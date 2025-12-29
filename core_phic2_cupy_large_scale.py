import os
import sys
import time
import warnings
import cupy as cp
import numpy as np
from scipy.stats import pearsonr
import psutil
import gc

cp.set_printoptions(precision=3, linewidth=200)
warnings.filterwarnings('ignore')

# STABLE version with adaptive learning rate for difficult matrices
# Fixes divergence issues by automatically reducing learning rate when cost increases

if not len(sys.argv) >= 2:
    print("usage:: python core_phic2_cupy_stable.py normalized-HiC-Contact-Matrix [--fp32] [--eta=1e-5]")
    print("  --fp32: Use float32 instead of float64")
    print("  --eta=N: Initial learning rate (default: 1e-5 for stability)")
    sys.exit()

fhic = str(sys.argv[1])

# Parse arguments
USE_FP32 = '--fp32' in sys.argv
INITIAL_ETA = 1.0e-5  # More conservative default

for arg in sys.argv:
    if arg.startswith('--eta='):
        INITIAL_ETA = float(arg.split('=')[1])

dtype = cp.float32 if USE_FP32 else cp.float64
print(f"Using precision: {'float32' if USE_FP32 else 'float64'}")
print(f"Initial learning rate (ETA): {INITIAL_ETA}")

def print_memory_usage():
    """Print current CPU and GPU memory usage"""
    process = psutil.Process()
    cpu_mem_gb = process.memory_info().rss / 1024**3
    gpu_mem = cp.get_default_memory_pool().used_bytes() / 1024**3
    gpu_total = cp.cuda.Device().mem_info[1] / 1024**3
    print(f"Memory - CPU: {cpu_mem_gb:.2f}GB, GPU: {gpu_mem_gb:.2f}/{gpu_total:.2f}GB")

def clear_gpu_memory():
    """Clear GPU memory cache"""
    cp.get_default_memory_pool().free_all_blocks()
    gc.collect()

def Init_K(K, N, INIT_K0):
    for i in range(1, N):
        j = i - 1
        K[i, j] = K[j, i] = INIT_K0
    return K

def K2P(K):
    N = K.shape[0]
    d = cp.sum(K, axis=0)
    L = cp.diag(d) - K
    L_sub = L[1:N, 1:N]
    
    try:
        Q = cp.linalg.inv(L_sub)
    except:
        print("Warning: Matrix inversion failed, using pseudo-inverse")
        Q = cp.linalg.pinv(L_sub)
    
    M = 0.5 * (Q + cp.transpose(Q))
    A = cp.diag(M)
    
    G = cp.zeros((N, N), dtype=dtype)
    G[1:N, 1:N] = -2*M + A + cp.reshape(A, (-1, 1))
    G[0, 1:N] = A
    G[1:N, 0] = A
    
    P = (1. + 3.*G)**(-1.5)
    return P

def Pdif2cost(P_dif):
    N = P_dif.shape[0]
    cost = cp.sqrt(cp.sum(P_dif**2)) / N
    return cost

def phic2_adaptive(K, P_obs, ETA_INIT=1.0e-5, ALPHA=1.0e-4, ITERATION_MAX=10000, verbose=True):
    """PHi-C2 with adaptive learning rate to prevent divergence"""
    stop_delta = ETA_INIT * ALPHA
    paras_fit = "%e\t%e\t%d\t" % (ETA_INIT, ALPHA, ITERATION_MAX)
    
    # Initial computation
    P_dif = K2P(K) - P_obs
    cost = Pdif2cost(P_dif)
    
    # Check if initial cost is reasonable
    if cp.isnan(cost) or cp.isinf(cost):
        print("ERROR: Initial cost is NaN or Inf. Check input matrix.")
        sys.exit(1)
    
    c_traj = cp.zeros((ITERATION_MAX + 1, 2), dtype=dtype)
    c_traj[0, 0] = cost
    c_traj[0, 1] = time.time()
    
    # Adaptive learning rate parameters
    eta = ETA_INIT
    eta_min = ETA_INIT * 0.001  # Don't go below 0.1% of initial
    eta_max = ETA_INIT * 2.0    # Don't go above 2× initial
    eta_decay = 0.5             # Reduce by 50% when diverging
    eta_increase = 1.05         # Increase by 5% when stable
    
    consecutive_good = 0
    consecutive_bad = 0
    
    iteration = 1
    last_print_time = time.time()
    
    print(f"\nStarting optimization...")
    print(f"Initial cost: {float(cost):.5e}")
    print(f"Stop delta: {stop_delta:.5e}\n")
    
    while True:
        cost_bk = cost
        
        # Update step with current eta
        K_new = K - eta * P_dif
        P_dif_new = K2P(K_new) - P_obs
        cost_new = Pdif2cost(P_dif_new)
        
        # Check for NaN or Inf
        if cp.isnan(cost_new) or cp.isinf(cost_new):
            # Divergence detected - reduce learning rate and retry
            eta = eta * eta_decay
            if eta < eta_min:
                print(f"\nERROR: Learning rate too small ({eta:.2e}). Optimization failed.")
                print(f"Final cost: {float(cost):.5e}")
                print(f"Try:")
                print(f"  1. Check input matrix for errors")
                print(f"  2. Use smaller initial learning rate: --eta=1e-6")
                print(f"  3. Check matrix symmetry and normalization")
                break
            
            consecutive_bad += 1
            consecutive_good = 0
            
            if verbose:
                print(f"Iter {iteration:6d} | Diverged! Reducing ETA: {eta:.2e}")
            
            # Don't increment iteration, just retry with smaller eta
            continue
        
        cost_dif = cost_bk - cost_new
        
        # Accept the update
        K = K_new
        P_dif = P_dif_new
        cost = cost_new
        
        c_traj[iteration, 0] = cost
        c_traj[iteration, 1] = time.time()
        
        # Adapt learning rate based on progress
        if cost_dif > 0:
            # Cost decreased - good!
            consecutive_good += 1
            consecutive_bad = 0
            
            # Gradually increase learning rate if consistently good
            if consecutive_good > 10 and eta < eta_max:
                eta = min(eta * eta_increase, eta_max)
        else:
            # Cost increased - bad!
            consecutive_bad += 1
            consecutive_good = 0
            
            # Reduce learning rate if cost increases
            if consecutive_bad > 3:
                eta = max(eta * eta_decay, eta_min)
                consecutive_bad = 0
        
        # Progress reporting
        current_time = time.time()
        if verbose and (current_time - last_print_time > 10):
            elapsed = current_time - c_traj[0, 1]
            iter_per_sec = iteration / elapsed if elapsed > 0 else 0
            eta_sec = (ITERATION_MAX - iteration) / iter_per_sec if iter_per_sec > 0 else 0
            
            print(f"Iter {iteration:7d} | Cost: {float(cost):.5e} | ΔCost: {float(cost_dif):+.5e} | "
                  f"ETA: {eta:.2e} | Speed: {iter_per_sec:.2f} it/s | Time: {elapsed/60:.1f}m")
            print_memory_usage()
            last_print_time = current_time
        
        # Convergence check
        if (0 < cost_dif < stop_delta):
            if verbose:
                print(f"\n✓ Converged: cost_dif ({float(cost_dif):.5e}) < stop_delta ({stop_delta:.5e})")
            break
        
        if iteration >= ITERATION_MAX:
            if verbose:
                print(f"\n⚠ Reached max iterations ({ITERATION_MAX})")
            break
        
        iteration += 1
        
        # Periodic memory cleanup
        if iteration % 100 == 0:
            clear_gpu_memory()
    
    c_traj = c_traj[:iteration + 1]
    
    if verbose:
        print(f"\nFinal statistics:")
        print(f"  Iterations: {iteration}")
        print(f"  Final cost: {float(cost):.5e}")
        print(f"  Final ETA: {eta:.2e}")
        print(f"  Status: {'Converged' if iteration < ITERATION_MAX and not cp.isnan(cost) else 'Incomplete'}")
    
    return [K, c_traj, paras_fit]

def saveLg(fn, xy, ct):
    fw = open(fn, 'w')
    fw.write(ct)
    n = len(xy)
    m = np.shape(xy)[1] if xy.ndim == 2 else 0
    for i in range(n):
        if m == 0:
            lt = "%11s " % ('NaN') if np.isnan(xy[i]) else "%11.5e" % (xy[i])
        else:
            lt = ''
            for v in xy[i]:
                lt += "%11s " % ('NaN') if np.isnan(v) else "%11.5e " % (v)
        fw.write(lt + '\n')
    fw.close()

def saveMx(fn, xy, ct):
    print(f"Saving matrix to {fn}...")
    fw = open(fn, 'w')
    fw.write(ct)
    n = len(xy)
    chunk_size = 1000
    for chunk_start in range(0, n, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n)
        for i in range(chunk_start, chunk_end):
            lt = ''
            for v in xy[i]:
                lt += "%11s " % ('NaN') if np.isnan(v) else "%11.5e " % (v)
            fw.write(lt + '\n')
        if chunk_start % 5000 == 0 and chunk_start > 0:
            print(f"  Written {chunk_start}/{n} rows...")
    fw.close()
    print(f"  Completed!")

if True:
    print("="*80)
    print("PHi-C2 STABLE - Adaptive Learning Rate")
    print("="*80)
    
    # Read Hi-C
    if not os.path.isfile(fhic):
        print('Cannot find ' + fhic)
        sys.exit()
    else:
        print(f"\nLoading Hi-C matrix from: {fhic}")
        start_load = time.time()
        
        P_obs = []
        line_count = 0
        with open(fhic) as fr:
            for line in fr:
                if not line[0] == '#':
                    lt = line.strip().split()
                    P_obs.append(list(map(float, lt)))
                    line_count += 1
                    if line_count % 5000 == 0:
                        print(f"  Loaded {line_count} rows...")
        
        P_obs = cp.array(P_obs, dtype=dtype)
        N = len(P_obs)
        load_time = time.time() - start_load
        
        print(f"Loaded matrix: {N}×{N} in {load_time:.1f}s")
        matrix_size_gb = N * N * (4 if USE_FP32 else 8) / 1024**3
        print(f"Matrix size: {matrix_size_gb:.2f}GB")
        print_memory_usage()
        
        cp.nan_to_num(P_obs, copy=False)
        P_obs = P_obs + cp.eye(N, dtype=dtype)
        
        # Check matrix properties
        print(f"\nMatrix properties:")
        print(f"  Non-zero elements: {cp.count_nonzero(P_obs)}")
        print(f"  Sparsity: {100 * (1 - cp.count_nonzero(P_obs) / (N*N)):.2f}%")
        print(f"  Min value: {float(cp.min(P_obs)):.5e}")
        print(f"  Max value: {float(cp.max(P_obs)):.5e}")
        print(f"  Mean value: {float(cp.mean(P_obs)):.5e}")
    
    # Minimization
    print("\n" + "="*80)
    print("Starting optimization with adaptive learning rate...")
    print("="*80)
    
    K_fit = cp.zeros((N, N), dtype=dtype)
    K_fit = Init_K(K_fit, N, INIT_K0=0.5)
    
    phic2_alpha = 1.0e-4  # Slightly relaxed for stability
    
    start_opt = time.time()
    K_fit, c_traj, paras_fit = phic2_adaptive(
        K_fit, P_obs, 
        ETA_INIT=INITIAL_ETA, 
        ALPHA=phic2_alpha,
        ITERATION_MAX=1000000, 
        verbose=True
    )
    opt_time = time.time() - start_opt
    
    print(f"\nOptimization completed in {opt_time:.1f}s ({opt_time/60:.1f} min)")
    
    # Check if we have a valid result
    if cp.isnan(c_traj[-1, 0]):
        print("\n" + "="*80)
        print("ERROR: Optimization failed (NaN in final cost)")
        print("="*80)
        print("\nTroubleshooting suggestions:")
        print("1. Try smaller learning rate: --eta=1e-6 or --eta=1e-7")
        print("2. Check input matrix:")
        print("   - Should be symmetric")
        print("   - Should be normalized (contact probabilities)")
        print("   - Should not have extreme values")
        print("3. Check for inf/nan in input:")
        P_obs_cpu = cp.asnumpy(P_obs)
        print(f"   Input has NaN: {np.any(np.isnan(P_obs_cpu))}")
        print(f"   Input has Inf: {np.any(np.isinf(P_obs_cpu))}")
        sys.exit(1)
    
    print("\nComputing final P_fit...")
    P_fit = K2P(K_fit)
    
    # Save results
    print("\n" + "="*80)
    print("Saving results...")
    print("="*80)
    
    c_traj = cp.asnumpy(c_traj)
    K_fit = cp.asnumpy(K_fit)
    P_fit = cp.asnumpy(P_fit)
    P_obs = cp.asnumpy(P_obs)
    
    clear_gpu_memory()
    
    precision_str = "fp32" if USE_FP32 else "fp64"
    dataDir = fhic[:fhic.rfind('.')] + f"_phic2_stable_a{phic2_alpha:7.1e}_{precision_str}"
    os.makedirs(dataDir, exist_ok=True)
    fo = "%s/N%d" % (dataDir, N)
    
    c_traj[:, 1] = c_traj[:, 1] - c_traj[0, 1]
    saveLg(fo + '.log', c_traj, "#%s\n#cost systemTime\n" % (paras_fit))
    saveMx(fo + '.K_fit', K_fit, 
           "#K_fit N %d min: %11.5e max: %11.5e\n" % (N, np.min(K_fit), np.max(K_fit)))
    
    print("\nComputing Pearson correlation...")
    triMask = np.where(np.triu(np.ones((N, N)), 1) > 0)
    pijMask = np.where(np.triu(P_obs, 1) > 0)
    
    p1 = pearsonr(P_fit[triMask], P_obs[triMask])[0]
    p2 = pearsonr(P_fit[pijMask], P_obs[pijMask])[0]
    
    ct = "#P_fit N %d min: %11.5e max: %11.5e pearson: %11.5e %11.5e\n" % \
         (N, np.nanmin(P_fit), np.nanmax(P_fit), p1, p2)
    saveMx(fo + '.P_fit', P_fit, ct)
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Matrix size: {N}×{N}")
    print(f"Precision: {precision_str}")
    print(f"Load time: {load_time:.1f}s")
    print(f"Optimization time: {opt_time:.1f}s ({opt_time/60:.1f} min)")
    print(f"Total iterations: {len(c_traj)-1}")
    print(f"Final cost: {c_traj[-1,0]:.5e}")
    print(f"Pearson correlation (all): {p1:.5f}")
    print(f"Pearson correlation (p>0): {p2:.5f}")
    print(f"Output directory: {dataDir}")
    print("="*80)
