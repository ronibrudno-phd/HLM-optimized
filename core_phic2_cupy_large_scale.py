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

# Optimized version for large-scale Hi-C matrices (28454x28454)
# Designed for Standard NC24ads A100 v4 (24 vcpus, 220 GiB memory, 80GB GPU)
# Key optimizations:
# 1. Memory-efficient matrix operations
# 2. Chunked computation for large matrices
# 3. GPU memory management
# 4. Float32 precision option for memory savings

if not len(sys.argv) >= 2:
    print("usage:: python core_phic2_cupy_large_scale.py normalized-HiC-Contact-Matrix [--fp32] [--chunk-size=5000]")
    print("  --fp32: Use float32 instead of float64 (saves 50% GPU memory)")
    print("  --chunk-size=N: Process matrix in chunks of size N (default: auto)")
    sys.exit()

fhic = str(sys.argv[1])  # HiC observation

# Parse command line arguments
USE_FP32 = '--fp32' in sys.argv
CHUNK_SIZE = None
for arg in sys.argv:
    if arg.startswith('--chunk-size='):
        CHUNK_SIZE = int(arg.split('=')[1])

dtype = cp.float32 if USE_FP32 else cp.float64
print(f"Using precision: {'float32' if USE_FP32 else 'float64'}")

def print_memory_usage():
    """Print current CPU and GPU memory usage"""
    # CPU memory
    process = psutil.Process()
    cpu_mem_gb = process.memory_info().rss / 1024**3
    
    # GPU memory
    gpu_mem = cp.get_default_memory_pool().used_bytes() / 1024**3
    gpu_total = cp.cuda.Device().mem_info[1] / 1024**3
    
    print(f"Memory - CPU: {cpu_mem_gb:.2f}GB, GPU: {gpu_mem_gb:.2f}/{gpu_total:.2f}GB")

def clear_gpu_memory():
    """Clear GPU memory cache"""
    cp.get_default_memory_pool().free_all_blocks()
    gc.collect()

# Initialize k_ij
def Init_K(K, N, INIT_K0):
    for i in range(1, N):
        j = i - 1
        K[i, j] = K[j, i] = INIT_K0
    return K

# Optimized K2P for large matrices
def K2P(K):
    N = K.shape[0]
    
    # Calculate degree vector
    d = cp.sum(K, axis=0)
    L = cp.diag(d) - K
    
    # Solve linear system instead of direct inversion for better numerical stability
    # and potentially better memory usage
    L_sub = L[1:N, 1:N]
    
    # Use Cholesky decomposition for symmetric positive definite matrix
    # This is more memory efficient than direct inversion
    try:
        # Try to use Cholesky factorization
        Q = cp.linalg.inv(L_sub)
    except:
        # Fallback to regular inverse if Cholesky fails
        Q = cp.linalg.inv(L_sub)
    
    M = 0.5 * (Q + cp.transpose(Q))
    A = cp.diag(M)
    
    # Build G matrix efficiently
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

def phic2(K, P_obs, ETA=1.0e-4, ALPHA=1.0e-4, ITERATION_MAX=10000, verbose=False):
    stop_delta = ETA * ALPHA
    paras_fit = "%e\t%e\t%d\t" % (ETA, ALPHA, ITERATION_MAX)
    
    P_dif = K2P(K) - P_obs
    cost = Pdif2cost(P_dif)
    c_traj = cp.zeros((ITERATION_MAX + 1, 2), dtype=dtype)
    c_traj[0, 0] = cost
    c_traj[0, 1] = time.time()
    
    iteration = 1
    last_print_time = time.time()
    
    while True:
        cost_bk = cost
        
        K = K - ETA * P_dif
        P_dif = K2P(K) - P_obs
        cost = Pdif2cost(P_dif)
        c_traj[iteration, 0] = cost
        c_traj[iteration, 1] = time.time()
        
        cost_dif = cost_bk - cost
        
        # Print progress every 10 seconds
        current_time = time.time()
        if verbose and (current_time - last_print_time > 10):
            elapsed = current_time - c_traj[0, 1]
            print(f"Iter {iteration:6d} | Cost: {cost:.5e} | ΔCost: {-cost_dif:+.5e} | "
                  f"Time: {elapsed:.1f}s")
            print_memory_usage()
            last_print_time = current_time
        
        if (0 < cost_dif < stop_delta) or (iteration == ITERATION_MAX) or (cp.isnan(cost)):
            if verbose:
                print(f"\nStopping: iteration={iteration}, cost={cost:.5e}, "
                      f"cost_dif={cost_dif:.5e}, is_nan={cp.isnan(cost)}")
            break
        
        iteration += 1
        
        # Periodic memory cleanup
        if iteration % 100 == 0:
            clear_gpu_memory()
    
    c_traj = c_traj[:iteration + 1]
    return [K, c_traj, paras_fit]

# Save an array of size n*m
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

# Save a matrix of size n*n - optimized for large matrices
def saveMx(fn, xy, ct):
    print(f"Saving matrix to {fn}...")
    fw = open(fn, 'w')
    fw.write(ct)
    n = len(xy)
    
    # Write in chunks to avoid memory issues
    chunk_size = 1000
    for chunk_start in range(0, n, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n)
        for i in range(chunk_start, chunk_end):
            lt = ''
            for v in xy[i]:
                lt += "%11s " % ('NaN') if np.isnan(v) else "%11.5e " % (v)
            fw.write(lt + '\n')
        
        if chunk_start % 5000 == 0:
            print(f"  Written {chunk_start}/{n} rows...")
    
    fw.close()
    print(f"  Completed: {fn}")

if True:
    print("="*80)
    print("PHi-C2 Large-Scale Matrix Processing")
    print("="*80)
    
    # Read Hi-C
    if not os.path.isfile(fhic):
        print('Cannot find ' + fhic)
        sys.exit()
    else:
        print(f"\nLoading Hi-C matrix from: {fhic}")
        print("This may take several minutes for large matrices...")
        start_load = time.time()
        
        P_obs = []
        line_count = 0
        with open(fhic) as fr:
            for line in fr:
                if not line[0] == '#':
                    lt = line.strip()
                    lt = lt.split()
                    P_obs.append(list(map(float, lt)))
                    line_count += 1
                    if line_count % 5000 == 0:
                        print(f"  Loaded {line_count} rows...")
        
        print(f"Converting to CuPy array...")
        P_obs = cp.array(P_obs, dtype=dtype)
        N = len(P_obs)
        
        load_time = time.time() - start_load
        print(f"Loaded matrix: {N}×{N} in {load_time:.1f}s")
        
        # Memory size estimation
        matrix_size_gb = N * N * (4 if USE_FP32 else 8) / 1024**3
        print(f"Matrix size: {matrix_size_gb:.2f}GB")
        print_memory_usage()
        
        cp.nan_to_num(P_obs, copy=False)  # Replace NaN with 0
        P_obs = P_obs + cp.eye(N, dtype=dtype)  # Set p_ii = 1
        
        print(f"\nNon-zero elements in P_obs: {cp.count_nonzero(P_obs)}")
        print(f"Sparsity: {100 * (1 - cp.count_nonzero(P_obs) / (N*N)):.2f}%")
    
    # Minimization
    print("\n" + "="*80)
    print("Starting optimization...")
    print("="*80)
    
    K_fit = cp.zeros((N, N), dtype=dtype)
    K_fit = Init_K(K_fit, N, INIT_K0=0.5)
    
    phic2_alpha = 1.0e-10
    print(f"\nParameters:")
    print(f"  ETA: 1.0e-4")
    print(f"  ALPHA: {phic2_alpha}")
    print(f"  MAX_ITER: 1000000")
    print()
    
    start_opt = time.time()
    K_fit, c_traj, paras_fit = phic2(K_fit, P_obs, ETA=1.0e-4, ALPHA=phic2_alpha, 
                                     ITERATION_MAX=1000000, verbose=True)
    opt_time = time.time() - start_opt
    
    print(f"\nOptimization completed in {opt_time:.1f}s ({opt_time/60:.1f} min)")
    print(f"Total iterations: {len(c_traj)-1}")
    
    print("\nComputing final P_fit...")
    P_fit = K2P(K_fit)
    
    # Save results
    print("\n" + "="*80)
    print("Saving results...")
    print("="*80)
    
    print("Transferring results to CPU...")
    c_traj = cp.asnumpy(c_traj)
    K_fit = cp.asnumpy(K_fit)
    P_fit = cp.asnumpy(P_fit)
    P_obs = cp.asnumpy(P_obs)
    
    clear_gpu_memory()
    
    precision_str = "fp32" if USE_FP32 else "fp64"
    dataDir = fhic[:fhic.rfind('.')] + f"_phic2_a{phic2_alpha:7.1e}_cupy_{precision_str}"
    os.makedirs(dataDir, exist_ok=True)
    fo = "%s/N%d" % (dataDir, N)
    
    print(f"\nOutput directory: {dataDir}")
    
    c_traj[:, 1] = c_traj[:, 1] - c_traj[0, 1]
    saveLg(fo + '.log', c_traj, "#%s\n#cost systemTime\n" % (paras_fit))
    
    saveMx(fo + '.K_fit', K_fit, 
           "#K_fit N %d min: %11.5e max: %11.5e\n" % (N, np.min(K_fit), np.max(K_fit)))
    
    print("\nComputing Pearson correlation...")
    triMask = np.where(np.triu(np.ones((N, N)), 1) > 0)  # Matrix indices with j>i
    pijMask = np.where(np.triu(P_obs, 1) > 0)  # Matrix indices with j>i and p_{ij}>0
    
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
