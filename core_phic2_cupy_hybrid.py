import os
import sys
import time
import warnings
import cupy as cp
import numpy as np
from scipy.stats import pearsonr
from scipy.linalg import solve as cpu_solve
import gc

cp.set_printoptions(precision=3, linewidth=200)
warnings.filterwarnings('ignore')

# HYBRID CPU-GPU version
# Uses GPU for K storage and updates, CPU for memory-intensive K2P calculation
# This is slower but guaranteed to work with any matrix size

if not len(sys.argv) == 2:
    print("usage:: python core_phic2_cupy_hybrid.py normalized-HiC-Contact-Matrix")
    sys.exit()
fhic = str(sys.argv[1])

mempool = cp.get_default_memory_pool()

def print_memory():
    """Print both GPU and system memory usage"""
    gpu_used = mempool.used_bytes() / 1024**3
    gpu_free = cp.cuda.Device().mem_info[0] / 1024**3
    print(f"GPU: {gpu_used:.2f} GB used, {gpu_free:.2f} GB free")

def clear_memory():
    """Clear both GPU and CPU memory"""
    mempool.free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    gc.collect()

def Init_K(K, N, INIT_K0):
    """Vectorized initialization of tridiagonal backbone"""
    indices = np.arange(1, N)
    K[indices, indices-1] = INIT_K0
    K[indices-1, indices] = INIT_K0
    return K

def K2P_cpu(K_cpu, N):
    """
    K2P calculation on CPU using numpy/scipy.
    Memory-intensive but uses system RAM instead of GPU.
    """
    # All operations on CPU
    d = np.sum(K_cpu, axis=0, dtype=np.float64)
    L = np.diag(d) - K_cpu
    
    # Solve on CPU
    identity = np.eye(N-1, dtype=np.float64)
    Q = cpu_solve(L[1:N, 1:N], identity, assume_a='pos', overwrite_a=True, overwrite_b=True)
    
    del identity, L, d
    gc.collect()
    
    # M = 0.5 * (Q + Q.T)
    M = 0.5 * (Q + Q.T)
    del Q
    
    A = np.diag(M).copy()
    
    # Build G
    G = np.zeros((N, N), dtype=np.float64)
    G[1:N, 1:N] = -2*M + A + A.reshape(-1, 1)
    G[0, 1:N] = A
    G[1:N, 0] = A
    
    del M, A
    gc.collect()
    
    # P = (1 + 3*G)^(-1.5)
    P = (1.0 + 3.0 * G) ** (-1.5)
    del G
    gc.collect()
    
    return P

def compute_cost_and_gradient_hybrid(K_gpu, P_obs_cpu, N):
    """
    Hybrid computation:
    - Move K to CPU for K2P calculation
    - Compute gradient on CPU
    - Return cost and gradient (gradient goes back to GPU)
    """
    # Transfer K to CPU
    K_cpu = cp.asnumpy(K_gpu)
    
    # Compute P on CPU
    P_calc_cpu = K2P_cpu(K_cpu, N)
    
    # Compute difference on CPU
    P_dif_cpu = P_calc_cpu - P_obs_cpu
    
    # Compute cost
    cost = float(np.sqrt(np.sum(P_dif_cpu**2)) / N)
    
    del P_calc_cpu, K_cpu
    gc.collect()
    
    return cost, P_dif_cpu

def phic2_hybrid(K_gpu, P_obs_cpu, N, ETA=1.0e-4, ALPHA=1.0e-4, ITERATION_MAX=10000,
                 checkpoint_interval=1000, patience=100):
    """
    Hybrid CPU-GPU PHi-C2.
    - K stored on GPU
    - K2P computed on CPU (uses system RAM)
    - Gradient transferred to GPU for update
    """
    stop_delta = ETA * ALPHA
    paras_fit = "%e\t%e\t%d\t" % (ETA, ALPHA, ITERATION_MAX)
    
    # Velocity on GPU (only NxN float32 array on GPU besides K)
    velocity_gpu = cp.zeros((N, N), dtype=cp.float32)
    momentum = 0.9
    
    eta = ETA
    eta_min = ETA * 0.1
    
    print("Computing initial cost (on CPU)...")
    print_memory()
    
    cost, P_dif_cpu = compute_cost_and_gradient_hybrid(K_gpu, P_obs_cpu, N)
    
    c_traj = []
    c_traj.append([cost, time.time()])
    
    best_cost = cost
    best_K_file = f"{dataDir}/best_K_temp.npy"
    patience_counter = 0
    
    iteration = 1
    last_print_time = time.time()
    
    print(f"Starting hybrid CPU-GPU optimization")
    print(f"N={N}, ETA={ETA}, ALPHA={ALPHA}")
    print(f"Initial cost: {cost:.6e}")
    print_memory()
    print(f"\nIteration\tCost\t\tCost_diff\tEta\t\tTime(s)")
    
    while True:
        cost_bk = cost
        
        # Transfer gradient to GPU for update
        P_dif_gpu = cp.array(P_dif_cpu, dtype=cp.float32)
        del P_dif_cpu
        
        # Momentum update on GPU
        velocity_gpu *= momentum
        velocity_gpu -= eta * P_dif_gpu
        K_gpu += velocity_gpu.astype(cp.float64)
        
        del P_dif_gpu
        clear_memory()
        
        # Compute new cost (on CPU)
        cost, P_dif_cpu = compute_cost_and_gradient_hybrid(K_gpu, P_obs_cpu, N)
        
        c_traj.append([cost, time.time()])
        cost_dif = cost_bk - cost
        
        # Adaptive learning rate
        if cost_dif > 0:
            eta = min(eta * 1.05, ETA * 2)
        else:
            eta = max(eta * 0.5, eta_min)
        
        # Print progress
        current_time = time.time()
        if current_time - last_print_time > 10 or iteration % 50 == 0:
            elapsed = current_time - c_traj[0][1]
            print(f"{iteration}\t\t{cost:.6e}\t{cost_dif:+.4e}\t{eta:.4e}\t{elapsed:.1f}")
            if iteration % 200 == 0:
                print_memory()
            last_print_time = current_time
        
        # Track best solution
        if cost < best_cost:
            best_cost = cost
            if iteration % 50 == 0:
                np.save(best_K_file, cp.asnumpy(K_gpu))
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Checkpoint saving
        if iteration % checkpoint_interval == 0:
            checkpoint_file = f"{dataDir}/checkpoint_iter{iteration}.npz"
            K_cpu = cp.asnumpy(K_gpu)
            np.savez_compressed(checkpoint_file, K=K_cpu, cost=cost, iteration=iteration)
            del K_cpu
            print(f"Checkpoint saved: {checkpoint_file}, cost={cost:.6e}")
            clear_memory()
        
        # Stopping criteria
        converged = (0 < cost_dif < stop_delta) and (iteration > 1000)
        max_iter_reached = (iteration == ITERATION_MAX)
        diverged = np.isnan(cost) or np.isinf(cost)
        early_stop = (patience_counter > patience)
        
        if converged:
            print(f"Converged at iteration {iteration}")
            break
        elif early_stop:
            print(f"Early stopping at iteration {iteration} (no improvement for {patience} iterations)")
            if os.path.exists(best_K_file):
                K_gpu = cp.array(np.load(best_K_file))
            break
        elif max_iter_reached:
            print(f"Maximum iterations reached: {ITERATION_MAX}")
            break
        elif diverged:
            print(f"Optimization diverged at iteration {iteration}")
            break
        
        iteration += 1
    
    del velocity_gpu, P_dif_cpu
    clear_memory()
    
    # Clean up temp file
    if os.path.exists(best_K_file):
        os.remove(best_K_file)
    
    c_traj_array = np.array(c_traj)
    return [K_gpu, c_traj_array, paras_fit]

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
    fw = open(fn, 'w')
    fw.write(ct)
    n = len(xy)
    for i in range(n):
        lt = ''
        for v in xy[i]:
            lt += "%11s " % ('NaN') if np.isnan(v) else "%11.5e " % (v)
        fw.write(lt + '\n')
    fw.close()

if __name__ == '__main__':
    if not os.path.isfile(fhic):
        print('Cannot find ' + fhic)
        sys.exit()
    
    print(f"Reading Hi-C matrix from {fhic}...")
    start_read = time.time()
    
    # Read on CPU
    try:
        P_obs_cpu = np.loadtxt(fhic, comments='#', dtype=np.float64)
    except:
        P_obs_list = []
        with open(fhic) as fr:
            for line in fr:
                if not line[0] == '#':
                    lt = line.strip().split()
                    P_obs_list.append(list(map(float, lt)))
        P_obs_cpu = np.array(P_obs_list, dtype=np.float64)
        del P_obs_list
    
    N = len(P_obs_cpu)
    mem_per_matrix_gb = N * N * 8 / 1024**3
    
    print(f"Matrix size: N={N} ({N}x{N} = {N*N:,} elements)")
    print(f"Memory per matrix: {mem_per_matrix_gb:.2f} GB")
    print(f"GPU usage: ~{mem_per_matrix_gb * 1.5:.2f} GB (K + velocity)")
    print(f"CPU usage: ~{mem_per_matrix_gb * 3:.2f} GB (P_obs + working)")
    print(f"Read time: {time.time()-start_read:.2f}s")
    
    # Process on CPU
    np.nan_to_num(P_obs_cpu, copy=False)
    P_obs_cpu += np.eye(N)
    
    # P_obs stays on CPU!
    print("\nP_obs kept on CPU (system RAM)")
    
    # Create output directory
    phic2_alpha = 1.0e-10
    dataDir = fhic[:fhic.rfind('.')] + "_phic2_a%7.1e_hybrid" % (phic2_alpha)
    os.makedirs(dataDir, exist_ok=True)
    
    # Initialize K on GPU
    print("Initializing K on GPU...")
    K_gpu = cp.zeros((N, N), dtype=cp.float64)
    K_gpu = cp.array(Init_K(cp.asnumpy(K_gpu), N, INIT_K0=0.5))
    
    print_memory()
    
    print("\n" + "="*70)
    print("HYBRID CPU-GPU MODE")
    print("="*70)
    print("Strategy:")
    print("  - K matrix stored on GPU (optimized variable)")
    print("  - P_obs kept on CPU (system RAM)")
    print("  - K2P computation on CPU (memory-intensive)")
    print("  - Gradient updates on GPU (fast)")
    print("Advantage: No GPU memory limits, handles any matrix size")
    print("Trade-off: Slower due to CPU-GPU transfers")
    print("="*70 + "\n")
    
    print("Starting optimization...")
    start_opt = time.time()
    
    K_fit, c_traj, paras_fit = phic2_hybrid(
        K_gpu,
        P_obs_cpu,
        N,
        ETA=1e-4,
        ALPHA=phic2_alpha,
        ITERATION_MAX=1000000,
        checkpoint_interval=5000,
        patience=500
    )
    
    opt_time = time.time() - start_opt
    print(f"\nOptimization completed in {opt_time:.2f}s ({opt_time/3600:.2f} hours)")
    
    # Final K2P on CPU
    print("Computing final P_fit on CPU...")
    K_fit_cpu = cp.asnumpy(K_fit)
    P_fit_cpu = K2P_cpu(K_fit_cpu, N)
    
    # Free GPU
    del K_fit
    clear_memory()
    
    # Save results
    fo = "%s/N%d" % (dataDir, N)
    
    print("Saving results...")
    c_traj[:, 1] = c_traj[:, 1] - c_traj[0, 1]
    saveLg(fo + '.log', c_traj, "#%s\n#cost systemTime\n" % (paras_fit))
    
    saveMx(fo + '.K_fit', K_fit_cpu, "#K_fit N %d min: %11.5e max: %11.5e\n" % 
           (N, np.min(K_fit_cpu), np.max(K_fit_cpu)))
    
    triMask = np.where(np.triu(np.ones((N, N)), 1) > 0)
    pijMask = np.where(np.triu(P_obs_cpu, 1) > 0)
    p1 = pearsonr(P_fit_cpu[triMask], P_obs_cpu[triMask])[0]
    p2 = pearsonr(P_fit_cpu[pijMask], P_obs_cpu[pijMask])[0]
    
    ct = "#P_fit N %d min: %11.5e max: %11.5e pearson: %11.5e %11.5e\n" % \
         (N, np.nanmin(P_fit_cpu), np.nanmax(P_fit_cpu), p1, p2)
    saveMx(fo + '.P_fit', P_fit_cpu, ct)
    
    print(f"\nResults saved to {dataDir}")
    print(f"Pearson correlations: {p1:.6f} (all), {p2:.6f} (non-zero)")
    print("\nDone!")
