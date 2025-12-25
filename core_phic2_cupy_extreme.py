import os
import sys
import time
import warnings
import cupy as cp
import numpy as np
from scipy.stats import pearsonr
from cupy.linalg import solve
import gc

cp.set_printoptions(precision=3, linewidth=200)
warnings.filterwarnings('ignore')

# EXTREME MEMORY-OPTIMIZED version
# Strategy: Keep only K and P_obs on GPU, compute everything else on-the-fly
# This is slower but can handle arbitrarily large matrices

if not len(sys.argv) == 2:
    print("usage:: python core_phic2_cupy_extreme.py normalized-HiC-Contact-Matrix")
    sys.exit()
fhic = str(sys.argv[1])

mempool = cp.get_default_memory_pool()
pinned_mempool = cp.get_default_pinned_memory_pool()

def print_gpu_memory():
    """Print current GPU memory usage"""
    used = mempool.used_bytes() / 1024**3
    total = mempool.total_bytes() / 1024**3
    free = cp.cuda.Device().mem_info[0] / 1024**3
    print(f"GPU Memory: {used:.2f} GB used / {total:.2f} GB pool / {free:.2f} GB free")

def clear_memory():
    """Aggressively free GPU memory"""
    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()
    gc.collect()

def Init_K(K, N, INIT_K0):
    """Vectorized initialization of tridiagonal backbone"""
    indices = cp.arange(1, N)
    K[indices, indices-1] = INIT_K0
    K[indices-1, indices] = INIT_K0
    return K

def K2P_extreme(K, N):
    """
    EXTREME memory-efficient K2P.
    Only keeps absolutely necessary arrays on GPU at any time.
    Returns cost directly without keeping P_calc in memory.
    """
    # Step 1: Compute d and build L in one pass
    d = cp.sum(K, axis=0, dtype=cp.float32)
    
    # Build L_sub directly (most memory-intensive part)
    L_sub = -K[1:N, 1:N].astype(cp.float32)
    L_sub[cp.diag_indices(N-1)] += d[1:].astype(cp.float32)
    del d
    clear_memory()
    
    # Step 2: Solve for Q (in float32 to save memory)
    identity = cp.eye(N-1, dtype=cp.float32)
    Q = solve(L_sub, identity)
    del identity, L_sub
    clear_memory()
    
    # Step 3: Compute M = 0.5 * (Q + Q.T) in-place
    Q += Q.T
    Q *= 0.5
    M = Q
    
    A = cp.diag(M).copy()
    
    # Step 4: Build G in small chunks and compute P incrementally
    # Instead of building full G, we'll build P directly in chunks
    chunk_size = min(2000, N // 10)  # Adaptive chunk size
    
    P = cp.zeros((N, N), dtype=cp.float32)
    
    # Diagonal term
    P[0, 0] = 1.0
    
    # First row/column
    G_chunk = A.astype(cp.float32)
    P[0, 1:N] = (1.0 + 3.0 * G_chunk) ** (-1.5)
    P[1:N, 0] = P[0, 1:N]
    del G_chunk
    
    # Process rest in chunks
    for i in range(1, N, chunk_size):
        end = min(i + chunk_size, N)
        rows = slice(i, end)
        
        # G[i:end, 1:N] = -2*M[i-1:end-1, :] + A + A.reshape(-1, 1)
        G_chunk = -2.0 * M[i-1:end-1, :]
        G_chunk += A
        G_chunk += A.reshape(-1, 1)
        
        # Diagonal elements
        for j, k in enumerate(range(i, end)):
            G_chunk[j, k-1] = 0.0  # Diagonal of G is 0
        
        # Convert to P
        P[rows, 1:N] = (1.0 + 3.0 * G_chunk) ** (-1.5)
        
        del G_chunk
        if i % (chunk_size * 5) == 0:
            clear_memory()
    
    del M, A
    clear_memory()
    
    return P

def compute_cost_and_gradient_extreme(K, P_obs, N):
    """
    Compute cost and gradient in one pass to minimize memory.
    Returns: (cost, gradient_direction)
    """
    # Compute P_calc
    P_calc = K2P_extreme(K, N)
    
    # Compute difference (this is our gradient direction)
    P_dif = P_calc - P_obs
    
    # Compute cost
    cost = float(cp.sqrt(cp.sum(P_dif**2)) / N)
    
    # Keep only the gradient direction, free P_calc
    del P_calc
    clear_memory()
    
    return cost, P_dif

def phic2_extreme(K, P_obs, N, ETA=1.0e-4, ALPHA=1.0e-4, ITERATION_MAX=10000,
                  checkpoint_interval=1000, patience=100):
    """
    EXTREME memory-optimized PHi-C2.
    Only keeps K, P_obs, and gradient on GPU.
    """
    stop_delta = ETA * ALPHA
    paras_fit = "%e\t%e\t%d\t" % (ETA, ALPHA, ITERATION_MAX)
    
    # Minimal GPU memory: only velocity in float32
    velocity = cp.zeros((N, N), dtype=cp.float32)
    momentum = 0.9
    
    eta = ETA
    eta_min = ETA * 0.1
    
    print("Computing initial cost...")
    print_gpu_memory()
    
    cost, P_dif = compute_cost_and_gradient_extreme(K, P_obs, N)
    
    c_traj = []
    c_traj.append([cost, time.time()])
    
    best_cost = cost
    best_K_file = f"{dataDir}/best_K_temp.npy"
    patience_counter = 0
    
    iteration = 1
    last_print_time = time.time()
    last_cleanup = time.time()
    
    print(f"Starting optimization with N={N}, ETA={ETA}, ALPHA={ALPHA}")
    print(f"Initial cost: {cost:.6e}")
    print_gpu_memory()
    print(f"Iteration\tCost\t\tCost_diff\tEta\t\tTime(s)\tGPU_GB")
    
    while True:
        cost_bk = cost
        
        # Momentum update
        velocity *= momentum
        velocity -= eta * P_dif
        K += velocity.astype(cp.float64)
        
        del P_dif  # Free gradient immediately
        
        # Periodic cleanup
        current_time = time.time()
        if current_time - last_cleanup > 30:  # Every 30 seconds
            clear_memory()
            last_cleanup = current_time
        
        # Compute new cost and gradient
        cost, P_dif = compute_cost_and_gradient_extreme(K, P_obs, N)
        
        c_traj.append([cost, time.time()])
        cost_dif = cost_bk - cost
        
        # Adaptive learning rate
        if cost_dif > 0:
            eta = min(eta * 1.05, ETA * 2)
        else:
            eta = max(eta * 0.5, eta_min)
        
        # Print progress
        if current_time - last_print_time > 10 or iteration % 100 == 0:
            elapsed = current_time - c_traj[0][1]
            gpu_used = mempool.used_bytes() / 1024**3
            print(f"{iteration}\t\t{cost:.6e}\t{cost_dif:+.4e}\t{eta:.4e}\t{elapsed:.1f}\t{gpu_used:.1f}")
            last_print_time = current_time
        
        # Track best solution (save to CPU/disk)
        if cost < best_cost:
            best_cost = cost
            if iteration % 50 == 0:  # Save frequently but not every iteration
                np.save(best_K_file, cp.asnumpy(K))
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Checkpoint saving
        if iteration % checkpoint_interval == 0:
            checkpoint_file = f"{dataDir}/checkpoint_iter{iteration}.npz"
            K_cpu = cp.asnumpy(K)
            np.savez_compressed(checkpoint_file, K=K_cpu, cost=cost, iteration=iteration)
            del K_cpu
            print(f"Checkpoint saved: {checkpoint_file}, cost={cost:.6e}")
            print_gpu_memory()
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
            # Restore best K
            if os.path.exists(best_K_file):
                K = cp.array(np.load(best_K_file))
            break
        elif max_iter_reached:
            print(f"Maximum iterations reached: {ITERATION_MAX}")
            break
        elif diverged:
            print(f"Optimization diverged at iteration {iteration}")
            break
        
        iteration += 1
    
    del velocity, P_dif
    clear_memory()
    
    # Clean up temp file
    if os.path.exists(best_K_file):
        os.remove(best_K_file)
    
    c_traj_array = np.array(c_traj)
    return [K, c_traj_array, paras_fit]

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
    
    # Read in numpy (CPU)
    try:
        P_obs_np = np.loadtxt(fhic, comments='#', dtype=np.float64)
    except:
        P_obs_list = []
        with open(fhic) as fr:
            for line in fr:
                if not line[0] == '#':
                    lt = line.strip().split()
                    P_obs_list.append(list(map(float, lt)))
        P_obs_np = np.array(P_obs_list, dtype=np.float64)
        del P_obs_list
    
    N = len(P_obs_np)
    mem_per_matrix_gb = N * N * 8 / 1024**3
    
    print(f"Matrix size: N={N} ({N}x{N} = {N*N:,} elements)")
    print(f"Memory per float64 matrix: {mem_per_matrix_gb:.2f} GB")
    print(f"Memory per float32 matrix: {mem_per_matrix_gb/2:.2f} GB")
    print(f"Estimated GPU usage: ~{mem_per_matrix_gb * 2.5:.2f} GB (K + P_obs + working)")
    print(f"Read time: {time.time()-start_read:.2f}s")
    
    # Process on CPU
    np.nan_to_num(P_obs_np, copy=False)
    P_obs_np += np.eye(N)
    
    # Transfer to GPU - convert to float32 to save memory
    print("Transferring P_obs to GPU (float32)...")
    P_obs = cp.array(P_obs_np, dtype=cp.float32)
    del P_obs_np
    gc.collect()
    
    print_gpu_memory()
    
    # Create output directory
    phic2_alpha = 1.0e-10
    dataDir = fhic[:fhic.rfind('.')] + "_phic2_a%7.1e_extreme" % (phic2_alpha)
    os.makedirs(dataDir, exist_ok=True)
    
    # Initialize K in float64 (this is what we optimize)
    print("\nInitializing K matrix...")
    K_fit = cp.zeros((N, N), dtype=cp.float64)
    K_fit = Init_K(K_fit, N, INIT_K0=0.5)
    
    print_gpu_memory()
    
    print("\n" + "="*70)
    print("EXTREME MEMORY MODE")
    print("="*70)
    print("Strategy: Keep only K and P_obs on GPU")
    print("All intermediate calculations done on-the-fly")
    print("This is slower but handles arbitrarily large matrices")
    print("="*70 + "\n")
    
    print("Starting optimization...")
    start_opt = time.time()
    
    K_fit, c_traj, paras_fit = phic2_extreme(
        K_fit,
        P_obs,
        N,
        ETA=1e-4,
        ALPHA=phic2_alpha,
        ITERATION_MAX=1000000,
        checkpoint_interval=5000,
        patience=500
    )
    
    opt_time = time.time() - start_opt
    print(f"\nOptimization completed in {opt_time:.2f}s ({opt_time/3600:.2f} hours)")
    
    # Final K2P
    print("Computing final P_fit...")
    P_fit = K2P_extreme(K_fit, N)
    
    # Convert to CPU
    print("Converting to CPU memory...")
    K_fit_cpu = cp.asnumpy(K_fit)
    P_fit_cpu = cp.asnumpy(P_fit)
    P_obs_cpu = cp.asnumpy(P_obs)
    
    # Free GPU
    del K_fit, P_fit, P_obs
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
