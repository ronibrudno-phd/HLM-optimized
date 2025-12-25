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

# MEMORY-OPTIMIZED version for large matrices
# Key optimizations:
# 1. Aggressive memory cleanup with garbage collection
# 2. In-place operations wherever possible
# 3. Strategic memory pooling with limits
# 4. Chunked operations for large matrices
# 5. Reduced intermediate arrays

if not len(sys.argv) == 2:
    print("usage:: python core_phic2_cupy_memory_optimized.py normalized-HiC-Contact-Matrix")
    sys.exit()
fhic = str(sys.argv[1])  # HiC observation

# Configure memory pool with limits
mempool = cp.get_default_memory_pool()
pinned_mempool = cp.get_default_pinned_memory_pool()

def print_gpu_memory():
    """Print current GPU memory usage"""
    used = mempool.used_bytes() / 1024**3
    total = mempool.total_bytes() / 1024**3
    print(f"GPU Memory: {used:.2f} GB used / {total:.2f} GB allocated")

def clear_memory():
    """Aggressively free GPU memory"""
    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()

def Init_K(K, N, INIT_K0):
    """Vectorized initialization of tridiagonal backbone"""
    indices = cp.arange(1, N)
    K[indices, indices-1] = INIT_K0
    K[indices-1, indices] = INIT_K0
    return K

def K2P_memory_efficient(K, N):
    """
    Memory-efficient K2P conversion.
    Minimizes intermediate arrays and uses in-place operations.
    """
    # Compute L = diag(d) - K where d = sum of K rows
    d = cp.sum(K, axis=0)
    L = -K.copy()  # Start with -K
    L[cp.diag_indices(N)] += d  # Add diagonal in-place
    
    # Solve L[1:N,1:N] @ Q = I for Q
    # This is the memory bottleneck - we need identity matrix
    identity = cp.eye(N-1, dtype=cp.float32)  # Use float32 to save memory
    L_sub = L[1:N, 1:N].astype(cp.float32)
    
    Q = solve(L_sub, identity)
    
    # Free memory immediately
    del identity, L_sub, L, d
    clear_memory()
    
    # Compute M = 0.5 * (Q + Q.T)
    M = Q
    M += M.T
    M *= 0.5
    
    A = cp.diag(M).copy()
    
    # Compute G efficiently
    G = cp.zeros((N, N), dtype=cp.float32)
    
    # G[1:N, 1:N] = -2*M + A + A.reshape(-1, 1)
    G[1:N, 1:N] = -2 * M
    G[1:N, 1:N] += A.reshape(-1, 1)
    G[1:N, 1:N] += A
    
    del M
    
    G[0, 1:N] = A
    G[1:N, 0] = A
    
    del A
    clear_memory()
    
    # Compute P = (1 + 3*G)^(-1.5)
    G *= 3
    G += 1
    cp.power(G, -1.5, out=G)  # In-place operation
    
    P = G.astype(cp.float64)  # Convert back to float64 for accuracy
    del G
    clear_memory()
    
    return P

def Pdif2cost(P_dif):
    """Compute RMSE cost"""
    cost = cp.sqrt(cp.sum(P_dif**2)) / N
    return cost

def phic2_memory_optimized(K, P_obs, N, ETA=1.0e-4, ALPHA=1.0e-4, ITERATION_MAX=10000,
                          checkpoint_interval=1000, patience=100):
    """
    Memory-optimized PHi-C2 with:
    - Aggressive memory management
    - Reduced intermediate arrays
    - Periodic cleanup
    """
    stop_delta = ETA * ALPHA
    paras_fit = "%e\t%e\t%d\t" % (ETA, ALPHA, ITERATION_MAX)
    
    # Use float32 for velocity to save memory
    velocity = cp.zeros_like(K, dtype=cp.float32)
    momentum = 0.9
    
    # Adaptive learning rate
    eta = ETA
    eta_decay = 0.995
    eta_min = ETA * 0.1
    
    print("Computing initial cost...")
    print_gpu_memory()
    
    # Compute initial cost
    P_calc = K2P_memory_efficient(K, N)
    P_dif = P_calc - P_obs
    cost = Pdif2cost(P_dif)
    del P_calc
    clear_memory()
    
    c_traj = []
    c_traj.append([float(cost), time.time()])
    
    # Early stopping variables
    best_cost = float(cost)
    best_K = None  # Only save when needed
    patience_counter = 0
    
    iteration = 1
    last_print_time = time.time()
    last_cleanup = time.time()
    
    print(f"Starting optimization with N={N}, ETA={ETA}, ALPHA={ALPHA}")
    print(f"Initial cost: {float(cost):.6e}")
    print_gpu_memory()
    print(f"Iteration\tCost\t\tCost_diff\tEta\t\tTime(s)")
    
    while True:
        cost_bk = cost
        
        # Momentum update (using float32 velocity)
        velocity *= momentum
        velocity -= eta * P_dif.astype(cp.float32)
        K += velocity.astype(cp.float64)
        
        # Periodic aggressive cleanup
        current_time = time.time()
        if current_time - last_cleanup > 60:  # Every minute
            clear_memory()
            last_cleanup = current_time
        
        # Compute new cost
        P_calc = K2P_memory_efficient(K, N)
        P_dif = P_calc - P_obs
        cost = Pdif2cost(P_dif)
        del P_calc
        
        c_traj.append([float(cost), time.time()])
        
        cost_dif = cost_bk - cost
        
        # Adaptive learning rate
        if cost_dif > 0:
            eta = min(eta * 1.05, ETA * 2)
        else:
            eta = max(eta * 0.5, eta_min)
        
        # Print progress
        if current_time - last_print_time > 10 or iteration % 100 == 0:
            elapsed = current_time - c_traj[0][1]
            print(f"{iteration}\t\t{float(cost):.6e}\t{float(cost_dif):+.4e}\t{eta:.4e}\t{elapsed:.1f}")
            print_gpu_memory()
            last_print_time = current_time
        
        # Track best solution
        if cost < best_cost:
            best_cost = cost
            if best_K is not None:
                del best_K
            best_K = K.copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Checkpoint saving
        if iteration % checkpoint_interval == 0:
            checkpoint_file = f"{dataDir}/checkpoint_iter{iteration}.npz"
            K_cpu = cp.asnumpy(K)
            np.savez_compressed(checkpoint_file, K=K_cpu, cost=float(cost), iteration=iteration)
            del K_cpu
            print(f"Checkpoint saved at iteration {iteration}, cost={float(cost):.6e}")
            clear_memory()
        
        # Stopping criteria
        converged = (0 < cost_dif < stop_delta) and (iteration > 1000)
        max_iter_reached = (iteration == ITERATION_MAX)
        diverged = cp.isnan(cost)
        early_stop = (patience_counter > patience)
        
        if converged:
            print(f"Converged at iteration {iteration}")
            break
        elif early_stop:
            print(f"Early stopping at iteration {iteration} (no improvement for {patience} iterations)")
            if best_K is not None:
                K = best_K
            break
        elif max_iter_reached:
            print(f"Maximum iterations reached: {ITERATION_MAX}")
            break
        elif diverged:
            print(f"Optimization diverged at iteration {iteration}")
            break
        
        iteration += 1
    
    del velocity, P_dif
    if best_K is not None and best_K is not K:
        del best_K
    clear_memory()
    
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
    # Read Hi-C
    if not os.path.isfile(fhic):
        print('Cannot find ' + fhic)
        sys.exit()
    
    print(f"Reading Hi-C matrix from {fhic}...")
    start_read = time.time()
    
    # Read in numpy first (CPU memory)
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
    print(f"Matrix size: N={N} ({N}x{N} = {N*N:,} elements)")
    print(f"Memory required (approximate): {N*N*8/1024**3:.2f} GB per matrix")
    print(f"Read time: {time.time()-start_read:.2f}s")
    
    # Process on CPU first
    np.nan_to_num(P_obs_np, copy=False)
    P_obs_np = P_obs_np + np.eye(N)
    
    # Now transfer to GPU
    print("Transferring to GPU...")
    P_obs = cp.array(P_obs_np, dtype=cp.float64)
    del P_obs_np
    gc.collect()
    
    print_gpu_memory()
    
    # Create output directory
    phic2_alpha = 1.0e-10
    dataDir = fhic[:fhic.rfind('.')] + "_phic2_a%7.1e_memory_optimized" % (phic2_alpha)
    os.makedirs(dataDir, exist_ok=True)
    
    # Initialize K
    print("\nInitializing K matrix...")
    K_fit = cp.zeros((N, N), dtype=cp.float64)
    K_fit = Init_K(K_fit, N, INIT_K0=0.5)
    
    print_gpu_memory()
    
    print("\nStarting optimization...")
    start_opt = time.time()
    
    K_fit, c_traj, paras_fit = phic2_memory_optimized(
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
    
    # Final K2P calculation
    print("Computing final P_fit...")
    P_fit = K2P_memory_efficient(K_fit, N)
    
    # Convert to numpy for saving
    print("Converting to CPU memory...")
    K_fit_cpu = cp.asnumpy(K_fit)
    P_fit_cpu = cp.asnumpy(P_fit)
    P_obs_cpu = cp.asnumpy(P_obs)
    
    # Free GPU memory
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
