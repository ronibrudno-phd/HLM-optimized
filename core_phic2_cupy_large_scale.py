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

# Auto-tuning version that adapts to matrix scale
# Automatically determines optimal learning rate based on matrix properties

if not len(sys.argv) >= 2:
    print("usage:: python core_phic2_cupy_auto_tuned.py normalized-HiC-Contact-Matrix [--fp32]")
    print("  --fp32: Use float32 instead of float64 (recommended)")
    sys.exit()

fhic = str(sys.argv[1])
USE_FP32 = '--fp32' in sys.argv

dtype = cp.float32 if USE_FP32 else cp.float64
np_dtype = np.float32 if USE_FP32 else np.float64

print(f"Using precision: {'float32' if USE_FP32 else 'float64'}")

def print_memory_usage():
    process = psutil.Process()
    cpu_mem_gb = process.memory_info().rss / 1024**3
    gpu_mem = cp.get_default_memory_pool().used_bytes() / 1024**3
    gpu_total = cp.cuda.Device().mem_info[1] / 1024**3
    print(f"Memory - CPU: {cpu_mem_gb:.2f}GB, GPU: {gpu_mem:.2f}/{gpu_total:.2f}GB")

def clear_gpu_memory():
    cp.get_default_memory_pool().free_all_blocks()
    gc.collect()

def Init_K(N, INIT_K0):
    K = cp.zeros((N, N), dtype=dtype)
    for i in range(1, N):
        K[i, i-1] = K[i-1, i] = INIT_K0
    return K

def K2P(K):
    N = K.shape[0]
    d = cp.sum(K, axis=0)
    L = cp.diag(d) - K
    L_sub = L[1:N, 1:N]
    
    Q = cp.linalg.inv(L_sub)
    M = 0.5 * (Q + Q.T)
    A = cp.diag(M)
    
    G = cp.zeros((N, N), dtype=dtype)
    G[1:N, 1:N] = -2*M + A[:, None] + A[None, :]
    G[0, 1:N] = A
    G[1:N, 0] = A
    
    P = cp.power(1.0 + 3.0 * G, -1.5)
    return P

def compute_cost(P_dif):
    return cp.sqrt(cp.mean(P_dif * P_dif))

def estimate_optimal_eta(P_obs, K_init):
    """
    Estimate optimal initial learning rate based on matrix properties
    Uses a few test iterations to find stable learning rate
    """
    print("\n" + "="*80)
    print("Auto-tuning learning rate...")
    print("="*80)
    
    # Compute initial gradient
    P_init = K2P(K_init)
    gradient = P_init - P_obs
    grad_norm = cp.sqrt(cp.mean(gradient * gradient))
    
    print(f"Initial gradient norm: {grad_norm:.5e}")
    
    # Try different learning rates
    test_etas = [1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6]
    best_eta = 1e-5
    
    for eta_test in test_etas:
        K_test = K_init - eta_test * gradient
        P_test = K2P(K_test)
        cost_new = compute_cost(P_test - P_obs)
        cost_old = grad_norm
        
        improvement = cost_old - cost_new
        print(f"  ETA={eta_test:.1e}: improvement={improvement:+.5e}")
        
        if improvement > 0 and improvement < cost_old * 0.1:
            # Found a good learning rate (makes progress but not too aggressive)
            best_eta = eta_test
            print(f"✓ Selected ETA: {best_eta:.1e}")
            break
        elif improvement < 0:
            # Too large, but we can still use smaller value
            continue
        elif improvement > cost_old * 0.5:
            # Too aggressive, try smaller
            continue
    
    return best_eta

def phic2_optimized(K, P_obs, ETA, ALPHA=1e-10, ITERATION_MAX=1000000):
    """
    Optimized PHi-C2 with conservative updates
    """
    N = K.shape[0]
    
    print("\n" + "="*80)
    print("Starting PHi-C2 Optimization")
    print("="*80)
    print(f"Parameters:")
    print(f"  Learning rate (ETA): {ETA:.2e}")
    print(f"  Convergence (ALPHA): {ALPHA:.2e}")
    print(f"  Max iterations: {ITERATION_MAX:,}")
    print(f"  Stop delta: {ETA * ALPHA:.2e}")
    print()
    
    # Initialize
    P_dif = K2P(K) - P_obs
    cost = compute_cost(P_dif)
    
    c_traj = []
    c_traj.append([cost, time.time()])
    
    print(f"Initial cost: {cost:.5e}")
    
    iteration = 0
    last_print_time = time.time()
    best_cost = cost
    no_improvement_count = 0
    
    # For detecting oscillations
    cost_window = [cost]
    window_size = 10
    
    stop_delta = ETA * ALPHA
    
    while iteration < ITERATION_MAX:
        iteration += 1
        cost_prev = cost
        
        # Simple gradient descent step
        K = K - ETA * P_dif
        
        # Compute new state
        P_dif = K2P(K) - P_obs
        cost = compute_cost(P_dif)
        
        cost_diff = cost_prev - cost
        
        # Track cost
        cost_window.append(cost)
        if len(cost_window) > window_size:
            cost_window.pop(0)
        
        c_traj.append([cost, time.time()])
        
        # Update best
        if cost < best_cost:
            best_cost = cost
            no_improvement_count = 0
        else:
            no_improvement_count += 1
        
        # Progress reporting
        current_time = time.time()
        if iteration == 1 or current_time - last_print_time > 10:
            elapsed = current_time - c_traj[0][1]
            iter_per_sec = iteration / elapsed if elapsed > 0 else 0
            eta_time = (ITERATION_MAX - iteration) / iter_per_sec if iter_per_sec > 0 else 0
            
            print(f"Iter {iteration:7d} | Cost: {cost:.5e} | ΔCost: {cost_diff:+.5e} | "
                  f"Best: {best_cost:.5e} | Speed: {iter_per_sec:.2f} it/s | "
                  f"Time: {elapsed/60:.1f}m")
            print_memory_usage()
            last_print_time = current_time
        
        # Convergence checks
        if abs(cost_diff) < stop_delta and iteration > 100:
            print(f"\n✓ Converged! Cost change ({abs(cost_diff):.2e}) < threshold ({stop_delta:.2e})")
            break
        
        if no_improvement_count > 5000:
            print(f"\n⚠ No improvement for 5000 iterations. Current cost: {cost:.5e}")
            # Check if we're oscillating around minimum
            if len(cost_window) == window_size:
                cost_std = np.std(cost_window)
                if cost_std < stop_delta * 10:
                    print(f"  Cost variance is low ({cost_std:.2e}). Likely at minimum.")
                    break
            
        if iteration > 10000 and no_improvement_count > 2000:
            print(f"\n⚠ Slow progress after {iteration} iterations.")
            print(f"  Current: {cost:.5e}, Best: {best_cost:.5e}")
            
            # Check if essentially converged
            recent_improvement = cost_window[0] - cost_window[-1]
            if abs(recent_improvement) < stop_delta * 10:
                print(f"  Recent improvement very small. Stopping.")
                break
        
        # Periodic cleanup
        if iteration % 100 == 0:
            clear_gpu_memory()
    
    c_traj = np.array(c_traj)
    paras_fit = f"{ETA}\t{ALPHA}\t{iteration}\t"
    
    print(f"\nOptimization completed:")
    print(f"  Total iterations: {iteration:,}")
    print(f"  Final cost: {cost:.5e}")
    print(f"  Best cost: {best_cost:.5e}")
    print(f"  Improvement: {c_traj[0,0] - best_cost:.5e}")
    
    return [K, c_traj, paras_fit]

def saveLg(fn, xy, ct):
    with open(fn, 'w') as fw:
        fw.write(ct)
        for row in xy:
            if xy.ndim == 2:
                line = ' '.join(f"{v:.5e}" if not np.isnan(v) else "NaN" for v in row)
            else:
                line = f"{row:.5e}" if not np.isnan(row) else "NaN"
            fw.write(line + '\n')

def saveMx(fn, xy, ct, chunk_size=1000):
    print(f"Saving {fn}...")
    with open(fn, 'w') as fw:
        fw.write(ct)
        n = len(xy)
        for chunk_start in range(0, n, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n)
            for i in range(chunk_start, chunk_end):
                line = ' '.join(f"{v:.5e}" if not np.isnan(v) else "NaN" for v in xy[i])
                fw.write(line + '\n')
            if chunk_start % 5000 == 0 and chunk_start > 0:
                print(f"  {chunk_start}/{n} rows...")
    print(f"  Done!")

# Main execution
if __name__ == "__main__":
    print("="*80)
    print("PHi-C2 Auto-Tuned for Large-Scale Hi-C Matrices")
    print("="*80)
    
    # Load matrix
    if not os.path.isfile(fhic):
        print(f'Error: Cannot find {fhic}')
        sys.exit(1)
    
    print(f"\nLoading: {fhic}")
    start_load = time.time()
    
    P_obs_list = []
    line_count = 0
    
    with open(fhic) as fr:
        for line in fr:
            if not line[0] == '#':
                row = list(map(float, line.strip().split()))
                P_obs_list.append(row)
                line_count += 1
                if line_count % 5000 == 0:
                    print(f"  {line_count} rows loaded...")
    
    N = len(P_obs_list)
    
    print(f"Converting to GPU...")
    P_obs = cp.array(P_obs_list, dtype=dtype)
    del P_obs_list
    gc.collect()
    
    load_time = time.time() - start_load
    print(f"Loaded {N}×{N} matrix in {load_time:.1f}s")
    print_memory_usage()
    
    # Matrix analysis
    print("\nMatrix properties:")
    cp.nan_to_num(P_obs, copy=False)
    
    nonzero = cp.count_nonzero(P_obs)
    print(f"  Non-zero elements: {int(nonzero):,}")
    print(f"  Sparsity: {100*(1 - nonzero/(N*N)):.2f}%")
    print(f"  Range: [{float(cp.min(P_obs)):.2e}, {float(cp.max(P_obs)):.2e}]")
    print(f"  Mean: {float(cp.mean(P_obs)):.2e}")
    print(f"  Std: {float(cp.std(P_obs)):.2e}")
    
    # Add diagonal
    P_obs = P_obs + cp.eye(N, dtype=dtype)
    
    # Initialize K
    print("\nInitializing spring constant matrix...")
    K_fit = Init_K(N, INIT_K0=0.5)
    
    # Auto-tune learning rate
    optimal_eta = estimate_optimal_eta(P_obs, K_fit)
    
    # Run optimization
    start_opt = time.time()
    K_fit, c_traj, paras_fit = phic2_optimized(
        K_fit, P_obs,
        ETA=optimal_eta,
        ALPHA=1e-10,
        ITERATION_MAX=1000000
    )
    opt_time = time.time() - start_opt
    
    print(f"\nTotal optimization time: {opt_time/60:.1f} minutes")
    
    # Compute final P
    print("\nComputing final contact probabilities...")
    P_fit = K2P(K_fit)
    
    # Transfer to CPU
    print("Transferring to CPU...")
    K_fit = cp.asnumpy(K_fit)
    P_fit = cp.asnumpy(P_fit)
    P_obs = cp.asnumpy(P_obs)
    clear_gpu_memory()
    
    # Save results
    print("\n" + "="*80)
    print("Saving Results")
    print("="*80)
    
    precision = "fp32" if USE_FP32 else "fp64"
    dataDir = fhic[:fhic.rfind('.')] + f"_phic2_auto_{precision}"
    os.makedirs(dataDir, exist_ok=True)
    fo = f"{dataDir}/N{N}"
    
    # Log
    c_traj[:, 1] -= c_traj[0, 1]
    saveLg(fo + '.log', c_traj, f"#{paras_fit}\n#cost time\n")
    
    # K matrix
    saveMx(fo + '.K_fit', K_fit,
           f"#K_fit N={N} range=[{np.min(K_fit):.5e}, {np.max(K_fit):.5e}]\n")
    
    # Pearson correlation
    print("\nComputing Pearson correlation...")
    tri_mask = np.triu_indices(N, k=1)
    nonzero_mask = np.where(P_obs[tri_mask] > 0)
    
    r_all = pearsonr(P_fit[tri_mask], P_obs[tri_mask])[0]
    r_nonzero = pearsonr(
        P_fit[tri_mask][nonzero_mask],
        P_obs[tri_mask][nonzero_mask]
    )[0]
    
    # P matrix
    saveMx(fo + '.P_fit', P_fit,
           f"#P_fit N={N} range=[{np.nanmin(P_fit):.5e}, {np.nanmax(P_fit):.5e}] "
           f"pearson_all={r_all:.5f} pearson_nonzero={r_nonzero:.5f}\n")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Matrix:               {N}×{N}")
    print(f"Precision:            {precision}")
    print(f"Learning rate:        {optimal_eta:.2e}")
    print(f"Load time:            {load_time:.1f}s")
    print(f"Optimization time:    {opt_time/60:.1f} min")
    print(f"Iterations:           {len(c_traj):,}")
    print(f"Initial cost:         {c_traj[0,0]:.5e}")
    print(f"Final cost:           {c_traj[-1,0]:.5e}")
    print(f"Pearson (all):        {r_all:.5f}")
    print(f"Pearson (nonzero):    {r_nonzero:.5f}")
    print(f"Output:               {dataDir}/")
    print("="*80)
