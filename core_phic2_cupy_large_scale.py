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

# Highly stable version for large-scale Hi-C matrices
# Key improvements:
# 1. Better initialization strategy
# 2. More conservative learning rate adaptation
# 3. Momentum-based optimization
# 4. Better convergence detection

if not len(sys.argv) >= 2:
    print("usage:: python core_phic2_cupy_stable_v2.py normalized-HiC-Contact-Matrix [--fp32] [--eta=1e-5]")
    print("  --fp32: Use float32 instead of float64 (recommended)")
    print("  --eta=X: Initial learning rate (default: 1e-5)")
    sys.exit()

fhic = str(sys.argv[1])

# Parse arguments
USE_FP32 = '--fp32' in sys.argv
INIT_ETA = 1e-6

for arg in sys.argv:
    if arg.startswith('--eta='):
        INIT_ETA = float(arg.split('=')[1])

dtype = cp.float32 if USE_FP32 else cp.float64
np_dtype = np.float32 if USE_FP32 else np.float64

print(f"Using precision: {'float32' if USE_FP32 else 'float64'}")
print(f"Initial learning rate (ETA): {INIT_ETA}")

def print_memory_usage():
    """Print current CPU and GPU memory usage"""
    process = psutil.Process()
    cpu_mem_gb = process.memory_info().rss / 1024**3
    
    gpu_mem = cp.get_default_memory_pool().used_bytes() / 1024**3
    gpu_total = cp.cuda.Device().mem_info[1] / 1024**3
    
    print(f"Memory - CPU: {cpu_mem_gb:.2f}GB, GPU: {gpu_mem:.2f}/{gpu_total:.2f}GB")

def clear_gpu_memory():
    """Clear GPU memory cache"""
    cp.get_default_memory_pool().free_all_blocks()
    gc.collect()

def Init_K(N, INIT_K0):
    """Initialize K matrix with better diagonal handling"""
    K = cp.zeros((N, N), dtype=dtype)
    # Off-diagonal elements for adjacent beads
    for i in range(1, N):
        j = i - 1
        K[i, j] = K[j, i] = INIT_K0
    return K

def K2P(K):
    """Convert spring constant matrix to contact probability"""
    N = K.shape[0]
    
    # Compute Laplacian
    d = cp.sum(K, axis=0)
    L = cp.diag(d) - K
    
    # Solve for inverse of submatrix
    L_sub = L[1:N, 1:N]
    
    try:
        # Use Cholesky if possible (faster and more stable)
        L_chol = cp.linalg.cholesky(L_sub)
        Q = cp.linalg.solve(L_chol.T, cp.linalg.solve(L_chol, cp.eye(N-1, dtype=dtype)))
    except:
        # Fall back to direct inversion
        Q = cp.linalg.inv(L_sub)
    
    # Compute mean squared distance matrix
    M = 0.5 * (Q + Q.T)  # Ensure symmetry
    A = cp.diag(M)
    
    # Build full distance matrix
    G = cp.zeros((N, N), dtype=dtype)
    G[1:N, 1:N] = -2*M + A[:, None] + A[None, :]
    G[0, 1:N] = A
    G[1:N, 0] = A
    
    # Convert to contact probability
    # P = (1 + 3*G)^(-3/2) with numerical stability
    G_safe = cp.maximum(G, -0.333)  # Prevent division issues
    P = cp.power(1.0 + 3.0 * G_safe, -1.5)
    
    return P

def compute_cost(P_dif):
    """Compute cost with numerical stability"""
    N = P_dif.shape[0]
    cost = cp.sqrt(cp.sum(P_dif * P_dif) / N)
    return cost

def phic2_momentum(K, P_obs, ETA=1e-5, ALPHA=1e-10, ITERATION_MAX=1000000, 
                   BETA=0.9, MIN_ETA=1e-10):
    """
    PHi-C2 optimization with momentum and adaptive learning rate
    
    Parameters:
    - BETA: Momentum coefficient (default 0.9)
    - MIN_ETA: Minimum learning rate before stopping
    """
    stop_delta = ETA * ALPHA
    N = K.shape[0]
    
    print("="*80)
    print("PHi-C2 with Momentum - Stable Version")
    print("="*80)
    print(f"Parameters:")
    print(f"  Initial ETA: {ETA}")
    print(f"  ALPHA: {ALPHA}")
    print(f"  Momentum (BETA): {BETA}")
    print(f"  Min ETA: {MIN_ETA}")
    print(f"  Max iterations: {ITERATION_MAX}")
    print()
    
    # Initialize
    print("Computing initial cost...")
    P_dif = K2P(K) - P_obs
    cost = compute_cost(P_dif)
    
    # Momentum term
    velocity = cp.zeros_like(K)
    
    # Tracking
    c_traj = []
    c_traj.append([cost, time.time(), ETA])
    
    print(f"Initial cost: {cost:.5e}")
    print(f"Stop delta: {stop_delta:.5e}")
    print()
    
    iteration = 0
    last_print_time = time.time()
    best_cost = cost
    stagnant_count = 0
    eta_reductions = 0
    
    # For adaptive ETA
    cost_history = [cost]
    window_size = 5
    
    while iteration < ITERATION_MAX:
        iteration += 1
        cost_prev = cost
        
        # Update with momentum
        gradient = P_dif
        velocity = BETA * velocity + (1 - BETA) * gradient
        K_new = K - ETA * velocity
        
        # Compute new cost
        P_dif_new = K2P(K_new) - P_obs
        cost_new = compute_cost(P_dif_new)
        
        cost_diff = cost_prev - cost_new
        
        # Check for improvement
        if cost_new < cost_prev:
            # Accept update
            K = K_new
            P_dif = P_dif_new
            cost = cost_new
            
            if cost < best_cost:
                best_cost = cost
                stagnant_count = 0
            else:
                stagnant_count += 1
            
            # Adaptive learning rate increase (carefully)
            if iteration > 10 and cost_diff > stop_delta * 2:
                # Only increase if we're making good progress
                if len(cost_history) >= window_size:
                    recent_improvement = cost_history[-window_size] - cost
                    if recent_improvement > 0:
                        ETA = min(ETA * 1.05, INIT_ETA * 2)  # Cap at 2x initial
        else:
            # Reject update and reduce learning rate
            eta_reductions += 1
            ETA *= 0.5
            velocity *= 0  # Reset momentum on reduction
            
            if ETA < MIN_ETA:
                print(f"\nLearning rate too small ({ETA:.2e}). Stopping.")
                break
            
            # Don't update K, P_dif, or cost
            print(f"Iter {iteration:6d} | Rejected step, reducing ETA to {ETA:.2e}")
            continue
        
        # Track cost
        cost_history.append(cost)
        if len(cost_history) > 100:
            cost_history.pop(0)
        
        c_traj.append([cost, time.time(), ETA])
        
        # Progress reporting
        current_time = time.time()
        if current_time - last_print_time > 10:
            elapsed = current_time - c_traj[0][1]
            iter_per_sec = iteration / elapsed if elapsed > 0 else 0
            eta_time = (ITERATION_MAX - iteration) / iter_per_sec if iter_per_sec > 0 else 0
            
            print(f"Iter {iteration:7d} | Cost: {cost:.5e} | ΔCost: {cost_diff:+.5e} | "
                  f"ETA: {ETA:.2e} | Speed: {iter_per_sec:.2f} it/s | "
                  f"Time: {elapsed/60:.1f}m | Est: {eta_time/60:.1f}m")
            print_memory_usage()
            last_print_time = current_time
        
        # Convergence checks
        if abs(cost_diff) < stop_delta and iteration > 100:
            print(f"\n✓ Converged at iteration {iteration}")
            print(f"  Cost change: {cost_diff:.5e} < {stop_delta:.5e}")
            break
        
        if stagnant_count > 1000:
            print(f"\n⚠ No improvement for 1000 iterations. Stopping.")
            break
        
        if eta_reductions > 20:
            print(f"\n⚠ Too many learning rate reductions ({eta_reductions}). Stopping.")
            break
        
        # Periodic cleanup
        if iteration % 100 == 0:
            clear_gpu_memory()
    
    c_traj = np.array(c_traj)
    paras_fit = f"{INIT_ETA}\t{ALPHA}\t{BETA}\t{iteration}\t"
    
    print(f"\nFinal statistics:")
    print(f"  Iterations: {iteration}")
    print(f"  Final cost: {cost:.5e}")
    print(f"  Best cost: {best_cost:.5e}")
    print(f"  Final ETA: {ETA:.5e}")
    print(f"  ETA reductions: {eta_reductions}")
    
    return [K, c_traj, paras_fit]

def saveLg(fn, xy, ct):
    """Save array to file"""
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

def saveMx(fn, xy, ct, chunk_size=1000):
    """Save matrix to file in chunks"""
    print(f"Saving matrix to {fn}...")
    fw = open(fn, 'w')
    fw.write(ct)
    n = len(xy)
    
    for chunk_start in range(0, n, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n)
        for i in range(chunk_start, chunk_end):
            lt = ''
            for v in xy[i]:
                lt += "%11s " % ('NaN') if np.isnan(v) else "%11.5e " % (v)
            fw.write(lt + '\n')
        
        if chunk_start % 5000 == 0 and chunk_start > 0:
            print(f"  {chunk_start}/{n} rows written...")
    
    fw.close()
    print(f"  Completed!")

# Main execution
if __name__ == "__main__":
    print("="*80)
    print("PHi-C2 Large-Scale - Stable Version with Momentum")
    print("="*80)
    
    # Load Hi-C matrix
    if not os.path.isfile(fhic):
        print(f'Cannot find {fhic}')
        sys.exit()
    
    print(f"\nLoading Hi-C matrix from: {fhic}")
    start_load = time.time()
    
    P_obs_list = []
    line_count = 0
    
    with open(fhic) as fr:
        for line in fr:
            if not line[0] == '#':
                lt = line.strip().split()
                row = list(map(float, lt))
                P_obs_list.append(row)
                line_count += 1
                
                if line_count % 5000 == 0:
                    print(f"  Loaded {line_count} rows...")
    
    N = len(P_obs_list)
    
    print(f"Converting to GPU array...")
    P_obs = cp.array(P_obs_list, dtype=dtype)
    del P_obs_list
    gc.collect()
    
    load_time = time.time() - start_load
    matrix_size_gb = N * N * (4 if USE_FP32 else 8) / 1024**3
    
    print(f"Loaded matrix: {N}×{N} in {load_time:.1f}s")
    print(f"Matrix size: {matrix_size_gb:.2f}GB")
    print_memory_usage()
    
    # Preprocessing
    print("Matrix properties:")
    nonzero_count = cp.count_nonzero(P_obs)
    sparsity = 1 - (nonzero_count / (N * N))
    print(f"  Non-zero elements: {int(nonzero_count)}")
    print(f"  Sparsity: {sparsity*100:.2f}%")
    print(f"  Min value: {cp.min(P_obs):.5e}")
    print(f"  Max value: {cp.max(P_obs):.5e}")
    print(f"  Mean value: {cp.mean(P_obs):.5e}")
    
    cp.nan_to_num(P_obs, copy=False)
    P_obs = P_obs + cp.eye(N, dtype=dtype)
    
    # Initialize K
    print("\nInitializing K matrix...")
    K_fit = Init_K(N, INIT_K0=0.5)
    
    # Optimization
    print("\n" + "="*80)
    print("Starting optimization with momentum...")
    print("="*80)
    
    start_opt = time.time()
    K_fit, c_traj, paras_fit = phic2_momentum(
        K_fit, P_obs,
        ETA=INIT_ETA,
        ALPHA=1e-10,
        ITERATION_MAX=1000000,
        BETA=0.5,
        MIN_ETA=1e-10
    )
    opt_time = time.time() - start_opt
    
    print(f"\nOptimization completed in {opt_time:.1f}s ({opt_time/60:.1f} min)")
    
    # Final P
    print("\nComputing final P_fit...")
    P_fit = K2P(K_fit)
    
    # Transfer to CPU
    print("Transferring results to CPU...")
    K_fit = cp.asnumpy(K_fit)
    P_fit = cp.asnumpy(P_fit)
    P_obs = cp.asnumpy(P_obs)
    
    clear_gpu_memory()
    
    # Save results
    print("\n" + "="*80)
    print("Saving results")
    print("="*80)
    
    precision_str = "fp32" if USE_FP32 else "fp64"
    dataDir = fhic[:fhic.rfind('.')] + f"_phic2_momentum_{precision_str}"
    os.makedirs(dataDir, exist_ok=True)
    fo = f"{dataDir}/N{N}"
    
    # Save log
    c_traj[:, 1] = c_traj[:, 1] - c_traj[0, 1]
    saveLg(fo + '.log', c_traj, f"#{paras_fit}\n#cost time eta\n")
    
    # Save K
    saveMx(fo + '.K_fit', K_fit,
           f"#K_fit N {N} min: {np.min(K_fit):.5e} max: {np.max(K_fit):.5e}\n")
    
    # Compute Pearson correlation
    print("\nComputing Pearson correlation...")
    triMask = np.where(np.triu(np.ones((N, N)), 1) > 0)
    pijMask = np.where(np.triu(P_obs, 1) > 0)
    
    p1 = pearsonr(P_fit[triMask], P_obs[triMask])[0]
    p2 = pearsonr(P_fit[pijMask], P_obs[pijMask])[0]
    
    # Save P_fit
    saveMx(fo + '.P_fit', P_fit,
           f"#P_fit N {N} min: {np.nanmin(P_fit):.5e} max: {np.nanmax(P_fit):.5e} "
           f"pearson: {p1:.5e} {p2:.5e}\n")
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"Matrix size:           {N}×{N}")
    print(f"Precision:             {precision_str}")
    print(f"Load time:             {load_time/60:.1f} min")
    print(f"Optimization time:     {opt_time/60:.1f} min")
    print(f"Total iterations:      {len(c_traj)}")
    print(f"Final cost:            {c_traj[-1,0]:.5e}")
    print(f"Pearson (all pairs):   {p1:.5f}")
    print(f"Pearson (p>0 pairs):   {p2:.5f}")
    print(f"Output directory:      {dataDir}")
    print("="*80)
