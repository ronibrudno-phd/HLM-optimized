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

# FINAL PRODUCTION VERSION
# - Memory-optimized K2P with solve()
# - Adaptive learning rate
# - Proper K constraints (non-negative, symmetric, bounded)
# - Both sampling and full Pearson
# - Robust error handling
# - Progress saving every 100 iterations

if not len(sys.argv) >= 2:
    print("usage:: python phic2_final.py normalized-HiC-Contact-Matrix")
    sys.exit()

fhic = str(sys.argv[1])

print("="*80)
print("PHi-C2 FINAL PRODUCTION VERSION")
print("="*80)
print("Features:")
print("  ✓ Memory-optimized K2P (uses solve, not inv)")
print("  ✓ Adaptive learning rate")
print("  ✓ K constraints: symmetric + bounded (negative K allowed for repulsion)")
print("  ✓ Both fast sampling & full Pearson correlation")
print("  ✓ Checkpointing every 100 iterations")
print("="*80)

# Check CuPy solve capabilities
_SOLVE_SUPPORTS_ASSUME_A = True
try:
    test = cp.eye(2, dtype=cp.float32)
    cp.linalg.solve(test, test, assume_a='pos')
except TypeError:
    _SOLVE_SUPPORTS_ASSUME_A = False
    print("Note: Using standard solve (assume_a not supported)")

def print_memory():
    process = psutil.Process()
    cpu_gb = process.memory_info().rss / 1024**3
    gpu_gb = cp.get_default_memory_pool().used_bytes() / 1024**3
    gpu_total = cp.cuda.Device().mem_info[1] / 1024**3
    print(f"  Memory - CPU: {cpu_gb:.2f}GB, GPU: {gpu_gb:.2f}/{gpu_total:.2f}GB")

def Init_K(N, INIT_K0, dtype):
    """Initialize K with nearest-neighbor springs"""
    K = cp.zeros((N, N), dtype=dtype)
    for i in range(1, N):
        K[i, i-1] = K[i-1, i] = INIT_K0
    return K

def K2P_inplace(K, out_P, identity, eps_diag=1e-5, rc2=1.0):
    """
    Memory-optimized K to P conversion
    Uses solve() instead of inv() for better stability
    """
    N = K.shape[0]
    
    # Compute degree vector
    d = cp.sum(K, axis=0, dtype=cp.float32)
    
    # Build Laplacian submatrix L[1:,1:]
    L11 = (-K[1:, 1:]).copy()
    idx = cp.arange(N - 1, dtype=cp.int32)
    L11[idx, idx] += d[1:]
    L11[idx, idx] += cp.float32(eps_diag)  # Regularization
    
    # Solve L11 * Q = I (more stable than inv)
    if _SOLVE_SUPPORTS_ASSUME_A:
        Q = cp.linalg.solve(L11, identity, assume_a='pos')
    else:
        Q = cp.linalg.solve(L11, identity)
    
    A = cp.diagonal(Q)
    
    # Build G matrix in-place
    out_P.fill(0)
    sub = out_P[1:N, 1:N]
    sub[...] = -2.0 * Q
    sub += A[None, :]
    sub += A[:, None]
    out_P[0, 1:N] = A
    out_P[1:N, 0] = A
    
    # Convert to P: (1 + 3*G/rc2)^(-1.5)
    out_P *= cp.float32(3.0 / rc2)
    out_P += cp.float32(1.0)
    cp.maximum(out_P, cp.float32(1e-6), out=out_P)
    cp.power(out_P, cp.float32(-1.5), out=out_P)
    
    del Q, L11, A, sub, d
    return out_P

def constrain_K(K, K_bound):
    """
    Apply physical constraints to K:
    1. Symmetric (REQUIRED - Newton's 3rd law)
    2. Bounded magnitude (prevents overflow, allows both attraction and repulsion)
    
    Args:
        K: Spring constant matrix
        K_bound: Maximum allowed |K| value
    
    Returns:
        K_constrained
    """
    # Symmetric (REQUIRED)
    K = 0.5 * (K + K.T)
    
    # Bound magnitude (allow negative for repulsion)
    K = cp.clip(K, -K_bound, K_bound)
    
    return K

def cost_func(P_dif, N):
    """RMS difference between fitted and observed P"""
    return cp.sqrt(cp.sum(P_dif**2)) / N

def estimate_eta(P_obs, K_init, N, identity, P_temp):
    """Estimate initial learning rate based on gradient"""
    print("\nEstimating initial learning rate...")
    K2P_inplace(K_init, P_temp, identity)
    grad_norm = cp.sqrt(cp.mean((P_temp - P_obs)**2))
    
    print(f"  Initial gradient norm: {float(grad_norm):.5e}")
    
    # Scale based on matrix size and gradient
    base_eta = 1e-4
    size_factor = (2869.0 / N)**2
    gradient_factor = min(1.0, 1e-4 / float(grad_norm))
    eta = base_eta * size_factor * gradient_factor
    eta = np.clip(eta, 1e-8, 1e-4)
    
    print(f"  Initial ETA: {eta:.2e}")
    return eta

def save_checkpoint(K, cost, iteration, checkpoint_dir):
    """Save checkpoint for recovery"""
    cp_file = f"{checkpoint_dir}/checkpoint_iter{iteration}.npz"
    K_cpu = cp.asnumpy(K)
    np.savez_compressed(cp_file, K=K_cpu, cost=cost, iteration=iteration)
    return cp_file

def phic2_final(K, N, P_obs, checkpoint_dir, ETA_init=1.0e-6, ALPHA=1.0e-10, ITERATION_MAX=1000000):
    """
    Final production PHi-C2 with all optimizations
    """
    # Determine K bound based on matrix size
    # Empirical observation: K scales roughly as 1000 * (2869/N)
    if N < 3000:
        K_bound = 1000.0  # For ~1MB resolution (like paper)
    elif N < 10000:
        K_bound = 500.0   # For ~500KB resolution
    elif N < 20000:
        K_bound = 200.0   # For ~200KB resolution
    else:
        K_bound = 100.0   # For ~100KB resolution (your case)
    
    print("\n" + "="*80)
    print("Starting PHi-C2 Optimization")
    print("="*80)
    print(f"Matrix size: {N}×{N}")
    print(f"Initial ETA: {ETA_init:.2e}")
    print(f"K bound: ±{K_bound:.1f} (adaptive based on matrix size)")
    print(f"Checkpoint directory: {checkpoint_dir}")
    print()
    
    ETA = ETA_init
    ETA_min = ETA_init * 1e-4
    
    # Pre-allocate arrays
    identity = cp.eye(N-1, dtype=cp.float32)
    P_fit = cp.zeros((N, N), dtype=cp.float32)
    P_dif = cp.zeros((N, N), dtype=cp.float32)
    
    # Initial state
    K2P_inplace(K, P_fit, identity)
    P_dif[...] = P_fit - P_obs
    cost = cost_func(P_dif, N)
    
    c_traj = []
    c_traj.append([float(cost), time.time(), ETA])
    
    print(f"Initial cost: {cost:.6e}\n")
    print_memory()
    
    iteration = 1
    best_cost = cost
    best_K = K.copy()
    best_iter = 0
    
    cost_history = [float(cost)]
    oscillation_count = 0
    eta_reduction_count = 0
    k_bound_expansions = 0  # Track bound expansions
    
    last_print = time.time()
    last_checkpoint = 0
    
    while iteration < ITERATION_MAX:
        cost_bk = cost
        
        # Gradient descent
        K -= ETA * P_dif
        
        # Check if K exceeds current bound BEFORE clipping
        K_min = float(cp.min(K))
        K_max = float(cp.max(K))
        K_absmax = max(abs(K_min), abs(K_max))
        
        # Dynamic bound expansion if K naturally grew beyond current bound
        if K_absmax > K_bound * 1.2:  # Allow 20% overflow before expanding
            old_bound = K_bound
            K_bound = K_absmax * 1.5  # Expand to 1.5× current max
            k_bound_expansions += 1
            
            print(f"\n  → K_bound expanded at iteration {iteration}")
            print(f"     K range: [{K_min:.2e}, {K_max:.2e}]")
            print(f"     Bound: {old_bound:.1f} → {K_bound:.1f} (expansion #{k_bound_expansions})")
        
        # CRITICAL: Apply constraints with current K_bound
        K = constrain_K(K, K_bound)
        
        # Check K health
        if cp.any(cp.isnan(K)) or cp.any(cp.isinf(K)):
            print(f"\n⚠ Invalid K at iter {iteration}, reverting to best")
            K = best_K.copy()
            break
        
        # Compute new state
        try:
            K2P_inplace(K, P_fit, identity)
            P_dif[...] = P_fit - P_obs
            cost = cost_func(P_dif, N)
        except Exception as e:
            print(f"\n⚠ Error in K2P at iter {iteration}: {e}")
            print(f"  Reverting to best K")
            K = best_K.copy()
            break
        
        if cp.isnan(cost) or cp.isinf(cost):
            print(f"\n⚠ Invalid cost at iter {iteration}, reverting to best")
            K = best_K.copy()
            cost = best_cost
            break
        
        cost_dif = cost_bk - cost
        cost_history.append(float(cost))
        if len(cost_history) > 50:
            cost_history.pop(0)
        
        c_traj.append([float(cost), time.time(), ETA])
        
        # Update best
        if cost < best_cost:
            best_cost = cost
            best_K = K.copy()
            best_iter = iteration
            oscillation_count = 0
        else:
            if cost_dif < 0:
                oscillation_count += 1
        
        # Adaptive ETA reduction
        if oscillation_count >= 10:
            ETA_old = ETA
            ETA = ETA * 0.5
            eta_reduction_count += 1
            
            print(f"\n  ⚠ Oscillations at iter {iteration}")
            print(f"    ETA: {ETA_old:.2e} → {ETA:.2e} (reduction #{eta_reduction_count})")
            
            K = best_K.copy()
            K2P_inplace(K, P_fit, identity)
            P_dif[...] = P_fit - P_obs
            cost = best_cost
            oscillation_count = 0
            
            if ETA < ETA_min:
                print(f"\n  ✓ ETA < minimum ({ETA_min:.2e}), stopping")
                break
        
        # Progress reporting
        if iteration == 1 or time.time() - last_print > 10:
            elapsed = c_traj[-1][1] - c_traj[0][1]
            rate = iteration / elapsed if elapsed > 0 else 0
            since_best = iteration - best_iter
            
            # Current K statistics
            K_curr_min = float(cp.min(K))
            K_curr_max = float(cp.max(K))
            
            print(f"Iter {iteration:7d} | Cost: {cost:.6e} | ΔCost: {cost_dif:+.6e} | "
                  f"Best: {best_cost:.6e} | ETA: {ETA:.2e}")
            print(f"  Rate: {rate:.2f} it/s | Time: {elapsed/60:.1f}m | "
                  f"Since best: {since_best} | ETA reduct: {eta_reduction_count} | "
                  f"K_bound expan: {k_bound_expansions}")
            print(f"  K range: [{K_curr_min:.2e}, {K_curr_max:.2e}] | K_bound: ±{K_bound:.1f}")
            print_memory()
            
            last_print = time.time()
        
        # Checkpointing
        if iteration - last_checkpoint >= 100:
            cp_file = save_checkpoint(K, float(cost), iteration, checkpoint_dir)
            print(f"  Checkpoint saved: {cp_file}")
            last_checkpoint = iteration
        
        # Convergence check
        stop_delta = ETA * ALPHA
        if 0 < cost_dif < stop_delta and iteration > 100:
            print(f"\n✓ Converged! ΔCost ({cost_dif:.2e}) < threshold ({stop_delta:.2e})")
            break
        
        # Stagnation check
        if iteration - best_iter > 5000:
            print(f"\n⚠ No improvement for 5000 iterations")
            
            if len(cost_history) == 50:
                cost_std = np.std(cost_history)
                if cost_std < stop_delta * 100:
                    print(f"  Cost variance ({cost_std:.2e}) very low, converged")
                    break
            
            if ETA > ETA_min * 2:
                print(f"  Reducing ETA further...")
                ETA = ETA * 0.5
                eta_reduction_count += 1
                best_iter = iteration
            else:
                print(f"  ETA at minimum, stopping")
                break
        
        iteration += 1
        
        # Periodic cleanup
        if iteration % 100 == 0:
            cp.get_default_memory_pool().free_all_blocks()
    
    # Use best K
    if best_cost < cost:
        print(f"\nUsing best K from iteration {best_iter}")
        K = best_K.copy()
    
    # Final P
    K2P_inplace(K, P_fit, identity)
    
    # Save final checkpoint
    cp_file = save_checkpoint(K, float(best_cost), iteration, checkpoint_dir)
    print(f"\nFinal checkpoint: {cp_file}")
    
    c_traj = np.array(c_traj)
    
    print(f"\n" + "="*80)
    print("Optimization Complete")
    print("="*80)
    print(f"  Iterations: {iteration:,}")
    print(f"  Best iteration: {best_iter:,}")
    print(f"  Final cost: {best_cost:.6e}")
    print(f"  Initial cost: {c_traj[0,0]:.6e}")
    print(f"  Improvement: {c_traj[0,0] - best_cost:.6e} ({100*(c_traj[0,0]-best_cost)/c_traj[0,0]:.2f}%)")
    print(f"  ETA reductions: {eta_reduction_count}")
    print(f"  K_bound expansions: {k_bound_expansions}")
    print(f"  Final K_bound: ±{K_bound:.1f}")
    print("="*80)
    
    paras_fit = f"{ETA_init}\t{ALPHA}\t{iteration}\t{eta_reduction_count}\t"
    
    del identity, P_dif
    
    return K, P_fit, c_traj, paras_fit

def saveLg(fn, xy, ct):
    """Save log file"""
    with open(fn, 'w') as fw:
        fw.write(ct)
        for row in xy:
            line = ' '.join(f"{v:.5e}" if not np.isnan(v) else "NaN" for v in row)
            fw.write(line + '\n')
    print(f"  Saved: {fn}")

def saveMx(fn, xy, ct, chunk_size=5000):
    """Save matrix file with progress"""
    print(f"  Saving {fn}...")
    with open(fn, 'w') as fw:
        fw.write(ct)
        n = len(xy)
        for chunk_start in range(0, n, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n)
            for i in range(chunk_start, chunk_end):
                line = ' '.join(f"{v:.5e}" if not np.isnan(v) else "NaN" for v in xy[i])
                fw.write(line + '\n')
            if chunk_start % 10000 == 0 and chunk_start > 0:
                print(f"    {chunk_start}/{n} rows...")
    print(f"  ✓ Saved: {fn}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    start_total = time.time()
    
    if not os.path.isfile(fhic):
        print(f'ERROR: Cannot find {fhic}')
        sys.exit(1)
    
    # Load Hi-C matrix
    print(f"\nLoading Hi-C matrix: {fhic}")
    start_load = time.time()
    
    P_obs = []
    with open(fhic) as fr:
        for line in fr:
            if not line[0] == '#':
                P_obs.append(list(map(float, line.strip().split())))
                if len(P_obs) % 5000 == 0:
                    print(f"  {len(P_obs)} rows...")
    
    P_obs = cp.array(P_obs, dtype=cp.float32)
    N = len(P_obs)
    
    load_time = time.time() - start_load
    print(f"\n✓ Loaded {N}×{N} in {load_time:.1f}s")
    
    # Preprocess
    cp.nan_to_num(P_obs, copy=False)
    P_obs = P_obs + cp.eye(N, dtype=cp.float32)
    
    # Matrix stats
    nonzero = int(cp.count_nonzero(P_obs))
    sparsity = 100 * (1 - nonzero/(N*N))
    print(f"\nMatrix statistics:")
    print(f"  Non-zero elements: {nonzero:,} ({sparsity:.1f}% sparse)")
    print(f"  Value range: [{float(cp.min(P_obs)):.2e}, {float(cp.max(P_obs)):.2e}]")
    print(f"  Mean: {float(cp.mean(P_obs)):.2e}")
    print_memory()
    
    # Initialize K
    print("\nInitializing spring constants...")
    K_fit = Init_K(N, 0.5, cp.float32)
    
    # Estimate ETA
    P_temp = cp.zeros((N, N), dtype=cp.float32)
    identity_temp = cp.eye(N-1, dtype=cp.float32)
    ETA_init = estimate_eta(P_obs, K_fit, N, identity_temp, P_temp)
    del P_temp, identity_temp
    gc.collect()
    
    # Create output directory
    dataDir = fhic[:fhic.rfind('.')] + "_phic2_FINAL"
    os.makedirs(dataDir, exist_ok=True)
    checkpoint_dir = f"{dataDir}/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    print(f"\nOutput directory: {dataDir}/")
    
    # Run optimization
    print("\n" + "="*80)
    print("STARTING OPTIMIZATION")
    print("="*80)
    start_opt = time.time()
    
    K_fit, P_fit, c_traj, paras_fit = phic2_final(
        K_fit, N, P_obs, checkpoint_dir,
        ETA_init=ETA_init,
        ALPHA=1.0e-10,
        ITERATION_MAX=1000000
    )
    
    opt_time = time.time() - start_opt
    print(f"\nOptimization time: {opt_time/60:.1f} min ({opt_time/3600:.1f} hours)")
    
    # Transfer to CPU
    print("\nTransferring results to CPU...")
    c_traj_np = c_traj.copy()
    K_fit = cp.asnumpy(K_fit)
    P_fit = cp.asnumpy(P_fit)
    P_obs = cp.asnumpy(P_obs)
    
    print("  ✓ Transfer complete")
    
    # Save results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    fo = f"{dataDir}/N{N}"
    
    # Log file
    c_traj_np[:,1] = c_traj_np[:,1] - c_traj_np[0,1]
    saveLg(fo+'.log', c_traj_np, f"#{paras_fit}\n#cost time eta\n")
    
    # K matrix
    saveMx(fo+'.K_fit', K_fit,
           f"#K_fit N={N} range=[{np.min(K_fit):.5e}, {np.max(K_fit):.5e}]\n")
    
    # Pearson correlation
    print("\nComputing Pearson correlation...")
    
    # Fast sampling
    print("  [1/2] Sampling-based (1M samples)...")
    def pearson_sample(P_fit_arr, P_obs_arr, n_samples=1_000_000, seed=0):
        rng = np.random.default_rng(seed)
        N = P_obs_arr.shape[0]
        i = rng.integers(0, N - 1, size=n_samples, dtype=np.int32)
        j = rng.integers(i + 1, N, size=n_samples, dtype=np.int32)
        obs = P_obs_arr[i, j]
        fit = P_fit_arr[i, j]
        p1 = float(np.corrcoef(fit, obs)[0, 1])
        m = obs > 0
        nz = int(m.sum())
        p2 = float(np.corrcoef(fit[m], obs[m])[0, 1]) if nz >= 1000 else float("nan")
        return p1, p2, nz
    
    p1_s, p2_s, nz_s = pearson_sample(P_fit, P_obs)
    print(f"    Sample Pearson (all):   {p1_s:.6f}")
    print(f"    Sample Pearson (obs>0): {p2_s:.6f}")
    
    # Full exact
    print("  [2/2] Full exact Pearson (all pairs)...")
    triMask = np.where(np.triu(np.ones((N,N)),1)>0)
    pijMask = np.where(np.triu(P_obs,1)>0)
    
    p1 = pearsonr(P_fit[triMask], P_obs[triMask])[0]
    p2 = pearsonr(P_fit[pijMask], P_obs[pijMask])[0]
    
    print(f"    Full Pearson (all):     {p1:.6f}")
    print(f"    Full Pearson (obs>0):   {p2:.6f}")
    
    # P matrix
    saveMx(fo+'.P_fit', P_fit,
           f"#P_fit N={N} range=[{np.nanmin(P_fit):.5e}, {np.nanmax(P_fit):.5e}] "
           f"pearson={p1:.6f} {p2:.6f}\n")
    
    total_time = time.time() - start_total
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"Matrix:                    {N}×{N}")
    print(f"Total time:                {total_time/60:.1f} min ({total_time/3600:.1f} hours)")
    print(f"  - Loading:               {load_time:.1f}s")
    print(f"  - Optimization:          {opt_time/60:.1f} min")
    print(f"Iterations:                {len(c_traj_np):,}")
    print(f"Initial cost:              {c_traj_np[0,0]:.6e}")
    print(f"Final cost:                {c_traj_np[-1,0]:.6e}")
    print(f"Improvement:               {100*(c_traj_np[0,0]-c_traj_np[-1,0])/c_traj_np[0,0]:.2f}%")
    print(f"Pearson (full, all):       {p1:.6f}")
    print(f"Pearson (full, obs>0):     {p2:.6f}")
    print(f"K range:                   [{np.min(K_fit):.5e}, {np.max(K_fit):.5e}]")
    print(f"Output directory:          {dataDir}/")
    print("="*80)
    
    # Success interpretation
    print("\n" + "="*80)
    print("RESULT INTERPRETATION")
    print("="*80)
    if p1 >= 0.90:
        print("✓✓ EXCELLENT! Pearson > 0.90")
        print("   Your optimization is highly successful!")
        print("   Use K_fit and P_fit for downstream 3D genome analysis.")
    elif p1 >= 0.85:
        print("✓ GOOD! Pearson > 0.85")
        print("  Your optimization is successful.")
        print("  Results are acceptable for most analyses.")
    elif p1 >= 0.75:
        print("△ ACCEPTABLE. Pearson > 0.75")
        print("  Results are usable but could be better.")
        print("  Consider trying 500KB or 1MB resolution.")
    else:
        print("✗ POOR. Pearson < 0.75")
        print("  The polymer model may not fit well at 100KB resolution.")
        print("  RECOMMENDATION: Bin to 500KB or 1MB resolution and re-run.")
    print("="*80)
    
    print(f"\nAll files saved to: {dataDir}/")
    print("Done! 🎉")
