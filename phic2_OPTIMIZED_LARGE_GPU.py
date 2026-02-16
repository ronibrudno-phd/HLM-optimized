# -*- coding: utf-8 -*-
import os
import sys
import time
import warnings
import cupy as cp
import numpy as np
from scipy.stats import pearsonr
import gc

# Try to import psutil, but make it optional
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("Warning: psutil not available")

cp.set_printoptions(precision=3, linewidth=200)
warnings.filterwarnings('ignore')

# OPTIMIZED VERSION FOR LARGE GPU
# Key improvements:
# 1. Better ETA estimation for large matrices
# 2. More aggressive optimization early on
# 3. Better convergence detection
# 4. Momentum-based updates
# 5. Better K_bound management

if not len(sys.argv) >= 2:
    print("usage:: python phic2_final.py normalized-HiC-Contact-Matrix")
    sys.exit()

fhic = str(sys.argv[1])

print("="*80)
print("PHi-C2 OPTIMIZED FOR LARGE GPU")
print("="*80)
print("Improvements:")
print("  [+] Optimized ETA estimation for large matrices")
print("  [+] Momentum-based gradient descent")
print("  [+] Better convergence detection")
print("  [+] Adaptive K_bound management")
print("="*80)

# Check CuPy solve capabilities
_SOLVE_SUPPORTS_ASSUME_A = True
try:
    test = cp.eye(2, dtype=cp.float32)
    cp.linalg.solve(test, test, assume_a='pos')
    del test
except TypeError:
    _SOLVE_SUPPORTS_ASSUME_A = False
    print("Note: Using standard solve (assume_a not supported)")

def print_memory():
    if HAS_PSUTIL:
        process = psutil.Process()
        cpu_gb = process.memory_info().rss / 1024**3
        gpu_gb = cp.get_default_memory_pool().used_bytes() / 1024**3
        gpu_total = cp.cuda.Device().mem_info[1] / 1024**3
        print(f"  Memory - CPU: {cpu_gb:.2f}GB, GPU: {gpu_gb:.2f}/{gpu_total:.2f}GB")
    else:
        gpu_gb = cp.get_default_memory_pool().used_bytes() / 1024**3
        gpu_total = cp.cuda.Device().mem_info[1] / 1024**3
        print(f"  Memory - GPU: {gpu_gb:.2f}/{gpu_total:.2f}GB")

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
    L11[idx, idx] += cp.float32(eps_diag)
    
    # Solve L11 * Q = I
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
    """Apply physical constraints to K"""
    # Symmetric (REQUIRED - Newton's 3rd law)
    K = 0.5 * (K + K.T)
    # Bound magnitude (allow negative for repulsion)
    K = cp.clip(K, -K_bound, K_bound)
    return K

def cost_func(P_dif, N):
    """RMS difference between fitted and observed P"""
    return cp.sqrt(cp.sum(P_dif**2)) / N

def estimate_eta(P_obs, K_init, N, identity, P_temp):
    """IMPROVED: Better ETA estimation for large matrices"""
    print("\nEstimating initial learning rate...")
    K2P_inplace(K_init, P_temp, identity)
    grad_norm = cp.sqrt(cp.mean((P_temp - P_obs)**2))
    
    print(f"  Initial gradient norm: {float(grad_norm):.5e}")
    
    # CRITICAL FIX: Much better scaling for large matrices
    base_eta = 2e-3  # Higher base rate
    
    if N < 5000:
        # Small matrices: original scaling
        size_factor = (2869.0 / N)**2
    elif N < 15000:
        # Medium matrices: moderate scaling
        size_factor = (5000.0 / N)**0.7
    else:
        # Large matrices (>15K): gentle scaling
        size_factor = (7500.0 / N)**0.5
    
    # More aggressive gradient factor
    gradient_factor = min(1.0, 5e-3 / float(grad_norm))
    eta = base_eta * size_factor * gradient_factor
    
    # MUCH wider range for large matrices
    if N > 20000:
        eta = np.clip(eta, 5e-6, 5e-2)  # 10x wider range
    elif N > 10000:
        eta = np.clip(eta, 1e-6, 1e-2)
    else:
        eta = np.clip(eta, 1e-7, 1e-3)
    
    print(f"  Size factor: {size_factor:.4f}")
    print(f"  Gradient factor: {gradient_factor:.4f}")
    print(f"  Initial ETA: {eta:.2e}")
    return eta

def save_checkpoint(K, cost, iteration, checkpoint_dir):
    """Save checkpoint for recovery"""
    cp_file = f"{checkpoint_dir}/checkpoint_iter{iteration}.npz"
    K_cpu = cp.asnumpy(K)
    np.savez_compressed(cp_file, K=K_cpu, cost=cost, iteration=iteration)
    return cp_file

def phic2_optimized(K, N, P_obs, checkpoint_dir, ETA_init=1.0e-6, ALPHA=1.0e-10, ITERATION_MAX=1000000):
    """
    OPTIMIZED PHi-C2 with momentum and better convergence
    """
    # Better K_bound initialization
    if N < 3000:
        K_bound = 1000.0
    elif N < 10000:
        K_bound = 500.0
    elif N < 20000:
        K_bound = 300.0  # Increased from 200
    else:
        # For very large matrices, start higher
        K_bound = max(100.0, 300.0 * (20000.0 / N)**0.5)
    
    print("\n" + "="*80)
    print("Starting PHi-C2 Optimization (OPTIMIZED)")
    print("="*80)
    print(f"Matrix size: {N}x{N}")
    print(f"Initial ETA: {ETA_init:.2e}")
    print(f"K bound: +/-{K_bound:.1f}")
    print(f"Checkpoint directory: {checkpoint_dir}")
    print()
    
    ETA = ETA_init
    ETA_min = ETA_init * 1e-5  # Allow going lower
    
    # Pre-allocate arrays
    identity = cp.eye(N-1, dtype=cp.float32)
    P_fit = cp.zeros((N, N), dtype=cp.float32)
    P_dif = cp.zeros((N, N), dtype=cp.float32)
    
    # Momentum for gradient descent
    momentum = cp.zeros_like(K)
    beta = 0.9  # Momentum coefficient
    
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
    
    cost_history = []
    oscillation_count = 0
    eta_reduction_count = 0
    k_bound_expansions = 0
    
    last_print = time.time()
    last_checkpoint = 0
    
    while iteration < ITERATION_MAX:
        cost_bk = cost
        
        # Momentum-based gradient descent
        momentum = beta * momentum + (1 - beta) * P_dif
        K -= ETA * momentum
        
        # Check K bounds
        K_min = float(cp.min(K))
        K_max = float(cp.max(K))
        K_absmax = max(abs(K_min), abs(K_max))
        
        # Dynamic bound expansion
        if K_absmax > K_bound * 1.2:
            old_bound = K_bound
            K_bound = K_absmax * 1.5
            k_bound_expansions += 1
            print(f"\n  -> K_bound expanded at iteration {iteration}")
            print(f"     Old: +/-{old_bound:.1f} -> New: +/-{K_bound:.1f}")
        
        # Apply constraints
        K = constrain_K(K, K_bound)
        
        # Check K health
        if cp.any(cp.isnan(K)) or cp.any(cp.isinf(K)):
            print(f"\n  [WARNING] Invalid K at iter {iteration}, reverting")
            K = best_K.copy()
            momentum.fill(0)
            ETA *= 0.5
            eta_reduction_count += 1
            continue
        
        # Forward pass
        try:
            K2P_inplace(K, P_fit, identity)
            P_dif[...] = P_fit - P_obs
            cost = cost_func(P_dif, N)
        except Exception as e:
            print(f"\n  [WARNING] K2P error at iter {iteration}: {e}")
            K = best_K.copy()
            momentum.fill(0)
            ETA *= 0.5
            eta_reduction_count += 1
            continue
        
        if cp.isnan(cost) or cp.isinf(cost):
            print(f"\n  [WARNING] Invalid cost at iter {iteration}")
            K = best_K.copy()
            momentum.fill(0)
            cost = best_cost
            ETA *= 0.5
            eta_reduction_count += 1
            continue
        
        # Track improvement
        cost_delta = cost_bk - cost
        cost_history.append(float(cost))
        
        # Keep only recent history
        if len(cost_history) > 1000:
            cost_history.pop(0)
        
        # Update best
        if cost < best_cost:
            best_cost = cost
            best_K = K.copy()
            best_iter = iteration
            oscillation_count = 0
        else:
            if cost_delta < 0:  # Cost increased
                oscillation_count += 1
        
        # Handle oscillations
        if oscillation_count >= 5:  # Faster response
            print(f"\n  [INFO] Oscillations detected at iter {iteration}")
            print(f"    ETA: {ETA:.2e} -> {ETA*0.5:.2e}")
            ETA *= 0.5
            eta_reduction_count += 1
            K = best_K.copy()
            momentum.fill(0)  # Reset momentum
            K2P_inplace(K, P_fit, identity)
            P_dif[...] = P_fit - P_obs
            cost = best_cost
            oscillation_count = 0
            
            if ETA < ETA_min:
                print(f"\n  [STOP] ETA < minimum ({ETA_min:.2e})")
                break
        
        # Adaptive ETA increase (when doing well)
        if cost_delta > 0 and oscillation_count == 0 and iteration % 50 == 0:
            if cost_delta > 1e-7 * cost_bk:  # Good progress
                ETA = min(ETA * 1.05, ETA_init * 2)  # Allow going higher than initial
        
        # Progress reporting
        if iteration == 1 or iteration % 100 == 0 or time.time() - last_print > 10:
            elapsed = time.time() - c_traj[0][1]
            rate = iteration / elapsed if elapsed > 0 else 0
            since_best = iteration - best_iter
            
            print(f"Iter {iteration:7d} | Cost: {cost:.6e} | dCost: {cost_delta:+.2e} | "
                  f"Best: {best_cost:.6e} | ETA: {ETA:.2e}")
            print(f"  Rate: {rate:.2f} it/s | Time: {elapsed/60:.1f}m | "
                  f"Since best: {since_best} | ETA red: {eta_reduction_count} | "
                  f"K_bound exp: {k_bound_expansions}")
            print_memory()
            
            c_traj.append([float(cost), time.time(), ETA])
            last_print = time.time()
        
        # Checkpointing every 500 iterations
        if iteration - last_checkpoint >= 500:
            cp_file = save_checkpoint(K, float(cost), iteration, checkpoint_dir)
            last_checkpoint = iteration
            if iteration % 2000 == 0:
                print(f"  -> Checkpoint: {os.path.basename(cp_file)}")
        
        # Convergence checks
        # 1. Cost delta check
        if iteration > 200:
            if abs(cost_delta) < ALPHA * ETA and cost_delta >= 0:
                print(f"\n[CONVERGED] Cost change < threshold")
                print(f"  dCost: {abs(cost_delta):.2e} < {ALPHA*ETA:.2e}")
                break
        
        # 2. Stagnation check (adaptive window)
        window = max(500, min(2000, N // 20))
        if iteration > window and iteration - best_iter > window:
            recent_std = np.std(cost_history[-window:]) if len(cost_history) >= window else np.inf
            if recent_std < 1e-9:
                print(f"\n[CONVERGED] Cost variance very low over {window} iterations")
                print(f"  Std: {recent_std:.2e}")
                break
        
        # 3. Max stagnation
        max_stagnation = max(5000, N // 5)
        if iteration - best_iter > max_stagnation:
            print(f"\n[STOP] No improvement for {max_stagnation} iterations")
            break
        
        # 4. Max ETA reductions
        if eta_reduction_count > 40:
            print(f"\n[STOP] Too many ETA reductions ({eta_reduction_count})")
            break
        
        iteration += 1
        
        # Periodic cleanup
        if iteration % 500 == 0:
            cp.get_default_memory_pool().free_all_blocks()
    
    # Use best K
    if best_cost < cost:
        print(f"\n[INFO] Using best K from iteration {best_iter}")
        K = best_K.copy()
        K2P_inplace(K, P_fit, identity)
        cost = best_cost
    
    # Save final checkpoint
    cp_file = save_checkpoint(K, float(best_cost), iteration, checkpoint_dir)
    print(f"\nFinal checkpoint: {os.path.basename(cp_file)}")
    
    c_traj = np.array(c_traj)
    
    print(f"\n" + "="*80)
    print("Optimization Complete")
    print("="*80)
    print(f"Iterations:          {iteration:,}")
    print(f"Best iteration:      {best_iter:,}")
    print(f"Initial cost:        {c_traj[0,0]:.6e}")
    print(f"Final cost:          {best_cost:.6e}")
    print(f"Improvement:         {100*(c_traj[0,0]-best_cost)/c_traj[0,0]:.2f}%")
    print(f"ETA reductions:      {eta_reduction_count}")
    print(f"K_bound expansions:  {k_bound_expansions}")
    print(f"Final K_bound:       +/-{K_bound:.1f}")
    print("="*80)
    
    paras_fit = f"N={N} ETA_init={ETA_init:.2e} iterations={iteration}"
    
    del identity, P_dif, momentum
    
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
    print(f"  [OK] Saved: {fn}")

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
    print(f"\n[OK] Loaded {N}x{N} in {load_time:.1f}s")
    
    # Preprocess
    cp.nan_to_num(P_obs, copy=False)
    P_obs = P_obs + cp.eye(N, dtype=cp.float32)
    
    # Matrix stats
    P_obs_cpu = cp.asnumpy(P_obs)
    nonzero = int(np.count_nonzero(P_obs_cpu))
    sparsity = 100 * (1 - nonzero/(N*N))
    print(f"\nMatrix statistics:")
    print(f"  Non-zero: {nonzero:,} ({sparsity:.1f}% sparse)")
    print(f"  Range: [{float(cp.min(P_obs)):.2e}, {float(cp.max(P_obs)):.2e}]")
    print(f"  Mean: {float(cp.mean(P_obs)):.2e}")
    del P_obs_cpu
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
    input_basename = os.path.basename(fhic)
    input_name = os.path.splitext(input_basename)[0]
    dataDir = f"{input_name}_phic2_OPTIMIZED"
    os.makedirs(dataDir, exist_ok=True)
    checkpoint_dir = f"{dataDir}/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    print(f"\nOutput directory: {dataDir}/")
    
    # Run optimization
    print("\n" + "="*80)
    print("STARTING OPTIMIZATION")
    print("="*80)
    start_opt = time.time()
    
    K_fit, P_fit, c_traj, paras_fit = phic2_optimized(
        K_fit, N, P_obs, checkpoint_dir,
        ETA_init=ETA_init,
        ALPHA=1.0e-10,
        ITERATION_MAX=1000000
    )
    
    opt_time = time.time() - start_opt
    print(f"\nOptimization time: {opt_time/60:.1f} min ({opt_time/3600:.1f} hours)")
    
    # Transfer to CPU
    print("\nTransferring results to CPU...")
    c_traj_np = c_traj
    K_fit = cp.asnumpy(K_fit)
    P_fit = cp.asnumpy(P_fit)
    P_obs = cp.asnumpy(P_obs)
    
    print("  [OK] Transfer complete")
    
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
    
    # Calculate total time
    total_time = time.time() - start_total
    
    # Save summary
    summary_file = f"{dataDir}/SUMMARY_{input_name}.txt"
    with open(summary_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("PHi-C2 OPTIMIZATION SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(f"Input file:                {os.path.basename(fhic)}\n")
        f.write(f"Matrix size:               {N}x{N}\n")
        f.write(f"Non-zero elements:         {nonzero:,} ({sparsity:.1f}% sparse)\n\n")
        f.write(f"Optimization:\n")
        f.write(f"  Total time:              {total_time/60:.1f} min ({total_time/3600:.1f} hours)\n")
        f.write(f"  Iterations:              {len(c_traj_np):,}\n")
        f.write(f"  Initial cost:            {c_traj_np[0,0]:.6e}\n")
        f.write(f"  Final cost:              {c_traj_np[-1,0]:.6e}\n")
        f.write(f"  Improvement:             {100*(c_traj_np[0,0]-c_traj_np[-1,0])/c_traj_np[0,0]:.2f}%\n\n")
        f.write(f"Results:\n")
        f.write(f"  Pearson (all):           {p1:.6f}\n")
        f.write(f"  Pearson (obs>0):         {p2:.6f}\n")
        f.write(f"  K range:                 [{np.min(K_fit):.5e}, {np.max(K_fit):.5e}]\n")
        f.write(f"  P range:                 [{np.nanmin(P_fit):.5e}, {np.nanmax(P_fit):.5e}]\n")
        f.write("="*80 + "\n")
        
        # Quality assessment
        if p1 >= 0.90:
            f.write("\nQUALITY: EXCELLENT (Pearson > 0.90)\n")
        elif p1 >= 0.85:
            f.write("\nQUALITY: GOOD (Pearson > 0.85)\n")
        elif p1 >= 0.75:
            f.write("\nQUALITY: ACCEPTABLE (Pearson > 0.75)\n")
        else:
            f.write("\nQUALITY: POOR (Pearson < 0.75)\n")
            f.write("Consider using coarser resolution (500KB or 1MB)\n")
    
    print(f"  [OK] Saved summary: {summary_file}")
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"Matrix:                    {N}x{N}")
    print(f"Total time:                {total_time/60:.1f} min ({total_time/3600:.1f} hours)")
    print(f"Iterations:                {len(c_traj_np):,}")
    print(f"Pearson (full, all):       {p1:.6f}")
    print(f"Pearson (full, obs>0):     {p2:.6f}")
    print(f"Output directory:          {dataDir}/")
    print("="*80)
    
    # Quality interpretation
    if p1 >= 0.90:
        print("\n[EXCELLENT] Pearson > 0.90 - Great fit!")
    elif p1 >= 0.85:
        print("\n[GOOD] Pearson > 0.85 - Acceptable results")
    elif p1 >= 0.75:
        print("\n[ACCEPTABLE] Pearson > 0.75 - Consider coarser resolution")
    else:
        print("\n[POOR] Pearson < 0.75 - Try 500KB or 1MB resolution")
    
    print(f"\nAll files saved to: {dataDir}/")
    print("Done!")
