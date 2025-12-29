import os
import sys
import time
import warnings
import cupy as cp
import numpy as np
from scipy.stats import pearsonr
cp.set_printoptions(precision=3, linewidth=200)
warnings.filterwarnings('ignore')

# Memory-optimized PHi-C2 with in-place K2P computation
# Based on optimized implementation using solve() instead of inv()

if not len(sys.argv) >= 2:
    print("usage:: python core_phic2_cupy_optimized.py normalized-HiC-Contact-Matrix")
    sys.exit()

fhic = str(sys.argv[1])

print("="*80)
print("PHi-C2 Memory-Optimized with Adaptive Learning Rate")
print("="*80)

# Check if solve supports assume_a parameter
_SOLVE_SUPPORTS_ASSUME_A = True
try:
    test = cp.eye(2, dtype=cp.float32)
    cp.linalg.solve(test, test, assume_a='pos')
except TypeError:
    _SOLVE_SUPPORTS_ASSUME_A = False
    print("Note: CuPy version doesn't support assume_a, using standard solve")

def Init_K(N, INIT_K0, dtype):
    K = cp.zeros((N, N), dtype=dtype)
    for i in range(1, N):
        K[i, i-1] = K[i-1, i] = INIT_K0
    return K

def K2P_inplace(K, out_P, identity, eps_diag=1e-5, rc2=1):
    """
    Memory-optimized K2P using in-place operations
    - Reuses out_P array (no new allocation)
    - Uses solve() instead of inv() (faster, more stable)
    - Minimal temporary arrays
    """
    N = K.shape[0]
    
    # Compute degree vector
    d = cp.sum(K, axis=0, dtype=cp.float32)
    
    # Build Laplacian submatrix L11 = D[1:,1:] - K[1:,1:]
    L11 = (-K[1:, 1:]).copy()
    idx = cp.arange(N - 1, dtype=cp.int32)
    L11[idx, idx] += d[1:]
    L11[idx, idx] += cp.float32(eps_diag)  # Regularization
    
    # Solve L11 * Q = I (more stable than Q = inv(L11))
    if _SOLVE_SUPPORTS_ASSUME_A:
        Q = cp.linalg.solve(L11, identity, assume_a='pos')
    else:
        Q = cp.linalg.solve(L11, identity)
    
    # Get diagonal of Q
    A = cp.diagonal(Q)
    
    # Build G matrix in-place using out_P
    out_P.fill(0)
    sub = out_P[1:N, 1:N]  # View, not copy
    sub[...] = -2.0 * Q
    sub += A[None, :]
    sub += A[:, None]
    out_P[0, 1:N] = A
    out_P[1:N, 0] = A
    
    # Convert G to P: P = (1 + 3*G/rc2)^(-1.5)
    out_P *= cp.float32(3.0 / rc2)
    out_P += cp.float32(1.0)
    cp.maximum(out_P, cp.float32(1e-6), out=out_P)  # Prevent division by zero
    cp.power(out_P, cp.float32(-1.5), out=out_P)  # In-place
    
    # Clean up
    del Q, L11, A, sub, d
    
    return out_P

def cost_func(P_dif, N):
    return cp.sqrt(cp.sum(P_dif**2)) / N

def constrain_K(K):
    K = cp.maximum(K, 0)
    K = 0.5 * (K + K.T)
    K = cp.minimum(K)
    return K

def estimate_eta(P_obs, K_init, N, identity, P_temp):
    print("\nEstimating initial learning rate...")
    K2P_inplace(K_init, P_temp, identity)
    grad_norm = cp.sqrt(cp.mean((P_temp - P_obs)**2))
    
    print(f"  Initial gradient norm: {grad_norm:.5e}")
    
    base_eta = 1e-4
    size_factor = (2869.0 / N)**2
    gradient_factor = min(1.0, 1e-4 / float(grad_norm))
    eta = base_eta * size_factor * gradient_factor
    eta = np.clip(eta, 1e-8, 1e-4)
    
    print(f"  Initial ETA: {eta:.2e}")
    return eta

def phic2_optimized(K, N, P_obs, ETA_init=1.0e-6, ALPHA=1.0e-10, ITERATION_MAX=1000000):
    """
    Memory-optimized PHi-C2 with adaptive learning rate
    """
    print("\n" + "="*80)
    print("Starting Memory-Optimized PHi-C2")
    print("="*80)
    print(f"Initial ETA: {ETA_init:.2e}")
    print(f"Memory optimization: In-place K2P computation")
    print()
    
    ETA = ETA_init
    ETA_min = ETA_init * 1e-4
    
    # Pre-allocate arrays (reused throughout)
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
    
    iteration = 1
    best_cost = cost
    best_K = K.copy()
    
    cost_history = [float(cost)]
    oscillation_count = 0
    eta_reduction_count = 0
    
    last_print = time.time()
    last_improvement_iter = 0
    
    while iteration < ITERATION_MAX:
        cost_bk = cost
        
        # Gradient descent (in-place)
        K -= ETA * P_dif
        constrain_K(K)
        
        if cp.any(cp.isnan(K)):
            print(f"\n⚠ NaN in K, reverting")
            K = best_K.copy()
            break
        
        # Compute new state
        try:
            K2P_inplace(K, P_fit, identity)
            P_dif[...] = P_fit - P_obs
            cost = cost_func(P_dif, N)
        except:
            print(f"\n⚠ Error at iter {iteration}, reverting")
            K = best_K.copy()
            break
        
        if cp.isnan(cost):
            print(f"\n⚠ NaN cost, reverting")
            K = best_K.copy()
            cost = best_cost
            break
        
        cost_dif = cost_bk - cost
        cost_history.append(float(cost))
        if len(cost_history) > 50:
            cost_history.pop(0)
        
        c_traj.append([float(cost), time.time(), ETA])
        
        # Track best
        if cost < best_cost:
            best_cost = cost
            best_K = K.copy()
            last_improvement_iter = iteration
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
                print(f"\n  ✓ ETA < minimum, stopping")
                break
        
        # Progress
        if iteration == 1 or time.time() - last_print > 10:
            elapsed = c_traj[-1][1] - c_traj[0][1]
            rate = iteration / elapsed if elapsed > 0 else 0
            iters_since = iteration - last_improvement_iter
            
            mem_gb = cp.get_default_memory_pool().used_bytes() / 1024**3
            
            print(f"Iter {iteration:7d} | Cost: {cost:.6e} | ΔCost: {cost_dif:+.6e} | "
                  f"Best: {best_cost:.6e} | ETA: {ETA:.2e}")
            print(f"  Rate: {rate:.2f} it/s | Time: {elapsed/60:.1f}m | "
                  f"No improve: {iters_since} | GPU: {mem_gb:.2f}GB")
            
            last_print = time.time()
        
        # Convergence
        stop_delta = ETA * ALPHA
        if 0 < cost_dif < stop_delta and iteration > 100:
            print(f"\n✓ Converged! ΔCost < threshold")
            break
        
        # Stagnation
        if iteration - last_improvement_iter > 5000:
            print(f"\n⚠ No improvement for 5000 iterations")
            
            if len(cost_history) == 50:
                cost_std = np.std(cost_history)
                if cost_std < stop_delta * 100:
                    print(f"  Cost variance very low, converged")
                    break
            
            if ETA > ETA_min * 2:
                print(f"  Reducing ETA...")
                ETA = ETA * 0.5
                eta_reduction_count += 1
                last_improvement_iter = iteration
            else:
                print(f"  ETA at minimum, stopping")
                break
        
        iteration += 1
        
        if iteration % 100 == 0:
            cp.get_default_memory_pool().free_all_blocks()
    
    # Use best K
    if best_cost < cost:
        K = best_K.copy()
    
    # Final P
    K2P_inplace(K, P_fit, identity)
    
    c_traj = np.array(c_traj)
    
    print(f"\nOptimization summary:")
    print(f"  Iterations: {iteration:,}")
    print(f"  Final cost: {best_cost:.6e}")
    print(f"  Initial cost: {c_traj[0,0]:.6e}")
    print(f"  Improvement: {c_traj[0,0] - best_cost:.6e} ({100*(c_traj[0,0]-best_cost)/c_traj[0,0]:.2f}%)")
    print(f"  ETA reductions: {eta_reduction_count}")
    
    paras_fit = f"{ETA_init}\t{ALPHA}\t{iteration}\t{eta_reduction_count}\t"
    
    # Clean up pre-allocated arrays
    del identity, P_dif
    
    return [K, P_fit, c_traj, paras_fit]

def saveLg(fn, xy, ct):
    with open(fn, 'w') as fw:
        fw.write(ct)
        for row in xy:
            line = ' '.join(f"{v:.5e}" if not np.isnan(v) else "NaN" for v in row)
            fw.write(line + '\n')

def saveMx(fn, xy, ct, chunk_size=5000):
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
    fw.close()
    print("  Done!")

if __name__ == "__main__":
    if not os.path.isfile(fhic):
        print(f'Cannot find {fhic}')
        sys.exit()
    
    print(f"\nLoading: {fhic}")
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
    
    print(f"\nLoaded {N}×{N} in {time.time()-start_load:.1f}s")
    
    cp.nan_to_num(P_obs, copy=False)
    P_obs = P_obs + cp.eye(N, dtype=cp.float32)
    
    print(f"  Non-zero: {int(cp.count_nonzero(P_obs)):,}")
    print(f"  Mean: {float(cp.mean(P_obs)):.2e}")
    
    # Initialize
    K_fit = Init_K(N, 0.5, cp.float32)
    
    # Estimate ETA (need temporary arrays)
    P_temp = cp.zeros((N, N), dtype=cp.float32)
    identity_temp = cp.eye(N-1, dtype=cp.float32)
    ETA_init = estimate_eta(P_obs, K_fit, N, identity_temp, P_temp)
    del P_temp, identity_temp
    
    # Run optimization
    start_opt = time.time()
    K_fit, P_fit, c_traj, paras_fit = phic2_optimized(
        K_fit, N, P_obs,
        ETA_init=ETA_init,
        ALPHA=1.0e-10,
        ITERATION_MAX=1000000
    )
    opt_time = time.time() - start_opt
    
    print(f"\nTotal time: {opt_time/60:.1f} min ({opt_time/3600:.1f} hours)")
    
    # Transfer to CPU
    print("\nTransferring to CPU...")
    c_traj_np = c_traj.copy()
    K_fit = cp.asnumpy(K_fit)
    P_fit = cp.asnumpy(P_fit)
    P_obs = cp.asnumpy(P_obs)
    
    # Save
    print("\nSaving results...")
    dataDir = fhic[:fhic.rfind('.')] + "_phic2_optimized"
    os.makedirs(dataDir, exist_ok=True)
    fo = f"{dataDir}/N{N}"
    
    c_traj_np[:,1] = c_traj_np[:,1] - c_traj_np[0,1]
    saveLg(fo+'.log', c_traj_np, f"#{paras_fit}\n#cost time eta\n")
    
    saveMx(fo+'.K_fit', K_fit,
           f"#K_fit N={N} range=[{np.min(K_fit):.5e}, {np.max(K_fit):.5e}]\n")
    
    # Pearson correlation
    print("\nComputing Pearson correlation...")
    print("  [1/2] Fast sampling (1M samples)...")
    
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
    
    print("  [2/2] Full exact Pearson...")
    triMask = np.where(np.triu(np.ones((N,N)),1)>0)
    pijMask = np.where(np.triu(P_obs,1)>0)
    
    p1 = pearsonr(P_fit[triMask], P_obs[triMask])[0]
    p2 = pearsonr(P_fit[pijMask], P_obs[pijMask])[0]
    
    print(f"    Full Pearson (all):     {p1:.6f}")
    print(f"    Full Pearson (obs>0):   {p2:.6f}")
    
    ct = "#P_fit N %d min: %11.5e max: %11.5e pearson: %11.5e %11.5e\n"%\
         (N, np.nanmin(P_fit), np.nanmax(P_fit), p1, p2)
    saveMx(fo+'.P_fit', P_fit, ct)
    
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"Matrix:                    {N}×{N}")
    print(f"Time:                      {opt_time/60:.1f} min")
    print(f"Iterations:                {len(c_traj_np):,}")
    print(f"Initial cost:              {c_traj_np[0,0]:.6e}")
    print(f"Final cost:                {c_traj_np[-1,0]:.6e}")
    print(f"Improvement:               {100*(c_traj_np[0,0]-c_traj_np[-1,0])/c_traj_np[0,0]:.2f}%")
    print(f"Pearson (full, all):       {p1:.6f}")
    print(f"Pearson (full, obs>0):     {p2:.6f}")
    print(f"Output:                    {dataDir}/")
    print("="*80)
