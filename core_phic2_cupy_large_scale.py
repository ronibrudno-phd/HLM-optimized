import os
import sys
import time
import warnings
import cupy as cp
import numpy as np
from scipy.stats import pearsonr
cp.set_printoptions(precision=3, linewidth=200)
warnings.filterwarnings('ignore')

# PHi-C2 with adaptive learning rate reduction
# Automatically reduces ETA when oscillations are detected

if not len(sys.argv) >= 2:
    print("usage:: python core_phic2_cupy_adaptive.py normalized-HiC-Contact-Matrix")
    sys.exit()

fhic = str(sys.argv[1])

print("="*80)
print("PHi-C2 with Adaptive Learning Rate")
print("="*80)

def Init_K(K, N, INIT_K0):
    for i in range(1, N):
        j = i-1
        K[i,j] = K[j,i] = INIT_K0
    return K

def K2P(K, N, regularization=1e-10):
    d = cp.sum(K, axis=0)
    L = cp.diag(d) - K
    L_sub = L[1:N, 1:N] + regularization * cp.eye(N-1, dtype=K.dtype)
    
    try:
        Q = cp.linalg.inv(L_sub)
    except:
        Q = cp.linalg.pinv(L_sub)
    
    M = 0.5*(Q + cp.transpose(Q))
    A = cp.diag(M)
    
    G = cp.zeros((N, N), dtype=K.dtype)
    G[1:N, 1:N] = -2*M + A + cp.reshape(A, (-1,1))
    G[0, 1:N] = A
    G[1:N, 0] = A
    G = cp.clip(G, -0.33, 1e6)
    
    P = cp.power(1.0 + 3.0*G, -1.5)
    P = cp.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)
    return P

def Pdif2cost(P_dif, N):
    return cp.sqrt(cp.sum(P_dif**2)) / N

def enforce_K_constraints(K):
    K = cp.maximum(K, 0)
    K = 0.5 * (K + K.T)
    K = cp.minimum(K, 1e3)
    return K

def estimate_eta(P_obs, K_init, N):
    print("\nEstimating initial learning rate...")
    P_init = K2P(K_init, N)
    P_dif = P_init - P_obs
    grad_norm = cp.sqrt(cp.mean(P_dif**2))
    
    print(f"  Initial gradient norm: {grad_norm:.5e}")
    
    base_eta = 1e-4
    size_factor = (2869.0 / N)**2
    gradient_factor = min(1.0, 1e-4 / float(grad_norm))
    eta = base_eta * size_factor * gradient_factor
    eta = np.clip(eta, 1e-8, 1e-4)
    
    print(f"  Initial ETA: {eta:.2e}")
    return eta

def phic2_adaptive(K, N, P_obs, ETA_init=1.0e-6, ALPHA=1.0e-10, ITERATION_MAX=1000000):
    """
    PHi-C2 with adaptive learning rate reduction
    """
    print("\n" + "="*80)
    print("Starting Adaptive PHi-C2 Optimization")
    print("="*80)
    print(f"Initial learning rate: {ETA_init:.2e}")
    print(f"Will reduce ETA when oscillations detected")
    print(f"Min ETA: {ETA_init * 1e-4:.2e}")
    print()
    
    ETA = ETA_init
    ETA_min = ETA_init * 1e-4
    
    # Initial state
    P_dif = K2P(K, N) - P_obs
    cost = Pdif2cost(P_dif, N)
    
    c_traj = []
    c_traj.append([float(cost), time.time(), ETA])
    
    print(f"Initial cost: {cost:.6e}\n")
    
    iteration = 1
    best_cost = cost
    best_K = K.copy()
    
    # Oscillation detection
    cost_history = [float(cost)]
    oscillation_count = 0
    eta_reduction_count = 0
    
    last_print = time.time()
    last_improvement_iter = 0
    
    while iteration < ITERATION_MAX:
        cost_bk = cost
        K_bk = K.copy()
        
        # Gradient descent
        K = K - ETA*P_dif
        K = enforce_K_constraints(K)
        
        if cp.any(cp.isnan(K)):
            print(f"\n⚠ NaN in K at iter {iteration}, reverting")
            K = best_K
            break
        
        # Compute new state
        try:
            P_dif = K2P(K, N) - P_obs
            cost = Pdif2cost(P_dif, N)
        except:
            print(f"\n⚠ Error at iter {iteration}, reverting")
            K = best_K
            break
        
        if cp.isnan(cost):
            print(f"\n⚠ NaN cost at iter {iteration}, reverting")
            K = best_K
            cost = best_cost
            break
        
        cost_dif = cost_bk - cost
        cost_history.append(float(cost))
        if len(cost_history) > 50:
            cost_history.pop(0)
        
        c_traj.append([float(cost), time.time(), ETA])
        
        # Update best
        if cost < best_cost:
            improvement = best_cost - cost
            best_cost = cost
            best_K = K.copy()
            last_improvement_iter = iteration
            oscillation_count = 0  # Reset oscillation counter
        else:
            # No improvement - check if oscillating
            if cost_dif < 0:  # Cost increased
                oscillation_count += 1
        
        # Detect sustained oscillations
        if oscillation_count >= 10:
            # Reduce learning rate
            ETA_old = ETA
            ETA = ETA * 0.5
            eta_reduction_count += 1
            
            print(f"\n  ⚠ Oscillations detected at iter {iteration}")
            print(f"    Reducing ETA: {ETA_old:.2e} → {ETA:.2e} (reduction #{eta_reduction_count})")
            
            # Revert to best K
            K = best_K.copy()
            P_dif = K2P(K, N) - P_obs
            cost = best_cost
            oscillation_count = 0
            
            # Check if ETA too small
            if ETA < ETA_min:
                print(f"\n  ✓ ETA ({ETA:.2e}) < minimum ({ETA_min:.2e})")
                print(f"    Stopping optimization")
                break
        
        # Progress
        if iteration == 1 or time.time() - last_print > 10:
            elapsed = c_traj[-1][1] - c_traj[0][1]
            rate = iteration / elapsed if elapsed > 0 else 0
            iters_since_improve = iteration - last_improvement_iter
            
            print(f"Iter {iteration:7d} | Cost: {cost:.6e} | ΔCost: {cost_dif:+.6e} | "
                  f"Best: {best_cost:.6e} | ETA: {ETA:.2e}")
            print(f"  Rate: {rate:.2f} it/s | Time: {elapsed/60:.1f}m | "
                  f"Since improve: {iters_since_improve} | Reductions: {eta_reduction_count}")
            
            last_print = time.time()
        
        # Convergence check
        stop_delta = ETA * ALPHA
        if 0 < cost_dif < stop_delta and iteration > 100:
            print(f"\n✓ Converged! ΔCost ({cost_dif:.2e}) < threshold ({stop_delta:.2e})")
            break
        
        # Stagnation check
        if iteration - last_improvement_iter > 5000:
            print(f"\n⚠ No improvement for 5000 iterations")
            
            # Check if really stuck (cost variance very low)
            if len(cost_history) == 50:
                cost_std = np.std(cost_history)
                if cost_std < stop_delta * 100:
                    print(f"  Cost variance ({cost_std:.2e}) very low, likely converged")
                    break
            
            # Try one more ETA reduction
            if ETA > ETA_min * 2:
                print(f"  Trying smaller ETA...")
                ETA = ETA * 0.5
                eta_reduction_count += 1
                last_improvement_iter = iteration  # Reset counter
            else:
                print(f"  ETA already at minimum, stopping")
                break
        
        iteration += 1
        
        if iteration % 100 == 0:
            cp.get_default_memory_pool().free_all_blocks()
    
    # Use best K
    if best_cost < cost:
        print(f"\nUsing best K (cost={best_cost:.6e} from iter {last_improvement_iter})")
        K = best_K
    
    c_traj = np.array(c_traj)
    
    print(f"\nOptimization summary:")
    print(f"  Total iterations: {iteration:,}")
    print(f"  Final cost: {best_cost:.6e}")
    print(f"  Initial cost: {c_traj[0,0]:.6e}")
    print(f"  Improvement: {c_traj[0,0] - best_cost:.6e} ({100*(c_traj[0,0]-best_cost)/c_traj[0,0]:.2f}%)")
    print(f"  ETA reductions: {eta_reduction_count}")
    print(f"  Final ETA: {ETA:.2e}")
    
    paras_fit = f"{ETA_init}\t{ALPHA}\t{iteration}\t{eta_reduction_count}\t"
    return [K, c_traj, paras_fit]

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
    line_count = 0
    with open(fhic) as fr:
        for line in fr:
            if not line[0] == '#':
                P_obs.append(list(map(float, line.strip().split())))
                line_count += 1
                if line_count % 5000 == 0:
                    print(f"  {line_count} rows...")
    
    P_obs = cp.array(P_obs, dtype=cp.float32)
    N = len(P_obs)
    
    print(f"\nLoaded {N}×{N} in {time.time()-start_load:.1f}s")
    
    cp.nan_to_num(P_obs, copy=False)
    P_obs = P_obs + cp.eye(N, dtype=cp.float32)
    
    print(f"  Non-zero: {int(cp.count_nonzero(P_obs)):,}")
    print(f"  Mean: {float(cp.mean(P_obs)):.2e}")
    
    K_fit = cp.zeros((N, N), dtype=cp.float32)
    K_fit = Init_K(K_fit, N, INIT_K0=0.5)
    
    ETA_init = estimate_eta(P_obs, K_fit, N)
    
    start_opt = time.time()
    K_fit, c_traj, paras_fit = phic2_adaptive(
        K_fit, N, P_obs,
        ETA_init=ETA_init,
        ALPHA=1.0e-10,
        ITERATION_MAX=1000000
    )
    opt_time = time.time() - start_opt
    
    print(f"\nTotal time: {opt_time/60:.1f} min ({opt_time/3600:.1f} hours)")
    
    print("\nComputing final P...")
    P_fit = K2P(K_fit, N)
    
    print("Transferring to CPU...")
    c_traj_np = c_traj.copy()
    K_fit = cp.asnumpy(K_fit)
    P_fit = cp.asnumpy(P_fit)
    P_obs = cp.asnumpy(P_obs)
    
    print("\nSaving results...")
    dataDir = fhic[:fhic.rfind('.')] + "_phic2_adaptive"
    os.makedirs(dataDir, exist_ok=True)
    fo = f"{dataDir}/N{N}"
    
    c_traj_np[:,1] = c_traj_np[:,1] - c_traj_np[0,1]
    saveLg(fo+'.log', c_traj_np, f"#{paras_fit}\n#cost time eta\n")
    
    
    
    print("\nComputing Pearson correlation (sampling-based)...")
    
    def pearson_sample(P_fit_gpu: cp.ndarray, P_obs_gpu: cp.ndarray, n_samples: int = 1_000_000, seed: int = 0):
        """Sample-based Pearson (upper triangle), and Pearson on sampled pairs with obs>0."""
        rng = np.random.default_rng(seed)
        N = int(P_obs_gpu.shape[0])
        i = rng.integers(0, N - 1, size=n_samples, dtype=np.int32)
        j = rng.integers(i + 1, N, size=n_samples, dtype=np.int32)
        i_gpu = cp.asarray(i)
        j_gpu = cp.asarray(j)
        obs = cp.asnumpy(P_obs_gpu[i_gpu, j_gpu])
        fit = cp.asnumpy(P_fit_gpu[i_gpu, j_gpu])
        p1 = float(np.corrcoef(fit, obs)[0, 1])
        m = obs > 0
        nz = int(m.sum())
        p2 = float(np.corrcoef(fit[m], obs[m])[0, 1]) if nz >= 1000 else float("nan")
        del i_gpu, j_gpu
        return p1, p2, nz
    
    # Convert back to GPU for sampling
    P_fit_gpu = cp.asarray(P_fit)
    P_obs_gpu = cp.asarray(P_obs)
    
    p1, p2, nz = pearson_sample(P_fit_gpu, P_obs_gpu, n_samples=1_000_000, seed=0)
    
    print(f"  Sample Pearson p1 (upper triangle): {p1:.6f}")
    print(f"  Sample Pearson p2 (obs>0):          {p2:.6f}   (nz samples={nz:,})")
    
    del P_fit_gpu, P_obs_gpu
    saveMx(fo+'.K_fit', K_fit,
           f"#K_fit N={N} range=[{np.min(K_fit):.5e}, {np.max(K_fit):.5e}]\n")
    saveMx(fo+'.P_fit', P_fit,
           f"#P_fit N={N} range=[{np.nanmin(P_fit):.5e}, {np.nanmax(P_fit):.5e}] "
           f"pearson_sampled={p1:.6f} pearson_obs>0={p2:.6f} nz_samples={nz}\n")
    
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"Matrix:                    {N}×{N}")
    print(f"Time:                      {opt_time/60:.1f} min")
    print(f"Iterations:                {len(c_traj_np):,}")
    print(f"Initial cost:              {c_traj_np[0,0]:.6e}")
    print(f"Final cost:                {c_traj_np[-1,0]:.6e}")
    print(f"Improvement:               {100*(c_traj_np[0,0]-c_traj_np[-1,0])/c_traj_np[0,0]:.2f}%")
    print(f"Pearson (sampled, all):    {p1:.6f}")
    print(f"Pearson (sampled, obs>0):  {p2:.6f}")
    print(f"Non-zero samples:          {nz:,} / 1,000,000")
    print(f"Output:                    {dataDir}/")
    print("="*80)
