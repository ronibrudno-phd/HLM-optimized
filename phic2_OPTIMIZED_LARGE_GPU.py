# -*- coding: utf-8 -*-
import os
import sys
import time
import warnings
import cupy as cp
import numpy as np
from scipy.stats import pearsonr
import gc

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

cp.set_printoptions(precision=3, linewidth=200)
warnings.filterwarnings('ignore')

# FINAL TUNED VERSION
# Based on empirical observation: For 27K matrices, need ETA ~1e-4 to 5e-4
# Previous versions were too conservative

if not len(sys.argv) >= 2:
    print("usage:: python phic2_final.py normalized-HiC-Contact-Matrix")
    sys.exit()

fhic = str(sys.argv[1])

print("="*80)
print("PHi-C2 FINAL TUNED VERSION")
print("="*80)
print("Key changes:")
print("  [!] MUCH higher initial learning rate for large matrices")
print("  [!] Empirically tuned for 27K×27K matrices")
print("  [!] More patience before reducing ETA")
print("="*80)

_SOLVE_SUPPORTS_ASSUME_A = True
try:
    test = cp.eye(2, dtype=cp.float32)
    cp.linalg.solve(test, test, assume_a='pos')
    del test
except TypeError:
    _SOLVE_SUPPORTS_ASSUME_A = False

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
    K = cp.zeros((N, N), dtype=dtype)
    for i in range(1, N):
        K[i, i-1] = K[i-1, i] = INIT_K0
    return K

def K2P_inplace(K, out_P, identity, eps_diag=1e-5, rc2=1.0):
    N = K.shape[0]
    d = cp.sum(K, axis=0, dtype=cp.float32)
    L11 = (-K[1:, 1:]).copy()
    idx = cp.arange(N - 1, dtype=cp.int32)
    L11[idx, idx] += d[1:]
    L11[idx, idx] += cp.float32(eps_diag)
    
    if _SOLVE_SUPPORTS_ASSUME_A:
        Q = cp.linalg.solve(L11, identity, assume_a='pos')
    else:
        Q = cp.linalg.solve(L11, identity)
    
    A = cp.diagonal(Q)
    out_P.fill(0)
    sub = out_P[1:N, 1:N]
    sub[...] = -2.0 * Q
    sub += A[None, :]
    sub += A[:, None]
    out_P[0, 1:N] = A
    out_P[1:N, 0] = A
    
    out_P *= cp.float32(3.0 / rc2)
    out_P += cp.float32(1.0)
    cp.maximum(out_P, cp.float32(1e-6), out=out_P)
    cp.power(out_P, cp.float32(-1.5), out=out_P)
    
    del Q, L11, A, sub, d
    return out_P

def constrain_K(K, K_bound):
    K = 0.5 * (K + K.T)
    K = cp.clip(K, -K_bound, K_bound)
    return K

def cost_func(P_dif, N):
    return cp.sqrt(cp.sum(P_dif**2)) / N

def estimate_eta(P_obs, K_init, N, identity, P_temp):
    """EMPIRICALLY TUNED for large matrices"""
    print("\nEstimating initial learning rate...")
    K2P_inplace(K_init, P_temp, identity)
    grad_norm = cp.sqrt(cp.mean((P_temp - P_obs)**2))
    
    print(f"  Initial gradient norm: {float(grad_norm):.5e}")
    
    # MUCH MORE AGGRESSIVE for large matrices
    # Based on empirical observation: need ~2e-4 for 27K matrix
    if N > 25000:
        # Very large matrices: use fixed aggressive rate
        eta = 2e-4
        print(f"  Using empirically tuned ETA for N={N}")
    elif N > 20000:
        eta = 3e-4
    elif N > 15000:
        eta = 4e-4
    elif N > 10000:
        eta = 5e-4
    else:
        # Smaller matrices: use formula
        base_eta = 5e-4
        size_factor = (5000.0 / N)**0.5
        eta = base_eta * size_factor
    
    # Adjust based on gradient magnitude (but not too much)
    grad_factor = min(2.0, max(0.5, 1e-3 / float(grad_norm)))
    eta = eta * grad_factor
    
    # Safety bounds
    eta = np.clip(eta, 1e-5, 1e-3)
    
    print(f"  Gradient adjustment factor: {grad_factor:.4f}")
    print(f"  Final ETA: {eta:.2e}")
    return eta

def save_checkpoint(K, cost, iteration, checkpoint_dir):
    cp_file = f"{checkpoint_dir}/checkpoint_iter{iteration}.npz"
    K_cpu = cp.asnumpy(K)
    np.savez_compressed(cp_file, K=K_cpu, cost=cost, iteration=iteration)
    return cp_file

def phic2_final_tuned(K, N, P_obs, checkpoint_dir, ETA_init=1.0e-6, ALPHA=1.0e-10, ITERATION_MAX=1000000):
    """Final tuned version with aggressive but stable learning"""
    
    # K_bound management
    if N < 3000:
        K_bound = 1000.0
    elif N < 10000:
        K_bound = 500.0
    elif N < 20000:
        K_bound = 200.0
    else:
        K_bound = 150.0  # Slightly higher for large matrices
    
    K_bound_max = K_bound * 10  # Allow significant expansion if needed
    
    print("\n" + "="*80)
    print("Starting PHi-C2 Optimization (FINAL TUNED)")
    print("="*80)
    print(f"Matrix size: {N}x{N}")
    print(f"Initial ETA: {ETA_init:.2e}")
    print(f"K bound: +/-{K_bound:.1f} (max: +/-{K_bound_max:.1f})")
    print()
    
    ETA = ETA_init
    ETA_min = ETA_init * 1e-3  # Can go quite low but not too low
    ETA_max = ETA_init * 5     # Allow significant increases
    
    identity = cp.eye(N-1, dtype=cp.float32)
    P_fit = cp.zeros((N, N), dtype=cp.float32)
    P_dif = cp.zeros((N, N), dtype=cp.float32)
    
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
    consecutive_bad = 0
    consecutive_good = 0
    eta_reduction_count = 0
    eta_increase_count = 0
    k_bound_expansions = 0
    
    last_print = time.time()
    last_checkpoint = 0
    
    while iteration < ITERATION_MAX:
        cost_bk = cost
        
        # Moderate gradient clipping
        grad_norm = float(cp.sqrt(cp.mean(P_dif**2)))
        clip_threshold = 5.0  # Very lenient
        if grad_norm > clip_threshold:
            P_dif = P_dif * (clip_threshold / grad_norm)
        
        # Gradient descent
        K -= ETA * P_dif
        
        # K bounds - liberal policy
        K_min = float(cp.min(K))
        K_max = float(cp.max(K))
        K_absmax = max(abs(K_min), abs(K_max))
        
        # Allow 20% overflow before expanding
        if K_absmax > K_bound * 1.2:
            if K_bound < K_bound_max:
                old_bound = K_bound
                K_bound = min(K_absmax * 1.4, K_bound_max)
                k_bound_expansions += 1
                if k_bound_expansions <= 5:
                    print(f"\n  -> K_bound: {old_bound:.1f} -> {K_bound:.1f}")
        
        K = constrain_K(K, K_bound)
        
        # Sanity checks
        if cp.any(cp.isnan(K)) or cp.any(cp.isinf(K)):
            print(f"\n  [WARNING] Invalid K at iter {iteration}")
            K = best_K.copy()
            ETA *= 0.8  # Gentler reduction
            eta_reduction_count += 1
            consecutive_bad = 0
            consecutive_good = 0
            continue
        
        # Forward pass
        try:
            K2P_inplace(K, P_fit, identity)
            P_dif[...] = P_fit - P_obs
            cost = cost_func(P_dif, N)
        except Exception as e:
            print(f"\n  [WARNING] K2P failed: {e}")
            K = best_K.copy()
            ETA *= 0.8
            eta_reduction_count += 1
            consecutive_bad = 0
            consecutive_good = 0
            continue
        
        if cp.isnan(cost) or cp.isinf(cost) or cost > 1.0:
            print(f"\n  [WARNING] Bad cost: {float(cost):.2e}")
            K = best_K.copy()
            K2P_inplace(K, P_fit, identity)
            P_dif[...] = P_fit - P_obs
            cost = best_cost
            ETA *= 0.8
            eta_reduction_count += 1
            consecutive_bad = 0
            consecutive_good = 0
            continue
        
        cost_delta = cost_bk - cost
        cost_history.append(float(cost))
        if len(cost_history) > 2000:
            cost_history.pop(0)
        
        # Update best
        if cost < best_cost:
            improvement = (best_cost - cost) / best_cost
            best_cost = cost
            best_K = K.copy()
            best_iter = iteration
            consecutive_bad = 0
            consecutive_good += 1
            
            # AGGRESSIVE ETA increases when making good progress
            if consecutive_good >= 5 and improvement > 1e-6:
                if ETA < ETA_max:
                    old_eta = ETA
                    ETA = min(ETA * 1.2, ETA_max)  # 1.2x increase
                    eta_increase_count += 1
                    if eta_increase_count <= 5:
                        print(f"\n  [+] ETA increased: {old_eta:.2e} -> {ETA:.2e}")
                    consecutive_good = 0
        else:
            if cost_delta < 0:
                consecutive_bad += 1
                consecutive_good = 0
        
        # MUCH MORE PATIENT before reducing ETA
        if consecutive_bad >= 15:  # Was 7 in BALANCED
            print(f"\n  [-] {consecutive_bad} bad updates at iter {iteration}")
            print(f"      ETA: {ETA:.2e} -> {ETA*0.8:.2e}")
            ETA *= 0.8  # Gentler than 0.7
            eta_reduction_count += 1
            K = best_K.copy()
            K2P_inplace(K, P_fit, identity)
            P_dif[...] = P_fit - P_obs
            cost = best_cost
            consecutive_bad = 0
            consecutive_good = 0
            
            if ETA < ETA_min:
                print(f"\n  [STOP] ETA < minimum ({ETA_min:.2e})")
                break
        
        # Progress reporting
        if iteration == 1 or iteration % 500 == 0 or time.time() - last_print > 20:
            elapsed = time.time() - c_traj[0][1]
            rate = iteration / elapsed if elapsed > 0 else 0
            since_best = iteration - best_iter
            improvement_pct = 100 * (c_traj[0][0] - best_cost) / c_traj[0][0]
            
            print(f"Iter {iteration:7d} | Cost: {cost:.6e} | dCost: {cost_delta:+.2e} | "
                  f"Best: {best_cost:.6e} | ETA: {ETA:.2e}")
            print(f"  Improvement: {improvement_pct:.2f}% | Since best: {since_best} | "
                  f"ETA red/inc: {eta_reduction_count}/{eta_increase_count} | K_bound: {K_bound:.1f}")
            print_memory()
            
            c_traj.append([float(cost), time.time(), ETA])
            last_print = time.time()
        
        # Checkpointing
        if iteration - last_checkpoint >= 2000:
            cp_file = save_checkpoint(K, float(cost), iteration, checkpoint_dir)
            last_checkpoint = iteration
            print(f"  -> Checkpoint: {os.path.basename(cp_file)}")
        
        # Convergence checks
        if iteration > 2000:
            recent_window = min(1000, len(cost_history))
            if recent_window >= 500:
                recent_improvement = (cost_history[-recent_window] - cost) / cost_history[-recent_window]
                if recent_improvement < 5e-5:  # 0.005% improvement
                    print(f"\n[CONVERGED] < 0.005% improvement over {recent_window} iterations")
                    break
        
        # Max stagnation
        max_stag = max(15000, N // 2)
        if iteration - best_iter > max_stag:
            print(f"\n[STOP] No improvement for {max_stag} iterations")
            break
        
        # Max iterations
        if iteration > 100000:
            print(f"\n[STOP] Reached 100K iterations")
            break
        
        iteration += 1
        
        if iteration % 1000 == 0:
            cp.get_default_memory_pool().free_all_blocks()
    
    # Use best K
    if best_cost < cost:
        print(f"\n[INFO] Using best K from iteration {best_iter}")
        K = best_K.copy()
        K2P_inplace(K, P_fit, identity)
    
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
    print(f"ETA increases:       {eta_increase_count}")
    print(f"K_bound expansions:  {k_bound_expansions}")
    print("="*80)
    
    paras_fit = f"N={N} ETA_init={ETA_init:.2e} iterations={iteration}"
    
    del identity, P_dif
    return K, P_fit, c_traj, paras_fit

def saveLg(fn, xy, ct):
    with open(fn, 'w') as fw:
        fw.write(ct)
        for row in xy:
            line = ' '.join(f"{v:.5e}" if not np.isnan(v) else "NaN" for v in row)
            fw.write(line + '\n')
    print(f"  Saved: {fn}")

def saveMx(fn, xy, ct, chunk_size=5000):
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

if __name__ == "__main__":
    start_total = time.time()
    
    if not os.path.isfile(fhic):
        print(f'ERROR: Cannot find {fhic}')
        sys.exit(1)
    
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
    
    cp.nan_to_num(P_obs, copy=False)
    P_obs = P_obs + cp.eye(N, dtype=cp.float32)
    
    P_obs_cpu = cp.asnumpy(P_obs)
    nonzero = int(np.count_nonzero(P_obs_cpu))
    sparsity = 100 * (1 - nonzero/(N*N))
    print(f"\nMatrix statistics:")
    print(f"  Non-zero: {nonzero:,} ({sparsity:.1f}% sparse)")
    print(f"  Range: [{float(cp.min(P_obs)):.2e}, {float(cp.max(P_obs)):.2e}]")
    print(f"  Mean: {float(cp.mean(P_obs)):.2e}")
    del P_obs_cpu
    print_memory()
    
    print("\nInitializing spring constants...")
    K_fit = Init_K(N, 0.5, cp.float32)
    
    P_temp = cp.zeros((N, N), dtype=cp.float32)
    identity_temp = cp.eye(N-1, dtype=cp.float32)
    ETA_init = estimate_eta(P_obs, K_fit, N, identity_temp, P_temp)
    del P_temp, identity_temp
    gc.collect()
    
    input_basename = os.path.basename(fhic)
    input_name = os.path.splitext(input_basename)[0]
    dataDir = f"{input_name}_phic2_FINAL_TUNED"
    os.makedirs(dataDir, exist_ok=True)
    checkpoint_dir = f"{dataDir}/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    print(f"\nOutput directory: {dataDir}/")
    
    print("\n" + "="*80)
    print("STARTING OPTIMIZATION")
    print("="*80)
    start_opt = time.time()
    
    K_fit, P_fit, c_traj, paras_fit = phic2_final_tuned(
        K_fit, N, P_obs, checkpoint_dir,
        ETA_init=ETA_init,
        ALPHA=1.0e-10,
        ITERATION_MAX=1000000
    )
    
    opt_time = time.time() - start_opt
    print(f"\nOptimization time: {opt_time/60:.1f} min ({opt_time/3600:.1f} hours)")
    
    print("\nTransferring results to CPU...")
    c_traj_np = c_traj
    K_fit = cp.asnumpy(K_fit)
    P_fit = cp.asnumpy(P_fit)
    P_obs = cp.asnumpy(P_obs)
    print("  [OK] Transfer complete")
    
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    fo = f"{dataDir}/N{N}"
    
    c_traj_np[:,1] = c_traj_np[:,1] - c_traj_np[0,1]
    saveLg(fo+'.log', c_traj_np, f"#{paras_fit}\n#cost time eta\n")
    
    saveMx(fo+'.K_fit', K_fit,
           f"#K_fit N={N} range=[{np.min(K_fit):.5e}, {np.max(K_fit):.5e}]\n")
    
    print("\nComputing Pearson correlation...")
    
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
    
    print("  [2/2] Full exact Pearson (all pairs)...")
    triMask = np.where(np.triu(np.ones((N,N)),1)>0)
    pijMask = np.where(np.triu(P_obs,1)>0)
    
    p1 = pearsonr(P_fit[triMask], P_obs[triMask])[0]
    p2 = pearsonr(P_fit[pijMask], P_obs[pijMask])[0]
    
    print(f"    Full Pearson (all):     {p1:.6f}")
    print(f"    Full Pearson (obs>0):   {p2:.6f}")
    
    saveMx(fo+'.P_fit', P_fit,
           f"#P_fit N={N} range=[{np.nanmin(P_fit):.5e}, {np.nanmax(P_fit):.5e}] "
           f"pearson={p1:.6f} {p2:.6f}\n")
    
    total_time = time.time() - start_total
    
    summary_file = f"{dataDir}/SUMMARY_{input_name}.txt"
    with open(summary_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("PHi-C2 OPTIMIZATION SUMMARY (FINAL TUNED)\n")
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
        f.write("="*80 + "\n")
        
        if p1 >= 0.90:
            f.write("\nQUALITY: EXCELLENT (>0.90)\n")
        elif p1 >= 0.85:
            f.write("\nQUALITY: GOOD (>0.85)\n")
        elif p1 >= 0.75:
            f.write("\nQUALITY: ACCEPTABLE (>0.75)\n")
        else:
            f.write("\nQUALITY: POOR (<0.75)\n")
            f.write("Consider coarser resolution (200KB, 500KB, or 1MB)\n")
    
    print(f"  [OK] Saved summary: {summary_file}")
    
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"Matrix:                    {N}x{N}")
    print(f"Total time:                {total_time/60:.1f} min ({total_time/3600:.1f} hours)")
    print(f"Iterations:                {len(c_traj_np):,}")
    print(f"Improvement:               {100*(c_traj_np[0,0]-c_traj_np[-1,0])/c_traj_np[0,0]:.2f}%")
    print(f"Pearson (full, all):       {p1:.6f}")
    print(f"Pearson (full, obs>0):     {p2:.6f}")
    print(f"Output directory:          {dataDir}/")
    print("="*80)
    
    if p1 >= 0.90:
        print("\n✓✓ EXCELLENT! Pearson > 0.90")
    elif p1 >= 0.85:
        print("\n✓ GOOD! Pearson > 0.85")
    elif p1 >= 0.75:
        print("\n△ ACCEPTABLE. Pearson > 0.75")
    else:
        print("\n✗ POOR. Pearson < 0.75")
        print("   RECOMMENDATION: Use coarser resolution (200KB-1MB)")
    
    print(f"\nAll files saved to: {dataDir}/")
    print("Done!")
