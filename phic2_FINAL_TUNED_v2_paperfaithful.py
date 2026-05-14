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

# PHi-C2 paper-faithful version
# Stays true to the paper's K2P. Replaces v2's silent numerical defenses
# with explicit detection that backs up + reduces ETA on problems.
# See phic2_FINAL_TUNED_v2.py for the v2 version with defenses enabled.

import argparse

_argp = argparse.ArgumentParser(
    description="PHi-C2 optimizer, paper-faithful version. "
                "Uses K2P with NO regularization (matches paper's K2P exactly). "
                "Detects and reports numerical problems instead of hiding them.")
_argp.add_argument("fhic",
                   help="Path to normalized Hi-C contact matrix")
_argp.add_argument("--spectrum-log-every", type=int, default=5000,
                   help="Compute and print eigvalsh(L[1:,1:]) every N "
                        "iterations. 0 disables. Default 5000.")
_argp.add_argument("--output-suffix", type=str, default="",
                   help="Optional suffix on the output directory name.")
_argp.add_argument("--K-runaway-threshold", type=float, default=1e4,
                   help="If max|K| exceeds this, back up. Default 1e4.")
_args = _argp.parse_args()

fhic = _args.fhic
SPECTRUM_LOG_EVERY = int(_args.spectrum_log_every)
OUTPUT_SUFFIX = _args.output_suffix
K_RUNAWAY_THRESHOLD = float(_args.K_runaway_threshold)

print(f"[paperfaithful] eps_diag    = 0.0  (paper convention)")
print(f"[paperfaithful] out_P floor = OFF  (will detect non-positive instead)")
print(f"[paperfaithful] grad clip   = OFF  (paper convention)")
print(f"[paperfaithful] K clip      = OFF  (detect runaway at |K|>{K_RUNAWAY_THRESHOLD:.0e})")
print(f"[paperfaithful] spectrum_log_every = {SPECTRUM_LOG_EVERY}")
if OUTPUT_SUFFIX:
    print(f"[paperfaithful] output_suffix = '{OUTPUT_SUFFIX}'")

print("="*80)
print("PHi-C2 paper-faithful version")
print("="*80)
print("Matches paper's K2P exactly (no eps_diag, no out_P floor).")
print("Kept from v2:")
print("  - assume_a='sym' (LDL^T factorization)")
print("  - K symmetrization each step (float64 hygiene)")
print("  - ETA from N at initialization")
print("  - Best-K tracking + backup on detected problems")
print("Removed from v2:")
print("  - eps_diag regularization (paper does not regularize)")
print("  - cp.maximum(out_P, 1e-6) floor (paper does not floor)")
print("  - gradient clipping (paper does not clip)")
print("  - K clipping (replaced with K runaway DETECTION)")
print("="*80)

_SOLVE_SUPPORTS_ASSUME_A = True
try:
    test = cp.eye(2, dtype=cp.float64)
    cp.linalg.solve(test, test, assume_a='sym')
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

def K2P_inplace(K, out_P, identity, rc2=1.0):
    """
    K to P conversion matching the paper's K2P (core_phic2_cupy.py) exactly:

        d = sum(K, axis=0)
        L = diag(d) - K
        Q = inv(L[1:N, 1:N])          # solved via LDL^T (assume_a='sym')
        M = 0.5*(Q + Q.T)
        A = diag(M)
        G[1:N, 1:N] = -2M + A_row + A_col
        G[0, 1:N] = G[1:N, 0] = A
        P = (1 + 3*G/rc2)**(-1.5)

    No regularization (eps_diag) is added to L. No floor on the
    (1 + 3*G/rc2) matrix before the power operation. If the matrix
    becomes indefinite enough that (1 + 3*G/rc2) goes non-positive on
    any entry, that entry would produce NaN under (...)^(-1.5). Instead
    of silently flooring, we COUNT and RETURN those entries so the
    caller can decide to back up.

    The only difference from the paper's K2P:
      - Uses cp.linalg.solve(..., assume_a='sym') instead of cp.linalg.inv.
        Both produce identical results when L11 is PSD. LDL^T is more
        robust on near-singular matrices that the paper's inv would
        silently mishandle.

    Returns
    -------
    out_P : modified in place; the predicted contact map.
    n_nonpos : int. Number of entries in (1+3*G/rc2) that were <= 0
               BEFORE the power operation. If > 0, the model has
               produced non-physical predictions and the gradient
               will contain NaN. Caller should back up.
    """
    N = K.shape[0]
    d = cp.sum(K, axis=0, dtype=cp.float64)
    L11 = (-K[1:, 1:]).copy()
    idx = cp.arange(N - 1, dtype=cp.int64)
    L11[idx, idx] += d[1:]
    # No eps_diag. No regularization. Paper's K2P does not add it.

    if _SOLVE_SUPPORTS_ASSUME_A:
        Q = cp.linalg.solve(L11, identity, assume_a='sym')
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

    out_P *= cp.float64(3.0 / rc2)
    out_P += cp.float64(1.0)

    # Detect non-positive entries instead of flooring. Caller will react.
    n_nonpos = int(cp.sum(out_P <= 0))

    # Apply the power operation. Where out_P > 0 this is well-defined;
    # where out_P <= 0 it will produce NaN, which the caller's NaN check
    # will catch. We don't floor.
    cp.power(out_P, cp.float64(-1.5), out=out_P)

    del Q, L11, A, sub, d
    return out_P, n_nonpos


def log_L_spectrum(K, label=""):
    """
    Compute eigvalsh of L[1:N, 1:N] on CPU/float64 and print spectrum stats.
    This is the matrix that K2P_inplace inverts (with NO regularization in
    this paperfaithful version). Min eigenvalue tells us whether the
    Laplacian is positive semi-definite.
    """
    import numpy as _np
    from scipy.linalg import eigvalsh as _eigvalsh

    N = K.shape[0]
    K_cpu = cp.asnumpy(K).astype(_np.float64)
    d_cpu = _np.sum(K_cpu, axis=0)
    L_cpu = _np.diag(d_cpu) - K_cpu
    lam = _eigvalsh(L_cpu[1:N, 1:N])
    n_neg = int((lam < -1e-12).sum())
    n_zero = int((_np.abs(lam) <= 1e-12).sum())
    pad = f" [{label}]" if label else ""
    print(f"  [spectrum]{pad} L[1:,1:] eigenvalues: "
          f"min={lam.min():.3e}, max={lam.max():.3e}, "
          f"n_negative={n_neg}, n_near_zero={n_zero}")
    return lam.min(), lam.max(), n_neg

def cost_func(P_dif, N):
    return cp.sqrt(cp.sum(P_dif**2)) / N

def estimate_eta(N):
    """
    Set initial ETA from matrix size only.

    Fix 4: previous version scaled ETA by (initial_cost / 2.7e-4) where
    2.7e-4 was hardcoded to one specific dataset. For other species this
    produced inconsistent starting ETAs — too large causes immediate NaN
    cascade, too small causes no progress. Size-based ETA is species-agnostic
    and stable across the full range of N in the species panel (N~1000 to ~35000).
    Hard cap at 1e-3 prevents instability for very small matrices (e.g.
    small-genome invertebrates at 1Mb resolution where N can be ~100-200).
    """
    if N > 25000:
        eta = 5e-5
    elif N > 20000:
        eta = 8e-5
    elif N > 15000:
        eta = 1e-4
    elif N > 10000:
        eta = 2e-4
    else:
        eta = 2e-4 * (10000.0 / N) ** 0.5
    eta = min(eta, 1e-3)
    print(f"\nInitial ETA: {eta:.2e}  (matrix size N={N})")
    return eta

def save_checkpoint(K, cost, iteration, checkpoint_dir):
    cp_file = f"{checkpoint_dir}/checkpoint_iter{iteration}.npz"
    K_cpu = cp.asnumpy(K)
    np.savez_compressed(cp_file, K=K_cpu, cost=cost, iteration=iteration)
    return cp_file

def phic2_final_tuned(K, N, P_obs, checkpoint_dir, ETA_init=1.0e-6, ALPHA=1.0e-10, ITERATION_MAX=1000000):
    """Final tuned version with aggressive but stable learning"""
    
    # K bounds: not used in paperfaithful version. The K_RUNAWAY_THRESHOLD
    # CLI argument acts as a soft cap (via backup, not clip).
    
    print("\n" + "="*80)
    print("Starting PHi-C2 Optimization (FINAL TUNED)")
    print("="*80)
    print(f"Matrix size: {N}x{N}")
    print(f"Initial ETA: {ETA_init:.2e}")
    print(f"K runaway threshold: |K| < {K_RUNAWAY_THRESHOLD:.0e} (backup-not-clip)")
    print()
    
    ETA = ETA_init
    # Fix 5: ETA_min raised from ETA_init*5e-4 to ETA_init*1e-2.
    # 4 orders of magnitude of ETA decay caused the optimizer to stop via
    # ETA exhaustion rather than genuine convergence. 2 orders of magnitude
    # is the appropriate maximum decay — beyond that, use stagnation/convergence
    # checks to stop instead.
    ETA_min = ETA_init * 1e-2
    ETA_max = ETA_init * 5     # Allow significant increases
    
    identity = cp.eye(N-1, dtype=cp.float64)
    P_fit = cp.zeros((N, N), dtype=cp.float64)
    P_dif = cp.zeros((N, N), dtype=cp.float64)
    
    _, _n_nonpos = K2P_inplace(K, P_fit, identity)
    P_dif[...] = P_fit - P_obs
    cost = cost_func(P_dif, N)
    
    c_traj = []
    c_traj.append([float(cost), time.time(), ETA])
    
    print(f"\n[paperfaithful] Initial L spectrum:")
    log_L_spectrum(K, label="initial")
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
    
    last_print = time.time()
    last_checkpoint = 0
    
    while iteration < ITERATION_MAX:
        cost_bk = cost
        
        # Gradient descent (paper convention: no gradient clipping)
        K -= ETA * P_dif
        
        # Symmetrize K (paper does not, but float64 introduces asymmetry
        # over many gradient updates; this is the one piece of numerical
        # hygiene I keep).
        K = 0.5 * (K + K.T)

        # DETECTION: K running away? Paper does not clip K, but if any entry
        # exceeds K_RUNAWAY_THRESHOLD the optimizer is in trouble and the
        # next K2P will likely fail. Back up rather than clip.
        K_absmax = float(cp.max(cp.abs(K)))
        if K_absmax > K_RUNAWAY_THRESHOLD:
            print(f"\n  [DETECT] iter {iteration}: max|K|={K_absmax:.2e} "
                  f"exceeds {K_RUNAWAY_THRESHOLD:.0e}. Backing up.")
            K = best_K.copy()
            ETA *= 0.8
            eta_reduction_count += 1
            consecutive_bad = 0
            consecutive_good = 0
            if ETA < ETA_min:
                print(f"\n  [STOP] ETA exhausted after K-runaway detection")
                break
            continue
        
        # Sanity checks
        if cp.any(cp.isnan(K)) or cp.any(cp.isinf(K)):
            print(f"\n  [WARNING] Invalid K at iter {iteration}")
            K = best_K.copy()
            ETA *= 0.8  # Gentler reduction
            eta_reduction_count += 1
            consecutive_bad = 0
            consecutive_good = 0
            continue
        
        # Forward pass (paper's K2P, no regularization, no floor)
        try:
            _, _n_nonpos = K2P_inplace(K, P_fit, identity)
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

        # DETECTION: did K2P produce non-positive (1 + 3*G/rc2) entries?
        # That means the Laplacian is indefinite enough that the model has
        # collapsed -- gradient information from those entries is meaningless.
        # Back up and reduce ETA. (v2 silently floored these entries; we detect.)
        if _n_nonpos > 0:
            if iteration <= 10 or iteration % 200 == 0:
                print(f"\n  [DETECT] iter {iteration}: K2P produced "
                      f"{_n_nonpos} non-positive (1+3G/rc2) entries. "
                      f"Backing up and reducing ETA.")
            K = best_K.copy()
            _, _ = K2P_inplace(K, P_fit, identity)
            P_dif[...] = P_fit - P_obs
            cost = best_cost
            ETA *= 0.8
            eta_reduction_count += 1
            consecutive_bad = 0
            consecutive_good = 0
            if ETA < ETA_min:
                print(f"\n  [STOP] ETA exhausted after non-positive detection")
                break
            continue
        
        if cp.isnan(cost) or cp.isinf(cost) or cost > 1.0:
            print(f"\n  [WARNING] Bad cost: {float(cost):.2e}")
            K = best_K.copy()
            _, _n_nonpos = K2P_inplace(K, P_fit, identity)
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
        
        # REDUCE ETA only after more bad updates (ADAPTIVE TO MATRIX SIZE)
        # Small matrices: faster convergence, need less patience
        # Large matrices: slower convergence, need more patience
        if N < 5000:
            bad_threshold = 10  # Small matrices
        elif N < 15000:
            bad_threshold = 12
        else:
            bad_threshold = 15  # Large matrices
        
        if consecutive_bad >= bad_threshold:
            print(f"\n  [-] {consecutive_bad} bad updates at iter {iteration}")
            print(f"      ETA: {ETA:.2e} -> {ETA*0.8:.2e}")
            ETA *= 0.8  # Gentler than 0.7
            eta_reduction_count += 1
            K = best_K.copy()
            _, _n_nonpos = K2P_inplace(K, P_fit, identity)
            P_dif[...] = P_fit - P_obs
            cost = best_cost
            consecutive_bad = 0
            consecutive_good = 0
            
            if ETA < ETA_min:
                print(f"\n  [STOP] ETA < minimum ({ETA_min:.2e})")
                break
        
        # Periodic spectrum monitoring (paperfaithful diagnostic)
        if SPECTRUM_LOG_EVERY > 0 and iteration % SPECTRUM_LOG_EVERY == 0:
            log_L_spectrum(K, label=f"iter {iteration}")

        # Progress reporting
        if iteration == 1 or iteration % 500 == 0 or time.time() - last_print > 20:
            elapsed = time.time() - c_traj[0][1]
            rate = iteration / elapsed if elapsed > 0 else 0
            since_best = iteration - best_iter
            improvement_pct = 100 * (c_traj[0][0] - best_cost) / c_traj[0][0]
            
            print(f"Iter {iteration:7d} | Cost: {cost:.6e} | dCost: {cost_delta:+.2e} | "
                  f"Best: {best_cost:.6e} | ETA: {ETA:.2e}")
            print(f"  Improvement: {improvement_pct:.2f}% | Since best: {since_best} | "
                  f"ETA red/inc: {eta_reduction_count}/{eta_increase_count}")
            print_memory()
            
            c_traj.append([float(cost), time.time(), ETA])
            last_print = time.time()
        
        # Checkpointing
        if iteration - last_checkpoint >= 10000:
            cp_file = save_checkpoint(K, float(cost), iteration, checkpoint_dir)
            last_checkpoint = iteration
            print(f"  -> Checkpoint: {os.path.basename(cp_file)}")
        
        # Convergence checks (ADAPTIVE TO MATRIX SIZE)
        # Small matrices converge faster but need minimum iterations
        min_iter_for_convergence = max(500, N // 10)  # At least 500 iterations
        
        if iteration > min_iter_for_convergence:
            recent_window = min(1000, len(cost_history))
            if recent_window >= 200:  # Reduced from 500 for small matrices
                recent_improvement = (cost_history[-recent_window] - cost) / cost_history[-recent_window]
                convergence_threshold = 1e-4 if N < 10000 else 5e-5  # Stricter for small matrices
                if recent_improvement < convergence_threshold:
                    print(f"\n[CONVERGED] < {convergence_threshold:.1e} improvement over {recent_window} iterations")
                    break
        
        # Max stagnation (ADAPTIVE TO MATRIX SIZE)
        if N < 5000:
            max_stag = max(2000, N)  # Small matrices: 2K-5K iterations
        elif N < 15000:
            max_stag = max(5000, N // 3)
        else:
            max_stag = max(15000, N // 2)  # Large matrices
        
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
        _, _n_nonpos = K2P_inplace(K, P_fit, identity)
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
    
    P_obs = cp.array(P_obs, dtype=cp.float64)
    N = len(P_obs)
    
    load_time = time.time() - start_load
    print(f"\n[OK] Loaded {N}x{N} in {load_time:.1f}s")
    
    cp.nan_to_num(P_obs, copy=False)
    P_obs = P_obs + cp.eye(N, dtype=cp.float64)
    
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
    K_fit = Init_K(N, 0.5, cp.float64)
    
    P_temp = cp.zeros((N, N), dtype=cp.float64)
    identity_temp = cp.eye(N-1, dtype=cp.float64)
    ETA_init = estimate_eta(N)
    del P_temp, identity_temp
    gc.collect()
    
    input_basename = os.path.basename(fhic)
    input_name = os.path.splitext(input_basename)[0]
    dataDir = f"{input_name}_phic2_paperfaithful{OUTPUT_SUFFIX}"
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
        i = rng.integers(0, N - 1, size=n_samples, dtype=np.int64)
        j = rng.integers(i + 1, N, size=n_samples, dtype=np.int64)
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
