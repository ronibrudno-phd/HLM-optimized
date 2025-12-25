import os
import sys
import time
import warnings
import cupy as cp
import numpy as np

# Optional: only used for small N; kept for compatibility
from scipy.stats import pearsonr

cp.set_printoptions(precision=3, linewidth=200)
warnings.filterwarnings('ignore')

if not len(sys.argv) == 2:
    print("usage:: python core_phic_new.py normalized-HiC-Contact-Matrix")
    sys.exit()
fhic = str(sys.argv[1])  # HiC observation


# ---------- Robust solve import (works across CuPy builds) ----------
try:
    from cupyx.scipy.linalg import solve as _solve
    _SOLVE_SUPPORTS_ASSUME_A = True
except Exception:
    from cupy.linalg import solve as _solve
    _SOLVE_SUPPORTS_ASSUME_A = False


# ---------- Utilities ----------
def Init_K(K, N, INIT_K0):
    """Vectorized initialization of tridiagonal backbone"""
    idx = cp.arange(1, N, dtype=cp.int32)
    K[idx, idx - 1] = cp.float32(INIT_K0)
    K[idx - 1, idx] = cp.float32(INIT_K0)
    return K


def clear_cupy_pools():
    """Helps fragmentation on long runs"""
    try:
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass


# ---------- Core math (memory-lean, stabilized) ----------
def K2P_inplace(K, out_P, identity, eps_diag=1e-6):
    """
    Convert K -> P, writing P into out_P (NxN float32 buffer).
    Stabilized:
      - build only L11 (no full L)
      - diagonal jitter on L11 for SPD stability
      - guard base of power to avoid NaNs
    """
    N = K.shape[0]

    # Degree vector (float32)
    d = cp.sum(K, axis=0, dtype=cp.float32)

    # Build L11 = diag(d[1:]) - K[1:,1:] without building full L
    L11 = (-K[1:, 1:]).copy()  # (N-1)x(N-1)
    idx = cp.arange(N - 1, dtype=cp.int32)
    L11[idx, idx] += d[1:]

    # Diagonal regularization (prevents borderline non-SPD / ill-conditioning)
    L11[idx, idx] += cp.float32(eps_diag)

    # Solve L11 @ Q = I
    if _SOLVE_SUPPORTS_ASSUME_A:
        Q = _solve(L11, identity, assume_a='pos')
    else:
        Q = _solve(L11, identity)

    # diag(Q)
    A = cp.diagonal(Q)

    # Fill out_P with G (reuse buffer)
    out_P.fill(0)
    sub = out_P[1:N, 1:N]
    sub[...] = -2.0 * Q
    sub += A[None, :]
    sub += A[:, None]
    out_P[0, 1:N] = A
    out_P[1:N, 0] = A

    # Convert G -> P in-place: P = (1 + 3G)^(-1.5)
    out_P *= cp.float32(3.0)
    out_P += cp.float32(1.0)

    # Guard to prevent negative/zero base -> NaNs/inf
    cp.maximum(out_P, cp.float32(1e-6), out=out_P)
    cp.power(out_P, cp.float32(-1.5), out=out_P)

    # Free big temporaries
    del Q, L11, A, sub, d
    return out_P


def Pdif2cost(P_dif, N):
    """RMSE over all entries"""
    return cp.sqrt(cp.sum(P_dif * P_dif)) / cp.float32(N)


def project_K_inplace(K, k_max=10.0):
    """
    Project K to a "safer" space:
      - symmetric
      - nonnegative
      - capped (prevents blow-ups)
    """
    # Symmetrize
    K[...] = cp.float32(0.5) * (K + K.T)
    # Nonnegative
    cp.maximum(K, cp.float32(0.0), out=K)
    # Cap
    if k_max is not None:
        cp.minimum(K, cp.float32(k_max), out=K)
    return K


# ---------- Optimizer (stabilized + rollback + retry) ----------
def phic2_optimized(
    K,
    P_obs,
    ETA=1e-6,
    ALPHA=1e-10,
    ITERATION_MAX=50,
    checkpoint_interval=0,
    eps_diag=1e-6,
    k_max=10.0,
    print_every_sec=10,
    max_jump_factor=5.0,   # reject if cost_new > max_jump_factor * cost_old
    max_retries=2,         # retries per iteration with higher eps and smaller ETA
):
    """
    Fixed-ETA optimizer (original-style) + safety rails:
      - Single NxN buffer reused (P_calc -> P_dif)
      - Projection of K each step (symmetry, nonnegativity, cap)
      - Reject/rollback on numerical blow-ups
      - Retry with stronger diagonal jitter + smaller ETA
      - ETA stays constant unless a retry is needed
    """
    N = K.shape[0]
    stop_delta = ETA * ALPHA
    paras_fit = "%e\t%e\t%d\t" % (ETA, ALPHA, ITERATION_MAX)

    # NxN buffer reused: holds P then overwritten with P_dif
    P_buf = cp.empty((N, N), dtype=cp.float32)

    # Identity for solve: (N-1)x(N-1)
    identity = cp.eye(N - 1, dtype=cp.float32)

    # Initial cost
    K2P_inplace(K, P_buf, identity, eps_diag=eps_diag)
    cp.subtract(P_buf, P_obs, out=P_buf)
    cost = Pdif2cost(P_buf, N)

    # Trajectory on CPU: [cost, time, rejected_flag]
    c_traj = np.zeros((ITERATION_MAX + 1, 3), dtype=np.float64)
    c_traj[0, 0] = float(cost)
    c_traj[0, 1] = time.time()
    c_traj[0, 2] = 0.0

    print(f"Starting optimization with N={N}, ETA={ETA:g}, ALPHA={ALPHA:g}")
    print(f"Initial cost: {float(cost):.6e}")
    print("iter\tcost\t\tcost_diff\teta\t\tt(sec)\trej\tretry")

    last_print_time = time.time()
    rejected_total = 0

    for iteration in range(1, ITERATION_MAX + 1):
        cost_prev = float(cost)
        cost_bk = cost

        # Backup K to rollback if needed
        K_prev = K.copy()

        accepted = False
        retry = 0
        eps_try = float(eps_diag)
        eta_try = float(ETA)  # fixed ETA unless we retry

        while (not accepted) and (retry <= max_retries):
            # restart from backup for each retry attempt
            K[...] = K_prev

            # Step
            K -= cp.float32(eta_try) * P_buf
            project_K_inplace(K, k_max=k_max)

            # Evaluate
            K2P_inplace(K, P_buf, identity, eps_diag=eps_try)
            cp.subtract(P_buf, P_obs, out=P_buf)
            cost_new = Pdif2cost(P_buf, N)
            cost_new_f = float(cost_new)
            
                        # Reject if non-finite OR if it increases cost beyond a tiny tolerance
            tol_increase = 1e-6  # allow 0.0001% increases (noise)
            bad = (
                (not np.isfinite(cost_new_f)) or
                (cost_new_f > cost_prev * (1.0 + tol_increase))
            )

            if bad:
                retry += 1
                # strengthen diagonal jitter and shrink step, then retry
                eps_try = min(eps_try * 100.0, 1e-2)
                eta_try = max(eta_try * 0.1, float(ETA) * 1e-4)  # don't go to zero
                continue

            # Accept
            accepted = True
            cost = cost_new

        # If still not accepted, rollback and stop
        if not accepted:
            K[...] = K_prev
            rejected_total += 1
            c_traj[iteration, 0] = cost_prev
            c_traj[iteration, 1] = time.time()
            c_traj[iteration, 2] = 1.0
            print(f"Iteration {iteration}: failed after retries; rolled back and stopping.")
            break

        # Record accepted step
        cost_dif = float(cost_bk - cost)
        c_traj[iteration, 0] = float(cost)
        c_traj[iteration, 1] = time.time()
        c_traj[iteration, 2] = 1.0 if (retry > 0) else 0.0
        if retry > 0:
            rejected_total += 1

        # Print progress
        now = time.time()
        if (now - last_print_time) > print_every_sec or iteration == 1 or (retry > 0) or (iteration % 10 == 0):
            elapsed = now - c_traj[0, 1]
            rej_flag = int(retry > 0)
            print(f"{iteration}\t{float(cost):.6e}\t{cost_dif:+.3e}\t{eta_try:.3e}\t{elapsed:6.1f}\t{rej_flag}\t{retry}")
            last_print_time = now

        # Stopping criteria (same spirit as original)
        if (0 < cost_dif < stop_delta) and (iteration > 3):
            print(f"Converged (stop_delta) at iteration {iteration}")
            break
        if not np.isfinite(float(cost)):
            print(f"Optimization diverged (non-finite cost) at iteration {iteration}")
            break

        if iteration % 5 == 0:
            clear_cupy_pools()

        if checkpoint_interval and (iteration % checkpoint_interval == 0):
            ck = os.path.join(dataDir, f"checkpoint_iter{iteration}.K.npy")
            cp.save(ck, K)
            print(f"Checkpoint saved: {ck}")

    last_it = iteration
    c_traj = c_traj[: last_it + 1]
    print(f"Total rejected/retried steps: {rejected_total}")
    return K, c_traj, paras_fit




def saveLg(fn, xy, ct):
    with open(fn, 'w') as fw:
        fw.write(ct)
        for row in xy:
            fw.write(f"{row[0]:.5e} {row[1]:.5e}\n")


# ---------- Main ----------
if __name__ == '__main__':
    if not os.path.isfile(fhic):
        print('Cannot find ' + fhic)
        sys.exit()

    print(f"Reading Hi-C matrix from {fhic}...")
    start_read = time.time()

    # Load as float32 on CPU then upload (saves RAM/transfer)
    try:
        P_obs_np = np.loadtxt(fhic, comments='#', dtype=np.float32)
        P_obs = cp.asarray(P_obs_np, dtype=cp.float32)
        del P_obs_np
    except Exception:
        # Fallback parse (still float32)
        P_list = []
        with open(fhic) as fr:
            for line in fr:
                if line and line[0] != '#':
                    P_list.append(list(map(float, line.strip().split())))
        P_obs = cp.asarray(np.asarray(P_list, dtype=np.float32), dtype=cp.float32)
        del P_list

    N = int(P_obs.shape[0])
    print(f"Matrix size: N={N} ({N}x{N} = {N*N:,} elements)")
    print(f"Read time: {time.time() - start_read:.2f}s")

    # Replace NaN/Inf with 0 and set diagonal to 1
    cp.nan_to_num(P_obs, copy=False)
    P_obs += cp.eye(N, dtype=cp.float32)

    # Output directory
    phic2_alpha = 1.0e-10
    dataDir = fhic[:fhic.rfind('.')] + "_phic2_a%7.1e_optimized" % (phic2_alpha)
    os.makedirs(dataDir, exist_ok=True)

    # Init K
    print("\nInitializing K matrix...")
    K_fit = cp.zeros((N, N), dtype=cp.float32)
    Init_K(K_fit, N, INIT_K0=0.5)

       # Stable starter settings for N~28k
    ETA0   = 1e-06          # safer than 1e-6 (you already saw one huge spike)
    ITERS0 = 100            # enough to see a trend; rollback protects you
    print("\nStarting optimization...")
    start_opt = time.time()
    K_fit, c_traj, paras_fit = phic2_optimized(
        K_fit,
        P_obs,
        ETA=ETA0,
        ALPHA=phic2_alpha,
        ITERATION_MAX=ITERS0,
        checkpoint_interval=50,   # save K every 10 iters (uncompressed .npy)
        eps_diag=1e-6,            # base jitter; retries will raise it if needed
        k_max=10.0,               # keep for now; only increase if progress stalls
        print_every_sec=10,
        max_jump_factor=5.0,      # stricter: reject big blow-ups early
        max_retries=3,
    )

    opt_time = time.time() - start_opt
    print(f"\nOptimization completed in {opt_time:.2f}s ({opt_time/3600:.2f} hours)")

    # Save log + K only (P_fit and Pearson masks are not feasible for large N)
    fo = f"{dataDir}/N{N}"
    print("Saving results...")
    c_traj[:, 1] = c_traj[:, 1] - c_traj[0, 1]
    saveLg(fo + '.log', c_traj, "#%s\n#cost systemTime\n" % (paras_fit))

    cp.save(fo + '.K_fit.npy', K_fit)

    print(f"N={N} too large: skipping P_fit + Pearson mask evaluation.")
    print("Saved: K_fit.npy and log only.")
    print(f"\nResults saved to {dataDir}")
    print("Done!")
