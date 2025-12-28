import os
import sys
import time
import warnings
import cupy as cp
import numpy as np

# Optional: only used for small N; kept for compatibility
from scipy.stats import pearsonr
def pearson_sample(P_fit_gpu, P_obs_gpu, n_samples=1_000_000, seed=0):
    """
    Estimates:
      p1 = Pearson(P_fit, P_obs) over sampled upper-triangle pairs
      p2 = Pearson(P_fit, P_obs) over sampled upper-triangle pairs where P_obs > 0
    """
    rng = np.random.default_rng(seed)
    N = int(P_obs_gpu.shape[0])

    # sample upper-triangle indices on CPU
    i = rng.integers(0, N - 1, size=n_samples, dtype=np.int32)
    j = rng.integers(i + 1, N, size=n_samples, dtype=np.int32)

    # move indices to GPU for gather
    i_gpu = cp.asarray(i)
    j_gpu = cp.asarray(j)

    # gather sampled entries to CPU
    obs = cp.asnumpy(P_obs_gpu[i_gpu, j_gpu])
    fit = cp.asnumpy(P_fit_gpu[i_gpu, j_gpu])

    # p1: all sampled
    p1 = float(np.corrcoef(fit, obs)[0, 1])

    # p2: sampled where obs>0
    m = obs > 0
    nz = int(m.sum())
    p2 = float(np.corrcoef(fit[m], obs[m])[0, 1]) if nz >= 1000 else float("nan")

    # free GPU temps
    del i_gpu, j_gpu
    return p1, p2, nz

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
def K2P_inplace(K, out_P, identity, eps_diag=1e-6, rc2=1.0):
    N = K.shape[0]
    d = cp.sum(K, axis=0, dtype=cp.float32)

    L11 = (-K[1:, 1:]).copy()
    idx = cp.arange(N - 1, dtype=cp.int32)
    L11[idx, idx] += d[1:]
    L11[idx, idx] += cp.float32(eps_diag)

    if _SOLVE_SUPPORTS_ASSUME_A:
        Q = _solve(L11, identity, assume_a='pos')
    else:
        Q = _solve(L11, identity)

    A = cp.diagonal(Q)

    out_P.fill(0)
    sub = out_P[1:N, 1:N]
    sub[...] = -2.0 * Q
    sub += A[None, :]
    sub += A[:, None]
    out_P[0, 1:N] = A
    out_P[1:N, 0] = A

    # (1 + 3G/rc2)^(-1.5)
    out_P *= cp.float32(3.0 / rc2)
    out_P += cp.float32(1.0)
    cp.maximum(out_P, cp.float32(1e-6), out=out_P)
    cp.power(out_P, cp.float32(-1.5), out=out_P)

    del Q, L11, A, sub, d
    return out_P


def Pdif2cost(P_dif, N):
    """RMSE over all entries"""
    return cp.sqrt(cp.sum(P_dif * P_dif)) / cp.float32(N)


def project_K_inplace(K, k_max=None):
    K[...] = cp.float32(0.5) * (K + K.T)
    cp.maximum(K, cp.float32(0.0), out=K)
    cp.fill_diagonal(K, cp.float32(0.0))
    if k_max is not None:
        cp.minimum(K, cp.float32(k_max), out=K)
    return K


def phic2_stable(
    K,
    P_obs,
    ETA=1e-6,
    ALPHA=1e-10,
    ITERATION_MAX=500,
    checkpoint_interval=0,
    eps_diag=1e-5,
    k_max=None,
    print_every_sec=10,
    patience=10,
    keep_best=True,
    decay=0.3,          # reduce ETA by this factor on plateau
    min_eta=1e-8,        # don't go below this
    max_decays=5,        # stop after this many decays
    # ---- Solution 2 (optional restart) ----
    enable_jitter_restart=True,
    jitter_after_decays=2,        # start jittering after this many decays
    jitter_sigma=1e-3,            # try 1e-4..1e-2 if needed
    rc2= 0.5
):
    """
    Original-style fixed-step gradient descent with two plateau escapes:
      1) On plateau: restore best_K BEFORE reducing ETA.
      2) Optional: jitter restart around best_K after a few decays.

    Requires:
      - K2P_inplace(K, P_buf, identity, eps_diag=...)
      - project_K_inplace(K, k_max=...)
      - Pdif2cost(P_dif, N)
      - clear_cupy_pools()
    """
    N = K.shape[0]
    stop_delta = ETA * ALPHA
    paras_fit = "%e\t%e\t%d\t" % (ETA, ALPHA, ITERATION_MAX)

    # Reused buffers
    P_buf = cp.empty((N, N), dtype=cp.float32)          # holds P then overwritten with P_dif
    identity = cp.eye(N - 1, dtype=cp.float32)

    # Initial cost + initial gradient (P_dif in P_buf)
    K2P_inplace(K, P_buf, identity, eps_diag=eps_diag, rc2=rc2)
    cp.subtract(P_buf, P_obs, out=P_buf)                # now P_buf is P_dif
    cost = Pdif2cost(P_buf, N)

    # trajectory on CPU: [cost, time, eta_used]
    c_traj = np.zeros((ITERATION_MAX + 1, 3), dtype=np.float64)
    c_traj[0, 0] = float(cost)
    c_traj[0, 1] = time.time()
    c_traj[0, 2] = float(ETA)

    best_cost = float(cost)
    best_iter = 0
    best_K = K.copy() if keep_best else None
    worse_count = 0
    decays_used = 0

    print(f"Starting optimization with N={N}, ETA={ETA:g}, ALPHA={ALPHA:g}")
    print(f"Initial cost: {float(cost):.6e}")
    print("iter\tcost\t\tcost_diff\teta\t\tt(sec)")

    last_print_time = time.time()

    for iteration in range(1, ITERATION_MAX + 1):
        cost_bk = cost

        # Fixed-step update (ETA may be reduced on plateau)
        K -= cp.float32(ETA) * P_buf
        project_K_inplace(K, k_max=k_max)

        # Recompute P_dif in-place
        K2P_inplace(K, P_buf, identity, eps_diag=eps_diag, rc2=rc2)
        cp.subtract(P_buf, P_obs, out=P_buf)
        cost = Pdif2cost(P_buf, N)

        cost_f = float(cost)
        cost_dif = float(cost_bk - cost)
        if iteration % 50 == 0:
        kmax = float(cp.max(K))
        kmean = float(cp.mean(K))
        print(f"   K stats: max={kmax:.3e} mean={kmean:.3e}")
            if (not np.isfinite(kmax)) or (kmax > 1e6):
                print("K exploded; stopping.")
                break
        c_traj[iteration, 0] = cost_f
        c_traj[iteration, 1] = time.time()
        c_traj[iteration, 2] = float(ETA)

        if not np.isfinite(cost_f):
            print(f"Diverged (non-finite cost) at iteration {iteration}")
            break

        # Best tracking
        if cost_f < best_cost:
            best_cost = cost_f
            best_iter = iteration
            worse_count = 0
            if keep_best:
                best_K = K.copy()
        else:
            worse_count += 1

        # ----------------------------
        # Solution 1 + Solution 2:
        # Plateau logic: restore best_K, reduce ETA, optional jitter restart
        # ----------------------------
        if worse_count >= patience:
            # Always jump back to the best-known basin before changing ETA
            if keep_best and best_K is not None:
                K = best_K.copy()

            # Stop if we can't/shouldn't decay further
            if ETA <= min_eta or decays_used >= max_decays:
                print(f"Stopping: plateau persists. Best at iter {best_iter} cost={best_cost:.6e}")
                if keep_best and best_K is not None:
                    K = best_K
                break

            # Reduce ETA
            ETA = max(ETA * decay, min_eta)
            decays_used += 1
            worse_count = 0

            did_restart = False

            # Optional: jitter restart around best_K after a few decays
            if enable_jitter_restart and (decays_used >= jitter_after_decays) and keep_best and (best_K is not None):
                sigma = cp.float32(jitter_sigma)
                noise = sigma * cp.random.standard_normal(K.shape, dtype=cp.float32)
                K = best_K.copy()
                K += cp.float32(0.5) * (noise + noise.T)
                project_K_inplace(K, k_max=k_max)
                did_restart = True

            # IMPORTANT: after restoring/restarting, recompute current gradient in P_buf
            K2P_inplace(K, P_buf, identity, eps_diag=eps_diag, rc2=rc2)
            cp.subtract(P_buf, P_obs, out=P_buf)

            if did_restart:
                print(f"Plateau: restored best_K + jitter restart, ETA -> {ETA:.2e} (decay {decays_used}/{max_decays})")
            else:
                print(f"Plateau: restored best_K, ETA -> {ETA:.2e} (decay {decays_used}/{max_decays})")

        # Print progress
        now = time.time()
        if (now - last_print_time) > print_every_sec or iteration == 1 or iteration % 10 == 0:
            elapsed = now - c_traj[0, 1]
            print(f"{iteration}\t{cost_f:.6e}\t{cost_dif:+.3e}\t{ETA:.1e}\t{elapsed:6.1f}")
            last_print_time = now

        # Optional convergence check (same spirit as original)
        if (0 < cost_dif < stop_delta) and (iteration > 3):
            print(f"Converged (stop_delta) at iteration {iteration}")
            break

        if iteration % 5 == 0:
            clear_cupy_pools()

        if checkpoint_interval and (iteration % checkpoint_interval == 0):
            ck = os.path.join(dataDir, f"checkpoint_iter{iteration}.K.npy")
            cp.save(ck, K)
            print(f"Checkpoint saved: {ck}")

    c_traj = c_traj[: iteration + 1]

    # Ensure we return best_K if requested (even if loop ended without plateau stop)
    if keep_best and best_K is not None:
        K = best_K

    return K, c_traj, paras_fit






def saveLg(fn, xy, ct):
    with open(fn, "w") as fw:
        fw.write(ct)
        for row in xy:
            if row.shape[0] >= 3:
                fw.write(f"{row[0]:.5e} {row[1]:.5e} {row[2]:.5e}\n")
            else:
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
    cp.fill_diagonal(P_obs, 1.0)

    # Output directory
    phic2_alpha = 1.0e-10
    dataDir = fhic[:fhic.rfind('.')] + "_phic2_a%7.1e_optimized" % (phic2_alpha)
    os.makedirs(dataDir, exist_ok=True)

    # Init K
    print("\nInitializing K matrix...")
    K_fit = cp.zeros((N, N), dtype=cp.float32)
    Init_K(K_fit, N, INIT_K0=0.1)

    # Settings
    ETA0 = 1e-6
    ITERS0 = 500  # you used 500 in the call; keep consistent

    print("\nStarting optimization...")
    start_opt = time.time()

    K_fit, c_traj, paras_fit = phic2_stable(
        K_fit, P_obs,
        ETA=ETA0,
        ALPHA=phic2_alpha,
        ITERATION_MAX=ITERS0,
        checkpoint_interval=100,
        eps_diag=1e-6,
        k_max=None,
        print_every_sec=10,
        patience=20,
        keep_best=True,
        decay=0.5,
        min_eta=1e-8,
        max_decays=8,
        enable_jitter_restart=False,
        rc2=0.5
    )

    opt_time = time.time() - start_opt
    print(f"\nOptimization completed in {opt_time:.2f}s ({opt_time/3600:.2f} hours)")

    # Save log + K
    fo = f"{dataDir}/N{N}"
    print("Saving results...")
    c_traj[:, 1] = c_traj[:, 1] - c_traj[0, 1]
    saveLg(fo + '.log', c_traj, "#%s\n#cost systemTime\n" % (paras_fit))
    cp.save(fo + '.K_fit.npy', K_fit)
    print(f"Saved: {fo}.K_fit.npy and log")

    # ---- Pearson check (sample-based) + adjacent diagnostic ----
    print("\nEstimating Pearson correlations by sampling...")
    clear_cupy_pools()

    P_fit_gpu = cp.empty((N, N), dtype=cp.float32)
    identity_eval = cp.eye(N - 1, dtype=cp.float32)

    print("Computing P_fit on GPU (one-time)...")
    K2P_inplace(K_fit, P_fit_gpu, identity_eval, eps_diag=1e-6, rc2=0.5)

    p1, p2, nz = pearson_sample(P_fit_gpu, P_obs, n_samples=1_000_000, seed=0)
    print(f"Sample Pearson p1 (upper triangle): {p1:.6f}")
    print(f"Sample Pearson p2 (obs>0):          {p2:.6f}   (nz samples={nz:,})")

    with open(f"{fo}.pearson_sample.txt", "w") as fw:
        fw.write(f"N {N}\n")
        fw.write("n_samples 1000000\n")
        fw.write(f"p1_upper_triangle {p1:.8f}\n")
        fw.write(f"p2_obs_gt_0 {p2:.8f}\n")
        fw.write(f"nz_samples {nz}\n")

    # Adjacent contact diagnostic (Lei Liu: good sign if ~0.1–0.4)
    adj_obs = cp.asnumpy(P_obs.diagonal(1))
    adj_fit = cp.asnumpy(P_fit_gpu.diagonal(1))

    def summarize_adj(x, name):
        x = x[np.isfinite(x)]
        return (f"{name}: mean={x.mean():.4f}, median={np.median(x):.4f}, "
                f"p10={np.quantile(x,0.10):.4f}, p90={np.quantile(x,0.90):.4f}, "
                f"min={x.min():.4f}, max={x.max():.4f}")

    print(summarize_adj(adj_obs, "P_obs diag+1"))
    print(summarize_adj(adj_fit, "P_fit diag+1"))

    # Free eval buffers
    del P_fit_gpu, identity_eval
    clear_cupy_pools()

    print("Done!")
