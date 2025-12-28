#!/usr/bin/env python3
"""core_phic2_cupy_updated.py

A practical update of the original `core_phic2_cupy.py` that stays close
in spirit (fixed-step gradient descent on P_fit - P_obs), but adds a few
safety/robustness improvements that matter for large N.

Key changes vs original:
  1) Input: load float32, NaN/Inf->0, set diagonal EXACTLY to 1.
  2) K projection each step: symmetrize, clamp to >=0, force diag(K)=0.
  3) K->P stabilized: solve instead of explicit inverse; eps_diag jitter;
     guard (1 + 3G/rc2) base to avoid NaNs.
  4) Plateau handling: keep best_K, decay ETA when not improving.
  5) Pearson: for large N, estimate by sampling (no full masks on CPU).

Usage:
  python core_phic2_cupy_updated.py <normalized_contact_matrix.txt>

Optional flags (see --help):
  --eta, --alpha, --iters, --init_k0, --eps_diag, --rc2
  --patience, --decay, --min_eta, --max_decays
  --target_adj (optional rescaling of off-diagonals to target median P[i,i+1])
  --pearson_samples

Notes:
- This script still allocates O(N^2) arrays on GPU (P_obs, K, P_buf, etc.).
  That is inherent to this formulation.
"""

import os
import sys
import time
import argparse
import warnings

import cupy as cp
import numpy as np

warnings.filterwarnings("ignore")
cp.set_printoptions(precision=3, linewidth=200)

# ---------- Robust solve import (works across CuPy builds) ----------
try:
    from cupyx.scipy.linalg import solve as _solve
    _SOLVE_SUPPORTS_ASSUME_A = True
except Exception:
    from cupy.linalg import solve as _solve
    _SOLVE_SUPPORTS_ASSUME_A = False


# ---------- IO ----------

def load_contact_matrix_txt(path: str) -> cp.ndarray:
    """Load whitespace-delimited matrix, ignoring '#' comment lines."""
    # Fast path: numpy loadtxt to float32 then upload
    try:
        mat = np.loadtxt(path, comments="#", dtype=np.float32)
        return cp.asarray(mat, dtype=cp.float32)
    except Exception:
        # Fallback parser (still float32)
        rows = []
        with open(path, "r") as f:
            for line in f:
                if not line or line[0] == "#":
                    continue
                rows.append([float(x) for x in line.split()])
        return cp.asarray(np.asarray(rows, dtype=np.float32), dtype=cp.float32)


# ---------- Model helpers ----------

def init_K_tridiag(K: cp.ndarray, init_k0: float) -> None:
    """Initialize the polymer backbone: K[i,i-1]=K[i-1,i]=init_k0 for i>=1."""
    N = K.shape[0]
    idx = cp.arange(1, N, dtype=cp.int32)
    K[idx, idx - 1] = cp.float32(init_k0)
    K[idx - 1, idx] = cp.float32(init_k0)


def project_K_inplace(K: cp.ndarray, k_max: float | None = None) -> None:
    """Keep K in a safer numerical region."""
    K[...] = cp.float32(0.5) * (K + K.T)
    cp.maximum(K, cp.float32(0.0), out=K)
    cp.fill_diagonal(K, cp.float32(0.0))
    if k_max is not None:
        cp.minimum(K, cp.float32(k_max), out=K)


def clear_cupy_pools() -> None:
    try:
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass


# ---------- Core math ----------

def K2P_inplace(
    K: cp.ndarray,
    out_P: cp.ndarray,
    identity: cp.ndarray,
    *,
    eps_diag: float = 1e-6,
    rc2: float = 1.0,
    base_floor: float = 1e-6,
) -> None:
    """Compute P from K into out_P (both NxN float32 on GPU)."""
    N = K.shape[0]

    # degree
    d = cp.sum(K, axis=0, dtype=cp.float32)

    # L11 = diag(d[1:]) - K[1:,1:]
    L11 = (-K[1:, 1:]).copy()
    idx = cp.arange(N - 1, dtype=cp.int32)
    L11[idx, idx] += d[1:]
    L11[idx, idx] += cp.float32(eps_diag)

    # Solve L11 Q = I  (equivalent to Q = inv(L11), but more stable)
    if _SOLVE_SUPPORTS_ASSUME_A:
        Q = _solve(L11, identity, assume_a="pos")
    else:
        Q = _solve(L11, identity)

    A = cp.diagonal(Q)

    # Build G into out_P (reuse buffer)
    out_P.fill(0)
    sub = out_P[1:N, 1:N]
    sub[...] = -2.0 * Q
    sub += A[None, :]
    sub += A[:, None]
    out_P[0, 1:N] = A
    out_P[1:N, 0] = A

    # P = (1 + 3G/rc2)^(-1.5)
    out_P *= cp.float32(3.0 / float(rc2))
    out_P += cp.float32(1.0)
    cp.maximum(out_P, cp.float32(base_floor), out=out_P)
    cp.power(out_P, cp.float32(-1.5), out=out_P)

    # release big temps
    del Q, L11, A, sub, d


def Pdif2cost(P_dif: cp.ndarray, N: int) -> cp.ndarray:
    """RMSE over all entries (same functional form as original)."""
    return cp.sqrt(cp.sum(P_dif * P_dif)) / cp.float32(N)


# ---------- Metrics ----------

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


def adjacent_stats(P: cp.ndarray):
    x = cp.asnumpy(P.diagonal(1))
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "p10": float("nan"), "p90": float("nan"), "max": float("nan")}
    return {
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "p10": float(np.quantile(x, 0.10)),
        "p90": float(np.quantile(x, 0.90)),
        "max": float(x.max()),
    }


# ---------- Optimizer (simple + plateau decay) ----------

def phic2_optimized(
    K: cp.ndarray,
    P_obs: cp.ndarray,
    *,
    ETA: float,
    ALPHA: float,
    ITERATION_MAX: int,
    eps_diag: float,
    rc2: float,
    k_max: float | None,
    patience: int,
    decay: float,
    min_eta: float,
    max_decays: int,
    checkpoint_interval: int,
    out_dir: str,
    print_every_sec: float,
    keep_best: bool,
):
    N = int(K.shape[0])
    stop_delta = ETA * ALPHA
    paras_fit = f"{ETA:e}\t{ALPHA:e}\t{ITERATION_MAX:d}\t"

    # Buffers
    P_buf = cp.empty((N, N), dtype=cp.float32)  # used as P then overwritten to P_dif
    identity = cp.eye(N - 1, dtype=cp.float32)

    # initial gradient + cost
    K2P_inplace(K, P_buf, identity, eps_diag=eps_diag, rc2=rc2)
    cp.subtract(P_buf, P_obs, out=P_buf)
    cost = Pdif2cost(P_buf, N)

    c_traj = np.zeros((ITERATION_MAX + 1, 3), dtype=np.float64)
    t0 = time.time()
    c_traj[0, 0] = float(cost)
    c_traj[0, 1] = t0
    c_traj[0, 2] = float(ETA)

    best_cost = float(cost)
    best_iter = 0
    best_K = K.copy() if keep_best else None

    worse_count = 0
    decays_used = 0

    print(f"Starting optimization with N={N}, ETA={ETA:g}, ALPHA={ALPHA:g}")
    print(f"Initial cost: {float(cost):.6e}")
    print("iter\tcost\t\tcost_diff\teta\t\tt(sec)")

    last_print = time.time()

    for it in range(1, ITERATION_MAX + 1):
        cost_bk = cost

        # gradient-like step: K <- K - ETA * (P_fit - P_obs)
        K -= cp.float32(ETA) * P_buf
        project_K_inplace(K, k_max=k_max)

        # refresh gradient proxy
        K2P_inplace(K, P_buf, identity, eps_diag=eps_diag, rc2=rc2)
        cp.subtract(P_buf, P_obs, out=P_buf)
        cost = Pdif2cost(P_buf, N)

        cost_f = float(cost)
        cost_dif = float(cost_bk - cost)

        c_traj[it, 0] = cost_f
        c_traj[it, 1] = time.time()
        c_traj[it, 2] = float(ETA)

        # NaN guard
        if not np.isfinite(cost_f):
            print(f"Diverged (non-finite cost) at iter {it}")
            break

        # best tracking
        if cost_f < best_cost:
            best_cost = cost_f
            best_iter = it
            worse_count = 0
            if keep_best:
                best_K = K.copy()
        else:
            worse_count += 1

        # plateau: restore best_K then decay ETA
        if worse_count >= patience:
            if keep_best and best_K is not None:
                K[...] = best_K
                # restore gradient at best_K
                K2P_inplace(K, P_buf, identity, eps_diag=eps_diag, rc2=rc2)
                cp.subtract(P_buf, P_obs, out=P_buf)

            if ETA <= min_eta or decays_used >= max_decays:
                print(f"Stopping: plateau persists. Best at iter {best_iter} cost={best_cost:.6e}")
                if keep_best and best_K is not None:
                    K[...] = best_K
                break

            ETA = max(ETA * decay, min_eta)
            decays_used += 1
            worse_count = 0
            print(f"Plateau: restored best_K, ETA -> {ETA:.2e} (decay {decays_used}/{max_decays})")

        # occasional K stats
        if it % 50 == 0:
            kmax = float(cp.max(K))
            kmean = float(cp.mean(K))
            print(f"   K stats: max={kmax:.3e} mean={kmean:.3e}")

        # progress print
        now = time.time()
        if it == 1 or it % 10 == 0 or (now - last_print) > print_every_sec:
            elapsed = now - t0
            print(f"{it}\t{cost_f:.6e}\t{cost_dif:+.3e}\t{ETA:.1e}\t{elapsed:6.1f}")
            last_print = now

        # original-style convergence check
        if (0 < cost_dif < stop_delta) and (it > 3):
            print(f"Converged (stop_delta) at iter {it}")
            break

        if it % 5 == 0:
            clear_cupy_pools()

        if checkpoint_interval and (it % checkpoint_interval == 0):
            ck = os.path.join(out_dir, f"checkpoint_iter{it}.K.npy")
            cp.save(ck, K)
            print(f"Checkpoint saved: {ck}")

    # trim
    last_it = min(it, ITERATION_MAX)
    c_traj = c_traj[: last_it + 1]

    if keep_best and best_K is not None:
        K[...] = best_K

    # normalize time to start
    c_traj[:, 1] -= c_traj[0, 1]

    return K, c_traj, paras_fit


# ---------- Save helpers ----------

def saveLg(path: str, arr: np.ndarray, header: str):
    with open(path, "w") as f:
        f.write(header)
        for row in arr:
            f.write(" ".join(f"{v:.5e}" for v in row) + "\n")


# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fhic", help="normalized Hi-C contact probability matrix (whitespace-delimited)")
    ap.add_argument("--eta", type=float, default=1e-6)
    ap.add_argument("--alpha", type=float, default=1e-10)
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--init_k0", type=float, default=0.5)
    ap.add_argument("--eps_diag", type=float, default=1e-6)
    ap.add_argument("--rc2", type=float, default=1.0)
    ap.add_argument("--k_max", type=float, default=None)
    ap.add_argument("--patience", type=int, default=200)
    ap.add_argument("--decay", type=float, default=0.5)
    ap.add_argument("--min_eta", type=float, default=1e-8)
    ap.add_argument("--max_decays", type=int, default=8)
    ap.add_argument("--checkpoint_interval", type=int, default=100)
    ap.add_argument("--print_every_sec", type=float, default=10.0)
    ap.add_argument("--keep_best", action="store_true", default=True)

    # Optional: rescale off-diagonals to target median P[i,i+1]
    ap.add_argument("--target_adj", type=float, default=None,
                    help="If set (e.g. 0.2), rescale OFF-diagonals so median(P[i,i+1]) matches target.")

    # Pearson sampling
    ap.add_argument("--pearson_samples", type=int, default=1_000_000)

    args = ap.parse_args()

    fhic = args.fhic
    if not os.path.isfile(fhic):
        print(f"Cannot find {fhic}")
        sys.exit(1)

    print(f"Reading Hi-C matrix from {fhic}...")
    t_read = time.time()
    P_obs = load_contact_matrix_txt(fhic)
    N = int(P_obs.shape[0])
    print(f"Matrix size: N={N} ({N}x{N} = {N*N:,} elements)")
    print(f"Read time: {time.time() - t_read:.2f}s")

    # sanitize
    cp.nan_to_num(P_obs, copy=False)
    cp.fill_diagonal(P_obs, cp.float32(1.0))

    # Optional off-diagonal rescale to target P[i,i+1]
    if args.target_adj is not None:
        st = adjacent_stats(P_obs)
        med = st["median"]
        if np.isfinite(med) and med > 0:
            s = float(args.target_adj) / float(med)
            print(f"P_obs diag+1 median={med:.7g} -> scaling off-diagonals by {s:.3f} to target {args.target_adj}")
            # scale only off-diagonals (keep diag=1)
            P_obs *= cp.float32(s)
            cp.fill_diagonal(P_obs, cp.float32(1.0))
        else:
            print("Warning: could not compute a positive median for diag+1; skipping scaling")

    # Report diag+1 stats (Lei Liu heuristic)
    st = adjacent_stats(P_obs)
    print(f"P_obs diag+1 stats: mean {st['mean']:.6g} median {st['median']:.6g} p10 {st['p10']:.6g} p90 {st['p90']:.6g} max {st['max']:.6g}")

    # output directory
    data_dir = fhic[: fhic.rfind(".")] + f"_phic2_a{args.alpha:7.1e}_updated"
    os.makedirs(data_dir, exist_ok=True)
    out_prefix = os.path.join(data_dir, f"N{N}")

    # init K
    print("\nInitializing K matrix...")
    K = cp.zeros((N, N), dtype=cp.float32)
    init_K_tridiag(K, args.init_k0)
    project_K_inplace(K, k_max=args.k_max)

    # optimize
    print("\nStarting optimization...")
    t0 = time.time()
    K, c_traj, paras_fit = phic2_optimized(
        K,
        P_obs,
        ETA=args.eta,
        ALPHA=args.alpha,
        ITERATION_MAX=args.iters,
        eps_diag=args.eps_diag,
        rc2=args.rc2,
        k_max=args.k_max,
        patience=args.patience,
        decay=args.decay,
        min_eta=args.min_eta,
        max_decays=args.max_decays,
        checkpoint_interval=args.checkpoint_interval,
        out_dir=data_dir,
        print_every_sec=args.print_every_sec,
        keep_best=args.keep_best,
    )

    opt_time = time.time() - t0
    print(f"\nOptimization completed in {opt_time:.2f}s ({opt_time/3600:.2f} hours)")

    # save
    print("Saving results...")
    saveLg(out_prefix + ".log", c_traj, f"#{paras_fit}\n#cost systemTime eta_used\n")
    cp.save(out_prefix + ".K_fit.npy", K)
    print(f"Saved: {out_prefix}.K_fit.npy and log")

    # Evaluate Pearson by sampling (requires one P_fit compute)
    print("\nEstimating Pearson correlations by sampling...")
    clear_cupy_pools()

    P_fit = cp.empty((N, N), dtype=cp.float32)
    identity_eval = cp.eye(N - 1, dtype=cp.float32)

    print("Computing P_fit on GPU (one-time)...")
    K2P_inplace(K, P_fit, identity_eval, eps_diag=args.eps_diag, rc2=args.rc2)

    p1, p2, nz = pearson_sample(P_fit, P_obs, n_samples=args.pearson_samples, seed=0)
    print(f"Sample Pearson p1 (upper triangle): {p1:.6f}")
    print(f"Sample Pearson p2 (obs>0):          {p2:.6f}   (nz samples={nz:,})")

    st_fit = adjacent_stats(P_fit)
    print(
        f"P_fit diag+1 stats: mean {st_fit['mean']:.6g} median {st_fit['median']:.6g} "
        f"p10 {st_fit['p10']:.6g} p90 {st_fit['p90']:.6g} max {st_fit['max']:.6g}"
    )

    with open(out_prefix + ".pearson_sample.txt", "w") as f:
        f.write(f"N {N}\n")
        f.write(f"pearson_samples {args.pearson_samples}\n")
        f.write(f"p1_upper_triangle {p1:.8f}\n")
        f.write(f"p2_obs_gt_0 {p2:.8f}\n")
        f.write(f"nz_samples {nz}\n")
        f.write(
            f"Pobs_adj_median {st['median']:.8g}\nPfit_adj_median {st_fit['median']:.8g}\n"
        )

    del P_fit, identity_eval
    clear_cupy_pools()

    print("Done!")


if __name__ == "__main__":
    main()
