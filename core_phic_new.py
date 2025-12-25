import os
import sys
import time
import warnings
import cupy as cp
import numpy as np

# NOTE: pearsonr + dense triMask is disabled for large N (will OOM on CPU)
from scipy.stats import pearsonr

# IMPORTANT: use cupyx.scipy.linalg.solve to get assume_a='pos'
try:
    from cupyx.scipy.linalg import solve  # may not exist in your CuPy
    _SOLVE_SUPPORTS_ASSUME_A = True
except Exception:
    from cupy.linalg import solve
    _SOLVE_SUPPORTS_ASSUME_A = False

cp.set_printoptions(precision=3, linewidth=200)
warnings.filterwarnings('ignore')

if not len(sys.argv) == 2:
    print("usage:: python core_phic2_cupy_optimized.py normalized-HiC-Contact-Matrix")
    sys.exit()
fhic = str(sys.argv[1])  # HiC observation


def Init_K(K, N, INIT_K0):
    """Vectorized initialization of tridiagonal backbone"""
    idx = cp.arange(1, N, dtype=cp.int32)
    K[idx, idx - 1] = INIT_K0
    K[idx - 1, idx] = INIT_K0
    return K


def K2P_inplace(K, out_P, identity):
    """
    Convert K -> P, writing P into out_P (NxN float32 buffer).
    Memory-lean version:
      - does NOT build full L (NxN)
      - does NOT build M = 0.5*(Q+Q.T)
      - uses Q directly
    """
    N = K.shape[0]

    # Degree vector (float32)
    d = cp.sum(K, axis=0, dtype=cp.float32)

    # Build L11 = diag(d[1:]) - K[1:,1:] WITHOUT creating full L
    L11 = (-K[1:, 1:]).copy()               # (N-1)x(N-1)
    idx = cp.arange(N - 1, dtype=cp.int32)
    L11[idx, idx] += d[1:]

    # Solve L11 @ Q = I  (Cholesky path because assume_a='pos')
    if _SOLVE_SUPPORTS_ASSUME_A:
        Q = solve(L11, identity, assume_a='pos')
    else:
        Q = solve(L11, identity)  # fallback (no assume_a)  # (N-1)x(N-1)

    # diag(Q)
    A = cp.diagonal(Q)

    # Fill out_P with G first (reuse buffer), then convert in-place to P
    out_P.fill(0)
    sub = out_P[1:N, 1:N]
    sub[...] = -2.0 * Q
    sub += A[None, :]
    sub += A[:, None]
    out_P[0, 1:N] = A
    out_P[1:N, 0] = A

    # Convert G -> P in-place: P = (1 + 3G)^(-1.5)
    out_P *= 3.0
    out_P += 1.0
    cp.power(out_P, -1.5, out=out_P)

    # Free big temporaries ASAP
    del Q, L11, A, sub, d
    return out_P


def Pdif2cost(P_dif, N):
    """Compute RMSE cost (P_dif is NxN)"""
    return cp.sqrt(cp.sum(P_dif * P_dif)) / N


def clear_cupy_pools():
    """Helps fragmentation on long runs"""
    try:
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass


def phic2_optimized(K, P_obs, ETA=1.0e-4, ALPHA=1.0e-10, ITERATION_MAX=10,
                    checkpoint_interval=0):
    """
    Lean optimizer for very large N:
      - No momentum (saves a full NxN buffer)
      - No best_K copy (saves another full NxN)
      - Single NxN buffer reused for P_calc and then overwritten with P_dif
      - Adaptive eta kept simple (optional)
    """
    N = K.shape[0]
    stop_delta = ETA * ALPHA
    paras_fit = "%e\t%e\t%d\t" % (ETA, ALPHA, ITERATION_MAX)

    # One reusable NxN buffer: holds P_calc then P_dif in-place
    P_buf = cp.empty((N, N), dtype=cp.float32)

    # Identity for solve: (N-1)x(N-1) is huge; keep float32
    identity = cp.eye(N - 1, dtype=cp.float32)

    # Initial cost
    K2P_inplace(K, P_buf, identity)                 # P in P_buf
    cp.subtract(P_buf, P_obs, out=P_buf)            # overwrite: P_dif in P_buf
    cost = Pdif2cost(P_buf, N)

    # Store cost trajectory on CPU (avoid huge GPU c_traj for big ITERATION_MAX)
    c_traj = np.zeros((ITERATION_MAX + 1, 2), dtype=np.float64)
    c_traj[0, 0] = float(cost)
    c_traj[0, 1] = time.time()

    eta = ETA
    eta_min = ETA * 0.1

    print(f"Starting optimization with N={N}, ETA={ETA}, ALPHA={ALPHA}")
    print(f"Initial cost: {float(cost):.6e}")
    print("Iteration\tCost\t\tCost_diff\tEta\t\tTime(s)")

    last_print_time = time.time()
    iteration = 1

    while iteration <= ITERATION_MAX:
        cost_bk = cost

        # Gradient step (no momentum): K -= eta * P_dif
        K -= eta * P_buf

        # Recompute cost
        K2P_inplace(K, P_buf, identity)             # P in P_buf
        cp.subtract(P_buf, P_obs, out=P_buf)        # P_dif in P_buf
        cost = Pdif2cost(P_buf, N)

        c_traj[iteration, 0] = float(cost)
        c_traj[iteration, 1] = time.time()

        cost_dif = cost_bk - cost

        # Simple adaptive eta
        if float(cost_dif) > 0:
            eta = min(eta * 1.05, ETA * 2.0)
        else:
            eta = max(eta * 0.5, eta_min)

        # Print progress
        current_time = time.time()
        if (current_time - last_print_time > 10) or (iteration % 10 == 0):
            elapsed = current_time - c_traj[0, 1]
            print(f"{iteration}\t\t{float(cost):.6e}\t{float(cost_dif):+.4e}\t{eta:.4e}\t{elapsed:.1f}")
            last_print_time = current_time

        # Optional checkpoint (save ONLY K as .npy; npz_compressed is painful at this size)
        if checkpoint_interval and (iteration % checkpoint_interval == 0):
            checkpoint_file = os.path.join(dataDir, f"checkpoint_iter{iteration}.K.npy")
            cp.save(checkpoint_file, K)
            print(f"Checkpoint saved: {checkpoint_file}")

        # Basic stopping
        diverged = bool(cp.isnan(cost))
        converged = (0 < float(cost_dif) < stop_delta) and (iteration > 3)

        if diverged:
            print(f"Optimization diverged at iteration {iteration}")
            break
        if converged:
            print(f"Converged at iteration {iteration}")
            break

        # Periodic memory pool cleanup (helps long runs)
        if iteration % 5 == 0:
            clear_cupy_pools()

        iteration += 1

    c_traj = c_traj[:iteration + 1]
    return K, c_traj, paras_fit


def saveLg(fn, xy, ct):
    with open(fn, 'w') as fw:
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


if __name__ == '__main__':
    if not os.path.isfile(fhic):
        print('Cannot find ' + fhic)
        sys.exit()

    print(f"Reading Hi-C matrix from {fhic}...")
    start_read = time.time()

    # Load as float32 on CPU first (saves RAM and transfer)
    try:
        P_obs_np = np.loadtxt(fhic, comments='#', dtype=np.float32)
        P_obs = cp.asarray(P_obs_np, dtype=cp.float32)
        del P_obs_np
    except Exception:
        # Fallback (still float32)
        P_list = []
        with open(fhic) as fr:
            for line in fr:
                if line and line[0] != '#':
                    lt = line.strip().split()
                    P_list.append(list(map(float, lt)))
        P_obs = cp.asarray(np.array(P_list, dtype=np.float32), dtype=cp.float32)
        del P_list

    N = int(P_obs.shape[0])
    print(f"Matrix size: N={N} ({N}x{N} = {N*N:,} elements)")
    print(f"Read time: {time.time() - start_read:.2f}s")

    # Replace NaN/Inf with 0 in-place, and set diagonal to 1
    cp.nan_to_num(P_obs, copy=False)
    P_obs += cp.eye(N, dtype=cp.float32)

    # Output directory
    phic2_alpha = 1.0e-10
    dataDir = fhic[:fhic.rfind('.')] + "_phic2_a%7.1e_optimized" % (phic2_alpha)
    os.makedirs(dataDir, exist_ok=True)

    # IMPORTANT: prevent CPU-side Pearson + mask crash for large N
    BIG_N = 8000

    # Initialize K
    print("\nInitializing K matrix...")
    K_fit = cp.zeros((N, N), dtype=cp.float32)
    K_fit = Init_K(K_fit, N, INIT_K0=0.5)

    # Run optimization
    print("\nStarting optimization...")
    start_opt = time.time()

    # For N~30895 you should start small (e.g., 1–5 iters) and scale up carefully.
    K_fit, c_traj, paras_fit = phic2_optimized(
        K_fit,
        P_obs,
        ETA=1e-4,
        ALPHA=phic2_alpha,
        ITERATION_MAX=5,              # <<< start small for N~31k
        checkpoint_interval=0
    )

    opt_time = time.time() - start_opt
    print(f"\nOptimization completed in {opt_time:.2f}s ({opt_time/3600:.2f} hours)")

    # Save results
    fo = "%s/N%d" % (dataDir, N)
    print("Saving results...")

    c_traj[:, 1] = c_traj[:, 1] - c_traj[0, 1]
    saveLg(fo + '.log', c_traj, "#%s\n#cost systemTime\n" % (paras_fit))

    # Save K as .npy (fast, compact-ish, avoids huge text write)
    cp.save(fo + '.K_fit.npy', K_fit)

    # For big N, skip dense P_fit generation + Pearson masks (CPU will suffer)
    if N <= BIG_N:
        print("Computing final P_fit (small N mode)...")
        P_buf = cp.empty((N, N), dtype=cp.float32)
        identity = cp.eye(N - 1, dtype=cp.float32)
        K2P_inplace(K_fit, P_buf, identity)
        P_fit = cp.asnumpy(P_buf)
        P_obs_cpu = cp.asnumpy(P_obs)

        triMask = np.where(np.triu(np.ones((N, N), dtype=np.uint8), 1) > 0)
        pijMask = np.where(np.triu(P_obs_cpu, 1) > 0)
        p1 = pearsonr(P_fit[triMask], P_obs_cpu[triMask])[0]
        p2 = pearsonr(P_fit[pijMask], P_obs_cpu[pijMask])[0]
        np.save(fo + '.P_fit.npy', P_fit)
        print(f"Pearson correlations: {p1:.6f} (all), {p2:.6f} (non-zero)")
    else:
        print(f"N={N} too large: skipping P_fit + Pearson mask evaluation.")
        print("Saved: K_fit.npy and log only.")

    print(f"\nResults saved to {dataDir}")
    print("Done!")
