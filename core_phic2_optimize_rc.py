import os
import sys
import time
import warnings
import cupy as cp
import numpy as np
from scipy.stats import pearsonr
cp.set_printoptions(precision=3, linewidth=200)
warnings.filterwarnings('ignore')

# PHi-C2 with automatic rc optimization
# Tries different rc values to find best fit

if not len(sys.argv) >= 2:
    print("usage:: python core_phic2_optimize_rc.py normalized-HiC-Contact-Matrix")
    sys.exit()

fhic = str(sys.argv[1])

print("="*80)
print("PHi-C2 with Capture Radius (rc) Optimization")
print("="*80)

_SOLVE_SUPPORTS_ASSUME_A = True
try:
    test = cp.eye(2, dtype=cp.float32)
    cp.linalg.solve(test, test, assume_a='pos')
except TypeError:
    _SOLVE_SUPPORTS_ASSUME_A = False

def Init_K(N, INIT_K0, dtype):
    K = cp.zeros((N, N), dtype=dtype)
    for i in range(1, N):
        K[i, i-1] = K[i-1, i] = INIT_K0
    return K

def K2P_inplace(K, out_P, identity, eps_diag=1e-5, rc2=1):
    """Memory-optimized K2P with configurable rc"""
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
    
    # Use configurable rc2
    out_P *= cp.float32(3.0 / rc2)
    out_P += cp.float32(1.0)
    cp.maximum(out_P, cp.float32(1e-6), out=out_P)
    cp.power(out_P, cp.float32(-1.5), out=out_P)
    
    del Q, L11, A, sub, d
    return out_P

def cost_func(P_dif, N):
    return cp.sqrt(cp.sum(P_dif**2)) / N

def quick_optimize(K, P_obs, N, identity, rc2, max_iter=200):
    """Quick optimization for a given rc value"""
    P_fit = cp.zeros((N, N), dtype=cp.float32)
    P_dif = cp.zeros((N, N), dtype=cp.float32)
    
    # Estimate ETA based on gradient
    K2P_inplace(K, P_fit, identity, rc2=rc2)
    grad_norm = cp.sqrt(cp.mean((P_fit - P_obs)**2))
    ETA = 1e-4 * (2869.0/N)**2 * min(1.0, 1e-4/float(grad_norm))
    ETA = np.clip(ETA, 1e-8, 1e-4)
    
    best_cost = float('inf')
    best_K = None
    
    for iteration in range(max_iter):
        K2P_inplace(K, P_fit, identity, rc2=rc2)
        P_dif[...] = P_fit - P_obs
        cost = cost_func(P_dif, N)
        
        if cost < best_cost:
            best_cost = cost
            best_K = K.copy()
        
        # Gradient step
        K -= ETA * P_dif
        K = cp.maximum(K, 0)
        K = 0.5 * (K + K.T)
        K = cp.minimum(K, 1e3)
        
        # Early stop if diverging
        if iteration > 50 and cost > best_cost * 1.2:
            break
    
    del P_fit, P_dif
    return best_K, best_cost

# Load matrix
print(f"\nLoading: {fhic}")
P_obs = []
with open(fhic) as fr:
    for line in fr:
        if not line[0] == '#':
            P_obs.append(list(map(float, line.strip().split())))
            if len(P_obs) % 5000 == 0:
                print(f"  {len(P_obs)} rows...")

P_obs = cp.array(P_obs, dtype=cp.float32)
N = len(P_obs)
print(f"\nLoaded {N}×{N}")

cp.nan_to_num(P_obs, copy=False)
P_obs = P_obs + cp.eye(N, dtype=cp.float32)

print(f"  Non-zero: {int(cp.count_nonzero(P_obs)):,}")
print(f"  Mean: {float(cp.mean(P_obs)):.2e}")

# Analyze contact distance distribution
print("\nAnalyzing contact distance distribution...")
# Sample contacts to estimate characteristic length scale
sample_size = min(1000000, N*N//2)
rng = np.random.default_rng(0)
i_samp = rng.integers(0, N-1, size=sample_size, dtype=np.int32)
j_samp = rng.integers(i_samp+1, N, size=sample_size, dtype=np.int32)
i_gpu = cp.asarray(i_samp)
j_gpu = cp.asarray(j_samp)
p_vals = cp.asnumpy(P_obs[i_gpu, j_gpu])
dists = j_samp - i_samp

# Compute weighted mean distance
nonzero = p_vals > 0
if nonzero.sum() > 0:
    mean_dist = np.average(dists[nonzero], weights=p_vals[nonzero])
    print(f"  Weighted mean contact distance: {mean_dist:.1f} bins")
    
    # Suggest rc based on data
    # Smaller mean_dist → smaller rc
    # Larger mean_dist → larger rc
    if mean_dist < 50:
        suggested_rc_range = [0.5, 0.7, 1.0, 1.5]
    elif mean_dist < 200:
        suggested_rc_range = [0.7, 1.0, 1.5, 2.0]
    else:
        suggested_rc_range = [1.0, 1.5, 2.0, 3.0]
    
    print(f"  Suggested rc range: {suggested_rc_range}")
else:
    suggested_rc_range = [0.5, 1.0, 1.5, 2.0]
    print(f"  Using default rc range: {suggested_rc_range}")

del i_gpu, j_gpu

# Pre-allocate
identity = cp.eye(N-1, dtype=cp.float32)

# Test different rc values
print("\n" + "="*80)
print(f"Testing {len(suggested_rc_range)} different rc values...")
print("="*80)

results = []

for idx, rc in enumerate(suggested_rc_range):
    rc2 = rc * rc
    print(f"\n[{idx+1}/{len(suggested_rc_range)}] Testing rc = {rc:.2f} (rc² = {rc2:.2f})")
    
    # Initialize K
    K_init = Init_K(N, 0.5, cp.float32)
    
    start = time.time()
    K_opt, final_cost = quick_optimize(K_init, P_obs, N, identity, rc2, max_iter=200)
    elapsed = time.time() - start
    
    results.append({
        'rc': rc,
        'rc2': rc2,
        'cost': float(final_cost),
        'K': K_opt.copy(),
        'time': elapsed
    })
    
    print(f"  Final cost: {final_cost:.6e} (time: {elapsed:.1f}s)")
    
    del K_init

# Find best rc
results.sort(key=lambda x: x['cost'])
best = results[0]

print("\n" + "="*80)
print("RC OPTIMIZATION RESULTS")
print("="*80)
for i, r in enumerate(results):
    marker = "  ← BEST" if i == 0 else ""
    improvement = (results[-1]['cost'] - r['cost']) / results[-1]['cost'] * 100
    print(f"rc={r['rc']:.2f}  →  Cost: {r['cost']:.6e}  "
          f"(improvement: {improvement:+.2f}%){marker}")

best_improvement = (results[-1]['cost'] - results[0]['cost']) / results[-1]['cost'] * 100
print(f"\nBest rc: {best['rc']:.2f}")
print(f"Improvement over worst: {best_improvement:.2f}%")

# Quick Pearson check with best rc
print(f"\n" + "="*80)
print("Quick Pearson Correlation Check (with best rc)")
print("="*80)

K_best = best['K']
P_fit = cp.zeros((N, N), dtype=cp.float32)
K2P_inplace(K_best, P_fit, identity, rc2=best['rc2'])

# Sample-based Pearson
def pearson_sample(P_fit_gpu, P_obs_gpu, n_samples=1_000_000):
    rng = np.random.default_rng(0)
    N = P_obs_gpu.shape[0]
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

p1, p2, nz = pearson_sample(P_fit, P_obs)
print(f"\nWith rc = {best['rc']:.2f}:")
print(f"  Pearson (sampled, all):   {p1:.6f}")
print(f"  Pearson (sampled, obs>0): {p2:.6f}")

print(f"\n" + "="*80)
print("RECOMMENDATION")
print("="*80)
print(f"\nBest capture radius found: rc = {best['rc']:.2f}")
print(f"  This gives lowest cost: {best['cost']:.6e}")
print(f"  Pearson correlation: {p1:.6f} (sampled)")
print()

if best['rc'] != 1.0:
    improvement_vs_default = (results[[r['rc'] for r in results].index(1.0)]['cost'] - best['cost']) / \
                            results[[r['rc'] for r in results].index(1.0)]['cost'] * 100 \
                            if 1.0 in [r['rc'] for r in results] else 0
    
    print(f"⚠ Note: rc = {best['rc']:.2f} is different from default (1.0)")
    if improvement_vs_default > 0:
        print(f"  This gives {improvement_vs_default:.2f}% better cost than rc=1.0!")
    print()

print("To run full optimization with optimal rc:")
print(f"  1. Edit core_phic2_cupy_optimized.py")
print(f"  2. Change: K2P_inplace(K, out_P, identity, rc2=1)")
print(f"     To:     K2P_inplace(K, out_P, identity, rc2={best['rc2']:.2f})")
print(f"  3. Run: python core_phic2_cupy_optimized.py {fhic}")
print()
print("Or use the create_with_optimal_rc.py script to do this automatically.")
print("="*80)

# Save best K for inspection
dataDir = fhic[:fhic.rfind('.')] + f"_phic2_rc_optimization"
os.makedirs(dataDir, exist_ok=True)

# Save summary
summary_file = f"{dataDir}/rc_optimization_summary.txt"
with open(summary_file, 'w') as f:
    f.write("PHi-C2 Capture Radius Optimization Results\n")
    f.write("="*80 + "\n\n")
    f.write(f"Matrix: {N}×{N}\n")
    f.write(f"Input: {fhic}\n\n")
    f.write("Results:\n")
    for r in results:
        f.write(f"  rc={r['rc']:.2f}: cost={r['cost']:.6e}\n")
    f.write(f"\nBest rc: {best['rc']:.2f}\n")
    f.write(f"Best cost: {best['cost']:.6e}\n")
    f.write(f"Pearson (sampled): {p1:.6f}\n")

print(f"\nSummary saved to: {summary_file}")
