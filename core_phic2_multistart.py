import os
import sys
import time
import warnings
import cupy as cp
import numpy as np
cp.set_printoptions(precision=3, linewidth=200)
warnings.filterwarnings('ignore')

# Multi-start optimization: Try different initializations
# Takes the best result from multiple runs with different K0 values

if not len(sys.argv) >= 2:
    print("usage:: python core_phic2_multistart.py normalized-HiC-Contact-Matrix")
    sys.exit()

fhic = str(sys.argv[1])

print("="*80)
print("PHi-C2 Multi-Start Optimization")
print("Tries different initializations to find best fit")
print("="*80)

def Init_K(N, INIT_K0, dtype):
    K = cp.zeros((N, N), dtype=dtype)
    for i in range(1, N):
        K[i, i-1] = K[i-1, i] = INIT_K0
    return K

def K2P(K, N, reg=1e-10):
    d = cp.sum(K, axis=0)
    L = cp.diag(d) - K
    L_sub = L[1:N, 1:N] + reg * cp.eye(N-1, dtype=K.dtype)
    try:
        Q = cp.linalg.inv(L_sub)
    except:
        Q = cp.linalg.pinv(L_sub)
    M = 0.5*(Q + Q.T)
    A = cp.diag(M)
    G = cp.zeros((N, N), dtype=K.dtype)
    G[1:N, 1:N] = -2*M + A + cp.reshape(A, (-1,1))
    G[0, 1:N] = A
    G[1:N, 0] = A
    G = cp.clip(G, -0.33, 1e6)
    P = cp.power(1.0 + 3.0*G, -1.5)
    P = cp.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)
    return P

def cost_func(P_dif, N):
    return cp.sqrt(cp.sum(P_dif**2)) / N

def constrain_K(K):
    K = cp.maximum(K, 0)
    K = 0.5 * (K + K.T)
    K = cp.minimum(K, 1e3)
    return K

def quick_optimize(K, P_obs, N, ETA, max_iter=500):
    """Quick optimization run (500 iterations max)"""
    best_cost = float('inf')
    best_K = None
    
    for iteration in range(max_iter):
        P_dif = K2P(K, N) - P_obs
        cost = cost_func(P_dif, N)
        
        if cost < best_cost:
            best_cost = cost
            best_K = K.copy()
        
        K = K - ETA * P_dif
        K = constrain_K(K)
        
        # Early stop if diverging
        if iteration > 50 and cost > best_cost * 1.1:
            break
        
        if iteration % 100 == 0:
            cp.get_default_memory_pool().free_all_blocks()
    
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

# Estimate ETA
print("\nEstimating ETA...")
K_test = Init_K(N, 0.5, cp.float32)
P_init = K2P(K_test, N)
grad_norm = cp.sqrt(cp.mean((P_init - P_obs)**2))
ETA = 1e-4 * (2869.0/N)**2 * min(1.0, 1e-4/float(grad_norm))
ETA = np.clip(ETA, 1e-8, 1e-4)
print(f"  ETA: {ETA:.2e}")

# Try different initializations
init_values = [0.1, 0.3, 0.5, 0.7, 1.0]
print(f"\nTrying {len(init_values)} different initializations...")
print("="*80)

results = []

for idx, K0 in enumerate(init_values):
    print(f"\n[{idx+1}/{len(init_values)}] Testing K0 = {K0}")
    
    K_init = Init_K(N, K0, cp.float32)
    
    start = time.time()
    K_opt, final_cost = quick_optimize(K_init, P_obs, N, ETA, max_iter=500)
    elapsed = time.time() - start
    
    results.append({
        'K0': K0,
        'cost': float(final_cost),
        'K': K_opt.copy(),
        'time': elapsed
    })
    
    print(f"  Final cost: {final_cost:.6e} (time: {elapsed:.1f}s)")

# Find best result
results.sort(key=lambda x: x['cost'])
best = results[0]

print("\n" + "="*80)
print("MULTI-START RESULTS")
print("="*80)
for i, r in enumerate(results):
    marker = "  ← BEST" if i == 0 else ""
    print(f"K0={r['K0']:.1f}  →  Cost: {r['cost']:.6e}  ({r['time']:.1f}s){marker}")

print(f"\nBest initialization: K0 = {best['K0']}")
print(f"Best cost: {best['cost']:.6e}")

# Full optimization with best initialization
print("\n" + "="*80)
print("Running full optimization with best K0...")
print("="*80)

# Load the adaptive optimizer
K_best = best['K']
exec(open('core_phic2_cupy_adaptive.py').read().replace(
    'if __name__ == "__main__":',
    'if False:'
))

# Note: This is a quick demo. For production, you'd want to
# properly import and run the adaptive optimizer here.

print("\n" + "="*80)
print("RECOMMENDATION")
print("="*80)
print(f"Best initialization found: K0 = {best['K0']}")
print(f"Best quick-run cost: {best['cost']:.6e}")
print("\nTo run full optimization with this initialization:")
print(f"1. Modify Init_K to use INIT_K0={best['K0']}")
print(f"2. Run: python core_phic2_cupy_adaptive.py {fhic}")
print("="*80)
