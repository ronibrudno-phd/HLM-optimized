import os
import sys
import time
import warnings
import cupy as cp
import numpy as np
from scipy.stats import pearsonr
cp.set_printoptions(precision=3, linewidth=200)
warnings.filterwarnings('ignore')

# Fixed version of PHi-C2 for large-scale Hi-C matrices
# Based on original core_phic2_cupy.py with stability improvements
# Key fixes:
# 1. Adaptive learning rate based on matrix size
# 2. Regularization for numerical stability
# 3. K bounds to prevent negative/extreme values
# 4. Better progress monitoring

if not len(sys.argv) >= 2:
    print("usage:: python core_phic2_cupy_fixed.py normalized-HiC-Contact-Matrix [--eta=AUTO]")
    print("  --eta=AUTO: Automatic learning rate (recommended)")
    print("  --eta=1e-6: Manual learning rate")
    sys.exit()

fhic = str(sys.argv[1])

# Parse learning rate
ETA_MANUAL = None
for arg in sys.argv:
    if arg.startswith('--eta=') and arg != '--eta=AUTO':
        ETA_MANUAL = float(arg.split('=')[1])

print("="*80)
print("PHi-C2 for Large-Scale Hi-C Matrices")
print("="*80)

# Initialize k_ij
def Init_K(K, N, INIT_K0):
    for i in range(1, N):
        j = i-1
        K[i,j] = K[j,i] = INIT_K0
    return K

# K2P with numerical stability improvements
def K2P(K, N, regularization=1e-10):
    """
    Convert spring constant matrix to contact probability
    With added numerical stability
    """
    d = cp.sum(K, axis=0)
    L = cp.diag(d) - K
    
    # Add regularization for stability
    L_sub = L[1:N, 1:N] + regularization * cp.eye(N-1, dtype=K.dtype)
    
    try:
        Q = cp.linalg.inv(L_sub)
    except:
        # Fallback to pseudoinverse if singular
        print("  Warning: Using pseudoinverse")
        Q = cp.linalg.pinv(L_sub)
    
    M = 0.5*(Q + cp.transpose(Q))
    A = cp.diag(M)
    
    G = cp.zeros((N, N), dtype=K.dtype)
    G[1:N, 1:N] = -2*M + A + cp.reshape(A, (-1,1))
    G[0, 1:N] = A
    G[1:N, 0] = A
    
    # Clip G to prevent numerical issues
    G = cp.clip(G, -0.33, 1e6)
    
    # Compute P with numerical safety
    P = cp.power(1.0 + 3.0*G, -1.5)
    P = cp.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)
    
    return P

def Pdif2cost(P_dif, N):
    cost = cp.sqrt(cp.sum(P_dif**2)) / N
    return cost

def enforce_K_constraints(K):
    """Keep K physically reasonable"""
    # Non-negative
    K = cp.maximum(K, 0)
    # Symmetric
    K = 0.5 * (K + K.T)
    # Bounded (prevent extreme values)
    K = cp.minimum(K, 1e3)
    return K

def estimate_eta(P_obs, K_init, N):
    """
    Estimate good learning rate based on initial gradient
    """
    print("\nEstimating optimal learning rate...")
    
    # Compute initial gradient magnitude
    P_init = K2P(K_init, N)
    P_dif = P_init - P_obs
    grad_norm = cp.sqrt(cp.mean(P_dif**2))
    
    print(f"  Initial gradient norm: {grad_norm:.5e}")
    
    # Scale ETA inversely with matrix size and gradient
    # For N=2869 (1MB), ETA=1e-4 works
    # For N=28454 (100KB), scale down by factor of ~100
    base_eta = 1e-4
    size_factor = (2869.0 / N)**2  # Quadratic scaling
    gradient_factor = min(1.0, 1e-4 / float(grad_norm))  # Scale by gradient
    
    eta = base_eta * size_factor * gradient_factor
    
    # Clamp to reasonable range
    eta = np.clip(eta, 1e-8, 1e-4)
    
    print(f"  Estimated ETA: {eta:.2e}")
    print(f"    Size factor: {size_factor:.2e}")
    print(f"    Gradient factor: {gradient_factor:.2e}")
    
    return eta

def phic2(K, N, P_obs, ETA=1.0e-4, ALPHA=1.0e-10, ITERATION_MAX=1000000, verbose=True):
    stop_delta = ETA * ALPHA
    paras_fit = "%e\t%e\t%d\t" % (ETA, ALPHA, ITERATION_MAX)
    
    print("\n" + "="*80)
    print("Starting PHi-C2 Optimization")
    print("="*80)
    print(f"Learning rate (ETA): {ETA:.2e}")
    print(f"Convergence (ALPHA): {ALPHA:.2e}")
    print(f"Stop delta: {stop_delta:.2e}")
    print(f"Max iterations: {ITERATION_MAX:,}")
    print()
    
    # Initial state
    P_dif = K2P(K, N) - P_obs
    cost = Pdif2cost(P_dif, N)
    
    c_traj = cp.zeros((ITERATION_MAX+1, 2), dtype=K.dtype)
    c_traj[0,0] = cost
    c_traj[0,1] = time.time()
    
    print(f"Initial cost: {cost:.6e}\n")
    
    iteration = 1
    best_cost = cost
    best_K = K.copy()
    no_improve_count = 0
    last_print = time.time()
    
    while True:
        cost_bk = cost
        
        # Gradient descent step
        K = K - ETA*P_dif
        
        # Enforce constraints
        K = enforce_K_constraints(K)
        
        # Check for NaN in K
        if cp.any(cp.isnan(K)):
            print(f"\n⚠ NaN detected in K at iteration {iteration}")
            print(f"  Reverting to best K")
            K = best_K
            break
        
        # Compute new state
        try:
            P_dif = K2P(K, N) - P_obs
            cost = Pdif2cost(P_dif, N)
        except:
            print(f"\n⚠ Error in K2P at iteration {iteration}")
            print(f"  Reverting to best K")
            K = best_K
            break
        
        # Check for NaN in cost
        if cp.isnan(cost):
            print(f"\n⚠ NaN cost at iteration {iteration}")
            print(f"  Reverting to best K")
            K = best_K
            cost = best_cost
            break
        
        c_traj[iteration,0] = cost
        c_traj[iteration,1] = time.time()
        
        cost_dif = cost_bk - cost
        
        # Track best
        if cost < best_cost:
            best_cost = cost
            best_K = K.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        # Progress reporting
        if verbose and (iteration == 1 or time.time() - last_print > 10):
            elapsed = c_traj[iteration,1] - c_traj[0,1]
            rate = iteration / elapsed if elapsed > 0 else 0
            
            print(f"Iter {iteration:7d} | Cost: {cost:.6e} | ΔCost: {cost_dif:+.6e} | "
                  f"Best: {best_cost:.6e} | Rate: {rate:.2f} it/s | Time: {elapsed/60:.1f}m")
            
            # Memory status
            gpu_mem = cp.get_default_memory_pool().used_bytes() / 1024**3
            print(f"  GPU memory: {gpu_mem:.2f}GB")
            
            last_print = time.time()
        
        # Convergence checks
        if 0 < cost_dif < stop_delta:
            print(f"\n✓ Converged! Cost change ({cost_dif:.2e}) < threshold ({stop_delta:.2e})")
            break
        
        if iteration >= ITERATION_MAX:
            print(f"\n⚠ Reached maximum iterations ({ITERATION_MAX:,})")
            break
        
        if no_improve_count > 10000:
            print(f"\n⚠ No improvement for 10000 iterations")
            print(f"  Current: {cost:.6e}, Best: {best_cost:.6e}")
            # Use best K
            K = best_K
            break
        
        iteration += 1
        
        # Periodic cleanup
        if iteration % 100 == 0:
            cp.get_default_memory_pool().free_all_blocks()
    
    # Use best K
    if best_cost < cost:
        print(f"\nUsing best K (cost={best_cost:.6e})")
        K = best_K
    
    c_traj = c_traj[:iteration+1]
    
    print(f"\nOptimization summary:")
    print(f"  Total iterations: {iteration:,}")
    print(f"  Final cost: {best_cost:.6e}")
    print(f"  Initial cost: {float(c_traj[0,0]):.6e}")
    print(f"  Improvement: {float(c_traj[0,0]) - best_cost:.6e}")
    
    return [K, c_traj, paras_fit]

# Save functions (unchanged from original)
def saveLg(fn, xy, ct):
    fw = open(fn, 'w')
    fw.write(ct)
    n = len(xy)
    m = np.shape(xy)[1] if xy.ndim==2 else 0
    for i in range(n):
        if m == 0:
            lt = "%11s "%('NaN') if np.isnan(xy[i]) else "%11.5e"%(xy[i])
        else:
            lt = ''
            for v in xy[i]:
                lt += "%11s "%('NaN') if np.isnan(v) else "%11.5e "%(v)
        fw.write(lt+'\n')
    fw.close()

def saveMx(fn, xy, ct, chunk_size=5000):
    print(f"Saving {fn}...")
    fw = open(fn, 'w')
    fw.write(ct)
    n = len(xy)
    for chunk_start in range(0, n, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n)
        for i in range(chunk_start, chunk_end):
            lt = ''
            for v in xy[i]:
                lt += "%11s "%('NaN') if np.isnan(v) else "%11.5e "%(v)
            fw.write(lt+'\n')
        if chunk_start % 5000 == 0 and chunk_start > 0:
            print(f"  {chunk_start}/{n} rows...")
    fw.close()
    print("  Done!")

if __name__ == "__main__":
    # Read Hi-C
    if not os.path.isfile(fhic):
        print('Cannot find '+fhic)
        sys.exit()
    
    print(f"\nLoading Hi-C matrix: {fhic}")
    start_load = time.time()
    
    P_obs = []
    line_count = 0
    with open(fhic) as fr:
        for line in fr:
            if not line[0] == '#':
                lt = line.strip()
                lt = lt.split()
                P_obs.append(list(map(float, lt)))
                line_count += 1
                if line_count % 5000 == 0:
                    print(f"  {line_count} rows loaded...")
    
    P_obs = cp.array(P_obs, dtype=cp.float32)  # Use float32 for large matrices
    N = len(P_obs)
    
    load_time = time.time() - start_load
    print(f"\nLoaded {N}×{N} matrix in {load_time:.1f}s")
    
    # Preprocessing
    cp.nan_to_num(P_obs, copy=False)
    P_obs = P_obs + cp.eye(N, dtype=cp.float32)
    
    # Matrix statistics
    print(f"\nMatrix statistics:")
    print(f"  Size: {N}×{N}")
    print(f"  Non-zero: {int(cp.count_nonzero(P_obs)):,}")
    print(f"  Range: [{float(cp.min(P_obs)):.2e}, {float(cp.max(P_obs)):.2e}]")
    print(f"  Mean: {float(cp.mean(P_obs)):.2e}")
    
    # Memory usage
    matrix_gb = N * N * 4 / 1024**3  # float32
    print(f"  Memory: {matrix_gb:.2f}GB")
    
    # Initialize K
    print("\nInitializing spring constants...")
    K_fit = cp.zeros((N, N), dtype=cp.float32)
    K_fit = Init_K(K_fit, N, INIT_K0=0.5)
    
    # Determine learning rate
    if ETA_MANUAL is None:
        ETA = estimate_eta(P_obs, K_fit, N)
    else:
        ETA = ETA_MANUAL
        print(f"\nUsing manual ETA: {ETA:.2e}")
    
    # Run optimization
    phic2_alpha = 1.0e-10
    
    start_opt = time.time()
    K_fit, c_traj, paras_fit = phic2(
        K_fit, N, P_obs,
        ETA=ETA,
        ALPHA=phic2_alpha,
        ITERATION_MAX=1000000,
        verbose=True
    )
    opt_time = time.time() - start_opt
    
    print(f"\nTotal optimization time: {opt_time/60:.1f} minutes ({opt_time/3600:.1f} hours)")
    
    # Compute final P
    print("\nComputing final contact probabilities...")
    P_fit = K2P(K_fit, N)
    
    # Transfer to CPU
    print("Transferring results to CPU...")
    c_traj = cp.asnumpy(c_traj)
    K_fit = cp.asnumpy(K_fit)
    P_fit = cp.asnumpy(P_fit)
    P_obs = cp.asnumpy(P_obs)
    
    # Save results
    print("\n" + "="*80)
    print("Saving Results")
    print("="*80)
    
    dataDir = fhic[:fhic.rfind('.')] + "_phic2_a%7.1e_cupy_fixed"%(phic2_alpha)
    os.makedirs(dataDir, exist_ok=True)
    fo = "%s/N%d"%(dataDir, N)
    
    c_traj[:,1] = c_traj[:,1] - c_traj[0,1]
    saveLg(fo+'.log', c_traj, "#%s\n#cost systemTime\n"%(paras_fit))
    
    saveMx(fo+'.K_fit', K_fit,
           "#K_fit N %d min: %11.5e max: %11.5e\n"%(N, np.min(K_fit), np.max(K_fit)))
    
    # Compute Pearson correlation
    print("\nComputing Pearson correlation...")
    triMask = np.where(np.triu(np.ones((N,N)),1)>0)
    pijMask = np.where(np.triu(P_obs,1)>0)
    
    p1 = pearsonr(P_fit[triMask], P_obs[triMask])[0]
    p2 = pearsonr(P_fit[pijMask], P_obs[pijMask])[0]
    
    ct = "#P_fit N %d min: %11.5e max: %11.5e pearson: %11.5e %11.5e\n"%\
         (N, np.nanmin(P_fit), np.nanmax(P_fit), p1, p2)
    saveMx(fo+'.P_fit', P_fit, ct)
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"Matrix size:        {N}×{N}")
    print(f"Learning rate:      {ETA:.2e}")
    print(f"Iterations:         {len(c_traj):,}")
    print(f"Optimization time:  {opt_time/60:.1f} min ({opt_time/3600:.1f} hours)")
    print(f"Initial cost:       {c_traj[0,0]:.6e}")
    print(f"Final cost:         {c_traj[-1,0]:.6e}")
    print(f"Pearson (all):      {p1:.5f}")
    print(f"Pearson (nonzero):  {p2:.5f}")
    print(f"Output directory:   {dataDir}/")
    print("="*80)
