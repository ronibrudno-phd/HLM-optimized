import os
import sys
import numpy as np
import cupy as cp

def estimate_memory_requirements(fhic):
    """
    Analyze a Hi-C matrix and estimate GPU memory requirements
    """
    print("="*70)
    print("GPU MEMORY REQUIREMENT ESTIMATOR")
    print("="*70)
    
    if not os.path.isfile(fhic):
        print(f'Cannot find {fhic}')
        sys.exit(1)
    
    # Read matrix size
    print(f"\nReading matrix from: {fhic}")
    try:
        P_obs = np.loadtxt(fhic, comments='#', dtype=np.float64)
    except:
        P_obs = []
        with open(fhic) as fr:
            for line in fr:
                if not line[0] == '#':
                    lt = line.strip().split()
                    P_obs.append(list(map(float, lt)))
        P_obs = np.array(P_obs, dtype=np.float64)
    
    N = len(P_obs)
    del P_obs
    
    print(f"Matrix dimensions: {N} x {N}")
    print(f"Total elements: {N*N:,}")
    
    # Memory calculations
    print("\n" + "="*70)
    print("MEMORY REQUIREMENTS (per matrix)")
    print("="*70)
    
    float64_per_matrix = N * N * 8 / 1024**3
    float32_per_matrix = N * N * 4 / 1024**3
    
    print(f"float64 (double precision): {float64_per_matrix:.2f} GB")
    print(f"float32 (single precision): {float32_per_matrix:.2f} GB")
    
    print("\n" + "="*70)
    print("ESTIMATED PEAK GPU MEMORY USAGE")
    print("="*70)
    
    # Original script needs:
    # - K (float64): N x N
    # - P_obs (float64): N x N  
    # - P_calc (float64): N x N (in K2P)
    # - G_buffer (float64): N x N (in K2P)
    # - Identity (float64): (N-1) x (N-1)
    # - L (float64): N x N (in K2P)
    # - Q (float64): (N-1) x (N-1)
    # - velocity (float64): N x N
    # - P_dif (float64): N x N
    
    original_peak = float64_per_matrix * 8  # Conservative estimate: 8 full matrices
    print(f"Original script:          ~{original_peak:.2f} GB")
    print(f"  (assumes ~8 simultaneous NxN float64 matrices)")
    
    # Memory-optimized version
    memory_opt_peak = float64_per_matrix * 4  # 4 full matrices
    print(f"Memory-optimized script:  ~{memory_opt_peak:.2f} GB")
    print(f"  (assumes ~4 simultaneous NxN float64 matrices)")
    
    # Ultra-optimized version (uses float32 for intermediates)
    ultra_opt_peak = float64_per_matrix * 2 + float32_per_matrix * 2  # 2 float64 + 2 float32
    print(f"Ultra-optimized script:   ~{ultra_opt_peak:.2f} GB")
    print(f"  (uses float32 for intermediate calculations)")
    
    print("\n" + "="*70)
    print("GPU INFORMATION")
    print("="*70)
    
    # Check GPU
    try:
        gpu_mem_total = cp.cuda.Device().mem_info[1] / 1024**3
        gpu_mem_free = cp.cuda.Device().mem_info[0] / 1024**3
        gpu_mem_used = gpu_mem_total - gpu_mem_free
        
        print(f"GPU Total Memory:  {gpu_mem_total:.2f} GB")
        print(f"GPU Free Memory:   {gpu_mem_free:.2f} GB")
        print(f"GPU Used Memory:   {gpu_mem_used:.2f} GB")
        
        print("\n" + "="*70)
        print("RECOMMENDATIONS")
        print("="*70)
        
        if original_peak < gpu_mem_free * 0.8:
            print("✓ Original script should work fine")
            print("  Recommended: core_phic2_cupy_optimized.py")
        elif memory_opt_peak < gpu_mem_free * 0.8:
            print("⚠ Original script may run out of memory")
            print("  Recommended: core_phic2_cupy_memory_optimized.py")
        elif ultra_opt_peak < gpu_mem_free * 0.8:
            print("⚠ Need aggressive memory optimization")
            print("  Recommended: core_phic2_cupy_ultra_optimized.py")
        else:
            print("✗ Matrix too large for single GPU")
            print("  Options:")
            print("  1. Use a GPU with more memory")
            print("  2. Reduce matrix resolution (bin/downsample)")
            print("  3. Use CPU-based optimization (much slower)")
            print("  4. Split optimization into chunks")
        
        print("\n" + "="*70)
        print("OPTIMIZATION STRATEGIES")
        print("="*70)
        
        if N > 50000:
            print("⚠ Very large matrix detected (N > 50,000)")
            print("  Consider:")
            print("  - Reducing checkpoint frequency (saves memory)")
            print("  - Using lower initial learning rate")
            print("  - Monitoring memory usage during run")
        
        if float64_per_matrix > 20:
            print("⚠ Each matrix requires > 20 GB")
            print("  Strongly recommend:")
            print("  - Use ultra-optimized version with float32")
            print("  - Set checkpoint_interval to 10000 or higher")
            print("  - Monitor 'nvidia-smi' during execution")
        
    except Exception as e:
        print(f"Could not access GPU: {e}")
        print("Make sure CUDA is properly installed")
    
    print("\n" + "="*70)
    print("USAGE EXAMPLES")
    print("="*70)
    print("\nFor your matrix:")
    print(f"  python core_phic2_cupy_memory_optimized.py {fhic}")
    print("or for ultra memory savings:")
    print(f"  python core_phic2_cupy_ultra_optimized.py {fhic}")
    print("="*70)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python diagnose_memory.py <hic-matrix-file>")
        print("\nThis script analyzes your Hi-C matrix and estimates")
        print("GPU memory requirements for the optimization.")
        sys.exit(1)
    
    fhic = sys.argv[1]
    estimate_memory_requirements(fhic)
