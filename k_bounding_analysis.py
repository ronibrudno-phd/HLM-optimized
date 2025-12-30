import cupy as cp
import numpy as np

def constrain_K_adaptive(K, iteration, K_bound=None, adaptation_factor=1.5):
    """
    Apply constraints with adaptive bounds
    
    Args:
        K: Spring constant matrix
        iteration: Current iteration number
        K_bound: Current bound (None = auto-detect)
        adaptation_factor: How much to scale bound when exceeded
    
    Returns:
        K_constrained, new_K_bound
    """
    # Symmetric (ALWAYS required)
    K = 0.5 * (K + K.T)
    
    # Adaptive bounding
    if K_bound is None:
        # Initial bound based on matrix size
        N = K.shape[0]
        if N < 3000:
            K_bound = 1000.0  # For ~1MB resolution
        elif N < 10000:
            K_bound = 500.0   # For ~500KB resolution
        else:
            K_bound = 200.0   # For ~100KB resolution
    
    # Check current K range
    K_min = float(cp.min(K))
    K_max = float(cp.max(K))
    K_absmax = max(abs(K_min), abs(K_max))
    
    # If K exceeds bound, expand bound (but log warning)
    if K_absmax > K_bound:
        old_bound = K_bound
        K_bound = K_absmax * adaptation_factor
        
        if iteration % 100 == 0:  # Don't spam
            print(f"  Warning: K exceeded bound at iter {iteration}")
            print(f"    K range: [{K_min:.2e}, {K_max:.2e}]")
            print(f"    Expanding bound: {old_bound:.1f} → {K_bound:.1f}")
    
    # Apply bounds
    K = cp.clip(K, -K_bound, K_bound)
    
    return K, K_bound


def constrain_K_percentile(K, percentile=99.9):
    """
    Clip based on percentile of |K| values
    More robust than fixed bounds
    
    Args:
        K: Spring constant matrix
        percentile: Percentile to use as bound (e.g., 99.9)
    
    Returns:
        K_constrained
    """
    # Symmetric
    K = 0.5 * (K + K.T)
    
    # Compute bound from percentile of absolute values
    K_abs = cp.abs(K)
    K_bound = float(cp.percentile(K_abs, percentile))
    
    # Ensure minimum bound
    K_bound = max(K_bound, 10.0)
    
    # Apply
    K = cp.clip(K, -K_bound, K_bound)
    
    return K


def constrain_K_statistical(K, n_sigma=5.0):
    """
    Clip outliers based on statistics
    Assumes most K values are reasonable, clips extreme outliers
    
    Args:
        K: Spring constant matrix
        n_sigma: Number of standard deviations for outlier detection
    
    Returns:
        K_constrained
    """
    # Symmetric
    K = 0.5 * (K + K.T)
    
    # Compute statistics
    K_mean = float(cp.mean(K))
    K_std = float(cp.std(K))
    
    # Bound based on statistics
    K_bound = abs(K_mean) + n_sigma * K_std
    K_bound = max(K_bound, 100.0)  # Minimum reasonable bound
    
    # Apply symmetric bounds
    K = cp.clip(K, K_mean - K_bound, K_mean + K_bound)
    
    return K


# Example usage patterns:

def example_fixed_bound():
    """Original approach - simple but may be too restrictive or too loose"""
    K = cp.random.randn(1000, 1000).astype(cp.float32)
    
    # Fixed bound
    K = 0.5 * (K + K.T)
    K = cp.clip(K, -1000, 1000)
    
    return K


def example_adaptive_bound():
    """Adaptive approach - adjusts during optimization"""
    K = Init_K(1000, 0.5, cp.float32)
    K_bound = None
    
    for iteration in range(10000):
        # ... gradient step ...
        K = K - eta * gradient
        
        # Adaptive constraint
        K, K_bound = constrain_K_adaptive(K, iteration, K_bound)
        
        # K_bound automatically grows if needed
    
    return K


def example_percentile_bound():
    """Percentile approach - based on actual K distribution"""
    K = cp.random.randn(1000, 1000).astype(cp.float32)
    
    # Clip based on 99.9th percentile
    # This keeps 99.9% of K values, only clips extreme outliers
    K = constrain_K_percentile(K, percentile=99.9)
    
    return K


def example_statistical_bound():
    """Statistical approach - based on mean and std"""
    K = cp.random.randn(1000, 1000).astype(cp.float32)
    
    # Clip values beyond mean ± 5σ
    K = constrain_K_statistical(K, n_sigma=5.0)
    
    return K


# Comparison of approaches:

def compare_approaches():
    """
    Compare different bounding strategies
    """
    # Simulated K matrix with some outliers
    N = 1000
    K = cp.random.randn(N, N).astype(cp.float32) * 10
    
    # Add some extreme outliers
    K[100, 200] = 5000
    K[200, 100] = 5000
    K[300, 400] = -3000
    K[400, 300] = -3000
    
    print("Original K statistics:")
    print(f"  Range: [{float(cp.min(K)):.1f}, {float(cp.max(K)):.1f}]")
    print(f"  Mean: {float(cp.mean(K)):.2f}, Std: {float(cp.std(K)):.2f}")
    print(f"  99.9 percentile of |K|: {float(cp.percentile(cp.abs(K), 99.9)):.1f}")
    print()
    
    # Method 1: Fixed bound
    K1 = cp.clip(K.copy(), -1000, 1000)
    print("Fixed bound (±1000):")
    print(f"  Range: [{float(cp.min(K1)):.1f}, {float(cp.max(K1)):.1f}]")
    print(f"  Clipped: {int(cp.sum((K != K1).astype(int)))} values")
    print()
    
    # Method 2: Percentile
    K2 = constrain_K_percentile(K.copy(), percentile=99.9)
    print("Percentile-based (99.9%):")
    print(f"  Range: [{float(cp.min(K2)):.1f}, {float(cp.max(K2)):.1f}]")
    print(f"  Clipped: {int(cp.sum((K != K2).astype(int)))} values")
    print()
    
    # Method 3: Statistical
    K3 = constrain_K_statistical(K.copy(), n_sigma=5.0)
    print("Statistical (mean ± 5σ):")
    print(f"  Range: [{float(cp.min(K3)):.1f}, {float(cp.max(K3)):.1f}]")
    print(f"  Clipped: {int(cp.sum((K != K3).astype(int)))} values")


if __name__ == "__main__":
    print("="*80)
    print("K Bounding Strategies Comparison")
    print("="*80)
    print()
    
    compare_approaches()
    
    print()
    print("="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    print()
    print("For your 100KB resolution data (N=28454):")
    print()
    print("Option 1: ADAPTIVE (RECOMMENDED)")
    print("  - Start with bound = 200")
    print("  - Automatically expands if K exceeds it")
    print("  - Most robust for unknown data")
    print()
    print("Option 2: PERCENTILE")
    print("  - Clip at 99.9th percentile")
    print("  - Data-driven, adapts to K distribution")
    print("  - Good if most K values are reasonable")
    print()
    print("Option 3: FIXED")
    print("  - Use ±1000 (based on paper)")
    print("  - Simple, works if data similar to paper")
    print("  - May be too loose for fine resolution")
    print()
    print("For production: Use ADAPTIVE with initial bound = 200")
    print("  (finer resolution → smaller typical K → smaller bound needed)")
    print("="*80)
