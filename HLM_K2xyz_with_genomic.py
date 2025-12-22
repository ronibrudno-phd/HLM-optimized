#!/usr/bin/env python3
"""
Generate 3D structures from K-matrix using HLM eigenvalue sampling method.
Saves XYZ coordinates WITH genomic location information.

Based on HLM-Genome's core_K2xyz-shape.py but modified to:
1. Actually save the XYZ coordinates (uncommented)
2. Include genomic location mapping
3. Output in simple text format

Usage:
    python HLM_K2xyz_with_genomic.py K_fit.txt bin_info.txt 100

Where bin_info.txt has format:
    # chr  start  end
    chr1  0  500000
    chr1  500000  1000000
    ...

Output:
    - structures/structure_0001.xyz (with genomic info in header)
    - structures/structure_0001_coords.txt (bead# chr start end x y z)
"""

import os
import sys
import numpy as np
from pathlib import Path

def read_K_matrix(fk):
    """Read K-matrix from file"""
    if not os.path.isfile(fk):
        print(f'Cannot find {fk}')
        sys.exit(1)
    
    K_fit = []
    with open(fk) as fr:
        for line in fr:
            if not line[0] == '#':
                lt = line.strip().split()
                K_fit.append(list(map(float, lt)))
    
    K_fit = np.array(K_fit)
    return K_fit

def read_bin_info(fbin):
    """
    Read genomic bin information.
    Format can be:
      - filtered_index  chr  start  end  [original_index]
      - chr  start  end
    """
    if not os.path.isfile(fbin):
        print(f'Cannot find {fbin}')
        sys.exit(1)
    
    bins = []
    with open(fbin) as f:
        for line in f:
            if not line.startswith('#'):
                parts = line.strip().split()
                if len(parts) >= 4:
                    # Format: filtered_index chr start end [original_index]
                    # Skip the filtered_index (first column)
                    chr_name = parts[1]
                    start = int(parts[2])
                    end = int(parts[3])
                    bins.append((chr_name, start, end))
                elif len(parts) >= 3:
                    # Format: chr start end
                    chr_name = parts[0]
                    start = int(parts[1])
                    end = int(parts[2])
                    bins.append((chr_name, start, end))
    
    return bins

def generate_structure(K_fit, seed=None):
    """
    Generate single structure using HLM eigenvalue sampling.
    
    This is the HLM-Genome method:
    1. Compute Laplacian L = diag(d) - K
    2. Eigendecompose: L = Q * Lambda * Q^T
    3. Sample X_i ~ N(0, 1/lambda_i) for i > 0
    4. Transform: R = Q * X
    """
    if seed is not None:
        np.random.seed(seed)
    
    Ng = len(K_fit)
    
    # K to Laplacian matrix
    d = np.sum(K_fit, axis=0) + np.diag(K_fit)
    Lap = np.diag(d) - K_fit
    
    # Eigenvalues and eigenvectors
    lam, Qs = np.linalg.eigh(Lap)
    
    # Collective coordinates X
    X = np.zeros((Ng, 3))
    for k in range(3):
        # Sample from equilibrium distribution
        # Use 1/lambda (not 1/3/lambda) as in HLM-Genome
        X[1:, k] = np.sqrt(1.0 / lam[1:]) * np.random.randn(Ng-1)
    
    # X -> 3D coordinates R
    xyz = np.zeros((Ng, 3))
    for k in range(3):
        xyz[:, k] = np.dot(Qs, X[:, k])
    
    # Center (necessary when diag(K) != 0)
    xyz = xyz - np.mean(xyz, axis=0)
    
    return xyz, lam

def save_xyz_with_genomic(xyz, bins, output_file, structure_id):
    """
    Save XYZ with genomic information.
    
    Format:
    N
    Structure #X: chr_start-chr_end
    bead_0 chr start end x y z
    bead_1 chr start end x y z
    ...
    """
    N = len(xyz)
    
    with open(output_file, 'w') as f:
        # Header
        f.write(f"{N}\n")
        
        # Comment line with genomic span
        if bins:
            first_chr, first_start, _ = bins[0]
            last_chr, _, last_end = bins[-1]
            f.write(f"Structure {structure_id}: {first_chr}:{first_start}-{last_chr}:{last_end}\n")
        else:
            f.write(f"Structure {structure_id}\n")
        
        # Coordinates with genomic info
        for i in range(N):
            if bins and i < len(bins):
                chr_name, start, end = bins[i]
                f.write(f"{i} {chr_name} {start} {end} {xyz[i,0]:.6f} {xyz[i,1]:.6f} {xyz[i,2]:.6f}\n")
            else:
                # No genomic info available
                f.write(f"{i} NA 0 0 {xyz[i,0]:.6f} {xyz[i,1]:.6f} {xyz[i,2]:.6f}\n")

def save_simple_xyz(xyz, output_file, structure_id):
    """Save in standard XYZ format (for visualization software)"""
    N = len(xyz)
    
    with open(output_file, 'w') as f:
        f.write(f"{N}\n")
        f.write(f"HLM structure {structure_id}\n")
        for i in range(N):
            f.write(f"C {xyz[i,0]:.6f} {xyz[i,1]:.6f} {xyz[i,2]:.6f}\n")

def compute_shape_stats(xyz):
    """Compute asphericity and shape factor"""
    Q = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            Q[i, j] = np.mean(xyz[:, i] * xyz[:, j])
    
    trQ = np.trace(Q)
    QH = Q - trQ/3.0 * np.eye(3)
    
    asp = 1.5 * np.trace(np.dot(QH, QH)) / trQ
    spf = 27. * np.linalg.det(QH) / (trQ**1.5)
    
    return asp, spf

def main():
    if len(sys.argv) < 3:
        print('Usage: python HLM_K2xyz_with_genomic.py K_fit.txt bin_info.txt [NSamples]')
        print('\nExample bin_info.txt format:')
        print('# chr  start  end')
        print('chr1  0  500000')
        print('chr1  500000  1000000')
        print('...')
        print('\nOr use "none" if no genomic info available')
        sys.exit(1)
    
    fk = str(sys.argv[1])
    fbin = str(sys.argv[2])
    ncfgs = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    
    print("="*60)
    print("HLM Structure Generator with Genomic Mapping")
    print("="*60)
    
    # Read K-matrix
    print(f"\nReading K-matrix: {fk}")
    K_fit = read_K_matrix(fk)
    Ng = len(K_fit)
    print(f"  K-matrix size: {Ng} x {Ng}")
    print(f"  K range: [{np.min(K_fit):.6e}, {np.max(K_fit):.6e}]")
    
    # Read genomic bins
    bins = None
    if fbin.lower() != 'none':
        print(f"\nReading genomic bins: {fbin}")
        bins = read_bin_info(fbin)
        print(f"  Number of bins: {len(bins)}")
        
        if len(bins) != Ng:
            print(f"\n⚠️  WARNING: K-matrix size ({Ng}) != bins ({len(bins)})")
            print(f"  Will use available bins, rest will be marked as NA")
    else:
        print("\n⚠️  No genomic information provided")
    
    # Create output directory
    output_dir = Path(fk).parent / f"{Path(fk).stem}_structures"
    output_dir.mkdir(exist_ok=True)
    print(f"\nOutput directory: {output_dir}/")
    print(f"Generating {ncfgs} structures...")
    
    # Generate structures
    np.random.seed(1274)  # HLM-Genome default seed
    
    shape_stats = []
    for c in range(ncfgs):
        # Generate structure
        xyz, eigenvalues = generate_structure(K_fit)
        
        # Compute shape
        asp, spf = compute_shape_stats(xyz)
        shape_stats.append((asp, spf))
        
        # Save with genomic info
        coords_file = output_dir / f"structure_{c+1:04d}_genomic.txt"
        save_xyz_with_genomic(xyz, bins, coords_file, c+1)
        
        # Save standard XYZ (for visualization)
        xyz_file = output_dir / f"structure_{c+1:04d}.xyz"
        save_simple_xyz(xyz, xyz_file, c+1)
        
        if (c+1) % max(1, ncfgs//10) == 0:
            print(f"  Generated {c+1}/{ncfgs} (asp={asp:.4f}, spf={spf:+.4f})")
    
    print(f"\n✓ Saved {ncfgs} structures to {output_dir}/")
    
    # Summary statistics
    asps = [s[0] for s in shape_stats]
    spfs = [s[1] for s in shape_stats]
    
    print("\n" + "="*60)
    print("SHAPE STATISTICS")
    print("="*60)
    print(f"Asphericity: {np.mean(asps):.6f} ± {np.std(asps):.6f}")
    print(f"Shape factor: {np.mean(spfs):+.6f} ± {np.std(spfs):.6f}")
    print("="*60)
    
    # Save summary
    summary_file = output_dir / "summary.txt"
    with open(summary_file, 'w') as f:
        f.write(f"# K-matrix: {fk}\n")
        f.write(f"# Genomic bins: {fbin}\n")
        f.write(f"# Structures: {ncfgs}\n")
        f.write(f"# Matrix size: {Ng}\n")
        f.write(f"#\n")
        f.write(f"# Structure  Asphericity  ShapeFactor\n")
        for i, (asp, spf) in enumerate(shape_stats, 1):
            f.write(f"{i:5d}  {asp:11.6f}  {spf:+12.6f}\n")
    
    print(f"\n✓ Saved summary: {summary_file}")
    
    print("\n" + "="*60)
    print("OUTPUT FILES:")
    print("="*60)
    print(f"1. structure_XXXX.xyz          - Standard XYZ (for visualization)")
    print(f"2. structure_XXXX_genomic.txt  - With chr, start, end, x, y, z")
    print(f"3. summary.txt                 - Shape statistics")
    print("="*60)

if __name__ == '__main__':
    main()