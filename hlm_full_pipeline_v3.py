#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HLM end-to-end pipeline v3
(K_fit + FILTERED bins -> HDF5 -> GC -> Sampled distances -> Analytical radial -> TSV)

This script combines:
  1) Structure sampling:
     - Sample 3D structures from K_fit using Laplacian eigendecomposition.
     - Regularize small eigenvalues (threshold set per-species by expression_hic_validation_v3.py).
     - Centers each structure at its bead-cloud centroid.

  2) GC content:
     - Compute GC per bin using `bedtools nuc`.

  3) Sampled distances (kept for morphology analysis):
     - Compute mean squared distance ⟨r²⟩_sampled per bin from structure ensemble.
     - Used for nuclear shape / asphericity / cell-to-cell variation analysis.
     - NOTE: these are SQUARED distances, not Euclidean. Column 'avg_r2' makes this explicit.

  4) Analytical radial position (NEW — the primary radial metric):
     - Computes σ_ii analytically from the Laplacian eigendecomposition via:
           σ_ii = Σ_{k=1}^{N-1}  Q[i,k]² / λ_k^reg
       This is Equation 8 of Shi et al. 2024 (Biophys J 123:2574).
     - Normalises to q_rad(i) = σ_ii / mean_j(σ_jj).
       q_rad < 1 → bin i is at the nuclear core; q_rad > 1 → periphery.
     - NOT affected by the chromosome-territory centroid-pulling artifact of the sampled
       approach, because the mathematical center of mass is fixed by the probability
       distribution of the Gaussian chain, not by any finite sample of structures.
     - Exact (not approximate): converges to the same value as infinite structure sampling.
     - Stored as 'sigma_ii', 'q_rad', and 'analytical_radial_bins' in the HDF5 file.

Key HDF5 datasets
-----------------
  chr, start, end          -- per bin, length N
  xyz_all                  -- (N, M, 3) float32 sampled structures
  gc_content               -- (N,) float32 GC fraction per bin
  avg_r2                   -- (N,) float32 mean SQUARED distance from centroid (sampled)
  sampled_radial_bins      -- (N,) int16  quantile bins based on avg_r2
  sigma_ii                 -- (N,) float64 analytical covariance diagonal (Eq. 8 Shi 2024)
  q_rad                    -- (N,) float64 normalised radial position (q_rad=1 → average)
  analytical_radial_bins   -- (N,) int16  quantile bins based on q_rad (PREFERRED)

Key TSV columns
---------------
  chr, start, end, gc_content,
  avg_r2, sampled_bin,      -- from structure sampling (useful for morphology)
  q_rad, analytical_bin     -- from analytical calculation (preferred for radial biology)

Threshold selection
-------------------
The min_eigenvalue threshold should be determined by expression_hic_validation_v3.py,
which tests multiple thresholds and selects the one that maximises:
  - Hi-C contact fidelity (hic_spearman)
  - Expression recovery at Hi-C resolution (expr_hic_recovery)
  - Analytical radial pattern (qrad_AB_ratio < 1 for conventional nuclei)
Pass that threshold via --min-eigenvalue (k2h5 / analytical-radial / run-all subcommands).

Filtered bins
-------------
Your bins file must correspond 1:1 to rows/cols of K_fit.
Supported formats:
  A) filtered_index  chr  start  end  [original_index]
  B) chr  start  end

Dependencies
------------
Python: numpy, h5py, pandas
Optional (for GC): bedtools available on PATH
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
import pandas as pd


# -----------------------------
# HLM: K -> structures
# -----------------------------
MIN_EIGENVALUE_THRESHOLD_DEFAULT = 1e-6  # matches your PRODUCTION script


@dataclass(frozen=True)
class BinInfo:
    chrom: str
    start: int
    end: int
    filtered_index: Optional[int] = None
    original_index: Optional[int] = None


def read_K_matrix(path: str) -> np.ndarray:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Cannot find K-matrix file: {path}")

    rows: List[List[float]] = []
    with open(path) as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            rows.append(list(map(float, line.strip().split())))
    K = np.array(rows, dtype=np.float64)

    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError(f"K must be square; got shape {K.shape}")
    return K


def read_bins_filtered(path: str) -> List[BinInfo]:
    """
    Read bins that correspond to K_fit (filtered bins).
    Supported formats:
      A) filtered_index  chr  start  end  [original_index]
      B) chr  start  end
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Cannot find bins file: {path}")

    out: List[BinInfo] = []
    with open(path) as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) >= 4:
                # Assume format A
                filt = int(parts[0])
                chrom = parts[1]
                start = int(parts[2])
                end = int(parts[3])
                orig = int(parts[4]) if len(parts) >= 5 else None
                out.append(BinInfo(chrom=chrom, start=start, end=end,
                                   filtered_index=filt, original_index=orig))
            elif len(parts) == 3:
                chrom = parts[0]
                start = int(parts[1])
                end = int(parts[2])
                out.append(BinInfo(chrom=chrom, start=start, end=end))
            else:
                raise ValueError(f"Bad bins line {ln}: {line}")

    # If filtered_index exists, ensure it is monotonic / consistent (optional sanity check)
    filt_idxs = [b.filtered_index for b in out if b.filtered_index is not None]
    if filt_idxs and (sorted(filt_idxs) != filt_idxs):
        print("WARNING: filtered_index column is not sorted; this is unusual but not fatal.", file=sys.stderr)

    return out


def precompute_eigendecomposition(K: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert K to Laplacian, then eigh(Lap). Returns |lambda| and eigenvectors Q.
    """
    d = np.sum(K, axis=0) + np.diag(K)
    Lap = np.diag(d) - K
    lam, Q = np.linalg.eigh(Lap)
    return np.abs(lam), Q


def compute_sigma_diag_global(
    lam_abs: np.ndarray,
    Q: np.ndarray,
    min_eigenvalue: float,
) -> np.ndarray:
    """
    Compute the diagonal of the genome-wide covariance matrix analytically.

    This implements Equation 8 of Shi et al. 2024 (Biophys J 123:2574-2583):

        σ_ii = Σ_{k=1}^{N-1}  Q[i,k]² / λ_k^reg

    where λ_k^reg = max(|λ_k|, min_eigenvalue).

    The result is the EXACT expected squared displacement of bin i from the
    centre of mass of the polymer model — identical to what you would get by
    averaging r²_i across infinitely many sampled structures, but computed
    in a single O(N × N_modes) matrix-vector product with no sampling noise
    and no chromosome-territory centroid-pulling artifact.

    The translational / zero mode (index 0) is excluded by design because it
    represents uniform translation of the entire genome and carries no
    information about internal structure.

    Args:
        lam_abs:       Absolute eigenvalues from precompute_eigendecomposition,
                       length N, sorted ascending. lam_abs[0] ≈ 0.
        Q:             Eigenvector matrix (N × N), columns are eigenvectors.
        min_eigenvalue: Regularization floor applied to small eigenvalues.
                        Should be the same threshold used for structure sampling.

    Returns:
        sigma_diag: shape (N,) float64 array of σ_ii values.
    """
    lam_reg = np.maximum(lam_abs, min_eigenvalue)
    # (Q[:, 1:]**2) has shape (N, N-1); (1/lam_reg[1:]) has shape (N-1,).
    # The @ operator performs a matrix-vector product giving σ_ii for every bin.
    return (Q[:, 1:] ** 2) @ (1.0 / lam_reg[1:])


def compute_analytical_radial_into_h5(
    h5_path: str,
    lam_abs: np.ndarray,
    Q: np.ndarray,
    min_eigenvalue: float,
    n_bins: int = 5,
) -> None:
    """
    Compute analytical radial positions (σ_ii and q_rad) and write to HDF5.

    This is the preferred radial metric for biological interpretation because
    it is exact, free from sampling noise, and free from the chromosome-territory
    centroid-pulling artifact that biases the sampled ⟨r²⟩ approach.

    Three datasets are written / overwritten in the HDF5 file:

    'sigma_ii'  (N,) float64
        Raw diagonal of the covariance matrix Σ.  Equal to ⟨δr²_i⟩/3 where
        ⟨δr²_i⟩ is the mean squared displacement of bin i from the polymer
        centre of mass.  Larger σ_ii → bin is more peripheral.

    'q_rad'  (N,) float64
        Normalised radial position from Eq. 8 of Shi et al. 2024:
            q_rad(i) = σ_ii / (mean_j σ_jj)
        q_rad < 1 → bin at nuclear core; q_rad > 1 → bin at periphery.
        For a conventional nucleus, A-compartment bins have q_rad < 1
        and B-compartment bins have q_rad > 1.

    'analytical_radial_bins'  (N,) int16
        Equal-count quintile bins (1 = innermost, 5 = outermost) based on
        q_rad.  Use these for GC / expression radial gradient analysis in
        preference to 'sampled_radial_bins', which are based on the
        chromosome-territory-biased sampled ⟨r²⟩.

    Args:
        h5_path:       Path to the HDF5 file (must already exist with chr/start/end).
        lam_abs:       Eigenvalues from precompute_eigendecomposition.
        Q:             Eigenvectors from precompute_eigendecomposition.
        min_eigenvalue: Regularisation threshold (same value used for sampling).
        n_bins:        Number of equal-count radial shells (default 5).
    """
    print("Computing analytical radial positions (σ_ii / q_rad)...", flush=True)
    t0 = time.time()

    sigma_diag = compute_sigma_diag_global(lam_abs, Q, min_eigenvalue)
    mean_sigma  = float(np.mean(sigma_diag))
    q_rad       = sigma_diag / mean_sigma

    # Quintile bins: 1 = smallest q_rad (innermost), n_bins = largest (outermost)
    percentiles = np.linspace(0, 100, n_bins + 1)
    edges       = np.percentile(q_rad, percentiles)
    rad_bins    = np.digitize(q_rad, edges[1:-1]) + 1
    rad_bins    = np.clip(rad_bins, 1, n_bins).astype(np.int16)

    with h5py.File(h5_path, "a") as f:
        # Overwrite if already present (allows re-running with different threshold)
        for key in ("sigma_ii", "q_rad", "analytical_radial_bins"):
            if key in f:
                del f[key]

        f.create_dataset("sigma_ii",  data=sigma_diag.astype(np.float64))
        f.create_dataset("q_rad",     data=q_rad.astype(np.float64))
        f.create_dataset("analytical_radial_bins", data=rad_bins)

        # Store the threshold used so results are reproducible
        f.attrs["analytical_radial_min_eigenvalue"] = float(min_eigenvalue)
        f.attrs["analytical_radial_mean_sigma"]     = mean_sigma
        f.attrs["n_analytical_radial_bins"]         = int(n_bins)

    print(f"  σ_ii range: [{sigma_diag.min():.3e}, {sigma_diag.max():.3e}]  "
          f"mean={mean_sigma:.3e}", flush=True)
    print(f"  q_rad range: [{q_rad.min():.3f}, {q_rad.max():.3f}]  "
          f"(1.0 = genome-wide mean)", flush=True)
    print(f"✓ Analytical radial datasets written to {h5_path} "
          f"in {time.time()-t0:.1f}s", flush=True)


def sample_structure_xyz(
    lam_abs: np.ndarray,
    Q: np.ndarray,
    rng: np.random.Generator,
    min_eigenvalue: float,
) -> np.ndarray:
    """
    Mirrors your PRODUCTION sampling:
      - regularize eigenvalues: lam_reg = max(|lam|, threshold)
      - sample X_k ~ N(0, 1/lam_reg) for modes 1..N-1 in each axis
      - xyz = Q @ X
      - subtract centroid (mean) to center at origin
    """
    N = lam_abs.shape[0]
    lam_reg = np.maximum(lam_abs, min_eigenvalue)

    X = np.zeros((N, 3), dtype=np.float64)
    # skip zero mode (index 0)
    scale = np.sqrt(1.0 / lam_reg[1:])
    X[1:, :] = rng.standard_normal(size=(N - 1, 3)) * scale[:, None]

    xyz = Q @ X  # (N,3)
    xyz = xyz - xyz.mean(axis=0, keepdims=True)
    return xyz.astype(np.float32)


def create_h5_from_k(
    k_matrix_path: str,
    bins_path: str,
    output_h5: str,
    species: str,
    n_structures: int,
    seed: int = 1274,
    min_eigenvalue: float = MIN_EIGENVALUE_THRESHOLD_DEFAULT,
    chunks_coords: int = 256,
    overwrite: bool = False,
    n_radial_bins: int = 5,
) -> str:
    """
    Generates structures and writes directly into an HDF5 file.

    After sampling, the same eigendecomposition is reused to compute the
    analytical radial position σ_ii at no extra cost — the eigendecomposition
    is the expensive step and it only runs once.

    xyz_all shape: (n_coords, n_structures, 3)
    """
    K = read_K_matrix(k_matrix_path)
    bins = read_bins_filtered(bins_path)
    N = K.shape[0]
    if len(bins) != N:
        raise ValueError(f"Filtered bins length ({len(bins)}) must match K size ({N}).")

    out_path = Path(output_h5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        if overwrite:
            out_path.unlink()
        else:
            raise FileExistsError(f"Output HDF5 exists: {output_h5} (use --overwrite to replace)")

    print("Computing eigendecomposition (one-time cost)...", flush=True)
    t0 = time.time()
    lam_abs, Q = precompute_eigendecomposition(K)
    n_zero = int(np.sum(lam_abs < 1e-12))
    print(f"✓ Eigendecomposition done in {time.time() - t0:.1f}s  "
          f"(N={N}, zero modes={n_zero}, threshold={min_eigenvalue:.2e})", flush=True)

    chr_arr   = np.array([b.chrom.encode("utf-8") for b in bins], dtype="S")
    start_arr = np.array([b.start for b in bins], dtype=np.int64)
    end_arr   = np.array([b.end   for b in bins], dtype=np.int64)

    coords_chunk = min(int(chunks_coords), N)

    with h5py.File(out_path, "w") as f:
        f.create_dataset("chr",   data=chr_arr)
        f.create_dataset("start", data=start_arr)
        f.create_dataset("end",   data=end_arr)

        # Placeholder GC; filled by `gc` step.
        f.create_dataset("gc_content", data=np.full((N,), np.nan, dtype=np.float32))

        xyz_ds = f.create_dataset(
            "xyz_all",
            shape=(N, n_structures, 3),
            dtype=np.float32,
            compression="gzip",
            chunks=(coords_chunk, 1, 3),
        )

        f.attrs["species"]                   = species
        f.attrs["n_files"]                   = int(n_structures)
        f.attrs["n_coords"]                  = int(N)
        f.attrs["min_eigenvalue_threshold"]  = float(min_eigenvalue)
        f.attrs["k_matrix_path"]             = os.path.abspath(k_matrix_path)
        f.attrs["bins_path"]                 = os.path.abspath(bins_path)
        f.attrs["seed_base"]                 = int(seed)

        rng = np.random.default_rng(seed)

        print(f"Generating {n_structures} structures and writing xyz_all...", flush=True)
        gen0 = time.time()
        for j in range(n_structures):
            if (j + 1) % max(1, n_structures // 10) == 0 or j == 0:
                elapsed = time.time() - gen0
                rate    = (j + 1) / elapsed if elapsed > 0 else 0.0
                print(f"  {j+1}/{n_structures}  ({rate:.2f} struct/s)", flush=True)

            xyz = sample_structure_xyz(lam_abs, Q, rng=rng, min_eigenvalue=min_eigenvalue)
            xyz_ds[:, j, :] = xyz

    print(f"✓ Wrote HDF5: {output_h5}", flush=True)

    # Compute analytical radial positions immediately — reuses lam_abs and Q
    # that are already in memory from the eigendecomposition above.
    # This is the key efficiency gain: eigendecomposition cost is paid once.
    compute_analytical_radial_into_h5(
        h5_path=output_h5,
        lam_abs=lam_abs,
        Q=Q,
        min_eigenvalue=min_eigenvalue,
        n_bins=n_radial_bins,
    )

    return str(out_path)


# -----------------------------
# Pack existing *_genomic.txt -> HDF5
# -----------------------------
def _list_genomic_files(structures_dir: str, pattern: str = "*_genomic.txt") -> List[str]:
    files = sorted(glob.glob(os.path.join(structures_dir, pattern)))
    if not files:
        raise FileNotFoundError(f"No genomic files matching {pattern} in: {structures_dir}")
    return files


def pack_h5_from_genomic_txt(
    structures_dir: str,
    output_h5: str,
    species: str,
    chunks_coords: int = 256,
    pattern: str = "*_genomic.txt",
    overwrite: bool = False,
) -> str:
    """
    Reads files like your structure_XXXX_genomic.txt and creates xyz_all.

    Expected columns per data line (after 2-line header):
      bead_idx chr start end x y z
    """
    files = _list_genomic_files(structures_dir, pattern=pattern)
    n_files = len(files)

    # Read coordinates from the first file
    df0 = pd.read_csv(files[0], sep=r"\s+", skiprows=2, header=None,
                      names=["bead", "chr", "start", "end", "x", "y", "z"], engine="python")
    N = len(df0)

    chr_arr = df0["chr"].astype(str).values.astype("S")
    start_arr = df0["start"].to_numpy(dtype=np.int64)
    end_arr = df0["end"].to_numpy(dtype=np.int64)

    out_path = Path(output_h5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        if overwrite:
            out_path.unlink()
        else:
            raise FileExistsError(f"Output HDF5 exists: {output_h5} (use --overwrite to replace)")

    coords_chunk = min(int(chunks_coords), N)

    with h5py.File(out_path, "w") as f:
        f.create_dataset("chr", data=chr_arr)
        f.create_dataset("start", data=start_arr)
        f.create_dataset("end", data=end_arr)

        f.create_dataset("gc_content", data=np.full((N,), np.nan, dtype=np.float32))

        xyz_ds = f.create_dataset(
            "xyz_all",
            shape=(N, n_files, 3),
            dtype=np.float32,
            compression="gzip",
            chunks=(coords_chunk, 1, 3),
        )

        f.attrs["species"] = species
        f.attrs["n_files"] = int(n_files)
        f.attrs["n_coords"] = int(N)

        print(f"Packing {n_files} genomic files into HDF5 xyz_all...", flush=True)
        t0 = time.time()
        for j, path in enumerate(files):
            if (j + 1) % max(1, n_files // 10) == 0 or j == 0:
                elapsed = time.time() - t0
                rate = (j + 1) / elapsed if elapsed > 0 else 0.0
                print(f"  {j+1}/{n_files}  ({rate:.2f} files/s)", flush=True)

            df = pd.read_csv(path, sep=r"\s+", skiprows=2, header=None,
                             names=["bead", "chr", "start", "end", "x", "y", "z"], engine="python")
            if len(df) != N:
                raise ValueError(f"{path}: expected {N} rows, got {len(df)}")

            xyz = df[["x", "y", "z"]].to_numpy(dtype=np.float32)
            xyz_ds[:, j, :] = xyz

    print(f"✓ Wrote HDF5: {output_h5}", flush=True)
    return str(out_path)


# -----------------------------
# GC content via bedtools nuc
# -----------------------------
def _normalize_chr(name: str, strip_chr_prefix: bool, add_chr_prefix: bool) -> str:
    """
    Harmonize chromosome naming across:
      - your structure/bin files (often 'chr1')
      - the FASTA headers / .fai (often '1' for Ensembl)
      - chrom.sizes (often '1' for Juicer)
    """
    s = str(name)
    if strip_chr_prefix and s.lower().startswith("chr"):
        s = s[3:]
    if add_chr_prefix and not s.lower().startswith("chr"):
        s = "chr" + s
    return s


def _read_chrom_sizes(chrom_sizes_path: str, strip_chr_prefix: bool, add_chr_prefix: bool) -> Dict[str, int]:
    sizes: Dict[str, int] = {}
    with open(chrom_sizes_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            chrom = _normalize_chr(parts[0], strip_chr_prefix, add_chr_prefix)
            sizes[chrom] = int(parts[1])
    return sizes


def _clip_end_to_chrom_sizes(chrs: Sequence[str], starts: np.ndarray, ends: np.ndarray,
                            sizes: Dict[str, int]) -> np.ndarray:
    ends2 = ends.copy()
    for i, c in enumerate(chrs):
        if c in sizes and ends2[i] > sizes[c]:
            ends2[i] = sizes[c]
    return ends2


def compute_gc_into_h5(
    h5_path: str,
    genome_fasta: str,
    chrom_sizes: Optional[str] = None,
    tmp_dir: Optional[str] = None,
    strip_chr_prefix: bool = False,
    add_chr_prefix: bool = False,
) -> None:
    """
    Writes / overwrites dataset 'gc_content' in the HDF5 using bedtools nuc.

    Robust behavior:
      - bedtools nuc can SKIP intervals if the chromosome doesn't exist in FASTA (e.g. chrMT missing).
      - We merge bedtools output back onto the original bins by (chr,start,end) to preserve order.
      - Missing GC values become NaN instead of crashing.
    """
    if strip_chr_prefix and add_chr_prefix:
        raise ValueError("Choose only one of strip_chr_prefix or add_chr_prefix")

    if tmp_dir is None:
        tmp_dir = os.path.join(os.path.dirname(os.path.abspath(h5_path)) or ".", "tmp_gc")
    os.makedirs(tmp_dir, exist_ok=True)

    bed_path = os.path.join(tmp_dir, "bins.bed")
    out_path = os.path.join(tmp_dir, "gc.tsv")

    with h5py.File(h5_path, "r") as f:
        chr_raw = f["chr"][:]
        chrs = [c.decode("utf-8") if isinstance(c, (bytes, np.bytes_)) else str(c) for c in chr_raw]
        starts = f["start"][:].astype(np.int64)
        ends = f["end"][:].astype(np.int64)

    # Normalize chromosome names to match FASTA / chrom.sizes
    chrs = [_normalize_chr(c, strip_chr_prefix, add_chr_prefix) for c in chrs]

    if chrom_sizes:
        sizes = _read_chrom_sizes(chrom_sizes, strip_chr_prefix, add_chr_prefix)
        ends = _clip_end_to_chrom_sizes(chrs, starts, ends, sizes)

    # Keep an in-memory reference table for merge-back (preserves original order!)
    bins_df = pd.DataFrame({"chr": chrs, "start": starts, "end": ends})

    # Write BED
    with open(bed_path, "w") as fw:
        for c, s, e in zip(chrs, starts, ends):
            fw.write(f"{c}\t{s}\t{e}\n")

    # Run bedtools nuc
    cmd = f"bedtools nuc -fi {genome_fasta} -bed {bed_path} > {out_path}"
    subprocess.run(cmd, shell=True, check=True)

    df = pd.read_csv(out_path, sep="\t")

    # Find a GC% column robustly
    candidates = [c for c in df.columns if ("gc" in c.lower() and "pct" in c.lower())]
    if candidates:
        gc_col = candidates[0]
    else:
        candidates2 = [c for c in df.columns if "gc" in c.lower()]
        if not candidates2:
            raise RuntimeError(f"Could not find a GC column in bedtools output. Columns: {list(df.columns)}")
        gc_col = candidates2[0]

    # bedtools nuc repeats the BED cols first (chr,start,end); use those explicitly
    if df.shape[1] < 4:
        raise RuntimeError(f"bedtools nuc output has too few columns: {list(df.columns)}")

    bed_c0, bed_c1, bed_c2 = df.columns[:3]

    nuc_df = df[[bed_c0, bed_c1, bed_c2, gc_col]].copy()
    nuc_df.columns = ["chr", "start", "end", "gc_content"]

    # Types
    nuc_df["start"] = nuc_df["start"].astype(np.int64)
    nuc_df["end"] = nuc_df["end"].astype(np.int64)
    nuc_df["gc_content"] = nuc_df["gc_content"].astype(np.float32)

    # Merge back to preserve original bin ordering; missing bins -> NaN
    merged = bins_df.merge(nuc_df, on=["chr", "start", "end"], how="left")

    n_expected = len(bins_df)
    n_got = int(merged["gc_content"].notna().sum())
    n_missing = n_expected - n_got

    if n_missing > 0:
        missing_counts = (
            merged.loc[merged["gc_content"].isna(), "chr"]
            .value_counts()
            .to_dict()
        )
        print(
            f"WARNING: bedtools nuc returned {n_got}/{n_expected} GC values; "
            f"filled {n_missing} missing bins with NaN. Missing-by-chr: {missing_counts}",
            flush=True,
        )

    gc = merged["gc_content"].to_numpy(dtype=np.float32)

    with h5py.File(h5_path, "a") as f:
        if "gc_content" in f:
            del f["gc_content"]
        f.create_dataset("gc_content", data=gc)

    print(f"✓ GC computed and saved to HDF5: {h5_path}", flush=True)



# -----------------------------
# Euclidean distances + radial bins
# -----------------------------
def _parse_center(center_str: str) -> Union[str, np.ndarray]:
    """
    center_str:
      - 'origin'
      - 'geometric'
      - 'x,y,z'
    """
    if center_str in ("origin", "geometric"):
        return center_str
    if "," in center_str:
        parts = [p.strip() for p in center_str.split(",")]
        if len(parts) != 3:
            raise ValueError("center must be origin, geometric, or x,y,z")
        return np.array([float(parts[0]), float(parts[1]), float(parts[2])], dtype=np.float64)
    raise ValueError("center must be origin, geometric, or x,y,z")


def _compute_geometric_center_from_h5(ds_xyz: h5py.Dataset, chunk_coords: int = 256) -> np.ndarray:
    """
    Geometric center = midpoint of global bounding box across all points in xyz_all
    (same idea as calculate_euclidean_distances.py uses for 'geometric'). 
    """
    n_coords, _, _ = ds_xyz.shape
    mins = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
    maxs = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)
    for i0 in range(0, n_coords, chunk_coords):
        i1 = min(n_coords, i0 + chunk_coords)
        block = ds_xyz[i0:i1, :, :]  # (chunk, n_files, 3)
        mins = np.minimum(mins, block.min(axis=(0, 1)))
        maxs = np.maximum(maxs, block.max(axis=(0, 1)))
    return (mins + maxs) / 2.0


def compute_distances_and_bins(
    h5_path: str,
    center: Union[str, np.ndarray] = "origin",
    n_bins: int = 5,
    chunk_coords: int = 256,
    save_all_distances: bool = True,
    export_tsv: Optional[str] = None,
) -> None:
    """
    Computes mean SQUARED distance ⟨r²⟩ per bin from the sampled structure ensemble
    and bins beads into equal-count radial shells.

    NOTE ON NAMING: The output dataset is called 'avg_r2' (not 'avg_distance') because
    these are genuinely squared distances, not Euclidean distances. The earlier name
    'euclidean_distances_avg' was misleading. The TSV column is also renamed to 'avg_r2'.

    NOTE ON PURPOSE: This function is kept for nuclear MORPHOLOGY analysis — asphericity,
    shape factor, cell-to-cell variation in nuclear shape. For radial BIOLOGY (A/B
    compartment gradient, GC gradient), use the analytical q_rad from
    compute_analytical_radial_into_h5, which is exact and unbiased.

    Datasets written:
      avg_r2              (N,)    float32  mean ⟨r²⟩ across structures per bin
      sampled_radial_bins (N,)    int16    quantile bins 1..n_bins based on avg_r2
      r2_all              (N, M)  float32  per-structure r² values (only if save_all_distances)
    """
    with h5py.File(h5_path, "a") as f:
        if "xyz_all" not in f:
            raise ValueError("HDF5 must contain dataset 'xyz_all'")

        ds_xyz = f["xyz_all"]
        N, M, _ = ds_xyz.shape

        # Nuclear center
        if isinstance(center, np.ndarray):
            nuclear_center = center.astype(np.float64)
        elif center == "origin":
            nuclear_center = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        elif center == "geometric":
            nuclear_center = _compute_geometric_center_from_h5(ds_xyz, chunk_coords=chunk_coords)
        else:
            raise ValueError("center must be origin, geometric, or x,y,z")

        print(f"Nuclear center = {nuclear_center.tolist()}", flush=True)

        # Remove old dataset names if present (backward compatibility)
        for old_key in ("euclidean_distances_avg", "euclidean_distances_all", "radial_bins"):
            if old_key in f:
                del f[old_key]

        # Create output datasets with corrected names
        for key in ("avg_r2", "sampled_radial_bins"):
            if key in f:
                del f[key]

        avg_ds = f.create_dataset("avg_r2", shape=(N,), dtype=np.float32)

        all_ds = None
        if save_all_distances:
            if "r2_all" in f:
                del f["r2_all"]
            coords_chunk = min(int(chunk_coords), N)
            all_ds = f.create_dataset(
                "r2_all",
                shape=(N, M),
                dtype=np.float32,
                compression="gzip",
                chunks=(coords_chunk, min(512, M)),
            )

        # Compute SQUARED distances per structure per bin
        print("Computing squared Euclidean distances from center...", flush=True)
        t0  = time.time()
        avg = np.zeros((N,), dtype=np.float32)

        for i0 in range(0, N, chunk_coords):
            i1    = min(N, i0 + chunk_coords)
            block = ds_xyz[i0:i1, :, :]            # (chunk, M, 3)

            displacement = block - nuclear_center[None, None, :]
            r2           = np.sum(displacement ** 2, axis=2).astype(np.float32)  # (chunk, M)

            if all_ds is not None:
                all_ds[i0:i1, :] = r2

            avg_chunk      = r2.mean(axis=1)        # ⟨r²⟩ per bead
            avg[i0:i1]     = avg_chunk
            avg_ds[i0:i1]  = avg_chunk

            if (i0 == 0) or (i0 // chunk_coords) % max(1, (N // chunk_coords) // 10) == 0:
                rate = i1 / max(1e-9, time.time() - t0)
                print(f"  {i1}/{N} beads ({rate:.1f} beads/s)", flush=True)

        f.attrs["nuclear_center"]        = nuclear_center
        f.attrs["nuclear_center_method"] = ("custom" if isinstance(center, np.ndarray)
                                             else center)
        f.attrs["distance_metric"]       = "squared_euclidean"

        print(f"✓ ⟨r²⟩ computed in {time.time()-t0:.1f}s", flush=True)

        # Quintile bins based on sampled ⟨r²⟩
        percentiles = np.linspace(0, 100, n_bins + 1)
        edges       = np.percentile(avg, percentiles)
        bins_arr    = np.digitize(avg, edges[1:-1]) + 1
        bins_arr    = np.clip(bins_arr, 1, n_bins).astype(np.int16)

        f.create_dataset("sampled_radial_bins", data=bins_arr)
        f.attrs["n_sampled_radial_bins"]   = int(n_bins)
        f.attrs["sampled_radial_bin_edges"] = edges

        print("✓ Saved avg_r2 and sampled_radial_bins into HDF5", flush=True)

    if export_tsv:
        export_radial_tsv(h5_path, export_tsv)


def export_radial_tsv(h5_path: str, out_tsv: str) -> None:
    """
    Export a per-bin TSV with both sampled and analytical radial metrics.

    Columns:
      chr, start, end, gc_content  -- genomic location and GC fraction
      avg_r2, sampled_bin          -- from structure sampling (use for morphology)
      q_rad, analytical_bin        -- from analytical σ_ii (use for radial biology)

    If a dataset is absent (e.g. analytical radial not yet computed, or dist step
    not yet run), the corresponding columns are filled with NaN / -1 and a warning
    is printed rather than crashing.
    """
    with h5py.File(h5_path, "r") as f:
        chrs    = [c.decode("utf-8") if isinstance(c, (bytes, np.bytes_)) else str(c)
                   for c in f["chr"][:]]
        starts  = f["start"][:]
        ends    = f["end"][:]
        gc      = f["gc_content"][:] if "gc_content" in f else np.full(len(chrs), np.nan)
        N       = len(chrs)

        # Sampled radial metrics — from compute_distances_and_bins
        if "avg_r2" in f:
            avg_r2 = f["avg_r2"][:]
        elif "euclidean_distances_avg" in f:
            # Backward compatibility with v2 dataset name
            avg_r2 = f["euclidean_distances_avg"][:]
            print("WARNING: reading legacy 'euclidean_distances_avg' — "
                  "re-run `dist` step to get 'avg_r2'", flush=True)
        else:
            print("WARNING: 'avg_r2' not found — run `dist` step first", flush=True)
            avg_r2 = np.full(N, np.nan, dtype=np.float32)

        if "sampled_radial_bins" in f:
            sampled_bins = f["sampled_radial_bins"][:]
        elif "radial_bins" in f:
            sampled_bins = f["radial_bins"][:]
            print("WARNING: reading legacy 'radial_bins' — "
                  "re-run `dist` step to get 'sampled_radial_bins'", flush=True)
        else:
            sampled_bins = np.full(N, -1, dtype=np.int16)

        # Analytical radial metrics — from compute_analytical_radial_into_h5
        if "q_rad" in f:
            q_rad = f["q_rad"][:]
        else:
            print("WARNING: 'q_rad' not found — run `analytical-radial` step", flush=True)
            q_rad = np.full(N, np.nan, dtype=np.float64)

        if "analytical_radial_bins" in f:
            analytical_bins = f["analytical_radial_bins"][:]
        else:
            analytical_bins = np.full(N, -1, dtype=np.int16)

    df = pd.DataFrame({
        "chr":           chrs,
        "start":         starts,
        "end":           ends,
        "gc_content":    gc,
        "avg_r2":        avg_r2,         # mean ⟨r²⟩ from sampled structures
        "sampled_bin":   sampled_bins,   # quantile bin from avg_r2
        "q_rad":         q_rad,          # analytical radial position (Eq. 8 Shi 2024)
        "analytical_bin": analytical_bins,  # quantile bin from q_rad (PREFERRED)
    }).sort_values(["chr", "start"])

    Path(out_tsv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_tsv, sep="\t", index=False)
    print(f"✓ Exported TSV: {out_tsv}  (rows={len(df)})", flush=True)


# -----------------------------
# CLI
# -----------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="hlm_full_pipeline.py")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # A) K -> H5
    ap_k2h5 = sub.add_parser("k2h5", help="Generate structures from K_fit and write directly to HDF5")
    ap_k2h5.add_argument("--k-matrix", required=True)
    ap_k2h5.add_argument("--bins", required=True, help="Filtered bins file matching K_fit")
    ap_k2h5.add_argument("--species", required=True)
    ap_k2h5.add_argument("--n-structures", type=int, required=True)
    ap_k2h5.add_argument("--output-h5", required=True)
    ap_k2h5.add_argument("--seed", type=int, default=1274)
    ap_k2h5.add_argument("--min-eigenvalue", type=float, default=MIN_EIGENVALUE_THRESHOLD_DEFAULT,
                         help="Regularisation threshold. Set by expression_hic_validation_v3.py "
                              "(default: %(default)s)")
    ap_k2h5.add_argument("--n-radial-bins", type=int, default=5,
                         help="Number of analytical radial shells (default: 5)")
    ap_k2h5.add_argument("--chunks-coords", type=int, default=256)
    ap_k2h5.add_argument("--overwrite", action="store_true")

    # B) pack existing
    ap_pack = sub.add_parser("pack", help="Pack existing *_genomic.txt into HDF5 xyz_all")
    ap_pack.add_argument("--structures-dir", required=True)
    ap_pack.add_argument("--species", required=True)
    ap_pack.add_argument("--output-h5", required=True)
    ap_pack.add_argument("--pattern", default="*_genomic.txt")
    ap_pack.add_argument("--chunks-coords", type=int, default=256)
    ap_pack.add_argument("--overwrite", action="store_true")

    # C) GC
    ap_gc = sub.add_parser("gc", help="Compute GC content per bin into existing HDF5 using bedtools nuc")
    ap_gc.add_argument("--h5", required=True)
    ap_gc.add_argument("--genome-fasta", required=True)
    ap_gc.add_argument("--chrom-sizes", default=None, help="Optional chrom.sizes to clip bin ends")
    ap_gc.add_argument("--tmp-dir", default=None)
    ap_gc.add_argument("--strip-chr-prefix", action="store_true",
                       help="Use if your bins are chr1 but FASTA/chrom.sizes are 1")
    ap_gc.add_argument("--add-chr-prefix", action="store_true",
                       help="Use if your bins are 1 but FASTA/chrom.sizes are chr1")

    # D) distances
    ap_dist = sub.add_parser("dist", help="Compute Euclidean distances + radial bins; optionally export TSV")
    ap_dist.add_argument("--h5", required=True)
    ap_dist.add_argument("--center", default="origin", help="origin | geometric | x,y,z")
    ap_dist.add_argument("--n-bins", type=int, default=5)
    ap_dist.add_argument("--chunk-coords", type=int, default=256)
    ap_dist.add_argument("--save-all-distances", action="store_true",
                         help="Store euclidean_distances_all (N x M). Can be huge.")
    ap_dist.add_argument("--export-tsv", default=None)

    # E) Analytical radial — run on existing HDF5 without regenerating structures
    ap_ar = sub.add_parser(
        "analytical-radial",
        help="Compute analytical σ_ii / q_rad from K matrix and add to existing HDF5. "
             "Use this to re-run with a different threshold without regenerating structures."
    )
    ap_ar.add_argument("--k-matrix", required=True)
    ap_ar.add_argument("--h5", required=True, help="Existing HDF5 file to update")
    ap_ar.add_argument("--min-eigenvalue", type=float, default=MIN_EIGENVALUE_THRESHOLD_DEFAULT,
                       help="Regularisation threshold, same value used for structure sampling")
    ap_ar.add_argument("--n-radial-bins", type=int, default=5)

    # F) Run all (K->H5 or pack->H5), then GC, then distances, then analytical radial
    ap_all = sub.add_parser("run-all", help="End-to-end: (k2h5 OR pack) -> gc -> dist -> analytical-radial -> tsv")
    src = ap_all.add_mutually_exclusive_group(required=True)
    src.add_argument("--k-matrix", default=None)
    src.add_argument("--structures-dir", default=None)
    ap_all.add_argument("--bins", default=None, help="Required if using --k-matrix (filtered bins file)")
    ap_all.add_argument("--species", required=True)
    ap_all.add_argument("--n-structures", type=int, default=None, help="Required if using --k-matrix")
    ap_all.add_argument("--output-h5", required=True)
    ap_all.add_argument("--pattern", default="*_genomic.txt")
    ap_all.add_argument("--overwrite", action="store_true")

    # GC args
    ap_all.add_argument("--genome-fasta", required=True)
    ap_all.add_argument("--chrom-sizes", default=None)
    ap_all.add_argument("--tmp-dir", default=None)
    ap_all.add_argument("--strip-chr-prefix", action="store_true")
    ap_all.add_argument("--add-chr-prefix", action="store_true")

    # distance args
    ap_all.add_argument("--center", default="origin")
    ap_all.add_argument("--n-bins", type=int, default=5)
    ap_all.add_argument("--chunk-coords", type=int, default=256)
    ap_all.add_argument("--save-all-distances", action="store_true")
    ap_all.add_argument("--export-tsv", required=True)

    # HLM / radial args
    ap_all.add_argument("--seed", type=int, default=1274)
    ap_all.add_argument("--min-eigenvalue", type=float, default=MIN_EIGENVALUE_THRESHOLD_DEFAULT,
                        help="Regularisation threshold from expression_hic_validation_v3.py")
    ap_all.add_argument("--n-radial-bins", type=int, default=5)
    ap_all.add_argument("--chunks-coords", type=int, default=256)

    args = ap.parse_args(argv)

    if args.cmd == "k2h5":
        create_h5_from_k(
            k_matrix_path=args.k_matrix,
            bins_path=args.bins,
            output_h5=args.output_h5,
            species=args.species,
            n_structures=args.n_structures,
            seed=args.seed,
            min_eigenvalue=args.min_eigenvalue,
            chunks_coords=args.chunks_coords,
            overwrite=args.overwrite,
            n_radial_bins=args.n_radial_bins,
        )
        return 0

    if args.cmd == "pack":
        pack_h5_from_genomic_txt(
            structures_dir=args.structures_dir,
            output_h5=args.output_h5,
            species=args.species,
            chunks_coords=args.chunks_coords,
            pattern=args.pattern,
            overwrite=args.overwrite,
        )
        return 0

    if args.cmd == "gc":
        compute_gc_into_h5(
            h5_path=args.h5,
            genome_fasta=args.genome_fasta,
            chrom_sizes=args.chrom_sizes,
            tmp_dir=args.tmp_dir,
            strip_chr_prefix=getattr(args, "strip_chr_prefix", False),
            add_chr_prefix=getattr(args, "add_chr_prefix", False),
        )
        return 0

    if args.cmd == "dist":
        center = _parse_center(args.center)
        compute_distances_and_bins(
            h5_path=args.h5,
            center=center,
            n_bins=args.n_bins,
            chunk_coords=args.chunk_coords,
            save_all_distances=args.save_all_distances,
            export_tsv=args.export_tsv,
        )
        return 0

    if args.cmd == "analytical-radial":
        # Load K, run eigendecomposition, compute σ_ii and write to existing HDF5.
        # This is the fast path: no structure generation needed.
        # Typical use: re-run with a different threshold without regenerating structures.
        print(f"Loading K matrix: {args.k_matrix}", flush=True)
        K = read_K_matrix(args.k_matrix)
        print("Computing eigendecomposition...", flush=True)
        t0 = time.time()
        lam_abs, Q = precompute_eigendecomposition(K)
        print(f"✓ Eigendecomposition done in {time.time()-t0:.1f}s", flush=True)
        compute_analytical_radial_into_h5(
            h5_path=args.h5,
            lam_abs=lam_abs,
            Q=Q,
            min_eigenvalue=args.min_eigenvalue,
            n_bins=args.n_radial_bins,
        )
        return 0

    if args.cmd == "run-all":
        if args.k_matrix:
            if not args.bins:
                raise SystemExit("--bins is required when using --k-matrix")
            if args.n_structures is None:
                raise SystemExit("--n-structures is required when using --k-matrix")

            # create_h5_from_k already computes analytical radial internally,
            # so we don't need a separate step for it in this branch.
            create_h5_from_k(
                k_matrix_path=args.k_matrix,
                bins_path=args.bins,
                output_h5=args.output_h5,
                species=args.species,
                n_structures=args.n_structures,
                seed=args.seed,
                min_eigenvalue=args.min_eigenvalue,
                chunks_coords=args.chunks_coords,
                overwrite=args.overwrite,
                n_radial_bins=args.n_radial_bins,
            )
        else:
            pack_h5_from_genomic_txt(
                structures_dir=args.structures_dir,
                output_h5=args.output_h5,
                species=args.species,
                chunks_coords=args.chunks_coords,
                pattern=args.pattern,
                overwrite=args.overwrite,
            )
            # For the pack path (pre-existing structures), we need the K matrix
            # to compute analytical radial — it must be provided separately via
            # the analytical-radial subcommand.
            print("NOTE: analytical radial not computed for packed structures. "
                  "Run 'analytical-radial --k-matrix ... --h5 ...' separately.", flush=True)

        compute_gc_into_h5(
            h5_path=args.output_h5,
            genome_fasta=args.genome_fasta,
            chrom_sizes=args.chrom_sizes,
            tmp_dir=args.tmp_dir,
            strip_chr_prefix=getattr(args, "strip_chr_prefix", False),
            add_chr_prefix=getattr(args, "add_chr_prefix", False),
        )

        center = _parse_center(args.center)
        compute_distances_and_bins(
            h5_path=args.output_h5,
            center=center,
            n_bins=args.n_bins,
            chunk_coords=args.chunk_coords,
            save_all_distances=args.save_all_distances,
            export_tsv=args.export_tsv,
        )
        return 0

    raise SystemExit("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
