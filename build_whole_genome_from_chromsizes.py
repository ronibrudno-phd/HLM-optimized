#!/usr/bin/env python3
"""
Build whole-genome contact probability matrix (intra + inter)
from a .hic file using Juicer's GW_KR normalization and an arbitrary
chrom.sizes file (e.g. Drosophila).

Steps:
  1. Read chrom.sizes: each line "chrom_name<TAB>size_bp".
  2. For each chrom and chrom pair, use:
        juicer_tools dump observed GW_KR hic chromA chromB BP RES
  3. Assemble the full genome-wide matrix.
  4. Compute mean diagonal <p_ii> over all bins.
  5. Divide entire matrix by <p_ii>, then set diag = 1.
  6. Save:
        <out_prefix>_P_whole.npy
        <out_prefix>_bins_metadata.json

Usage example (Drosophila, resolution 1Mb):
  python build_whole_genome_from_chromsizes.py \\
      --juicer-jar /path/to/juicer_tools.jar \\
      --hic-file  dmel.hic \\
      --chromsizes-file dromel_v6_merged_BDGP6.46.58.chrom.sizes.sorted_whole_chroms_only \\
      --out-prefix dmel_1Mb_GWKR \\
      --resolution 1000000
"""

import argparse
import subprocess
import sys
import math
import json
from collections import OrderedDict

import numpy as np


def run_cmd(cmd):
    """Run a shell command and return stdout as text, or raise on error."""
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        sys.stderr.write("Command failed:\n{}\n\n".format(" ".join(cmd)))
        sys.stderr.write("STDERR:\n{}\n".format(e.stderr))
        raise
    return result.stdout


def read_chromsizes(chromsizes_file):
    """
    Read chrom.sizes style file: 'name<tab>size' per line.
    Returns OrderedDict(name -> size_bp) preserving file order.
    """
    chrom_sizes = OrderedDict()
    with open(chromsizes_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            name = parts[0]
            size_bp = int(parts[1])
            chrom_sizes[name] = size_bp
    return chrom_sizes


def dump_observed_GWKR(juicer_jar, hic_file, chrom1, chrom2, resolution):
    """
    Dump GW_KR observed contacts for a given chrom pair at given resolution.

    Returns a list of (start1_bp, start2_bp, value) tuples.
    """
    cmd = [
        "java", "-Xmx10g", "-jar", juicer_jar,
        "dump", "observed", "GW_KR",
        hic_file,
        chrom1, chrom2,
        "BP", str(resolution)
    ]
    out = run_cmd(cmd)

    data = []
    for line in out.strip().splitlines():
        if not line.strip():
            continue
        f = line.strip().split()
        if len(f) < 3 or len(f) > 3:
            continue
        start1 = int(float(f[0]))  # some juicer versions give floats
        start2 = int(float(f[1]))
        val = float(f[2])
        if val == 0.0:
            continue
        data.append((start1, start2, val))
    return data


def main():
    parser = argparse.ArgumentParser(
        description="Build whole-genome GW_KR-rescaled matrix from .hic using chrom.sizes"
    )
    parser.add_argument("--juicer-jar", required=True,
                        help="Path to juicer_tools.jar")
    parser.add_argument("--hic-file", required=True,
                        help="Path to input .hic file")
    parser.add_argument("--chromsizes-file", required=True,
                        help="Path to chrom.sizes file (name<TAB>size)")
    parser.add_argument("--out-prefix", required=True,
                        help="Prefix for outputs (P_whole.npy, bins_metadata.json)")
    parser.add_argument("--resolution", type=int, default=1_000_000,
                        help="Bin size in bp (default: 1,000,000 = 1Mb)")
    parser.add_argument(
        "--include-chrom",
        nargs="*",
        help="Optional subset of chromosomes/scaffolds to include by exact name "
             "(e.g. HiC_scaffold_1 HiC_scaffold_2). "
             "If not given, all entries in chrom.sizes are used."
    )

    args = parser.parse_args()

    juicer_jar = args.juicer_jar
    hic_file = args.hic_file
    chromsizes_file = args.chromsizes_file
    out_prefix = args.out_prefix
    resolution = args.resolution

    # 1) Read chromosome sizes from file
    print(f"Reading chrom.sizes from {chromsizes_file} ...", flush=True)
    chrom_sizes_all = read_chromsizes(chromsizes_file)

    if args.include_chrom:
        # Filter to user-specified subset, preserving file order
        chroms = [c for c in chrom_sizes_all.keys() if c in args.include_chrom]
        chrom_sizes = OrderedDict((c, chrom_sizes_all[c]) for c in chroms)
    else:
        chrom_sizes = chrom_sizes_all
        chroms = list(chrom_sizes.keys())

    print("Chromosomes/scaffolds used:", ", ".join(chroms))

    # 2) Compute number of bins and global offsets
    n_bins_chr = {}
    offsets = {}
    offset = 0
    for c, size_bp in chrom_sizes.items():
        n_bins = math.ceil(size_bp / resolution)
        n_bins_chr[c] = n_bins
        offsets[c] = offset
        offset += n_bins

    N_total = offset
    print(f"Total bins (whole genome): {N_total}")

    # 3) Allocate full genome-wide matrix
    print("Allocating whole-genome matrix ...", flush=True)
    P_counts = np.zeros((N_total, N_total), dtype=float)

    # 4) Fill intra-chrom blocks
    print("Filling intra-chromosome blocks ...", flush=True)
    for c in chroms:
        print(f"  Intra: {c} vs {c}", flush=True)
        data = dump_observed_GWKR(juicer_jar, hic_file, c, c, resolution)
        offset_c = offsets[c]
        n_bins = n_bins_chr[c]
        for start1, start2, val in data:
            i_local = start1 // resolution
            j_local = start2 // resolution
            if i_local >= n_bins or j_local >= n_bins:
                continue
            i = offset_c + i_local
            j = offset_c + j_local
            if i >= N_total or j >= N_total:
                continue
            P_counts[i, j] = val
            P_counts[j, i] = val  # symmetric

    # 5) Fill inter-chrom blocks
    print("Filling inter-chromosome blocks ...", flush=True)
    for i_idx, ca in enumerate(chroms):
        for cb in chroms[i_idx + 1:]:
            print(f"  Inter: {ca} vs {cb}", flush=True)
            data = dump_observed_GWKR(juicer_jar, hic_file, ca, cb, resolution)
            offset_a = offsets[ca]
            offset_b = offsets[cb]
            n_a = n_bins_chr[ca]
            n_b = n_bins_chr[cb]
            for start1, start2, val in data:
                i_local = start1 // resolution
                j_local = start2 // resolution
                if i_local >= n_a or j_local >= n_b:
                    continue
                i = offset_a + i_local
                j = offset_b + j_local
                if i >= N_total or j >= N_total:
                    continue
                P_counts[i, j] = val
                P_counts[j, i] = val  # symmetric

    # 6) Global rescaling: mean diagonal across whole genome -> 1
    print("Rescaling matrix: mean diag -> 1, then diag = 1 ...", flush=True)
    diag = np.diag(P_counts)
    mask = diag > 0
    if not np.any(mask):
        raise RuntimeError("No positive entries on diagonal; check normalization / resolution.")
    mean_diag = diag[mask].mean()
    print(f"Mean diagonal before rescaling: {mean_diag}")

    P = P_counts / mean_diag
    np.fill_diagonal(P, 1.0)

    # 7) Save outputs
    mat_file = f"{out_prefix}_P_whole.npy"
    meta_file = f"{out_prefix}_bins_metadata.json"

    print(f"Saving matrix to {mat_file}", flush=True)
    np.save(mat_file, P)

    # Metadata mapping
    print(f"Saving metadata to {meta_file}", flush=True)
    bin_info = []
    for c in chroms:
        off = offsets[c]
        n = n_bins_chr[c]
        for k in range(n):
            global_idx = off + k
            start_bp = k * resolution
            end_bp = (k + 1) * resolution
            bin_info.append({
                "global_index": int(global_idx),
                "chrom_hic_name": c,   # used by filter script
                "chrom_label": c,      # same as name here
                "bin_index": int(k),
                "start_bp": int(start_bp),
                "end_bp": int(end_bp),
            })

    meta = {
        "resolution_bp": int(resolution),
        "chrom_sizes_bp": {c: int(size) for c, size in chrom_sizes.items()},
        "offsets": {c: int(off) for c, off in offsets.items()},
        "n_bins_chr": {c: int(n) for c, n in n_bins_chr.items()},
        "bins": bin_info,
    }

    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()
