#!/usr/bin/env python3
"""
Filtering + export for HLM-genome input.

Steps:
1. Load whole-genome contact matrix P (after KR + global rescaling, diag=1).
2. For each bin (chromatin segment), look at contacts with OTHER segments
   on the SAME chromosome only.
   If more than 90% of those values are zero, drop that bin.
3. Build a filtered whole-genome matrix (rows/cols of dropped bins removed).
4. Set the diagonal of the filtered matrix to NaN.
5. Save:
   - TXT matrix for HLM-genome (space-separated, NaN on diagonal, header line)
   - Updated metadata JSON with new bin order.

Usage:
  python filter_and_export_hlm_matrix.py \
      --matrix-npy mydata_1Mb_GWKR_P_whole.npy \
      --metadata-json mydata_1Mb_GWKR_bins_metadata.json \
      --out-matrix-txt mydata_1Mb_GWKR_HLM_input.txt \
      --out-metadata-json mydata_1Mb_GWKR_filtered_bins_metadata.json
"""

import argparse
import json
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Filter low-contact bins and export HLM-genome matrix"
    )
    parser.add_argument("--matrix-npy", required=True,
                        help="Input .npy file with whole-genome P matrix")
    parser.add_argument("--metadata-json", required=True,
                        help="Input metadata JSON from previous step")
    parser.add_argument("--out-matrix-txt", required=True,
                        help="Output TXT file: filtered matrix for HLM-genome")
    parser.add_argument("--out-metadata-json", required=True,
                        help="Output JSON: updated metadata after filtering")
    parser.add_argument("--zero-threshold", type=float, default=0.9,
                        help="Fraction of zero intra-chrom contacts above which "
                             "a bin is removed (default: 0.9 -> >90%% zeros)")

    args = parser.parse_args()

    # 1. Load data
    print("Loading matrix and metadata ...", flush=True)
    P = np.load(args.matrix_npy)
    with open(args.metadata_json) as f:
        meta = json.load(f)

    N = P.shape[0]
    assert P.shape[0] == P.shape[1], "Matrix must be square"

    bins = meta["bins"]  # list of dicts, one per global index in original matrix
    assert len(bins) == N, "Number of bins in metadata must match matrix size"

    # Group bin indices by chromosome (hic name)
    chrom_to_indices = {}
    for b in bins:
        cid = b["chrom_hic_name"]
        idx = b["global_index"]
        chrom_to_indices.setdefault(cid, []).append(idx)

    # Make sure indices within each chrom are sorted by original bin_index
    for cid, idx_list in chrom_to_indices.items():
        chrom_to_indices[cid] = sorted(idx_list, key=lambda i: bins[i]["bin_index"])

    print("Chromosomes found:", ", ".join(chrom_to_indices.keys()))

    # 2. Determine which bins to keep based on intra-chrom zero fraction
    keep = np.ones(N, dtype=bool)
    n_removed_total = 0

    for cid, idx_list in chrom_to_indices.items():
        idx_array = np.array(idx_list, dtype=int)
        n_chrom_bins = len(idx_array)
        print(f"Checking chromosome {cid} with {n_chrom_bins} bins ...", flush=True)

        for pos_in_chrom, global_i in enumerate(idx_array):
            # Contacts of bin i with all bins on the same chromosome
            row = P[global_i, idx_array]

            # Exclude self-contact from this check
            row_no_self = row.copy()
            row_no_self[pos_in_chrom] = 0.0  # ignore self

            n_total = n_chrom_bins - 1  # "other segments"
            if n_total <= 0:
                continue  # single-bin chrom, keep it

            n_zero = np.count_nonzero(row_no_self == 0.0)
            frac_zero = n_zero / float(n_total)

            if frac_zero > args.zero_threshold:
                keep[global_i] = False
                n_removed_total += 1

        n_kept_chrom = np.count_nonzero(keep[idx_array])
        print(f"  Chrom {cid}: kept {n_kept_chrom} / {n_chrom_bins} bins", flush=True)

    print(f"Total bins removed: {n_removed_total} out of {N}")
    print(f"Total bins kept   : {np.count_nonzero(keep)}")

    # 3. Build filtered matrix
    print("Building filtered matrix ...", flush=True)
    kept_indices = np.where(keep)[0]
    P_filtered = P[np.ix_(kept_indices, kept_indices)]

    # 4. Set diagonal to NaN for HLM-genome input
    print("Setting diagonal to NaN ...", flush=True)
    np.fill_diagonal(P_filtered, np.nan)

    # 5. Save matrix as TXT in HLM format
    print(f"Saving filtered matrix to {args.out_matrix_txt}", flush=True)
    M = P_filtered.shape[0]

    # Compute min/max over non-NaN, non-zero entries
    mask_valid = ~np.isnan(P_filtered) & (P_filtered > 0)
    if not np.any(mask_valid):
        raise RuntimeError("No positive, non-NaN entries to compute min/max.")
    min_val = P_filtered[mask_valid].min()
    max_val = P_filtered[mask_valid].max()

    with open(args.out_matrix_txt, "w") as f:
        # Header line matching your example style
        f.write(
            f"#shape: {M} {M} min: {min_val:.6g} max: {max_val:.6g}\n"
        )

        # Matrix rows: space-separated, NaN as 'NaN', values as '%.6g'
        for row in P_filtered:
            entries = []
            for v in row:
                if np.isnan(v):
                    entries.append("NaN")
                else:
                    entries.append(f"{v:.6g}")
            f.write(" ".join(entries) + "\n")

    # 6. Build updated metadata
    print(f"Building and saving updated metadata to {args.out_metadata_json}", flush=True)

    # Map old index -> new index
    old_to_new = {int(old): int(new) for new, old in enumerate(kept_indices)}

    # New bins list in the new order (0..M-1)
    new_bins = []
    for new_idx, old_idx in enumerate(kept_indices):
        b_old = bins[int(old_idx)]
        b_new = dict(b_old)  # copy
        b_new["old_global_index"] = int(old_idx)
        b_new["new_global_index"] = int(new_idx)
        new_bins.append(b_new)

    # Recompute per-chrom offsets and n_bins_chr in the new matrix
    new_chrom_to_indices = {}
    for b in new_bins:
        cid = b["chrom_hic_name"]
        new_idx = b["new_global_index"]
        new_chrom_to_indices.setdefault(cid, []).append(new_idx)

    new_offsets = {}
    new_n_bins_chr = {}
    offset = 0
    for cid in sorted(new_chrom_to_indices.keys(), key=lambda x: x):
        idx_list = sorted(new_chrom_to_indices[cid])
        n = len(idx_list)
        new_offsets[cid] = offset
        new_n_bins_chr[cid] = n
        offset += n

    new_meta = {
        "resolution_bp": meta.get("resolution_bp", None),
        "chrom_sizes_bp": meta.get("chrom_sizes_bp", None),
        "original_offsets": meta.get("offsets", None),
        "original_n_bins_chr": meta.get("n_bins_chr", None),
        "original_matrix_size": N,
        "filtered_matrix_size": int(P_filtered.shape[0]),
        "new_offsets": new_offsets,
        "new_n_bins_chr": new_n_bins_chr,
        "bins": new_bins,
    }

    with open(args.out_metadata_json, "w") as f:
        json.dump(new_meta, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()
