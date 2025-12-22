#!/usr/bin/env python3
"""
Convert JSON bin metadata to simple text format for use with structure generation scripts.

Usage:
    python json_to_bin_info.py mydata_bins.json output_bins.txt
"""

import json
import sys

def main():
    if len(sys.argv) < 3:
        print("Usage: python json_to_bin_info.py input.json output.txt")
        sys.exit(1)
    
    input_json = sys.argv[1]
    output_txt = sys.argv[2]
    
    print(f"Reading JSON: {input_json}")
    with open(input_json, 'r') as f:
        data = json.load(f)
    
    # Extract bin information
    bins = data['bins']
    resolution = data['resolution_bp']
    
    print(f"  Resolution: {resolution} bp")
    print(f"  Total bins: {len(bins)}")
    print(f"  Filtered size: {data['filtered_matrix_size']}")
    
    # Write to text file
    print(f"\nWriting to: {output_txt}")
    with open(output_txt, 'w') as f:
        f.write(f"# Genomic bins at {resolution} bp resolution (filtered)\n")
        f.write(f"# Original matrix: {data['original_matrix_size']} bins\n")
        f.write(f"# Filtered matrix: {data['filtered_matrix_size']} bins\n")
        f.write(f"# Format: new_index  chr  start  end  old_index\n")
        f.write(f"#\n")
        
        for bin_info in bins:
            new_idx = bin_info['new_global_index']
            chr_name = f"chr{bin_info['chrom_label']}"
            start = bin_info['start_bp']
            end = bin_info['end_bp']
            old_idx = bin_info['old_global_index']
            
            f.write(f"{new_idx}\t{chr_name}\t{start}\t{end}\t{old_idx}\n")
    
    print(f"  ✓ Wrote {len(bins)} bins")
    
    # Summary by chromosome
    print("\nBins per chromosome:")
    chr_counts = {}
    for bin_info in bins:
        chr_name = bin_info['chrom_label']
        chr_counts[chr_name] = chr_counts.get(chr_name, 0) + 1
    
    for chr_name in sorted(chr_counts.keys(), key=lambda x: (x.replace('X','23').replace('Y','24'), x)):
        print(f"  chr{chr_name}: {chr_counts[chr_name]}")
    
    print("\n" + "="*60)
    print("SUCCESS!")
    print("="*60)
    print(f"Output: {output_txt}")
    print(f"Total bins: {len(bins)}")
    print("\nThis file can now be used with:")
    print(f"  python HLM_K2xyz_with_genomic.py K_matrix.txt {output_txt} 1000")
    print("="*60)

if __name__ == '__main__':
    main()
