#!/usr/bin/env python3
"""
Merge and Balance Multiple UAV Geo-Localization Datasets

This script merges GTA-UAV (synthetic), UAV-VisLoc (real), and VPAIR (real) datasets
into a single JSON file with:
- Dataset source tracking (gta, visloc, vpair)
- Altitude metadata for real datasets
- Per-dataset data_root for path resolution
- Configurable duplication for class balancing

Usage:
    # Training (with duplication for balancing)
    python merge_json_balanced.py \
        --gta_json /path/to/GTA-UAV/cross-area-drone2sate-train.json \
        --visloc_json /path/to/UAV_VisLoc/cross-area-drone2sate-train.json \
        --vpair_json /path/to/VPair/cross-area-drone2drone-train.json \
        --output combined-lora-train.json \
        --visloc_multiplier 10 \
        --vpair_multiplier 10 \
        --gta_root /path/to/GTA-UAV/data/GTA-UAV \
        --visloc_root /path/to/UAV_VisLoc_dataset \
        --vpair_root /path/to/VPair

    # Test (no duplication)
    python merge_json_balanced.py \
        --gta_json /path/to/GTA-UAV/cross-area-drone2sate-test.json \
        --visloc_json /path/to/UAV_VisLoc/cross-area-drone2sate-test.json \
        --output combined-lora-test.json \
        --visloc_multiplier 1 \
        --gta_root /path/to/GTA-UAV/data/GTA-UAV \
        --visloc_root /path/to/UAV_VisLoc_dataset

Author: UAV Geo-Localization Pipeline
"""

import os
import json
import random
import argparse
import copy
from typing import List, Dict, Optional


def load_json(filepath: str) -> List[Dict]:
    """Load JSON file and return list of entries."""
    if not filepath or not os.path.exists(filepath):
        return []
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def add_dataset_metadata(
    entries: List[Dict],
    dataset_source: str,
    data_root: str,
    altitude_meters: Optional[float] = None
) -> List[Dict]:
    """
    Add metadata fields to each entry.
    
    Args:
        entries: List of dataset entries
        dataset_source: One of 'gta', 'visloc', 'vpair'
        data_root: Root directory for this dataset
        altitude_meters: Flight altitude (only for real datasets)
    
    Returns:
        Modified entries with metadata fields added
    """
    modified = []
    for entry in entries:
        entry_copy = copy.deepcopy(entry)
        entry_copy['dataset_source'] = dataset_source
        entry_copy['data_root'] = data_root
        if altitude_meters is not None:
            entry_copy['altitude_meters'] = altitude_meters
        modified.append(entry_copy)
    return modified


def duplicate_entries(
    entries: List[Dict],
    multiplier: int
) -> List[Dict]:
    """
    Duplicate entries N times for class balancing.
    
    Each duplicate gets a unique 'duplicate_id' field to track
    which virtual copy it is. This enables on-the-fly augmentation
    to produce different crops for the same source image.
    
    Args:
        entries: List of dataset entries
        multiplier: How many times to duplicate (1 = no duplication)
    
    Returns:
        List with entries duplicated multiplier times
    """
    if multiplier <= 1:
        # Add duplicate_id=0 even without duplication
        for entry in entries:
            entry['duplicate_id'] = 0
        return entries
    
    duplicated = []
    for entry in entries:
        for dup_idx in range(multiplier):
            entry_copy = copy.deepcopy(entry)
            entry_copy['duplicate_id'] = dup_idx
            duplicated.append(entry_copy)
    
    return duplicated


def merge_datasets(
    gta_entries: List[Dict],
    visloc_entries: List[Dict],
    vpair_entries: List[Dict],
    shuffle: bool = True,
    seed: int = 42
) -> List[Dict]:
    """
    Merge all dataset entries into a single list.
    
    Args:
        gta_entries: GTA-UAV entries (synthetic)
        visloc_entries: UAV-VisLoc entries (real)
        vpair_entries: VPAIR entries (real)
        shuffle: Whether to shuffle the merged list
        seed: Random seed for reproducibility
    
    Returns:
        Merged list of all entries
    """
    merged = gta_entries + visloc_entries + vpair_entries
    
    if shuffle:
        random.seed(seed)
        random.shuffle(merged)
    
    return merged


def print_statistics(
    gta_entries: List[Dict],
    visloc_entries: List[Dict],
    vpair_entries: List[Dict],
    merged: List[Dict]
):
    """Print dataset statistics."""
    print("\n" + "=" * 60)
    print("DATASET MERGE STATISTICS")
    print("=" * 60)
    print(f"GTA-UAV (synthetic):  {len(gta_entries):>8} entries")
    print(f"UAV-VisLoc (real):    {len(visloc_entries):>8} entries")
    print(f"VPAIR (real):         {len(vpair_entries):>8} entries")
    print("-" * 60)
    print(f"TOTAL MERGED:         {len(merged):>8} entries")
    print("=" * 60)
    
    # Count by source in merged
    sources = {}
    for entry in merged:
        src = entry.get('dataset_source', 'unknown')
        sources[src] = sources.get(src, 0) + 1
    
    print("\nMerged distribution:")
    for src, count in sorted(sources.items()):
        pct = 100.0 * count / len(merged) if merged else 0
        print(f"  {src}: {count} ({pct:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Merge and balance UAV geo-localization datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input files
    parser.add_argument('--gta_json', type=str, default=None,
                        help='Path to GTA-UAV train/test JSON')
    parser.add_argument('--visloc_json', type=str, default=None,
                        help='Path to UAV-VisLoc train/test JSON')
    parser.add_argument('--vpair_json', type=str, default=None,
                        help='Path to VPAIR train/test JSON')
    
    # Data roots
    parser.add_argument('--gta_root', type=str, default=None,
                        help='Root directory for GTA-UAV dataset')
    parser.add_argument('--visloc_root', type=str, default=None,
                        help='Root directory for UAV-VisLoc dataset')
    parser.add_argument('--vpair_root', type=str, default=None,
                        help='Root directory for VPAIR dataset')
    
    # Duplication multipliers
    parser.add_argument('--gta_multiplier', type=int, default=1,
                        help='Duplication factor for GTA-UAV (usually 1)')
    parser.add_argument('--visloc_multiplier', type=int, default=10,
                        help='Duplication factor for UAV-VisLoc')
    parser.add_argument('--vpair_multiplier', type=int, default=10,
                        help='Duplication factor for VPAIR')
    
    # Altitude metadata (for real datasets)
    parser.add_argument('--visloc_altitude', type=float, default=400.0,
                        help='UAV-VisLoc flight altitude in meters')
    parser.add_argument('--vpair_altitude', type=float, default=400.0,
                        help='VPAIR flight altitude in meters')
    
    # Output
    parser.add_argument('--output', type=str, required=True,
                        help='Output JSON file path')
    
    # Options
    parser.add_argument('--no_shuffle', action='store_true',
                        help='Do not shuffle the merged dataset')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for shuffling')
    
    args = parser.parse_args()
    
    # Load datasets
    print("\nLoading datasets...")
    
    gta_raw = load_json(args.gta_json) if args.gta_json else []
    visloc_raw = load_json(args.visloc_json) if args.visloc_json else []
    vpair_raw = load_json(args.vpair_json) if args.vpair_json else []
    
    print(f"  GTA-UAV: {len(gta_raw)} raw entries")
    print(f"  UAV-VisLoc: {len(visloc_raw)} raw entries")
    print(f"  VPAIR: {len(vpair_raw)} raw entries")
    
    # Add metadata
    print("\nAdding metadata...")
    
    gta_entries = []
    if gta_raw and args.gta_root:
        gta_meta = add_dataset_metadata(gta_raw, 'gta', args.gta_root)
        gta_entries = duplicate_entries(gta_meta, args.gta_multiplier)
        print(f"  GTA-UAV: {len(gta_raw)} × {args.gta_multiplier} = {len(gta_entries)}")
    
    visloc_entries = []
    if visloc_raw and args.visloc_root:
        visloc_meta = add_dataset_metadata(
            visloc_raw, 'visloc', args.visloc_root, args.visloc_altitude
        )
        visloc_entries = duplicate_entries(visloc_meta, args.visloc_multiplier)
        print(f"  UAV-VisLoc: {len(visloc_raw)} × {args.visloc_multiplier} = {len(visloc_entries)}")
    
    vpair_entries = []
    if vpair_raw and args.vpair_root:
        vpair_meta = add_dataset_metadata(
            vpair_raw, 'vpair', args.vpair_root, args.vpair_altitude
        )
        vpair_entries = duplicate_entries(vpair_meta, args.vpair_multiplier)
        print(f"  VPAIR: {len(vpair_raw)} × {args.vpair_multiplier} = {len(vpair_entries)}")
    
    # Merge
    merged = merge_datasets(
        gta_entries, visloc_entries, vpair_entries,
        shuffle=not args.no_shuffle,
        seed=args.seed
    )
    
    # Print statistics
    print_statistics(gta_entries, visloc_entries, vpair_entries, merged)
    
    # Save
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Saved merged dataset to: {args.output}")
    print(f"   File size: {os.path.getsize(args.output) / 1024 / 1024:.2f} MB")


if __name__ == '__main__':
    main()
