#!/usr/bin/env python3
"""
Merge GTA-UAV and OrthoLoc Datasets for Combined Training

This script merges GTA-UAV (synthetic) and ortholoc_converted (real) datasets
into a single JSON file with:
- Dataset source tracking (gta, ortholoc)
- Altitude metadata for real datasets
- Per-dataset data_root for path resolution
- No balancing (use all data as-is)

Usage:
    python merge_datasets.py \
        --gta_json /media/aniel/storage/GTA_dataset/cross-area-drone2sate-train.json \
        --ortholoc_json /media/aniel/storage/ortholoc_converted/cross-area-drone2sate-train.json \
        --gta_root /media/aniel/storage/GTA_dataset \
        --ortholoc_root /media/aniel/storage/ortholoc_converted \
        --output /media/aniel/storage/GTA-UAV/Game4Loc/combined-train.json

Author: Game4Loc Combined Training Pipeline
"""

import os
import json
import random
import argparse
import copy
from typing import List, Dict, Optional


def load_json(filepath: str) -> List[Dict]:
    """
    Load JSON file and return list of entries.

    Args:
        filepath: Path to JSON file

    Returns:
        List of dataset entries

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If JSON is invalid or not a list
    """
    if not filepath:
        raise ValueError("JSON filepath cannot be empty")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"JSON file not found: {filepath}")

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in {filepath}: {e}")

    if not isinstance(data, list):
        raise ValueError(f"Expected list of entries, got {type(data)}")

    return data


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
        dataset_source: One of 'gta', 'ortholoc'
        data_root: Root directory for this dataset
        altitude_meters: Flight altitude (None for GTA, ~110.0 for OrthoLoc)

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


def merge_datasets(
    gta_entries: List[Dict],
    ortholoc_entries: List[Dict],
    shuffle: bool = True,
    seed: int = 42
) -> List[Dict]:
    """
    Merge GTA and OrthoLoc entries into a single list.

    Args:
        gta_entries: GTA-UAV entries (synthetic)
        ortholoc_entries: OrthoLoc entries (real)
        shuffle: Whether to shuffle the merged list
        seed: Random seed for reproducibility

    Returns:
        Merged list of all entries
    """
    merged = gta_entries + ortholoc_entries

    if shuffle:
        random.seed(seed)
        random.shuffle(merged)

    return merged


def print_statistics(
    gta_entries: List[Dict],
    ortholoc_entries: List[Dict],
    merged: List[Dict]
):
    """Print dataset statistics."""
    print("\n" + "=" * 63)
    print("DATASET MERGE STATISTICS")
    print("=" * 63)
    print(f"GTA-UAV (synthetic):     {len(gta_entries):>8} entries")
    print(f"OrthoLoc (real):         {len(ortholoc_entries):>8} entries")
    print("-" * 63)
    print(f"TOTAL MERGED:            {len(merged):>8} entries")
    print("=" * 63)

    # Count by source in merged
    sources = {}
    for entry in merged:
        src = entry.get('dataset_source', 'unknown')
        sources[src] = sources.get(src, 0) + 1

    print("\nMerged distribution:")
    for src, count in sorted(sources.items()):
        pct = 100.0 * count / len(merged) if merged else 0
        print(f"  {src}: {count} ({pct:.1f}%)")


def validate_data_root(data_root: str, dataset_name: str):
    """
    Validate dataset root exists and contains expected structure.

    Args:
        data_root: Root directory to validate
        dataset_name: Name for error messages (e.g., 'GTA-UAV', 'OrthoLoc')
    """
    if not os.path.exists(data_root):
        raise FileNotFoundError(f"{dataset_name} data_root not found: {data_root}")

    # Check for expected subdirectories (warn only, don't fail)
    drone_dir = os.path.join(data_root, 'drone', 'images')
    sat_dir = os.path.join(data_root, 'satellite')

    if not os.path.exists(drone_dir):
        print(f"Warning: {dataset_name} drone directory not found at {drone_dir}")
    if not os.path.exists(sat_dir):
        print(f"Warning: {dataset_name} satellite directory not found at {sat_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge GTA-UAV and OrthoLoc datasets for combined training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Input files (required)
    parser.add_argument('--gta_json', type=str, required=True,
                        help='Path to GTA-UAV train JSON')
    parser.add_argument('--ortholoc_json', type=str, required=True,
                        help='Path to OrthoLoc train JSON')

    # Data roots (required)
    parser.add_argument('--gta_root', type=str, required=True,
                        help='Root directory for GTA-UAV dataset')
    parser.add_argument('--ortholoc_root', type=str, required=True,
                        help='Root directory for OrthoLoc dataset')

    # Altitude metadata
    parser.add_argument('--ortholoc_altitude', type=float, default=110.0,
                        help='OrthoLoc flight altitude in meters')

    # Output (required)
    parser.add_argument('--output', type=str, required=True,
                        help='Output JSON file path')

    # Options
    parser.add_argument('--no_shuffle', action='store_true',
                        help='Do not shuffle the merged dataset')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for shuffling')

    args = parser.parse_args()

    # Validate data roots
    print("\nValidating data roots...")
    validate_data_root(args.gta_root, 'GTA-UAV')
    validate_data_root(args.ortholoc_root, 'OrthoLoc')

    # Load datasets
    print("\nLoading datasets...")

    try:
        gta_raw = load_json(args.gta_json)
        print(f"  GTA-UAV: {len(gta_raw)} entries loaded from {args.gta_json}")
    except Exception as e:
        print(f"Error loading GTA JSON: {e}")
        return 1

    try:
        ortholoc_raw = load_json(args.ortholoc_json)
        print(f"  OrthoLoc: {len(ortholoc_raw)} entries loaded from {args.ortholoc_json}")
    except Exception as e:
        print(f"Error loading OrthoLoc JSON: {e}")
        return 1

    # Add metadata
    print("\nAdding metadata...")

    gta_entries = add_dataset_metadata(gta_raw, 'gta', args.gta_root)
    print(f"  GTA-UAV: Added metadata to {len(gta_entries)} entries")

    ortholoc_entries = add_dataset_metadata(
        ortholoc_raw, 'ortholoc', args.ortholoc_root, args.ortholoc_altitude
    )
    print(f"  OrthoLoc: Added metadata to {len(ortholoc_entries)} entries (altitude: {args.ortholoc_altitude}m)")

    # Merge
    merged = merge_datasets(
        gta_entries, ortholoc_entries,
        shuffle=not args.no_shuffle,
        seed=args.seed
    )

    # Print statistics
    print_statistics(gta_entries, ortholoc_entries, merged)

    # Save
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"\nSaving merged dataset...")
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Saved merged dataset to: {args.output}")
    print(f"   File size: {os.path.getsize(args.output) / 1024 / 1024:.2f} MB")
    print("=" * 63)


if __name__ == '__main__':
    exit(main() or 0)
