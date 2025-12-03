#!/usr/bin/env python3
"""
Merge multiple GTA-format JSON files into unified train/test datasets.

This script combines UAV_VisLoc and VPair JSONs into single files that can
be used with GTADatasetTrain or VirtualExpandDataset.

Usage:
    python merge_datasets.py --output_dir /path/to/output

The script will look for datasets in parent directories and create:
    - combined-train.json
    - combined-test.json
"""

import os
import json
import argparse
from pathlib import Path


def load_json(filepath):
    """Load JSON file and return data."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data, filepath):
    """Save data to JSON file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(data)} entries to {filepath}")


def adjust_paths(entries, dataset_prefix):
    """
    Adjust image directory paths to be relative to combined root.
    
    Args:
        entries: List of JSON entries
        dataset_prefix: Prefix to add (e.g., 'UAV_VisLoc_dataset' or 'vpair')
    
    Returns:
        Modified entries with adjusted paths
    """
    adjusted = []
    for entry in entries:
        new_entry = entry.copy()
        # Prepend dataset prefix to image directories
        new_entry['drone_img_dir'] = os.path.join(dataset_prefix, entry['drone_img_dir'])
        new_entry['sate_img_dir'] = os.path.join(dataset_prefix, entry['sate_img_dir'])
        adjusted.append(new_entry)
    return adjusted


def merge_datasets(config):
    """
    Merge multiple datasets into combined train/test files.
    
    Args:
        config: Dict mapping dataset_name -> {train_json, test_json, prefix}
    
    Returns:
        (combined_train, combined_test) lists
    """
    combined_train = []
    combined_test = []
    
    for name, paths in config.items():
        print(f"\nProcessing {name}...")
        
        # Load train data
        if paths.get('train_json') and os.path.exists(paths['train_json']):
            train_data = load_json(paths['train_json'])
            train_data = adjust_paths(train_data, paths['prefix'])
            combined_train.extend(train_data)
            print(f"  Train: {len(train_data)} entries")
        else:
            print(f"  Train: NOT FOUND at {paths.get('train_json')}")
        
        # Load test data
        if paths.get('test_json') and os.path.exists(paths['test_json']):
            test_data = load_json(paths['test_json'])
            test_data = adjust_paths(test_data, paths['prefix'])
            combined_test.extend(test_data)
            print(f"  Test: {len(test_data)} entries")
        else:
            print(f"  Test: NOT FOUND at {paths.get('test_json')}")
    
    return combined_train, combined_test


def main():
    parser = argparse.ArgumentParser(description='Merge UAV geo-localization datasets')
    parser.add_argument('--datasets_root', type=str, 
                        default='/home/aniel/skyline_drone/datasets',
                        help='Root directory containing all datasets')
    parser.add_argument('--output_dir', type=str,
                        default='/home/aniel/skyline_drone/datasets',
                        help='Output directory for combined JSONs')
    parser.add_argument('--include_visloc', action='store_true', default=True,
                        help='Include UAV_VisLoc dataset')
    parser.add_argument('--include_vpair', action='store_true', default=True,
                        help='Include VPair dataset')
    args = parser.parse_args()
    
    datasets_root = Path(args.datasets_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure datasets to merge
    config = {}
    
    if args.include_visloc:
        visloc_root = datasets_root / 'UAV_VisLoc_dataset'
        config['UAV_VisLoc'] = {
            'train_json': str(visloc_root / 'cross-area-drone2sate-train.json'),
            'test_json': str(visloc_root / 'cross-area-drone2sate-test.json'),
            'prefix': 'UAV_VisLoc_dataset'
        }
    
    if args.include_vpair:
        vpair_root = datasets_root / 'vpair'
        config['VPair'] = {
            'train_json': str(vpair_root / 'cross-area-drone2drone-train.json'),
            'test_json': str(vpair_root / 'cross-area-drone2drone-test.json'),
            'prefix': 'vpair'
        }
    
    # Merge datasets
    combined_train, combined_test = merge_datasets(config)
    
    # Save combined files
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    
    if combined_train:
        train_path = output_dir / 'combined-train.json'
        save_json(combined_train, train_path)
        print(f"Total train pairs: {len(combined_train)}")
    
    if combined_test:
        test_path = output_dir / 'combined-test.json'
        save_json(combined_test, test_path)
        print(f"Total test pairs: {len(combined_test)}")
    
    print(f"\nOutput directory: {output_dir}")
    print("\nTo use with training:")
    print(f"  data_root: {datasets_root}")
    print(f"  train_json: combined-train.json")
    print(f"  test_json: combined-test.json")


if __name__ == '__main__':
    main()
