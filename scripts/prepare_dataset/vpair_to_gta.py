# ---------------------------------------------------------------
# Copyright (c) 2024-2025 Yuxiang Ji. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

"""
VPair to GTA-UAV Format Converter

This script converts the VPair dataset (drone-to-drone visual place recognition)
to GTA-UAV compatible JSON format for training with the Game4Loc framework.

VPair Dataset Structure:
- poses.csv: Contains filename, lat, lon, altitude, roll, pitch, yaw
- queries/: 2,706 query drone images
- reference_views/: 5,412 reference drone images (gallery)
- distractors/: 10,000 distractor images (optional)

Output:
- vpair-drone2drone-train.json: Training pairs (80% of queries)
- vpair-drone2drone-test.json: Test pairs (20% of queries)
"""

import os
import pandas as pd
import json
import numpy as np
from tqdm import tqdm
from geopy.distance import geodesic
import argparse


def load_poses_csv(vpair_root):
    """
    Load poses.csv file containing GPS and orientation data.

    Returns:
        DataFrame with columns: filename, lat, lon, altitude, roll, pitch, yaw
    """
    poses_file = os.path.join(vpair_root, 'poses.csv')
    if not os.path.exists(poses_file):
        raise FileNotFoundError(f"poses.csv not found at {poses_file}")

    poses_df = pd.read_csv(poses_file)
    print(f"Loaded {len(poses_df)} poses from {poses_file}")
    print(f"Columns: {poses_df.columns.tolist()}")

    return poses_df


def compute_distance_matrix(queries_df, references_df):
    """
    Compute pairwise GPS distances between queries and reference images.

    Args:
        queries_df: DataFrame with query image poses
        references_df: DataFrame with reference image poses

    Returns:
        distance_matrix: NxM matrix where N=queries, M=references
    """
    n_queries = len(queries_df)
    n_refs = len(references_df)

    distance_matrix = np.zeros((n_queries, n_refs))

    print(f"\nComputing distance matrix ({n_queries} queries x {n_refs} references)...")

    for i, (_, query_row) in enumerate(tqdm(queries_df.iterrows(), total=n_queries)):
        query_loc = (query_row['lat'], query_row['lon'])

        for j, (_, ref_row) in enumerate(references_df.iterrows()):
            ref_loc = (ref_row['lat'], ref_row['lon'])
            distance = geodesic(query_loc, ref_loc).meters
            distance_matrix[i, j] = distance

    return distance_matrix


def distance_to_weight(distance, scale_factor=50.0):
    """
    Convert GPS distance to similarity weight using exponential decay.

    Args:
        distance: GPS distance in meters
        scale_factor: Controls decay rate (smaller = faster decay)

    Returns:
        weight: Similarity weight in range [0, 1]
    """
    weight = np.exp(-distance / scale_factor)
    return weight


def generate_pairs(queries_df, references_df, distance_matrix,
                   pos_threshold=25.0, semipos_threshold=100.0,
                   scale_factor=50.0):
    """
    Generate positive and semi-positive pairs based on GPS distance.

    Args:
        queries_df: DataFrame with query images
        references_df: DataFrame with reference images
        distance_matrix: NxM distance matrix
        pos_threshold: Distance threshold for positive pairs (meters)
        semipos_threshold: Distance threshold for semi-positive pairs (meters)
        scale_factor: Weight decay scale factor

    Returns:
        pairs_metadata: List of dicts in GTA-UAV format
    """
    pairs_metadata = []
    n_queries = len(queries_df)

    queries_list = queries_df.reset_index(drop=True)
    references_list = references_df.reset_index(drop=True)

    print(f"\nGenerating pairs (pos_threshold={pos_threshold}m, semipos_threshold={semipos_threshold}m)...")

    n_no_positives = 0
    n_no_semipositives = 0

    for i in tqdm(range(n_queries)):
        query_row = queries_list.iloc[i]
        query_filename = query_row['filename']
        query_lat = query_row['lat']
        query_lon = query_row['lon']
        query_altitude = query_row['altitude']

        # Get distances to all references
        distances = distance_matrix[i, :]

        # Find positive matches (distance < pos_threshold)
        pos_indices = np.where(distances < pos_threshold)[0]
        pos_distances = distances[pos_indices]

        # Find semi-positive matches (pos_threshold <= distance < semipos_threshold)
        semipos_indices = np.where((distances >= pos_threshold) & (distances < semipos_threshold))[0]
        semipos_distances = distances[semipos_indices]

        # Combined positive + semi-positive
        all_indices = np.where(distances < semipos_threshold)[0]
        all_distances = distances[all_indices]

        # Skip if no positive matches
        if len(pos_indices) == 0:
            n_no_positives += 1
            # Fall back to using closest reference as positive
            closest_idx = np.argmin(distances)
            pos_indices = np.array([closest_idx])
            pos_distances = np.array([distances[closest_idx]])

        # Skip if no semi-positive matches
        if len(all_indices) == 0:
            n_no_semipositives += 1
            # Use only positives
            all_indices = pos_indices
            all_distances = pos_distances

        # Convert distances to weights
        pos_weights = [distance_to_weight(d, scale_factor) for d in pos_distances]
        all_weights = [distance_to_weight(d, scale_factor) for d in all_distances]

        # Get reference filenames
        pos_ref_names = [references_list.iloc[idx]['filename'] for idx in pos_indices]
        all_ref_names = [references_list.iloc[idx]['filename'] for idx in all_indices]

        # Get reference locations (for evaluation)
        pos_ref_locs = [[references_list.iloc[idx]['lat'],
                        references_list.iloc[idx]['lon']] for idx in pos_indices]
        all_ref_locs = [[references_list.iloc[idx]['lat'],
                        references_list.iloc[idx]['lon']] for idx in all_indices]

        # Create metadata entry in GTA-UAV format
        # Note: Using drone_loc_x_y key for compatibility with train_gta.py
        # Values are lat/lon but key name matches GTA-UAV format
        pair_entry = {
            "drone_img_dir": "queries",
            "drone_img_name": query_filename,
            "drone_loc_x_y": [query_lat, query_lon],  # GTA-UAV compatible key
            "sate_img_dir": "reference_views",
            "pair_pos_sate_img_list": pos_ref_names,
            "pair_pos_sate_weight_list": pos_weights,
            "pair_pos_sate_loc_x_y_list": pos_ref_locs,  # GTA-UAV compatible key
            "pair_pos_semipos_sate_img_list": all_ref_names,
            "pair_pos_semipos_sate_weight_list": all_weights,
            "pair_pos_semipos_sate_loc_x_y_list": all_ref_locs,  # GTA-UAV compatible key
            "drone_metadata": {
                "height": query_altitude,
                "drone_roll": query_row.get('roll', None),
                "drone_pitch": query_row.get('pitch', None),
                "drone_yaw": query_row.get('yaw', None),
                "cam_roll": None,
                "cam_pitch": None,
                "cam_yaw": None
            }
        }

        pairs_metadata.append(pair_entry)

    print(f"\nGenerated {len(pairs_metadata)} query pairs")
    print(f"Queries with no positives (< {pos_threshold}m): {n_no_positives}")
    print(f"Queries with no semi-positives (< {semipos_threshold}m): {n_no_semipositives}")

    # Compute statistics
    n_positives = sum([len(p['pair_pos_sate_img_list']) for p in pairs_metadata])
    n_semipositives = sum([len(p['pair_pos_semipos_sate_img_list']) for p in pairs_metadata])

    print(f"Average positives per query: {n_positives / len(pairs_metadata):.2f}")
    print(f"Average semi-positives per query: {n_semipositives / len(pairs_metadata):.2f}")

    return pairs_metadata


def split_train_test(pairs_metadata, split_type='cross-area', train_ratio=0.8, random_seed=42):
    """
    Split pairs into train and test sets.

    Following GTA-UAV's cross-area philosophy: train and test on different geographic regions
    to ensure model generalization to unseen locations.

    Args:
        pairs_metadata: List of pair dicts
        split_type: 'cross-area' (geographic split) or 'same-area' (random split)
        train_ratio: Fraction for training (default: 0.8), only used for same-area
        random_seed: Random seed for reproducibility

    Returns:
        train_pairs, test_pairs
    """
    if split_type == 'cross-area':
        # Geographic split: North vs South (following GTA-UAV's cross-area concept)
        # This ensures train and test are in different locations, testing generalization
        print("\nPerforming CROSS-AREA split (geographic regions)...")
        print("  Following GTA-UAV philosophy: different locations for train/test")

        # Get median latitude to split north/south
        all_lats = [p['drone_loc_x_y'][0] for p in pairs_metadata]
        median_lat = np.median(all_lats)

        print(f"  Median latitude: {median_lat:.6f}")
        print(f"  Train region: South (lat < {median_lat:.6f})")
        print(f"  Test region:  North (lat >= {median_lat:.6f})")

        # Split based on latitude (different geographic areas)
        train_pairs = [p for p in pairs_metadata if p['drone_loc_x_y'][0] < median_lat]
        test_pairs = [p for p in pairs_metadata if p['drone_loc_x_y'][0] >= median_lat]

    else:  # same-area
        # Random split (train and test from same region)
        print("\nPerforming SAME-AREA split (random)...")
        np.random.seed(random_seed)
        n_total = len(pairs_metadata)
        n_train = int(n_total * train_ratio)

        # Shuffle indices
        indices = np.random.permutation(n_total)
        train_indices = indices[:n_train]
        test_indices = indices[n_train:]

        train_pairs = [pairs_metadata[i] for i in train_indices]
        test_pairs = [pairs_metadata[i] for i in test_indices]

    print(f"\nSplit complete: {len(train_pairs)} train, {len(test_pairs)} test")

    # Verify no overlap in geographic regions for cross-area
    if split_type == 'cross-area':
        train_lats = [p['drone_loc_x_y'][0] for p in train_pairs]
        test_lats = [p['drone_loc_x_y'][0] for p in test_pairs]
        print(f"  Train lat range: {min(train_lats):.6f} to {max(train_lats):.6f}")
        print(f"  Test lat range:  {min(test_lats):.6f} to {max(test_lats):.6f}")

    return train_pairs, test_pairs


def save_json(pairs_metadata, output_path):
    """
    Save pairs metadata to JSON file.
    """
    with open(output_path, 'w') as f:
        json.dump(pairs_metadata, f, indent=2)

    print(f"Saved {len(pairs_metadata)} pairs to {output_path}")


def main(args):
    """
    Main preprocessing pipeline.
    """
    vpair_root = args.vpair_root

    print(f"VPair root directory: {vpair_root}")
    print(f"=" * 70)

    # 1. Load poses.csv
    poses_df = load_poses_csv(vpair_root)

    # 2. Identify queries and references
    # All images in poses.csv can be queries or references
    # VPair typically uses images from poses.csv as both queries and references
    # For this conversion, we'll use all images as references and a subset as queries

    # Check which images exist in queries/ and reference_views/
    queries_dir = os.path.join(vpair_root, 'queries')
    references_dir = os.path.join(vpair_root, 'reference_views')

    if not os.path.exists(queries_dir):
        raise FileNotFoundError(f"queries directory not found at {queries_dir}")
    if not os.path.exists(references_dir):
        raise FileNotFoundError(f"reference_views directory not found at {references_dir}")

    # Get list of actual files
    query_files = set(os.listdir(queries_dir))
    reference_files = set(os.listdir(references_dir))

    print(f"\nFound {len(query_files)} files in queries/")
    print(f"Found {len(reference_files)} files in reference_views/")

    # Filter poses_df by actual files
    queries_df = poses_df[poses_df['filename'].isin(query_files)].copy()
    references_df = poses_df[poses_df['filename'].isin(reference_files)].copy()

    print(f"\nMatched {len(queries_df)} queries with poses")
    print(f"Matched {len(references_df)} references with poses")

    if len(queries_df) == 0:
        raise ValueError("No queries matched with poses.csv. Check filename formats.")
    if len(references_df) == 0:
        raise ValueError("No references matched with poses.csv. Check filename formats.")

    # 3. Compute distance matrix
    distance_matrix = compute_distance_matrix(queries_df, references_df)

    # Print distance statistics
    print(f"\nDistance statistics:")
    print(f"  Min distance: {np.min(distance_matrix):.2f}m")
    print(f"  Max distance: {np.max(distance_matrix):.2f}m")
    print(f"  Mean distance: {np.mean(distance_matrix):.2f}m")
    print(f"  Median distance: {np.median(distance_matrix):.2f}m")

    # 4. Generate pairs
    pairs_metadata = generate_pairs(
        queries_df,
        references_df,
        distance_matrix,
        pos_threshold=args.pos_threshold,
        semipos_threshold=args.semipos_threshold,
        scale_factor=args.scale_factor
    )

    # 5. Split train/test (following GTA-UAV's cross-area methodology)
    train_pairs, test_pairs = split_train_test(
        pairs_metadata,
        split_type=args.split_type,
        train_ratio=args.train_ratio
    )

    # 6. Save JSON files
    train_output = os.path.join(vpair_root, args.train_output)
    test_output = os.path.join(vpair_root, args.test_output)

    save_json(train_pairs, train_output)
    save_json(test_pairs, test_output)

    print(f"\n{'=' * 70}")
    print(f"Preprocessing completed successfully!")
    print(f"Training file: {train_output}")
    print(f"Test file: {test_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert VPair dataset to GTA-UAV format")

    parser.add_argument('--vpair_root', type=str,
                       default='/home/aniel/skyline_drone/datasets/vpair',
                       help='Path to vpair dataset root directory')

    parser.add_argument('--pos_threshold', type=float, default=25.0,
                       help='Distance threshold for positive pairs (meters)')

    parser.add_argument('--semipos_threshold', type=float, default=100.0,
                       help='Distance threshold for semi-positive pairs (meters)')

    parser.add_argument('--scale_factor', type=float, default=50.0,
                       help='Weight decay scale factor (smaller = faster decay)')

    parser.add_argument('--split_type', type=str, default='cross-area',
                       choices=['cross-area', 'same-area'],
                       help='Split type following GTA-UAV: cross-area (different regions) or same-area (random)')

    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='Fraction of data for training (default: 0.8, only used for same-area)')

    parser.add_argument('--train_output', type=str,
                       default='cross-area-drone2drone-train.json',
                       help='Output filename for training JSON')

    parser.add_argument('--test_output', type=str,
                       default='cross-area-drone2drone-test.json',
                       help='Output filename for test JSON')

    args = parser.parse_args()

    main(args)
