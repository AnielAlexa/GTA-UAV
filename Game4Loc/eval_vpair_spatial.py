#!/usr/bin/env python3
"""
VPAIR Spatial Recall Evaluation (AnyLoc Protocol)

Evaluates visual place recognition using spatial proximity matching
instead of exact image ID matching, following the AnyLoc/VPAIR paper protocol.

Usage:
    python eval_vpair_spatial.py \
        --data_root /path/to/vpair \
        --test_pairs_meta_file cross-area-drone2drone-test.json \
        --checkpoint_start /path/to/checkpoint.pth \
        --radius 25.0 \
        --gpu_ids 0
"""

import os
import sys
import torch
import argparse
import numpy as np
from dataclasses import dataclass
from torch.utils.data import DataLoader
from tqdm import tqdm
from geopy.distance import geodesic

from game4loc.dataset.gta import GTADatasetEval, get_transforms
from game4loc.models.model import DesModel


def parse_tuple(s):
    try:
        return tuple(map(int, s.split(',')))
    except ValueError:
        raise argparse.ArgumentTypeError("Tuple must be integers separated by commas")


@dataclass
class Configuration:
    # Model
    model: str = 'vit_small_patch16_dinov3.lvd1689m'
    img_size: int = 384

    # Evaluation
    batch_size: int = 128
    verbose: bool = True
    gpu_ids: tuple = (0)
    normalize_features: bool = True

    # Spatial recall parameters
    radius_thresholds: list = None  # [10, 25, 50, 100] meters

    # num_workers
    num_workers: int = 0 if os.name == 'nt' else 4

    # device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Dataset
    query_mode: str = 'D2S'  # Actually D2D for VPAIR

    # Paths
    checkpoint_start: str = None
    data_root: str = "/path/to/vpair"
    test_pairs_meta_file: str = 'cross-area-drone2drone-test.json'
    sate_img_dir: str = 'reference_views'

    def __post_init__(self):
        if self.radius_thresholds is None:
            self.radius_thresholds = [10, 25, 50, 100]


def extract_features(model, dataloader, device, verbose=True):
    """Extract features from images."""
    model.eval()
    all_features = []

    with torch.no_grad():
        iterator = tqdm(dataloader, desc="Extracting features") if verbose else dataloader
        for batch in iterator:
            images = batch.to(device)
            features = model(images)

            # Normalize features (important for cosine similarity)
            features = torch.nn.functional.normalize(features, p=2, dim=1)
            all_features.append(features.cpu())

    return torch.cat(all_features, dim=0)


def calculate_spatial_recall(query_locs, gallery_locs, similarity_matrix,
                             radius_threshold=25.0, k_values=[1, 5, 10, 20]):
    """
    Calculate recall using spatial proximity (AnyLoc protocol).

    A retrieval is considered correct if ANY of the top-K retrieved images
    are within the radius threshold of the query location.

    Args:
        query_locs: (N, 2) array of query GPS coordinates [lat, lon]
        gallery_locs: (M, 2) array of gallery GPS coordinates [lat, lon]
        similarity_matrix: (N, M) similarity scores
        radius_threshold: Distance threshold in meters (default: 25m)
        k_values: List of K values to evaluate

    Returns:
        results: Dict with Recall@K for each K value
    """
    n_queries = len(query_locs)

    # Get top-K indices for each query
    max_k = max(k_values)
    top_k_indices = torch.topk(similarity_matrix, k=max_k, dim=1)[1].numpy()

    results = {}

    for k in k_values:
        correct = 0
        distances_to_closest = []

        for i in range(n_queries):
            query_loc = query_locs[i]  # [lat, lon]
            top_k_idx = top_k_indices[i, :k]

            # Get locations of top-K retrieved images
            retrieved_locs = gallery_locs[top_k_idx]

            # Calculate distances to all top-K candidates
            distances = []
            for retrieved_loc in retrieved_locs:
                dist = geodesic(query_loc, retrieved_loc).meters
                distances.append(dist)

            # Find minimum distance among top-K
            min_distance = min(distances)
            distances_to_closest.append(min_distance)

            # Check if ANY of top-K is within radius
            if min_distance <= radius_threshold:
                correct += 1

        recall = correct / n_queries
        avg_distance = np.mean(distances_to_closest)
        median_distance = np.median(distances_to_closest)

        results[k] = {
            'recall': recall * 100,  # Convert to percentage
            'avg_distance': avg_distance,
            'median_distance': median_distance
        }

    return results


def evaluate_spatial(config):
    """Main evaluation function with spatial recall."""

    print("\n" + "="*80)
    print("VPAIR SPATIAL RECALL EVALUATION (AnyLoc Protocol)")
    print("="*80)
    print(f"\nModel: {config.model}")
    print(f"Image Size: {config.img_size}×{config.img_size}")
    print(f"Checkpoint: {config.checkpoint_start}")
    print(f"Data Root: {config.data_root}")
    print(f"Radius Thresholds: {config.radius_thresholds} meters")

    # Load model
    model = DesModel(
        config.model,
        pretrained=True,
        img_size=config.img_size,
        share_weights=True
    )

    if config.checkpoint_start and os.path.exists(config.checkpoint_start):
        print(f"\nLoading checkpoint: {config.checkpoint_start}")
        checkpoint = torch.load(config.checkpoint_start, map_location='cpu')

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        # Remove 'module.' prefix if present (DataParallel)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v

        model.load_state_dict(new_state_dict, strict=False)
        print("✓ Checkpoint loaded successfully")

    device = torch.device(config.device)
    model = model.to(device)
    model.eval()

    # Get transforms
    val_transforms, _, _ = get_transforms((config.img_size, config.img_size))

    # Load datasets
    print("\nLoading VPAIR datasets...")

    query_dataset = GTADatasetEval(
        data_root=config.data_root,
        pairs_meta_file=config.test_pairs_meta_file,
        view="drone",
        transforms=val_transforms,
        mode='pos',
        query_mode=config.query_mode,
    )

    gallery_dataset = GTADatasetEval(
        data_root=config.data_root,
        pairs_meta_file=config.test_pairs_meta_file,
        view="sate",
        transforms=val_transforms,
        sate_img_dir=config.sate_img_dir,
        mode='pos',
        query_mode=config.query_mode,
    )

    query_loader = DataLoader(
        query_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=False,
        pin_memory=True
    )

    gallery_loader = DataLoader(
        gallery_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=False,
        pin_memory=True
    )

    print(f"Query Images: {len(query_dataset)}")
    print(f"Gallery Images (before filtering): {len(gallery_dataset)}")

    # Get GPS coordinates
    query_locs = np.array(query_dataset.images_center_loc_xy)  # (N, 2) [lat, lon]
    gallery_locs = np.array(gallery_dataset.images_center_loc_xy)  # (M, 2) [lat, lon]

    # Filter out gallery images with invalid coordinates (0, 0)
    # This happens when the dataset loader can't find location data in the JSON
    valid_gallery_mask = ~((gallery_locs[:, 0] == 0.0) | (gallery_locs[:, 1] == 0.0))
    valid_gallery_indices = np.where(valid_gallery_mask)[0]
    
    print(f"\nFiltering gallery images with invalid (0,0) coordinates...")
    print(f"  Images with valid coordinates: {valid_gallery_mask.sum()}/{len(gallery_dataset)}")
    print(f"  Images with (0,0) coordinates: {(~valid_gallery_mask).sum()}")
    
    if (~valid_gallery_mask).sum() > 0:
        print(f"  Examples of filtered images: {[gallery_dataset.images_name[i] for i in np.where(~valid_gallery_mask)[0][:5]]}")

    # Keep only valid gallery entries
    gallery_locs = gallery_locs[valid_gallery_mask]
    
    print(f"\nQuery locations shape: {query_locs.shape}")
    print(f"Gallery locations shape (after filtering): {gallery_locs.shape}")

    # Extract features
    print("\n" + "-"*80)
    print("FEATURE EXTRACTION")
    print("-"*80)

    print("\nExtracting query features...")
    query_features = extract_features(model, query_loader, device, verbose=True)

    print("\nExtracting gallery features...")
    gallery_features_all = extract_features(model, gallery_loader, device, verbose=True)
    
    # Filter gallery features to match valid coordinates
    gallery_features = gallery_features_all[valid_gallery_indices]

    print(f"\nQuery features shape: {query_features.shape}")
    print(f"Gallery features shape (after filtering): {gallery_features.shape}")

    # Compute similarity matrix (cosine similarity)
    print("\nComputing similarity matrix...")
    query_features = query_features.to(device)
    gallery_features = gallery_features.to(device)

    similarity_matrix = torch.mm(query_features, gallery_features.t())  # (N, M)
    similarity_matrix = similarity_matrix.cpu()

    print(f"Similarity matrix shape: {similarity_matrix.shape}")

    # Evaluate with different radius thresholds
    print("\n" + "="*80)
    print("SPATIAL RECALL RESULTS (AnyLoc Protocol)")
    print("="*80)

    for radius in config.radius_thresholds:
        print(f"\n{'─'*80}")
        print(f"Localization Radius: {radius}m")
        print(f"{'─'*80}")

        results = calculate_spatial_recall(
            query_locs=query_locs,
            gallery_locs=gallery_locs,
            similarity_matrix=similarity_matrix,
            radius_threshold=radius,
            k_values=[1, 5, 10, 20]
        )

        # Print results in table format
        print(f"\n{'Metric':<20} {'Value':<15} {'Avg Dist':<15} {'Median Dist':<15}")
        print("-" * 65)

        for k in [1, 5, 10, 20]:
            recall = results[k]['recall']
            avg_dist = results[k]['avg_distance']
            median_dist = results[k]['median_distance']
            print(f"Recall@{k:<15} {recall:>6.2f}%        {avg_dist:>6.2f}m        {median_dist:>6.2f}m")

    # Summary comparison to AnyLoc
    print("\n" + "="*80)
    print("COMPARISON TO STATE-OF-THE-ART")
    print("="*80)

    # Standard VPAIR evaluation uses 25m radius
    radius_25m_results = calculate_spatial_recall(
        query_locs=query_locs,
        gallery_locs=gallery_locs,
        similarity_matrix=similarity_matrix,
        radius_threshold=25.0,
        k_values=[1, 5, 10]
    )

    print(f"\n{'Method':<30} {'Backbone':<25} {'R@1 (25m)':<15} {'R@5 (25m)':<15}")
    print("-" * 85)
    print(f"{'AnyLoc-VLAD (SOTA)':<30} {'DINOv2 ViT-L':<25} {'43.0%':<15} {'66-70%':<15}")
    print(f"{'Your Model':<30} {config.model:<25} {radius_25m_results[1]['recall']:<14.2f}% {radius_25m_results[5]['recall']:<14.2f}%")
    print(f"{'Baseline VPR':<30} {'Various':<25} {'19.5%':<15} {'53%':<15}")

    print("\n" + "="*80)

    # Performance tier
    r1_25m = radius_25m_results[1]['recall']
    if r1_25m >= 40:
        tier = "🏆 STATE-OF-THE-ART"
    elif r1_25m >= 30:
        tier = "✅ COMPETITIVE"
    elif r1_25m >= 20:
        tier = "⚫ GOOD"
    else:
        tier = "⚠️ BELOW BASELINE"

    print(f"\nPerformance Tier: {tier}")
    print(f"Your R@1 (25m radius): {r1_25m:.2f}%")
    print(f"AnyLoc SOTA: 43.0%")
    print(f"Relative Performance: {(r1_25m/43.0)*100:.1f}% of SOTA")

    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="VPAIR Spatial Recall Evaluation (AnyLoc Protocol)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--data_root', type=str,
                       default='/media/aniel/storage/vpair',
                       help='VPAIR dataset root directory')

    parser.add_argument('--test_pairs_meta_file', type=str,
                       default='cross-area-drone2drone-test.json',
                       help='Test metafile path')

    parser.add_argument('--model', type=str,
                       default='vit_small_patch16_dinov3.lvd1689m',
                       help='Model architecture')

    parser.add_argument('--img_size', type=int, default=384,
                       help='Input image size')

    parser.add_argument('--gpu_ids', type=parse_tuple, default=(0,),
                       help='GPU IDs')

    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size')

    parser.add_argument('--checkpoint_start', type=str, required=True,
                       help='Path to model checkpoint')

    parser.add_argument('--sate_img_dir', type=str, default='reference_views',
                       help='Gallery image directory name')

    parser.add_argument('--radius', type=float, nargs='+',
                       default=[10, 25, 50, 100],
                       help='Localization radius thresholds in meters')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    config = Configuration()
    config.data_root = args.data_root
    config.test_pairs_meta_file = args.test_pairs_meta_file
    config.batch_size = args.batch_size
    config.gpu_ids = args.gpu_ids
    config.checkpoint_start = args.checkpoint_start
    config.model = args.model
    config.img_size = args.img_size
    config.sate_img_dir = args.sate_img_dir
    config.radius_thresholds = args.radius

    evaluate_spatial(config)
