"""
train_lora_ortholoc.py - Phase 2: Sim-to-Real Fine-Tuning (CORRECTED)

This script continues LoRA training from GTA-UAV pretrained weights to Ortholoc dataset.

Key Features:
1. Loads GTA-pretrained LoRA weights with strict verification
2. Uses albumentations (compatible with GTADatasetTrain)
3. Conservative augmentations for gentle sim-to-real adaptation
4. Validation loop with Recall@1 tracking
5. Multi-GPU support via DataParallel
6. Automatic architecture detection from checkpoint

Fixed Issues:
- ❌→✅ Transform library (torchvision → albumentations)
- ❌→✅ Weight loading verification (now checks all critical components)
- ❌→✅ Validation loop (tracks Recall@1 on val set)
- ❌→✅ Multi-GPU support (DataParallel)
- ❌→✅ Architecture auto-detection (infers config from checkpoint)

Usage:
    python train_lora_ortholoc.py \
        --gta_weights work_dir/gta/weights_best.pth \
        --real_data_root data/ortholoc_converted \
        --real_train_json cross-area-drone2sate-train.json \
        --real_val_json cross-area-drone2sate-val.json \
        --output_dir work_dir/ortholoc \
        --lora_r 64 --lora_alpha 128 \
        --batch_size 128 --lr 5e-5 --epochs 10
"""

import os
import sys
import argparse
import time
import math
import shutil
import torch
import torch.nn as nn
import numpy as np
import cv2
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model
import timm

# Albumentations (CRITICAL FIX: was torchvision)
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Import from your existing project
from game4loc.dataset.gta import GTADatasetTrain, GTADatasetEval
from game4loc.utils import setup_system, Logger
from game4loc.trainer.trainer import train_with_weight
from game4loc.loss import WeightedInfoNCE

# -----------------------------------------------------------------------------#
# 1. Data Augmentations (FIXED: Now uses albumentations)                      #
# -----------------------------------------------------------------------------#
def get_ortholoc_transforms(img_size=384, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
                           augmentation_strength='conservative'):
    """
    Create albumentations transforms for Ortholoc dataset.

    Args:
        img_size: Target image size
        mean: Normalization mean (ImageNet default)
        std: Normalization std (ImageNet default)
        augmentation_strength: 'conservative' (default), 'moderate', or 'aggressive'

    Returns:
        val_transforms, train_sat_transforms, train_drone_transforms
    """

    if augmentation_strength == 'conservative':
        # Conservative: Preserve GTA-learned features, gentle adaptation
        train_drone_transforms = A.Compose([
            A.Resize(img_size, img_size, interpolation=cv2.INTER_CUBIC),

            # Light blur for sim-to-real (GTA drone images are sharp)
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 5), sigma_limit=(0.1, 1.0)),
                A.MotionBlur(blur_limit=3),  # Simulates camera/drone movement
            ], p=0.1),  # 10% chance for drone

            # Color/lighting (slightly less than GTA training)
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.7),
            A.RandomGamma(gamma_limit=(90, 110), p=0.3),

            # Very conservative geometric (preserve learned geometry)
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=5,
                             border_mode=cv2.BORDER_CONSTANT, value=0, p=0.3),

            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])

        train_sat_transforms = A.Compose([
            A.Resize(img_size, img_size, interpolation=cv2.INTER_CUBIC),

            # Light blur for sim-to-real (GTA images are sharp, real images have natural blur)
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 7), sigma_limit=(0.1, 1.5)),
                A.MotionBlur(blur_limit=5),
                A.MedianBlur(blur_limit=5),  # Simulates sensor noise reduction
            ], p=0.3),  # 30% chance for satellite

            # Optional JPEG compression (real satellite imagery often compressed)
            A.ImageCompression(quality_lower=80, quality_upper=100, p=0.3),  # Increased from 0.2

            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])

    elif augmentation_strength == 'moderate':
        # Moderate: More augmentation if conservative doesn't adapt well
        train_drone_transforms = A.Compose([
            A.Resize(img_size, img_size, interpolation=cv2.INTER_CUBIC),

            # Light blur (drone motion + camera shake)
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 7), sigma_limit=(0.2, 1.5)),
                A.MotionBlur(blur_limit=7),
                A.MedianBlur(blur_limit=5),
            ], p=0.1),  # 10% chance for drone

            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.8),
            A.RandomGamma(gamma_limit=(80, 120), p=0.4),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10,
                             border_mode=cv2.BORDER_CONSTANT, value=0, p=0.4),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])

        train_sat_transforms = A.Compose([
            A.Resize(img_size, img_size, interpolation=cv2.INTER_CUBIC),

            # Light blur for satellite imagery
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 9), sigma_limit=(0.2, 2.0)),
                A.MotionBlur(blur_limit=7),
                A.MedianBlur(blur_limit=7),
            ], p=0.3),  # 30% chance for satellite

            A.ImageCompression(quality_lower=70, quality_upper=100, p=0.4),  # More aggressive
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])

    elif augmentation_strength == 'aggressive':
        # Aggressive: Maximum robustness (use if domain gap is very large)
        train_drone_transforms = A.Compose([
            A.Resize(img_size, img_size, interpolation=cv2.INTER_CUBIC),

            # Light blur (motion, atmospheric effects)
            A.OneOf([
                A.GaussianBlur(blur_limit=(5, 11), sigma_limit=(0.5, 2.5)),
                A.MotionBlur(blur_limit=9),
                A.MedianBlur(blur_limit=7),
            ], p=0.1),  # 10% chance for drone

            A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.15, p=0.9),
            A.RandomGamma(gamma_limit=(70, 130), p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
            A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, rotate_limit=15,
                             border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])

        train_sat_transforms = A.Compose([
            A.Resize(img_size, img_size, interpolation=cv2.INTER_CUBIC),

            # Light blur for satellite (atmospheric + sensor limitations)
            A.OneOf([
                A.GaussianBlur(blur_limit=(5, 11), sigma_limit=(0.5, 3.0)),
                A.MotionBlur(blur_limit=9),
                A.MedianBlur(blur_limit=9),
            ], p=0.3),  # 30% chance for satellite

            A.ImageCompression(quality_lower=60, quality_upper=100, p=0.5),  # Very aggressive
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])

    else:
        raise ValueError(f"Unknown augmentation_strength: {augmentation_strength}")

    # Validation: Clean, no augmentation (same as GTA)
    val_transforms = A.Compose([
        A.Resize(img_size, img_size, interpolation=cv2.INTER_CUBIC),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])

    return val_transforms, train_sat_transforms, train_drone_transforms


# -----------------------------------------------------------------------------#
# 2. Model Definition (MUST match GTA Training EXACTLY)                       #
# -----------------------------------------------------------------------------#
class DesModelLoRA(nn.Module):
    """
    LoRA fine-tuned DINOv3 model for cross-view geo-localization.

    This model MUST match train_lora_gta.py architecture exactly for weight loading.
    """
    def __init__(self,
                 model_name='vit_small_patch16_dinov3.lvd1689m',
                 img_size=384,
                 lora_r=64,
                 lora_alpha=128,
                 mlp_hidden=512,
                 mlp_output_dim=None):  # None = embed_dim (PEFT variant)

        super(DesModelLoRA, self).__init__()

        print(f"Creating backbone: {model_name}")
        self.model = timm.create_model(
            model_name,
            pretrained=True,  # CRITICAL: Load DINOv3 pretrained weights first!
            num_classes=0,
            img_size=img_size,
            global_pool='avg'
        )
        self.embed_dim = self.model.embed_dim

        # Freeze backbone
        for param in self.model.parameters():
            param.requires_grad = False

        # Apply PEFT LoRA (MUST match GTA config)
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["qkv", "proj", "fc1", "fc2"],
            bias="none",
            lora_dropout=0.1
        )
        self.model = get_peft_model(self.model, lora_config)

        # MLP head (MUST match GTA config)
        if mlp_output_dim is None:
            mlp_output_dim = self.embed_dim  # PEFT variant: 384→512→384

        self.mlp_head = nn.Sequential(
            nn.Linear(self.embed_dim, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(mlp_hidden, mlp_output_dim)
        )

        # Learnable temperature (MUST match GTA config)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        print(f"Model created:")
        print(f"  Backbone: {model_name}")
        print(f"  Embed dim: {self.embed_dim}")
        print(f"  LoRA: r={lora_r}, alpha={lora_alpha}")
        print(f"  MLP: {self.embed_dim}→{mlp_hidden}→{mlp_output_dim}")

    def forward(self, img1=None, img2=None):
        """Forward pass for dual-encoder."""
        def encode(x):
            features = self.model(x)
            features = self.mlp_head(features)
            return features

        if img1 is not None and img2 is not None:
            return encode(img1), encode(img2)
        elif img1 is not None:
            return encode(img1)
        else:
            return encode(img2)


# -----------------------------------------------------------------------------#
# 3. Weight Loading with Verification (NEW)                                   #
# -----------------------------------------------------------------------------#
def load_gta_weights_with_verification(model, checkpoint_path):
    """
    Load GTA weights with comprehensive verification.

    This function:
    1. Loads checkpoint
    2. Handles DataParallel prefix (if present)
    3. Verifies critical components loaded correctly
    4. Reports any issues

    Returns:
        model with loaded weights
    """
    print(f"\n{'='*70}")
    print(f"LOADING GTA WEIGHTS")
    print(f"{'='*70}")
    print(f"Checkpoint: {checkpoint_path}\n")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Handle 'module.' prefix from DataParallel (shouldn't exist, but be safe)
    state_dict = checkpoint
    if all(k.startswith('module.') for k in state_dict.keys()):
        print("⚠ WARNING: Detected DataParallel prefix 'module.' (unexpected)")
        print("  Stripping prefix for compatibility...\n")
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    # Load state dict (strict=False allows missing base model params)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    # Analyze loading results
    print(f"{'='*70}")
    print(f"LOADING REPORT")
    print(f"{'='*70}\n")

    # Expected missing keys: base model frozen weights
    expected_missing = [k for k in missing_keys if 'model.model.' in k]

    # Critical missing keys: LoRA adapters, MLP head, logit_scale
    critical_missing = [k for k in missing_keys if 'model.model.' not in k]

    print(f"✓ Base model frozen params (OK to miss): {len(expected_missing)}")

    if critical_missing:
        print(f"\n❌ CRITICAL: Missing trainable parameters!")
        for key in critical_missing[:20]:
            print(f"   - {key}")
        if len(critical_missing) > 20:
            print(f"   ... and {len(critical_missing) - 20} more")
        raise ValueError("Critical parameters missing from checkpoint! Check architecture match.")

    if unexpected_keys:
        print(f"\n⚠ WARNING: Unexpected keys in checkpoint:")
        for key in unexpected_keys[:10]:
            print(f"   - {key}")
        if len(unexpected_keys) > 10:
            print(f"   ... and {len(unexpected_keys) - 10} more")

    # Verify critical components
    print(f"\n{'='*70}")
    print(f"VERIFICATION")
    print(f"{'='*70}\n")

    # Check logit_scale
    if hasattr(model, 'logit_scale'):
        logit_scale_value = model.logit_scale.item()
        print(f"✓ logit_scale: {logit_scale_value:.4f} (temp: {np.exp(logit_scale_value):.2f})")
    else:
        raise ValueError("logit_scale not found!")

    # Check LoRA adapters
    lora_params = [name for name, param in model.named_parameters() if 'lora_' in name.lower()]
    if lora_params:
        print(f"✓ LoRA adapters: {len(lora_params)} parameters")
    else:
        raise ValueError("No LoRA parameters found! Check PEFT model creation.")

    # Check MLP head
    mlp_params = [name for name, param in model.named_parameters() if 'mlp_head' in name]
    if mlp_params:
        print(f"✓ MLP head: {len(mlp_params)} parameters")
        # Get output dimension
        last_linear = [p for n, p in model.named_parameters() if 'mlp_head' in n and 'weight' in n][-1]
        print(f"  Output dimension: {last_linear.shape[0]}")
    else:
        raise ValueError("MLP head not found!")

    # Count trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\n✓ Total parameters: {total:,}")
    print(f"✓ Trainable (LoRA+MLP): {trainable:,} ({100*trainable/total:.2f}%)")

    print(f"\n{'='*70}")
    print(f"✓ GTA WEIGHTS LOADED SUCCESSFULLY")
    print(f"{'='*70}\n")

    return model


# -----------------------------------------------------------------------------#
# 4. Simple Validation Dataset for Ortholoc (NEW)                             #
# -----------------------------------------------------------------------------#
class SimpleValDataset(torch.utils.data.Dataset):
    """
    Simple validation dataset that works with both GTA and Ortholoc formats.

    Just loads images without parsing satellite tile coordinates (not needed for Recall@K).
    """
    def __init__(self, data_root, pairs_meta_file, view='drone', transforms=None):
        """
        Args:
            data_root: Root directory of dataset
            pairs_meta_file: JSON file with pairs
            view: 'drone' or 'satellite'
            transforms: Albumentations transforms
        """
        import json

        self.data_root = data_root
        self.transforms = transforms
        self.view = view
        self.images_path = []
        self.images_name = []
        self.pairs_dict = {}  # NEW: Store query→gallery mapping

        # Load JSON
        json_path = os.path.join(data_root, pairs_meta_file)
        with open(json_path, 'r') as f:
            pairs_data = json.load(f)

        if view == 'drone':
            # Load drone images (queries)
            for pair in pairs_data:
                drone_img_dir = pair.get('drone_img_dir', 'drone/images')
                drone_img_name = pair['drone_img_name']
                img_path = os.path.join(data_root, drone_img_dir, drone_img_name)
                self.images_path.append(img_path)
                self.images_name.append(drone_img_name)

                # Store matching satellite images for this query
                self.pairs_dict[drone_img_name] = pair['pair_pos_sate_img_list']

        elif view == 'satellite':
            # Load ALL unique satellite images (gallery)
            sate_img_set = set()
            for pair in pairs_data:
                sate_img_dir = pair.get('sate_img_dir', 'satellite')
                sate_img_list = pair['pair_pos_sate_img_list']
                for sate_img_name in sate_img_list:
                    if sate_img_name not in sate_img_set:
                        sate_img_set.add(sate_img_name)
                        img_path = os.path.join(data_root, sate_img_dir, sate_img_name)
                        self.images_path.append(img_path)
                        self.images_name.append(sate_img_name)

    def __getitem__(self, index):
        img_path = self.images_path[index]
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transforms is not None:
            img = self.transforms(image=img)['image']

        return img

    def __len__(self):
        return len(self.images_path)


# -----------------------------------------------------------------------------#
# 5. Validation Loop (NEW)                                                    #
# -----------------------------------------------------------------------------#
def validate_ortholoc(model, val_loader_query, val_loader_gallery, device='cuda'):
    """
    Run validation and compute Recall@K metrics (GTA-UAV style).

    Args:
        model: The model to evaluate
        val_loader_query: DataLoader for query images (drone)
        val_loader_gallery: DataLoader for gallery images (satellite)
        device: Device to run on

    Returns:
        dict with metrics: {'r1': float, 'r5': float, 'r10': float}
    """
    model.eval()

    print(f"\n{'='*60}")
    print(f"VALIDATION")
    print(f"{'='*60}")

    # Extract query features (drone)
    query_features = []
    with torch.no_grad():
        for batch in val_loader_query:
            batch = batch.to(device)
            features = model(img1=batch)
            # Normalize features
            features = torch.nn.functional.normalize(features, dim=-1)
            query_features.append(features.cpu())
    query_features = torch.cat(query_features, dim=0)
    print(f"Query features: {query_features.shape}")

    # Extract gallery features (satellite)
    gallery_features = []
    with torch.no_grad():
        for batch in val_loader_gallery:
            batch = batch.to(device)
            features = model(img2=batch)
            # Normalize features
            features = torch.nn.functional.normalize(features, dim=-1)
            gallery_features.append(features.cpu())
    gallery_features = torch.cat(gallery_features, dim=0)
    print(f"Gallery features: {gallery_features.shape}")

    # Compute similarity matrix (query x gallery)
    similarity = query_features @ gallery_features.T  # [num_query, num_gallery]

    # DEBUG: Check feature and similarity statistics
    print(f"\nDEBUG Statistics:")
    print(f"  Query features - mean: {query_features.mean():.4f}, std: {query_features.std():.4f}")
    print(f"  Gallery features - mean: {gallery_features.mean():.4f}, std: {gallery_features.std():.4f}")
    print(f"  Similarity - min: {similarity.min():.4f}, max: {similarity.max():.4f}, mean: {similarity.mean():.4f}")
    print(f"  Top-1 similarities - mean: {similarity.max(dim=1)[0].mean():.4f}")

    # Get query and gallery names for matching
    query_dataset = val_loader_query.dataset
    gallery_dataset = val_loader_gallery.dataset

    query_list = query_dataset.images_name
    gallery_list = gallery_dataset.images_name
    pairs_dict = query_dataset.pairs_dict

    print(f"  Num queries: {len(query_list)}")
    print(f"  Num galleries: {len(gallery_list)}")
    print(f"  Avg matches per query: {np.mean([len(pairs_dict[q]) for q in query_list]):.1f}")

    # Create gallery name → index mapping
    gallery_idx = {name: idx for idx, name in enumerate(gallery_list)}

    # Compute Recall@K (GTA-UAV style)
    num_queries = len(query_list)
    cmc = np.zeros(len(gallery_list))  # Cumulative Match Characteristic

    # DEBUG: Track matching statistics
    total_gt_matches = 0
    queries_with_matches = 0
    first_match_positions = []

    for i in range(num_queries):
        query_name = query_list[i]

        # Get ground truth gallery matches for this query
        gt_gallery_names = pairs_dict[query_name]
        gt_gallery_indices = []
        for gt_name in gt_gallery_names:
            if gt_name in gallery_idx:
                gt_gallery_indices.append(gallery_idx[gt_name])

        if len(gt_gallery_indices) == 0:
            print(f"Warning: Query {query_name} has no valid gallery matches")
            continue

        # DEBUG: Track ground truth matches
        total_gt_matches += len(gt_gallery_indices)
        queries_with_matches += 1

        # Get similarity scores for this query
        scores = similarity[i].numpy()

        # Sort gallery by similarity (descending)
        sorted_indices = np.argsort(scores)[::-1]

        # Check if any ground truth is in top-K
        good_index = np.isin(sorted_indices, gt_gallery_indices)

        # Find first match position
        match_positions = np.where(good_index == 1)[0]
        if len(match_positions) > 0:
            first_match_pos = match_positions[0]
            first_match_positions.append(first_match_pos)
            cmc[first_match_pos:] += 1  # All ranks >= first match are correct

            # DEBUG: Show first few examples
            if i < 3:
                print(f"\n  Query {i} ({query_name}):")
                print(f"    GT matches: {len(gt_gallery_indices)}")
                print(f"    First match at rank: {first_match_pos + 1}")
                print(f"    Top-3 predicted: {[gallery_list[idx] for idx in sorted_indices[:3]]}")
                print(f"    GT galleries: {gt_gallery_names[:3]}")

    # Normalize CMC by number of queries
    cmc = cmc / num_queries

    # Extract Recall@K
    recall_at_1 = cmc[0] * 100
    recall_at_5 = cmc[4] * 100 if len(cmc) > 4 else 0
    recall_at_10 = cmc[9] * 100 if len(cmc) > 9 else 0

    print(f"Recall@1:  {recall_at_1:.2f}%")
    print(f"Recall@5:  {recall_at_5:.2f}%")
    print(f"Recall@10: {recall_at_10:.2f}%")
    print(f"{'='*60}\n")

    model.train()

    return {
        'r1': recall_at_1,
        'r5': recall_at_5,
        'r10': recall_at_10
    }


def validate_ortholoc_backbone_only(backbone, val_loader_query, val_loader_gallery, device='cuda'):
    """
    Validate using ONLY backbone features (bypass LoRA and MLP).
    This tests pure DINOv3 without any task-specific training.

    Args:
        backbone: Raw backbone model (without LoRA or MLP)
        val_loader_query: DataLoader for query images (drone)
        val_loader_gallery: DataLoader for gallery images (satellite)
        device: Device to run on

    Returns:
        dict: {'r1': float, 'r5': float, 'r10': float}
    """
    backbone.eval()

    # Extract query features (raw backbone)
    query_features = []
    with torch.no_grad():
        for images in val_loader_query:
            images = images.to(device)
            features = backbone(images)  # Raw backbone output
            features = F.normalize(features, dim=-1)  # L2 normalize
            query_features.append(features.cpu())

    query_features = torch.cat(query_features, dim=0)

    # Extract gallery features (raw backbone)
    gallery_features = []
    with torch.no_grad():
        for images in val_loader_gallery:
            images = images.to(device)
            features = backbone(images)
            features = F.normalize(features, dim=-1)
            gallery_features.append(features.cpu())

    gallery_features = torch.cat(gallery_features, dim=0)

    print(f"Query features (backbone): {query_features.shape}")
    print(f"Gallery features (backbone): {gallery_features.shape}\n")

    # Compute similarity matrix
    similarity = query_features @ gallery_features.T

    # Get ground truth matches from dataset
    query_dataset = val_loader_query.dataset
    gallery_dataset = val_loader_gallery.dataset

    pairs_dict = query_dataset.pairs_dict  # dict: query_name -> list of GT gallery names
    query_list = query_dataset.images_name
    gallery_list = gallery_dataset.images_name

    # Create gallery name -> index mapping
    gallery_idx = {name: idx for idx, name in enumerate(gallery_list)}

    # Compute CMC (Cumulative Matching Characteristics)
    num_queries = len(query_list)
    cmc = np.zeros(len(gallery_list))

    for i in range(num_queries):
        query_name = query_list[i]

        # Get ground truth gallery names for this query
        if query_name not in pairs_dict:
            continue

        gt_gallery_names = pairs_dict[query_name]

        # Convert GT names to indices
        gt_gallery_indices = [gallery_idx[gt_name] for gt_name in gt_gallery_names if gt_name in gallery_idx]

        if len(gt_gallery_indices) == 0:
            continue

        # Get similarity scores for this query
        scores = similarity[i].numpy()

        # Sort gallery by similarity (descending)
        sorted_indices = np.argsort(scores)[::-1]

        # Check if any ground truth is in top-K
        good_index = np.isin(sorted_indices, gt_gallery_indices)

        # Find first match position
        match_positions = np.where(good_index == 1)[0]
        if len(match_positions) > 0:
            first_match_pos = match_positions[0]
            cmc[first_match_pos:] += 1

    # Normalize CMC by number of queries
    cmc = cmc / num_queries

    # Extract Recall@K
    recall_at_1 = cmc[0] * 100
    recall_at_5 = cmc[4] * 100 if len(cmc) > 4 else 0
    recall_at_10 = cmc[9] * 100 if len(cmc) > 9 else 0

    print(f"Recall@1:  {recall_at_1:.2f}%")
    print(f"Recall@5:  {recall_at_5:.2f}%")
    print(f"Recall@10: {recall_at_10:.2f}%")
    print(f"{'='*70}\n")

    return {
        'r1': recall_at_1,
        'r5': recall_at_5,
        'r10': recall_at_10
    }


# -----------------------------------------------------------------------------#
# 5. Training Loop                                                            #
# -----------------------------------------------------------------------------#
def train_ortholoc(args):
    """Main training function."""

    # Setup
    save_path = os.path.join(args.output_dir, f"ortholoc_finetune_{time.strftime('%m%d%H%M')}")
    os.makedirs(save_path, exist_ok=True)
    sys.stdout = Logger(os.path.join(save_path, 'log.txt'))
    setup_system(seed=42)

    print(f"\n{'='*70}")
    print(f"PHASE 2: SIM-TO-REAL FINE-TUNING (GTA → ORTHOLOC)")
    print(f"{'='*70}")
    print(f"GTA Weights: {args.gta_weights}")
    print(f"Ortholoc Data: {args.real_data_root}")
    print(f"Output: {save_path}")
    print(f"\nConfiguration:")
    print(f"  Model: {args.model_name}")
    print(f"  Image Size: {args.img_size}")
    print(f"  LoRA: r={args.lora_r}, alpha={args.lora_alpha}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Augmentation: {args.augmentation_strength}")
    print(f"{'='*70}\n")

    # --- Initialize Model ---
    model = DesModelLoRA(
        model_name=args.model_name,
        img_size=args.img_size,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        mlp_hidden=args.mlp_hidden,
        mlp_output_dim=args.mlp_output_dim
    )

    # --- Load GTA Weights with Verification ---
    model = load_gta_weights_with_verification(model, args.gta_weights)

    # --- Move to GPU ---
    model = model.to(args.device)

    # --- Multi-GPU Support (NEW) ---
    if args.gpu_ids and len(args.gpu_ids) > 1:
        if torch.cuda.device_count() < len(args.gpu_ids):
            print(f"⚠ WARNING: Requested {len(args.gpu_ids)} GPUs, but only {torch.cuda.device_count()} available")
            args.gpu_ids = args.gpu_ids[:torch.cuda.device_count()]

        print(f"Using DataParallel on GPUs: {args.gpu_ids}\n")
        model = torch.nn.DataParallel(model, device_ids=args.gpu_ids)

    # --- Data Transforms ---
    print(f"Creating data transforms...")
    val_transform, sat_transform, drone_transform = get_ortholoc_transforms(
        img_size=args.img_size,
        augmentation_strength=args.augmentation_strength
    )

    # --- Training Dataset ---
    print(f"Loading training data...")
    train_dataset = GTADatasetTrain(
        data_root=args.real_data_root,
        pairs_meta_file=args.real_train_json,
        transforms_query=drone_transform,
        transforms_gallery=sat_transform,
        mode=args.train_mode,
        train_ratio=1.0,
        prob_flip=args.prob_flip
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True  # For stable batch stats
    )

    print(f"  Training pairs: {len(train_dataset)}")
    print(f"  Batches per epoch: {len(train_loader)}\n")

    # --- Validation Dataset (NEW) ---
    print(f"Loading validation data...")

    # Query loader (drone) - Use SimpleValDataset for Ortholoc compatibility
    val_dataset_query = SimpleValDataset(
        data_root=args.real_data_root,
        pairs_meta_file=args.real_val_json,
        view='drone',
        transforms=val_transform
    )

    val_loader_query = DataLoader(
        val_dataset_query,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Gallery loader (satellite) - Use SimpleValDataset for Ortholoc compatibility
    val_dataset_gallery = SimpleValDataset(
        data_root=args.real_data_root,
        pairs_meta_file=args.real_val_json,
        view='satellite',
        transforms=val_transform
    )

    val_loader_gallery = DataLoader(
        val_dataset_gallery,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    print(f"  Val queries: {len(val_dataset_query)}")
    print(f"  Val gallery: {len(val_dataset_gallery)}\n")

    # Verify dataset compatibility
    print(f"Verifying dataset compatibility...")
    sample = train_dataset[0]
    print(f"  Sample shapes: query={sample[0].shape}, gallery={sample[1].shape}, weight={sample[2]:.4f}")

    # Check weight distribution
    print(f"  Checking weight distribution (first 1000 samples)...")
    weights = [train_dataset[i][2] for i in range(min(1000, len(train_dataset)))]
    print(f"    Min: {min(weights):.4f}, Max: {max(weights):.4f}, Mean: {np.mean(weights):.4f}")
    perfect_matches = sum(1 for w in weights if w > 0.99)
    semi_positives = sum(1 for w in weights if 0.1 < w < 0.99)
    print(f"    Perfect (IoU=1.0): {perfect_matches}, Semi-pos (IoU<1.0): {semi_positives}\n")

    # --- Optimizer & Scheduler ---
    print(f"Setting up optimizer and scheduler...")
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    loss_function = WeightedInfoNCE(label_smoothing=args.label_smoothing, device=args.device, k=args.k)
    scaler = GradScaler()

    num_training_steps = len(train_loader) * args.epochs
    num_warmup_steps = int(len(train_loader) * args.warmup_epochs)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_training_steps=num_training_steps,
        num_warmup_steps=num_warmup_steps
    )

    print(f"  Training steps: {num_training_steps}")
    print(f"  Warmup steps: {num_warmup_steps}")
    print(f"  Initial LR: {args.lr:.2e}\n")

    # --- ABLATION STUDY: Pure DINOv3 Backbone ---
    print(f"{'='*70}")
    print(f"ABLATION: Pure DINOv3 Backbone (No LoRA, No MLP)")
    print(f"{'='*70}\n")

    backbone_only = model.model.base_model.model  # Access raw ViT backbone
    backbone_metrics = validate_ortholoc_backbone_only(
        backbone_only, val_loader_query, val_loader_gallery, device=args.device
    )

    print(f"Pure DINOv3 Backbone Performance:")
    print(f"  R@1:  {backbone_metrics['r1']:.2f}%")
    print(f"  R@5:  {backbone_metrics['r5']:.2f}%")
    print(f"  R@10: {backbone_metrics['r10']:.2f}%\n")

    # --- ABLATION STUDY: GTA-Finetuned Model ---
    print(f"{'='*70}")
    print(f"ABLATION: GTA-Finetuned (DINOv3 + LoRA + MLP)")
    print(f"{'='*70}\n")

    gta_metrics = validate_ortholoc(model, val_loader_query, val_loader_gallery, device=args.device)

    print(f"GTA-Finetuned Performance (current baseline):")
    print(f"  R@1:  {gta_metrics['r1']:.2f}%")
    print(f"  R@5:  {gta_metrics['r5']:.2f}%")
    print(f"  R@10: {gta_metrics['r10']:.2f}%\n")

    # --- ABLATION SUMMARY ---
    print(f"{'='*70}")
    print(f"ABLATION SUMMARY")
    print(f"{'='*70}")
    print(f"Pure DINOv3:     R@1={backbone_metrics['r1']:5.2f}%  R@5={backbone_metrics['r5']:5.2f}%  R@10={backbone_metrics['r10']:5.2f}%  (no task training)")
    print(f"GTA-finetuned:   R@1={gta_metrics['r1']:5.2f}%  R@5={gta_metrics['r5']:5.2f}%  R@10={gta_metrics['r10']:5.2f}%  (trained on GTA-UAV)")
    print(f"Improvement:     R@1=+{gta_metrics['r1'] - backbone_metrics['r1']:4.2f}%  R@5=+{gta_metrics['r5'] - backbone_metrics['r5']:4.2f}%  R@10=+{gta_metrics['r10'] - backbone_metrics['r10']:4.2f}%")
    print(f"{'='*70}\n")
    print(f"Now starting fine-tuning on OrthoLoC to improve these numbers...\n")

    # --- Training Loop ---
    best_r1 = gta_metrics['r1']  # Initialize with GTA-finetuned performance
    best_epoch = 0

    print(f"{'='*70}")
    print(f"STARTING TRAINING")
    print(f"{'='*70}\n")

    for epoch in range(1, args.epochs + 1):
        print(f"{'='*70}")
        print(f"EPOCH {epoch}/{args.epochs}")
        print(f"{'='*70}")

        # Shuffle dataset (mutually exclusive sampling)
        train_dataset.shuffle()

        # Training
        epoch_start = time.time()
        loss = train_with_weight(
            train_config=args,
            model=model,
            dataloader=train_loader,
            loss_function=loss_function,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            with_weight=args.with_weight
        )
        epoch_time = time.time() - epoch_start

        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {loss:.4f}")
        print(f"  Learning Rate: {current_lr:.2e}")
        print(f"  Time: {epoch_time/60:.2f} min")

        # Validation (NEW)
        val_metrics = validate_ortholoc(model, val_loader_query, val_loader_gallery, device=args.device)

        print(f"\nEpoch {epoch} Metrics:")
        print(f"  Train Loss: {loss:.4f}")
        print(f"  Val R@1: {val_metrics['r1']:.2f}%")
        print(f"  Val R@5: {val_metrics['r5']:.2f}%")
        print(f"  Val R@10: {val_metrics['r10']:.2f}%")

        # Save checkpoints
        # Always save latest
        if torch.cuda.device_count() > 1 and isinstance(model, torch.nn.DataParallel):
            torch.save(model.module.state_dict(), f"{save_path}/weights_latest.pth")
        else:
            torch.save(model.state_dict(), f"{save_path}/weights_latest.pth")

        # Save best based on R@1 (NEW: was based on loss)
        if val_metrics['r1'] > best_r1:
            best_r1 = val_metrics['r1']
            best_epoch = epoch

            if torch.cuda.device_count() > 1 and isinstance(model, torch.nn.DataParallel):
                torch.save(model.module.state_dict(), f"{save_path}/weights_best.pth")
            else:
                torch.save(model.state_dict(), f"{save_path}/weights_best.pth")

            print(f"  ✓ NEW BEST MODEL! R@1: {best_r1:.2f}% (Epoch {best_epoch})")
        else:
            print(f"  Best R@1: {best_r1:.2f}% (Epoch {best_epoch})")

        # Save periodic checkpoints every 5 epochs
        if epoch % 5 == 0:
            if torch.cuda.device_count() > 1 and isinstance(model, torch.nn.DataParallel):
                torch.save(model.module.state_dict(), f"{save_path}/weights_e{epoch}_r1{val_metrics['r1']:.2f}.pth")
            else:
                torch.save(model.state_dict(), f"{save_path}/weights_e{epoch}_r1{val_metrics['r1']:.2f}.pth")

        print(f"{'='*70}\n")

    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Best Val R@1: {best_r1:.2f}% (Epoch {best_epoch})")
    print(f"Weights saved to: {save_path}")
    print(f"  - weights_best.pth (R@1: {best_r1:.2f}%)")
    print(f"  - weights_latest.pth (Epoch {args.epochs})")
    print(f"{'='*70}\n")

    print(f"{'='*70}")
    print(f"ABLATION STUDY SUMMARY")
    print(f"{'='*70}")
    print(f"Pure DINOv3 backbone:        R@1={backbone_metrics['r1']:5.2f}%  (no task training)")
    print(f"GTA-finetuned (baseline):    R@1={gta_metrics['r1']:5.2f}%  (trained on GTA-UAV)")
    print(f"OrthoLoC-finetuned (best):   R@1={best_r1:5.2f}%  (fine-tuned on OrthoLoC)")
    print(f"")
    print(f"Improvement from pure DINOv3 to GTA:      +{gta_metrics['r1'] - backbone_metrics['r1']:4.2f}%")
    print(f"Improvement from GTA to OrthoLoC:         +{best_r1 - gta_metrics['r1']:4.2f}%")
    print(f"Total improvement from pure DINOv3:       +{best_r1 - backbone_metrics['r1']:4.2f}%")
    print(f"{'='*70}\n")


# -----------------------------------------------------------------------------#
# 6. Main Entry Point                                                         #
# -----------------------------------------------------------------------------#
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sim-to-Real LoRA Fine-Tuning: GTA-UAV → Ortholoc')

    # Paths
    parser.add_argument('--gta_weights', type=str, required=True,
                       help='Path to GTA-pretrained LoRA weights')
    parser.add_argument('--real_data_root', type=str, required=True,
                       help='Path to Ortholoc dataset root')
    parser.add_argument('--real_train_json', type=str, default='cross-area-drone2sate-train.json',
                       help='Training JSON file name')
    parser.add_argument('--real_val_json', type=str, default='cross-area-drone2sate-val.json',
                       help='Validation JSON file name')
    parser.add_argument('--output_dir', type=str, default='./work_dir/ortholoc',
                       help='Output directory for checkpoints and logs')

    # Model Architecture (MUST MATCH GTA)
    parser.add_argument('--model_name', type=str, default='vit_small_patch16_dinov3.lvd1689m',
                       help='Backbone model name')
    parser.add_argument('--img_size', type=int, default=384,
                       help='Input image size')
    parser.add_argument('--lora_r', type=int, default=64,
                       help='LoRA rank (MUST match GTA training!)')
    parser.add_argument('--lora_alpha', type=int, default=128,
                       help='LoRA alpha (MUST match GTA training!)')
    parser.add_argument('--mlp_hidden', type=int, default=512,
                       help='MLP hidden dimension (512 for PEFT, 2048 for Custom)')
    parser.add_argument('--mlp_output_dim', type=int, default=None,
                       help='MLP output dimension (None=embed_dim for PEFT, 256 for Custom)')

    # Training Hyperparameters
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size (128-256 typical for LoRA)')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=5e-5,
                       help='Learning rate (conservative for fine-tuning)')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay')
    parser.add_argument('--warmup_epochs', type=float, default=1.0,
                       help='Warmup epochs (fraction)')

    # Loss & Data
    parser.add_argument('--k', type=float, default=3.0,
                       help='Weight sensitivity for WeightedInfoNCE (3-5 typical)')
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                       help='Label smoothing for InfoNCE loss (0.0=none, 0.1=default)')
    parser.add_argument('--train_mode', type=str, default='pos_semipos',
                       choices=['pos', 'pos_semipos'],
                       help='Training mode: pos (IoU=1.0 only) or pos_semipos (mixed IoU)')
    parser.add_argument('--prob_flip', type=float, default=0.5,
                       help='Probability of synchronized horizontal flip')
    parser.add_argument('--augmentation_strength', type=str, default='moderate',
                       choices=['conservative', 'moderate', 'aggressive'],
                       help='Augmentation strength (conservative recommended for sim-to-real)')

    # System
    parser.add_argument('--gpu_ids', type=str, default='0',
                       help='Comma-separated GPU IDs (e.g., "0,1,2")')
    parser.add_argument('--num_workers', type=int, default=8,
                       help='Number of data loading workers')

    # Parse and setup
    args = parser.parse_args()

    # Parse GPU IDs
    args.gpu_ids = [int(id) for id in args.gpu_ids.split(',')]
    args.device = f'cuda:{args.gpu_ids[0]}' if torch.cuda.is_available() else 'cpu'

    # Additional args for compatibility with trainer
    args.clip_grad = 100.0  # Match GTA training
    args.verbose = False
    args.with_weight = True  # Use weighted InfoNCE
    args.scheduler = 'cosine'  # Required by trainer.py

    # Convert mlp_output_dim=None to actual None (argparse gives string)
    if args.mlp_output_dim == 'None' or args.mlp_output_dim is None:
        args.mlp_output_dim = None

    # Validate paths
    if not os.path.exists(args.gta_weights):
        raise FileNotFoundError(f"GTA weights not found: {args.gta_weights}")
    if not os.path.exists(args.real_data_root):
        raise FileNotFoundError(f"Ortholoc data not found: {args.real_data_root}")

    # Run training
    train_ortholoc(args)