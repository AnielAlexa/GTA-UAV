#!/usr/bin/env python3
"""
LoRA Fine-tuning for Low-Altitude UAV Geo-Localization

This script fine-tunes a DINOv3 ViT-Small backbone using LoRA (Low-Rank Adaptation)
for drone-to-satellite matching at low altitudes (70-150m).

Key Features:
- LoRA adapters on attention (qkv, proj) and MLP (fc1, fc2) layers
- 100% frozen backbone (only LoRA adapters + MLP head trainable)
- On-the-fly altitude simulation for real high-altitude images
- Per-dataset evaluation (GTA vs VisLoc breakdown)
- Mixed precision training with gradient checkpointing support

Architecture:
- Backbone: DINOv3 ViT-Small (vit_small_patch16_dinov3.lvd1689m)
- LoRA: r=128, alpha=256 on qkv, proj, fc1, fc2
- Head: MLP (384 → 2048 → 256) with ReLU + Dropout
- Temperature: Learnable logit_scale

Usage:
    python train_lora_fly.py \
        --train_pairs_meta_file combined-lora-train.json \
        --test_pairs_meta_file combined-lora-test.json \
        --lora_r 128 --lora_alpha 256 \
        --batch_size 128 --lr 3e-4 --epochs 10

Author: UAV Geo-Localization Pipeline
"""

import os
import sys
import time
import math
import shutil
import random
import argparse
import copy
import json
import gc
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.core.transforms_interface import ImageOnlyTransform

from transformers import get_cosine_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup

# Add Game4Loc to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from game4loc.utils import setup_system, Logger, AverageMeter
from game4loc.loss import WeightedInfoNCE
from game4loc.transforms import Cut


# =============================================================================
# CUSTOM TRANSFORMS
# =============================================================================

class LowAltitudeSimulation(ImageOnlyTransform):
    """
    Simulates low-altitude (70-150m) view from high-altitude (~400m) images.
    
    This transform applies:
    1. Random center crop (scale 0.15-0.30 of original size)
    2. Random center jitter (±10% GPS noise simulation)
    3. Resize back to original dimensions
    
    The crop scale simulates altitude ratio:
    - 400m → 100m: crop ~25% of image (scale=0.25)
    - 400m → 70m:  crop ~17.5% of image (scale=0.175)
    
    Args:
        crop_scale_range: Tuple of (min_scale, max_scale) for random crop
        jitter_ratio: Maximum center offset as fraction of crop size (0.1 = ±10%)
        always_apply: Always apply this transform
        p: Probability of applying (default 0.8 for training)
    """
    
    def __init__(
        self,
        crop_scale_range: Tuple[float, float] = (0.15, 0.30),
        jitter_ratio: float = 0.1,
        always_apply: bool = False,
        p: float = 0.8
    ):
        super().__init__(always_apply, p)
        self.crop_scale_range = crop_scale_range
        self.jitter_ratio = jitter_ratio
    
    def apply(self, image, **params):
        h, w = image.shape[:2]
        
        # Random crop scale
        scale = random.uniform(self.crop_scale_range[0], self.crop_scale_range[1])
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Center position with jitter
        center_x, center_y = w // 2, h // 2
        
        # Add GPS noise (±jitter_ratio of crop size)
        max_jitter_x = int(new_w * self.jitter_ratio)
        max_jitter_y = int(new_h * self.jitter_ratio)
        
        jitter_x = random.randint(-max_jitter_x, max_jitter_x) if max_jitter_x > 0 else 0
        jitter_y = random.randint(-max_jitter_y, max_jitter_y) if max_jitter_y > 0 else 0
        
        center_x += jitter_x
        center_y += jitter_y
        
        # Calculate crop bounds
        start_x = max(0, center_x - new_w // 2)
        start_y = max(0, center_y - new_h // 2)
        
        # Ensure we don't go out of bounds
        start_x = min(start_x, w - new_w)
        start_y = min(start_y, h - new_h)
        
        # Crop
        cropped = image[start_y:start_y + new_h, start_x:start_x + new_w]
        
        # Resize back to original size
        resized = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
        
        return resized
    
    def get_transform_init_args_names(self):
        return ("crop_scale_range", "jitter_ratio")


class RandomCenterCropZoom(ImageOnlyTransform):
    """Standard center crop zoom (used for GTA-UAV synthetic data)."""
    
    def __init__(
        self,
        scale_limit: Tuple[float, float] = (0.25, 0.55),
        always_apply: bool = False,
        p: float = 0.5
    ):
        super().__init__(always_apply, p)
        self.scale_limit = scale_limit
    
    def apply(self, image, **params):
        h, w = image.shape[:2]
        scale = random.uniform(self.scale_limit[0], self.scale_limit[1])
        
        new_h, new_w = int(h * scale), int(w * scale)
        start_x = (w - new_w) // 2
        start_y = (h - new_h) // 2
        
        cropped = image[start_y:start_y + new_h, start_x:start_x + new_w]
        resized = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
        
        return resized
    
    def get_transform_init_args_names(self):
        return ("scale_limit",)


def get_transforms_lora(
    img_size: Tuple[int, int],
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225],
    ground_cutting: int = 0
) -> Dict[str, A.Compose]:
    """
    Get transforms for LoRA training with source-specific augmentation.
    
    Returns a dictionary with:
    - 'real_query': For VisLoc/VPAIR drone images (aggressive low-altitude sim)
    - 'gta_query': For GTA-UAV drone images (standard augmentation)
    - 'gallery': For all satellite/reference images (standard augmentation)
    - 'val': For validation (no augmentation except resize)
    """
    
    # Real high-altitude drone images → simulate low altitude
    real_query_transforms = A.Compose([
        Cut(cutting=ground_cutting, p=1.0),
        LowAltitudeSimulation(
            crop_scale_range=(0.15, 0.30),  # Aggressive crop for 400m → 70-120m
            jitter_ratio=0.1,               # ±10% GPS noise
            p=0.85                          # High probability
        ),
        A.ImageCompression(quality_lower=85, quality_upper=100, p=0.5),
        A.Resize(img_size[0], img_size[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.OneOf([
            A.AdvancedBlur(p=1.0),
            A.Sharpen(p=1.0),
            A.GaussNoise(p=1.0),
        ], p=0.3),
        A.Normalize(mean, std),
        ToTensorV2(),
    ])
    
    # GTA-UAV synthetic drone images → standard augmentation
    gta_query_transforms = A.Compose([
        Cut(cutting=ground_cutting, p=1.0),
        RandomCenterCropZoom(scale_limit=(0.25, 0.55), p=0.5),
        A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5),
        A.Resize(img_size[0], img_size[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
        A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15, p=0.5),
        A.OneOf([
            A.AdvancedBlur(p=1.0),
            A.Sharpen(p=1.0),
        ], p=0.3),
        A.OneOf([
            A.GridDropout(ratio=0.4, p=1.0),
            A.CoarseDropout(
                max_holes=25, max_height=int(0.2 * img_size[0]), max_width=int(0.2 * img_size[0]),
                min_holes=10, min_height=int(0.1 * img_size[0]), min_width=int(0.1 * img_size[0]),
                p=1.0
            ),
        ], p=0.3),
        A.Normalize(mean, std),
        ToTensorV2(),
    ])
    
    # Satellite/reference images → standard augmentation
    gallery_transforms = A.Compose([
        RandomCenterCropZoom(scale_limit=(0.25, 0.55), p=0.5),
        A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5),
        A.Resize(img_size[0], img_size[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
        A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15, p=0.5),
        A.OneOf([
            A.AdvancedBlur(p=1.0),
            A.Sharpen(p=1.0),
        ], p=0.3),
        A.OneOf([
            A.GridDropout(ratio=0.4, p=1.0),
            A.CoarseDropout(
                max_holes=25, max_height=int(0.2 * img_size[0]), max_width=int(0.2 * img_size[0]),
                min_holes=10, min_height=int(0.1 * img_size[0]), min_width=int(0.1 * img_size[0]),
                p=1.0
            ),
        ], p=0.3),
        A.Normalize(mean, std),
        ToTensorV2(),
    ])
    
    # Validation transforms (no augmentation)
    val_transforms = A.Compose([
        Cut(cutting=ground_cutting, p=1.0),
        A.Resize(img_size[0], img_size[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
        A.Normalize(mean, std),
        ToTensorV2(),
    ])
    
    return {
        'real_query': real_query_transforms,
        'gta_query': gta_query_transforms,
        'gallery': gallery_transforms,
        'val': val_transforms,
    }


# =============================================================================
# CUSTOM DATASET
# =============================================================================

class GTADatasetTrainLoRA(Dataset):
    """
    Training dataset with source-specific augmentation for LoRA fine-tuning.
    
    Key differences from GTADatasetTrain:
    - Reads 'dataset_source' and 'data_root' from each entry
    - Applies different transforms based on source:
        - VisLoc/VPAIR: LowAltitudeSimulation (aggressive crop + jitter)
        - GTA: Standard RandomCenterCropZoom
    - Satellite images always get standard augmentation
    
    Args:
        pairs_meta_file: Path to merged JSON file (from merge_json_balanced.py)
        transforms_dict: Dictionary of transforms from get_transforms_lora()
        prob_flip: Probability of synchronized horizontal flip
        shuffle_batch_size: Batch size for mutually exclusive sampling
        mode: 'pos_semipos' or 'pos' for pair selection
        train_ratio: Fraction of data to use
        group_len: Number of satellite matches per drone (for grouping)
    """
    
    def __init__(
        self,
        pairs_meta_file: str,
        transforms_dict: Dict[str, A.Compose],
        prob_flip: float = 0.5,
        shuffle_batch_size: int = 128,
        mode: str = 'pos_semipos',
        train_ratio: float = 1.0,
        group_len: int = 2
    ):
        super().__init__()
        
        # Load merged JSON (path is absolute since data_root is per-entry)
        with open(pairs_meta_file, 'r', encoding='utf-8') as f:
            pairs_meta_data = json.load(f)
        
        self.transforms_dict = transforms_dict
        self.prob_flip = prob_flip
        self.shuffle_batch_size = shuffle_batch_size
        self.group_len = group_len
        
        self.pairs = []
        self.pairs_sate2drone_dict = {}
        self.pairs_drone2sate_dict = {}
        self.pairs_match_set = set()
        
        for entry in pairs_meta_data:
            data_root = entry.get('data_root', '')
            dataset_source = entry.get('dataset_source', 'unknown')
            
            drone_img_dir = entry['drone_img_dir']
            drone_img_name = entry['drone_img_name']
            sate_img_dir = entry['sate_img_dir']
            
            # Get pair lists based on mode
            pair_sate_img_list = entry.get(f'pair_{mode}_sate_img_list', 
                                           entry.get('pair_pos_sate_img_list', []))
            pair_sate_weight_list = entry.get(f'pair_{mode}_sate_weight_list',
                                              entry.get('pair_pos_sate_weight_list', []))
            
            # Build full paths
            drone_img_file = os.path.join(data_root, drone_img_dir, drone_img_name)
            
            for pair_sate_img, pair_sate_weight in zip(pair_sate_img_list, pair_sate_weight_list):
                sate_img_file = os.path.join(data_root, sate_img_dir, pair_sate_img)
                self.pairs.append((
                    drone_img_file, 
                    sate_img_file, 
                    pair_sate_weight,
                    dataset_source  # Track source for transform selection
                ))
            
            # Build graph for mutually exclusive sampling
            pair_all_sate_img_list = entry.get('pair_pos_semipos_sate_img_list',
                                               entry.get('pair_pos_sate_img_list', []))
            for pair_sate_img in pair_all_sate_img_list:
                self.pairs_drone2sate_dict.setdefault(drone_img_name, []).append(pair_sate_img)
                self.pairs_sate2drone_dict.setdefault(pair_sate_img, []).append(drone_img_name)
                self.pairs_match_set.add((drone_img_name, pair_sate_img))
        
        # Apply train_ratio
        num_pairs = len(self.pairs)
        num_pairs_train = int(train_ratio * num_pairs)
        random.shuffle(self.pairs)
        self.pairs = self.pairs[:num_pairs_train]
        
        self.samples = copy.deepcopy(self.pairs)
        
        print(f"GTADatasetTrainLoRA: {len(self.pairs)} pairs loaded")
        
        # Print source distribution
        sources = {}
        for _, _, _, src in self.pairs:
            sources[src] = sources.get(src, 0) + 1
        for src, count in sorted(sources.items()):
            print(f"  - {src}: {count} ({100*count/len(self.pairs):.1f}%)")
    
    def __getitem__(self, index):
        query_img_path, gallery_img_path, positive_weight, dataset_source = self.samples[index]
        
        # Load images
        query_img = cv2.imread(query_img_path)
        if query_img is None:
            raise FileNotFoundError(f"Cannot load: {query_img_path}")
        query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
        
        gallery_img = cv2.imread(gallery_img_path)
        if gallery_img is None:
            raise FileNotFoundError(f"Cannot load: {gallery_img_path}")
        gallery_img = cv2.cvtColor(gallery_img, cv2.COLOR_BGR2RGB)
        
        # Synchronized horizontal flip
        if np.random.random() < self.prob_flip:
            query_img = cv2.flip(query_img, 1)
            gallery_img = cv2.flip(gallery_img, 1)
        
        # Select transforms based on dataset source
        if dataset_source in ('visloc', 'vpair'):
            # Real high-altitude → aggressive low-altitude simulation
            query_transform = self.transforms_dict['real_query']
        else:
            # GTA synthetic → standard augmentation
            query_transform = self.transforms_dict['gta_query']
        
        gallery_transform = self.transforms_dict['gallery']
        
        # Apply transforms
        query_img = query_transform(image=query_img)['image']
        gallery_img = gallery_transform(image=gallery_img)['image']
        
        return query_img, gallery_img, positive_weight
    
    def __len__(self):
        return len(self.samples)
    
    def shuffle(self):
        """Simple shuffle for each epoch."""
        random.shuffle(self.samples)
        print(f"Dataset shuffled: {len(self.samples)} samples")


class GTADatasetEvalLoRA(Dataset):
    """
    Evaluation dataset for LoRA fine-tuning.
    
    Loads query (drone) or gallery (satellite) images with their metadata
    for evaluation. Tracks dataset_source for per-dataset metrics.
    """
    
    def __init__(
        self,
        pairs_meta_file: str,
        view: str,  # 'drone' or 'sate'
        transforms: A.Compose,
        mode: str = 'pos_semipos',
        query_mode: str = 'D2S'
    ):
        super().__init__()
        
        with open(pairs_meta_file, 'r', encoding='utf-8') as f:
            pairs_meta_data = json.load(f)
        
        self.transforms = transforms
        self.view = view
        
        self.images_path = []
        self.images_name = []
        self.images_center_loc_xy = []
        self.images_topleft_loc_xy = []
        self.images_dataset_source = []  # Track source for per-dataset eval
        self.pairs_drone2sate_dict = {}
        
        if view == 'drone':
            # Query (drone) images
            seen_drones = set()
            for entry in pairs_meta_data:
                data_root = entry.get('data_root', '')
                dataset_source = entry.get('dataset_source', 'unknown')
                
                drone_img_dir = entry['drone_img_dir']
                drone_img_name = entry['drone_img_name']
                
                if drone_img_name in seen_drones:
                    continue
                seen_drones.add(drone_img_name)
                
                drone_img_file = os.path.join(data_root, drone_img_dir, drone_img_name)
                drone_loc = entry.get('drone_loc_x_y', [0, 0])
                
                self.images_path.append(drone_img_file)
                self.images_name.append(drone_img_name)
                self.images_center_loc_xy.append(drone_loc)
                self.images_dataset_source.append(dataset_source)
                
                # Build pairs dict
                pair_sate_list = entry.get(f'pair_{mode}_sate_img_list',
                                           entry.get('pair_pos_sate_img_list', []))
                self.pairs_drone2sate_dict[drone_img_name] = pair_sate_list
        
        else:
            # Gallery (satellite) images
            seen_sates = set()
            for entry in pairs_meta_data:
                data_root = entry.get('data_root', '')
                sate_img_dir = entry['sate_img_dir']
                
                pair_sate_list = entry.get(f'pair_{mode}_sate_img_list',
                                           entry.get('pair_pos_sate_img_list', []))
                pair_loc_list = entry.get(f'pair_{mode}_sate_loc_x_y_list',
                                          entry.get('pair_pos_sate_loc_x_y_list', []))
                
                for sate_img, sate_loc in zip(pair_sate_list, pair_loc_list):
                    if sate_img in seen_sates:
                        continue
                    seen_sates.add(sate_img)
                    
                    sate_img_file = os.path.join(data_root, sate_img_dir, sate_img)
                    
                    self.images_path.append(sate_img_file)
                    self.images_name.append(sate_img)
                    self.images_center_loc_xy.append(sate_loc)
                    self.images_topleft_loc_xy.append(sate_loc)  # Approx
        
        print(f"GTADatasetEvalLoRA ({view}): {len(self.images_path)} images")
    
    def __getitem__(self, index):
        img_path = self.images_path[index]
        
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Cannot load: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.transforms is not None:
            img = self.transforms(image=img)['image']
        
        return img
    
    def __len__(self):
        return len(self.images_path)


# =============================================================================
# LORA MODEL
# =============================================================================

class LoRALinear(nn.Module):
    """
    Low-Rank Adaptation layer for linear transformations.
    
    Implements: W' = W + BA where B ∈ R^{d×r}, A ∈ R^{r×k}
    
    Args:
        in_features: Input dimension
        out_features: Output dimension
        r: LoRA rank
        alpha: LoRA scaling factor
        dropout: Dropout rate for LoRA path
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 16,
        alpha: int = 32,
        dropout: float = 0.1
    ):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        
        # LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Initialize
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor, original_output: torch.Tensor) -> torch.Tensor:
        """Add LoRA contribution to original output."""
        lora_out = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
        return original_output + lora_out * self.scaling


class DesModelLoRA(nn.Module):
    """
    DINOv3 ViT with LoRA adapters for geo-localization.
    
    Architecture:
    - Backbone: DINOv3 ViT-Small (frozen)
    - LoRA adapters on: qkv, proj, fc1, fc2 in each transformer block
    - MLP head: 384 → hidden_dim → output_dim
    - Learnable temperature (logit_scale)
    
    Only LoRA adapters and MLP head are trainable (~12-15M params).
    """
    
    def __init__(
        self,
        model_name: str = 'vit_small_patch16_dinov3.lvd1689m',
        pretrained: bool = True,
        img_size: int = 384,
        lora_r: int = 128,
        lora_alpha: int = 256,
        lora_dropout: float = 0.1,
        mlp_hidden_dim: int = 2048,
        mlp_output_dim: int = 256,
        share_weights: bool = True
    ):
        super().__init__()
        
        self.share_weights = share_weights
        self.model_name = model_name
        self.img_size = img_size
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        
        # Load DINOv3 backbone
        print(f"Loading backbone: {model_name}")
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            img_size=img_size
        )
        
        # Freeze all backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Get backbone dimensions
        self.embed_dim = self.backbone.embed_dim  # 384 for ViT-Small
        num_blocks = len(self.backbone.blocks)
        
        print(f"Backbone embed_dim: {self.embed_dim}, blocks: {num_blocks}")
        
        # Add LoRA adapters to each transformer block
        self.lora_adapters = nn.ModuleDict()
        
        for block_idx, block in enumerate(self.backbone.blocks):
            prefix = f"block_{block_idx}"
            
            # QKV projection (in_features=embed_dim, out_features=3*embed_dim)
            qkv_in = self.embed_dim
            qkv_out = block.attn.qkv.out_features  # 3 * embed_dim
            self.lora_adapters[f"{prefix}_qkv"] = LoRALinear(
                qkv_in, qkv_out, r=lora_r, alpha=lora_alpha, dropout=lora_dropout
            )
            
            # Output projection
            proj_dim = self.embed_dim
            self.lora_adapters[f"{prefix}_proj"] = LoRALinear(
                proj_dim, proj_dim, r=lora_r, alpha=lora_alpha, dropout=lora_dropout
            )
            
            # MLP fc1 (embed_dim → mlp_ratio*embed_dim)
            mlp_hidden = block.mlp.fc1.out_features
            self.lora_adapters[f"{prefix}_fc1"] = LoRALinear(
                self.embed_dim, mlp_hidden, r=lora_r, alpha=lora_alpha, dropout=lora_dropout
            )
            
            # MLP fc2 (mlp_ratio*embed_dim → embed_dim)
            self.lora_adapters[f"{prefix}_fc2"] = LoRALinear(
                mlp_hidden, self.embed_dim, r=lora_r, alpha=lora_alpha, dropout=lora_dropout
            )
        
        # MLP head for descriptor projection
        self.mlp_head = nn.Sequential(
            nn.Linear(self.embed_dim, mlp_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(mlp_hidden_dim, mlp_output_dim)
        )
        
        # Learnable temperature
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        # Store original forward functions
        self._patch_backbone()
        
        # Print parameter counts
        self._print_param_counts()
    
    def _patch_backbone(self):
        """Patch backbone forward to include LoRA."""
        for block_idx, block in enumerate(self.backbone.blocks):
            prefix = f"block_{block_idx}"
            
            # Patch attention forward
            original_attn_forward = block.attn.forward
            lora_qkv = self.lora_adapters[f"{prefix}_qkv"]
            lora_proj = self.lora_adapters[f"{prefix}_proj"]
            
            def make_attn_forward(orig_forward, lora_qkv, lora_proj, block_attn):
                def patched_forward(x):
                    B, N, C = x.shape
                    
                    # Original QKV
                    qkv = block_attn.qkv(x)
                    # Add LoRA
                    qkv = lora_qkv(x, qkv)
                    
                    qkv = qkv.reshape(B, N, 3, block_attn.num_heads, C // block_attn.num_heads)
                    qkv = qkv.permute(2, 0, 3, 1, 4)
                    q, k, v = qkv.unbind(0)
                    
                    # Attention
                    attn = (q @ k.transpose(-2, -1)) * block_attn.scale
                    attn = attn.softmax(dim=-1)
                    attn = block_attn.attn_drop(attn)
                    
                    x = (attn @ v).transpose(1, 2).reshape(B, N, C)
                    
                    # Original projection
                    proj_out = block_attn.proj(x)
                    # Add LoRA
                    proj_out = lora_proj(x, proj_out)
                    
                    x = block_attn.proj_drop(proj_out)
                    return x
                
                return patched_forward
            
            block.attn.forward = make_attn_forward(
                original_attn_forward, lora_qkv, lora_proj, block.attn
            )
            
            # Patch MLP forward
            original_mlp_forward = block.mlp.forward
            lora_fc1 = self.lora_adapters[f"{prefix}_fc1"]
            lora_fc2 = self.lora_adapters[f"{prefix}_fc2"]
            
            def make_mlp_forward(orig_forward, lora_fc1, lora_fc2, block_mlp):
                def patched_forward(x):
                    # FC1 + LoRA
                    fc1_out = block_mlp.fc1(x)
                    fc1_out = lora_fc1(x, fc1_out)
                    
                    x = block_mlp.act(fc1_out)
                    x = block_mlp.drop1(x)
                    
                    # FC2 + LoRA
                    fc2_out = block_mlp.fc2(x)
                    fc2_out = lora_fc2(x, fc2_out)
                    
                    x = block_mlp.drop2(fc2_out)
                    return x
                
                return patched_forward
            
            block.mlp.forward = make_mlp_forward(
                original_mlp_forward, lora_fc1, lora_fc2, block.mlp
            )
    
    def _print_param_counts(self):
        """Print trainable vs total parameters."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        print(f"\nParameter counts:")
        print(f"  Total: {total_params:,}")
        print(f"  Trainable: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
        print(f"  Frozen: {frozen_params:,} ({100*frozen_params/total_params:.2f}%)")
        
        # Breakdown
        lora_params = sum(p.numel() for p in self.lora_adapters.parameters())
        mlp_params = sum(p.numel() for p in self.mlp_head.parameters())
        print(f"  LoRA adapters: {lora_params:,}")
        print(f"  MLP head: {mlp_params:,}")
    
    def get_config(self):
        """Get data config for transforms."""
        data_config = timm.data.resolve_model_data_config(self.backbone)
        return data_config
    
    def set_grad_checkpointing(self, enable: bool = True):
        """Enable gradient checkpointing."""
        self.backbone.set_grad_checkpointing(enable)
    
    def forward(self, img1=None, img2=None):
        """Forward pass through backbone + LoRA + MLP head."""
        
        def encode(x):
            # Get backbone features (LoRA is applied via patched forwards)
            features = self.backbone(x)
            # Project through MLP head
            features = self.mlp_head(features)
            return features
        
        if img1 is not None and img2 is not None:
            return encode(img1), encode(img2)
        elif img1 is not None:
            return encode(img1)
        else:
            return encode(img2)


# =============================================================================
# EVALUATION
# =============================================================================

def predict(config, model, dataloader):
    """Extract features from a dataloader."""
    model.eval()
    
    img_features_list = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features"):
            with autocast():
                batch = batch.to(config.device)
                features = model(img1=batch)
                
                if config.normalize_features:
                    features = F.normalize(features, dim=-1)
                
                img_features_list.append(features.float())
    
    return torch.cat(img_features_list, dim=0)


def evaluate(
    config,
    model,
    query_loader,
    gallery_loader,
    query_list,
    gallery_list,
    pairs_dict,
    ranks_list=[1, 5, 10],
    query_dataset_sources=None
):
    """
    Evaluate retrieval performance.
    
    Returns:
        r1: Recall@1 score
        results_by_source: Dict of {source: {r1, r5, r10, count}}
    """
    print("Extract Features and Compute Scores:")
    model.eval()
    
    # Extract query features
    query_features = predict(config, model, query_loader)
    
    # Compute scores batch-wise
    all_scores = []
    with torch.no_grad():
        for gallery_batch in tqdm(gallery_loader, desc="Computing scores"):
            with autocast():
                gallery_batch = gallery_batch.to(config.device)
                gallery_features = model(img2=gallery_batch)
                
                if config.normalize_features:
                    gallery_features = F.normalize(gallery_features, dim=-1)
            
            scores = query_features @ gallery_features.T
            all_scores.append(scores.cpu())
    
    all_scores = torch.cat(all_scores, dim=1).numpy()
    
    # Build gallery index
    gallery_idx = {img: idx for idx, img in enumerate(gallery_list)}
    
    # Get matches for each query
    matches_list = []
    for query_img in query_list:
        pairs = pairs_dict.get(query_img, [])
        matches = [gallery_idx[p] for p in pairs if p in gallery_idx]
        matches_list.append(np.array(matches))
    
    # Compute metrics
    query_num = len(query_list)
    cmc = np.zeros(len(gallery_list))
    
    # Per-source tracking
    source_results = {}
    if query_dataset_sources:
        for src in set(query_dataset_sources):
            source_results[src] = {'cmc': np.zeros(len(gallery_list)), 'count': 0}
    
    for i in range(query_num):
        score = all_scores[i]
        index = np.argsort(score)[::-1]
        
        matches = matches_list[i]
        if len(matches) == 0:
            continue
        
        good_index = np.isin(index, matches)
        match_rank = np.where(good_index)[0]
        
        if len(match_rank) > 0:
            cmc[match_rank[0]:] += 1
            
            # Per-source
            if query_dataset_sources:
                src = query_dataset_sources[i]
                source_results[src]['cmc'][match_rank[0]:] += 1
                source_results[src]['count'] += 1
    
    # Normalize CMC
    cmc = cmc / query_num
    
    # Print overall results
    results_str = []
    for r in ranks_list:
        results_str.append(f"R@{r}: {cmc[r-1]*100:.2f}%")
    print(f"[OVERALL] Queries: {query_num}, " + ", ".join(results_str))
    
    # Print per-source results
    results_by_source = {}
    if query_dataset_sources:
        for src, data in source_results.items():
            if data['count'] > 0:
                src_cmc = data['cmc'] / data['count']
                src_str = []
                for r in ranks_list:
                    src_str.append(f"R@{r}: {src_cmc[r-1]*100:.2f}%")
                print(f"[{src.upper()}] Queries: {data['count']}, " + ", ".join(src_str))
                
                results_by_source[src] = {
                    f'r{r}': src_cmc[r-1] for r in ranks_list
                }
                results_by_source[src]['count'] = data['count']
    
    # Cleanup
    del query_features, all_scores
    gc.collect()
    
    return cmc[0], results_by_source


# =============================================================================
# TRAINING
# =============================================================================

def train_epoch(
    config,
    model,
    dataloader,
    loss_function,
    optimizer,
    scheduler,
    scaler,
    epoch
):
    """Train for one epoch."""
    model.train()
    losses = AverageMeter()
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for query, reference, weight in pbar:
        optimizer.zero_grad()
        
        with autocast():
            query = query.to(config.device)
            reference = reference.to(config.device)
            weight = weight.to(config.device)
            
            # Forward
            features1, features2 = model(img1=query, img2=reference)
            
            # Get logit scale
            if hasattr(model, 'module'):
                logit_scale = model.module.logit_scale.exp()
            else:
                logit_scale = model.logit_scale.exp()
            
            # Compute loss
            loss_dict = loss_function(features1, features2, logit_scale, weight)
            loss = sum(loss_dict.values())
            
            losses.update(loss.item())
        
        # Backward
        scaler.scale(loss).backward()
        
        # Gradient clipping
        if config.clip_grad:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_value_(model.parameters(), config.clip_grad)
        
        scaler.step(optimizer)
        scaler.update()
        
        if scheduler:
            scheduler.step()
        
        pbar.set_postfix({'loss': f'{losses.avg:.4f}'})
    
    return losses.avg


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class Configuration:
    """Training configuration."""
    
    # Model
    model: str = 'vit_small_patch16_dinov3.lvd1689m'
    img_size: int = 384
    
    # LoRA (r=32 is standard; increase to 64/128 if underfitting)
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.1
    
    # MLP head (smaller = less overfitting risk)
    mlp_hidden_dim: int = 1024
    mlp_output_dim: int = 256
    
    # Training
    epochs: int = 10
    batch_size: int = 128
    lr: float = 3e-4
    warmup_epochs: float = 0.5
    
    # Loss
    with_weight: bool = True
    label_smoothing: float = 0.1
    k: float = 5
    
    # Optimizer
    clip_grad: float = 100.0
    
    # Data
    train_pairs_meta_file: str = 'combined-lora-train.json'
    test_pairs_meta_file: str = 'combined-lora-test.json'
    mode: str = 'pos_semipos'
    prob_flip: float = 0.5
    train_ratio: float = 1.0
    
    # Eval
    batch_size_eval: int = 128
    eval_every_n_epoch: int = 1
    normalize_features: bool = True
    
    # Hardware
    gpu_ids: tuple = (0,)
    num_workers: int = 8
    device: str = 'cuda'
    grad_checkpointing: bool = False
    
    # Output
    model_path: str = './work_dir/lora'
    verbose: bool = True
    
    # Misc
    seed: int = 42
    zero_shot: bool = True


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for UAV geo-localization")
    
    # Data
    parser.add_argument('--train_pairs_meta_file', type=str, required=True)
    parser.add_argument('--test_pairs_meta_file', type=str, required=True)
    parser.add_argument('--mode', type=str, default='pos_semipos')
    
    # Model
    parser.add_argument('--model', type=str, default='vit_small_patch16_dinov3.lvd1689m')
    parser.add_argument('--img_size', type=int, default=384)
    
    # LoRA (r=32 is standard; increase if underfitting)
    parser.add_argument('--lora_r', type=int, default=32)
    parser.add_argument('--lora_alpha', type=int, default=64)
    parser.add_argument('--lora_dropout', type=float, default=0.1)
    
    # MLP
    parser.add_argument('--mlp_hidden_dim', type=int, default=1024)
    parser.add_argument('--mlp_output_dim', type=int, default=256)
    
    # Training
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--warmup_epochs', type=float, default=0.5)
    
    # Loss
    parser.add_argument('--with_weight', action='store_true')
    parser.add_argument('--k', type=float, default=5)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    
    # Hardware
    parser.add_argument('--gpu_ids', type=str, default='0')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--grad_checkpointing', action='store_true')
    
    # Eval
    parser.add_argument('--eval_every_n_epoch', type=int, default=1)
    parser.add_argument('--batch_size_eval', type=int, default=128)
    
    # Output
    parser.add_argument('--model_path', type=str, default='./work_dir/lora')
    
    # Misc
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no_zero_shot', action='store_true')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create config
    config = Configuration()
    for key, value in vars(args).items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    # Parse GPU IDs
    if isinstance(args.gpu_ids, str):
        config.gpu_ids = tuple(int(x) for x in args.gpu_ids.split(','))
    
    config.zero_shot = not args.no_zero_shot
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Setup
    setup_system(seed=config.seed)
    
    # Create output directory
    save_time = time.strftime("%m%d%H%M%S")
    model_path = os.path.join(config.model_path, config.model.replace('.', '_'), save_time)
    os.makedirs(model_path, exist_ok=True)
    
    # Logger
    sys.stdout = Logger(os.path.join(model_path, 'log.txt'))
    
    print("=" * 70)
    print("LORA FINE-TUNING FOR UAV GEO-LOCALIZATION")
    print("=" * 70)
    print(f"Model: {config.model}")
    print(f"LoRA: r={config.lora_r}, alpha={config.lora_alpha}")
    print(f"Output: {model_path}")
    
    # =========================================================================
    # MODEL
    # =========================================================================
    print("\n" + "=" * 70)
    print("MODEL SETUP")
    print("=" * 70)
    
    model = DesModelLoRA(
        model_name=config.model,
        pretrained=True,
        img_size=config.img_size,
        lora_r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        mlp_hidden_dim=config.mlp_hidden_dim,
        mlp_output_dim=config.mlp_output_dim
    )
    
    # Get transforms config
    data_config = model.get_config()
    mean = data_config['mean']
    std = data_config['std']
    img_size = (config.img_size, config.img_size)
    
    # Gradient checkpointing
    if config.grad_checkpointing:
        model.set_grad_checkpointing(True)
        print("Gradient checkpointing: ENABLED")
    
    # Multi-GPU
    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
        print(f"Using {len(config.gpu_ids)} GPUs: {config.gpu_ids}")
        model = nn.DataParallel(model, device_ids=config.gpu_ids)
    
    model = model.to(config.device)
    
    # =========================================================================
    # DATA
    # =========================================================================
    print("\n" + "=" * 70)
    print("DATASET SETUP")
    print("=" * 70)
    
    transforms = get_transforms_lora(img_size, mean, std)
    
    # Training dataset
    train_dataset = GTADatasetTrainLoRA(
        pairs_meta_file=config.train_pairs_meta_file,
        transforms_dict=transforms,
        prob_flip=config.prob_flip,
        mode=config.mode,
        train_ratio=config.train_ratio
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    # Test datasets
    query_dataset = GTADatasetEvalLoRA(
        pairs_meta_file=config.test_pairs_meta_file,
        view='drone',
        transforms=transforms['val'],
        mode=config.mode
    )
    
    gallery_dataset = GTADatasetEvalLoRA(
        pairs_meta_file=config.test_pairs_meta_file,
        view='sate',
        transforms=transforms['val'],
        mode=config.mode
    )
    
    query_loader = DataLoader(
        query_dataset,
        batch_size=config.batch_size_eval,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    gallery_loader = DataLoader(
        gallery_dataset,
        batch_size=config.batch_size_eval,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test queries: {len(query_dataset)}")
    print(f"Test gallery: {len(gallery_dataset)}")
    
    # =========================================================================
    # LOSS & OPTIMIZER
    # =========================================================================
    print("\n" + "=" * 70)
    print("TRAINING SETUP")
    print("=" * 70)
    
    loss_function = WeightedInfoNCE(
        device=config.device,
        label_smoothing=config.label_smoothing,
        k=config.k
    )
    
    # Only optimize trainable parameters (LoRA + MLP + logit_scale)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=config.lr)
    
    # Scheduler
    train_steps = len(train_loader) * config.epochs
    warmup_steps = int(len(train_loader) * config.warmup_epochs)
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_training_steps=train_steps,
        num_warmup_steps=warmup_steps
    )
    
    scaler = GradScaler()
    
    print(f"Optimizer: AdamW, lr={config.lr}")
    print(f"Scheduler: Cosine, warmup={warmup_steps} steps")
    print(f"Total steps: {train_steps}")
    
    # =========================================================================
    # ZERO-SHOT EVALUATION
    # =========================================================================
    if config.zero_shot:
        print("\n" + "=" * 70)
        print("ZERO-SHOT EVALUATION")
        print("=" * 70)
        
        r1, _ = evaluate(
            config, model,
            query_loader, gallery_loader,
            query_dataset.images_name,
            gallery_dataset.images_name,
            query_dataset.pairs_drone2sate_dict,
            query_dataset_sources=query_dataset.images_dataset_source
        )
    
    # =========================================================================
    # TRAINING LOOP
    # =========================================================================
    best_r1 = 0.0
    best_visloc_r1 = 0.0
    
    for epoch in range(1, config.epochs + 1):
        print("\n" + "=" * 70)
        print(f"EPOCH {epoch}/{config.epochs}")
        print("=" * 70)
        
        # Shuffle
        train_dataset.shuffle()
        
        # Train
        train_loss = train_epoch(
            config, model, train_loader,
            loss_function, optimizer, scheduler, scaler, epoch
        )
        
        print(f"Train Loss: {train_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Evaluate
        if epoch % config.eval_every_n_epoch == 0 or epoch == config.epochs:
            print("\n" + "-" * 40)
            print("EVALUATION")
            print("-" * 40)
            
            r1, results_by_source = evaluate(
                config, model,
                query_loader, gallery_loader,
                query_dataset.images_name,
                gallery_dataset.images_name,
                query_dataset.pairs_drone2sate_dict,
                query_dataset_sources=query_dataset.images_dataset_source
            )
            
            # Get VisLoc R@1 for target domain tracking
            visloc_r1 = results_by_source.get('visloc', {}).get('r1', 0)
            
            # Save checkpoints
            state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            
            # Best overall
            if r1 > best_r1:
                best_r1 = r1
                torch.save(state_dict, os.path.join(model_path, f'best_overall_e{epoch}_r1_{r1:.4f}.pth'))
                print(f"✅ New best overall R@1: {r1*100:.2f}%")
            
            # Best VisLoc (target domain)
            if visloc_r1 > best_visloc_r1:
                best_visloc_r1 = visloc_r1
                torch.save(state_dict, os.path.join(model_path, f'best_visloc_e{epoch}_r1_{visloc_r1:.4f}.pth'))
                print(f"✅ New best VisLoc R@1: {visloc_r1*100:.2f}%")
    
    # Save final
    state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    torch.save(state_dict, os.path.join(model_path, 'final.pth'))
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best Overall R@1: {best_r1*100:.2f}%")
    print(f"Best VisLoc R@1: {best_visloc_r1*100:.2f}%")
    print(f"Checkpoints saved to: {model_path}")


if __name__ == '__main__':
    main()
