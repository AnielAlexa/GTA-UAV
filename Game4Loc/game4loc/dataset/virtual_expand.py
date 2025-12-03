"""
Virtual Expansion Dataset Wrapper for UAV Visual Localization

This module provides a dataset wrapper that virtually expands small datasets
by applying on-the-fly random augmentations. Instead of creating N copies of
each image, it maps expanded indices back to original data and applies
different random transforms each time.

Key features:
- N× virtual expansion without storing extra images
- Per-dataset expansion factors (e.g., VisLoc ×10, VPair ×5)
- Compatible with existing GTADatasetTrain shuffle_group() logic
- Integrates with RandomCenterCropZoom for altitude simulation
"""

import os
import cv2
import numpy as np
import random
import copy
import json
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from game4loc.transforms import RandomCenterCropZoom, Cut


class VirtualExpandDataset(Dataset):
    """
    Dataset wrapper that virtually expands training data by factor N.
    
    Each original sample can be accessed N times with different random
    augmentations applied. The index mapping is:
        original_idx = expanded_idx % len(original_dataset)
        
    This achieves effective dataset size multiplication without storing
    additional images on disk.
    
    Args:
        pairs_meta_file: Path to JSON metadata file (GTA format)
        data_root: Root directory for images
        expansion_factor: How many times to virtually expand the dataset
        transforms_query: Albumentations transforms for drone/query images
        transforms_gallery: Albumentations transforms for satellite/gallery images
        prob_flip: Probability of horizontal flip (applied to both images)
        shuffle_batch_size: Batch size for mutually exclusive sampling
        mode: Training mode ('pos_semipos' or 'pos')
        train_ratio: Fraction of data to use for training
        group_len: Number of satellite matches per drone image (for grouping)
        altitude_aug_prob: Extra altitude augmentation probability
        altitude_scale_range: Range for altitude simulation crop
    """
    
    def __init__(self,
                 pairs_meta_file,
                 data_root,
                 expansion_factor=1,
                 transforms_query=None,
                 transforms_gallery=None,
                 prob_flip=0.5,
                 shuffle_batch_size=128,
                 mode='pos_semipos',
                 train_ratio=1.0,
                 group_len=2,
                 altitude_aug_prob=0.0,
                 altitude_scale_range=(0.2, 0.5)):
        super().__init__()
        
        with open(os.path.join(data_root, pairs_meta_file), 'r', encoding='utf-8') as f:
            pairs_meta_data = json.load(f)
        
        self.data_root = data_root
        self.expansion_factor = expansion_factor
        self.group_len = group_len
        
        self.pairs = []
        self.pairs_sate2drone_dict = {}
        self.pairs_drone2sate_dict = {}
        self.pairs_match_set = set()

        for pair_drone2sate in pairs_meta_data:
            drone_img_dir = pair_drone2sate['drone_img_dir']
            drone_img_name = pair_drone2sate['drone_img_name']
            sate_img_dir = pair_drone2sate['sate_img_dir']
            
            # Training with Positive-only or Positive+Semi-positive data
            pair_sate_img_list = pair_drone2sate.get(f'pair_{mode}_sate_img_list', 
                                                     pair_drone2sate.get('pair_pos_sate_img_list', []))
            pair_sate_weight_list = pair_drone2sate.get(f'pair_{mode}_sate_weight_list',
                                                        pair_drone2sate.get('pair_pos_sate_weight_list', []))
            
            drone_img_file = os.path.join(data_root, drone_img_dir, drone_img_name)

            for pair_sate_img, pair_sate_weight in zip(pair_sate_img_list, pair_sate_weight_list):
                sate_img_file = os.path.join(data_root, sate_img_dir, pair_sate_img)
                self.pairs.append((drone_img_file, sate_img_file, pair_sate_weight))

            # Build Graph with All Edges (drone, sate)
            pair_all_sate_img_list = pair_drone2sate.get('pair_pos_semipos_sate_img_list',
                                                          pair_drone2sate.get('pair_pos_sate_img_list', []))
            for pair_sate_img in pair_all_sate_img_list:
                self.pairs_drone2sate_dict.setdefault(drone_img_name, []).append(pair_sate_img)
                self.pairs_sate2drone_dict.setdefault(pair_sate_img, []).append(drone_img_name)
                self.pairs_match_set.add((drone_img_name, pair_sate_img))

        self.transforms_query = transforms_query
        self.transforms_gallery = transforms_gallery
        self.prob_flip = prob_flip
        self.shuffle_batch_size = shuffle_batch_size
        
        # Additional altitude augmentation (on top of transforms)
        self.altitude_aug_prob = altitude_aug_prob
        self.altitude_scale_range = altitude_scale_range

        # Training with sparse data
        num_pairs = len(self.pairs)
        num_pairs_train = int(train_ratio * num_pairs)
        random.shuffle(self.pairs)
        self.pairs = self.pairs[:num_pairs_train]
        
        # Virtual expansion: store original pairs, samples will be expanded
        self.original_pairs = copy.deepcopy(self.pairs)
        self.samples = self._expand_samples()
        
        print(f"VirtualExpandDataset: {len(self.original_pairs)} original pairs "
              f"× {expansion_factor} = {len(self.samples)} virtual samples")
    
    def _expand_samples(self):
        """Expand samples by creating N references to each original pair."""
        expanded = []
        for i, pair in enumerate(self.original_pairs):
            for exp_idx in range(self.expansion_factor):
                # Store (original_pair_idx, expansion_idx) for varied augmentation
                expanded.append((i, exp_idx, pair[0], pair[1], pair[2]))
        return expanded
    
    def __getitem__(self, index):
        orig_idx, exp_idx, query_img_path, gallery_img_path, positive_weight = self.samples[index]
        
        # Load images
        query_img = cv2.imread(query_img_path)
        query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
        
        gallery_img = cv2.imread(gallery_img_path)
        gallery_img = cv2.cvtColor(gallery_img, cv2.COLOR_BGR2RGB)
        
        # Synchronized horizontal flip
        if np.random.random() < self.prob_flip:
            query_img = cv2.flip(query_img, 1)
            gallery_img = cv2.flip(gallery_img, 1)
        
        # Additional altitude augmentation (different each virtual copy)
        if self.altitude_aug_prob > 0 and np.random.random() < self.altitude_aug_prob:
            query_img = self._apply_altitude_aug(query_img)
            gallery_img = self._apply_altitude_aug(gallery_img)
        
        # Apply standard transforms
        if self.transforms_query is not None:
            query_img = self.transforms_query(image=query_img)['image']
            
        if self.transforms_gallery is not None:
            gallery_img = self.transforms_gallery(image=gallery_img)['image']
        
        return query_img, gallery_img, positive_weight
    
    def _apply_altitude_aug(self, image):
        """Apply random center crop zoom for altitude simulation."""
        h, w = image.shape[:2]
        scale = random.uniform(self.altitude_scale_range[0], self.altitude_scale_range[1])
        
        new_h, new_w = int(h * scale), int(w * scale)
        start_x = (w - new_w) // 2
        start_y = (h - new_h) // 2
        
        image = image[start_y:start_y+new_h, start_x:start_x+new_w, :]
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
        
        return image

    def __len__(self):
        return len(self.samples)

    def shuffle_group(self):
        """
        Implementation of Mutually Exclusive Sampling with virtual expansion.
        Operates on original pairs to ensure proper grouping, then expands.
        """
        print(f"\nShuffle Dataset in Batches (Virtual Expansion ×{self.expansion_factor}):")
        
        pair_pool = copy.deepcopy(self.original_pairs)
        random.shuffle(pair_pool)
        
        sate_batch = set()
        drone_batch = set()
        pairs_epoch = set()
        
        batches = []
        current_batch = []
        
        break_cnt = 0

        for pair in pair_pool:
            drone_img_path, sate_img_path, weight = pair
            drone_name = os.path.basename(drone_img_path)
            sate_name = os.path.basename(sate_img_path)
            
            if drone_name not in drone_batch and sate_name not in sate_batch:
                current_batch.append(pair)
                drone_batch.add(drone_name)
                sate_batch.add(sate_name)
                pairs_epoch.add((drone_name, sate_name))
                
                if len(current_batch) >= self.shuffle_batch_size:
                    batches.append(current_batch)
                    current_batch = []
                    sate_batch = set()
                    drone_batch = set()
            else:
                break_cnt += 1

        # Expand batches by expansion_factor
        expanded_samples = []
        for batch in batches:
            for pair in batch:
                for exp_idx in range(self.expansion_factor):
                    orig_idx = self.original_pairs.index(pair) if pair in self.original_pairs else 0
                    expanded_samples.append((orig_idx, exp_idx, pair[0], pair[1], pair[2]))
        
        self.samples = expanded_samples
        
        print(f"Original batches: {len(batches)}, "
              f"Samples after expansion: {len(self.samples)}, "
              f"Break count: {break_cnt}")


class CombinedVirtualDataset(Dataset):
    """
    Combines multiple VirtualExpandDataset instances into one unified dataset.
    
    Each sub-dataset can have its own expansion factor and augmentation params.
    Useful for combining UAV_VisLoc (×10) + VPair (×5) into one training set.
    
    Args:
        datasets: List of VirtualExpandDataset instances
    """
    
    def __init__(self, datasets):
        self.datasets = datasets
        self.cumulative_lengths = []
        
        total = 0
        for ds in datasets:
            total += len(ds)
            self.cumulative_lengths.append(total)
        
        self.total_length = total
        
        # Merge matching dictionaries for proper negative sampling
        self.pairs_sate2drone_dict = {}
        self.pairs_drone2sate_dict = {}
        self.pairs_match_set = set()
        
        for ds in datasets:
            self.pairs_sate2drone_dict.update(ds.pairs_sate2drone_dict)
            self.pairs_drone2sate_dict.update(ds.pairs_drone2sate_dict)
            self.pairs_match_set.update(ds.pairs_match_set)
        
        print(f"CombinedVirtualDataset: {len(datasets)} datasets, "
              f"{self.total_length} total samples")
    
    def __getitem__(self, index):
        # Find which dataset this index belongs to
        for i, cum_len in enumerate(self.cumulative_lengths):
            if index < cum_len:
                if i == 0:
                    local_idx = index
                else:
                    local_idx = index - self.cumulative_lengths[i-1]
                return self.datasets[i][local_idx]
        raise IndexError(f"Index {index} out of range")
    
    def __len__(self):
        return self.total_length
    
    def shuffle_group(self):
        """Shuffle each sub-dataset independently."""
        for ds in self.datasets:
            ds.shuffle_group()
        
        # Rebuild combined samples
        self.cumulative_lengths = []
        total = 0
        for ds in self.datasets:
            total += len(ds)
            self.cumulative_lengths.append(total)
        self.total_length = total


class CombinedDatasetEval(Dataset):
    """
    Evaluation dataset for combined JSONs that loads gallery images from the JSON
    instead of scanning directories.
    
    This is needed because combined datasets have satellite images in different
    subdirectories (UAV_VisLoc_dataset/satellite, vpair/reference_views, etc.)
    
    Args:
        pairs_meta_file: Path to combined JSON file
        data_root: Root directory for images
        view: 'drone' for query images, 'sate' for gallery images
        mode: 'pos' or 'pos_semipos' for which satellite list to use
        transforms: Albumentations transforms
    """
    
    def __init__(self,
                 pairs_meta_file,
                 data_root,
                 view,
                 mode='pos_semipos',
                 transforms=None):
        super().__init__()
        
        with open(os.path.join(data_root, pairs_meta_file), 'r', encoding='utf-8') as f:
            pairs_meta_data = json.load(f)
        
        self.data_root = data_root
        self.transforms = transforms
        
        self.images_path = []
        self.images_name = []
        self.images_center_loc_xy = []
        self.images_topleft_loc_xy = []
        
        self.pairs_sate2drone_dict = {}
        self.pairs_drone2sate_dict = {}
        self.pairs_match_set = set()
        
        if view == 'drone':
            # Load drone/query images
            for pair in pairs_meta_data:
                drone_img_name = pair['drone_img_name']
                drone_img_dir = pair['drone_img_dir']
                drone_loc = pair.get('drone_loc_x_y', [0, 0])
                
                # Get satellite matches for this drone
                pair_sate_img_list = pair.get(f'pair_{mode}_sate_img_list', 
                                               pair.get('pair_pos_semipos_sate_img_list', []))
                
                # Build match dictionaries
                for sate_img in pair_sate_img_list:
                    self.pairs_drone2sate_dict.setdefault(drone_img_name, []).append(sate_img)
                    self.pairs_sate2drone_dict.setdefault(sate_img, []).append(drone_img_name)
                    self.pairs_match_set.add((drone_img_name, sate_img))
                
                # Only add if there are matches
                if len(pair_sate_img_list) > 0:
                    self.images_path.append(os.path.join(data_root, drone_img_dir, drone_img_name))
                    self.images_name.append(drone_img_name)
                    self.images_center_loc_xy.append((drone_loc[0], drone_loc[1]))
        
        elif view == 'sate':
            # Load satellite/gallery images from JSON (not by scanning directories)
            seen_sate = set()
            for pair in pairs_meta_data:
                sate_img_dir = pair['sate_img_dir']
                pair_sate_img_list = pair.get(f'pair_{mode}_sate_img_list',
                                               pair.get('pair_pos_semipos_sate_img_list', []))
                pair_sate_loc_list = pair.get(f'pair_{mode}_sate_loc_x_y_list',
                                               pair.get('pair_pos_semipos_sate_loc_x_y_list', []))
                
                for i, sate_img in enumerate(pair_sate_img_list):
                    if sate_img in seen_sate:
                        continue
                    seen_sate.add(sate_img)
                    
                    self.images_path.append(os.path.join(data_root, sate_img_dir, sate_img))
                    self.images_name.append(sate_img)
                    
                    # Get location if available
                    if i < len(pair_sate_loc_list):
                        loc = pair_sate_loc_list[i]
                        self.images_center_loc_xy.append((loc[0], loc[1]))
                    else:
                        self.images_center_loc_xy.append((0, 0))
                    
                    # Top-left location (not used for combined datasets)
                    self.images_topleft_loc_xy.append((0, 0))
        
        print(f"CombinedDatasetEval ({view}): {len(self.images_name)} images")
    
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
        return len(self.images_name)


def get_transforms_finetune(image_size_sat=(384, 384),
                            img_size_ground=(384, 384),
                            mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225],
                            ground_cutting=0,
                            altitude_aug_prob=0.5,
                            altitude_scale_range=(0.15, 0.45)):
    """
    Get transforms for fine-tuning on real-world datasets.
    
    Compared to GTA training transforms:
    - More aggressive altitude augmentation for low-altitude drones (70-150m)
    - Slightly reduced dropout/noise since real images have natural variation
    
    Args:
        image_size_sat: (H, W) for satellite/reference images
        img_size_ground: (H, W) for drone/query images  
        mean, std: Normalization parameters (ImageNet default)
        ground_cutting: Pixels to cut from top/bottom of drone images
        altitude_aug_prob: Probability of altitude augmentation
        altitude_scale_range: (min, max) scale for center crop zoom
    
    Returns:
        (satellite_transforms, ground_transforms)
    """
    
    satellite_transforms = A.Compose([
        RandomCenterCropZoom(scale_limit=altitude_scale_range, p=altitude_aug_prob),
        A.ImageCompression(quality_lower=85, quality_upper=100, p=0.4),
        A.Resize(image_size_sat[0], image_size_sat[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, always_apply=False, p=0.5),
        A.OneOf([
            A.AdvancedBlur(p=1.0),
            A.Sharpen(p=1.0),
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
        ], p=0.25),
        A.OneOf([
            A.GridDropout(ratio=0.3, p=1.0),
            A.CoarseDropout(max_holes=20,
                            max_height=int(0.15*image_size_sat[0]),
                            max_width=int(0.15*image_size_sat[0]),
                            min_holes=5,
                            min_height=int(0.05*image_size_sat[0]),
                            min_width=int(0.05*image_size_sat[0]),
                            p=1.0),
        ], p=0.2),
        A.Normalize(mean, std),
        ToTensorV2(),
    ])

    ground_transforms = A.Compose([
        Cut(cutting=ground_cutting, p=1.0),
        RandomCenterCropZoom(scale_limit=altitude_scale_range, p=altitude_aug_prob),
        A.ImageCompression(quality_lower=85, quality_upper=100, p=0.4),
        A.Resize(img_size_ground[0], img_size_ground[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, always_apply=False, p=0.5),
        A.OneOf([
            A.AdvancedBlur(p=1.0),
            A.Sharpen(p=1.0),
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
        ], p=0.25),
        A.OneOf([
            A.GridDropout(ratio=0.4, p=1.0),
            A.CoarseDropout(max_holes=20,
                            max_height=int(0.15*img_size_ground[0]),
                            max_width=int(0.15*img_size_ground[0]),
                            min_holes=5,
                            min_height=int(0.05*img_size_ground[0]),
                            min_width=int(0.05*img_size_ground[0]),
                            p=1.0),
        ], p=0.2),
        A.Normalize(mean, std),
        ToTensorV2(),
    ])
    
    return satellite_transforms, ground_transforms


def get_transforms_val(image_size_sat=(384, 384),
                       img_size_ground=(384, 384),
                       mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225],
                       ground_cutting=0):
    """Validation transforms (no augmentation)."""
    
    satellite_transforms = A.Compose([
        A.Resize(image_size_sat[0], image_size_sat[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
        A.Normalize(mean, std),
        ToTensorV2(),
    ])
    
    ground_transforms = A.Compose([
        Cut(cutting=ground_cutting, p=1.0),
        A.Resize(img_size_ground[0], img_size_ground[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
        A.Normalize(mean, std),
        ToTensorV2(),
    ])
    
    return satellite_transforms, ground_transforms
