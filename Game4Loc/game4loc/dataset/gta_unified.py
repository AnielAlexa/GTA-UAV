# ---------------------------------------------------------------
# Unified GTA-compatible Dataset for UAV_VisLoc, VPair, and GTA-UAV
# Supports both lat/lon and x/y coordinate systems
# ---------------------------------------------------------------

import os
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
import copy
from tqdm import tqdm
import time
import random
import itertools
import json
import math
from game4loc.transforms import RandomCenterCropZoom


def latlon_to_meters(lat1, lon1, lat2, lon2):
    """
    Convert lat/lon distance to approximate meters using Haversine formula.
    Returns (dx, dy) in meters from point 1 to point 2.
    """
    R = 6378137  # Earth radius in meters
    
    # Convert to radians
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    
    # Approximate conversion
    dy = R * math.radians(lat2 - lat1)
    dx = R * math.radians(lon2 - lon1) * math.cos((lat1_rad + lat2_rad) / 2)
    
    return dx, dy


def get_sate_data(root_dir):
    """Get all satellite images from directory."""
    sate_img_dir_list = []
    sate_img_list = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg', '.tif')):
                sate_img_dir_list.append(root)
                sate_img_list.append(file)
    return sate_img_dir_list, sate_img_list


class GTAUnifiedDatasetTrain(Dataset):
    """
    Unified training dataset compatible with GTA-UAV pipeline.
    Supports datasets with either:
    - drone_loc_x_y (GTA-UAV format)
    - drone_loc_lat_lon (UAV_VisLoc, VPair format)
    """
    
    def __init__(self,
                 pairs_meta_file,
                 data_root,
                 transforms_query=None,
                 transforms_gallery=None,
                 prob_flip=0.5,
                 shuffle_batch_size=128,
                 mode='pos_semipos',
                 train_ratio=1.0,
                 group_len=2):
        super().__init__()
        
        with open(os.path.join(data_root, pairs_meta_file), 'r', encoding='utf-8') as f:
            pairs_meta_data = json.load(f)
        self.data_root = data_root
        self.group_len = group_len

        self.pairs = []
        self.pairs_sate2drone_dict = {}
        self.pairs_drone2sate_dict = {}
        self.pairs_match_set = set()

        for pair_drone2sate in pairs_meta_data:
            drone_img_dir = pair_drone2sate['drone_img_dir']
            drone_img_name = pair_drone2sate['drone_img_name']
            sate_img_dir = pair_drone2sate['sate_img_dir']
            
            # Training with Positive-only data or Positive+Semi-positive data
            pair_sate_img_list = pair_drone2sate.get(f'pair_{mode}_sate_img_list', [])
            pair_sate_weight_list = pair_drone2sate.get(f'pair_{mode}_sate_weight_list', [])
            
            # Handle missing weights (default to 1.0)
            if not pair_sate_weight_list:
                pair_sate_weight_list = [1.0] * len(pair_sate_img_list)
            
            drone_img_file = os.path.join(data_root, drone_img_dir, drone_img_name)

            for pair_sate_img, pair_sate_weight in zip(pair_sate_img_list, pair_sate_weight_list):
                sate_img_file = os.path.join(data_root, sate_img_dir, pair_sate_img)
                if os.path.exists(sate_img_file):  # Only add if file exists
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

        # Training with sparse data
        num_pairs = len(self.pairs)
        num_pairs_train = int(train_ratio * num_pairs)
        random.shuffle(self.pairs)
        self.pairs = self.pairs[:num_pairs_train]
        
        self.samples = copy.deepcopy(self.pairs)
        
        print(f"Loaded {len(self.pairs)} training pairs from {pairs_meta_file}")
    
    def __getitem__(self, index):
        
        query_img_path, gallery_img_path, positive_weight = self.samples[index]
        
        # Load query (drone) image
        query_img = cv2.imread(query_img_path)
        if query_img is None:
            raise ValueError(f"Could not load image: {query_img_path}")
        query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
        
        # Load gallery (satellite) image
        gallery_img = cv2.imread(gallery_img_path)
        if gallery_img is None:
            raise ValueError(f"Could not load image: {gallery_img_path}")
        gallery_img = cv2.cvtColor(gallery_img, cv2.COLOR_BGR2RGB)
        
        # Random horizontal flip (applied to both images)
        if np.random.random() < self.prob_flip:
            query_img = cv2.flip(query_img, 1)
            gallery_img = cv2.flip(gallery_img, 1) 
        
        # Apply transforms
        if self.transforms_query is not None:
            query_img = self.transforms_query(image=query_img)['image']
            
        if self.transforms_gallery is not None:
            gallery_img = self.transforms_gallery(image=gallery_img)['image']
        
        return query_img, gallery_img, positive_weight

    def __len__(self):
        return len(self.samples)

    def shuffle(self):
        """
        Implementation of Mutually Exclusive Sampling process.
        Ensures no duplicate drone or satellite images in the same batch.
        """
        
        print("\nShuffle Dataset:")
        
        pair_pool = copy.deepcopy(self.pairs)
        random.shuffle(pair_pool)
        
        sate_batch = set()
        drone_batch = set()
        pairs_epoch = set()
        
        batches = []
        current_batch = []
        break_counter = 0

        while True:
            if len(pair_pool) > 0:
                pair = pair_pool.pop(0)
                
                drone_img, sate_img, _ = pair
                drone_img_name = drone_img.split('/')[-1]
                sate_img_name = sate_img.split('/')[-1]

                pair_name = (drone_img_name, sate_img_name)

                if drone_img_name not in drone_batch and sate_img_name not in sate_batch and pair_name not in pairs_epoch:
                    current_batch.append(pair)
                    pairs_epoch.add(pair_name)
                    
                    # Add all related sate images to batch exclusion
                    pairs_drone2sate = self.pairs_drone2sate_dict.get(drone_img_name, [])
                    for sate in pairs_drone2sate:
                        sate_batch.add(sate)
                    
                    # Add all related drone images to batch exclusion
                    pairs_sate2drone = self.pairs_sate2drone_dict.get(sate_img_name, [])
                    for drone in pairs_sate2drone:
                        drone_batch.add(drone)
                    
                    break_counter = 0
                else:
                    if pair_name not in pairs_epoch:
                        pair_pool.append(pair)
                    break_counter += 1
                    
                if break_counter >= 16384:
                    break
            else:
                break

            if len(current_batch) >= self.shuffle_batch_size:
                batches.extend(current_batch)
                sate_batch = set()
                drone_batch = set()
                current_batch = []
        
        self.samples = batches
        
        print(f"Original Length: {len(self.pairs)} - Length after Shuffle: {len(self.samples)}")
        print(f"Break Counter: {break_counter}")
        if len(self.samples) > 0:
            print(f"First Element: {self.samples[0][0]} - Last Element: {self.samples[-1][0]}")

    def shuffle_group(self):
        """
        Implementation of Mutually Exclusive Sampling process with group.
        For group-based contrastive learning.
        """
        print("\nShuffle Dataset in Batches (Group Mode):")
        
        pair_pool = copy.deepcopy(self.pairs)
        random.shuffle(pair_pool)
        
        sate_batch = set()
        drone_batch = set()
        pairs_epoch = set()
        
        batches = []
        current_batch = []
        break_counter = 0

        while True:
            if len(pair_pool) > 0:
                if break_counter >= 16384:
                    break

                pair = pair_pool.pop(0)
                
                drone_img_path, sate_img_path, _ = pair
                drone_img_dir = os.path.dirname(drone_img_path)
                sate_img_dir = os.path.dirname(sate_img_path)

                drone_img_name_i = drone_img_path.split('/')[-1]
                sate_img_name_i = sate_img_path.split('/')[-1]

                pair_name = (drone_img_name_i, sate_img_name_i)

                if drone_img_name_i in drone_batch or pair_name in pairs_epoch:
                    if pair_name not in pairs_epoch:
                        pair_pool.append(pair)
                    break_counter += 1
                    continue

                pairs_drone2sate = self.pairs_drone2sate_dict.get(drone_img_name_i, [])
                if not pairs_drone2sate:
                    break_counter += 1
                    continue
                    
                random.shuffle(pairs_drone2sate)

                subset_sate_len = itertools.combinations(pairs_drone2sate, min(self.group_len, len(pairs_drone2sate)))
                
                subset_drone = None
                subset_sate = None
                
                for subset_sate_i in subset_sate_len:
                    flag = True
                    sate2drone_inter_set = None

                    for sate_img in subset_sate_i:
                        if sate_img in sate_batch:
                            flag = False
                            break
                        
                        if sate2drone_inter_set is None:
                            sate2drone_inter_set = set(self.pairs_sate2drone_dict.get(sate_img, []))
                        else:
                            sate2drone_inter_set = sate2drone_inter_set.intersection(
                                self.pairs_sate2drone_dict.get(sate_img, []))
                        
                    if not flag or sate2drone_inter_set is None or len(sate2drone_inter_set) < self.group_len:
                        continue

                    sate2drone_inter_set = list(sate2drone_inter_set)
                    random.shuffle(sate2drone_inter_set)
                    subset_drone_len = itertools.combinations(sate2drone_inter_set, self.group_len)
                    
                    for subset_drone_i in subset_drone_len:
                        if drone_img_name_i not in subset_drone_i:
                            continue
                        flag = True
                        for drone_img in subset_drone_i:
                            if drone_img in drone_batch or not flag:
                                flag = False
                                break
                            for sate_img in subset_sate_i:
                                pair_tmp = (drone_img, sate_img)
                                if pair_tmp in pairs_epoch:
                                    flag = False
                                    break
                        if flag:
                            subset_drone = subset_drone_i
                            subset_sate = subset_sate_i
                            break
                
                if subset_drone is not None and subset_sate is not None:
                    for drone_img_name, sate_img_name in zip(subset_drone, subset_sate):
                        drone_img_path = os.path.join(self.data_root, drone_img_dir.replace(self.data_root, '').lstrip('/'), drone_img_name)
                        sate_img_path = os.path.join(self.data_root, sate_img_dir.replace(self.data_root, '').lstrip('/'), sate_img_name)
                        current_batch.append((drone_img_path, sate_img_path, 1.0))
                        pairs_epoch.add((drone_img_name, sate_img_name))
                    
                    for drone_img in subset_drone:
                        for sate in self.pairs_drone2sate_dict.get(drone_img, []):
                            sate_batch.add(sate)
                    for sate_img in subset_sate:
                        for drone in self.pairs_sate2drone_dict.get(sate_img, []):
                            drone_batch.add(drone)
                else:
                    if pair_name not in pairs_epoch:
                        pair_pool.append(pair)
                    break_counter += 1

                if break_counter >= 16384:
                    break
            else:
                break
                
            if len(current_batch) >= self.shuffle_batch_size:
                batches.extend(current_batch)
                sate_batch = set()
                drone_batch = set()
                current_batch = []
        
        self.samples = batches
        
        print(f"Original Length: {len(self.pairs)} - Length after Shuffle: {len(self.samples)}")
        print(f"Break Counter: {break_counter}")


class GTAUnifiedDatasetEval(Dataset):
    """
    Unified evaluation dataset compatible with GTA-UAV pipeline.
    Supports datasets with either:
    - drone_loc_x_y (GTA-UAV format)
    - drone_loc_lat_lon (UAV_VisLoc, VPair format)
    """
    
    def __init__(self,
                 pairs_meta_file,
                 data_root,
                 view,
                 mode='pos',
                 sate_img_dir='',
                 query_mode='D2S',
                 pairs_sate2drone_dict=None,
                 transforms=None,
                 use_latlon=True,  # Whether coordinates are lat/lon (True) or x/y (False)
                 ):
        super().__init__()
        
        with open(os.path.join(data_root, pairs_meta_file), 'r', encoding='utf-8') as f:
            pairs_meta_data = json.load(f)
        self.data_root = data_root
        self.use_latlon = use_latlon
        sate_img_dir_full = os.path.join(data_root, sate_img_dir)

        self.images_path = []
        self.images_name = []
        self.images_center_loc_xy = []
        self.images_topleft_loc_xy = []

        self.pairs_sate2drone_dict = {}
        self.pairs_drone2sate_dict = {}
        self.pairs_match_set = set()

        if view == 'drone':
            for pair_drone2sate in pairs_meta_data:
                drone_img_name = pair_drone2sate['drone_img_name']
                drone_img_dir = pair_drone2sate['drone_img_dir']
                
                # Support both coordinate formats
                if 'drone_loc_x_y' in pair_drone2sate:
                    drone_loc = pair_drone2sate['drone_loc_x_y']
                elif 'drone_loc_lat_lon' in pair_drone2sate:
                    drone_loc = pair_drone2sate['drone_loc_lat_lon']
                else:
                    drone_loc = [0, 0]  # Default if no location
                
                self.pairs_drone2sate_dict[drone_img_name] = []
                pair_sate_img_list = pair_drone2sate.get(f'pair_{mode}_sate_img_list', [])
                
                for pair_sate_img in pair_sate_img_list:
                    self.pairs_drone2sate_dict.setdefault(drone_img_name, []).append(pair_sate_img)
                    self.pairs_sate2drone_dict.setdefault(pair_sate_img, []).append(drone_img_name)
                    self.pairs_match_set.add((drone_img_name, pair_sate_img))
                
                if len(pair_sate_img_list) != 0:
                    img_path = os.path.join(data_root, drone_img_dir, drone_img_name)
                    if os.path.exists(img_path):
                        self.images_path.append(img_path)
                        self.images_name.append(drone_img_name)
                        self.images_center_loc_xy.append((drone_loc[0], drone_loc[1]))

        elif view == 'sate':
            # Build sate location lookup from metadata
            sate_loc_dict = {}
            for pair_drone2sate in pairs_meta_data:
                # Support both coordinate formats
                loc_key = 'pair_pos_sate_loc_x_y_list' if 'pair_pos_sate_loc_x_y_list' in pair_drone2sate else 'pair_pos_sate_loc_lat_lon_list'
                semipos_loc_key = 'pair_pos_semipos_sate_loc_x_y_list' if 'pair_pos_semipos_sate_loc_x_y_list' in pair_drone2sate else 'pair_pos_semipos_sate_loc_lat_lon_list'
                
                pair_sate_img_list = pair_drone2sate.get('pair_pos_sate_img_list', [])
                pair_sate_loc_list = pair_drone2sate.get(loc_key, [])
                
                for sate_img, sate_loc in zip(pair_sate_img_list, pair_sate_loc_list):
                    if sate_img not in sate_loc_dict:
                        sate_loc_dict[sate_img] = sate_loc
                
                # Also add semipos locations
                pair_semipos_sate_img_list = pair_drone2sate.get('pair_pos_semipos_sate_img_list', [])
                pair_semipos_sate_loc_list = pair_drone2sate.get(semipos_loc_key, [])
                
                for sate_img, sate_loc in zip(pair_semipos_sate_img_list, pair_semipos_sate_loc_list):
                    if sate_img not in sate_loc_dict:
                        sate_loc_dict[sate_img] = sate_loc
            
            if query_mode == 'D2S':
                sate_img_dir_list, sate_img_list = get_sate_data(sate_img_dir_full)
                for sate_dir, sate_img in zip(sate_img_dir_list, sate_img_list):
                    img_path = os.path.join(sate_dir, sate_img)
                    if os.path.exists(img_path):
                        self.images_path.append(img_path)
                        self.images_name.append(sate_img)
                        
                        # Get location from lookup or default
                        loc = sate_loc_dict.get(sate_img, [0, 0])
                        self.images_center_loc_xy.append((loc[0], loc[1]) if loc else (0, 0))
                        self.images_topleft_loc_xy.append((loc[0], loc[1]) if loc else (0, 0))
            else:
                # S2D mode - only include sate images that have matches
                sate_img_dir_list, sate_img_list = get_sate_data(sate_img_dir_full)
                for sate_dir, sate_img in zip(sate_img_dir_list, sate_img_list):
                    if pairs_sate2drone_dict and sate_img not in pairs_sate2drone_dict:
                        continue
                    img_path = os.path.join(sate_dir, sate_img)
                    if os.path.exists(img_path):
                        self.images_path.append(img_path)
                        self.images_name.append(sate_img)
                        
                        loc = sate_loc_dict.get(sate_img, [0, 0])
                        self.images_center_loc_xy.append((loc[0], loc[1]) if loc else (0, 0))
                        self.images_topleft_loc_xy.append((loc[0], loc[1]) if loc else (0, 0))

        self.transforms = transforms
        print(f"Loaded {len(self.images_path)} {view} images for evaluation")

    def __getitem__(self, index):
        img_path = self.images_path[index]
        
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Could not load image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.transforms is not None:
            img = self.transforms(image=img)['image']
        
        return img

    def __len__(self):
        return len(self.images_name)


def get_transforms(img_size,
                   mean=[0.485, 0.456, 0.406],
                   std=[0.229, 0.224, 0.225],
                   sat_rot=False,
                   altitude_aug_prob=0.7,  # Higher probability for low-altitude training
                   altitude_scale_range=(0.15, 0.45)):  # Aggressive crop for 70-150m altitude
    """
    Get transforms for training and validation.
    
    For low-altitude drones (70-150m), use:
    - altitude_aug_prob=0.7 (high probability of crop augmentation)
    - altitude_scale_range=(0.15, 0.45) (aggressive center crop to simulate altitude variation)
    
    For high-altitude drones (300-400m like VPair), use:
    - altitude_aug_prob=0.5
    - altitude_scale_range=(0.25, 0.55)
    """

    val_transforms = A.Compose([
        A.Resize(img_size[0], img_size[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
        A.Normalize(mean, std),
        ToTensorV2(),
    ])
    
    p_rot = 1.0 if sat_rot else 0.0
                                
    train_sat_transforms = A.Compose([
        RandomCenterCropZoom(scale_limit=altitude_scale_range, p=altitude_aug_prob),
        A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5),
        A.Resize(img_size[0], img_size[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
        A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15, always_apply=False, p=0.5),
        A.OneOf([
            A.AdvancedBlur(p=1.0),
            A.Sharpen(p=1.0),
        ], p=0.3),
        A.OneOf([
            A.GridDropout(ratio=0.4, p=1.0),
            A.CoarseDropout(max_holes=25,
                            max_height=int(0.2*img_size[0]),
                            max_width=int(0.2*img_size[0]),
                            min_holes=10,
                            min_height=int(0.1*img_size[0]),
                            min_width=int(0.1*img_size[0]),
                            p=1.0),
        ], p=0.3),
        A.RandomRotate90(p=p_rot),
        A.Normalize(mean, std),
        ToTensorV2(),
    ])

    train_drone_transforms = A.Compose([
        RandomCenterCropZoom(scale_limit=altitude_scale_range, p=altitude_aug_prob),
        A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5),
        A.Resize(img_size[0], img_size[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
        A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15, always_apply=False, p=0.5),
        A.OneOf([
            A.AdvancedBlur(p=1.0),
            A.Sharpen(p=1.0),
        ], p=0.3),
        A.OneOf([
            A.GridDropout(ratio=0.4, p=1.0),
            A.CoarseDropout(max_holes=25,
                            max_height=int(0.2*img_size[0]),
                            max_width=int(0.2*img_size[0]),
                            min_holes=10,
                            min_height=int(0.1*img_size[0]),
                            min_width=int(0.1*img_size[0]),
                            p=1.0),
        ], p=0.3),
        A.RandomRotate90(p=1.0),
        A.Normalize(mean, std),
        ToTensorV2(),
    ])
    
    return val_transforms, train_sat_transforms, train_drone_transforms


if __name__ == "__main__":
    # Test the unified dataset
    print("Testing GTAUnifiedDatasetTrain...")
    
    # Example: Test with VPair dataset
    # dataset = GTAUnifiedDatasetTrain(
    #     pairs_meta_file='vpair_train.json',
    #     data_root='/path/to/datasets',
    #     mode='pos_semipos'
    # )
    # print(f"Dataset size: {len(dataset)}")
