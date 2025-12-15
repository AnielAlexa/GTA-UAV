#!/usr/bin/env python3
# ---------------------------------------------------------------
# OrthoLoC to GTA-UAV Dataset Conversion Script
# Converts OrthoLoC dataset to GTA-UAV format for training
# ---------------------------------------------------------------

import os
import json
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import shutil
import argparse
import random

# Constants
CROP_SIZE = 384
DOP_SIZE = 1024

# Target physical offsets (meters)
MIN_OFFSET_METERS = 20  # 20 meters
MAX_OFFSET_METERS = 30  # 30 meters

# Note: Pixel offsets calculated dynamically based on actual GSD per image

# IoU thresholds (GTA-UAV standard)
THRESHOLD_POS = 0.39
THRESHOLD_SEMI = 0.14

# Geographic split
TRAIN_LOCS = list(range(1, 41))   # L01-L40
VAL_LOCS = list(range(41, 47))    # L41-L46
TEST_LOCS = list(range(47, 52))   # L47-L51


def extract_geotransform(dop_image_pil):
    """
    Extract GeoTIFF geotransform from DOP image.

    Returns:
        (origin_x, pixel_width, origin_y, pixel_height)
    """
    try:
        # ModelPixelScaleTag (33550): Pixel scale in world units
        pixel_scale = dop_image_pil.tag_v2.get(33550)
        if pixel_scale is None:
            raise ValueError("ModelPixelScaleTag not found")
        pixel_width, pixel_height = pixel_scale[0], pixel_scale[1]

        # ModelTiepointTag (33922): Maps pixel coords to world coords
        tiepoint = dop_image_pil.tag_v2.get(33922)
        if tiepoint is None:
            raise ValueError("ModelTiepointTag not found")
        # Format: (pixel_x, pixel_y, pixel_z, world_x, world_y, world_z)
        origin_x = tiepoint[3]
        origin_y = tiepoint[4]

        return origin_x, pixel_width, origin_y, pixel_height
    except Exception as e:
        raise RuntimeError(f"Failed to extract geotransform: {e}")


def world_to_pixel(drone_x, drone_y, geotransform):
    """
    Convert drone world coordinates to DOP pixel coordinates.

    Note: GeoTIFF uses top-down pixel coordinates (Y increases downward)
    while world coordinates use standard Cartesian (Y increases upward).
    Therefore, Y-axis calculation is inverted.

    Returns:
        (pixel_x, pixel_y)
    """
    origin_x, pixel_width, origin_y, pixel_height = geotransform
    pixel_x = int((drone_x - origin_x) / pixel_width)
    pixel_y = int((origin_y - drone_y) / pixel_height)  # Inverted Y-axis
    return pixel_x, pixel_y


def validate_crop_bounds(drone_pixel_x, drone_pixel_y, crop_size=CROP_SIZE, dop_size=DOP_SIZE):
    """
    Check if crop around drone position fits within DOP bounds.

    Returns: True if valid, False if near edge
    """
    half_crop = crop_size // 2  # 192

    if (drone_pixel_x < half_crop or
        drone_pixel_x > dop_size - half_crop or
        drone_pixel_y < half_crop or
        drone_pixel_y > dop_size - half_crop):
        return False  # Out of bounds

    return True


def calculate_crop_iou(x1, y1, x2, y2, size=CROP_SIZE):
    """
    Calculate IoU between two square crops.

    Args:
        x1, y1: Top-left of crop 1
        x2, y2: Top-left of crop 2
        size: Crop size (384)

    Returns:
        IoU value [0, 1]
    """
    # Bounding boxes
    box1 = [x1, y1, x1 + size, y1 + size]
    box2 = [x2, y2, x2 + size, y2 + size]

    # Intersection
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2], box2[2])
    inter_y2 = min(box1[3], box2[3])

    if inter_x2 < inter_x1 or inter_y2 < inter_y1:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)

    # Union
    box1_area = size * size
    box2_area = size * size
    union_area = box1_area + box2_area - inter_area

    iou = inter_area / union_area
    return iou


def extract_pose_from_extrinsics(extrinsics):
    """
    Extract camera pose (roll, pitch, yaw, height, x, y) from 3x4 extrinsics.

    Args:
        extrinsics: 3x4 numpy array [R | t]

    Returns:
        dict with keys: roll, pitch, yaw (degrees), height, x, y
    """
    # Split into rotation and translation
    R = extrinsics[:, :3]  # 3x3 rotation matrix
    t = extrinsics[:, 3]   # Translation vector

    # Extract Euler angles from rotation matrix
    rotation = Rotation.from_matrix(R)
    roll, pitch, yaw = rotation.as_euler('xyz', degrees=True)

    # Extract position
    x, y, height = t[0], t[1], abs(t[2])

    return {
        'roll': float(roll),
        'pitch': float(pitch),
        'yaw': float(yaw),
        'height': float(height),
        'x': float(x),
        'y': float(y)
    }


def generate_satellite_crops(dop_img, drone_pixel_x, drone_pixel_y, location, reference, pose, gsd):
    """
    Generate 4 satellite crops: 1 center + 3 random offsets.

    Args:
        gsd: Ground Sample Distance (meters per pixel) for this DOP

    Returns:
        List of (crop_img, crop_name, offset_x, offset_y, iou, world_x, world_y) or None
    """
    half_crop = CROP_SIZE // 2

    # Validate boundaries
    if not validate_crop_bounds(drone_pixel_x, drone_pixel_y):
        return None

    # Calculate pixel offsets based on GSD for this image
    min_offset_pixels = int(MIN_OFFSET_METERS / gsd)
    max_offset_pixels = int(MAX_OFFSET_METERS / gsd)

    crops = []

    # Center crop (positive pair, IoU = 1.0)
    center_x = drone_pixel_x - half_crop
    center_y = drone_pixel_y - half_crop
    center_crop = dop_img[center_y:center_y+CROP_SIZE, center_x:center_x+CROP_SIZE]
    center_name = f"{location}_0_{reference}_center.png"
    crops.append((center_crop, center_name, 0, 0, 1.0, pose['x'], pose['y']))

    # Generate 3 random offset crops (semi-positive pairs)
    for i in range(3):
        # Random offset (pixel-based, scaled by GSD)
        offset_x = random.randint(-max_offset_pixels, max_offset_pixels)
        offset_y = random.randint(-max_offset_pixels, max_offset_pixels)

        # Calculate crop position
        crop_x = center_x + offset_x
        crop_y = center_y + offset_y

        # Clamp to valid range
        crop_x = max(0, min(crop_x, DOP_SIZE - CROP_SIZE))
        crop_y = max(0, min(crop_y, DOP_SIZE - CROP_SIZE))

        # Extract crop
        offset_crop = dop_img[crop_y:crop_y+CROP_SIZE, crop_x:crop_x+CROP_SIZE]
        offset_name = f"{location}_0_{reference}_offset{i}.png"

        # Calculate IoU
        iou = calculate_crop_iou(center_x, center_y, crop_x, crop_y, CROP_SIZE)

        # Calculate world coordinates (adjusted by offset and GSD)
        # Note: Y-axis inverted (pixel Y+ = world Y-)
        world_x = pose['x'] + (offset_x * gsd)
        world_y = pose['y'] - (offset_y * gsd)

        crops.append((offset_crop, offset_name, offset_x, offset_y, iou, world_x, world_y))

    return crops


def process_single_query(args):
    """
    Process a single query image.

    Args:
        Tuple of (base_name, queries_dir, dops_dir, cameras_dir,
                  drone_output_dir, sate_output_dir)

    Returns:
        Metadata dict or None if error/boundary violation
    """
    base_name, queries_dir, dops_dir, cameras_dir, drone_out, sate_out = args

    try:
        # Paths
        query_path = os.path.join(queries_dir, f"{base_name}.jpg")
        dop_path = os.path.join(dops_dir, f"{base_name}.tif")
        camera_path = os.path.join(cameras_dir, f"{base_name}.json")

        # Check existence
        if not all([os.path.exists(p) for p in [query_path, dop_path, camera_path]]):
            return None

        # 1. Copy query image
        shutil.copy(query_path, os.path.join(drone_out, f"{base_name}.jpg"))

        # 2. Load camera JSON
        with open(camera_path, 'r') as f:
            camera_data = json.load(f)

        extrinsics = np.array(camera_data['extrinsics'])
        pose = extract_pose_from_extrinsics(extrinsics)

        # 3. Load DOP as PIL image first (for geotransform), then convert to numpy
        dop_pil = Image.open(dop_path)
        geotransform = extract_geotransform(dop_pil)
        gsd = geotransform[1]  # Pixel width (GSD)
        dop_img = np.array(dop_pil)

        # 4. Convert drone world coords to DOP pixel coords
        drone_pixel_x, drone_pixel_y = world_to_pixel(pose['x'], pose['y'], geotransform)

        # 5. Validate boundaries
        if not validate_crop_bounds(drone_pixel_x, drone_pixel_y):
            return None  # Skip samples near edge

        # 6. Generate satellite crops
        location = base_name.split('_')[0]  # Extract 'L01' from 'L01_R0000'
        reference = base_name.split('_')[1]  # Extract 'R0000'

        crops = generate_satellite_crops(dop_img, drone_pixel_x, drone_pixel_y,
                                        location, reference, pose, gsd)

        if crops is None:
            return None

        # 7. Save crops and build metadata
        pos_list = []
        pos_weights = []
        pos_locs = []
        semipos_list = []
        semipos_weights = []
        semipos_locs = []

        for crop_img, crop_name, offset_x, offset_y, iou, world_x, world_y in crops:
            # Save crop
            crop_path = os.path.join(sate_out, crop_name)
            Image.fromarray(crop_img).save(crop_path)

            # Add to semi-positive list
            semipos_list.append(crop_name)
            semipos_weights.append(float(iou))
            semipos_locs.append([world_x, world_y])

            # Add to positive list if IoU > threshold
            if iou > THRESHOLD_POS:
                pos_list.append(crop_name)
                pos_weights.append(float(iou))
                pos_locs.append([world_x, world_y])

        # 8. Create metadata entry
        pair_entry = {
            "drone_img_dir": "drone/images",
            "drone_img_name": f"{base_name}.jpg",
            "drone_loc_x_y": [pose['x'], pose['y']],
            "sate_img_dir": "satellite",
            "pair_pos_sate_img_list": pos_list,
            "pair_pos_sate_weight_list": pos_weights,
            "pair_pos_sate_loc_x_y_list": pos_locs,
            "pair_pos_semipos_sate_img_list": semipos_list,
            "pair_pos_semipos_sate_weight_list": semipos_weights,
            "pair_pos_semipos_sate_loc_x_y_list": semipos_locs,
            "drone_metadata": {
                "height": pose['height'],
                "drone_roll": pose['roll'],
                "drone_pitch": pose['pitch'],
                "drone_yaw": pose['yaw'],
                "cam_roll": None,
                "cam_pitch": None,
                "cam_yaw": None
            }
        }

        return pair_entry

    except Exception as e:
        print(f"Error processing {base_name}: {e}")
        return None


def get_split_for_location(location_id):
    """
    Determine split (train/val/test) for a location.

    Args:
        location_id: String like 'L01', 'L23', etc.

    Returns:
        'train', 'val', or 'test'
    """
    loc_num = int(location_id[1:])  # Extract number from 'L01'

    if loc_num in TRAIN_LOCS:
        return 'train'
    elif loc_num in VAL_LOCS:
        return 'val'
    elif loc_num in TEST_LOCS:
        return 'test'
    else:
        raise ValueError(f"Invalid location ID: {location_id}")


def process_split(split_root, drone_output_dir, sate_output_dir):
    """
    Process all queries in a split.

    Returns:
        List of valid metadata entries
    """
    queries_dir = os.path.join(split_root, 'queries')
    dops_dir = os.path.join(split_root, 'DOPs')
    cameras_dir = os.path.join(split_root, 'cameras')

    if not os.path.exists(queries_dir):
        print(f"Warning: {queries_dir} not found, skipping")
        return []

    query_files = sorted([f for f in os.listdir(queries_dir) if f.endswith('.jpg')])

    # Prepare arguments for multiprocessing
    args_list = []
    for query_file in query_files:
        base_name = query_file.replace('.jpg', '')
        args_list.append((
            base_name,
            queries_dir,
            dops_dir,
            cameras_dir,
            drone_output_dir,
            sate_output_dir
        ))

    # Process with multiprocessing
    with Pool(cpu_count()) as pool:
        results = list(tqdm(
            pool.imap(process_single_query, args_list),
            total=len(args_list),
            desc=f"Processing {os.path.basename(split_root)}"
        ))

    # Filter out None results
    valid_results = [r for r in results if r is not None]
    skipped = len(results) - len(valid_results)

    print(f"  Processed: {len(valid_results)}, Skipped (boundary violations): {skipped}")

    return valid_results


def main(ortholoc_raw_root, output_root):
    """
    Main conversion pipeline.
    """
    print("=" * 60)
    print("OrthoLoC to GTA-UAV Conversion")
    print("=" * 60)

    # Setup directories
    drone_dir = os.path.join(output_root, 'drone', 'images')
    sate_dir = os.path.join(output_root, 'satellite')
    os.makedirs(drone_dir, exist_ok=True)
    os.makedirs(sate_dir, exist_ok=True)

    # Storage for all pairs by geographic split
    all_pairs_train = []
    all_pairs_val = []
    all_pairs_test = []

    # Process each split
    for split in ['train', 'val', 'test_inPlace', 'test_outPlace']:
        split_root = os.path.join(ortholoc_raw_root, split)

        if not os.path.exists(split_root):
            print(f"Skipping {split} (not found)")
            continue

        print(f"\nProcessing {split}...")
        pairs = process_split(split_root, drone_dir, sate_dir)

        # Distribute to train/val/test based on location
        for pair in pairs:
            location = pair['drone_img_name'].split('_')[0]  # Extract 'L01'
            target_split = get_split_for_location(location)

            if target_split == 'train':
                all_pairs_train.append(pair)
            elif target_split == 'val':
                all_pairs_val.append(pair)
            elif target_split == 'test':
                all_pairs_test.append(pair)

    # Save JSON files
    print("\nSaving JSON files...")
    train_path = os.path.join(output_root, 'cross-area-drone2sate-train.json')
    val_path = os.path.join(output_root, 'cross-area-drone2sate-val.json')
    test_path = os.path.join(output_root, 'cross-area-drone2sate-test.json')

    with open(train_path, 'w') as f:
        json.dump(all_pairs_train, f, indent=2)

    with open(val_path, 'w') as f:
        json.dump(all_pairs_val, f, indent=2)

    with open(test_path, 'w') as f:
        json.dump(all_pairs_test, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("Conversion Complete!")
    print("=" * 60)
    print(f"Train: {len(all_pairs_train)} pairs")
    print(f"Val:   {len(all_pairs_val)} pairs")
    print(f"Test:  {len(all_pairs_test)} pairs")
    print(f"Total: {len(all_pairs_train) + len(all_pairs_val) + len(all_pairs_test)} pairs")
    print(f"\nOutput directory: {output_root}")
    print(f"  - {train_path}")
    print(f"  - {val_path}")
    print(f"  - {test_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert OrthoLoC dataset to GTA-UAV format")
    parser.add_argument("--ortholoc_raw", type=str, required=True,
                       help="Path to OrthoLoC_Raw directory")
    parser.add_argument("--output", type=str, required=True,
                       help="Output directory for converted dataset")

    args = parser.parse_args()

    main(args.ortholoc_raw, args.output)
