import os
import json
import pandas as pd
import argparse

def process_vpair(root_dir, output_path):
    # Define paths relative to the dataset root
    vpair_root = "vpair"
    drone_dir = os.path.join(vpair_root, "queries")
    sate_dir = os.path.join(vpair_root, "reference_views")
    
    # Read poses.csv to get the list of valid files
    poses_path = os.path.join(root_dir, "vpair", "poses.csv")
    
    if not os.path.exists(poses_path):
        print(f"Error: poses.csv not found at {poses_path}")
        return

    print(f"Reading metadata from {poses_path}...")
    df = pd.read_csv(poses_path)
    
    data_list = []
    
    for index, row in df.iterrows():
        filename = row['filename']
        lat = row['lat']
        lon = row['lon']
        alt = row['altitude']
        roll = row['roll']
        pitch = row['pitch']
        yaw = row['yaw']
        
        # Construct the entry
        # VPair has 1-to-1 correspondence between query and reference with same filename
        # We populate all fields to match GTA-UAV / VisLoc format exactly
        
        entry = {
            "drone_img_dir": drone_dir,
            "drone_img_name": filename,
            "drone_loc_lat_lon": [lat, lon],
            "sate_img_dir": sate_dir,
            
            # Positive pairs (exact match)
            "pair_pos_sate_img_list": [filename],
            "pair_pos_sate_weight_list": [1.0],
            "pair_pos_sate_loc_lat_lon_list": [[lat, lon]],
            
            # Positive + Semi-positive pairs (same as positive for VPair 1:1)
            "pair_pos_semipos_sate_img_list": [filename],
            "pair_pos_semipos_sate_weight_list": [1.0],
            "pair_pos_semipos_sate_loc_lat_lon_list": [[lat, lon]],
            
            "drone_metadata": {
                "height": alt,
                "drone_roll": roll,
                "drone_pitch": pitch,
                "drone_yaw": yaw,
                "cam_roll": None, # Not provided in poses.csv
                "cam_pitch": None,
                "cam_yaw": None,
            }
        }
        data_list.append(entry)
        
    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(data_list, f, indent=4)
    
    print(f"Processed {len(data_list)} pairs. Saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare VPair dataset for Game4Loc")
    parser.add_argument("--data_root", type=str, default="/home/aniel/skyline_drone/datasets", help="Root directory of datasets")
    parser.add_argument("--output", type=str, default="vpair_train.json", help="Output JSON file name")
    
    args = parser.parse_args()
    
    process_vpair(args.data_root, args.output)
