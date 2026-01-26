import os
import sys
import torch
import torch.nn.functional as F
import argparse
import numpy as np
import cv2
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add module path
sys.path.append(os.getcwd())

from game4loc.dataset.gta import GTADatasetEval, get_transforms
from game4loc.models.model_convnext_boq import DesModelConvNextBoQ

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True, help='Root directory of data')
    parser.add_argument('--test_pairs_meta_file', type=str, required=True, help='Path to test json file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--gpu_ids', type=str, default='0', help='GPU IDs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation')
    parser.add_argument('--img_size', type=int, default=256, help='Image size')
    parser.add_argument('--num_queries', type=int, default=8, help='BoQ queries')
    parser.add_argument('--vis_dir', type=str, default='vis_output', help='Output directory for visualization')
    parser.add_argument('--vis_count', type=int, default=10, help='Number of queries to visualize')
    return parser.parse_args()

def load_model(args, device):
    print(f"Loading model from {args.checkpoint}...")
    # Initialize model with same params as training
    model = DesModelConvNextBoQ(
        model_name='convnext_tiny.dinov3_lvd1689m',
        pretrained=False, # Load weights from checkpoint
        img_size=args.img_size,
        num_queries=args.num_queries,
        mlp_output_dim=512,
        use_lora=True, 
        lora_r=8,
        lora_alpha=16
    )
    
    # Load state dict
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    # PEFT model saving mechanism can be tricky. 
    # If saved with `torch.save(model.state_dict())`, keys usually have `base_model.model...` prefix if LoRA.
    # DesModelConvNextBoQ structure: self.backbone (PeftModel) -> ...
    
    # Let's inspect keys briefly
    msg = model.load_state_dict(checkpoint, strict=False)
    print(f"Load status: {msg}")
    
    model.to(device)
    model.eval()
    return model

def predict(model, loader, device):
    features = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting features"):
            batch = batch.to(device)
            # Forward depends on if it is drone (img1) or satellite (img2)
            # The loader returns just images. 
            # In DesModelConvNextBoQ, forward(img1=x) works for query.
            # forward(img2=x) works for gallery.
            # But here we just want features. forward_one is what we need.
            # However, we can use the main forward with correct arg.
            # But the dataset loader in Eval mode returns (img, index) or just img?
            # GTADatasetEval returns (img, index) usually.
            
            # Since DesModelConvNextBoQ uses `forward_one` internally with backbone/proj, 
            # we can assume symmetric extraction for now OR we use `img1` for everything if weights shared.
            # Weights ARE shared by default in our training script.
            
            emb = model(img1=batch) # Use img1 path
            
            if isinstance(emb, tuple): # BoQ might return (desc, attn)
                emb = emb[0]
                
            features.append(emb.cpu())
    return torch.cat(features, dim=0)

def visualize_matches(args, query_dataset, gallery_dataset, query_feats, gallery_feats, indices, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # Normalize
    query_feats = F.normalize(query_feats, p=2, dim=1)
    gallery_feats = F.normalize(gallery_feats, p=2, dim=1)
    
    # Similarity
    sim_matrix = torch.mm(query_feats, gallery_feats.t())
    
    # Top-k
    k = 5
    _, topk_indices = torch.topk(sim_matrix, k=k, dim=1)
    
    for i in range(min(len(indices), args.vis_count)):
        idx = indices[i] # Index in query dataset
        
        # Get query image path
        # Use images_path directly if available
        if hasattr(query_dataset, 'images_path'):
            q_path = query_dataset.images_path[idx]
        else:
            q_name = query_dataset.images_name[idx]
            q_path = os.path.join(args.data_root, q_name)
            
        q_name = query_dataset.images_name[idx]
        
        # Load query image
        q_img = cv2.imread(q_path)
        if q_img is None:
            print(f"Failed to read {q_path}")
            continue
        q_img = cv2.resize(q_img, (args.img_size, args.img_size))
        
        # Create canvas
        # Query | Match 1 | Match 2 ...
        canvas = [q_img]
        
        for rank, g_idx in enumerate(topk_indices[idx]):
            if hasattr(gallery_dataset, 'images_path'):
                g_path = gallery_dataset.images_path[g_idx]
            else:
                g_name = gallery_dataset.images_name[g_idx]
                g_path = os.path.join(args.data_root, g_name)
            
            g_name = gallery_dataset.images_name[g_idx]
            
            g_img = cv2.imread(g_path)
            if g_img is None:
                g_img = np.zeros_like(q_img)
            else:
                g_img = cv2.resize(g_img, (args.img_size, args.img_size))
            
            # Add border: Green if ground truth (how to check?), Red otherwise?
            # GTADatasetEval has pairs_drone2sate_dict: query_index -> list(gallery_names)
            # Actually pairs_drone2sate_dict maps Q_NAME to list(G_NAMES).
            gt_names = query_dataset.pairs_drone2sate_dict.get(q_name, [])
             
            # Sometimes names in dict might include subfolders, verify format
            # Let's assume loose matching if exact fails
            is_match = g_name in gt_names
            
            color = (0, 255, 0) if is_match else (0, 0, 255)
            cv2.rectangle(g_img, (0,0), (args.img_size-1, args.img_size-1), color, 5)
            
            canvas.append(g_img)
            
        # Concatenate
        row = np.hstack(canvas)
        out_path = os.path.join(output_dir, f"query_{i}_{os.path.basename(q_name)}")
        cv2.imwrite(out_path, row)
        
    print(f"Visualizations saved to {output_dir}")

def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu_ids}" if torch.cuda.is_available() else "cpu")
    
    # Data params matching training
    # We need mean/std from usual DINOv3 config or fallback
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    
    val_transforms, _, _ = get_transforms((args.img_size, args.img_size), mean=mean, std=std)
    
    # Load Datasets
    # Query (Drone)
    query_dataset = GTADatasetEval(
        data_root=args.data_root,
        pairs_meta_file=args.test_pairs_meta_file,
        view='drone',
        transforms=val_transforms,
        mode='pos_semipos',
        sate_img_dir='satellite', # Adjust if needed logic inside Dataset
        query_mode='D2S'
    )
    
    # Gallery (Satellite)
    gallery_dataset = GTADatasetEval(
        data_root=args.data_root,
        pairs_meta_file=args.test_pairs_meta_file,
        view='sate',
        transforms=val_transforms,
        mode='pos_semipos',
        sate_img_dir='satellite', # Adjust if needed
        query_mode='D2S'
    )
    
    query_loader = DataLoader(query_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    gallery_loader = DataLoader(gallery_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Load Model
    model = load_model(args, device)
    
    # Extract
    print("Extracting query features...")
    query_feats = predict(model, query_loader, device)
    print("Extracting gallery features...")
    gallery_feats = predict(model, gallery_loader, device)
    
    # Normalize
    query_feats = F.normalize(query_feats, p=2, dim=1)
    gallery_feats = F.normalize(gallery_feats, p=2, dim=1)
    
    # Calculate R@k
    sim = torch.mm(query_feats, gallery_feats.t())
    
    # Ground truth mapping
    # GTADatasetEval: self.pairs_drone2sate_dict[drone_name] = [sat_name1, sat_name2...]
    # We need to map indices.
    # query_dataset.images_name list aligns with query_feats rows
    # gallery_dataset.images_name list aligns with gallery_feats cols
    
    # Build GT matrix
    # This might be slow if loop. But dataset size is small (~400 queries).
    gt_map = []
    
    # Map gallery name to index for fast lookup
    gal_name_to_idx = {name: i for i, name in enumerate(gallery_dataset.images_name)}
    
    recall_1 = 0
    recall_5 = 0
    recall_10 = 0
    
    topk_indices = torch.topk(sim, k=10, dim=1).indices.cpu().numpy()
    
    for i, q_name in enumerate(query_dataset.images_name):
        gt_names = query_dataset.pairs_drone2sate_dict.get(q_name, [])
        gt_indices = [gal_name_to_idx[g] for g in gt_names if g in gal_name_to_idx]
        
        if i == 0:
            print(f"DEBUG: q_name={q_name}")
            print(f"DEBUG: gt_names (first 3)={gt_names[:3]}")
            print(f"DEBUG: Found gt_indices count={len(gt_indices)}")
            if len(gt_indices) == 0:
                print(f"DEBUG: Sample gallery keys={list(gal_name_to_idx.keys())[:5]}")
        
        if not gt_indices:
            continue
            
        retrieved_indices = topk_indices[i]
        
        if np.intersect1d(retrieved_indices[:1], gt_indices).size > 0:
            recall_1 += 1
        if np.intersect1d(retrieved_indices[:5], gt_indices).size > 0:
            recall_5 += 1
        if np.intersect1d(retrieved_indices[:10], gt_indices).size > 0:
            recall_10 += 1
            
    n = len(query_dataset.images_name)
    print(f"R@1: {recall_1/n:.4f}")
    print(f"R@5: {recall_5/n:.4f}")
    print(f"R@10: {recall_10/n:.4f}")
    
    # Visualizations
    visualize_matches(
        args, 
        query_dataset, 
        gallery_dataset, 
        query_feats, 
        gallery_feats, 
        range(n), 
        args.vis_dir
    )

if __name__ == '__main__':
    main()
