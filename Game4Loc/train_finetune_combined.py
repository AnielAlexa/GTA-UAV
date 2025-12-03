#!/usr/bin/env python3
"""
Fine-tuning script for UAV Visual Localization

This script fine-tunes a pre-trained GTA-UAV model on combined real-world datasets
(UAV_VisLoc + VPair) using virtual expansion for data augmentation.

Key features:
- Virtual expansion (N× oversampling with on-the-fly augmentation)
- Lower learning rate for fine-tuning (1e-5 vs 1e-3 for training from scratch)
- Aggressive altitude augmentation for low-altitude drones (70-150m)
- Combined dataset training to avoid catastrophic forgetting

Usage:
    python train_finetune_combined.py --checkpoint /path/to/weights.pth
    
    # Quick smoke test (2 epochs, subset of data)
    python train_finetune_combined.py --checkpoint /path/to/weights.pth --smoke_test
"""

import os
import time
import math
import shutil
import sys
import torch
import argparse
from dataclasses import dataclass
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from transformers import get_constant_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup, get_cosine_schedule_with_warmup

from game4loc.dataset.gta import GTADatasetEval
from game4loc.dataset.virtual_expand import (
    VirtualExpandDataset, 
    CombinedVirtualDataset, 
    CombinedDatasetEval,
    get_transforms_finetune, 
    get_transforms_val
)
from game4loc.utils import setup_system, Logger
from game4loc.trainer.trainer import train, train_with_weight
from game4loc.evaluate.gta import evaluate
from game4loc.loss import InfoNCE, WeightedInfoNCE, GroupInfoNCE, TripletLoss
from game4loc.models.model import DesModel
from game4loc.models.model_netvlad import DesModelWithVLAD


@dataclass
class Configuration:
    """Fine-tuning configuration for combined UAV datasets."""
    
    # Model - DINOv3 ViT-Small (matching your pre-trained weights)
    model: str = 'vit_small_patch16_dinov3.lvd1689m'
    model_hub: str = 'timm'
    with_netvlad: bool = False
    
    # Image size (ViT uses 224 by default, but can use 384)
    img_size: int = 384
 
    # Freeze layers (optional - set to True to freeze early layers)
    freeze_layers: bool = False
    frozen_stages = [0, 0, 0, 0]  # [1,1,0,0] to freeze first 2 stages

    # Training with sharing weights
    share_weights: bool = True
    
    # Training with weighted-InfoNCE
    with_weight: bool = True

    # Group settings
    train_in_group: bool = True
    group_len: int = 2
    loss_type = ["whole_slice", "part_slice"]

    # Unused in fine-tuning
    train_with_mix_data: bool = False
    train_with_recon: bool = False
    recon_weight: float = 0.1
    
    # Training settings
    mixed_precision: bool = True
    custom_sampling: bool = True
    seed: int = 42
    epochs: int = 20
    batch_size: int = 16                # Reduced for 8GB GPU with 384x384 images
    verbose: bool = False
    gpu_ids: tuple = (0,)               # Single GPU by default

    # Virtual expansion factors
    expansion_visloc: int = 10          # UAV_VisLoc: 768 × 10 = 7,680
    expansion_vpair: int = 5            # VPair: 1,353 × 5 = 6,765
    # Total effective: ~14,445 training pairs per epoch

    # Training with sparse data
    train_ratio: float = 1.0

    # Eval settings
    batch_size_eval: int = 64
    eval_every_n_epoch: int = 1
    normalize_features: bool = True
    eval_gallery_n: int = -1

    # Optimizer 
    clip_grad: float = 100.0
    decay_exclue_bias: bool = False
    grad_checkpointing: bool = False
    
    # Loss
    label_smoothing: float = 0.1
    k: float = 3
    
    # Learning Rate - MUCH LOWER for fine-tuning
    lr: float = 1e-5                     # 10× lower than training from scratch
    scheduler: str = "cosine"
    warmup_epochs: float = 0.5           # More warmup for stability
    lr_end: float = 1e-7

    # Augmentation - more aggressive for altitude variation
    prob_flip: float = 0.5
    altitude_aug_prob: float = 0.7       # High probability for altitude variation
    altitude_scale_range: tuple = (0.15, 0.45)  # For 70-150m altitude
    
    # Savepath for model checkpoints
    model_path: str = "./work_dir/finetune_combined"

    query_mode: str = "D2S"
    train_mode: str = "pos_semipos"
    test_mode: str = "pos_semipos"      # Use pos_semipos since pos list can be empty

    # Eval before training
    zero_shot: bool = True
    
    # Pre-trained checkpoint (REQUIRED for fine-tuning)
    checkpoint_start: str = None

    # Workers
    num_workers: int = 0 if os.name == 'nt' else 4 
    
    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu' 
    
    # CUDNN settings
    cudnn_benchmark: bool = True
    cudnn_deterministic: bool = False

    # Data paths - Combined dataset
    data_root: str = "/home/aniel/skyline_drone/datasets"
    train_pairs_meta_file: str = "combined-train.json"
    test_pairs_meta_file: str = "combined-test.json"
    sate_img_dir: str = "satellite"  # Not used for combined dataset

    # Per-dataset paths (for separate loading with different expansion)
    visloc_train_json: str = "UAV_VisLoc_dataset/cross-area-drone2sate-train.json"
    visloc_test_json: str = "UAV_VisLoc_dataset/cross-area-drone2sate-test.json"
    vpair_train_json: str = "vpair/cross-area-drone2drone-train.json"
    vpair_test_json: str = "vpair/cross-area-drone2drone-test.json"

    # Logging
    log_to_file: bool = False
    log_path: str = ""


def train_script(config):
    """Main training script with virtual expansion."""

    if config.log_to_file:
        f = open(config.log_path, 'w')
        sys.stdout = f

    save_time = "{}".format(time.strftime("%m%d%H%M%S"))
    model_path = "{}/{}/{}".format(config.model_path, config.model, save_time)

    if not os.path.exists(model_path):
        os.makedirs(model_path)
    shutil.copyfile(os.path.basename(__file__), "{}/train.py".format(model_path))

    # Redirect print to both console and log file
    sys.stdout = Logger(os.path.join(model_path, 'log.txt'))

    setup_system(seed=config.seed,
                 cudnn_benchmark=config.cudnn_benchmark,
                 cudnn_deterministic=config.cudnn_deterministic)
    
    print("=" * 70)
    print("FINE-TUNING ON COMBINED UAV DATASETS")
    print("=" * 70)
    print(f"Training save in path: {model_path}")
    print(f"Starting from checkpoint: {config.checkpoint_start}")
    print(f"Virtual expansion: VisLoc ×{config.expansion_visloc}, VPair ×{config.expansion_vpair}")

    #-----------------------------------------------------------------------------#
    # Model                                                                       #
    #-----------------------------------------------------------------------------#
    print(f"\nModel: {config.model}")

    if config.with_netvlad:
        model = DesModelWithVLAD(model_name=config.model, 
                    pretrained=True,
                    img_size=config.img_size,
                    share_weights=config.share_weights)
    else:
        model = DesModel(model_name=config.model, 
                        pretrained=True,
                        img_size=config.img_size,
                        share_weights=config.share_weights)
                        
    data_config = model.get_config()
    print(data_config)
    mean = data_config["mean"]
    std = data_config["std"]
    img_size = (config.img_size, config.img_size)
    
    # Activate gradient checkpointing
    if config.grad_checkpointing:
        model.set_grad_checkpointing(True)
    
    # Load pretrained Checkpoint (REQUIRED for fine-tuning)
    if config.checkpoint_start is not None:  
        print(f"\n>>> Loading pre-trained checkpoint: {config.checkpoint_start}")
        model_state_dict = torch.load(config.checkpoint_start, map_location='cpu')
        model.load_state_dict(model_state_dict, strict=False)
        print(">>> Checkpoint loaded successfully!")
    else:
        print("\n!!! WARNING: No checkpoint specified - training from scratch !!!")

    # Freeze layers if specified
    print(f"Freeze model layers: {config.freeze_layers}, stages: {config.frozen_stages}")
    if config.freeze_layers:
        model.freeze_layers(config.frozen_stages)

    # Data parallel
    print(f"GPUs available: {torch.cuda.device_count()}")
    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)
            
    model = model.to(config.device)

    print(f"\nImage Size: {img_size}")
    print(f"Mean: {mean}")
    print(f"Std: {std}")

    #-----------------------------------------------------------------------------#
    # DataLoader with Virtual Expansion                                           #
    #-----------------------------------------------------------------------------#
    print("\n" + "=" * 70)
    print("DATASET CONFIGURATION")
    print("=" * 70)

    # Transforms for fine-tuning (more aggressive augmentation)
    sat_transforms, drone_transforms = get_transforms_finetune(
        image_size_sat=img_size,
        img_size_ground=img_size,
        mean=mean,
        std=std,
        ground_cutting=0,
        altitude_aug_prob=config.altitude_aug_prob,
        altitude_scale_range=config.altitude_scale_range
    )
    
    val_sat_transforms, val_drone_transforms = get_transforms_val(
        image_size_sat=img_size,
        img_size_ground=img_size,
        mean=mean,
        std=std,
        ground_cutting=0
    )

    # Create datasets with different expansion factors
    # Note: Each dataset has its own data_root since image paths in JSON are relative to dataset dir
    print("\nCreating UAV_VisLoc dataset with virtual expansion...")
    visloc_data_root = os.path.join(config.data_root, "UAV_VisLoc_dataset")
    visloc_dataset = VirtualExpandDataset(
        data_root=visloc_data_root,
        pairs_meta_file="cross-area-drone2sate-train.json",
        expansion_factor=config.expansion_visloc,
        transforms_query=drone_transforms,
        transforms_gallery=sat_transforms,
        prob_flip=config.prob_flip,
        shuffle_batch_size=config.batch_size,
        mode=config.train_mode,
        train_ratio=config.train_ratio,
        group_len=config.group_len,
    )

    print("\nCreating VPair dataset with virtual expansion...")
    vpair_data_root = os.path.join(config.data_root, "vpair")
    vpair_dataset = VirtualExpandDataset(
        data_root=vpair_data_root,
        pairs_meta_file="cross-area-drone2drone-train.json",
        expansion_factor=config.expansion_vpair,
        transforms_query=drone_transforms,
        transforms_gallery=sat_transforms,
        prob_flip=config.prob_flip,
        shuffle_batch_size=config.batch_size,
        mode=config.train_mode,
        train_ratio=config.train_ratio,
        group_len=config.group_len,
    )

    # Combine datasets
    print("\nCombining datasets...")
    train_dataset = CombinedVirtualDataset([visloc_dataset, vpair_dataset])
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=not config.custom_sampling,
        pin_memory=True
    )
    
    print(f"\nTotal training samples (with expansion): {len(train_dataset)}")
    print(f"Batches per epoch: {len(train_dataloader)}")

    #-----------------------------------------------------------------------------#
    # Test DataLoader (use combined test set with CombinedDatasetEval)            #
    #-----------------------------------------------------------------------------#
    if config.query_mode == 'D2S':
        query_view = 'drone'
        gallery_view = 'sate'
    else:
        query_view = 'sate'
        gallery_view = 'drone'
    
    # Use CombinedDatasetEval which loads gallery from JSON instead of scanning dirs
    query_dataset_test = CombinedDatasetEval(
        data_root=config.data_root,
        pairs_meta_file=config.test_pairs_meta_file,
        view=query_view,
        mode=config.test_mode,
        transforms=val_drone_transforms,
    )
    query_img_list = query_dataset_test.images_name
    query_center_loc_xy_list = query_dataset_test.images_center_loc_xy
    pairs_drone2sate_dict = query_dataset_test.pairs_drone2sate_dict
    
    query_dataloader_test = DataLoader(
        query_dataset_test,
        batch_size=config.batch_size_eval,
        num_workers=config.num_workers,
        shuffle=False,
        pin_memory=True
    )
    
    gallery_dataset_test = CombinedDatasetEval(
        data_root=config.data_root,
        pairs_meta_file=config.test_pairs_meta_file,
        view=gallery_view,
        mode=config.test_mode,
        transforms=val_sat_transforms,
    )
    gallery_center_loc_xy_list = gallery_dataset_test.images_center_loc_xy
    gallery_topleft_loc_xy_list = gallery_dataset_test.images_topleft_loc_xy
    gallery_img_list = gallery_dataset_test.images_name
    
    gallery_dataloader_test = DataLoader(
        gallery_dataset_test,
        batch_size=config.batch_size_eval,
        num_workers=config.num_workers,
        shuffle=False,
        pin_memory=True
    )
    
    print(f"\nQuery Images Test: {len(query_dataset_test)}")
    print(f"Gallery Images Test: {len(gallery_dataset_test)}")
    
    #-----------------------------------------------------------------------------#
    # Loss                                                                        #
    #-----------------------------------------------------------------------------#
    print(f"\nLoss: WeightedInfoNCE (weight={config.with_weight}, k={config.k})")
    
    loss_function_normal = WeightedInfoNCE(
        device=config.device,
        label_smoothing=config.label_smoothing,
        k=config.k,
    )

    if config.mixed_precision:
        scaler = GradScaler(init_scale=2.**10)
    else:
        scaler = None
        
    #-----------------------------------------------------------------------------#
    # Optimizer                                                                   #
    #-----------------------------------------------------------------------------#
    print(f"\nOptimizer: AdamW (lr={config.lr})")

    if config.decay_exclue_bias:
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias"]
        optimizer_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_parameters, lr=config.lr)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    #-----------------------------------------------------------------------------#
    # Scheduler                                                                   #
    #-----------------------------------------------------------------------------#
    train_steps = len(train_dataloader) * config.epochs
    warmup_steps = int(len(train_dataloader) * config.warmup_epochs)
       
    if config.scheduler == "polynomial":
        print(f"\nScheduler: polynomial - max LR: {config.lr} - end LR: {config.lr_end}")
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_training_steps=train_steps,
            lr_end=config.lr_end,
            power=1.5,
            num_warmup_steps=warmup_steps
        )
    elif config.scheduler == "cosine":
        print(f"\nScheduler: cosine - max LR: {config.lr}")
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_training_steps=train_steps,
            num_warmup_steps=warmup_steps
        )
    elif config.scheduler == "constant":
        print(f"\nScheduler: constant - max LR: {config.lr}")
        scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps
        )
    else:
        scheduler = None
        
    print(f"Warmup Epochs: {config.warmup_epochs} - Warmup Steps: {warmup_steps}")
    print(f"Train Epochs: {config.epochs} - Train Steps: {train_steps}")
        
    #-----------------------------------------------------------------------------#
    # Zero Shot Evaluation                                                        #
    #-----------------------------------------------------------------------------#
    if config.zero_shot:
        print("\n" + "=" * 30 + "[Zero Shot]" + "=" * 30)

        r1_test = evaluate(
            config=config,
            model=model,
            query_loader=query_dataloader_test,
            gallery_loader=gallery_dataloader_test, 
            query_list=query_img_list,
            gallery_list=gallery_img_list,
            pairs_dict=pairs_drone2sate_dict,
            ranks_list=[1, 5, 10],
            query_center_loc_xy_list=query_center_loc_xy_list,
            gallery_center_loc_xy_list=gallery_center_loc_xy_list,
            gallery_topleft_loc_xy_list=gallery_topleft_loc_xy_list,
            step_size=1000,
            cleanup=True
        )
           
    #-----------------------------------------------------------------------------#
    # Training Loop                                                               #
    #-----------------------------------------------------------------------------#
    best_score = 0
    
    for epoch in range(1, config.epochs + 1):
        
        print("\n" + "=" * 30 + f"[Epoch: {epoch}]" + "=" * 30)

        if config.custom_sampling:
            train_dataloader.dataset.shuffle_group()
        
        train_loss = train_with_weight(
            config,
            model,
            dataloader=train_dataloader,
            loss_function=loss_function_normal,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler
        )

        print(f"Train Loss: {train_loss:.4f}")

        # Evaluation
        if (epoch % config.eval_every_n_epoch == 0 and epoch != 0) or epoch == config.epochs:
            print("\n" + "-" * 30 + "[Evaluation]" + "-" * 30)

            r1_test = evaluate(
                config=config,
                model=model,
                query_loader=query_dataloader_test,
                gallery_loader=gallery_dataloader_test,
                query_list=query_img_list,
                gallery_list=gallery_img_list,
                pairs_dict=pairs_drone2sate_dict,
                ranks_list=[1, 5, 10],
                query_center_loc_xy_list=query_center_loc_xy_list,
                gallery_center_loc_xy_list=gallery_center_loc_xy_list,
                gallery_topleft_loc_xy_list=gallery_topleft_loc_xy_list,
                step_size=1000,
                cleanup=True
            )
            
            # Save best model
            if r1_test > best_score:
                best_score = r1_test
                
                if isinstance(model, torch.nn.DataParallel):
                    torch.save(model.module.state_dict(), f'{model_path}/weights_best.pth')
                else:
                    torch.save(model.state_dict(), f'{model_path}/weights_best.pth')
                print(f">>> New best model saved! R@1: {best_score:.4f}")

        # Save checkpoint every epoch
        if isinstance(model, torch.nn.DataParallel):
            torch.save(model.module.state_dict(), f'{model_path}/weights_e{epoch}.pth')
        else:
            torch.save(model.state_dict(), f'{model_path}/weights_e{epoch}.pth')

    print("\n" + "=" * 70)
    print(f"Training Complete! Best R@1: {best_score:.4f}")
    print(f"Model saved to: {model_path}")
    print("=" * 70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fine-tune GTA-UAV model on combined datasets')
    
    # Required
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to pre-trained checkpoint (required)')
    
    # Training
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='Learning rate')
    
    # Virtual expansion
    parser.add_argument('--expansion_visloc', type=int, default=10,
                        help='Expansion factor for UAV_VisLoc')
    parser.add_argument('--expansion_vpair', type=int, default=5,
                        help='Expansion factor for VPair')
    
    # GPU
    parser.add_argument('--gpu', type=int, nargs='+', default=[0],
                        help='GPU IDs to use')
    
    # Data
    parser.add_argument('--data_root', type=str, 
                        default='/home/aniel/skyline_drone/datasets',
                        help='Root directory for datasets')
    
    # Smoke test mode
    parser.add_argument('--smoke_test', action='store_true',
                        help='Quick test with 2 epochs and reduced expansion')
    
    args = parser.parse_args()
    
    # Create configuration
    config = Configuration()
    
    # Apply command line arguments
    config.checkpoint_start = args.checkpoint
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.lr = args.lr
    config.expansion_visloc = args.expansion_visloc
    config.expansion_vpair = args.expansion_vpair
    config.gpu_ids = tuple(args.gpu)
    config.data_root = args.data_root
    
    # Smoke test mode
    if args.smoke_test:
        print(">>> SMOKE TEST MODE <<<")
        config.epochs = 2
        config.expansion_visloc = 2
        config.expansion_vpair = 2
        config.eval_every_n_epoch = 1
    
    # Run training
    train_script(config)
