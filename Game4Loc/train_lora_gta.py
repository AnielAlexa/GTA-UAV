"""
train_lora_gta.py - LoRA fine-tuning version of train_gta.py

This is train_gta.py but with:
1. Frozen backbone + PEFT LoRA adapters (instead of full fine-tune)
2. MLP head for feature projection
3. Same dataset, augmentations, loss, evaluation as train_gta.py
"""

import os
import time
import math
import shutil
import sys
import torch
import torch.nn as nn
import argparse
import numpy as np
from dataclasses import dataclass
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from transformers import get_constant_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType

import timm
import torch.nn.functional as F
from torch.cuda.amp import autocast
from sklearn.metrics import average_precision_score
from tqdm import tqdm

from game4loc.dataset.gta import GTADatasetEval, GTADatasetTrain, get_transforms
from game4loc.utils import setup_system, Logger
from game4loc.trainer.trainer import train, train_with_weight
from game4loc.loss import InfoNCE, WeightedInfoNCE, GroupInfoNCE, TripletLoss


#-----------------------------------------------------------------------------#
# Evaluate Function (inlined to avoid problematic GimDKM import)              #
#-----------------------------------------------------------------------------#

def predict(train_config, model, dataloader):
    """Extract features from dataloader."""
    model.eval()
    time.sleep(0.1)
    
    if train_config.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader
        
    img_features_list = []
    
    with torch.no_grad():
        for img in bar:
            with autocast():
                img = img.to(train_config.device)
                img_feature = model(img1=img)
            
                if train_config.normalize_features:
                    img_feature = F.normalize(img_feature, dim=-1)
            
            img_features_list.append(img_feature.to(torch.float32))

        img_features = torch.cat(img_features_list, dim=0) 
        
    if train_config.verbose:
        bar.close()
    
    return img_features


def evaluate(
        config,
        model,
        query_loader,
        gallery_loader,
        query_list,
        query_center_loc_xy_list,
        gallery_list,
        gallery_center_loc_xy_list,
        gallery_topleft_loc_xy_list,
        pairs_dict,
        ranks_list=[1, 5, 10],
        sdmk_list=[1, 3, 5],
        disk_list=[1, 3, 5],
        step_size=1000,
        cleanup=True,
        dis_threshold_list=[4*(i+1) for i in range(50)],
        plot_acc_threshold=False,
        top10_log=False,
        with_match=False,
    ):
    """Evaluate model - simplified version without GimDKM matcher."""
    
    print("Extract Features and Compute Scores:")
    model.eval()
    img_features_query = predict(config, model, query_loader)

    all_scores = []
    with torch.no_grad():
        for gallery_batch in gallery_loader:
            with autocast():
                gallery_batch = gallery_batch.to(device=config.device)
                gallery_features_batch = model(img2=gallery_batch)
                if config.normalize_features:
                    gallery_features_batch = F.normalize(gallery_features_batch, dim=-1)

            scores_batch = img_features_query @ gallery_features_batch.T
            all_scores.append(scores_batch.cpu())
    
    all_scores = torch.cat(all_scores, dim=1).numpy()

    gallery_idx = {}
    for idx, gallery_img in enumerate(gallery_list):
        gallery_idx[gallery_img] = idx

    matches_list = []
    for query_i in query_list:
        pairs_list_i = pairs_dict[query_i]
        matches_i = []
        for pair in pairs_list_i:
            matches_i.append(gallery_idx[pair])
        matches_list.append(np.array(matches_i))

    matches_tensor = [torch.tensor(matches, dtype=torch.long) for matches in matches_list]

    query_num = img_features_query.shape[0]

    all_ap = []
    cmc = np.zeros(len(gallery_list))

    for i in tqdm(range(query_num), desc="Processing each query"):
        score = all_scores[i]    
        index = np.argsort(score)[::-1]
        
        good_index_i = np.isin(index, matches_tensor[i]) 
        
        y_true = good_index_i.astype(int)
        y_scores = np.arange(len(y_true), 0, -1)
        if np.sum(y_true) > 0:
            ap = average_precision_score(y_true, y_scores)
            all_ap.append(ap)
        
        match_rank = np.where(good_index_i == 1)[0]
        if len(match_rank) > 0:
            cmc[match_rank[0]:] += 1
    
    mAP = np.mean(all_ap)
    cmc = cmc / query_num

    top1 = round(len(gallery_list)*0.01)

    string = []
    for i in ranks_list:
        string.append('Recall@{}: {:.4f}'.format(i, cmc[i-1]*100))
        
    string.append('Recall@top1: {:.4f}'.format(cmc[top1]*100))
    string.append('AP: {:.4f}'.format(mAP*100))   
    
    print(' - '.join(string))
    
    return cmc[0]  # Return R@1 for model selection


def parse_tuple(s):
    try:
        return tuple(map(int, s.split(',')))
    except ValueError:
        raise argparse.ArgumentTypeError("Tuple must be integers separated by commas")


#-----------------------------------------------------------------------------#
# LoRA Model with MLP Head                                                    #
#-----------------------------------------------------------------------------#

class DesModelLoRA(nn.Module):
    """
    DesModel with LoRA adapters using PEFT library.
    
    - Backbone: 100% frozen
    - LoRA adapters on attention layers (qkv, proj) and MLP layers (fc1, fc2)
    - MLP head for descriptor projection
    - Learnable logit_scale (same as DesModel)
    """

    def __init__(self, 
                 model_name='vit_small_patch16_dinov3.lvd1689m',
                 pretrained=True,
                 img_size=384,
                 share_weights=True,
                 lora_r=16,
                 lora_alpha=32,
                 lora_dropout=0.1,
                 lora_targets=None,
                 mlp_hidden=512):
                 
        super(DesModelLoRA, self).__init__()
        self.share_weights = share_weights
        self.model_name = model_name
        self.img_size = img_size
        
        # 1. Load backbone
        print(f"Loading backbone: {model_name}")
        self.model = timm.create_model(
            model_name, 
            pretrained=pretrained, 
            num_classes=0, 
            img_size=img_size,
            global_pool='avg'  # Required for DINOv3
        )
        
        self.embed_dim = self.model.embed_dim
        print(f"Backbone embed_dim: {self.embed_dim}")
        
        # 2. Apply PEFT LoRA
        if lora_targets is None:
            lora_targets = ["qkv", "proj", "fc1", "fc2"]
        
        # Build target modules list for PEFT
        target_modules = []
        for name, module in self.model.named_modules():
            for target in lora_targets:
                if target in name and isinstance(module, nn.Linear):
                    target_modules.append(name)
                    break
        
        # Remove duplicates and sort
        target_modules = sorted(list(set(target_modules)))
        print(f"LoRA target modules ({len(target_modules)}): {target_modules[:5]}...")
        
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
        )
        
        self.model = get_peft_model(self.model, lora_config)
        
        # 3. MLP Head: embed_dim -> mlp_hidden -> embed_dim
        self.mlp_head = nn.Sequential(
            nn.Linear(self.embed_dim, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(mlp_hidden, self.embed_dim)
        )
        
        # 4. Learnable temperature (same as DesModel)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        # 5. Print parameter summary
        self._print_params()

    def _print_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        
        # Count LoRA params
        lora_params = 0
        for name, param in self.model.named_parameters():
            if 'lora' in name.lower() and param.requires_grad:
                lora_params += param.numel()
        
        mlp_params = sum(p.numel() for p in self.mlp_head.parameters())
        
        print(f"\n{'='*50}")
        print(f"PARAMETER SUMMARY")
        print(f"{'='*50}")
        print(f"Total:      {total:>12,}")
        print(f"Trainable:  {trainable:>12,} ({100*trainable/total:.2f}%)")
        print(f"Frozen:     {frozen:>12,} ({100*frozen/total:.2f}%)")
        print(f"{'='*50}")
        print(f"LoRA:       {lora_params:>12,}")
        print(f"MLP Head:   {mlp_params:>12,}")
        print(f"{'='*50}\n")

    def get_config(self):
        """Compatible with DesModel.get_config()"""
        return timm.data.resolve_model_data_config(self.model)
    
    def set_grad_checkpointing(self, enable=True):
        """Compatible with DesModel.set_grad_checkpointing()"""
        self.model.set_grad_checkpointing(enable)

    def forward(self, img1=None, img2=None):
        """Compatible with DesModel.forward() interface."""
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


#-----------------------------------------------------------------------------#
# Configuration                                                               #
#-----------------------------------------------------------------------------#

@dataclass
class Configuration:
    
    # Model - DINOv3 by default
    model: str = 'vit_small_patch16_dinov3.lvd1689m'
    model_hub: str = 'timm'
    
    # Override model image size
    img_size: int = 384
    
    # LoRA Config
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_targets: list = None  # Will default to ["qkv", "proj", "fc1", "fc2"]
    
    # MLP Head
    mlp_hidden: int = 512

    # Training with sharing weights
    share_weights: bool = True
    
    # Training with weighted-InfoNCE
    with_weight: bool = True

    # Please Ignore
    train_in_group: bool = True
    group_len = 2
    # Please Ignore
    loss_type = ["whole_slice", "part_slice"]
    
    # Training 
    mixed_precision: bool = True
    custom_sampling: bool = True         # use custom sampling instead of random
    seed = 1
    epochs: int = 10
    batch_size: int = 64                 # Can use larger batch since less VRAM needed
    verbose: bool = False
    gpu_ids: tuple = (0,)                # GPU ids for training

    # Training with sparse data
    train_ratio: float = 1.0

    # Eval
    batch_size_eval: int = 128
    eval_every_n_epoch: int = 1          # eval every n Epoch
    normalize_features: bool = True
    eval_gallery_n: int = -1             # -1 for all or int

    # Optimizer 
    clip_grad = 100.                     # None | float
    decay_exclue_bias: bool = False
    grad_checkpointing: bool = False     # Gradient Checkpointing
    
    # Loss
    label_smoothing: float = 0.1
    k: float = 3
    
    # Learning Rate - higher for LoRA
    lr: float = 3e-4                     # Higher LR for LoRA
    scheduler: str = "cosine"            # "polynomial" | "cosine" | "constant" | None
    warmup_epochs: int = 0.1
    lr_end: float = 0.0001               # only for "polynomial"

    # Augment Images
    prob_flip: float = 0.5               # flipping the sat image and drone image simultaneously
    
    # Savepath for model checkpoints
    model_path: str = "./work_dir/gta_lora"

    query_mode: str = "D2S"              # Retrieval in Drone to Satellite

    train_mode: str = "pos_semipos"      # Train with positive + semi-positive pairs
    test_mode: str = "pos"               # Test with positive pairs

    # Eval before training
    zero_shot: bool = True
    
    # Checkpoint to start from
    checkpoint_start = None

    # set num_workers to 0 if on Windows
    num_workers: int = 0 if os.name == 'nt' else 4 
    
    # train on GPU if available
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu' 
    
    # for better performance
    cudnn_benchmark: bool = True
    
    # make cudnn deterministic
    cudnn_deterministic: bool = False

    data_root: str = "./data/GTA-UAV-data"

    train_pairs_meta_file = 'cross-area-drone2sate-train.json'
    test_pairs_meta_file = 'cross-area-drone2sate-test.json'
    sate_img_dir = 'satellite'


#-----------------------------------------------------------------------------#
# Train Config                                                                #
#-----------------------------------------------------------------------------#

def train_script(config):

    if config.log_to_file:
        f = open(config.log_path, 'w')
        sys.stdout = f

    save_time = "{}".format(time.strftime("%m%d%H%M%S"))
    model_path = "{}/{}/{}".format(config.model_path,
                                       config.model,
                                       save_time)

    if not os.path.exists(model_path):
        os.makedirs(model_path)
    shutil.copyfile(os.path.basename(__file__), "{}/train.py".format(model_path))

    # Redirect print to both console and log file
    sys.stdout = Logger(os.path.join(model_path, 'log.txt'))

    setup_system(seed=config.seed,
                 cudnn_benchmark=config.cudnn_benchmark,
                 cudnn_deterministic=config.cudnn_deterministic)
    
    print("training save in path: {}".format(model_path))
    print("training start from", config.checkpoint_start)

    #-----------------------------------------------------------------------------#
    # Model - LoRA Version                                                        #
    #-----------------------------------------------------------------------------#
    print("\nModel: {} (LoRA)".format(config.model))
    print(f"LoRA Config: r={config.lora_r}, alpha={config.lora_alpha}, dropout={config.lora_dropout}")

    model = DesModelLoRA(
        model_name=config.model, 
        pretrained=True,
        img_size=config.img_size,
        share_weights=config.share_weights,
        lora_r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        lora_targets=config.lora_targets,
        mlp_hidden=config.mlp_hidden,
    )
                        
    data_config = model.get_config()
    print(data_config)
    mean = data_config["mean"]
    std = data_config["std"]
    img_size = (config.img_size, config.img_size)
    
    # Activate gradient checkpointing
    if config.grad_checkpointing:
        model.set_grad_checkpointing(True)
    
    # Load pretrained Checkpoint    
    if config.checkpoint_start is not None:  
        print("Start from:", config.checkpoint_start)
        model_state_dict = torch.load(config.checkpoint_start)  
        model.load_state_dict(model_state_dict, strict=False)     

    # Data parallel
    print("GPUs available:", torch.cuda.device_count())  
    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)
            
    # Model to device   
    model = model.to(config.device)

    print("\nImage Size Query:", img_size)
    print("Image Size Ground:", img_size)
    print("Mean: {}".format(mean))
    print("Std:  {}\n".format(std)) 

    print("Use custom sampling: {}".format(config.custom_sampling))


    #-----------------------------------------------------------------------------#
    # DataLoader (same as train_gta.py)                                           #
    #-----------------------------------------------------------------------------#

    # Transforms
    if 'cross-area' in config.train_pairs_meta_file:
        sat_rot = True
    else:
        sat_rot = False
    val_transforms, train_sat_transforms, train_drone_transforms = \
        get_transforms(img_size, mean=mean, std=std, sat_rot=sat_rot)
                                                                                                                
    # Train
    train_dataset = GTADatasetTrain(data_root=config.data_root,
                                    pairs_meta_file=config.train_pairs_meta_file,
                                    transforms_query=train_drone_transforms,
                                    transforms_gallery=train_sat_transforms,
                                    group_len=config.group_len,
                                    prob_flip=config.prob_flip,
                                    shuffle_batch_size=config.batch_size,
                                    mode=config.train_mode,
                                    train_ratio=config.train_ratio,
                                    )
    
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config.batch_size,
                                  num_workers=config.num_workers,
                                  shuffle=not config.custom_sampling,
                                  pin_memory=True)
    
    # Test query
    if config.query_mode == 'D2S':
        query_view = 'drone'
        gallery_view = 'sate'
    else:
        query_view = 'sate'
        gallery_view = 'drone'
    query_dataset_test = GTADatasetEval(data_root=config.data_root,
                                        pairs_meta_file=config.test_pairs_meta_file,
                                        view=query_view,
                                        transforms=val_transforms,
                                        mode=config.test_mode,
                                        sate_img_dir=config.sate_img_dir,
                                        query_mode=config.query_mode,
                                        )
    query_img_list = query_dataset_test.images_name
    query_center_loc_xy_list = query_dataset_test.images_center_loc_xy
    pairs_drone2sate_dict = query_dataset_test.pairs_drone2sate_dict
    
    query_dataloader_test = DataLoader(query_dataset_test,
                                       batch_size=config.batch_size_eval,
                                       num_workers=config.num_workers,
                                       shuffle=False,
                                       pin_memory=True)
    
    # Test gallery
    gallery_dataset_test = GTADatasetEval(data_root=config.data_root,
                                          pairs_meta_file=config.test_pairs_meta_file,
                                          view=gallery_view,
                                          transforms=val_transforms,
                                          mode=config.test_mode,
                                          sate_img_dir=config.sate_img_dir,
                                          query_mode=config.query_mode,
                                         )
    gallery_center_loc_xy_list = gallery_dataset_test.images_center_loc_xy
    gallery_topleft_loc_xy_list = gallery_dataset_test.images_topleft_loc_xy
    gallery_img_list = gallery_dataset_test.images_name
    
    gallery_dataloader_test = DataLoader(gallery_dataset_test,
                                       batch_size=config.batch_size_eval,
                                       num_workers=config.num_workers,
                                       shuffle=False,
                                       pin_memory=True)
    
    print("Query Images Test:", len(query_dataset_test))
    print("Gallery Images Test:", len(gallery_dataset_test))
    
    #-----------------------------------------------------------------------------#
    # Loss (same as train_gta.py)                                                 #
    #-----------------------------------------------------------------------------#
    print("Train with weight?", config.with_weight, "k=", config.k)
    
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
    # Optimizer - Only trainable params (LoRA + MLP)                              #
    #-----------------------------------------------------------------------------#
    
    # Only optimize trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"\nOptimizing {len(trainable_params)} parameter groups")

    if config.decay_exclue_bias:
        param_optimizer = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
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
        optimizer = torch.optim.AdamW(trainable_params, lr=config.lr)


    #-----------------------------------------------------------------------------#
    # Scheduler (same as train_gta.py)                                            #
    #-----------------------------------------------------------------------------#

    train_steps = len(train_dataloader) * config.epochs
    warmup_steps = len(train_dataloader) * config.warmup_epochs
       
    if config.scheduler == "polynomial":
        print("\nScheduler: polynomial - max LR: {} - end LR: {}".format(config.lr, config.lr_end))  
        scheduler = get_polynomial_decay_schedule_with_warmup(optimizer,
                                                              num_training_steps=train_steps,
                                                              lr_end = config.lr_end,
                                                              power=1.5,
                                                              num_warmup_steps=warmup_steps)
        
    elif config.scheduler == "cosine":
        print("\nScheduler: cosine - max LR: {}".format(config.lr))   
        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                    num_training_steps=train_steps,
                                                    num_warmup_steps=warmup_steps)
        
    elif config.scheduler == "constant":
        print("\nScheduler: constant - max LR: {}".format(config.lr))   
        scheduler =  get_constant_schedule_with_warmup(optimizer,
                                                       num_warmup_steps=warmup_steps)
           
    else:
        scheduler = None
        
    print("Warmup Epochs: {} - Warmup Steps: {}".format(str(config.warmup_epochs).ljust(2), warmup_steps))
    print("Train Epochs:  {} - Train Steps:  {}".format(config.epochs, train_steps))
        
        
    #-----------------------------------------------------------------------------#
    # Zero Shot                                                                   #
    #-----------------------------------------------------------------------------#
    if config.zero_shot:
        print("\n{}[{}]{}".format(30*"-", "Zero Shot", 30*"-"))  

        r1_test = evaluate(config=config,
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
                           cleanup=True)
           
            
    #-----------------------------------------------------------------------------#
    # Train (same as train_gta.py)                                                #
    #-----------------------------------------------------------------------------#
    start_epoch = 0   
    best_score = 0
    

    for epoch in range(1, config.epochs+1):
        
        print("\n{}[Epoch: {}]{}".format(30*"-", epoch, 30*"-"))

        if config.custom_sampling:
            train_dataloader.dataset.shuffle()
        
        train_loss = train_with_weight(config,
                           model,
                           dataloader=train_dataloader,
                           loss_function=loss_function_normal,
                           optimizer=optimizer,
                           scheduler=scheduler,
                           scaler=scaler,
                           with_weight=config.with_weight)
        
        print("Epoch: {}, Train Loss = {:.3f}, Lr = {:.6f}".format(epoch,
                                                                   train_loss,
                                                                   optimizer.param_groups[0]['lr']))
        
        # evaluate
        if (epoch % config.eval_every_n_epoch == 0 and epoch != 0) or epoch == config.epochs:
        
            print("\n{}[{}]{}".format(30*"-", "Evaluate", 30*"-"))
        
            r1_test = evaluate(config=config,
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
                                cleanup=True)
                
            if r1_test > best_score or epoch == config.epochs:

                best_score = r1_test

                if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
                    torch.save(model.module.state_dict(), '{}/weights_e{}_{:.4f}.pth'.format(model_path, epoch, r1_test))
                else:
                    torch.save(model.state_dict(), '{}/weights_e{}_{:.4f}.pth'.format(model_path, epoch, r1_test))
                
    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
        torch.save(model.module.state_dict(), '{}/weights_end.pth'.format(model_path))
    else:
        torch.save(model.state_dict(), '{}/weights_end.pth'.format(model_path))  

    if config.log_to_file:
        f.close()
        sys.stdout = sys.__stdout__          


def parse_args():
    parser = argparse.ArgumentParser(description="LoRA training script for GTA (based on train_gta.py)")

    parser.add_argument('--log_to_file', action='store_true', help='Log saving to file')
    parser.add_argument('--log_path', type=str, default=None, help='Log file path')
    parser.add_argument('--data_root', type=str, default='./data/GTA-UAV-data', help='Data root')
    parser.add_argument('--train_pairs_meta_file', type=str, default='cross-area-drone2sate-train.json', help='Training metafile path')
    parser.add_argument('--test_pairs_meta_file', type=str, default='cross-area-drone2sate-test.json', help='Test metafile path')
    parser.add_argument('--model', type=str, default='vit_small_patch16_dinov3.lvd1689m', help='Model architecture')
    parser.add_argument('--img_size', type=int, default=384, help='Image size')
    
    # LoRA specific
    parser.add_argument('--lora_r', type=int, default=16, help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=32, help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.1, help='LoRA dropout')
    parser.add_argument('--mlp_hidden', type=int, default=512, help='MLP hidden size')
    
    parser.add_argument('--no_share_weights', action='store_true', help='Train without sharing weights')
    parser.add_argument('--epochs', type=int, default=10, help='Epochs')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--gpu_ids', type=parse_tuple, default=(0,), help='GPU ID')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--checkpoint_start', type=str, default=None, help='Training from checkpoint')
    parser.add_argument('--train_mode', type=str, default='pos_semipos', help='Train with positive or positive+semi-positive pairs')
    parser.add_argument('--test_mode', type=str, default='pos', help='Test with positive pairs')
    parser.add_argument('--query_mode', type=str, default='D2S', help='Retrieval with drone to satellite')
    parser.add_argument('--train_in_group', action='store_true', help='Train in group')
    parser.add_argument('--group_len', type=int, default=2, help='Group length')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='Label smoothing value for loss')
    parser.add_argument('--with_weight', action='store_true', help='Train with weight')
    parser.add_argument('--k', type=float, default=3, help='weighted k')
    parser.add_argument('--no_custom_sampling', action='store_true', help='Train without custom sampling')
    parser.add_argument('--train_ratio', type=float, default=1.0, help='Train on ratio of data')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    config = Configuration()
    config.data_root = args.data_root
    config.train_pairs_meta_file = args.train_pairs_meta_file
    config.test_pairs_meta_file = args.test_pairs_meta_file
    config.log_to_file = args.log_to_file
    config.log_path = args.log_path
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.train_in_group = args.train_in_group
    config.group_len = args.group_len
    config.gpu_ids = args.gpu_ids
    config.label_smoothing = args.label_smoothing
    config.with_weight = args.with_weight
    config.k = args.k
    config.checkpoint_start = args.checkpoint_start
    config.model = args.model
    config.img_size = args.img_size
    config.lr = args.lr
    config.share_weights = not(args.no_share_weights)
    config.custom_sampling = not(args.no_custom_sampling)
    config.train_mode = args.train_mode
    config.test_mode = args.test_mode
    config.query_mode = args.query_mode
    config.train_ratio = args.train_ratio
    
    # LoRA specific
    config.lora_r = args.lora_r
    config.lora_alpha = args.lora_alpha
    config.lora_dropout = args.lora_dropout
    config.mlp_hidden = args.mlp_hidden

    train_script(config)
