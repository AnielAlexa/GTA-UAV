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

from game4loc.dataset.gta import GTADatasetEval, GTADatasetTrain, get_transforms
from game4loc.utils import setup_system, Logger
from game4loc.trainer.trainer import train, train_with_weight
from game4loc.evaluate.gta import evaluate
from game4loc.loss import InfoNCE, WeightedInfoNCE, GroupInfoNCE, TripletLoss
from game4loc.models.model import DesModel
from game4loc.models.model_netvlad import DesModelWithVLAD
from game4loc.models.model_dinov3_boq import DesModelDINOv3BoQ
from game4loc.models.model_dinov3_msra import DesModelDINOv3MSRA # Authenticated
from game4loc.models.model_convnext_boq import DesModelConvNextBoQ
from game4loc.models.model_matchanything_boq import DesModelMatchAnythingBoQ


def parse_tuple(s):
    try:
        return tuple(map(int, s.split(',')))
    except ValueError:
        raise argparse.ArgumentTypeError("Tuple must be integers separated by commas")


@dataclass
class Configuration:
    
    # Model
    model: str = 'vit_small_patch16_dinov3.lvd1689m'
    model_hub: str = 'timm'
    with_netvlad: bool = False
    with_dinov3_boq: bool = False
    with_dinov3_msra: bool = False # New flag
    with_matchanything_boq: bool = False

    # BoQ parameters (Kept for compatibility)
    num_queries: int = 64
    boq_nheads: int = 8
    gem_p: float = 3.0
    mlp_hidden_dim: int = 1024
    mlp_output_dim: int = 512
    unfreeze_n_blocks: int = 2

    # MSRA parameters
    msra_K: int = 64
    msra_d_state: int = 16
    use_paper_index_embed: bool = True  # Use paper's W_LN linear transform (True) vs embedding lookup (False)

    # MatchAnything parameters
    matchanything_ckpt: str = ''

    # Override model image size
    img_size: int = 256
 
    # Please Ignore
    freeze_layers: bool = False
    frozen_stages = [0,0,0,0]

    # Training with sharing weights
    share_weights: bool = True
    
    # Training with weighted-InfoNCE
    with_weight: bool = True

    # Please Ignore
    train_in_group: bool = True
    group_len = 2
    # Please Ignore
    loss_type = ["whole_slice", "part_slice"]

    # Please Ignore
    train_with_mix_data: bool = False
    # Please Ignore
    train_with_recon: bool = False
    recon_weight: float = 0.1
    
    # Training 
    mixed_precision: bool = True
    custom_sampling: bool = True         # use custom sampling instead of random
    seed = 1
    epochs: int = 25
    batch_size: int = 32                # keep in mind real_batch_size = 2 * batch_size
    verbose: bool = False
    gpu_ids: tuple = (0,)           # GPU ids for training

    # Training with sparse data
    train_ratio: float = 1.0

    # Eval
    batch_size_eval: int = 32
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
    
    # Learning Rate
    lr: float = 0.0003                   # Slightly higher for LoRA
    scheduler: str = "cosine"            # "polynomial" | "cosine" | "constant" | None
    warmup_epochs: int = 1
    lr_end: float = 0.00001              #  only for "polynomial"

    # Augment Images
    prob_flip: float = 0.5               # flipping the sat image and drone image simultaneously
    
    # Savepath for model checkpoints
    model_path: str = "./work_dir/gta_msra"
    log_to_file: bool = False # Added missing field based on usage
    log_path: str = "log.txt"

    query_mode: str = "D2S"               # Retrieval in Drone to Satellite

    train_mode: str = "pos_semipos"       # Train with positive + semi-positive pairs
    test_mode: str = "pos"                # Test with positive pairs

    # Eval before training
    zero_shot: bool = True
    
    # Checkpoint to start from
    checkpoint_start = None

    # LoRA
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1

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
    # Model                                                                       #
    #-----------------------------------------------------------------------------#
    print("\nModel: {}".format(config.model))

    if config.with_dinov3_msra:
        print("Using DINOv3 + ConvLoRA + MSRA Aggregator (Mamba)")
        model = DesModelDINOv3MSRA(model_name=config.model,
                                  pretrained=True,
                                  img_size=config.img_size,
                                  share_weights=config.share_weights,
                                  msra_K=config.msra_K,
                                  msra_d_state=config.msra_d_state,
                                  use_paper_index_embed=config.use_paper_index_embed,
                                  use_lora=config.use_lora,
                                  lora_r=config.lora_r,
                                  lora_alpha=config.lora_alpha,
                                  lora_dropout=config.lora_dropout)
                                  
    elif config.with_dinov3_boq:
        if "convnext" in config.model:
             model = DesModelConvNextBoQ(model_name=config.model,
                                  pretrained=True,
                                  img_size=config.img_size,
                                  share_weights=config.share_weights,
                                  num_queries=config.num_queries,
                                  mlp_output_dim=config.mlp_output_dim,
                                  unfreeze_n_blocks=config.unfreeze_n_blocks,
                                  use_lora=config.use_lora,
                                  lora_r=config.lora_r,
                                  lora_alpha=config.lora_alpha,
                                  lora_dropout=config.lora_dropout)
        else:
            model = DesModelDINOv3BoQ(model_name=config.model,
                                  pretrained=True,
                                  img_size=config.img_size,
                                  share_weights=config.share_weights,
                                  num_queries=config.num_queries,
                                  boq_nheads=config.boq_nheads,
                                  gem_p=config.gem_p,
                                  mlp_hidden_dim=config.mlp_hidden_dim,
                                  mlp_output_dim=config.mlp_output_dim,
                                  unfreeze_n_blocks=config.unfreeze_n_blocks,
                                  use_lora=config.use_lora,
                                  lora_r=config.lora_r,
                                  lora_alpha=config.lora_alpha,
                                  lora_dropout=config.lora_dropout)
    else:
        # Fallbacks
        model = DesModel(model_name=config.model,
                        pretrained=True,
                        img_size=config.img_size,
                        share_weights=config.share_weights)
                        
    # No .get_config() method on some custom models?
    if hasattr(model, 'get_config'):
        data_config = model.get_config()
    else:
        # Fallback for DINOv3
        data_config = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}

    print(data_config)
    mean = data_config.get("mean", [0.485, 0.456, 0.406])
    std = data_config.get("std", [0.229, 0.224, 0.225])
    img_size = (config.img_size, config.img_size)
    
    # Activate gradient checkpointing
    if config.grad_checkpointing:
        if hasattr(model, 'set_grad_checkpointing'):
            model.set_grad_checkpointing(True)
    
    # Load pretrained Checkpoint    
    if config.checkpoint_start is not None:  
        print("Start from:", config.checkpoint_start)
        model_state_dict = torch.load(config.checkpoint_start)  
        model.load_state_dict(model_state_dict, strict=False)     

    print("Freeze model layers:", config.freeze_layers, config.frozen_stages)
    if config.freeze_layers:
        if hasattr(model, 'freeze_layers'):
            model.freeze_layers(config.frozen_stages)

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
    # DataLoader                                                                  #
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
    # Optimizer                                                                   #
    #-----------------------------------------------------------------------------#
    
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
        # For DINO-MSRA, we primarily train LoRA params and the Aggregator.
        # Ensure only gradients=True params are passed?
        params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = torch.optim.AdamW(params, lr=config.lr, weight_decay=0.01)

    # LR Scheduler
    if config.scheduler == "polynomial":
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=len(train_dataloader) * config.warmup_epochs,
            num_training_steps=len(train_dataloader) * config.epochs,
            lr_end=config.lr_end,
            power=1.0,
        )
    elif config.scheduler == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=len(train_dataloader) * config.warmup_epochs,
            num_training_steps=len(train_dataloader) * config.epochs,
        )
    elif config.scheduler == "constant":
        scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=len(train_dataloader) * config.warmup_epochs,
        )
    else:
        scheduler = None

    #-----------------------------------------------------------------------------#
    # Loss                                                                        #
    #-----------------------------------------------------------------------------#
    print("\nLoss: WeightedInfoNCE")
    loss_fn = WeightedInfoNCE(k=config.k, label_smoothing=config.label_smoothing)

    if config.mixed_precision:
        scaler = GradScaler()
    else:
        scaler = None

    #-----------------------------------------------------------------------------#
    # Train Loop                                                                  #
    #-----------------------------------------------------------------------------#
    
    print("\nStart Training...")
    
    # Zero Shot
    if config.zero_shot:
        print("\nZero-Shot Evaluation...")
        evaluate(config=config,
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

    for epoch in range(1, config.epochs + 1):
        print("\n{}[Epoch: {}]{}".format(30*"-", epoch, 30*"-"))
        
        if config.custom_sampling:
            train_dataloader.dataset.shuffle()
            
        if config.with_weight:
            train_loss = train_with_weight(config, 
                                           model, 
                                           train_dataloader, 
                                           loss_fn, 
                                           optimizer, 
                                           scheduler, 
                                           scaler)
        else:
            train_loss = train(config, 
                               model, 
                               train_dataloader, 
                               loss_fn, 
                               optimizer, 
                               scheduler, 
                               scaler)
                               
        print(f"Epoch {epoch} Loss: {train_loss}")
        
        # Evaluation
        if epoch % config.eval_every_n_epoch == 0:
            print("\nEvaluating...")
            evaluate(config=config,
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
            
            # Save Checkpoint
            torch.save(model.state_dict(), os.path.join(model_path, f"epoch_{epoch}.pth"))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train GTA-UAV DINO-MSRA')
    
    parser.add_argument('--project_name', type=str, default='GTA-UAV')
    parser.add_argument('--model', type=str, default='vit_small_patch16_dinov3.lvd1689m')
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--gpu_ids', type=parse_tuple, default=(0,))
    
    # Flags
    parser.add_argument('--with_dinov3_msra', action='store_true', default=True, help='Use DINO-MSRA')
    
    # MSRA
    parser.add_argument('--msra_K', type=int, default=64)
    parser.add_argument('--msra_d_state', type=int, default=16)
    parser.add_argument('--use_paper_index_embed', action='store_true', default=True,
                        help='Use paper W_LN linear transform (default). Use --no_paper_index_embed for embedding lookup.')
    parser.add_argument('--no_paper_index_embed', dest='use_paper_index_embed', action='store_false',
                        help='Use embedding lookup instead of paper W_LN linear transform')
    
    # LoRA
    parser.add_argument('--use_lora', action='store_true', default=True)
    parser.add_argument('--lora_r', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--lora_dropout', type=float, default=0.1)

    # IO
    parser.add_argument('--data_root', type=str, default='./data/GTA-UAV-data')
    parser.add_argument('--model_path', type=str, default='./work_dir/gta_msra')
    parser.add_argument('--train_pairs_meta_file', type=str, default='cross-area-drone2sate-train.json')
    parser.add_argument('--test_pairs_meta_file', type=str, default='cross-area-drone2sate-test.json')
    parser.add_argument('--checkpoint_start', type=str, default=None)
    
    # Loss & Training
    parser.add_argument('--k', type=float, default=3)
    parser.add_argument('--test_mode', type=str, default='pos')
    parser.add_argument('--train_ratio', type=float, default=1.0)
    parser.add_argument('--freeze_layers', action='store_true')

    args = parser.parse_args()

    # Create config
    config = Configuration()
    
    # Update config with args
    for k, v in vars(args).items():
        if hasattr(config, k):
            setattr(config, k, v)
            
    train_script(config)
