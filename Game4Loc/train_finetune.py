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


def parse_tuple(s):
    try:
        return tuple(map(int, s.split(',')))
    except ValueError:
        raise argparse.ArgumentTypeError("Tuple must be integers separated by commas")


@dataclass
class Configuration:
    
    # Model
    model: str = 'vit_base_patch14_dinov2.lvd142m' # Default to Dinov2
    model_hub: str = 'timm'
    with_netvlad: bool = False
    
    # Override model image size
    img_size: int = 384 # Dinov2 usually uses larger images, check if 224 or 384
 
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
    epochs: int = 5 # Fewer epochs for finetuning
    batch_size: int = 16 # Adjust based on GPU memory
    verbose: bool = False
    gpu_ids: tuple = (0,)           # GPU ids for training

    # Training with sparse data
    train_ratio: float = 1.0

    # Eval
    batch_size_eval: int = 64
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
    
    # Augmentation for Rotation Invariance and Sim-to-Real Domain Adaptation
    hard_rotation: bool = True           # Apply ±180° rotation to drone images (instead of ±90°)
    sim2real_aug: bool = True            # Apply aggressive color jitter for sim-to-real adaptation
    
    # Learning Rate
    lr: float = 1e-5                     # Lower LR for finetuning
    scheduler: str = "cosine"            # "polynomial" | "cosine" | "constant" | None
    warmup_epochs: int = 0.1
    lr_end: float = 1e-6               #  only for "polynomial"

    # Augment Images
    prob_flip: float = 0.5               # flipping the sat image and drone image simultaneously
    
    # Savepath for model checkpoints
    model_path: str = "./work_dir/finetune"
    
    log_to_file: bool = True
    log_path: str = "./work_dir/finetune/log.txt"

    query_mode: str = "D2S"               # Retrieval in Drone to Satellite

    train_mode: str = "pos_semipos"       # Train with positive + semi-positive pairs
    test_mode: str = "pos"                # Test with positive pairs

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

    data_root: str = "/home/aniel/skyline_drone/datasets"

    train_pairs_meta_file = 'vpair_train.json'
    test_pairs_meta_file = 'vpair_train.json' # Use same for now or split
    sate_img_dir = 'vpair/reference_views'


#-----------------------------------------------------------------------------#
# Train Config                                                                #
#-----------------------------------------------------------------------------#

def train_script(config):

    if config.log_to_file:
        if not os.path.exists(os.path.dirname(config.log_path)):
            os.makedirs(os.path.dirname(config.log_path))
        # f = open(config.log_path, 'w')
        # sys.stdout = f

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
    
    # Load pretrained Checkpoint    
    if config.checkpoint_start is not None:  
        print("Start from:", config.checkpoint_start)
        model_state_dict = torch.load(config.checkpoint_start)  
        model.load_state_dict(model_state_dict, strict=False)     

    print("Freeze model layers:", config.freeze_layers, config.frozen_stages)
    if config.freeze_layers:
        model.freeze_layers(config.frozen_stages)

    # Data parallel
    print("GPUs available:", torch.cuda.device_count())  
    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)
        
    model.to(config.device)

    #-----------------------------------------------------------------------------#
    # DataLoader                                                                  #
    #-----------------------------------------------------------------------------#
    
    # Transforms
    val_transforms, train_sat_transforms, train_drone_transforms = \
        get_transforms(img_size, mean=mean, std=std, sat_rot=False,
                      hard_rotation=config.hard_rotation,
                      sim2real_aug=config.sim2real_aug)

    # Train
    train_dataset = GTADatasetTrain(pairs_meta_file=config.train_pairs_meta_file,
                                    data_root=config.data_root,
                                    transforms_query=train_drone_transforms,
                                    transforms_gallery=train_sat_transforms,
                                    prob_flip=config.prob_flip,
                                    shuffle_batch_size=config.batch_size,
                                    mode=config.train_mode,
                                    train_ratio=config.train_ratio,
                                    group_len=config.group_len)
    
    train_loader = DataLoader(train_dataset,
                              batch_size=config.batch_size,
                              num_workers=config.num_workers,
                              shuffle=not config.custom_sampling,
                              pin_memory=True,
                              collate_fn=None)
    
    # Evaluation is disabled for VPair finetuning as GTADatasetEval expects specific GTA-UAV metadata/filenames
    eval_loader = None
    
    #-----------------------------------------------------------------------------#
    # Optimizer                                                                   #
    #-----------------------------------------------------------------------------#
    
    param_groups = [
        {'params': [p for n, p in model.named_parameters() if "bias" in n], 'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters() if "bias" not in n], 'weight_decay': 0.01}
    ]
    
    optimizer = torch.optim.AdamW(param_groups, lr=config.lr, betas=(0.9, 0.999), eps=1e-8)
    
    #-----------------------------------------------------------------------------#
    # Scheduler                                                                   #
    #-----------------------------------------------------------------------------#
    
    num_steps = len(train_loader) * config.epochs
    warmup_steps = len(train_loader) * config.warmup_epochs
    
    if config.scheduler == "polynomial":
        scheduler = get_polynomial_decay_schedule_with_warmup(optimizer,
                                                              num_warmup_steps=warmup_steps,
                                                              num_training_steps=num_steps,
                                                              lr_end=config.lr_end,
                                                              power=1.0)
    elif config.scheduler == "cosine":
        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=warmup_steps,
                                                    num_training_steps=num_steps)
    elif config.scheduler == "constant":
        scheduler = get_constant_schedule_with_warmup(optimizer,
                                                      num_warmup_steps=warmup_steps)
    else:
        scheduler = None

    #-----------------------------------------------------------------------------#
    # Loss                                                                        #
    #-----------------------------------------------------------------------------#
    
    if config.with_weight:
        criterion = WeightedInfoNCE(label_smoothing=config.label_smoothing)
    else:
        criterion = InfoNCE(label_smoothing=config.label_smoothing)
    
    #-----------------------------------------------------------------------------#
    # Train                                                                       #
    #-----------------------------------------------------------------------------#
    
    print("\nStart Training...")
    
    if config.zero_shot and eval_loader is not None:
        print("\nZero-Shot Eval...")
        evaluate(model, eval_loader, config.device, config.normalize_features, config.verbose)

    for epoch in range(1, config.epochs + 1):
        
        if config.with_weight:
            train_with_weight(model, train_loader, criterion, optimizer, scheduler, epoch, config.device, config.clip_grad, config.log_path, config.verbose)
        else:
            train(model, train_loader, criterion, optimizer, scheduler, epoch, config.device, config.clip_grad, config.log_path, config.verbose)
        
        if epoch % config.eval_every_n_epoch == 0:
            print("\nEval Epoch: {}".format(epoch))
            if eval_loader is not None:
                evaluate(model, eval_loader, config.device, config.normalize_features, config.verbose)
            
            # Save Checkpoint
            torch.save(model.state_dict(), os.path.join(model_path, 'epoch_{}.pth'.format(epoch)))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train Script')
    
    # Model
    parser.add_argument('--model', default=Configuration.model, type=str, help='Model name')
    parser.add_argument('--model_hub', default=Configuration.model_hub, type=str, help='Model Hub')
    parser.add_argument('--with_netvlad', default=Configuration.with_netvlad, type=bool, help='With NetVLAD')
    parser.add_argument('--img_size', default=Configuration.img_size, type=int, help='Image Size')
    
    # Training
    parser.add_argument('--batch_size', default=Configuration.batch_size, type=int, help='Batch Size')
    parser.add_argument('--epochs', default=Configuration.epochs, type=int, help='Epochs')
    parser.add_argument('--lr', default=Configuration.lr, type=float, help='Learning Rate')
    parser.add_argument('--gpu_ids', default=Configuration.gpu_ids, type=parse_tuple, help='GPU IDs')
    parser.add_argument('--seed', default=Configuration.seed, type=int, help='Seed')
    parser.add_argument('--verbose', default=Configuration.verbose, type=bool, help='Verbose')
    
    # Data
    parser.add_argument('--data_root', default=Configuration.data_root, type=str, help='Data Root')
    parser.add_argument('--train_pairs_meta_file', default=Configuration.train_pairs_meta_file, type=str, help='Train Pairs Meta File')
    parser.add_argument('--test_pairs_meta_file', default=Configuration.test_pairs_meta_file, type=str, help='Test Pairs Meta File')
    parser.add_argument('--sate_img_dir', default=Configuration.sate_img_dir, type=str, help='Satellite Image Dir')
    
    # Checkpoint
    parser.add_argument('--checkpoint_start', default=Configuration.checkpoint_start, type=str, help='Checkpoint Start')
    parser.add_argument('--model_path', default=Configuration.model_path, type=str, help='Model Path')

    args = parser.parse_args()

    config = Configuration(
        model=args.model,
        model_hub=args.model_hub,
        with_netvlad=args.with_netvlad,
        img_size=args.img_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        gpu_ids=args.gpu_ids,
        seed=args.seed,
        verbose=args.verbose,
        data_root=args.data_root,
        train_pairs_meta_file=args.train_pairs_meta_file,
        test_pairs_meta_file=args.test_pairs_meta_file,
        sate_img_dir=args.sate_img_dir,
        checkpoint_start=args.checkpoint_start,
        model_path=args.model_path
    )

    train_script(config)
