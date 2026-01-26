import os
import sys
import torch
import argparse
from dataclasses import dataclass
from torch.utils.data import DataLoader

from game4loc.dataset.gta import GTADatasetEval, get_transforms
from game4loc.evaluate.gta import evaluate
from game4loc.models.model import DesModel
from game4loc.models.model_dinov3_boq import DesModelDINOv3BoQ
from game4loc.models.model_convnext_boq import DesModelConvNextBoQ


def parse_tuple(s):
    try:
        return tuple(map(int, s.split(',')))
    except ValueError:
        raise argparse.ArgumentTypeError("Tuple must be integers separated by commas")


@dataclass
class Configuration:

    # Model
    # model: str = 'convnext_base.fb_in22k_ft_in1k_384'
    model: str = 'vit_base_patch16_rope_reg1_gap_256.sbb_in1k'
    
    # Override model image size
    img_size: int = 384
    
    # Evaluation
    batch_size: int = 128
    verbose: bool = True
    gpu_ids: tuple = (0)
    normalize_features: bool = True

    # New Grayscale option
    grayscale: bool = False

    # With Fine Matching
    with_match: bool = False
    
    # BoQ / LoRA / DINOv3 Settings
    with_dinov3_boq: bool = False
    num_queries: int = 64
    boq_nheads: int = 8
    mlp_hidden_dim: int = 1024
    mlp_output_dim: int = 512
    unfreeze_n_blocks: int = 2
    
    use_lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    
    use_intermediate_layer: bool = False
    intermediate_layer_idx: int = 9
    intermediate_facet: str = 'token'

    # set num_workers to 0 if on Windows
    num_workers: int = 0 if os.name == 'nt' else 4 
    
    # train on GPU if available
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu' 

    # Dataset
    query_mode: str = 'D2S'
    # query_mode: str = 'S2D'

    # Checkpoint to start from
    # checkpoint_start = '/home/xmuairmud/jyx/GTA-UAV/Game4Loc/pretrained/gta/same_area/selavpr.pth'
    checkpoint_start = 'pretrained/gta/cross_area/game4loc.pth'

    # data_root: str = "/home/xmuairmud/data/GTA-UAV-data/GTA-UAV-Lidar/GTA-UAV-Lidar"
    data_root: str = "/home/xmuairmud/data/GTA-UAV-data/GTA-UAV-official/GTA-UAV-LR-hf"

    train_pairs_meta_file = 'cross-area-drone2sate-train.json'
    test_pairs_meta_file = 'cross-area-drone2sate-test.json'
    sate_img_dir = 'satellite'


def eval_script(config):

    if config.log_to_file:
        f = open(config.log_path, 'w')
        sys.stdout = f

    #-----------------------------------------------------------------------------#
    # Model                                                                       #
    #-----------------------------------------------------------------------------#
    
    print("\nModel: {}".format(config.model))


    if config.with_dinov3_boq:
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
                                  mlp_hidden_dim=config.mlp_hidden_dim,
                                  mlp_output_dim=config.mlp_output_dim,
                                  unfreeze_n_blocks=config.unfreeze_n_blocks,
                                  use_lora=config.use_lora,
                                  lora_r=config.lora_r,
                                  lora_alpha=config.lora_alpha,
                                  lora_dropout=config.lora_dropout,
                                  use_intermediate_layer=config.use_intermediate_layer,
                                  intermediate_layer_idx=config.intermediate_layer_idx,
                                  intermediate_facet=config.intermediate_facet)
    else:
        model = DesModel(config.model,
                    pretrained=True,
                    img_size=config.img_size,
                    share_weights=config.share_weights)
                          
    data_config = model.get_config()
    print(data_config)
    mean = data_config["mean"]
    std = data_config["std"]
    img_size = (config.img_size, config.img_size)
    

    # load pretrained Checkpoint    
    if config.checkpoint_start is not None:  
        print("Start from:", config.checkpoint_start)
        model_state_dict = torch.load(config.checkpoint_start)  
        model.load_state_dict(model_state_dict, strict=True)     

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


    #-----------------------------------------------------------------------------#
    # DataLoader                                                                  #
    #-----------------------------------------------------------------------------#

    # Transforms
    val_transforms, train_sat_transforms, train_drone_transforms = get_transforms(img_size, mean=mean, std=std, grayscale=config.grayscale)


    # Test query
    if config.query_mode == 'D2S':
        query_dataset_test = GTADatasetEval(data_root=config.data_root,
                                            pairs_meta_file=config.test_pairs_meta_file,
                                            view="drone",
                                            transforms=val_transforms,
                                            mode=config.test_mode,
                                            query_mode=config.query_mode,
                                            )
        gallery_dataset_test = GTADatasetEval(data_root=config.data_root,
                                            pairs_meta_file=config.test_pairs_meta_file,
                                            view="sate",
                                            transforms=val_transforms,
                                            sate_img_dir=config.sate_img_dir,
                                            mode=config.test_mode,
                                            query_mode=config.query_mode,
                                            )
        pairs_dict = query_dataset_test.pairs_drone2sate_dict
    elif config.query_mode == 'S2D':
        gallery_dataset_test = GTADatasetEval(data_root=config.data_root,
                                            pairs_meta_file=config.test_pairs_meta_file,
                                            view="drone",
                                            transforms=val_transforms,
                                            mode=config.test_mode,
                                            query_mode=config.query_mode,
                                            )
        pairs_dict = gallery_dataset_test.pairs_sate2drone_dict
        query_dataset_test = GTADatasetEval(data_root=config.data_root,
                                            pairs_meta_file=config.test_pairs_meta_file,
                                            view="sate",
                                            transforms=val_transforms,
                                            query_mode=config.query_mode,
                                            pairs_sate2drone_dict=pairs_dict,
                                            sate_img_dir=config.sate_img_dir,
                                            mode=config.test_mode,
                                        )
    query_img_list = query_dataset_test.images_name
    query_center_loc_xy_list = query_dataset_test.images_center_loc_xy

    gallery_center_loc_xy_list = gallery_dataset_test.images_center_loc_xy
    gallery_topleft_loc_xy_list = gallery_dataset_test.images_topleft_loc_xy
    gallery_img_list = gallery_dataset_test.images_name

    query_dataloader_test = DataLoader(query_dataset_test,
                                    batch_size=config.batch_size,
                                    num_workers=config.num_workers,
                                    shuffle=False,
                                    pin_memory=True)
    gallery_dataloader_test = DataLoader(gallery_dataset_test,
                                       batch_size=config.batch_size,
                                       num_workers=config.num_workers,
                                       shuffle=False,
                                       pin_memory=True)
    
    print("Query Images Test:", len(query_dataset_test))
    print("Gallery Images Test:", len(gallery_dataset_test))

    # For Test Log (distance threshold) 
    dis_threshold_list = None
    if 'cross' in config.test_pairs_meta_file:
        ####### Cross-area for total 500m/10m
        print("cross-area eval")
        dis_threshold_list = [10*(i+1) for i in range(50)]
    else:
        ####### Same-area for total 200m/4m
        print("same-area eval")
        dis_threshold_list = [4*(i+1) for i in range(50)]
    
    print("\n{}[{}]{}".format(30*"-", "Evaluating GTA-UAV", 30*"-"))  

    r1_test = evaluate(config=config,
                           model=model,
                           query_loader=query_dataloader_test,
                           gallery_loader=gallery_dataloader_test, 
                           query_list=query_img_list,
                           gallery_list=gallery_img_list,
                           pairs_dict=pairs_dict,
                           ranks_list=[1, 5, 10],
                           query_center_loc_xy_list=query_center_loc_xy_list,
                           gallery_center_loc_xy_list=gallery_center_loc_xy_list,
                           gallery_topleft_loc_xy_list=gallery_topleft_loc_xy_list,
                           step_size=1000,
                           dis_threshold_list=dis_threshold_list,
                           cleanup=True,
                           plot_acc_threshold=False,
                           top10_log=False,
                           with_match=config.with_match)

    if config.log_to_file:
        f.close()
        sys.stdout = sys.__stdout__  
 


def parse_args():
    parser = argparse.ArgumentParser(description="Training script for gta.")

    parser.add_argument('--log_to_file', action='store_true', help='Log saving to file')

    parser.add_argument('--log_path', type=str, default=None, help='Log file path')

    parser.add_argument('--data_root', type=str, default='./data/GTA-UAV-data', help='Data root')
   
    parser.add_argument('--test_pairs_meta_file', type=str, default='cross-area-drone2sate-test.json', help='Test metafile path')

    parser.add_argument('--model', type=str, default='vit_base_patch16_rope_reg1_gap_256.sbb_in1k', help='Model architecture')

    parser.add_argument('--no_share_weights', action='store_true', help='Model not sharing wieghts')

    parser.add_argument('--with_match', action='store_true', help='Test with post-process image matching (GIM, etc)')

    # Added DINOv3/BoQ/LoRA args
    parser.add_argument('--with_dinov3_boq', action='store_true')
    parser.add_argument('--num_queries', type=int, default=64)
    parser.add_argument('--boq_nheads', type=int, default=8)
    parser.add_argument('--mlp_hidden_dim', type=int, default=1024)
    parser.add_argument('--mlp_output_dim', type=int, default=512)
    parser.add_argument('--unfreeze_n_blocks', type=int, default=2)

    parser.add_argument('--use_lora', action='store_true')
    parser.add_argument('--lora_r', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_dropout', type=float, default=0.1)

    parser.add_argument('--use_intermediate_layer', action='store_true')
    parser.add_argument('--intermediate_layer_idx', type=int, default=9)
    parser.add_argument('--intermediate_facet', type=str, default='token')

    parser.add_argument('--gpu_ids', type=parse_tuple, default=(0,1), help='GPU ID')

    parser.add_argument('--batch_size', type=int, default=40, help='Batch size')

    parser.add_argument('--checkpoint_start', type=str, default=None, help='Training from checkpoint')

    parser.add_argument('--test_mode', type=str, default='pos', help='Test with positive pairs')

    parser.add_argument('--query_mode', type=str, default='D2S', help='Retrieval with drone to satellite')

    # Gray Argument
    parser.add_argument('--grayscale', action='store_true', help='Use grayscale images')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    config = Configuration()
    config.data_root = args.data_root
    config.test_pairs_meta_file = args.test_pairs_meta_file
    config.log_to_file = args.log_to_file
    config.log_path = args.log_path
    config.batch_size = args.batch_size
    config.gpu_ids = args.gpu_ids
    config.checkpoint_start = args.checkpoint_start
    config.model = args.model
    config.share_weights = not(args.no_share_weights)
    config.test_mode = args.test_mode
    config.query_mode = args.query_mode
    config.with_match = args.with_match
    
    config.with_dinov3_boq = args.with_dinov3_boq
    config.num_queries = args.num_queries
    config.boq_nheads = args.boq_nheads
    config.mlp_hidden_dim = args.mlp_hidden_dim
    config.mlp_output_dim = args.mlp_output_dim
    config.unfreeze_n_blocks = args.unfreeze_n_blocks
    config.use_lora = args.use_lora
    config.lora_r = args.lora_r
    config.lora_alpha = args.lora_alpha
    config.lora_dropout = args.lora_dropout
    config.use_intermediate_layer = args.use_intermediate_layer
    config.intermediate_layer_idx = args.intermediate_layer_idx
    config.intermediate_facet = args.intermediate_facet

    # Add grayscale to config
    config.grayscale = args.grayscale

    eval_script(config)