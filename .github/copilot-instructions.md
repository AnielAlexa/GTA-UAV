# Game4Loc Copilot Instructions

## Project Overview

**Game4Loc** is a UAV geo-localization benchmark that trains models to match drone-view images with satellite imagery. The project uses GTA-V simulated data (GTA-UAV dataset) and supports real-world datasets like UAV-VisLoc.

### Architecture

```
GTA-UAV/
├── Game4Loc/           # Main training/evaluation framework (PyTorch)
│   ├── game4loc/       # Core library
│   │   ├── dataset/    # Dataset loaders (gta.py, visloc.py, etc.)
│   │   ├── models/     # Vision models (timm-based DesModel)
│   │   ├── trainer/    # Training loops with weighted InfoNCE loss
│   │   ├── evaluate/   # Evaluation with Recall@K, SDM metrics
│   │   └── matcher/    # Optional image matching for finer localization
│   ├── train_*.py      # Training scripts per dataset
│   └── eval_*.py       # Evaluation scripts per dataset
├── DeepGTAV/           # GTA-V plugin for data collection (C++/Python)
└── scripts/            # Dataset preprocessing utilities
```

## Key Concepts

### Dataset Pairing Strategy
- **Positive pairs**: Drone-satellite pairs with high IoU overlap (>0.39)
- **Semi-positive pairs**: Lower overlap pairs (>0.14) for training robustness
- Use `train_mode='pos_semipos'` for training, `test_mode='pos'` for evaluation
- JSON metadata files: `{cross|same}-area-drone2sate-{train|test}.json`

### Training Configuration Pattern
All training scripts use a `@dataclass Configuration` pattern. Key parameters:
```python
model: str = 'vit_base_patch16_rope_reg1_gap_256.sbb_in1k'  # timm model name
img_size: int = 384
with_weight: bool = True    # Use weighted InfoNCE loss (IoU-based weights)
share_weights: bool = True  # Single encoder for drone+satellite
k: float = 5                # Temperature parameter for weight function
gpu_ids: tuple = (0, 1)     # Multi-GPU via DataParallel
```

### Model Architecture
The `DesModel` class (`game4loc/models/model.py`) wraps timm models:
- Supports ViT, ConvNeXt, Swin variants
- Learnable `logit_scale` parameter (CLIP-style)
- `share_weights=True`: Same encoder for both views (default)

## Common Commands

```bash
# Install dependencies
cd Game4Loc && pip install -r requirements.txt

# Train on GTA-UAV (cross-area setting)
python train_gta.py \
    --data_root /path/to/GTA-UAV \
    --train_pairs_meta_file "cross-area-drone2sate-train.json" \
    --test_pairs_meta_file "cross-area-drone2sate-test.json" \
    --model "vit_base_patch16_rope_reg1_gap_256.sbb_in1k" \
    --gpu_ids 0,1 --lr 0.0001 --batch_size 64 --with_weight --k 5 --epoch 5

# Evaluate with pre-trained checkpoint
python eval_gta.py \
    --data_root /path/to/GTA-UAV \
    --test_pairs_meta_file "cross-area-drone2sate-test.json" \
    --checkpoint_start /path/to/weights.pth --gpu_ids 0
```

## Dataset-Specific Scripts

| Dataset | Train Script | Eval Script | Dataset Class |
|---------|--------------|-------------|---------------|
| GTA-UAV | `train_gta.py` | `eval_gta.py` | `game4loc/dataset/gta.py` |
| UAV-VisLoc | `train_visloc.py` | `eval_visloc.py` | `game4loc/dataset/visloc.py` |
| DenseUAV | `train_denseuav_extend.py` | `eval_denseuav.py` | `game4loc/dataset/denseuav_extend.py` |
| University-1652 | `train_university.py` | `eval_university.py` | `game4loc/dataset/university.py` |

## Code Patterns

### Adding a New Dataset
1. Create dataset class in `game4loc/dataset/` following `gta.py` pattern
2. Implement `__getitem__` returning `(query_img, gallery_img, weight)`
3. Implement `shuffle_group()` for Mutually Exclusive Sampling
4. Create corresponding `train_*.py` and `eval_*.py` scripts

### Data Augmentation
Transforms defined in `game4loc/transforms.py` using albumentations:
- Training: ColorJitter, blur, dropout augmentations
- Eval: Resize + normalize only
- Shared horizontal flip applied to both drone and satellite images

### Loss Function
`WeightedInfoNCE` (`game4loc/loss.py`): IoU-weighted contrastive loss
- Higher IoU → lower label smoothing → stronger positive signal
- Formula: `eps = 1 - 1/(1 + exp(-k * weight))`

## Preprocessing Real-World Data

Use `scripts/prepare_dataset/visloc.py` as reference for converting datasets to GTA-UAV format:
- Tile satellite maps at multiple zoom levels
- Calculate IoU weights between drone FoV projections and tiles
- Output JSON with `pair_pos_sate_*` and `pair_pos_semipos_sate_*` lists

## Notes

- Checkpoints saved to `work_dir/{dataset}/{model}/{timestamp}/`
- Set `num_workers=0` on Windows (auto-detected via `os.name`)
- Use `--zero_shot` flag to evaluate before training starts
- For finer localization, enable `with_match=True` in eval scripts (uses GIM matcher)
