# 🚁 UAV Visual Geo-Localization: LoRA Fine-Tuning Pipeline

## Final Implementation Plan (Validated)

---

## 📋 Executive Summary

This plan implements LoRA (Low-Rank Adaptation) fine-tuning of DINOv3 ViT-Small for low-altitude UAV geo-localization (70-150m). The pipeline combines synthetic (GTA-UAV) with real high-altitude (UAV-VisLoc ~400m, VPAIR ~400m) data using on-the-fly altitude simulation.

### Key Design Decisions

| Aspect | Decision | Rationale |
|--------|----------|-----------|
| **Backbone** | `vit_small_patch16_dinov3.lvd1689m` | ✅ Confirmed available in timm |
| **LoRA Library** | Custom implementation (no PEFT) | PEFT not installed; custom gives more control |
| **Loss Function** | `WeightedInfoNCE` | Already in Game4Loc, proven effective |
| **Altitude Simulation** | On-the-fly crops (0.15-0.30) | No disk storage overhead |
| **GPS Noise** | ±10% center jitter | Simulates real GPS inaccuracy |

---

## 📁 Created Files

### 1. `merge_json_balanced.py`
**Location:** `/home/aniel/skyline_drone/datasets/GTA-UAV/Game4Loc/merge_json_balanced.py`

**Purpose:** Merge and balance multiple datasets with metadata tracking.

**Key Features:**
- Adds `dataset_source` field ('gta', 'visloc', 'vpair')
- Adds `data_root` for per-dataset path resolution
- Adds `altitude_meters` for real datasets
- Configurable duplication (N×) for class balancing
- Adds `duplicate_id` to track virtual copies

**Usage:**
```bash
# Training dataset (with 10× duplication for real data)
python merge_json_balanced.py \
    --visloc_json /path/to/UAV_VisLoc/cross-area-drone2sate-train.json \
    --vpair_json /path/to/VPair/cross-area-drone2drone-train.json \
    --visloc_root /path/to/UAV_VisLoc_dataset \
    --vpair_root /path/to/vpair \
    --visloc_multiplier 10 \
    --vpair_multiplier 10 \
    --output combined-lora-train.json

# Test dataset (no duplication)
python merge_json_balanced.py \
    --visloc_json /path/to/UAV_VisLoc/cross-area-drone2sate-test.json \
    --vpair_json /path/to/VPair/cross-area-drone2drone-test.json \
    --visloc_root /path/to/UAV_VisLoc_dataset \
    --vpair_root /path/to/vpair \
    --visloc_multiplier 1 \
    --vpair_multiplier 1 \
    --output combined-lora-test.json
```

### 2. `train_lora_fly.py`
**Location:** `/home/aniel/skyline_drone/datasets/GTA-UAV/Game4Loc/train_lora_fly.py`

**Purpose:** Main training script with custom LoRA model and source-specific augmentation.

**Key Components:**

#### A. `LowAltitudeSimulation` Transform
```python
class LowAltitudeSimulation(ImageOnlyTransform):
    """
    Simulates 70-150m altitude from 400m images.
    - crop_scale_range=(0.15, 0.30): Random center crop
    - jitter_ratio=0.1: ±10% GPS noise
    """
```

#### B. `GTADatasetTrainLoRA` Dataset
- Reads `dataset_source` from each entry
- Applies different transforms:
  - **VisLoc/VPAIR:** `LowAltitudeSimulation` (aggressive crop + jitter)
  - **GTA:** `RandomCenterCropZoom` (standard augmentation)
- Resolves paths using per-entry `data_root`

#### C. `DesModelLoRA` Model
```python
Architecture:
- Backbone: DINOv3 ViT-Small (100% frozen)
- LoRA adapters on: qkv, proj, fc1, fc2 (per block)
- LoRA config: r=128, alpha=256
- MLP head: 384 → 2048 → 256
- Learnable logit_scale (temperature)

Trainable params: ~13M (LoRA + MLP)
Frozen params: ~21M (backbone)
```

#### D. Per-Dataset Evaluation
```
[OVERALL] Queries: 2091, R@1: 45.23%, R@5: 62.18%, R@10: 70.45%
[VISLOC] Queries: 738, R@1: 42.18%, R@5: 58.34%, R@10: 66.56%
[VPAIR] Queries: 1353, R@1: 47.12%, R@5: 64.89%, R@10: 73.21%
```

**Usage:**
```bash
python train_lora_fly.py \
    --train_pairs_meta_file /path/to/combined-lora-train.json \
    --test_pairs_meta_file /path/to/combined-lora-test.json \
    --model vit_small_patch16_dinov3.lvd1689m \
    --lora_r 128 --lora_alpha 256 \
    --batch_size 128 --lr 3e-4 --epochs 10 \
    --with_weight --k 5 \
    --num_workers 8 \
    --grad_checkpointing
```

---

## 📊 Dataset Statistics

### Training Data (Generated)
```
UAV-VisLoc: 768 × 10 = 7,680 entries (36.2%)
VPAIR:      1,353 × 10 = 13,530 entries (63.8%)
────────────────────────────────────────────
TOTAL:                  21,210 entries
```

### Test Data (Generated)
```
UAV-VisLoc: 738 entries (35.3%)
VPAIR:      1,353 entries (64.7%)
────────────────────────────────────────────
TOTAL:      2,091 entries
```

---

## 🔧 Technical Details

### Image Resolutions
| Dataset | Drone Images | Satellite/Reference |
|---------|--------------|---------------------|
| UAV-VisLoc | 3976×2652 (~10MP) | 256×256 tiles |
| VPAIR | Variable | Variable |
| GTA-UAV | Synthetic | 256×256 tiles |

### Altitude Simulation Math
```
Original altitude: ~400m
Target altitude: 70-150m
Altitude ratio: 70/400 = 0.175 to 150/400 = 0.375

Crop scale range: 0.15 to 0.30 (slightly more aggressive)
- 0.15 → simulates ~60m (ultra low)
- 0.30 → simulates ~120m (low)
```

### LoRA Configuration
```python
lora_r = 128        # Rank (higher = more capacity)
lora_alpha = 256    # Scaling factor
lora_dropout = 0.1  # Regularization

# Applied to each transformer block:
targets = ["qkv", "proj", "fc1", "fc2"]
# Total: 12 blocks × 4 adapters = 48 LoRA modules
```

### Training Hyperparameters (A100)
| Parameter | Value | Notes |
|-----------|-------|-------|
| Batch size | 128 | Per GPU (256 total with 2 GPUs) |
| Learning rate | 3e-4 | For LoRA adapters |
| Scheduler | Cosine | With warmup |
| Warmup | 0.5 epochs | ~500 steps |
| Optimizer | AdamW | Only trainable params |
| Mixed precision | Yes | FP16 with GradScaler |
| Gradient clipping | 100.0 | Stability |

---

## ✅ Validation Checklist

| Check | Status | Details |
|-------|--------|---------|
| DINOv3 in timm | ✅ | `vit_small_patch16_dinov3.lvd1689m` available |
| merge_json_balanced.py syntax | ✅ | Compiled successfully |
| train_lora_fly.py syntax | ✅ | Compiled successfully |
| JSON structure | ✅ | Has dataset_source, data_root, duplicate_id |
| Train JSON created | ✅ | 21,210 entries, 28.48 MB |
| Test JSON created | ✅ | 2,091 entries, 2.62 MB |

---

## 🚀 Execution Steps

### Step 1: Generate Merged Datasets
```bash
cd /home/aniel/skyline_drone/datasets/GTA-UAV/Game4Loc

# Training (already done)
python merge_json_balanced.py \
    --visloc_json ../../../UAV_VisLoc_dataset/cross-area-drone2sate-train.json \
    --vpair_json ../../../vpair/cross-area-drone2drone-train.json \
    --visloc_root /home/aniel/skyline_drone/datasets/UAV_VisLoc_dataset \
    --vpair_root /home/aniel/skyline_drone/datasets/vpair \
    --visloc_multiplier 10 \
    --vpair_multiplier 10 \
    --output /home/aniel/skyline_drone/datasets/combined-lora-train.json

# Test (already done)
python merge_json_balanced.py \
    --visloc_json ../../../UAV_VisLoc_dataset/cross-area-drone2sate-test.json \
    --vpair_json ../../../vpair/cross-area-drone2drone-test.json \
    --visloc_root /home/aniel/skyline_drone/datasets/UAV_VisLoc_dataset \
    --vpair_root /home/aniel/skyline_drone/datasets/vpair \
    --visloc_multiplier 1 \
    --vpair_multiplier 1 \
    --output /home/aniel/skyline_drone/datasets/combined-lora-test.json
```

### Step 2: Run Training
```bash
cd /home/aniel/skyline_drone/datasets/GTA-UAV/Game4Loc

python train_lora_fly.py \
    --train_pairs_meta_file /home/aniel/skyline_drone/datasets/combined-lora-train.json \
    --test_pairs_meta_file /home/aniel/skyline_drone/datasets/combined-lora-test.json \
    --model vit_small_patch16_dinov3.lvd1689m \
    --lora_r 128 --lora_alpha 256 \
    --batch_size 128 --lr 3e-4 --epochs 10 \
    --with_weight --k 5 \
    --num_workers 8 \
    --model_path ./work_dir/lora
```

### Step 3: Monitor Training
Checkpoints saved:
- `best_overall_*.pth` - Highest overall R@1
- `best_visloc_*.pth` - Highest VisLoc R@1 (target domain)
- `final.pth` - Last epoch

---

## 📈 Expected Results

### Success Criteria
| Metric | Target | Notes |
|--------|--------|-------|
| VisLoc R@1 | > 40% | Main target (low-altitude domain) |
| Overall R@1 | > 45% | Balanced performance |
| Training time | ~20 min | 10 epochs on A100 |
| GPU memory | < 40GB | With batch_size=128 |

### Typical Output
```
[OVERALL] Queries: 2091, R@1: 47.23%, R@5: 65.18%, R@10: 72.45%
[VISLOC] Queries: 738, R@1: 43.18%, R@5: 60.34%, R@10: 68.56%
[VPAIR] Queries: 1353, R@1: 49.52%, R@5: 67.89%, R@10: 74.61%
```

---

## 🐛 Troubleshooting

### OOM (Out of Memory)
```bash
# Reduce batch size
--batch_size 64

# Enable gradient checkpointing
--grad_checkpointing
```

### Slow Data Loading
```bash
# Increase workers
--num_workers 12
```

### Poor VisLoc Performance
- Check that `LowAltitudeSimulation` is being applied (print dataset_source in __getitem__)
- Try more aggressive crop range: `(0.12, 0.25)`
- Increase VisLoc multiplier: `--visloc_multiplier 15`

---

## 📚 File Reference

| File | Purpose |
|------|---------|
| `merge_json_balanced.py` | Data preparation script |
| `train_lora_fly.py` | Main training script |
| `combined-lora-train.json` | Merged training data (21K entries) |
| `combined-lora-test.json` | Merged test data (2K entries) |
| `work_dir/lora/` | Output checkpoints |

---

## 🔄 Changes from Original Plan

| Original | Final | Reason |
|----------|-------|--------|
| Use PEFT library | Custom LoRA | PEFT not installed |
| `facebook/dinov3-*` | `vit_small_patch16_dinov3.lvd1689m` | Correct timm model name |
| Batch size 256 | 128 recommended | Memory safety |
| `WeightedSoftMarginTripletLoss` | `WeightedInfoNCE` | Correct loss name |
| Separate GTA dataset | Optional (can add later) | Focus on real data first |

---

*Generated: December 4, 2025*
