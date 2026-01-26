import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os

# Add MatchAnything to Python path
sys.path.insert(0, '/media/aniel/storage/dataset_to_analize/MatchAnything/imcui/third_party/MatchAnything/src')

from loftr.backbone import build_backbone
from config.default import get_cfg_defaults

# Import BoQ from the existing DINOv3+BoQ model
from .model_dinov3_boq import BoQ


class DesModelMatchAnythingBoQ(nn.Module):
    """
    Model Architecture:
    MatchAnything ResNet+FPN (Frozen) -> BoQ Aggregator (Cascaded) -> Descriptor

    This model uses the MatchAnything ResNet+FPN backbone for coarse feature extraction,
    keeping it fully frozen to preserve fine matching capabilities. Features are then
    aggregated using Bag-of-Queries (BoQ) for retrieval.
    """
    def __init__(self,
                 checkpoint_path='/media/aniel/storage/dataset_to_analize/MatchAnything/matchanything_eloftr.ckpt',
                 img_size=384,
                 share_weights=True,
                 num_queries=64,
                 num_layers=2,
                 mlp_output_dim=512):
        super(DesModelMatchAnythingBoQ, self).__init__()
        self.share_weights = share_weights
        self.img_size = img_size

        print(f"Initializing MatchAnything BoQ Model")

        # 1. Load MatchAnything Config
        config = get_cfg_defaults()
        # Set to extract only coarse features (skip fine features for efficiency)
        config.LOFTR.COARSE_FEAT_ONLY = True
        config.LOFTR.RESNETFPN.COARSE_FEAT_ONLY = True

        # Build backbone config dict
        backbone_config = {
            'backbone_type': config.LOFTR.BACKBONE_TYPE,
            'align_corner': config.LOFTR.ALIGN_CORNER,
            'resolution': config.LOFTR.RESOLUTION,
            'resnetfpn': {
                'initial_dim': config.LOFTR.RESNETFPN.INITIAL_DIM,
                'block_dims': config.LOFTR.RESNETFPN.BLOCK_DIMS,
                'coarse_feat_only': config.LOFTR.RESNETFPN.COARSE_FEAT_ONLY,
                'inter_feat': config.LOFTR.RESNETFPN.INTER_FEAT,
                'leaky': config.LOFTR.RESNETFPN.LEAKY,
                'repvggmodel': config.LOFTR.RESNETFPN.REPVGGMODEL,
            }
        }

        # 2. Build Backbone
        print("Building ResNet+FPN backbone...")
        self.backbone = build_backbone(backbone_config)

        # 3. Load Checkpoint and Freeze
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')

            # Extract state dict
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            # Filter backbone weights (keys start with 'backbone.')
            backbone_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('backbone.'):
                    # Remove 'backbone.' prefix
                    new_key = key[9:]  # len('backbone.') = 9
                    backbone_state_dict[new_key] = value

            # Load weights
            missing, unexpected = self.backbone.load_state_dict(backbone_state_dict, strict=False)
            if missing:
                print(f"Warning: Missing keys in backbone: {missing}")
            if unexpected:
                print(f"Warning: Unexpected keys in backbone: {unexpected}")

            print("Checkpoint loaded successfully")
        else:
            print(f"Warning: Checkpoint not found at {checkpoint_path}, using random initialization")

        # 4. Freeze ALL backbone parameters (no LoRA, fully frozen)
        print("Freezing all backbone parameters...")
        trainable_count = 0
        frozen_count = 0
        for param in self.backbone.parameters():
            param.requires_grad = False
            frozen_count += param.numel()

        print(f"Frozen {frozen_count:,} parameters in backbone")

        # Get feature dimensions from backbone
        # ResNet+FPN outputs coarse features with 256 channels (last block_dim)
        # Input: [B, 1, H, W] -> Output: {'feats_c': [B, 256, H/8, W/8], 'feats_f': None}
        embed_dim = 256  # From config.LOFTR.RESNETFPN.BLOCK_DIMS[-1]

        print(f"Backbone coarse feature dimension: {embed_dim}")

        # 5. Aggregator (BoQ)
        # We want final output `mlp_output_dim` (e.g. 512).
        # BoQ output size is `proj_channels * row_dim`.
        # Standard: proj_channels=512, row_dim=1 for 512-dim output

        proj_channels = 512
        row_dim = mlp_output_dim // proj_channels
        if row_dim < 1:
            # If requested output is small (e.g. 256), adjust proj_channels
            proj_channels = mlp_output_dim
            row_dim = 1

        print(f"BoQ Configuration: In={embed_dim}, Proj={proj_channels}, NumQueries={num_queries}, NumLayers={num_layers}, RowDim={row_dim}, Out={proj_channels*row_dim}")

        self.aggregator = BoQ(
            in_channels=embed_dim,
            proj_channels=proj_channels,
            num_queries=num_queries,
            num_layers=num_layers,
            row_dim=row_dim
        )

        # Count trainable parameters in aggregator
        trainable_count = sum(p.numel() for p in self.aggregator.parameters() if p.requires_grad)
        print(f"Trainable parameters in BoQ: {trainable_count:,}")

        # InfoNCE temperature (learned)
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        if not share_weights:
            print("Initializing second backbone for asymmetric retrieval...")
            self.backbone2 = build_backbone(backbone_config)
            if checkpoint_path and os.path.exists(checkpoint_path):
                self.backbone2.load_state_dict(backbone_state_dict, strict=False)
            # Freeze second backbone too
            for param in self.backbone2.parameters():
                param.requires_grad = False

    def rgb_to_grayscale(self, rgb_img):
        """
        Convert RGB images to grayscale using standard luminosity formula.

        Args:
            rgb_img: [B, 3, H, W] RGB images

        Returns:
            gray_img: [B, 1, H, W] Grayscale images
        """
        # Standard RGB to grayscale conversion: Y = 0.299*R + 0.587*G + 0.114*B
        gray = 0.299 * rgb_img[:, 0] + 0.587 * rgb_img[:, 1] + 0.114 * rgb_img[:, 2]
        return gray.unsqueeze(1)  # [B, 1, H, W]

    def spatial_to_sequence(self, feats_c):
        """
        Convert spatial feature maps to sequence format for BoQ processing.

        Args:
            feats_c: [B, C, H, W] Spatial feature maps

        Returns:
            feats_seq: [B, H*W, C] Sequence of features
        """
        B, C, H, W = feats_c.shape
        # Flatten spatial dimensions: [B, C, H, W] -> [B, C, H*W]
        feats_flat = feats_c.flatten(2)  # [B, C, H*W]
        # Permute to sequence format: [B, C, H*W] -> [B, H*W, C]
        feats_seq = feats_flat.permute(0, 2, 1)  # [B, H*W, C]
        return feats_seq

    def _forward_single(self, x, backbone):
        """
        Forward pass for a single image.

        Args:
            x: [B, 3, H, W] RGB image
            backbone: ResNet+FPN backbone

        Returns:
            descriptor: [B, mlp_output_dim] L2-normalized descriptor
        """
        # 1. Convert RGB to Grayscale
        # Input: [B, 3, 384, 384] -> [B, 1, 384, 384]
        gray = self.rgb_to_grayscale(x)

        # 2. Extract Coarse Features via Backbone
        # Input: [B, 1, 384, 384] -> Output: {'feats_c': [B, 256, 48, 48], 'feats_f': None}
        feats = backbone(gray)
        feats_c = feats['feats_c']  # [B, 256, 48, 48] at 1/8 resolution

        # 3. Convert Spatial Features to Sequence
        # [B, 256, 48, 48] -> [B, 2304, 256]
        feats_seq = self.spatial_to_sequence(feats_c)

        # 4. Aggregation (BoQ - Cascaded)
        # Input: [B, 2304, 256] -> Output: [B, 512]
        descriptor, _ = self.aggregator(feats_seq)

        # 5. L2 Normalization (Essential for retrieval)
        descriptor = F.normalize(descriptor, p=2, dim=1)

        return descriptor

    def forward(self, img1=None, img2=None):
        """
        Forward pass handling single or dual image input.

        Args:
            img1: [B, 3, H, W] First image (query/drone)
            img2: [B, 3, H, W] Second image (gallery/satellite)

        Returns:
            If both images: (emb1, emb2) tuple
            If single image: emb (single descriptor)
        """
        if self.share_weights:
            if img1 is not None and img2 is not None:
                emb1 = self._forward_single(img1, self.backbone)
                emb2 = self._forward_single(img2, self.backbone)
                return emb1, emb2
            elif img1 is not None:
                return self._forward_single(img1, self.backbone)
            else:
                return self._forward_single(img2, self.backbone)
        else:
            if img1 is not None and img2 is not None:
                emb1 = self._forward_single(img1, self.backbone)
                emb2 = self._forward_single(img2, self.backbone2)
                return emb1, emb2
            elif img1 is not None:
                return self._forward_single(img1, self.backbone)
            else:
                # Typically gallery is second
                return self._forward_single(img2, self.backbone2)

    def get_config(self):
        """
        Return a dummy config for compatibility.
        MatchAnything doesn't use timm, so we return a simple dict.
        """
        return {
            'input_size': (3, self.img_size, self.img_size),
            'mean': (0.485, 0.456, 0.406),
            'std': (0.229, 0.224, 0.225),
        }

    def set_grad_checkpointing(self, enable=True):
        """
        Gradient checkpointing not implemented for ResNet+FPN.
        Backbone is frozen anyway, so this is not needed.
        """
        if enable:
            print("Warning: Gradient checkpointing not implemented for MatchAnything backbone")
        pass

    def freeze_layers(self, frozen_stages=[0,0,0,0]):
        """
        Freeze backbone layers. Already frozen in __init__, so this is a no-op.
        """
        print("Backbone already frozen in __init__")
        pass
