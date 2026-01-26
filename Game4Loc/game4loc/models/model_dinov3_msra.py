import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import math
import numpy as np
from mamba_ssm import Mamba

from .layers.conv_lora import inject_conv_lora, mark_only_lora_as_trainable


class MSRAAggregator(nn.Module):
    """
    MSRA (Mamba-based Spatial Relation-Aware) Feature Aggregator

    Paper: DINO-MSRA (Journal of Geo-information Science, 2025)

    Key formula from paper Section 2.3:
        Ê = E + HardTanh(E_pe + W_LN × M_idx)

    Where:
        - M, M_idx = max_pool(F_l) along channel axis
        - E = first_embed(M) - projects max values to K vectors
        - E_pe = learnable positional encoding
        - W_LN = learnable linear transformation for index-aware position embedding
        - Ê is processed through Mamba encoder
        - Final output: f = BN(flatten(E2 × E))

    Args:
        in_channels: Feature dimension from backbone (C)
        H, W: Spatial grid dimensions
        K: Number of projection vectors / descriptor dimension
        d_state: Mamba state dimension
        d_conv: Mamba convolution kernel size
        expand: Mamba expansion factor
        num_prefix_tokens: Number of prefix tokens to skip (e.g., CLS token)
        use_paper_index_embed: If True, use paper's linear transformation W_LN.
                               If False, use embedding lookup (original implementation).
    """

    def __init__(self, in_channels, H=16, W=16, K=64, d_state=16, d_conv=4, expand=2,
                 num_prefix_tokens=0, use_paper_index_embed=True):
        super().__init__()
        self.H = H
        self.W = W
        self.K = K
        self.in_channels = in_channels
        self.num_prefix_tokens = num_prefix_tokens
        self.use_paper_index_embed = use_paper_index_embed

        # 1. Project Max Value (M) to K descriptors
        # M is scalar per pixel (amplitude). Project to K dims.
        # Paper: "Map M to K projection vectors E=[e1, e2, ..., ek]"
        self.proj_M = nn.Linear(1, K)

        # 2. Index-aware Position Embedding
        # Paper formula: W_LN × M_idx where W_LN ∈ R^((h×w)×(K×(h×w)/2))
        # This projects channel indices to spatial feature vectors
        if use_paper_index_embed:
            # Paper-faithful: Linear transformation on one-hot encoded indices
            # W_LN transforms from channel space to K-dimensional spatial features
            # Using embedding weights as the linear transformation matrix
            self.W_LN = nn.Linear(in_channels, K, bias=False)
        else:
            # Original implementation: Discrete embedding lookup
            self.embed_idx = nn.Embedding(in_channels, K)

        # 3. Positional Embedding (E_pe)
        # Paper: "Learnable E_pe"
        self.pos_embed = nn.Parameter(torch.randn(1, H*W, K))

        # 4. Mamba Block
        # Paper: "Process Ê through Mamba for sequence modeling"
        self.mamba = Mamba(
            d_model=K,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )

        # 5. Second Projection (E2)
        # Paper: "Project Mamba output to K geometric layout descriptors E2"
        self.proj_E2 = nn.Linear(K, K)

        # 6. BatchNorm for final descriptor
        # Paper: "f = BN(flatten(E2 × E))"
        self.bn = nn.BatchNorm1d(K * H * W)
        
    def forward(self, x):
        """
        Forward pass of MSRA aggregator.

        Args:
            x: Input features [B, N, D] where N=num_tokens, D=feature_dim

        Returns:
            Normalized descriptor [B, K*H*W]
        """
        # x: [B, N, D]
        B, N, D = x.shape

        # Remove prefix tokens (e.g., CLS token)
        if N > self.num_prefix_tokens:
            features = x[:, self.num_prefix_tokens:, :]
        else:
            # Should not happen with valid input
            features = x

        # Ensure shape matches H*W
        current_L = features.shape[1]

        # Reshape for max pooling
        # [B, L, D] -> [B, D, L]
        feat_trans = features.transpose(1, 2)

        # 1. Max Pooling along channel axis (dim=1)
        # Paper: "M, M_idx = F_l.max(l)"
        # M: [B, L] - aggregated feature values
        # M_idx: [B, L] - position indices of max values (which channel contributed)
        M, M_idx = torch.max(feat_trans, dim=1)

        # Expand dims for projection
        M_flat = M.unsqueeze(-1)  # [B, L, 1]

        # 2. First Embedding Layer
        # Paper: "Map M to K projection vectors E=[e1, e2, ..., ek]"
        E = self.proj_M(M_flat)  # [B, L, K]

        # 3. Index-aware Position Embedding
        # Paper formula: Ê = E + HardTanh(E_pe + W_LN × M_idx)
        if self.use_paper_index_embed:
            # Paper-faithful: Linear transformation W_LN on one-hot encoded indices
            # Convert M_idx to one-hot: [B, L] -> [B, L, in_channels]
            M_idx_onehot = F.one_hot(M_idx, num_classes=self.in_channels).float()
            # Apply linear transformation: [B, L, in_channels] @ [in_channels, K] -> [B, L, K]
            E_idx = self.W_LN(M_idx_onehot)
        else:
            # Original implementation: Embedding lookup
            E_idx = self.embed_idx(M_idx)  # [B, L, K]

        # Positional Encoding
        if current_L == self.pos_embed.shape[1]:
            E_pe = self.pos_embed
        else:
            # Interpolate PE if size mismatch (e.g., 384 vs 256 image)
            E_pe = F.interpolate(
                self.pos_embed.transpose(1, 2),
                size=current_L, mode='linear'
            ).transpose(1, 2)

        # Combine: Ê = E + HardTanh(E_pe + W_LN × M_idx)
        E_hat = E + F.hardtanh(E_pe + E_idx)

        # 4. Mamba Encoder
        # Paper: "Process Ê through Mamba for sequence modeling"
        mamba_out = self.mamba(E_hat)

        # 5. Second Embedding Layer
        # Paper: "Project Mamba output to K geometric layout descriptors E2"
        E2 = self.proj_E2(mamba_out)  # [B, L, K]

        # 6. Fusion
        # Paper: "f = BN(flatten(E2 × E))"
        out = E2 * E

        # Flatten
        out_flat = out.reshape(B, -1)  # [B, L*K]

        # BatchNorm
        if out_flat.shape[1] == self.bn.num_features:
            out_norm = self.bn(out_flat)
        else:
            # Skip BN if size varies (shouldn't happen in normal operation)
            out_norm = out_flat

        # L2 Normalize (not explicitly in paper, but standard for retrieval)
        return F.normalize(out_norm, p=2, dim=1)


class DesModelDINOv3MSRA(nn.Module):
    """
    DINOv3 backbone with MSRA (Mamba-based Spatial Relation-Aware) aggregator.

    This model combines:
    - DINOv3 Vision Transformer backbone for feature extraction
    - MSRA aggregator for spatial relationship modeling via Mamba SSM
    - Optional Conv-LoRA for parameter-efficient fine-tuning

    Args:
        model_name: timm model name for DINOv3 backbone
        pretrained: Whether to use pretrained weights
        img_size: Input image size
        share_weights: Whether to share backbone weights between query and gallery
        msra_K: Number of projection vectors in MSRA
        msra_d_state: Mamba state dimension
        use_paper_index_embed: If True, use paper's W_LN linear transformation.
                               If False, use embedding lookup (original implementation).
        use_lora: Whether to use Conv-LoRA
        lora_r: LoRA rank
        lora_alpha: LoRA alpha scaling factor
        lora_dropout: LoRA dropout rate
    """

    def __init__(self,
                 model_name='vit_small_patch16_dinov3.lvd1689m',
                 pretrained=True,
                 img_size=256,
                 share_weights=True,
                 # MSRA Args
                 msra_K=64,
                 msra_d_state=16,
                 use_paper_index_embed=True,
                 # LoRA Args
                 use_lora=True,
                 lora_r=8,
                 lora_alpha=16,
                 lora_dropout=0.1):
        super().__init__()
        self.share_weights = share_weights

        print(f"Initializing DINOv3 MSRA Model: {model_name}, LoRA={use_lora}")
        print(f"  MSRA config: K={msra_K}, d_state={msra_d_state}, paper_index_embed={use_paper_index_embed}")
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, img_size=img_size)
        
        # Get prefix tokens count
        if hasattr(self.backbone, 'num_prefix_tokens'):
            self.num_prefix_tokens = self.backbone.num_prefix_tokens
        else:
            self.num_prefix_tokens = 1 # Warning: Assumption
            print("Warning: num_prefix_tokens not found, assuming 1 (CLS)")
            
        print(f"Model num_prefix_tokens: {self.num_prefix_tokens}")
        
        # Inject Conv-LoRA
        if use_lora:
            print(f"Injecting ConvLoRA (r={lora_r}) into backbone...")
            # Pass prefix tokens so ConvLoRA handles spatial reshape correctly
            self.backbone = inject_conv_lora(self.backbone, r=lora_r, alpha=lora_alpha, dropout=lora_dropout, num_prefix_tokens=self.num_prefix_tokens)
            
            # Freeze non-LoRA params
            mark_only_lora_as_trainable(self.backbone)
            self.use_lora = True
        else:
            self.use_lora = False
            
        # Get feature dim
        dummy = torch.randn(1, 3, img_size, img_size)
        with torch.no_grad():
            feats = self.backbone.forward_features(dummy)
            dim = feats.shape[-1]
            num_tokens = feats.shape[1]
            # Infer H, W
            # Assuming square spatial grid
            hw_tokens = num_tokens - self.num_prefix_tokens
            H = W = int(math.sqrt(hw_tokens))
            
        print(f"Backbone dim: {dim}, Grid: {H}x{W} (tokens={num_tokens}, prefix={self.num_prefix_tokens})")
        
        # Aggregator
        self.aggregator = MSRAAggregator(
            in_channels=dim,
            H=H, W=W,
            K=msra_K,
            d_state=msra_d_state,
            num_prefix_tokens=self.num_prefix_tokens,
            use_paper_index_embed=use_paper_index_embed
        )
        
        # InfoNCE Scale
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        if not share_weights:
            self.backbone2 = timm.create_model(model_name, pretrained=pretrained, num_classes=0, img_size=img_size)
            if use_lora:
                self.backbone2 = inject_conv_lora(self.backbone2, r=lora_r, alpha=lora_alpha, dropout=lora_dropout, num_prefix_tokens=self.num_prefix_tokens)
                mark_only_lora_as_trainable(self.backbone2)

    def forward(self, img1=None, img2=None):
        def forward_one(x, backbone):
            tokens = backbone.forward_features(x)
            desc = self.aggregator(tokens)
            return desc
            
        if self.share_weights:
            if img1 is not None and img2 is not None:
                return forward_one(img1, self.backbone), forward_one(img2, self.backbone)
            elif img1 is not None:
                return forward_one(img1, self.backbone)
            else:
                return forward_one(img2, self.backbone)
        else:
             if img1 is not None and img2 is not None:
                return forward_one(img1, self.backbone), forward_one(img2, self.backbone2)
