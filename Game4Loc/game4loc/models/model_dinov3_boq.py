import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np

class GeM(nn.Module):
    """
    Generalized Mean Pooling
    p: power parameter (learnable)
    eps: small value for numerical stability
    """
    def __init__(self, p=3.0, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        # x: [B, N, D] or [B, D, N] - we assume [B, N, D] for transformer features
        # we want to pool over N (sequence length)
        # Apply numerical stability: ReLU or clamp
        # Transformers outputs can be negative, so we clamp to eps before power
        # x = x.clamp(min=self.eps) 
        # Alternatively, use absolute value or Softplus. 
        # But standard GeM is often on ReLU features. 
        # For DINOv3 features, let's use clamp(min=eps) to avoid complex numbers.
        
        return F.avg_pool1d(x.clamp(min=self.eps).pow(self.p).transpose(1, 2), kernel_size=x.size(1)).pow(1./self.p).transpose(1, 2).squeeze(1)


class BagOfQueries(nn.Module):
    """
    Bag-of-Queries (BoQ) Aggregation Module
    Uses learnable queries to aggregate dense features via attention.
    """
    def __init__(self, dim, num_queries, num_heads=8):
        super(BagOfQueries, self).__init__()
        self.num_queries = num_queries
        self.dim = dim
        self.queries = nn.Parameter(torch.randn(1, num_queries, dim))
        
        # Self-Attention for queries to interact with each other
        self.self_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm_q = nn.LayerNorm(dim)

        # Cross-Attention: Queries attend to Patches
        # Q = Queries, K = Patches, V = Patches
        self.cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm_c = nn.LayerNorm(dim)

    def forward(self, x):
        # x: [B, num_patches, dim]
        B = x.size(0)
        
        # Expand queries for batch
        queries = self.queries.expand(B, -1, -1)  # [B, num_queries, dim]
        
        # 1. Self-Attention (Query refinement)
        q_refined, _ = self.self_attn(queries, queries, queries)
        queries = self.norm_q(queries + q_refined)
        
        # 2. Cross-Attention
        # Q: queries, K: x, V: x
        # Note: Standard MultiheadAttention expects (batch, seq, feature) if batch_first=True
        aggregated, _ = self.cross_attn(query=queries, key=x, value=x)
        aggregated = self.norm_c(queries + aggregated) # Residual connection
        
        return aggregated


class MLPHeader(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super(MLPHeader, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def forward(self, x):
        return self.net(x)


class DesModelDINOv3BoQ(nn.Module):
    """
    Model Architecture:
    DINOv3 -> Patches + CLS
    Patches -> BoQ -> GeM -> Pooled
    Concat(CLS, Pooled) -> MLP -> Embedding
    """
    def __init__(self, 
                 model_name='vit_small_patch16_dinov3.lvd1689m',
                 pretrained=True,
                 img_size=384,
                 share_weights=True,
                 num_queries=64,
                 boq_nheads=8,
                 gem_p=3.0,
                 mlp_hidden_dim=1024,
                 mlp_output_dim=512):
                 
        super(DesModelDINOv3BoQ, self).__init__()
        self.share_weights = share_weights
        self.img_size = img_size
        
        print(f"Initializing DINOv3 BoQ Model: {model_name}")
        
        # 1. Backbone (DINOv3)
        # Create model but removing the head typically isn't enough as we need patches
        # We will use forward_features() method of timm models
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, img_size=img_size)
        
        # Determine feature dimension
        with torch.no_grad():
            dummy = torch.randn(1, 3, img_size, img_size)
            features = self.backbone.forward_features(dummy)
            # features shape: [B, N, D] (N = patches + 1 cls)
            if isinstance(features, tuple): # Some models return tuple
                 features = features[0]
            self.feat_dim = features.shape[-1]
            print(f"Backbone feature dimension: {self.feat_dim}")
            print(f"Backbone token count (Including CLS): {features.shape[1]}")

        # 2. Aggregation (BoQ + GeM)
        self.boq = BagOfQueries(dim=self.feat_dim, num_queries=num_queries, num_heads=boq_nheads)
        self.gem = GeM(p=gem_p)
        
        # 3. Head (MLP)
        # Input to MLP is CLS token (feat_dim) + BoQ pooled (feat_dim)
        input_dim_mlp = self.feat_dim * 2
        # Increased dropout to 0.2 for regularization on small dataset
        self.mlp = MLPHeader(input_dim=input_dim_mlp, hidden_dim=mlp_hidden_dim, output_dim=mlp_output_dim, dropout=0.2)
        
        # InfoNCE temperature (learned) - standard in this repo
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # Handle non-shared weights case
        if not share_weights:
            self.backbone2 = timm.create_model(model_name, pretrained=pretrained, num_classes=0, img_size=img_size)
            # Note: For simplicity in this advanced arch, we usually assume the Aggregation/Head are shared 
            # even if backbones aren't, but let's strictly follow the flag for backbone.
            # However, typically people share the BoQ/MLP.
            # If user asks for unshared, we duplicate everything? Usually backbone is what's unshared (e.g. satellite vs drone view diffs).
            # For now, let's keep BoQ/MLP shared to save params, but duplicate backbone if requested.
            pass

    def get_config(self):
        return timm.data.resolve_model_data_config(self.backbone)

    def set_grad_checkpointing(self, enable=True):
        self.backbone.set_grad_checkpointing(enable)
        if not self.share_weights and hasattr(self, 'backbone2'):
            self.backbone2.set_grad_checkpointing(enable)

    def freeze_layers(self, frozen_stages=[0,0,0,0]):
        # DINOv3 (ViT) doesn't have "stages" in the same way as ConvNets.
        # But we can freeze the entire backbone if requested.
        # Since this method is only called if config.freeze_layers is True,
        # we will freeze the backbone regardless of the frozen_stages values.
        
        print("Freezing Backbone parameters...")
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        if not self.share_weights and hasattr(self, 'backbone2'):
            for param in self.backbone2.parameters():
                param.requires_grad = False

    def _forward_single(self, x, backbone):
        # 1. Extract Features
        # Output: [B, N, D]
        all_tokens = backbone.forward_features(x)
        
        # 2. Split CLS and Patches
        # For DINOv3 (and v2 with registers), we must handle prefix tokens correctly.
        # num_prefix_tokens = 1 (CLS) + 4 (Registers) = 5
        num_prefix = backbone.num_prefix_tokens if hasattr(backbone, 'num_prefix_tokens') else 1
        
        # CLS is always at 0
        cls_token = all_tokens[:, 0, :]   # [B, D]
        
        # Patches start after all prefix tokens
        patch_tokens = all_tokens[:, num_prefix:, :] # [B, N-prefix, D]
        
        # 3. Bag-of-Queries Aggregation
        # [B, num_queries, D]
        boq_features = self.boq(patch_tokens)
        
        # 4. GeM Pooling
        # [B, D]
        pooled_features = self.gem(boq_features)
        
        # 5. Concatenate
        combined = torch.cat([cls_token, pooled_features], dim=1) # [B, 2*D]
        
        # 6. MLP Projection
        embedding = self.mlp(combined) # [B, out_dim]
        
        return embedding

    def forward(self, img1=None, img2=None):
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
