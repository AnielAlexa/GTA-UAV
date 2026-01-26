import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np

try:
    from peft import get_peft_model, LoraConfig, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

class BoQBlock(nn.Module):
    def __init__(self, in_dim, num_queries, nheads=8):
        super(BoQBlock, self).__init__()
        
        self.encoder = nn.TransformerEncoderLayer(d_model=in_dim, nhead=nheads, dim_feedforward=4*in_dim, batch_first=True, dropout=0.)
        self.queries = nn.Parameter(torch.randn(1, num_queries, in_dim))
        
        # Self-attention for queries
        self.self_attn = nn.MultiheadAttention(in_dim, num_heads=nheads, batch_first=True)
        self.norm_q = nn.LayerNorm(in_dim)
        
        # Cross-attention: Queries attend to X
        self.cross_attn = nn.MultiheadAttention(in_dim, num_heads=nheads, batch_first=True)
        self.norm_out = nn.LayerNorm(in_dim)
        
    def forward(self, x):
        B = x.size(0)
        x = self.encoder(x) # Update image features
        
        q = self.queries.repeat(B, 1, 1)
        
        # Self-Attention on Queries
        q_attn, _ = self.self_attn(q, q, q)
        q = q + q_attn
        q = self.norm_q(q)
        
        # Cross-Attention
        out, attn = self.cross_attn(q, x, x)        
        out = self.norm_out(out)
        
        # Return updated image features for next layer, and the query output
        return x, out, attn.detach()


class BoQSequence(nn.Module):
    """
    BoQ aggregator that expects a sequence input (B, N, D).
    Unlike standard BoQ, this class assumes the input is already projected 
    and flattened into a sequence.
    """
    def __init__(self, in_channels, num_queries=32, num_layers=2, row_dim=32):
        super().__init__()
        # Input normalization
        self.norm_input = nn.LayerNorm(in_channels)
        
        self.boqs = nn.ModuleList([
            BoQBlock(in_channels, num_queries, nheads=in_channels//64) for _ in range(num_layers)])
        
        self.fc = nn.Linear(num_layers * num_queries, row_dim)
        
    def forward(self, x):
        # x: [B, N, in_channels]
        x = self.norm_input(x)
        
        outs = []
        attns = []
        for i in range(len(self.boqs)):
            x, out, attn = self.boqs[i](x)
            outs.append(out)
            attns.append(attn)

        # Concatenate outputs from all layers
        out = torch.cat(outs, dim=1) # [B, num_layers*num_queries, in_channels]
        
        # Compress the 'number of queries' dimension
        out = self.fc(out.permute(0, 2, 1)) # -> [B, in_channels, row_dim]
        
        return out.flatten(1), attns # [B, in_channels * row_dim]


class DesModelConvNextBoQ(nn.Module):
    """
    Model Architecture:
    ConvNeXt (Stage 3 & 4) -> 2x Conv Projection -> Concat -> BoQ Aggregator
    """
    def __init__(self, 
                 model_name='convnext_tiny.dinov3_lvd1689m',
                 pretrained=True,
                 img_size=384,
                 share_weights=True,
                 num_queries=64,
                 mlp_output_dim=512,
                 unfreeze_n_blocks=2,
                 use_lora=True,
                 lora_r=16,
                 lora_alpha=32,
                 lora_dropout=0.1):
        super(DesModelConvNextBoQ, self).__init__()
        self.share_weights = share_weights
        self.use_lora = use_lora
        
        print(f"Initializing ConvNeXt DINOv3 BoQ Model: {model_name}")
        
        # 1. Backbone (Extract features from Stage 4)
        # out_indices=[3] corresponds to Stage 4
        self.backbone = timm.create_model(
            model_name, 
            pretrained=pretrained, 
            features_only=True, 
            out_indices=[3]
        )
        
        # Get feature channels
        feature_info = self.backbone.feature_info
        channels = feature_info.channels()
        print(f"Feature channels from Stage 4: {channels}")
        dim_s4 = channels[0]
        
        # Setup LoRA or Freeze
        if use_lora and PEFT_AVAILABLE:
            print(f"Applying LoRA to ConvNeXt backbone (r={lora_r}, alpha={lora_alpha})...")
            # Target mlp layers in ConvNeXt blocks (fc1, fc2)
            # We exclude conv_dw because LoRA on grouped convs has constraints (rank must divide groups, etc)
            target_modules = ["mlp.fc1", "mlp.fc2"]
            config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type=None
            )
            # PEFT wraps the backbone. Note: backbone is now a PeftModel.
            self.backbone = get_peft_model(self.backbone, config)
            self.backbone.print_trainable_parameters()
        else:
            print(f"Freezing backbone layers but unfreezing last {unfreeze_n_blocks} blocks...")
            # Freeze all
            for param in self.backbone.parameters():
                param.requires_grad = False
                
            # Unfreeze simple logic
            # ConvNeXt is stages [0,1,2,3]. self.backbone created with out_indices=[3] might just be the full model up to stage 4?
            # timm features_only=True returns a FeatureGraphNet usually.
            # Let's see if we can access the underlying blocks.
            # Usually: backbone.stages[3].blocks...
            
            # Since self.backbone is a FeatureListNet/GraphNet, accessing internals might be tricky or different.
            # We can rely on named_parameters loops.
            # ConvNeXt parameters look like: stages.3.blocks.2.mlp.fc1.weight
            
            if unfreeze_n_blocks > 0:
                # We target stage 3 blocks (0-indexed stages: 0,1,2,3)
                # The convnext_tiny has depths=[3, 3, 9, 3] usually.
                # So stage 3 has 3 blocks.
                # We want to unfreeze the last N blocks of the LAST stage.
                # Or last N blocks of the hierarchy.
                
                # Check stages
                # Assuming standard ConvNeXt structure wrapped or not
                # Let's try to unfreeze based on names
                
                # Count total blocks?
                # Simplify: just unfreeze stage 3 blocks.
                # Actually, `features_only` might not have `stages` attribute directly accessible if it's a wrapper.
                # But let's assuming we can iterate named_parameters.
                
                # For safety, let's just use named_parameters and look for "stages.3.blocks"
                # If unfreeze_n_blocks=2, we want stages.3.blocks.2 and stages.3.blocks.1 (if 3 blocks total)
                
                # Or simplier, just unfreeze the whole last stage?
                # User asked for "last 2 layers" (blocks).
                # Let's try to find them.
                
                params_to_unfreeze = []
                for name, param in self.backbone.named_parameters():
                    # Example: stages.3.blocks.2.gamma
                    if "stages.3.blocks" in name:
                        # Parse block index
                        # logical check
                        parts = name.split('.')
                        # find where 'blocks' is
                        try:
                            b_idx = parts.index('blocks')
                            block_num = int(parts[b_idx+1])
                            # Hardcoded knowledge: convnext tiny has 3 blocks in stage 3 (the 4th stage)
                            if block_num >= (3 - unfreeze_n_blocks): 
                                param.requires_grad = True
                        except:
                            pass
                    # Also unfreeze stage 3 norm
                    if "stages.3.downsample" in name or "norm" in name:
                         # Maybe unfreeze final norms?
                         # Let's just unfreeze blocks for now as requested.
                         pass
        
        # 2. Adapters (Conv2d Projections)
        # We project to proj_channels = dim_s4 (768) if possible, or keep it 768.
        # User requested: "let 768 do not project lower", so we set proj_channels to dim_s4.
        
        proj_channels = dim_s4 # 768
        self.proj_s4 = nn.Conv2d(dim_s4, proj_channels, kernel_size=3, padding=1, bias=False)
        
        # 3. Aggregator (BoQ)
        # Determine proj_channels and row_dim based on mlp_output_dim
        # dim_s4 is 768 for tiny.
        proj_channels = dim_s4
        row_dim = mlp_output_dim // proj_channels
        
        if mlp_output_dim % proj_channels != 0 or row_dim < 1:
            # If requested dim is not a multiple, force fit.
            # If mlp_output_dim=768 and proj_channels=768, row_dim=1. Correct.
            pass
        if row_dim < 1:
             row_dim = 1
             
        # Optional: if we want to strictly match user request, we might need to adjust logic,
        # but 768 output with 768 channels works perfectly with row_dim=1.
        
        print(f"BoQ Configuration: In={dim_s4}, Proj={proj_channels}, RowDim={row_dim}, Out={proj_channels*row_dim}")
        
        self.aggregator = BoQSequence(
            in_channels=proj_channels, 
            num_queries=num_queries,
            num_layers=2, 
            row_dim=row_dim
        )
        row_dim = mlp_output_dim // proj_channels
        if row_dim < 1:
             # If target 512 < 768, we HAVE to project down or BoQ output will be too large?
             # Or we set row_dim to 1 and accept larger output?
             # But the user said "let 768 do not project lower" in BoQ input.
             # The final dimension is (proj_channels * row_dim).
             # If proj_channels=768 and row_dim=1, output=768.
             # Training script expects mlp_output_dim (default 512).
             # We should probably respect the input channel size but maybe project the FINAL output?
             # BoQ Sequence class ends with `self.fc = nn.Linear(..., row_dim)`.
             # output is [B, proj_channels * row_dim].
             # If we want 512 output from 768 channels, we'd need fractional row_dim which is impossible.
             
             # Case: keeping 768 channels in BoQ transformer.
             # If we want exact mlp_output_dim (512) at the end, we might need a separate linear layer 
             # AFTER BoQ or just accept 768 dimension if the training script handles mismatch.
             # However, let's stick to valid integers.
             
             print(f"Warning: mlp_output_dim ({mlp_output_dim}) < FeatureDim ({dim_s4}). Setting row_dim=1, Output will be {dim_s4}.")
             row_dim = 1
        
        print(f"BoQ Configuration: Proj={proj_channels}, RowDim={row_dim}, Out={proj_channels*row_dim}")
        
        self.aggregator = BoQSequence(
            in_channels=proj_channels, 
            num_queries=num_queries, 
            num_layers=2,
            row_dim=row_dim
        )
        
        # InfoNCE temperature
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        if not share_weights:
            print("Initializing second backbone for asymmetric retrieval...")
            self.backbone2 = timm.create_model(
                model_name, 
                pretrained=pretrained, 
                features_only=True, 
                out_indices=[3]
            )
            if use_lora and PEFT_AVAILABLE:
                config2 = LoraConfig(
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    target_modules=["mlp.fc1", "mlp.fc2"],
                    lora_dropout=lora_dropout,
                    bias="none",
                    task_type=None
                )
                self.backbone2 = get_peft_model(self.backbone2, config2)
            
            # Duplicated projections for second backbone
            self.proj_s4_2 = nn.Conv2d(dim_s4, proj_channels, kernel_size=3, padding=1, bias=False)

    def forward_one(self, x, backbone, proj4):
        # Extract features
        features = backbone(x)
        # Assuming single output since out_indices=[3]
        f4 = features[0]
        
        # Project
        f4 = proj4(f4) # [B, 768, H4, W4]
        
        # Flatten and Permute to [B, N, C]
        tokens = f4.flatten(2).transpose(1, 2)
        
        # Aggregate
        desc, attns = self.aggregator(tokens)
        
        # Normalize
        desc = F.normalize(desc, p=2, dim=1)
        
        return desc

    def forward(self, img1=None, img2=None):
        if img1 is not None and img2 is None:
            # Single pass query/img1
            return self.forward_one(img1, self.backbone, self.proj_s4)
        
        if img1 is None and img2 is not None:
             # Single pass gallery/img2
            if self.share_weights:
                return self.forward_one(img2, self.backbone, self.proj_s4)
            else:
                return self.forward_one(img2, self.backbone2, self.proj_s4_2)
                
        # Siamese pass
        desc1 = self.forward_one(img1, self.backbone, self.proj_s4)
        
        if self.share_weights:
            desc2 = self.forward_one(img2, self.backbone, self.proj_s4)
        else:
            desc2 = self.forward_one(img2, self.backbone2, self.proj_s4_2)
            
        return desc1, desc2
    
    def get_config(self):
        if self.use_lora:
            return timm.data.resolve_model_data_config(self.backbone.base_model.model)
        return timm.data.resolve_model_data_config(self.backbone)

    def set_grad_checkpointing(self, enable=True):
        if self.use_lora:
             self.backbone.base_model.model.set_grad_checkpointing(enable)
        else:
            self.backbone.set_grad_checkpointing(enable)
            
        if not self.share_weights and hasattr(self, 'backbone2'):
            if self.use_lora:
                self.backbone2.base_model.model.set_grad_checkpointing(enable)
            else:
                self.backbone2.set_grad_checkpointing(enable)

    def freeze_layers(self, frozen_stages=[0,0,0,0]):
        if self.use_lora:
            print("LoRA enabled: Skipping explicit backbone freezing (PEFT handles base model freezing automatically).")
            return
        # Implement explicit freezing if needed for non-LoRA
        # For now, consistent with DINOv3 logic which might just freeze everything if instructed or handle stages.
        # But here we just leave it or implement similarly.
        pass