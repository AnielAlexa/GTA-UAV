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


class IntermediateLayerExtractor:
    """Extracts features from intermediate layers using forward hooks (AnyLoc-style).

    Research has shown that intermediate ViT layers (especially layer 10-11 in a 12-layer model)
    provide more stable and discriminative features for visual place recognition compared to
    final layer features. Additionally, extracting specific attention facets (key, query, or value)
    can improve performance.
    """

    def __init__(self, model, layer_idx=10, facet='value', use_lora=False):
        """
        Args:
            model: The ViT backbone (or PEFT-wrapped model)
            layer_idx: Which block to extract from (0-indexed, default 10 for 12-layer ViT)
            facet: Which attention facet to use:
                   - 'token': use block output tokens (standard)
                   - 'key'/'query'/'value': extract from attention QKV projection
            use_lora: Whether the model is wrapped with PEFT/LoRA
        """
        self.features = None
        self.facet = facet
        self.layer_idx = layer_idx
        self._hooks = []

        # Get the actual model (handle PEFT wrapper)
        if use_lora and hasattr(model, 'base_model'):
            blocks = model.base_model.model.blocks
        else:
            blocks = model.blocks

        num_blocks = len(blocks)
        if layer_idx >= num_blocks:
            raise ValueError(f"layer_idx={layer_idx} is out of range for model with {num_blocks} blocks (0-indexed)")

        # Register hook on the specified block
        if facet in ['key', 'query', 'value']:
            # Hook the attention QKV projection to extract facets
            target = blocks[layer_idx].attn.qkv
            self._hooks.append(target.register_forward_hook(self._qkv_hook))
            print(f"Registered QKV hook on block {layer_idx} for '{facet}' facet extraction")
        else:
            # Hook the entire block output (token facet)
            target = blocks[layer_idx]
            self._hooks.append(target.register_forward_hook(self._block_hook))
            print(f"Registered block hook on block {layer_idx} for 'token' facet extraction")

    def _block_hook(self, module, input, output):
        """Captures the output of a transformer block."""
        self.features = output

    def _qkv_hook(self, module, input, output):
        """Captures QKV projection output and extracts the requested facet."""
        # QKV output shape: [B, N, 3*D] -> split into Q, K, V
        B, N, _ = output.shape
        D = output.shape[-1] // 3
        qkv = output.reshape(B, N, 3, D)

        facet_idx = {'query': 0, 'key': 1, 'value': 2}[self.facet]
        self.features = qkv[:, :, facet_idx, :]  # [B, N, D]

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    def __del__(self):
        """Cleanup hooks when the extractor is destroyed."""
        self.remove_hooks()

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


class BoQ(nn.Module):
    def __init__(self, in_channels=384, proj_channels=512, num_queries=32, num_layers=2, row_dim=512):
        super().__init__()
        # Input projection: Transformer features are already (B, N, D).
        # We use a Linear layer to project D -> proj_channels
        self.proj_l = nn.Linear(in_channels, proj_channels)
        self.norm_input = nn.LayerNorm(proj_channels)
        
        self.num_layers = num_layers
        self.boqs = nn.ModuleList([
            BoQBlock(proj_channels, num_queries, nheads=proj_channels//64) for _ in range(num_layers)])
        
        # Final compression: (num_layers * num_queries) -> row_dim
        # The repo uses a Linear layer on the Transposed output to reduce the 'sequence' length of the result?
        # Let's check the repo logic again:
        # out = torch.cat(outs, dim=1)  -> (B, layers*queries, D)
        # out = self.fc(out.permute(0, 2, 1)) -> (B, D, layers*queries) @ (layers*queries, row_dim) -> (B, D, row_dim)? 
        # Wait, nn.Linear applies to the LAST dimension.
        # Repo: self.fc = nn.Linear(num_layers*num_queries, row_dim)
        # Repo: out = self.fc(out.permute(0, 2, 1)) 
        # Input to FC: (B, D, layers*queries). 
        # Output of FC: (B, D, row_dim).
        # Repo: out = out.flatten(1) -> (B, D * row_dim) ? 
        # Let's re-read the repo carefully.
        # self.fc = torch.nn.Linear(num_layers*num_queries, row_dim)
        # out = torch.cat(outs, dim=1) # [B, N*Q, D]
        # out = self.fc(out.permute(0, 2, 1)) # [B, D, N*Q] -> [B, D, row_dim]
        # out = out.flatten(1) # [B, D*row_dim]
        # output is huge? The repo default row_dim=32. Reference D=512. Result = 512*32 = 16384. This seems large for a descriptor.
        # Let's check defaults. 
        # Ah, row_dim=32. Yes. 
        # If we want output 512, and D is 512. We can set row_dim=1. Then flatten is 512.
        # OR we set row_dim such that D*row_dim is our target.
        
        self.fc = nn.Linear(num_layers * num_queries, row_dim)
        self.row_dim = row_dim

    def forward(self, x):
        # x: [B, N, in_channels] (DINOv3 patches)
        x = self.proj_l(x) # [B, N, proj_channels]
        x = self.norm_input(x)
        
        outs = []
        attns = []
        for i in range(len(self.boqs)):
            x, out, attn = self.boqs[i](x)
            outs.append(out)
            attns.append(attn)

        # Concatenate outputs from all layers
        out = torch.cat(outs, dim=1) # [B, num_layers*num_queries, proj_channels]
        
        # Compress the 'number of queries' dimension
        # out.permute(0, 2, 1) -> [B, proj_channels, num_layers*num_queries]
        out = self.fc(out.permute(0, 2, 1)) # -> [B, proj_channels, row_dim]
        
        return out.flatten(1), attns # [B, proj_channels * row_dim]


class DesModelDINOv3BoQ(nn.Module):
    """
    Model Architecture:
    DINOv3 (Frozen/LoRA) -> BoQ Aggregator (Cascaded) -> Descriptor
    """
    def __init__(self,
                 model_name='vit_small_patch16_dinov3.lvd1689m',
                 pretrained=True,
                 img_size=384,
                 share_weights=True,
                 num_queries=8,
                 boq_nheads=8,
                 gem_p=3.0,
                 mlp_hidden_dim=1024,
                 mlp_output_dim=512,
                 unfreeze_n_blocks=2,
                 use_lora=False,
                 lora_r=16,
                 lora_alpha=16,
                 lora_dropout=0.1,
                 use_intermediate_layer=False,
                 intermediate_layer_idx=10,
                 intermediate_facet='value'):
        super(DesModelDINOv3BoQ, self).__init__()
        self.share_weights = share_weights
        self.img_size = img_size
        self.use_lora = use_lora
        self.use_intermediate_layer = use_intermediate_layer
        self.intermediate_layer_idx = intermediate_layer_idx
        self.intermediate_facet = intermediate_facet
        
        # 1. Backbone
        print(f"Initializing DINOv3 BoQ Model: {model_name}")
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, img_size=img_size)
        
        # Setup LoRA or Freeze
        if use_lora and PEFT_AVAILABLE:
            print(f"Applying LoRA to backbone (r={lora_r}, alpha={lora_alpha})...")
            # Apply LoRA to qkv and proj layers in Attention, and fc1/fc2 in MLP
            target_modules = ["qkv", "proj", "fc1", "fc2"]
            config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type=None
            )
            self.backbone = get_peft_model(self.backbone, config)
            self.backbone.print_trainable_parameters()
        else:
            # Check for timm ViT structure
            if hasattr(self.backbone, 'blocks'):
                print(f"Freezing backbone but unfreezing last {unfreeze_n_blocks} blocks...")
                # Freeze everything first
                for param in self.backbone.parameters():
                    param.requires_grad = False
                
                # Unfreeze last N blocks
                if unfreeze_n_blocks > 0:
                    for i in range(-unfreeze_n_blocks, 0):
                        for param in self.backbone.blocks[i].parameters():
                            param.requires_grad = True
                    
                    # Also unfreeze norm if present
                    if hasattr(self.backbone, 'norm'):
                         for param in self.backbone.norm.parameters():
                            param.requires_grad = True
            else:
                print("Backbone structure unknown (no .blocks), Freezing all backbone layers...")
                for param in self.backbone.parameters():
                    param.requires_grad = False
        
        # Get feature dimensions
        # Create a dummy input to get output dim (must match configured img_size)
        dummy = torch.randn(1, 3, img_size, img_size)
        if use_lora:
            # PEFT model wrapper specific
            # We need to run a forward pass through the base model or use config
            # But let's just use the num_features from the underlying timm model if possible,
            # or infer from a forward pass (safer)
            with torch.no_grad():
                # timm vit forward_features returns (B, N, D)
                # Note: PEFT model forward might differ, so we access .base_model.model
                features = self.backbone.base_model.model.forward_features(dummy)
                embed_dim = features.shape[-1]
                self.num_prefix_tokens = self.backbone.base_model.model.num_prefix_tokens
        else:
            with torch.no_grad():
                features = self.backbone.forward_features(dummy)
                embed_dim = features.shape[-1]
                self.num_prefix_tokens = self.backbone.num_prefix_tokens

        print(f"Backbone feature dimension: {embed_dim}")

        # Initialize intermediate layer extractor if enabled (AnyLoc-style)
        self.intermediate_extractor = None
        if use_intermediate_layer:
            print(f"Enabling intermediate layer extraction: layer={intermediate_layer_idx}, facet={intermediate_facet}")
            self.intermediate_extractor = IntermediateLayerExtractor(
                self.backbone,
                layer_idx=intermediate_layer_idx,
                facet=intermediate_facet,
                use_lora=use_lora
            )

        # 2. Aggregator (BoQ)
        # Determine proj_channels and row_dim based on mlp_output_dim
        proj_channels = 512
        row_dim = mlp_output_dim // proj_channels
        
        if mlp_output_dim % proj_channels != 0 or row_dim < 1:
            # If requested dim is not a multiple of 512 (e.g. 768), 
            # set proj_channels to the target dim and row_dim to 1.
            proj_channels = mlp_output_dim
            row_dim = 1
            
        print(f"BoQ Configuration: In={embed_dim}, Proj={proj_channels}, RowDim={row_dim}, Out={proj_channels*row_dim}")
        
        self.aggregator = BoQ(
            in_channels=embed_dim, 
            proj_channels=proj_channels, 
            num_queries=num_queries, 
            num_layers=2, # As per repo default
            row_dim=row_dim
        )
        
        # InfoNCE temperature (learned)
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.intermediate_extractor2 = None
        if not share_weights:
            print("Initializing second backbone for asymmetric retrieval...")
            self.backbone2 = timm.create_model(model_name, pretrained=pretrained, num_classes=0, img_size=img_size)
            if use_lora and PEFT_AVAILABLE:
                config2 = LoraConfig(
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    target_modules=["qkv", "proj", "fc1", "fc2"],
                    lora_dropout=lora_dropout,
                    bias="none",
                    task_type=None
                )
                self.backbone2 = get_peft_model(self.backbone2, config2)

            # Initialize intermediate layer extractor for second backbone if enabled
            if use_intermediate_layer:
                self.intermediate_extractor2 = IntermediateLayerExtractor(
                    self.backbone2,
                    layer_idx=intermediate_layer_idx,
                    facet=intermediate_facet,
                    use_lora=use_lora
                )
        
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
        # DINOv3 (ViT) doesn't have "stages" in the same way as ConvNets.
        # But we can freeze the entire backbone if requested.
        # Since this method is only called if config.freeze_layers is True,
        # we will freeze the backbone regardless of the frozen_stages values.
        
        if self.use_lora:
            print("LoRA enabled: Skipping explicit backbone freezing (PEFT handles base model freezing automatically).")
            # We must NOT loop through parameters setting requires_grad=False because that would freeze the LoRA adapters too!
            return

        print("Freezing Backbone parameters...")
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        if not self.share_weights and hasattr(self, 'backbone2'):
            for param in self.backbone2.parameters():
                param.requires_grad = False

    def _forward_single(self, x, backbone, extractor=None):
        # 1. Extract Features
        # Output: [B, N, D]
        if self.use_lora:
            all_tokens = backbone.base_model.model.forward_features(x)
        else:
            all_tokens = backbone.forward_features(x)

        # If intermediate layer extraction is enabled, use the captured features
        if self.use_intermediate_layer and extractor is not None:
            # The forward pass above triggered the hooks, now get captured features
            all_tokens = extractor.features

        # 2. Extract Patches (ignoring CLS for BoQ)
        num_prefix = self.num_prefix_tokens
        patch_tokens = all_tokens[:, num_prefix:, :]  # [B, N-prefix, D]

        # 3. Aggregation (BoQ - Cascaded)
        # Returns: descriptor [B, OutDim], attentions
        descriptor, _ = self.aggregator(patch_tokens)

        # Normalization (L2) - Essential for retrieval
        descriptor = F.normalize(descriptor, p=2, dim=1)

        return descriptor

    def forward(self, img1=None, img2=None):
        if self.share_weights:
            if img1 is not None and img2 is not None:
                emb1 = self._forward_single(img1, self.backbone, self.intermediate_extractor)
                emb2 = self._forward_single(img2, self.backbone, self.intermediate_extractor)
                return emb1, emb2
            elif img1 is not None:
                return self._forward_single(img1, self.backbone, self.intermediate_extractor)
            else:
                return self._forward_single(img2, self.backbone, self.intermediate_extractor)
        else:
            if img1 is not None and img2 is not None:
                emb1 = self._forward_single(img1, self.backbone, self.intermediate_extractor)
                emb2 = self._forward_single(img2, self.backbone2, self.intermediate_extractor2)
                return emb1, emb2
            elif img1 is not None:
                return self._forward_single(img1, self.backbone, self.intermediate_extractor)
            else:
                # Typically gallery is second
                return self._forward_single(img2, self.backbone2, self.intermediate_extractor2)
