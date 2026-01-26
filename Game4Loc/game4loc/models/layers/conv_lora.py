import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvLoRA(nn.Module):
    def __init__(self, linear_layer, r=8, alpha=16, dropout=0.1, num_experts=3, num_prefix_tokens=1):
        super().__init__()
        # Store original linear layer
        self.linear = linear_layer
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.num_prefix_tokens = num_prefix_tokens
        
        # Freezing the original linear layer
        for param in self.linear.parameters():
            param.requires_grad = False
        
        # LoRA A (Encoder): Linear projection to low rank
        self.lora_A = nn.Linear(self.in_features, r, bias=False)
        
        # LoRA B (Decoder): Linear projection back to high rank
        self.lora_B = nn.Linear(r, self.out_features, bias=False)
        
        # Dropout
        self.dropout = nn.Dropout(p=dropout)
        
        # Experts (Convolutional)
        # Experts with different kernel sizes to capture different scales.
        # e.g., 3x3, 5x5, 7x7. Padding to keep same size.
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(r, r, kernel_size=3 + 2*i, padding=1 + i, groups=r, bias=False), # Depthwise
                nn.ReLU()
            ) for i in range(num_experts)
        ])
        
        # Gating network (Linear on channel dim)
        self.gate = nn.Linear(r, num_experts)
        
        # Initialize
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        
    def forward(self, x):
        # x: [B, N, D]
        
        # Standard Linear Pass (Frozen)
        original_out = self.linear(x)
        
        # LoRA Path
        # 1. Encode
        x_dropped = self.dropout(x)
        x_lora = self.lora_A(x_dropped) # [B, N, r]
        
        # 2. Experts (Spatial)
        # We process only spatial tokens through convolutional experts
        if x_lora.dim() == 3:
            B, N, R = x_lora.shape
            
            # Simple split based on prefix count
            prefix_tokens = x_lora[:, :self.num_prefix_tokens, :]
            spatial_tokens = x_lora[:, self.num_prefix_tokens:, :] 
            
            # Reshape spatial tokens to grid [B, R, H, W]
            num_spatial = N - self.num_prefix_tokens
            H = int(math.sqrt(num_spatial))
            W = H # Assuming square grid
            
            if H * W == num_spatial:
                spatial = spatial_tokens.transpose(1, 2).reshape(B, R, H, W)
                
                # Apply Experts (Parallel Convolutions)
                expert_outputs = [expert(spatial) for expert in self.experts]
                stacked_experts = torch.stack(expert_outputs, dim=1) # [B, NumExperts, R, H, W]
                
                # Apply Gate (Pixel-wise selection)
                gate_logits = self.gate(spatial_tokens) # [B, HW, NumExperts]
                gate_weights = F.softmax(gate_logits, dim=-1)
                gate_weights = gate_weights.transpose(1, 2).reshape(B, self.num_experts, 1, H, W)
                
                # Weighted Sum
                mixed = (stacked_experts * gate_weights).sum(dim=1) # [B, R, H, W]
                
                # Flatten back
                mixed_flat = mixed.flatten(2).transpose(1, 2) # [B, HW, R]
                
                # Re-concatenate with prefix (which skipped experts)
                x_mixed = torch.cat([prefix_tokens, mixed_flat], dim=1)
            else:
                # Fallback if shape mismatch (should not happen in standard training)
                x_mixed = x_lora
        else:
            x_mixed = x_lora

        # 3. Decode
        delta_w = self.lora_B(x_mixed)
        
        return original_out + delta_w * self.scaling

def inject_conv_lora(model, r=8, alpha=16, dropout=0.1, num_prefix_tokens=1):
    target_suffixes = ["qkv", "proj", "fc1", "fc2"]
    
    # We collect targets first to avoid modifying dictionary while iterating
    targets = []
    for name, module in model.named_modules():
        if any(name.endswith(suffix) for suffix in target_suffixes):
            if isinstance(module, nn.Linear):
                targets.append((name, module))
                
    for name, module in targets:
        path = name.split('.')
        parent = model
        for p in path[:-1]:
            parent = getattr(parent, p)
        layer_name = path[-1]
        
        # print(f"Injecting ConvLoRA into {name}")
        lora_layer = ConvLoRA(module, r=r, alpha=alpha, dropout=dropout, num_prefix_tokens=num_prefix_tokens)
        setattr(parent, layer_name, lora_layer)
        
    return model

def mark_only_lora_as_trainable(model):
    print("Freezing non-LoRA parameters...")
    for n, p in model.named_parameters():
        if 'lora_' in n or 'gate' in n or 'experts' in n:
            p.requires_grad = True
        else:
            p.requires_grad = False
