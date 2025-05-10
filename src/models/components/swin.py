import json
import torch
import torch.nn as nn
import open_clip
from huggingface_hub import hf_hub_download
from open_clip.transformer import VisionTransformer
import torch.nn.functional as F
from transformers import AutoImageProcessor, SwinModel


class Expert(nn.Module):
    def __init__(self, hidden_dims=[96, 192, 384, 768], output_dim=768):
        super().__init__()
        self.output_dim = output_dim
        self.num_scales = len(hidden_dims)

        # Project each scale to the common output dimension
        self.proj_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(dim, output_dim, kernel_size=1),
                nn.ReLU()
            ) for dim in hidden_dims
        ])

        # Attention mechanism over scales (output_dim → scalar per scale)
        self.attn_proj = nn.Sequential(
            nn.Linear(output_dim, output_dim // 2),
            nn.ReLU(),
            nn.Linear(output_dim // 2, 1)
        )

    def forward(self, multi_scale_feats):
        """
        multi_scale_feats: list of Tensors with shape [B, P_i, D_i] per scale
        Returns: fused feature tensor [B, P, output_dim]
        """
        upsampled = []
        max_len = max(f.shape[1] for f in multi_scale_feats)  # target patch count

        for f, conv in zip(multi_scale_feats, self.proj_convs):
            f = conv(f.transpose(1, 2))  # B, D, P_i
            f = F.interpolate(f, size=max_len, mode='linear', align_corners=False)
            upsampled.append(f)  # B, D, max_len

        # for f in upsampled:
        #     print(f.shape)


        # Stack: [num_scales, B, D, P] → [B, D, P, num_scales]
        fused_tensor = torch.stack(upsampled, dim=0)

        # print(fused_tensor.shape)

        fused_for_attn = fused_tensor.permute(1,3,0,2)

        # print(fused_for_attn.shape)

        B, P, S, D = fused_for_attn.shape
        attn_input = fused_for_attn.reshape(B * P * S, D)

        # print(attn_input.shape)

        attn_logits = self.attn_proj(attn_input)
        attn_logits = attn_logits.view(B, P, S)

        # print(attn_logits.shape)

        attn_weights = F.softmax(attn_logits, dim=-1)  # [B, P, S]

        # print(fused_for_attn.shape, attn_weights.shape)

        # # Compute attention over scales: [B, D, P, num_scales] → [B, P, num_scales]
        # attn_weights = self.attn_proj(fused_tensor.permute(0, 2, 3, 1))  # → [B, P, D, num_scales]
        
        # attn_weights = F.softmax(attn_weights, dim=-1)

        # Apply attention: [B, D, P, num_scales] × [B, 1, P, num_scales]
        weighted = fused_for_attn.permute(0,3,1,2) * attn_weights.unsqueeze(1)  # B, D, P, S
        fused = weighted.sum(dim=-1)  # [B, D, P]
        return fused.transpose(1, 2)  # [B, P, D]

class MoE(nn.Module):
    def __init__(self, num_experts=6, hidden_dims=[96, 192, 384, 768], output_dim=768, router_input_dim=768):
        super().__init__()
        self.experts = nn.ModuleList([
            Expert(hidden_dims, output_dim) for _ in range(num_experts)
        ])
        self.router = nn.Sequential(
            nn.Linear(router_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_experts)
        )
    
    def forward(self, multi_scale_feats, global_feat):
        """
        multi_scale_feats: list of scale features from Swin
        """
        router_logits = self.router(global_feat)  # [B, K]
        router_logits = torch.softmax(router_logits, dim=-1)    # [B, K]

        # print(router_logits.shape)

        # Each expert returns [B, P, D]
        expert_outputs = [expert(multi_scale_feats) for expert in self.experts]
        expert_outputs = torch.stack(expert_outputs, dim=1)  # [B, K, P, D]

        gates = router_logits.unsqueeze(-1).unsqueeze(-1)  # [B, K, 1, 1]
        fused_output = (gates * expert_outputs).sum(dim=1)  # [B, P, D]
        fused_output = fused_output.transpose(1, 2)  # [B, 256, D]
        fused_output = F.adaptive_avg_pool1d(fused_output, output_size=256)  # [B, D, 256]
        fused_output = fused_output.transpose(1, 2)  # [B, 256, D]
        B, P, D = fused_output.shape
        H = W = int(P ** 0.5)
        fused_output = fused_output.transpose(1, 2).reshape(B, D, H, W)  # [B, D, 56, 56]

        return fused_output, router_logits

class SWIN(nn.Module):
    def __init__(self, pretrained=True, lora=False, lora_r=8, lora_alpha=16, lora_dropout=0.1, use_moe=True):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.moe = MoE() if use_moe else None
        model_id = 'microsoft/swin-tiny-patch4-window7-224'

        self.model = SwinModel.from_pretrained(model_id)
        self.preprocessor = AutoImageProcessor.from_pretrained(model_id)


    def forward(self, x):
        inputs = self.preprocessor(x, return_tensors="pt").to(self.device)
        # x = torch.stack([self.preprocessor(img) for img in x]).to(self.device)
        out = self.model(**inputs, output_hidden_states=True) 

        final_hidden = out.last_hidden_state
        global_feat = final_hidden.mean(dim=1) 

        stage_feats = [out.hidden_states[i] for i in range(4)]

        if self.moe:
            local_feat, router_logits = self.moe(stage_feats, global_feat)
        else:
            local_feat = out.last_hidden_state
            router_logits = None


        return global_feat, local_feat, router_logits

