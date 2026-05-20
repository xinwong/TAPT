from collections import OrderedDict
from typing import Tuple, Union
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        return x[0]


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


def _inverse_softplus(value: float) -> float:
    value = max(float(value), 1e-6)
    return math.log(math.expm1(value))


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class ResidualAttentionBlock_IVLP(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, add_prompt=False,
                 text_layer=False, i=0, design_details=None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        # Only add learnable tokens if flag is set True
        # For the first iteration i, we should not add the learnable parameters
        # as it is already been taken care of in the very start, for both text
        # and the visual branch
        self.text_layer = text_layer
        self.attn_mask = attn_mask
        if i != 0:
            self.add_prompt = add_prompt
            if self.add_prompt:
                if self.text_layer:
                    self.n_ctx_text = design_details["language_ctx"]  # hyperparameter
                    ctx_vectors = torch.empty(self.n_ctx_text, d_model)
                else:
                    self.n_ctx_visual = design_details["vision_ctx"]  # hyperparameter
                    ctx_vectors = torch.empty(self.n_ctx_visual, d_model)
                # Code snippet for per layer visual prompts
                nn.init.normal_(ctx_vectors, std=0.02)
                self.VPT_shallow = nn.Parameter(ctx_vectors)
        else:
            self.add_prompt = False

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        # Will need to append the learnable tokens for this layer here
        # Check if flag was set for this layer or not
        if self.add_prompt:
            # Also see if this is textual transformer layer or not
            if not self.text_layer:
                # Remove the outputs produced by learnable tokens of previous layer
                prefix = x[0:x.shape[0] - self.n_ctx_visual, :, :]
                # Create/configure learnable tokens of this layer
                visual_context = self.VPT_shallow.expand(x.shape[1], -1, -1).permute(1, 0, 2) #.half()
                # Add the learnable tokens of this layer with the input, by replacing the previous
                # layer learnable tokens
                x = torch.cat([prefix, visual_context], dim=0)
            else:
                # Appending the learnable tokens in different way
                # x -> [77, NCLS, DIM]
                # First remove the learnable tokens from previous layer
                prefix = x[:1, :, :]
                suffix = x[1 + self.n_ctx_text:, :, :]
                # Create/configure learnable tokens of this layer
                textual_context = self.VPT_shallow.expand(x.shape[1], -1, -1).permute(1, 0, 2) #.half()
                # Add the learnable tokens of this layer with the input, replaced by previous
                # layer learnable tokens
                x = torch.cat([prefix, textual_context, suffix], dim=0)

        # # Store feature in forward pass for alignment loss
        if not self.text_layer:
            self.visual_feat = x

        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class ResidualAttentionBlock_IVLP_MoE(nn.Module):
    """
    A Residual Attention Block modified to incorporate Mixture-of-Experts (MoE)
    for multiple learnable deep prompts (3 experts in this case).
    """
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, add_prompt=False,
                 text_layer=False, i=0, design_details=None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        # Only add learnable tokens if flag is set True
        # For the first iteration i, we should not add the learnable parameters
        # as it is already been taken care of in the very start, for both text
        # and the visual branch
        self.text_layer = text_layer
        self.attn_mask = attn_mask

        # --- MoE Prompt Specific Initialization ---
        self.num_experts = design_details["num_experts"]  
        self.n_ctx_text = 0
        self.n_ctx_visual = 0

        if i != 0:
            self.add_prompt = add_prompt
            if self.add_prompt:
                if self.text_layer:
                    self.n_ctx_text = design_details["language_ctx"]  # hyperparameter
                    expert_prompts_tensor = torch.empty(self.num_experts, self.n_ctx_text, d_model)
                else:
                    self.n_ctx_visual = design_details["vision_ctx"]  # hyperparameter
                    expert_prompts_tensor = torch.empty(self.num_experts, self.n_ctx_visual, d_model)
                # Code snippet for per layer visual prompts
                nn.init.normal_(expert_prompts_tensor, std=0.02)
                self.expert_prompts = nn.Parameter(expert_prompts_tensor) # 形状: (E, C, D)
                self.gate = nn.Linear(d_model, self.num_experts)
        else:
            # 此层不添加提示（i=0或add_prompt=False）
            self.add_prompt = False # 如果i == 0，确保它为false

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        # Will need to append the learnable tokens for this layer here
        # Check if flag was set for this layer or not
        orig_dtype = x.dtype  # 保存原始数据类型
        
        if self.add_prompt:
            # Also see if this is textual transformer layer or not
            if not self.text_layer:
                # 视觉层的MoE提示处理
                # prefix 维度: [L, B, D]  (不含上层 prompt)
                prefix = x[0:x.shape[0] - self.n_ctx_visual, :, :]
                
                ## Token-wise logits: 每个 token 分别计算专家 logits，再沿 token 维求平均
                gate_input = prefix.permute(1, 0, 2)                                 # [B, L, D]
                gate_logits = self.gate(gate_input)                                  # [B, L, num_experts]
                gate_weights = F.softmax(gate_logits, dim=-1)                        # [B, L, num_experts]
                gate_weights = gate_weights.mean(dim=1)                              # [B, num_experts]
                
                # expert_prompts: [E, C, D] → 扩展 batch 维度
                expert_prompts = self.expert_prompts.to(orig_dtype)                 # [E, C, D]
                expert_prompts = expert_prompts.unsqueeze(0)                        # [1, E, C, D]
                gate_weights = gate_weights.unsqueeze(-1).unsqueeze(-1)             # [B, E, 1, 1]
                combined_prompts = (gate_weights * expert_prompts).sum(dim=1)       # [B, C, D]
                
                # 重新排列成 [C, B, D] 以便与 prefix 进行 concat
                visual_context = combined_prompts.permute(1, 0, 2)                  # [C, B, D]
                
                # 连接原始输入和新的 prompt
                x = torch.cat([prefix, visual_context], dim=0)
            else:
                # 文本层的 MoE 提示处理
                prefix = x[:1, :, :]                                             # CLS token
                suffix = x[1 + self.n_ctx_text:, :, :]                           # 其余 token
                
                ## Token-wise logits
                all_tokens = torch.cat([prefix, suffix], dim=0)                   # [L, B, D]
                gate_input = all_tokens.permute(1, 0, 2)                          # [B, L, D]
                gate_logits = self.gate(gate_input)                               # [B, L, num_experts]
                gate_weights = F.softmax(gate_logits, dim=-1)                     # [B, L, num_experts]
                gate_weights = gate_weights.mean(dim=1)                           # [B, num_experts]

                expert_prompts = self.expert_prompts.to(orig_dtype)              # [E, C, D]
                expert_prompts = expert_prompts.unsqueeze(0)                     # [1, E, C, D]
                gate_weights = gate_weights.unsqueeze(-1).unsqueeze(-1)          # [B, E, 1, 1]
                combined_prompts = (gate_weights * expert_prompts).sum(dim=1)    # [B, C, D]

                textual_context = combined_prompts.permute(1, 0, 2)              # [C, B, D]
                
                x = torch.cat([prefix, textual_context, suffix], dim=0)

        # # Store feature in forward pass for alignment loss
        if not self.text_layer:
            self.visual_feat = x

        # 正常的Transformer层计算
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class ResidualAttentionBlock_IVLP_MoE_Aware(nn.Module):
    """
    Alignment-Aware Soft MoE for IVLP/VPT.
    Key improvements over IVLP_MoE:
      1. Configurable CLS/token-mean/hybrid gating on normalized features
      2. Anchored prompt = base_prompt + alpha * (mixed_prompt - base_prompt)
      3. Bounded alpha/tau to stabilize optimization
      4. expert_diversity_loss() preventing expert collapse
      5. load_balance_loss() for balanced routing
    """
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, add_prompt=False,
                 text_layer=False, i=0, design_details=None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.text_layer = text_layer
        self.attn_mask = attn_mask

        self.num_experts = design_details["num_experts"]
        self.delta_scale_init = design_details.get("delta_scale_init", 0.1)
        self.gate_mode = design_details.get("gate_mode", "hybrid")
        self.gate_hybrid_lambda = design_details.get("gate_hybrid_lambda", 0.7)
        self.alpha_min = design_details.get("alpha_min", 0.0)
        self.alpha_max = design_details.get("alpha_max", 2.0)
        self.tau_min = design_details.get("tau_min", 0.3)
        self.tau_max = design_details.get("tau_max", 3.0)
        self.n_ctx_text = 0
        self.n_ctx_visual = 0
        self._last_gate_weights = None

        if self.gate_mode not in {"cls", "token_mean", "hybrid"}:
            raise ValueError(f"Unsupported gate mode: {self.gate_mode}")

        if i != 0:
            self.add_prompt = add_prompt
            if self.add_prompt:
                if self.text_layer:
                    self.n_ctx_text = design_details["language_ctx"]
                    expert_prompts_tensor = torch.empty(self.num_experts, self.n_ctx_text, d_model)
                else:
                    self.n_ctx_visual = design_details["vision_ctx"]
                    expert_prompts_tensor = torch.empty(self.num_experts, self.n_ctx_visual, d_model)
                nn.init.normal_(expert_prompts_tensor, std=0.02)
                base_prompt_tensor = expert_prompts_tensor.mean(dim=0).clone()
                self.base_prompt = nn.Parameter(base_prompt_tensor)         # [C, D]
                self.expert_prompts = nn.Parameter(expert_prompts_tensor)   # [E, C, D]
                self.gate = nn.Linear(d_model, self.num_experts)
                tau_init = min(max(1.0, self.tau_min), self.tau_max)
                alpha_init = min(max(self.delta_scale_init, max(self.alpha_min, 1e-6)), self.alpha_max)
                self.tau = nn.Parameter(torch.full((1,), _inverse_softplus(tau_init)))
                self.alpha = nn.Parameter(torch.full((1,), _inverse_softplus(alpha_init)))
        else:
            self.add_prompt = False

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def _zero_loss(self):
        return self.ln_1.weight.new_tensor(0.0)

    def _bounded_scale(self, value: torch.Tensor, min_value: float, max_value: float):
        scale = F.softplus(value.float())
        if min_value is not None:
            scale = scale.clamp(min=min_value)
        if max_value is not None:
            scale = scale.clamp(max=max_value)
        return scale

    def _router_logits(self, gate_source: torch.Tensor):
        tokens = self.ln_1(gate_source)
        cls_logits = self.gate(tokens[0])
        if self.gate_mode == "cls":
            return cls_logits

        token_mean_logits = self.gate(tokens.mean(dim=0))
        if self.gate_mode == "token_mean":
            return token_mean_logits

        lambda_cls = max(0.0, min(1.0, float(self.gate_hybrid_lambda)))
        return lambda_cls * cls_logits + (1.0 - lambda_cls) * token_mean_logits

    def _mix_expert_prompt(self, gate_source: torch.Tensor, orig_dtype: torch.dtype):
        gate_logits = self._router_logits(gate_source)                       # [B, E]
        tau = self._bounded_scale(self.tau, self.tau_min, self.tau_max)
        gate_weights = F.softmax(
            gate_logits.float() / tau, dim=-1)                               # [B, E]
        self._last_gate_weights = gate_weights

        expert_prompts = self.expert_prompts.to(orig_dtype).unsqueeze(0)     # [1, E, C, D]
        weights = gate_weights.to(orig_dtype).unsqueeze(-1).unsqueeze(-1)    # [B, E, 1, 1]
        mixed = (weights * expert_prompts).sum(dim=1)                        # [B, C, D]
        return mixed.permute(1, 0, 2).to(orig_dtype)                         # [C, B, D]

    def expert_diversity_loss(self):
        """Penalise high cosine similarity between expert prompts."""
        if not self.add_prompt:
            return self._zero_loss()
        E = self.expert_prompts.shape[0]
        if E <= 1:
            return self._zero_loss()
        flat = F.normalize(self.expert_prompts.reshape(E, -1), dim=-1)
        sim = flat @ flat.T
        mask = torch.triu(torch.ones(E, E, device=flat.device), diagonal=1).bool()
        return sim[mask].clamp(min=0).mean()

    def load_balance_loss(self):
        """Encourage the batch-average router usage to stay near uniform."""
        if self._last_gate_weights is None or not self.add_prompt:
            return self._zero_loss()
        avg_weights = self._last_gate_weights.mean(dim=0)
        target = torch.full_like(avg_weights, 1.0 / self.num_experts)
        return F.mse_loss(avg_weights, target)

    def moe_aux_losses(self):
        return self.load_balance_loss(), self.expert_diversity_loss()

    def forward(self, x: torch.Tensor):
        orig_dtype = x.dtype
        self._last_gate_weights = None

        if self.add_prompt:
            if not self.text_layer:
                prefix = x[0:x.shape[0] - self.n_ctx_visual, :, :]
                base_prompt = self.base_prompt.to(orig_dtype)
                base_prompt = base_prompt.unsqueeze(0).expand(x.shape[1], -1, -1)
                base_prompt = base_prompt.permute(1, 0, 2)                   # [C, B, D]
                mixed_prompt = self._mix_expert_prompt(prefix, orig_dtype)
                moe_delta = mixed_prompt - base_prompt
                alpha = self._bounded_scale(self.alpha, self.alpha_min, self.alpha_max).to(orig_dtype)
                visual_context = base_prompt + alpha.view(1, 1, 1) * moe_delta
                visual_context = visual_context.to(orig_dtype)

                x = torch.cat([prefix, visual_context], dim=0)
            else:
                prefix = x[:1, :, :]
                suffix = x[1 + self.n_ctx_text:, :, :]
                gate_source = torch.cat([prefix, suffix], dim=0)
                base_prompt = self.base_prompt.to(orig_dtype)
                base_prompt = base_prompt.unsqueeze(0).expand(x.shape[1], -1, -1)
                base_prompt = base_prompt.permute(1, 0, 2)                   # [C, B, D]
                mixed_prompt = self._mix_expert_prompt(gate_source, orig_dtype)
                moe_delta = mixed_prompt - base_prompt
                alpha = self._bounded_scale(self.alpha, self.alpha_min, self.alpha_max).to(orig_dtype)
                textual_context = base_prompt + alpha.view(1, 1, 1) * moe_delta
                textual_context = textual_context.to(orig_dtype)

                x = torch.cat([prefix, textual_context, suffix], dim=0)

        if not self.text_layer:
            self.visual_feat = x

        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class ResidualAttentionBlock_MaPLe(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, design_details=None,
                 text_layer=False, i=0):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        # For the first iteration i, we do not need to add the learnable parameters here
        # as it will be added in the beginning, for both text and the vision branch
        self.text_layer = text_layer
        self.attn_mask = attn_mask
        # This must be consistent with the config file prompt
        self.compound_prompt_nctx = design_details['maple_length']
        if i == 0:
            self.first_layer = True
        else:
            self.first_layer = False
        
        self.visual_feature = torch.empty(0)
        self.text_feature = torch.empty(0)

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, inputs):
        # For the first layer, we do not need to add any duplicate, as it is already added
        # as the shallow version
        x = inputs[0]
        compound_prompts_deeper = inputs[1]
        counter = inputs[2]
        if not self.first_layer:
            if len(compound_prompts_deeper) > 0:
                # This means that deeper compound prompts are turned on
                # Here it behaves differently for text and visual side
                # Forward function is same for both

                if not self.text_layer:
                    # First check if the ith layer needs compound prompts or not
                    if not (counter > len(compound_prompts_deeper) - 1):
                        # Remove the outputs produced by learnable tokens of previous layer
                        prefix = x[0:x.shape[0] - self.compound_prompt_nctx, :, :]
                        # Create/configure learnable tokens of this layer
                        visual_context = compound_prompts_deeper[counter]  # extract the correct index
                        visual_context = visual_context.expand(x.shape[1], -1, -1).permute(1, 0, 2) #.half()
                        # Add the learnable tokens of this layer with the input, by replacing previous
                        # layer learnable tokens
                        x = torch.cat([prefix, visual_context], dim=0)

                        # Once done, update the counter, so that the next time, it does not use same learnable tokens
                        counter += 1
                else:
                    # First check if the ith layer needs compound prompts or not
                    if not (counter > len(compound_prompts_deeper) - 1):
                        # Appending the learnable tokens in different way
                        # x -> [77, NCLS, DIM]
                        # First remove the learnable tokens from previous layer
                        prefix = x[:1, :, :]
                        suffix = x[1 + self.compound_prompt_nctx:, :, :]
                        # Create/configure learnable tokens of this layer
                        textual_context = compound_prompts_deeper[counter]
                        textual_context = textual_context.expand(x.shape[1], -1, -1).permute(1, 0, 2) #.half()
                        # Add the learnable tokens of this layer with the input, replaced by previous
                        # layer learnable tokens
                        x = torch.cat([prefix, textual_context, suffix], dim=0)
                        # Once done, update the counter, so that the next time, it does not use same learnable tokens
                        counter += 1
        
        # # Store feature in forward pass
        if not self.text_layer:
            self.visual_feat = x

        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))

        return [x, compound_prompts_deeper, counter]  # return again as a list, so that nn.seq can work


class ResidualAttentionBlock_MaPLe_MoE(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None,
                 design_details=None, text_layer=False, i: int = 0):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        # For the first iteration i, we do not need to add the learnable parameters here
        # as it will be added in the beginning, for both text and the vision branch
        self.text_layer = text_layer
        self.attn_mask = attn_mask
        # This must be consistent with the config file prompt
        self.compound_prompt_nctx = design_details['maple_length']
        if i == 0:
            self.first_layer = True
        else:
            self.first_layer = False

        # --- MoE Prompt Specific Initialization ---
        self.num_experts = design_details["num_experts"]
        self.n_ctx = design_details["maple_length"]

        expert_prompts = torch.empty(self.num_experts, self.n_ctx, d_model)
        nn.init.normal_(expert_prompts, std=0.02)
        self.expert_prompts = nn.Parameter(expert_prompts)  # [E, C, D]
        
        self.gate = nn.Linear(d_model, self.num_experts)

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def _mix_expert_prompt(self, prefix: torch.Tensor, orig_dtype: torch.dtype):
        # prefix: [L, B, D]
        gate_in = prefix.permute(1, 0, 2)                # [B, L, D]
        logits = self.gate(gate_in)                       # [B, L, E]
        weights = F.softmax(logits, dim=-1).mean(dim=1)   # [B, E]

        expert = self.expert_prompts.to(orig_dtype)       # [E, C, D]
        expert = expert.unsqueeze(0)                      # [1, E, C, D]
        weights = weights.unsqueeze(-1).unsqueeze(-1)     # [B, E, 1, 1]
        mixed = (weights * expert).sum(dim=1)             # [B, C, D]
        return mixed.permute(1, 0, 2)                     # [C, B, D]

    def forward(self, inputs):
        # For the first layer, we do not need to add any duplicate, as it is already added
        # as the shallow version
        x = inputs[0]
        compound_prompts_deeper = inputs[1]
        counter = inputs[2]
        orig_dtype = x.dtype

        # 若不是第一层，可执行 prompt 替换与 MoE
        if not self.first_layer:
            # --------- 1) 先执行 MaPLe 深层 prompt 替换 ---------
            if len(compound_prompts_deeper) > 0:
                if not self.text_layer:
                    if not (counter > len(compound_prompts_deeper) - 1):
                        prefix = x[0:x.shape[0] - self.compound_prompt_nctx, :, :]
                        visual_ctx = compound_prompts_deeper[counter]
                        visual_ctx = visual_ctx.expand(x.shape[1], -1, -1).permute(1, 0, 2).half()
                        x = torch.cat([prefix, visual_ctx], dim=0)
                        counter += 1
                else:
                    if not (counter > len(compound_prompts_deeper) - 1):
                        prefix_cls = x[:1, :, :]
                        suffix = x[1 + self.compound_prompt_nctx:, :, :]
                        textual_ctx = compound_prompts_deeper[counter]
                        textual_ctx = textual_ctx.expand(x.shape[1], -1, -1).permute(1, 0, 2).half()
                        x = torch.cat([prefix_cls, textual_ctx, suffix], dim=0)
                        counter += 1

            # --------- 2) 再执行 MoE 混合 ---------
            if not self.text_layer:
                prefix = x[0:x.shape[0] - self.compound_prompt_nctx, :, :]
                moe_prompt = self._mix_expert_prompt(prefix, orig_dtype)
                x = torch.cat([prefix, moe_prompt], dim=0)
            else:
                prefix_cls = x[:1, :, :]
                suffix = x[1 + self.compound_prompt_nctx:, :, :]
                body = torch.cat([prefix_cls, suffix], dim=0)
                moe_prompt = self._mix_expert_prompt(body, orig_dtype)
                x = torch.cat([prefix_cls, moe_prompt, suffix], dim=0)

        # # Store feature in forward pass
        if not self.text_layer:
            self.visual_feat = x
            
        # 正常 Transformer 残差
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return [x, compound_prompts_deeper, counter]


class ResidualAttentionBlock_MaPLe_MoE_Aware(nn.Module):
    """
    Alignment-Aware Soft MoE for MaPLe.
    Key improvements over MaPLe_MoE:
      1. Configurable CLS/token-mean/hybrid gating on normalized features
      2. Bounded alpha/tau to stabilize optimization
      3. Residual fusion: compound_prompt + alpha * (mixed_prompt - compound_prompt)
      4. expert_diversity_loss() preventing expert collapse
      5. load_balance_loss() for balanced routing
    """
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None,
                 design_details=None, text_layer=False, i: int = 0):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.text_layer = text_layer
        self.attn_mask = attn_mask
        self.compound_prompt_nctx = design_details['maple_length']
        self.first_layer = (i == 0)

        self.num_experts = design_details["num_experts"]
        self.n_ctx = design_details["maple_length"]
        self.delta_scale_init = design_details.get("delta_scale_init", 0.1)
        self.gate_mode = design_details.get("gate_mode", "hybrid")
        self.gate_hybrid_lambda = design_details.get("gate_hybrid_lambda", 0.7)
        self.alpha_min = design_details.get("alpha_min", 0.0)
        self.alpha_max = design_details.get("alpha_max", 2.0)
        self.tau_min = design_details.get("tau_min", 0.3)
        self.tau_max = design_details.get("tau_max", 3.0)
        self._last_gate_weights = None
        self.add_prompt = not self.first_layer

        if self.gate_mode not in {"cls", "token_mean", "hybrid"}:
            raise ValueError(f"Unsupported gate mode: {self.gate_mode}")

        if self.add_prompt:
            expert_prompts = torch.empty(self.num_experts, self.n_ctx, d_model)
            nn.init.normal_(expert_prompts, std=0.02)
            self.expert_prompts = nn.Parameter(expert_prompts)  # [E, C, D]
            self.gate = nn.Linear(d_model, self.num_experts)
            tau_init = min(max(1.0, self.tau_min), self.tau_max)
            alpha_init = min(max(self.delta_scale_init, max(self.alpha_min, 1e-6)), self.alpha_max)
            self.tau = nn.Parameter(torch.full((1,), _inverse_softplus(tau_init)))
            self.alpha = nn.Parameter(torch.full((1,), _inverse_softplus(alpha_init)))

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def _zero_loss(self):
        return self.ln_1.weight.new_tensor(0.0)

    def _bounded_scale(self, value: torch.Tensor, min_value: float, max_value: float):
        scale = F.softplus(value.float())
        if min_value is not None:
            scale = scale.clamp(min=min_value)
        if max_value is not None:
            scale = scale.clamp(max=max_value)
        return scale

    def _router_logits(self, gate_source: torch.Tensor):
        tokens = self.ln_1(gate_source)
        cls_logits = self.gate(tokens[0])
        if self.gate_mode == "cls":
            return cls_logits

        token_mean_logits = self.gate(tokens.mean(dim=0))
        if self.gate_mode == "token_mean":
            return token_mean_logits

        lambda_cls = max(0.0, min(1.0, float(self.gate_hybrid_lambda)))
        return lambda_cls * cls_logits + (1.0 - lambda_cls) * token_mean_logits

    def _mix_expert_prompt(self, prefix: torch.Tensor, orig_dtype: torch.dtype):
        """Soft mixture of expert prompts with configurable routing."""
        logits = self._router_logits(prefix)                     # [B, E]
        tau = self._bounded_scale(self.tau, self.tau_min, self.tau_max)
        weights = F.softmax(
            logits.float() / tau, dim=-1)                        # [B, E]
        self._last_gate_weights = weights

        expert = self.expert_prompts.to(orig_dtype).unsqueeze(0) # [1, E, C, D]
        w = weights.to(orig_dtype).unsqueeze(-1).unsqueeze(-1)   # [B, E, 1, 1]
        mixed = (w * expert).sum(dim=1)                          # [B, C, D]
        return mixed.permute(1, 0, 2).to(orig_dtype)             # [C, B, D]

    def expert_diversity_loss(self):
        """Penalise high cosine similarity between expert prompts."""
        if not self.add_prompt:
            return self._zero_loss()
        E = self.expert_prompts.shape[0]
        if E <= 1:
            return self._zero_loss()
        flat = F.normalize(self.expert_prompts.reshape(E, -1), dim=-1)
        sim = flat @ flat.T
        mask = torch.triu(torch.ones(E, E, device=flat.device), diagonal=1).bool()
        return sim[mask].clamp(min=0).mean()

    def load_balance_loss(self):
        if self._last_gate_weights is None or not self.add_prompt:
            return self._zero_loss()
        avg_weights = self._last_gate_weights.mean(dim=0)
        target = torch.full_like(avg_weights, 1.0 / self.num_experts)
        return F.mse_loss(avg_weights, target)

    def moe_aux_losses(self):
        return self.load_balance_loss(), self.expert_diversity_loss()

    def forward(self, inputs):
        x = inputs[0]
        compound_prompts_deeper = inputs[1]
        counter = inputs[2]
        orig_dtype = x.dtype
        self._last_gate_weights = None

        if self.add_prompt:
            if not self.text_layer:
                prefix = x[:x.shape[0] - self.compound_prompt_nctx, :, :]

                # Retrieve MaPLe compound prompt as the shared cross-modal base
                maple_prompt = None
                if len(compound_prompts_deeper) > 0 and not (counter > len(compound_prompts_deeper) - 1):
                    maple_prompt = compound_prompts_deeper[counter]
                    maple_prompt = maple_prompt.expand(x.shape[1], -1, -1).permute(1, 0, 2)
                    counter += 1

                mixed_prompt = self._mix_expert_prompt(prefix, orig_dtype)
                alpha = self._bounded_scale(self.alpha, self.alpha_min, self.alpha_max).to(orig_dtype)

                if maple_prompt is not None:
                    maple_prompt = maple_prompt.to(orig_dtype)
                    fused = maple_prompt + alpha.view(1, 1, 1) * (mixed_prompt - maple_prompt)
                else:
                    fused = mixed_prompt
                fused = fused.to(orig_dtype)

                x = torch.cat([prefix, fused], dim=0)
            else:
                prefix_cls = x[:1, :, :]
                suffix = x[1 + self.compound_prompt_nctx:, :, :]

                maple_prompt = None
                if len(compound_prompts_deeper) > 0 and not (counter > len(compound_prompts_deeper) - 1):
                    maple_prompt = compound_prompts_deeper[counter]
                    maple_prompt = maple_prompt.expand(x.shape[1], -1, -1).permute(1, 0, 2)
                    counter += 1

                body = torch.cat([prefix_cls, suffix], dim=0)
                mixed_prompt = self._mix_expert_prompt(body, orig_dtype)
                alpha = self._bounded_scale(self.alpha, self.alpha_min, self.alpha_max).to(orig_dtype)

                if maple_prompt is not None:
                    maple_prompt = maple_prompt.to(orig_dtype)
                    fused = maple_prompt + alpha.view(1, 1, 1) * (mixed_prompt - maple_prompt)
                else:
                    fused = mixed_prompt
                fused = fused.to(orig_dtype)

                x = torch.cat([prefix_cls, fused, suffix], dim=0)

        if not self.text_layer:
            self.visual_feat = x

        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return [x, compound_prompts_deeper, counter]


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, prompts_needed=0,
                 text_layer=False, design_details=None):
        super().__init__()
        self.width = width
        self.layers = layers
        # Implements respective encoder blocks for a given design choice
        current_trainer = design_details['trainer']
        if current_trainer == 'IVLP' or current_trainer == 'VPT':
            self.resblocks = nn.Sequential(*[ResidualAttentionBlock_IVLP(width, heads, attn_mask, True,
                                                                         text_layer, i,
                                                                         design_details) if prompts_needed > i
                                             else ResidualAttentionBlock_IVLP(width, heads, attn_mask, False,
                                                                              text_layer, i, design_details)
                                             for i in range(layers)])
        elif current_trainer == 'MaPLe':
            self.resblocks = nn.Sequential(
                *[ResidualAttentionBlock_MaPLe(width, heads, attn_mask, design_details, text_layer, i)
                  for i in range(layers)])
        elif current_trainer == 'MaPLe_MoE':
            self.resblocks = nn.Sequential(
                *[ResidualAttentionBlock_MaPLe_MoE(width, heads, attn_mask, design_details, text_layer, i)
                  for i in range(layers)])
        elif current_trainer == 'IVLP_MoE' or current_trainer == 'VPT_MoE':
            self.resblocks = nn.Sequential(*[ResidualAttentionBlock_IVLP_MoE(width, heads, attn_mask, True,
                                                                         text_layer, i,
                                                                         design_details) if prompts_needed > i
                                             else ResidualAttentionBlock_IVLP_MoE(width, heads, attn_mask, False,
                                                                              text_layer, i, design_details)
                                             for i in range(layers)])
        elif current_trainer == 'MaPLe_MoE_Aware' or current_trainer == 'TAME_VLJ':
            self.resblocks = nn.Sequential(
                *[ResidualAttentionBlock_MaPLe_MoE_Aware(width, heads, attn_mask, design_details, text_layer, i)
                  for i in range(layers)])
        elif current_trainer in ('IVLP_MoE_Aware', 'VPT_MoE_Aware', 'TAME_V', 'TAME_VLI'):
            self.resblocks = nn.Sequential(*[ResidualAttentionBlock_IVLP_MoE_Aware(width, heads, attn_mask, True,
                                                                         text_layer, i,
                                                                         design_details) if prompts_needed > i
                                             else ResidualAttentionBlock_IVLP_MoE_Aware(width, heads, attn_mask, False,
                                                                              text_layer, i, design_details)
                                             for i in range(layers)])
        else:
            # Corresponds to default CoOp or CoCoOp
            assert current_trainer == 'CoOp' or current_trainer == 'CoCoOp'
            self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def moe_aux_losses(self):
        if len(self.resblocks) == 0:
            zero = torch.tensor(0.0)
            return zero, zero

        zero = self.resblocks[0].ln_1.weight.new_tensor(0.0)
        balance = zero
        diversity = zero
        for block in self.resblocks:
            if hasattr(block, "moe_aux_losses"):
                block_balance, block_diversity = block.moe_aux_losses()
                balance = balance + block_balance
                diversity = diversity + block_diversity
        return balance, diversity

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int,
                 output_dim: int, design_details):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
        if design_details["vision_depth"] == 0:
            self.VPT_shallow = False
        else:
            self.VPT_shallow = True
        if self.VPT_shallow:
            # Add visual prompt tokens here
            n_ctx = design_details["vision_ctx"]  # hyperparameter
            ctx_vectors = torch.empty(n_ctx, width)
            nn.init.normal_(ctx_vectors, std=0.02)
            self.VPT = nn.Parameter(ctx_vectors)
            # self.VPT.half()
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)
        # hyper-parameter if need to add prompt embeddings inside to the input
        # of transformer block or not:
        self.prompt_till_layer_visual = design_details["vision_depth"]
        self.transformer = Transformer(width, layers, heads, prompts_needed=self.prompt_till_layer_visual,
                                       design_details=design_details)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)

        # After positional embeddings, we will attach prompts with the model, remember only those
        # are trainable parameters here in whole image encoder.
        if self.VPT_shallow:
            visual_ctx = self.VPT.expand(x.shape[0], -1, -1) #.half()
            x = torch.cat([x, visual_ctx], dim=1)
        else:
            assert self.prompt_till_layer_visual == 0

        # Normal code as before
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x


class VisionTransformer_MaPLe(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int,
                 design_details):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
        self.VPT_shallow = True
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)
        # hyper-parameter if need to add prompt embeddings inside to the input
        # of transformer block or not:
        self.prompt_till_layer_visual = 0
        self.transformer = Transformer(width, layers, heads, design_details=design_details)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor, shared_ctx, compound_deeper_prompts):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)

        # After positional embeddings, we will attach prompts with the model, remember only those
        # are trainable parameters here in whole image encoder.
        if self.VPT_shallow:
            visual_ctx = shared_ctx.expand(x.shape[0], -1, -1) #.half()
            x = torch.cat([x, visual_ctx], dim=1)
        else:
            assert self.prompt_till_layer_visual == 0

        # Normal code as before
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        # Again combine the inputs, so nn.sequential can work
        outputs = self.transformer([x, compound_deeper_prompts, 0])  # third argument is counter
        x = outputs[0]
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 design_details
                 ):
        super().__init__()

        self.context_length = context_length
        trainer = design_details['trainer']

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            if trainer in ("MaPLe", "MaPLe_MoE", "MaPLe_MoE_Aware", "TAME_VLJ"):
                self.visual = VisionTransformer_MaPLe(
                    input_resolution=image_resolution,
                    patch_size=vision_patch_size,
                    width=vision_width,
                    layers=vision_layers,
                    heads=vision_heads,
                    output_dim=embed_dim,
                    design_details=design_details
                )
            else:
                self.visual = VisionTransformer(
                    input_resolution=image_resolution,
                    patch_size=vision_patch_size,
                    width=vision_width,
                    layers=vision_layers,
                    heads=vision_heads,
                    output_dim=embed_dim,
                    design_details=design_details
                )
        # hyper-parameter if need to add prompt embeddings inside to the input
        # of transformer block or not:
        prompt_till_layer_text = design_details['language_depth']
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),
            prompts_needed=prompt_till_layer_text,
            text_layer=True,
            design_details=design_details
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict, design_details):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in
                        [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers, design_details
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    # convert_weights(model)
    try:
        model.load_state_dict(state_dict)
    except:
        missing_keys, _ = model.load_state_dict(state_dict, strict=False)
        print('Weights not found for some missing keys: ', missing_keys)
    return model.eval()
