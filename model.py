# vf_estimator.py
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# Speed knobs (optional)
# ----------------------------
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# ----------------------------
# GRN (for ConvNeXt V2 block)
# ----------------------------
class GRN(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, N, C)
        # Global Response Normalization (ConvNeXt V2)
        Gx = torch.norm(x, p=2, dim=-1, keepdim=True)                 # (B, N, 1)
        Nx = Gx / (Gx.mean(dim=-2, keepdim=True) + self.eps)          # (B, N, 1)
        return self.gamma * (x * Nx) + self.beta + x

# ----------------------------
# Your ConvNeXtV2Block (as given)
# ----------------------------
class ConvNeXtV2Block(nn.Module):
    def __init__(self, dim: int, intermediate_dim: int, dilation: int = 1):
        super().__init__()
        padding = (dilation * (7 - 1)) // 2
        self.dwconv = nn.Conv1d(
            dim, dim, kernel_size=7, padding=padding, groups=dim, dilation=dilation
        )  # depthwise conv
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)  # pointwise/1x1 convs
        self.act = nn.GELU()
        self.grn = GRN(intermediate_dim)
        self.pwconv2 = nn.Linear(intermediate_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, N, D)
        residual = x
        x = x.transpose(1, 2)             # (B, D, N)
        x = self.dwconv(x)
        x = x.transpose(1, 2)             # (B, N, D)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        return residual + x

# ----------------------------
# Time embedding (Grad-TTS style)
# ----------------------------
def sinusoidal_t_embed(t: torch.Tensor, dim: int = 64) -> torch.Tensor:
    """
    t: (B,) in [0,1], returns (B, dim) sinusoidal time embedding.
    """
    device = t.device
    half = dim // 2
    # 2*pi for better coverage; consistent with many diffusion impls
    freqs = torch.exp(
        torch.linspace(0, math.log(10000), half, device=device)
        * (-1)
    )
    # shape: (B, half)
    args = 2.0 * math.pi * t[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb  # (B, dim)

# ----------------------------
# VF Estimator
# ----------------------------
@dataclass
class VFConfig:
    d_in: int = 144        # compressed latent dim (C*Kc)
    d_model: int = 256     # hidden channels inside VF
    d_time: int = 64       # time embedding dim (before projection)
    n_repeats: int = 4     # Nm in the paper
    n_post: int = 4        # post ConvNeXt blocks count
    inter_dim: int = 1024  # ConvNeXt MLP hidden
    kernel_size: int = 5
    num_heads: int = 4
    head_dim: int = 64

class VFEstimator(nn.Module):
    """
    Implements the VF Estimator (Fig. 4(c) & Appx A.2.3) of Supertonic-TTS.
    - Input: noisy compressed latents z_t (B, T, 144)
    - Conditions: time t (B,), text_kv (B, Nt, Dt), ref_kv (B, Nr, Dr)
    - Output: vector field v(z_t, text, ref, t) with shape (B, T, 144)
    """
    def __init__(self, cfg: VFConfig = VFConfig()):
        super().__init__()
        self.cfg = cfg

        # input/output projections
        self.proj_in  = nn.Linear(cfg.d_in, cfg.d_model)
        self.proj_out = nn.Linear(cfg.d_model, cfg.d_in)

        # time conditioning: 64-d sinusoidal -> project to d_model, then global add
        self.time_proj = nn.Linear(cfg.d_time, cfg.d_model)

        # a helper to build blocks
        def convnext_block(dilation: int = 1):
            return ConvNeXtV2Block(dim=cfg.d_model, intermediate_dim=cfg.inter_dim, dilation=dilation)

        # main repeated stages
        self.stages = nn.ModuleList()
        for _ in range(cfg.n_repeats):
            stage = nn.ModuleDict(
                dict(
                    dilated_1=convnext_block(dilation=1),
                    dilated_2=convnext_block(dilation=2),
                    dilated_3=convnext_block(dilation=4),
                    dilated_4=convnext_block(dilation=8),
                    # after time add, two standard ConvNeXt blocks around cross-attn
                    pre_text = convnext_block(dilation=1),
                    post_text= convnext_block(dilation=1),
                )
            )
            self.stages.append(stage)

        # post blocks
        self.post_blocks = nn.Sequential(*[convnext_block(dilation=1) for _ in range(cfg.n_post)])

        # lightweight norms on the path
        self.pre_text_ln = nn.LayerNorm(cfg.d_model)
        self.post_text_ln = nn.LayerNorm(cfg.d_model)
        self.pre_ref_ln = nn.LayerNorm(cfg.d_model)
        self.post_ref_ln = nn.LayerNorm(cfg.d_model)

    def forward(
        self,
        zt: torch.Tensor,               # (B, T, 144)
        t: torch.Tensor,                # (B,)
        text_kv: torch.Tensor,          # (B, Nt, 128)
        ref_kv: torch.Tensor,           # (B, Nr, 128)
    ) -> torch.Tensor:
        x = self.proj_in(zt)  # (B, T, d_model)

        # precompute time embedding once
        t_emb = sinusoidal_t_embed(t, dim=self.cfg.d_time)
        t_add = self.time_proj(t_emb)  # (B, d_model)

        for stage in self.stages:
            # 4 dilated ConvNeXt blocks
            x = stage["dilated_1"](x)
            x = stage["dilated_2"](x)
            x = stage["dilated_3"](x)
            x = stage["dilated_4"](x)

            # time conditioning: global add (broadcast over time)
            x = x + t_add[:, None, :]

            # ConvNeXt -> Text cross-attn -> ConvNeXt
            x = stage["pre_text"](x)
            x = self.pre_text_ln(x)
            x = x + self.text_attn(x, text_kv)  # residual
            x = stage["post_text"](x)

            # LayerNorm -> Ref cross-attn (residual)
            x = self.pre_ref_ln(x)
            x = x + self.ref_attn(x, ref_kv)
            x = self.post_ref_ln(x)

        # tail ConvNeXt stack + projection to 144-dim
        x = self.post_blocks(x)
        out = self.proj_out(x)  # (B, T, 144)
        return out