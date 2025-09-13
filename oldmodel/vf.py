import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.transformer import CrossAttention
from model.convnext import ConvNeXt1DBlock

def gradtts_time_embedding(t: torch.Tensor, dim: int = 64, max_period: float = 10000.0) -> torch.Tensor:
    """
    Grad-TTS style sinusoidal time embedding (dim must be even).
    t: (B,) in [0, 1] or any scalar schedule space; treat as continuous.
    Returns: (B, dim)
    """
    assert dim % 2 == 0
    half = dim // 2
    device = t.device
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(0, half, device=device).float() / half
    )
    args = t[:, None] * freqs[None, :]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    return emb

class TimeCondBlock(nn.Module):
    """
    Time conditioning: project 64-d time embedding to channel dim and add globally.
    """
    def __init__(self, channel_dim: int, time_emb_dim: int = 64):
        super().__init__()
        self.proj = nn.Linear(time_emb_dim, channel_dim)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C), t_emb: (B, time_emb_dim)
        add = self.proj(t_emb)[:, None, :]  # (B, 1, C)
        return x + add


class TextCondBlock(nn.Module):
    """
    Cross-attention with text embeddings (speaker-adaptive text from TextEncoder).
    Keys/Values: text embeddings; Queries: current hidden states.
    """
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.ca = CrossAttention(q_dim=dim, kv_dim=dim, num_heads=num_heads, dropout=dropout)

    def forward(self, x: torch.Tensor, text_embed: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C), text_embed: (B, T_text, C)
        return self.ca(x, k_inp=text_embed, v_inp=text_embed)


class RefCondBlock(nn.Module):
    """
    Cross-attention with reference speech representation.
    Keys/Values: reference value vectors (fixed-size); Queries: current hidden states.
    Optionally, keys can be replaced by the 50 learned keys reused from TextEncoder.
    """
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.ca = CrossAttention(q_dim=dim, kv_dim=dim, num_heads=num_heads, dropout=dropout)

    def forward(self, x: torch.Tensor, ref_value: torch.Tensor, ref_key_override: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        ref_value: (B, T_ref, C)
        ref_key_override: if provided, use these as keys by concatenating with values along feature dim
                          Here we follow paper intent by 'reusing 50 learned keys in VF' via
                          replacing/augmenting K with learnable keys while V remains ref_value.
        """
        if ref_key_override is None:
            kv = ref_value
            return self.ca(x, k_inp=kv, v_inp=kv)
        else:
            # fuse keys and values into single kv tensor of shape (B, Tk, C)
            # simplest: keep V = ref_value but replace its features with ref_key features via a small proj
            # to align with q_dim (C). Here we just take the same channel dim and average them.
            # (Many equivalent implementations are possible; paper specifies key reuse, not exact fusion op.)
            # We'll concatenate then project back to C as a pragmatic choice.
            B, Tk, C = ref_value.shape
            keys = ref_key_override  # (B, 50, C) broadcasted from (50, C) at call site
            kv_cat = torch.cat([keys, ref_value], dim=1)  # (B, 50+Tk, C)
            kv = kv_cat
            return self.ca(x, k_inp=keys, v_inp=ref_value)
        


class VFEstimator(nn.Module):
    """
    Vector Field Estimator (text-to-latent module)
      - Input noisy compressed latents: dim_in = 144
      - First Linear: 144 -> 256
      - Main block repeated Nm=4:
          [ TimeCond + (DilatedConvNeXt x4 with dilations 1,2,4,8)
            + (ConvNeXt x2) + TextCond + RefCond ]
      - Tail: ConvNeXt x4
      - Output Linear: 256 -> 144

    ConvNeXt: kernel=5, intermediate_dim=1024 (all), per paper.
    Time embedding: 64-d via Grad-TTS method, added globally after projection.
    Text/Ref conditioning: cross-attention (Q=current states; K,V = text embeddings / reference value).
    Reuse 50 learnable keys from TextEncoder by passing them in ref_key_override.
    """
    def __init__(
        self,
        dim_in: int = 144,
        dim_model: int = 256,
        conv_hidden: int = 1024,
        num_heads: int = 4,
        Nm: int = 4,
        text_dim:int = 128,
        ref_dim:int = 128,
    ):
        super().__init__()
        self.inp = nn.Linear(dim_in, dim_model)

        if text_dim != dim_model:
            self.text_proj = nn.Linear(text_dim, dim_model)
        if ref_dim != dim_model:
            self.ref_proj = nn.Linear(ref_dim, dim_model)

        # One set of dilations used inside each main block
        self.dilations = [1, 2, 4, 8]

        # Build repeated main blocks
        self.main_blocks = nn.ModuleList()
        for _ in range(Nm):
            blocks = nn.ModuleDict({
                "time": TimeCondBlock(channel_dim=dim_model, time_emb_dim=64),
                "dilated": nn.ModuleList([ConvNeXt1DBlock(dim_model, hidden_dim=conv_hidden, kernel_size=5, dilation=d) for d in self.dilations]),
                "std1": ConvNeXt1DBlock(dim_model, hidden_dim=conv_hidden, kernel_size=5, dilation=1),
                "std2": ConvNeXt1DBlock(dim_model, hidden_dim=conv_hidden, kernel_size=5, dilation=1),
                "text": TextCondBlock(dim=dim_model, num_heads=num_heads),
                "ref": RefCondBlock(dim=dim_model, num_heads=num_heads),
            })
            self.main_blocks.append(blocks)

        # Tail: 4 additional ConvNeXt blocks
        self.tail = nn.ModuleList([ConvNeXt1DBlock(dim_model, hidden_dim=conv_hidden) for _ in range(4)])

        self.out = nn.Linear(dim_model, dim_in)

    def forward(
        self,
        noisy_latents: torch.Tensor,       # (B, T, 144)
        time_t: torch.Tensor,              # (B,) scalarized time (e.g., diffusion/flow t)
        text_embed: torch.Tensor,          # (B, T_text, 256?) -> must be 256; project if needed
        ref_value: torch.Tensor,           # (B, T_ref, 256?) -> must be 256; project if needed
        learned_keys_50: Optional[torch.Tensor] = None,  # (50, 256) from TextEncoder, reused as K
        audio_mask: torch.Tensor = None,   # (B, T)
        text_mask: torch.Tensor = None,    # (B, T_text)
    ) -> torch.Tensor:
        """
        Returns predicted vector field: (B, T, 144)
        Note: If text/ref embeddings are 128-d (as in paper), pass simple linear adapters to 256.
        """
        x = self.inp(noisy_latents)  # (B, T, 256)

        # Prepare condition streams (project to model dim if necessary)
        if self.text_proj is not None:
            text_embed = self.text_proj(text_embed)
        else:
            text_embed = text_embed
        if self.ref_proj is not None:
            ref_value = self.ref_proj(ref_value)
        else:
            ref_value = ref_value

        # Broadcast learned keys to batch and (optionally) project
        ref_key_override = None
        if learned_keys_50 is not None:
            keys = learned_keys_50
            if keys.dim() == 2:
                keys = keys.unsqueeze(0).expand(x.size(0), -1, -1)  # (B, 50, C_key)
            # If key dim != model dim, add simple projector
            if keys.size(-1) != x.size(-1):
                proj = nn.Linear(keys.size(-1), x.size(-1)).to(x.device)
                keys = proj(keys)
            ref_key_override = keys

        # Time embedding once
        t_emb = gradtts_time_embedding(time_t, dim=64)

        # Repeated main structure
        for blk in self.main_blocks:
            # time cond
            x = blk["time"](x, t_emb)
            # 4 dilated convnext
            for dblk in blk["dilated"]:
                x = dblk(x)
            # 2 standard convnext
            x = blk["std1"](x)
            x = blk["std2"](x)
            # text cond
            x = blk["text"](x, text_embed)
            # ref cond (reuse learned keys as K)
            x = blk["ref"](x, ref_value, ref_key_override=ref_key_override)

        # tail convnext x4
        for tblk in self.tail:
            x = tblk(x)

        # out projection
        vf = self.out(x)  # (B, T, 144)
        return vf
