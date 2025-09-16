import math
import torch
import torch.nn as nn
from model.transformer import MultiHeadAttention
from model.convnext import ConvNeXt1DBlock

class SimpleDownsample(nn.Module):
    """Downsample by factor ds with learnable softmax weights"""
    def __init__(self, downsample: int = 2):
        super().__init__()
        self.ds = downsample
        self.bias = nn.Parameter(torch.zeros(downsample))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        B, T, C = x.shape
        ds = self.ds
        dT = (T + ds - 1) // ds  # ceil division

        # pad last frame if not divisible
        pad = dT * ds - T
        if pad > 0:
            x = torch.cat([x, x[:, -1:].expand(B, pad, C)], dim=1)  # (B, T+pad, C)

        # group into (B, dT, ds, C)
        x = x.view(B, dT, ds, C)

        # softmax weights across ds
        w = self.bias.softmax(dim=0).view(1, 1, ds, 1)
        x = (x * w).sum(dim=2)  # (B, dT, C)
        return x


class SimpleUpsample(nn.Module):
    """Nearest-neighbor repeat upsample by factor us"""
    def __init__(self, upsample: int = 2):
        super().__init__()
        self.us = upsample

    def forward(self, x: torch.Tensor, target_len: int) -> torch.Tensor:
        # x: (B, T', C)
        x = x.unsqueeze(2).expand(-1, -1, self.us, -1)  # (B, T', us, C)
        x = x.reshape(x.size(0), x.size(1) * self.us, -1)  # (B, T'*us, C)
        return x[:, :target_len]

class OutCombiner(nn.Module):
    """
    Per-channel residual gate:
        out = x_orig + (x_proc - x_orig) * s
            = (1 - s) * x_orig + s * x_proc
    where s is a learnable vector of shape (C,).
    """
    def __init__(self, channels: int, init: float = 0.5, scale_min: float = 0.2, scale_max: float = 1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.full((channels,), float(init)))
        self.scale_min = scale_min
        self.scale_max = scale_max

    def forward(self, x_orig: torch.Tensor, x_proc: torch.Tensor) -> torch.Tensor:
        # x_*: (B, T, C)
        s = torch.clamp(self.scale, self.scale_min, self.scale_max)  # (C,)
        return x_orig + (x_proc - x_orig) * s  # broadcast to (B, T, C)

# --- (A) 마스크 다운샘플 도우미 ---
def downsample_mask(mask: torch.Tensor, ds: int) -> torch.Tensor:
    """
    mask: (B, T)  - True=유효, False=패딩(혹은 무시)
    ds  : 다운샘플 비율
    반환: (B, ceil(T/ds))
    규칙: 윈도우 내에 하나라도 유효(True)가 있으면 유효로 간주(ANY pooling)
    """
    B, T = mask.shape
    dT = (T + ds - 1) // ds
    pad = dT * ds - T
    if pad > 0:
        pad_zeros = torch.zeros(B, pad, dtype=mask.dtype, device=mask.device)
        mask = torch.cat([mask, pad_zeros], dim=1)  # 패딩은 False로 채움
    # (B, dT, ds)로 묶어 윈도우 단위로 any pooling
    mask = mask.view(B, dT, ds)
    mask = mask.any(dim=2)
    return mask


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

import torch
import torch.nn as nn

class TimeCondBlock(nn.Module):
    """
    Time conditioning with scale & shift (FiLM style).
    Projects time embedding to (scale, shift) and applies:
        out = x * (1 + scale) + shift
    """
    def __init__(self, channel_dim: int, time_emb_dim: int = 64):
        super().__init__()
        self.proj = nn.Linear(time_emb_dim, channel_dim * 2, bias=False)
        nn.init.zeros_(self.proj.weight)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        scale_shift = self.proj(t_emb)  # (B, 2C)
        scale, shift = scale_shift.chunk(2, dim=-1)  # (B, C), (B, C)

        scale = scale[:, None, :]
        shift = shift[:, None, :]

        return x * (1 + scale) + shift

class TextCondBlock(nn.Module):
    """
    Cross-attention with text embeddings (speaker-adaptive text from TextEncoder).
    Keys/Values: text embeddings; Queries: current hidden states.
    """
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.ca = MultiHeadAttention(q_dim=dim, kv_dim=dim, num_heads=num_heads)

    def forward(self, x: torch.Tensor, text_embed: torch.Tensor, text_attention_mask: torch.Tensor = None) -> torch.Tensor:
        # x: (B, T, C), text_embed: (B, T_text, C)
        return self.ca(q_inp=x, kv_inp=text_embed, attention_mask=text_attention_mask)

class VFEstimator(nn.Module):
    def __init__(
        self,
        dim_in: int = 144,
        dim_model: int = 256,
        conv_hidden: int = 1024,
        num_heads: int = 4,
        Nm: int = 5,
        text_dim:int = 128,
        downsample_factors: list = [1, 2, 4, 2, 1]
    ):
        super().__init__()
        # self.inp = nn.Linear(dim_in, dim_model)
        self.inp = nn.Sequential(
            nn.Linear(dim_in*2, dim_model),
            nn.SiLU(),
            nn.Linear(dim_model, dim_model),
        )

        self.text_proj = nn.Linear(text_dim, dim_model)

        # One set of dilations used inside each main block
        self.dilations = [1, 2, 4, 8]
        self.downsample_factors = downsample_factors

        # Build repeated main blocks
        self.main_blocks = nn.ModuleList()
        for i in range(Nm):
            blocks = nn.ModuleDict({
                "downsample": SimpleDownsample(downsample=downsample_factors[i]),
                "dilated": nn.ModuleList([ConvNeXt1DBlock(dim_model, hidden_dim=conv_hidden, kernel_size=5, dilation=d) for d in self.dilations]),
                "time": TimeCondBlock(channel_dim=dim_model, time_emb_dim=64),
                "std1": ConvNeXt1DBlock(dim_model, hidden_dim=conv_hidden, kernel_size=5, dilation=1),
                "text": TextCondBlock(dim=dim_model, num_heads=num_heads),
                "text_ln": nn.LayerNorm(dim_model),
                "std2": ConvNeXt1DBlock(dim_model, hidden_dim=conv_hidden, kernel_size=5, dilation=1),
                "self": MultiHeadAttention(q_dim=dim_model, kv_dim=dim_model, num_heads=num_heads),
                "self_ln": nn.LayerNorm(dim_model),
                "upsample": SimpleUpsample(upsample=downsample_factors[i])
            })
            self.main_blocks.append(blocks)
        self.out_combiners = nn.ModuleList([OutCombiner(dim_model) for _ in range(Nm)])

        # Tail: 4 additional ConvNeXt blocks
        self.tail = nn.ModuleList([ConvNeXt1DBlock(dim_model, hidden_dim=conv_hidden) for _ in range(4)])

        self.out = nn.Linear(dim_model, dim_in)
    
    @torch.no_grad()
    def forward_cfg(
        self,
        x_t: torch.Tensor,                 # (B, T, Cx)  여기선 (B, 100, T)
        context: torch.Tensor,             # (B, 100, T)
        times: torch.Tensor,               # (B,)
        text_embed: torch.Tensor,          # (B, T_text, text_dim)
        *,
        audio_mask: torch.Tensor | None = None,   # (B, T) or None
        text_mask: torch.Tensor | None = None,    # (B, T_text) or None
        guidance_scale: float = 1.0,
        concat: bool = True                # True: 한 번에 배치 concat해서 forward
    ) -> torch.Tensor:
        """
        CFG: vf = vf_uncond + s * (vf_cond - vf_uncond)
        - uncond는 text_embed/텍스트 마스크를 '비운' 상태로 계산
        - concat=True면 (2B)로 합쳐 1회 forward (빠름)
        """
        if guidance_scale == 0.0:
            return self.forward(x_t, context, times, text_embed, audio_mask, text_mask)

        B = x_t.shape[0]
        device = x_t.device

        # 무조건(uncond) 텍스트: 임베드 0, 마스크 전부 False
        text_embed_uncond = torch.zeros_like(text_embed, device=device)
        text_mask_uncond = (torch.zeros_like(text_mask, dtype=torch.bool, device=device)
                            if text_mask is not None else None)

        if not concat:
            # 두 번 호출(간단하지만 느릴 수 있음)
            vf_uncond = self.forward(x_t, context, times, text_embed_uncond, audio_mask, text_mask_uncond)
            vf_cond   = self.forward(x_t, context, times, text_embed,         audio_mask, text_mask)
            return vf_uncond + guidance_scale * (vf_cond - vf_uncond)

        # concat 경로: (uncond, cond) 순서로 붙여 한 번에 forward
        x_t_cat      = torch.cat([x_t,      x_t],      dim=0)
        context_cat  = torch.cat([context,  context],  dim=0)
        times_cat    = torch.cat([times,    times],    dim=0)
        text_cat     = torch.cat([text_embed_uncond, text_embed], dim=0)

        if audio_mask is not None:
            audio_mask_cat = torch.cat([audio_mask, audio_mask], dim=0)
        else:
            audio_mask_cat = None

        if text_mask is not None:
            text_mask_cat = torch.cat([text_mask_uncond, text_mask], dim=0)
        else:
            text_mask_cat = None

        vf_cat = self.forward(x_t_cat, context_cat, times_cat, text_cat,
                              audio_mask_cat, text_mask_cat)          # (2B, 100, T)

        vf_uncond, vf_cond = vf_cat[:B], vf_cat[B:]
        return vf_uncond + guidance_scale * (vf_cond - vf_uncond)

    def forward(
        self,
        x_t: torch.Tensor,                 # (B, 100, T)
        context: torch.Tensor,             # (B, 100, T)
        times: torch.Tensor,               # (B,) scalarized time (e.g., diffusion/flow t)
        text_embed: torch.Tensor,          # (B, T_text, 256?) -> must be 256; project if needed
        audio_mask: torch.Tensor = None,   # (B, T)
        text_mask: torch.Tensor = None,    # (B, T_text)
    ) -> torch.Tensor:
        """
        Returns predicted vector field: (B, 100, T)
        Note: If text/ref embeddings are 128-d (as in paper), pass simple linear adapters to 256.
        """
        x_t = torch.cat([x_t, context], dim=-1)
        x = self.inp(x_t)  # (B, T, 256)

        # Prepare condition streams (project to model dim if necessary)
        text_embed = self.text_proj(text_embed)
        
        # Time embedding once
        t_emb = gradtts_time_embedding(times, dim=64)

        # Repeated main structure
        for i, blk in enumerate(self.main_blocks):
            origin_x = x
            orig_len = origin_x.shape[1]

            attn_mask_cur = None
            if audio_mask is not None:
                ds_i = self.downsample_factors[i]
                attn_mask_cur = downsample_mask(audio_mask, ds_i)  # (B, dT)

            x = blk["downsample"](x)

            for dblk in blk["dilated"]:
                x = dblk(x)

            x = blk["time"](x, t_emb)

            x = blk["std1"](x)
            x = x + blk["text"](blk["text_ln"](x), text_embed, text_attention_mask=text_mask)
            x = blk["std2"](x)

            x = x + blk["self"](blk["self_ln"](x), attention_mask=attn_mask_cur)

            x = blk["upsample"](x, orig_len)

            x = self.out_combiners[i](origin_x, x)

        # tail convnext x4
        for tblk in self.tail:
            x = tblk(x)

        # out projection
        vf = self.out(x)  # (B, T, 144)
        return vf
