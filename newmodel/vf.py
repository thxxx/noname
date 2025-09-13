import math
import torch
import torch.nn as nn
from newmodel.transformer import MultiHeadAttention
from newmodel.convnext import ConvNeXt1DBlock
from newmodel.downsample import SimpleDownsample, SimpleUpsample

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
    Time conditioning with scale & shift (FiLM style).
    Projects time embedding to (scale, shift) and applies:
        out = x * (1 + scale) + shift
    """
    def __init__(self, channel_dim: int, time_emb_dim: int = 64):
        super().__init__()
        self.proj = nn.Linear(time_emb_dim, channel_dim * 2)  # scale + shift

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        scale_shift = self.proj(t_emb)
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
        return self.ca(x, kv_inp=text_embed, attention_mask=text_attention_mask)


class VFEstimator(nn.Module):
    def __init__(
        self,
        dim_in: int = 144,
        dim_model: int = 256,
        conv_hidden: int = 1024,
        num_heads: int = 4,
        Nm: int = 4,
        text_dim:int = 128,
        downsample_factors: list = [1, 2, 4, 2, 1]
    ):
        super().__init__()
        self.inp = nn.Linear(dim_in, dim_model)

        self.text_proj = nn.Linear(text_dim, dim_model)

        # One set of dilations used inside each main block
        self.dilations = [1, 2, 4, 8]

        # Build repeated main blocks
        self.main_blocks = nn.ModuleList()
        for i in range(Nm):
            blocks = nn.ModuleDict({
                "downsample": SimpleDownsample(downsample=downsample_factors[i]),
                "time": TimeCondBlock(channel_dim=dim_model, time_emb_dim=64),
                "dilated": nn.ModuleList([ConvNeXt1DBlock(dim_model, hidden_dim=conv_hidden, kernel_size=5, dilation=d) for d in self.dilations]),
                "std1": ConvNeXt1DBlock(dim_model, hidden_dim=conv_hidden, kernel_size=5, dilation=1),
                "std2": ConvNeXt1DBlock(dim_model, hidden_dim=conv_hidden, kernel_size=5, dilation=1),
                "text": TextCondBlock(dim=dim_model, num_heads=num_heads),
                "self": MultiHeadAttention(q_dim=dim_model, kv_dim=dim_model, num_heads=num_heads),
                "upsample": SimpleUpsample(upsample=downsample_factors[i])
            })
            self.main_blocks.append(blocks)

        # Tail: 4 additional ConvNeXt blocks
        self.tail = nn.ModuleList([ConvNeXt1DBlock(dim_model, hidden_dim=conv_hidden) for _ in range(4)])

        self.out = nn.Linear(dim_model, dim_in)

    def forward(
        self,
        x_t: torch.Tensor,                 # (B, T, 144)
        context: torch.Tensor,             # (B, T, 144)
        times: torch.Tensor,              # (B,) scalarized time (e.g., diffusion/flow t)
        text_embed: torch.Tensor,          # (B, T_text, 256?) -> must be 256; project if needed
        audio_mask: torch.Tensor = None,   # (B, T)
        text_mask: torch.Tensor = None,    # (B, T_text)
    ) -> torch.Tensor:
        """
        Returns predicted vector field: (B, T, 144)
        Note: If text/ref embeddings are 128-d (as in paper), pass simple linear adapters to 256.
        """
        x = self.inp(x_t)  # (B, T, 256)

        # Prepare condition streams (project to model dim if necessary)
        text_embed = self.text_proj(text_embed)
        
        # Time embedding once
        t_emb = gradtts_time_embedding(times, dim=64)

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
            x = blk["text"](x, text_embed, text_attention_mask=text_mask)

            # self attn
            x = blk["self"](x, attention_mask=audio_mask)

        # tail convnext x4
        for tblk in self.tail:
            x = tblk(x)

        # out projection
        vf = self.out(x)  # (B, T, 144)
        return vf
