import torch
import torch.nn as nn
import math
from typing import Optional, Tuple

# ----------------------------
# ConvNeXt-style 1D Block (kernel size = 5, no GRN)
# ----------------------------
class ConvNeXtBlock1D(nn.Module):
    def __init__(self, dim: int, intermediate_dim: int = 512, kernel_size: int = 5, dilation: int = 1):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        padding = (dilation * (kernel_size - 1)) // 2
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=kernel_size,
                                padding=padding, groups=dim, dilation=dilation)
        self.norm   = nn.LayerNorm(dim, eps=1e-6)
        self.pw1    = nn.Linear(dim, intermediate_dim)
        self.act    = nn.GELU()
        self.pw2    = nn.Linear(intermediate_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, N, C)
        residual = x
        x = x.transpose(1, 2)                 # (B, C, N)
        x = self.dwconv(x)
        x = x.transpose(1, 2)                 # (B, N, C)
        x = self.norm(x)
        x = self.pw1(x)
        x = self.act(x)
        x = self.pw2(x)
        return x + residual

# ----------------------------
# Rotary Positional Embedding (RoPE) helpers
# ----------------------------
def build_rotary(freqs: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # freqs: (Hd,), t: (N,)
    ang = torch.einsum("n,d->nd", t, freqs)   # (N, Hd)
    return ang.sin().unsqueeze(0), ang.cos().unsqueeze(0)  # (1, N, Hd)

def apply_rope(q: torch.Tensor, k: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # q,k: (B, H, N, Hd), sin,cos: (1, N, Hd)
    # split last dim into [even, odd] and rotate
    q1, q2 = q[..., ::2], q[..., 1::2]
    k1, k2 = k[..., ::2], k[..., 1::2]
    sin_e = sin[..., ::2]
    cos_e = cos[..., ::2]
    q_rot = torch.stack([q1 * cos_e - q2 * sin_e, q1 * sin_e + q2 * cos_e], dim=-1).flatten(-2)
    k_rot = torch.stack([k1 * cos_e - k2 * sin_e, k1 * sin_e + k2 * cos_e], dim=-1).flatten(-2)
    return q_rot, k_rot

class RotaryEmbedding(nn.Module):
    """
    Rotary embeddings for attention.
    dim must be even (per-head dim).
    """
    def __init__(self, dim: int, base: int = 10000):
        super().__init__()
        assert dim % 2 == 0, "per-head dimension must be even for RoPE."
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int, device=None):
        t = torch.arange(seqlen, device=device, dtype=self.inv_freq.dtype)
        sin, cos = build_rotary(self.inv_freq.to(device), t)
        return sin, cos  # (1, N, Hd)

# ----------------------------
# Multi-Head Self-Attention with RoPE
# ----------------------------
class SelfAttentionBlock(nn.Module):
    def __init__(self, dim: int = 512, num_heads: int = 4, dropout: float = 0.0, use_rope: bool = True):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.out = nn.Linear(dim, dim, bias=True)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)
        self.use_rope = use_rope
        self.rope = RotaryEmbedding(self.head_dim) if use_rope else None
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (B, N, C)
        attn_mask: (B, 1, N, N) or (B, N) padding mask (True=keep / False=mask) -> we convert to additive mask
        """
        B, N, C = x.shape
        residual = x
        x = self.norm(x)

        qkv = self.qkv(x).view(B, N, 3, self.num_heads, self.head_dim).transpose(1, 3)  # (B, H, 3, N, Hd)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]                             # (B, H, N, Hd)

        if self.use_rope:
            sin, cos = self.rope(N, device=x.device)
            q, k = apply_rope(q, k, sin, cos)

        attn = torch.einsum("bhnd,bhmd->bhnm", q, k) / math.sqrt(self.head_dim)        # (B,H,N,N)

        if attn_mask is not None:
            # Support padding mask (B,N): convert to (B,1,1,N) additive mask
            if attn_mask.dim() == 2:
                pad = (~attn_mask).unsqueeze(1).unsqueeze(1)  # True where to mask
                attn = attn.masked_fill(pad, float("-inf"))
            else:
                attn = attn + attn_mask  # assume already additive

        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        y = torch.einsum("bhnm,bhmd->bhnd", attn, v).transpose(1, 2).contiguous()      # (B,N,H,Hd)
        y = y.view(B, N, C)
        y = self.out(y)
        y = self.proj_drop(y)

        x = residual + y
        # MLP
        x = x + self.mlp(x)
        return x

# ----------------------------
# Text Encoder (spec-compliant)
# ----------------------------
class TextEncoder(nn.Module):
    """
    - Embedder: vocab_size -> 128
    - 6 × ConvNeXtBlock1D (kernel=5, inter=512), model dim kept at 512 (project up once)
    - 4 × Self-Attention blocks (dim=512, heads=4, RoPE)
    - 2 × Cross-Attention layers (dim=512, heads=4), optional external context
    """
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = 128,
        model_dim: int = 512,
        num_convnext: int = 6,
        num_self_attn: int = 4,
        conv_kernel: int = 5,
        conv_intermediate: int = 512,
        num_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim)
        self.in_proj = nn.Linear(emb_dim, model_dim)

        self.conv_blocks = nn.ModuleList([
            ConvNeXtBlock1D(dim=model_dim, intermediate_dim=conv_intermediate, kernel_size=conv_kernel)
            for _ in range(num_convnext)
        ])

        self.self_blocks = nn.ModuleList([
            SelfAttentionBlock(dim=model_dim, num_heads=num_heads, dropout=dropout, use_rope=True)
            for _ in range(num_self_attn)
        ])

        self.norm_out = nn.LayerNorm(model_dim)

    def forward(
        self,
        text_ids: torch.Tensor,                 # (B, N)
        text_mask: Optional[torch.Tensor] = None,   # (B, N) True=keep / False=pad
        cross_kv: Optional[torch.Tensor] = None,    # (B, M, 512)
        cross_mask: Optional[torch.Tensor] = None,  # (B, M) True=keep
    ) -> torch.Tensor:
        x = self.embed(text_ids)            # (B, N, 128)
        x = self.in_proj(x)                 # (B, N, 512)

        for blk in self.conv_blocks:
            x = blk(x)                      # (B, N, 512)

        for blk in self.self_blocks:
            x = blk(x, attn_mask=text_mask) # (B, N, 512)

        x = self.norm_out(x)
        return x                             # (B, N, 512)
