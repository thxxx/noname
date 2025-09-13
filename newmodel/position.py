from typing import Optional, Tuple
import torch
import torch.nn as nn

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.base = base

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, T, H, D_head) or (B, T, D) before splitting heads.
        Returns cos, sin shaped to broadcast to x
        """
        if x.dim() == 4:
            d = x.size(-1)
            T = x.size(1)
        else:
            d = x.size(-1)
            T = x.size(1)
        inv_freq = 1.0 / (self.base ** (torch.arange(0, d, 2, device=x.device).float() / d))
        t = torch.arange(T, device=x.device).float()
        freqs = torch.einsum("t,d->td", t, inv_freq)  # (T, d/2)
        emb = torch.cat((freqs, freqs), dim=-1)  # (T, d)
        cos = emb.cos().unsqueeze(0).unsqueeze(2)  # (1, T, 1, d)
        sin = emb.sin().unsqueeze(0).unsqueeze(2)  # (1, T, 1, d)
        return cos, sin

def apply_rope(q: torch.Tensor, k: torch.Tensor, rope: RotaryPositionalEmbedding):
    """
    q, k: (B, T, H, Dh)
    """
    cos, sin = rope(q)
    # rotate last dim in pairs
    def _rotate(x):
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        x_rot = torch.stack((-x2, x1), dim=-1).reshape_as(x)
        return x_rot
    q = q * cos + _rotate(q) * sin
    k = k * cos + _rotate(k) * sin
    return q, k
