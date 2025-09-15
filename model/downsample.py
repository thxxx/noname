import torch
import torch.nn as nn

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
