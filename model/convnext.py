import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNeXt1DBlock(nn.Module):
    def __init__(self, dim: int, hidden_dim: int = 412, kernel_size: int = 5, dilation: int = 1):
        super().__init__()
        padding = (kernel_size // 2) * dilation
        self.dw = nn.Conv1d(dim, dim, kernel_size, padding=padding, dilation=dilation, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pw1 = nn.Conv1d(dim, hidden_dim, kernel_size=1)
        self.act = nn.GELU()
        self.pw2 = nn.Conv1d(hidden_dim, dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        residual = x
        x = x.transpose(1, 2)            # (B, C, T)
        x = self.dw(x)
        x = x.transpose(1, 2)            # (B, T, C)
        x = self.norm(x)
        x = x.transpose(1, 2)
        x = self.pw1(x)
        x = self.act(x)
        x = self.pw2(x)
        x = x.transpose(1, 2)
        return x + residual