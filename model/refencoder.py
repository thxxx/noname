import torch
import torch.nn as nn
import torch.nn.functional as F
from model.convnext import ConvNeXt1DBlock
from model.transformer import CrossAttention

# -----------------------------
# Reference Encoder (A.2.1) w/ SDPA
# -----------------------------
class ReferenceEncoder(nn.Module):
    """
    입력 : (B, T, 144)
    구조 : Linear(144→128) → ConvNeXt×6(k=5, hidden=512) → Cross-Attn×2 (SDPA)
    - Cross-Attn #1: 학습 가능한 쿼리 50개(128차원)가 원본 시퀀스에 어텐션 → (B, 50, 128)
    - Cross-Attn #2: 위 (B,50,128)을 쿼리로 다시 원본 시퀀스에 어텐션(정제)
    출력 : reference_value (B, 50, 128)
    """
    def __init__(self, in_dim: int = 100, hidden_dim: int = 128,
                 conv_hidden: int = 512, num_conv: int = 6,
                 num_queries: int = 50, num_heads: int = 1):
        super().__init__()
        assert hidden_dim == 128
        self.in_proj = nn.Linear(in_dim, hidden_dim)

        self.conv_blocks = nn.ModuleList([
            ConvNeXt1DBlock(dim=hidden_dim, hidden_dim=conv_hidden, kernel_size=5)
            for _ in range(num_conv)
        ])

        # 학습 쿼리 (고정 길이 50)
        self.learned_queries = nn.Parameter(torch.randn(num_queries, hidden_dim) * 0.02)

        # SDPA 기반 Cross-Attn 2개
        self.cross1 = CrossAttention(q_dim=hidden_dim, kv_dim=hidden_dim, num_heads=num_heads, dropout=0.0)
        self.cross2 = CrossAttention(q_dim=hidden_dim, kv_dim=hidden_dim, num_heads=num_heads, dropout=0.0)

        # 안정화용 LayerNorm
        self.pre_ca_norm = nn.LayerNorm(hidden_dim)
        self.post_ca1_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, 100)
        return: (B, 50, 128)
        """
        B, T, _ = x.shape

        x = self.in_proj(x)  # (B, T, 128)

        # ConvNeXt × 6
        for blk in self.conv_blocks:
            x = blk(x)

        x = self.pre_ca_norm(x) # (B, T, 128)

        # Cross-Attn #1 (학습 쿼리 50개)
        q1 = self.learned_queries.unsqueeze(0).expand(B, -1, -1)  # (B, 50, 128)
        ref_50 = self.cross1(q1, x, x)   # (B, 50, 128)
        ref_50 = self.post_ca1_norm(ref_50)

        # Cross-Attn #2 (정제)
        ref_50 = self.cross2(ref_50, x, x)  # (B, 50, 128)

        return ref_50
