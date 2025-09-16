import torch
import torch.nn as nn
from model.convnext import ConvNeXt1DBlock
from model.transformer import TransformerBlock
from typing import Tuple


# --- TextEncoder (as specified in the paper) ---
class TextEncoder(nn.Module):
    """
    Character-level Text Encoder:
      - Embedding: 128-dim
      - 6 x ConvNeXt1DBlock (kernel=5, hidden=512)
      - 4 x TransformerBlock (heads=4, FF=512) with RoPE
      - 2 x CrossAttention
        * First cross-attention: K = 50 learnable vectors (128-d), V = reference value (from ref-encoder)
        * Second cross-attention: K,V = provided (defaults to same as first)
    """
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = 128,
        conv_hidden: int = 512,
        num_convnext: int = 6,
        num_sa_blocks: int = 4,
        sa_heads: int = 4,
        sa_ff: int = 512,
        num_cross: int = 2,
        learned_key_tokens: int = 50,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim)

        self.convnext = nn.ModuleList(
            [ConvNeXt1DBlock(emb_dim, hidden_dim=conv_hidden, kernel_size=5) for _ in range(num_convnext)]
        )

        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(emb_dim) for _ in range(num_sa_blocks)]
        )

        self.sa_blocks = nn.ModuleList(
            [TransformerBlock(emb_dim, num_heads=sa_heads, ff_hidden=sa_ff) for _ in range(num_sa_blocks)]
        )

    def forward(
        self,
        text_ids: torch.Tensor,                   # (B, T_text)
        text_attention_mask: torch.Tensor = None, # (B, T_text)
    ) -> Tuple[torch.Tensor]:
        """
        Returns:
          text_emb: (B, T_text, 128) speaker-adaptive text embeddings
          learned_keys: (50, 128) the learnable key vectors (for reuse in VF estimator)
        """
        x = self.embed(text_ids.int())  # (B, T, 128)

        # 6x ConvNeXt
        for blk in self.convnext:
            x = blk(x)

        # 4x Self-Attention (Transformer) with RoPE
        for blk, ln in zip(self.sa_blocks, self.layer_norms):
            x = x + blk(ln(x), text_attention_mask)

        return x