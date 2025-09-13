import torch
import torch.nn as nn
from model.convnext import ConvNeXt1DBlock
from model.transformer import TransformerBlock, CrossAttention
from typing import Tuple, Optional


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

        self.sa_blocks = nn.ModuleList(
            [TransformerBlock(emb_dim, num_heads=sa_heads, ff_hidden=sa_ff) for _ in range(num_sa_blocks)]
        )

        # Learnable reference keys (used in the 1st cross-attention)
        self.learned_ref_keys = nn.Parameter(torch.randn(learned_key_tokens, emb_dim) * 0.02)

        self.cross_attn = nn.ModuleList(
            [CrossAttention(q_dim=emb_dim, kv_dim=emb_dim, num_heads=sa_heads) for _ in range(num_cross)]
        )

    def forward(
        self,
        text_ids: torch.Tensor,                 # (B, T_text)
        ref_value: torch.Tensor,               # (B, T_ref, 128) from Reference Encoder ("Ref. Value")
        ref_key: Optional[torch.Tensor] = None # (B, T_key, 128) optional; if None, use learned 50 keys
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          text_emb: (B, T_text, 128) speaker-adaptive text embeddings
          learned_keys: (50, 128) the learnable key vectors (for reuse in VF estimator)
        """
        B, T = text_ids.shape
        x = self.embed(text_ids)  # (B, T, 128)

        # 6x ConvNeXt
        for blk in self.convnext:
            x = blk(x)

        # 4x Self-Attention (Transformer) with RoPE
        for blk in self.sa_blocks:
            x = blk(x)

        # 1st Cross-Attention: keys = 50 learnable vectors (broadcast to batch), values = ref_value
        learned_k = self.learned_ref_keys.unsqueeze(0).expand(B, -1, -1)  # (B, 50, 128)
        x = self.cross_attn[0](x, learned_k, ref_value)

        # 2nd Cross-Attention: if ref_key is provided, use it; else reuse learned_k
        k2 = ref_key if ref_key is not None else learned_k
        v2 = ref_value
        x = self.cross_attn[1](x, k2, v2)

        return x, self.learned_ref_keys