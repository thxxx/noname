import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from model.position import RotaryPositionalEmbedding, apply_rope
from einops import rearrange

class MultiHeadAttention(nn.Module):
    def __init__(self, q_dim, kv_dim=None, num_heads=8, use_rope=True):
        super().__init__()
        kv_dim = kv_dim or q_dim
        self.h = num_heads
        self.d = q_dim // num_heads
        assert q_dim % num_heads == 0 and self.d % 2 == 0

        self.q_proj = nn.Linear(q_dim, q_dim)
        self.k_proj = nn.Linear(kv_dim, q_dim)
        self.v_proj = nn.Linear(kv_dim, q_dim)
        self.out = nn.Linear(q_dim, q_dim)

        self.q_norm = nn.RMSNorm(self.d)
        self.k_norm = nn.RMSNorm(self.d)
        self.rope = RotaryPositionalEmbedding(self.d) if use_rope else None

    def forward(self, q_inp, kv_inp=None, attention_mask=None):
        if kv_inp is None: kv_inp = q_inp  # self-attn

        q = self.q_proj(q_inp)
        k = self.k_proj(kv_inp)
        v = self.v_proj(kv_inp)

        q = rearrange(q, "b t (h d) -> b h t d", h=self.h)
        k = rearrange(k, "b t (h d) -> b h t d", h=self.h)
        v = rearrange(v, "b t (h d) -> b h t d", h=self.h)

        q, k = self.q_norm(q), self.k_norm(k)
        q, k = apply_rope(q, k, self.rope)

        attn_mask = None if attention_mask is None else attention_mask[:, None, None, :]

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)  # (B,H,Tq,D)
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.out(out)

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=4, ff_hidden=512, ff_dropout=0.0, attn_dropout=0.0):
        super().__init__()
        self.attn = MultiHeadAttention(dim, num_heads=num_heads)
        self.ln1 = nn.RMSNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_hidden), 
            nn.GELU(), 
            nn.Linear(ff_hidden, dim)
        )
        self.ln2 = nn.RMSNorm(dim)
        self.ff_drop = nn.Dropout(ff_dropout)
        self.attn_drop = nn.Dropout(attn_dropout)

    def forward(self, x, context=None, attention_mask=None):
        x = x + self.attn_drop(self.attn(self.ln1(x), context, attention_mask))
        x = x + self.ff_drop(self.ff(self.ln2(x)))
        return x
