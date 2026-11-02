import math
import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int = 64, num_heads: int = 4, attn_dropout: float = 0.1, proj_dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj_drop = nn.Dropout(proj_dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None, return_attn: bool = False):
        B, n, _ = x.shape

        Q = self.W_q(x).view(B, n, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, n, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, n, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores + mask

        attn_weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(self.attn_drop(attn_weights), V)
        context = context.transpose(1, 2).contiguous().view(B, n, self.d_model)
        out = self.proj_drop(self.W_o(context))

        if return_attn:
            return out, attn_weights
        return out


class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model: int = 64, num_heads: int = 4, d_ff: int = 256, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.msa = MultiHeadSelfAttention(d_model, num_heads, dropout, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        if return_attn:
            msa_out, attn = self.msa(self.norm1(x), return_attn=True)
            x = x + msa_out
            x = x + self.ffn(self.norm2(x))
            return x, attn
        x = x + self.msa(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x
