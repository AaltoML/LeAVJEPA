import torch
import torch.nn as nn

from transformer import Block


class CrossAttentionBlock(nn.Module):

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: int = 4):
        super().__init__()
        self.norm_q = nn.LayerNorm(hidden_size)
        self.norm_kv = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm_mlp = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_ratio * hidden_size),
            nn.GELU(),
            nn.Linear(mlp_ratio * hidden_size, hidden_size),
        )

    def forward(self, q, x):
        kv = self.norm_kv(x)
        out, _ = self.attn(self.norm_q(q), kv, kv, need_weights=False)
        q = q + out
        q = q + self.mlp(self.norm_mlp(q))
        return q


class VJEPAProbe(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, num_classes: int,
                 depth: int = 4, mlp_ratio: int = 4):
        super().__init__()
        cfg = {
            "hidden_size": hidden_size,
            "num_attention_heads": num_heads,
            "intermediate_size": mlp_ratio * hidden_size,
            "attention_probs_dropout_prob": 0.0,
            "hidden_dropout_prob": 0.0,
            "qkv_bias": True,
        }
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(depth - 1)])
        self.query = nn.Parameter(torch.zeros(1, 1, hidden_size))
        nn.init.trunc_normal_(self.query, std=0.02)
        self.cross = CrossAttentionBlock(hidden_size, num_heads, mlp_ratio)
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, num_classes)

    def forward(self, tokens):
        x = tokens
        for blk in self.blocks:
            x = blk(x)
        q = self.query.expand(x.size(0), -1, -1)
        q = self.cross(q, x).squeeze(1)
        return self.head(self.norm(q))
