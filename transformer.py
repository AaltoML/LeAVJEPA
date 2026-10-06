import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_attention_heads = config["num_attention_heads"]
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        qkv_bias = config["qkv_bias"]
        self.query = nn.Linear(self.hidden_size, self.all_head_size, bias=qkv_bias)
        self.key = nn.Linear(self.hidden_size, self.all_head_size, bias=qkv_bias)
        self.value = nn.Linear(self.hidden_size, self.all_head_size, bias=qkv_bias)

        self.attn_dropout_prob = config["attention_probs_dropout_prob"]
        self.output_projection = nn.Linear(self.all_head_size, self.hidden_size)
        self.output_dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        def heads(t):
            return t.view(
                batch_size, seq_len, self.num_attention_heads, self.attention_head_size
            ).transpose(1, 2)

        q, k, v = heads(self.query(x)), heads(self.key(x)), heads(self.value(x))

        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_dropout_prob if self.training else 0.0
        )
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.all_head_size)
        return self.output_dropout(self.output_projection(out))


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense_1 = nn.Linear(config["hidden_size"], config["intermediate_size"])
        self.activation = nn.GELU()
        self.dense_2 = nn.Linear(config["intermediate_size"], config["hidden_size"])
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x):
        return self.dropout(self.dense_2(self.activation(self.dense_1(x))))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.layernorm_1 = nn.LayerNorm(config["hidden_size"])
        self.mlp = MLP(config)
        self.layernorm_2 = nn.LayerNorm(config["hidden_size"])

    def forward(self, x):
        x = x + self.attention(self.layernorm_1(x))
        x = x + self.mlp(self.layernorm_2(x))
        return x


class Encoder(nn.Module):
    def __init__(self, config, gradient_checkpointing=False):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        self.blocks = nn.ModuleList([Block(config) for _ in range(config["num_hidden_layers"])])

    def forward(self, x):
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = grad_checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return x
