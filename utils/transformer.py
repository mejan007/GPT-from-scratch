import torch
import torch.nn as nn

from utils.attention import MultiHeadAttention
from utils.feedforward import FeedForward
from utils.layernorm import LayerNorm

# class TransformerBlock(nn.Module):

#     def __init__(self, config):
#         super().__init__()

#         self.att = MultiHeadAttention(
#             d_in = config["emb_dim"],
#             d_out = config["emb_dim"],
#             context_length = config["context_length"],
#             num_heads = config["n_heads"],
#             dropout = config["drop_rate"],
#             qkv_bias = config["qkv_bias"],
#         )

#         self.ff = FeedForward(config)

#         self.norm1 = LayerNorm(config["emb_dim"])

#         self.norm2 = LayerNorm(config["emb_dim"])

#         self.drop_shortcut = nn.Dropout(config["drop_rate"])

#     def forward(self, x):
        
#         shortcut = x # Shortcut connection for attention block
#         x = self.norm1(x) # Normalize the embedding vectors
#         x = self.att(x) # Get context vectors of Shape [batch_size, num_tokens, emb_dim] 
#         x = self.drop_shortcut(x) # dropout layer

#         x = x + shortcut # Add the original input back

#         shortcut = x # Shortcut connection for feedforward block
#         x = self.norm2(x)
#         x = self.ff(x)
#         x = self.drop_shortcut(x)
#         x = x + shortcut

#         return x

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)   # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed-forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        return x
