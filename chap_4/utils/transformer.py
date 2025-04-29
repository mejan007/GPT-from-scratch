import torch
import torch.nn as nn

from utils.attention import MultiHeadAttention
from utils.feedforward import FeedForward
from utils.layernorm import LayerNorm

class TransformerBlock(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.att = MultiHeadAttention(
            d_in = config["emb_dim"],
            d_out = config["emb_dim"],
            context_length = config["context_length"],
            num_heads = config["n_heads"],
            dropout = config["drop_rate"],
            qkv_bias = config["qkv_bias"],
        )

        self.ff = FeedForward(config)

        self.norm1 = LayerNorm(config["emb_dim"])

        self.norm2 = LayerNorm(config["emb_dim"])

        self.drop_shortcut = nn.Dropout(config["drop_rate"])

    def forward(self, x):
        
        shortcut = x # Shortcut connection for attention block
        x = self.norm1(x) # Normalize the embedding vectors
        x = self.att(x) # Get context vectors of Shape [batch_size, num_tokens, emb_dim] 
        x = self.drop_shortcut(x) # dropout layer

        x = x + shortcut # Add the original input back

        shortcut = x # Shortcut connection for feedforward block
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x

