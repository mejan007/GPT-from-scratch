import torch
import torch.nn as nn

from utils.transformer import TransformerBlock
from utils.layernorm import LayerNorm

# class GPTModel(nn.Module):
    
#     def __init__(self, config):

#         super().__init__()

#         self.tok_emb = nn.Embedding(config["vocab_size"], config["emb_dim"])
#         # for gpt - 2 small, vocab_size = 50257 and emb_dim = 768
#         # This matrix is also called lookup table we pass in the token id and get the corresponding embedding
#         # Also a trainable parameter

#         self.pos_emb = nn.Embedding(config["context_length"], config["emb_dim"])

#         self.drop_emb = nn.Dropout(config["drop_rate"])


#         # This is a sequential container that contains n_layers transformer blocks
#         self.trf_blocks = nn.Sequential(
#             *[TransformerBlock(config) for _ in range(config["n_layers"])]
#         )
#         ''' The * unpacks the list so that nn.Sequential receives block1, block2, block3 as separate arguments, matching its expected input.'''
#         # Placeholder for the LayerNorm
#         self.final_norm = LayerNorm(config["emb_dim"])
        
#         self.out_head = nn.Linear(config["emb_dim"], config["vocab_size"], bias=False)




#     def forward(self, in_idx):
#         # in_idx is the input token ids
#         batch_size, seq_len = in_idx.shape
        
#         # What we are doing here is looking up the token ids in token embedding matrix created in the constructor 
#         # to get the 768 embedding vectors for each token id of the input sequence
#         # So, if one batch has 4 tokens, for each batch we will get 4 embedding vectors of size 768
#         tok_embeds = self.tok_emb(in_idx) 
#         ''' For each token ID, nn.Embedding retrieves a vector of shape (emb_dim,), so tok_embeds has shape (batch_size, seq_len, emb_dim).'''


#         # We are creating a position embedding matrix of size (seq_len, emb_dim)
#         # What arange is doing is creating a tensor of size (seq_len) with values from 0 to seq_len - 1
#         # And then we are passing this tensor to position embedding to get only that many position embeddings
#         pos_embeds = self.pos_emb(torch.arange(seq_len, device = in_idx.device))
#         '''
#         pos_embeds is not the entire positional embedding matrix. It’s a tensor of shape (seq_len, emb_dim) containing only the embeddings for the positions [0, 1, ..., seq_len-1]. Only seq_len rows of the (context_length, emb_dim) matrix are used.
#         '''
#         # We are adding the token embeddings and position embeddings (both 768 dimensional vectors)
#         # This is the final embedding matrix we will pass to the transformer
#         x = tok_embeds + pos_embeds

#         '''
#         Here, pos_embeds (shape (seq_len, emb_dim)) is broadcasted to match tok_embeds (shape (batch_size, seq_len, emb_dim)).
#         Broadcasting replicates pos_embeds across the batch_size dimension, effectively treating it as (batch_size, seq_len, emb_dim) for the addition.

#         The result x has shape (batch_size, seq_len, emb_dim).
#         '''

#         # We are applying dropout to the embedding matrix which is a regularization technique
#         x = self.drop_emb(x)

#         x = self.trf_blocks(x)  # (batch_size, num_tokens, emb_dim)
        
#         x = self.final_norm(x)  # (batch_size, num_tokens, emb_dim)

#         logits = self.out_head(x)
#         # This will output a tensor of shape (batch_size, num_tokens, vocab_size)
#         return logits


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
