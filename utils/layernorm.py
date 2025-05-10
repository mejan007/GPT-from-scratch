
import torch 
import torch.nn as nn
# class LayerNorm(nn.Module):
#     def __init__(self, emb_dim):
#         super().__init__()
#         self.eps = 1e-5 # epsilon to avoid division by zero
        
#         # Two trainable parameters scale and shift are initialized to 1 and 0 respectively
#         self.scale = nn.Parameter(torch.ones(emb_dim))
#         self.shift = nn.Parameter(torch.zeros(emb_dim))

#     def forward(self, x):
#         mean = x.mean(dim = -1, keepdim = True)
#         var = x.var(dim = -1, keepdim = True, unbiased = False)
#         norm_x = (x - mean) / torch.sqrt(var + self.eps)
        
#         return self.scale * norm_x + self.shift

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift
    