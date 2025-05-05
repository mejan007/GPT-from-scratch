import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias = False):
        
        super().__init__()
        assert(d_out % num_heads == 0), \
        "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias = qkv_bias)

        self.out_proj = nn.Linear(d_out, d_out)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask",
                             torch.triu(torch.ones(context_length, context_length), diagonal = 1))
        

    def forward(self, x):
        ''' Step 1'''
        # Start with input
        b, num_tokens, d_in = x.shape

        ''' Step 2'''
        # Initialize trainable weight matrices for key, query and value (in constrcutor actually)
        # Their dimensions are (d_in, d_out)
        # Multiply the input with these matrices to get the keys, queries and values
        # The output of the multiplication will be of shape (b, num_tokens, d_out)    
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # So, each row represents a token, and each column represents a dimension of the embedding

        ''' Step 3'''
        # Unroll the last dimension of the keys, queries and values to include num_heads and head_dim
        # Since, head_dim = d_out // num_heads i.e. d_out = num_heads * head_dim
        # So, we reshape the last dimension of the keys, queries and values to (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        ''' Step 4'''
        # Currently, these matrices are grouped by Number of tokens
        # But we want to group them by number of heads (For computing attention scores for each heads separately)
        # So, we transpose the dimensions to get the shape (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        ''' Step 5'''
        # Compute the attention scores by multiplying queris and keys
        # This involves another transpose of keys so that matrix multiplication is possible
        # Similar to implementation in causal attention but this time we dimensions to transpose keys is (2, 3)

        attn_scores = queries @ keys.transpose(2, 3)

        # This ensures that the attention scores are computed for each head separately
        # So, the shape of attn_scores is (b, num_heads, num_tokens, num_tokens)
        # This makes sense as attenttion scores are computed for each token with respect to all other tokens
        
        ''' Step 6'''
        # Mask the attention scores to ignore future tokens
        attn_scores.masked_fill(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf
        )

        ''' Step 7'''
        # Compute the attention weights by applying softmax to the attention scores
        # Before applying softmax, we divide the attention scores by the square root of the head_dim (key ko last dim bhanekei tei ho)
        # This is done to prevent the softmax from being too large or too small

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]** 0.5, dim = -1)
        attn_weights = self.dropout(attn_weights)

        ''' Step 8'''
        # Compute the context vectors by multiplying the attention weights with the values
        # The shape of context_vecs is (b, num_heads, num_tokens, head_dim)
        
        # BUT we need the context vectors to be of shape (b, num_tokens, d_out)
        
        context_vec = (attn_weights @ values).transpose(1,2)
        # So, we need to transpose the context vectors to get the shape (b, num_tokens, num_heads, head_dim)
        # Then we need to reshape the context vectors to get the shape (b, num_tokens, d_out)
        
        ''' Step 9'''
        # Combine the context vectors from all heads
        # We need contiguous to ensure that the memory is contiguous
        context_vec = context_vec.contiguous().view(
            b, num_tokens, self.d_out
        )

        ''' Optional Projection'''
        context_vec = self.out_proj(context_vec)
        return context_vec