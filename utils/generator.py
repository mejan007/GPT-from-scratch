import torch

def generate_text_simple(model, idx, max_new_tokens, context_size):
    """
    Generate text iteratively using the model until:
    number of iterations = max_new_tokens.
    """
    # Here, model is an instance of GPTModel. 
    # idx is the input text tokenized and converted to tensor
    # and max_new_tokens is the number of tokens to generate.
    # and context_size is the size of the context window (how many tokens model has to process).

    # idx example:
    # tensor([[6109, 3626, 6100,  345],
        # [6109, 1110, 6622,  257]])
    
    # idx shape = (batch, n_tokens)
    
    for _ in range(max_new_tokens):

        '''
        Crops current context if it exceeds the supported context size, 
        e.g., if LLM supports only 5 tokens, and the context size is 10, 
        then only the last 5 tokens are used as context 
        '''
        idx_cond = idx[:, -context_size:]
        # idx_cond is the last context_size tokens of idx
        # idx_cond example:
        '''
        idx = tensor([[10, 23, 45, 67, 89, 123, 56, 78], # First sequence/batch
                    [9, 8, 7, 6, 5, 4, 3, 2]]) # Second sequence/batch
        
        If context_size = 6 then:
        idx_cond = tensor([[45, 67, 89, 123, 56, 78], # First sequence/batch
                           [6, 5, 4, 3, 2]])
        '''

        # Pass the current input to the model and get the logits
        with torch.no_grad(): 
            # No need to compute gradients for inference
            logits = model(idx_cond)
        
        # The shape of logits is (batch_size, context_size, vocab_size)
        # Each logit vector represents scores for all possible tokens in the vocabulary for each position in the input sequence.
        
        # Now, we extract the logits for the last position of each sequence
        # This is because, in autoregressive generation, we’re only interested in predicting the next token after the current sequence.
        # So, (batch_size, context_size, vocab_size) -> (batch_size, vocab_size)
        logits = logits[:, -1, :]

        # Applying softmax to get the probabilities of the next token
        probabs = torch.softmax(logits, dim = -1)  # (batch_size, vocab_size)

        # Get the index of the next token with the highest probability for each batch
        idx_next = torch.argmax(probabs, dim = -1, keepdim=True) # (batch_size, 1)

        # Append the predicted token to the input sequence

        idx = torch.cat((idx, idx_next), dim = 1) # (batch_size, n_tokens + 1)

    return idx


# def generate(model, idx, max_new_tokens, context_size,
#              temperature = 0.0, top_k = None, eos_id = None):
#     """
#     Generate text iteratively using the model applying temperature scaling, 
#     top-k sampling and multinomial sampling until:
#     number of iterations = max_new_tokens.
#     """
#     # Here, model is an instance of GPTModel. 
#     # idx is the input text tokenized and converted to tensor
#     # and max_new_tokens is the number of tokens to generate.
#     # and context_size is the size of the context window (how many tokens model has to process).

#     # idx example:
#     # tensor([[6109, 3626, 6100,  345],
#         # [6109, 1110, 6622,  257]])
    
#     # idx shape = (batch, n_tokens)
    
#     for _ in range(max_new_tokens):

#         '''
#         Crops current context if it exceeds the supported context size, 
#         e.g., if LLM supports only 5 tokens, and the context size is 10, 
#         then only the last 5 tokens are used as context 
#         '''
#         idx_cond = idx[:, -context_size:]
#         # idx_cond is the last context_size tokens of idx
#         # idx_cond example:
#         '''
#         # idx = tensor([[10, 23, 45, 67, 89, 123, 56, 78], # First sequence/batch
#                     [9, 8, 7, 6, 5, 4, 3, 2]]) # Second sequence/batch
        
#         If context_size = 6 then:
#         # idx_cond = tensor([[45, 67, 89, 123, 56, 78], # First sequence/batch
#                            [6, 5, 4, 3, 2]])
#         '''

#         # Pass the current input to the model and get the logits
#         with torch.no_grad(): 
#             # No need to compute gradients for inference
#             logits = model(idx_cond)
        
#         # The shape of logits is (batch_size, context_size, vocab_size)
#         # Each logit vector represents scores for all possible tokens in the vocabulary for each position in the input sequence.
        
#         # Now, we extract the logits for the last position of each sequence
#         # This is because, in autoregressive generation, we’re only interested in predicting the next token after the current sequence.
#         # So, (batch_size, context_size, vocab_size) -> (batch_size, vocab_size)
#         logits = logits[:, -1, :]

#         '''Modifications'''

#         if top_k is not None:
#             top_logits, _ = torch.topk(logits, top_k)
            
#             min_val = top_logits[:, -1]

#             logits = torch.where(
#                 logits < min_val,
#                 torch.tensor(float('-inf')).to(logits.device),
#                 logits
#             )
        
#         if temperature > 0.0:

#             logits = logits / temperature
#             probs = torch.softmax(logits, dim = -1)
#             idx_next = torch.multinomial(probs, num_samples=1)

#         else:

#             idx_next = torch.argmax(logits, dim=-1, keepdim=True)

#         if idx_next == eos_id:
#             break
#         # Append the predicted token to the input sequence

#         idx = torch.cat((idx, idx_next), dim = 1) # (batch_size, n_tokens + 1)

#     return idx


def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):

    # For-loop is the same as before: Get logits, and only focus on last time step
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        # New: Filter logits with top_k sampling
        if top_k is not None:
            # Keep only top_k values
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)

        # New: Apply temperature scaling
        if temperature > 0.0:
            logits = logits / temperature

            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)

            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

        # Otherwise same as before: get idx of the vocab entry with the highest logits value
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

        if idx_next == eos_id:  # Stop generating early if end-of-sequence token is encountered and eos_id is specified
            break

        # Same as before: append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)

    return idx
    