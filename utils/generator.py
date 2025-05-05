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

    