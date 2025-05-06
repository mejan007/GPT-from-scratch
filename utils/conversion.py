import torch

def text_to_token_ids(text, tokenizer):
    """
    Convert text to token ids using the tokenizer.
    """
    encoded = tokenizer.encode(text, allowed_special = {'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # Add batch dimension
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    """
    Convert token ids to text using the tokenizer.
    """
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())