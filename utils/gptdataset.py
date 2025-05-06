import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken
class GPTDatasetV1(Dataset):

    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)

        # Loop over the token_ids to create input-target pairs with a sliding window
        # - Start at index 0
        # - Stop at len(token_ids) - max_length to ensure each input chunk has max_length tokens
        # - Increment by stride to control overlap between consecutive chunks

        '''
        The loop condition for i in range(0, len(token_ids) - max_length, stride) 
        ensures that we can extract valid input-target pairs of length 
        max_length without going out of bounds in the token_ids list.
        '''
        for i in range(0, len(token_ids) - max_length, stride):
            # Extract input chunk of max_length tokens starting at index i
            input_chunk = token_ids[i: i + max_length]

            # Extract target chunk, which is shifted by 1 token (for autoregressive prediction)
            # Starts at index i+1 and has the same length as input_chunk
            target_chunk = token_ids[i + 1: i + max_length + 1]

            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
    

def create_dataloader_v1(txt, batch_size=4, max_length=256,
 stride=128, shuffle=True, drop_last=True,
 num_workers=1):
    
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle = shuffle,
        drop_last = drop_last,
        num_workers = num_workers
    )

    return dataloader