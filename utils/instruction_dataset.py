import torch
from torch.utils.data import Dataset, DataLoader

def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    
    input_text = (
        f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    )

    return instruction_text + input_text


class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):

        self.data = data
        self.encoded_texts = []
        
        for entry in data: 
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"

            
            full_text = instruction_plus_input + response_text # Alpaca-style formatting is complete

            # Tokenize the full input/output text 
            # and append each input/output pair to the list
            self.encoded_texts.append(
                tokenizer.encode(full_text)
            )
 
    def __getitem__(self, index):
        return self.encoded_texts[index]
 
    def __len__(self):
        return len(self.data)
    

def custom_collate_fn(batch, pad_token_id=50256, ignore_index = -100, 
                      allowed_max_length=None, device="cpu"):

    # The +1 is necessary because the function appends an extra token (the pad_token_id) 
    # to each sequence before padding and then constructs inputs and targets with a one-token shift.
    batch_max_length = max(len(item)+1 for item in batch) 

    inputs_lst, targets_lst = [], []

    for item in batch: 

        new_item = item.copy()

        new_item += [pad_token_id]

        
        padded = (
        new_item + [pad_token_id] * 
        (batch_max_length - len(new_item))
        )
        # Removes extra padded token
        # Truncates the last token for inputs
        inputs = torch.tensor(padded[:-1])
        # Shifts +1 to the right for targets
        targets = torch.tensor(padded[1:])

        # Replace all but the first occurrence of the pad_token_id with ignore_index
        mask = targets == pad_token_id 
        indices = torch.nonzero(mask).squeeze() 

        if indices.numel() > 1: 
            targets[indices[1:]] = ignore_index 
        
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length] 
            targets = targets[:allowed_max_length] 

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device) 
    
    return inputs_tensor, targets_tensor

def create_dataloader(dataset, batch_size, custom_collate_func, shuffle=True, drop_last = True, num_workers=0):
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=custom_collate_func,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )