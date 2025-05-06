import torch


def calc_loss_batch(input_batch, target_batch, model, device):

    input_batch = input_batch.to(device)
        
    target_batch = target_batch.to(device) # Input shifted by 1 token 

    logits = model(input_batch)
    # logits shape: (batch_size, nmax_length_tokens, vocab_size)
    # target_batch shape: (batch_size, nmax_length_tokens)

    # Flatten the logits and target_batch to compute the loss
    # logits shape: (batch_size * nmax_length_tokens, vocab_size)
    loss = torch.nn.functional.cross_entropy(
            logits.flatten(0, 1), target_batch.flatten()
    )

    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):

    total_loss = 0.

    
    if len(data_loader) == 0:
        # Data loader being empty means error in the dataset
        return float("nan")
        
    elif num_batches is None:
        # If num_batches is None, use the entire data loader
        # For example, in our case for train loader, it would be 9
        # and for val loader it would be 1
        num_batches = len(data_loader) 

    else:
        # If num_batches is specified, use the minimum of num_batches and the length of data_loader

        # Reduces the number of batches to match the total number of batches in the data
        # loader if num_batches exceeds the number of batches in the data loader

        num_batches = min(num_batches, len(data_loader)) 

    for i, (input_batch, target_batch) in enumerate(data_loader):

        if i < num_batches:
            # Calculate the loss for each batch
            loss = calc_loss_batch(
                    input_batch, target_batch, model, device
            )
            # Sum the loss for each batch
            total_loss += loss.item() 
        else:
            break

    return total_loss / num_batches