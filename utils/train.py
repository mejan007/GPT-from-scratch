import torch

from utils.loss import calc_loss_loader, calc_loss_batch
from utils.generator import generate_text_simple, generate
from utils.conversion import text_to_token_ids, token_ids_to_text

def evaluate_model(model, train_loader, val_loader, device, eval_iter):

    '''
    Evaluates the performance of a GPT-2 model by computing the average loss on a specified 
    number of batches from both the training and validation data loaders. This function is 
    used to assess the model's generalization during training without updating model parameters.  

    eval_iter: The number of batches to process from each loader to compute the average loss.
    '''

    # Set the model to evaluation mode, disabling dropout and other training-specific layers
    model.eval() 
    # Disables gradient tracking, which is not required during evaluation, to reduce
    # the computational overhead
    with torch.no_grad(): 
        # Calculate average training loss over a specified number of batches
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        # Calculate average validation loss over a specified number of batches
        val_loss = calc_loss_loader(
            val_loader, model, device, num_batches=eval_iter
        )
    # Switch the model back to training mode
    model.train()
    # Return the computed training and validation losses
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    
    '''
    Generates a sample text sequence from the GPT-2 model given an initial context 
    and prints it. This function is used to qualitatively assess the model's text 
    generation capabilities during training. 
    '''
    # Set the model to evaluation mode for consistent text generation
    model.eval()
    
    # Retrieve the context size from the model's positional embedding layer
    context_size = model.pos_emb.weight.shape[0]
    
    # Encode the start context into token IDs and move to the specified device
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    
    # Disable gradient computation for efficiency during generation
    with torch.no_grad():
        # Generate a sequence of new tokens based on the encoded context
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
        
        # Generate using a new implementation of generate function (using temperature scaling, top-k and multinomial sampling)
        # token_ids = generate(
        #     model=model,
        #     idx=encoded,
        #     max_new_tokens=50,
        #     context_size=context_size,
        #     top_k=40,
        #     temperature=1.4,
        # )
    # Decode the generated token IDs back into text
    decoded_text = token_ids_to_text(token_ids, tokenizer)

    # Print the generated text, replacing newlines with spaces for readability
    print(decoded_text.replace("\n", " ")) 

    # Switch the model back to training mode
    model.train()


def train_model_simple(model, train_loader, val_loader,
    optimizer, device, num_epochs,
    eval_freq, eval_iter, start_context, tokenizer):

    '''
     Trains a GPT-2 model using a simple training loop over a specified number of epochs. 
     It processes batches from the training data, updates model parameters, periodically 
     evaluates the model on training and validation sets, generates sample text, and tracks losses and tokens processed.
    '''
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], [] 
    tokens_seen, global_step = 0, -1
    
    # Training loop
    for epoch in range(num_epochs): 
        model.train()

        for input_batch, target_batch in train_loader:
            # Reset loss gradients from previous batch iteration
            optimizer.zero_grad() 

            # Compute the loss for the current batch
            loss = calc_loss_batch(
                    input_batch, target_batch, model, device
                )
            
            # Backpropagate the loss to compute gradients
            loss.backward() 

            # Update model parameters using the computed gradients
            optimizer.step()
            # Track the total number of tokens processed (based on input batch size)
            tokens_seen += input_batch.numel()

            # Track total number of training steps (batches processed)
            global_step += 1
        
            # Periodically evaluate the model based on the evaluation frequency 
            if global_step % eval_freq == 0: 
                # Compute training and validation losses
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                
                # Append losses and tokens seen to their respective tracking lists
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                
                # Print the current epoch, step, and loss values
                print(f"Epoch {epoch+1} (Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, "
                    f"Val loss {val_loss:.3f}"
                    )
                
        # Generate and print a sample text output after each epoch
        generate_and_print_sample( 
            model, tokenizer, device, start_context
        )

    return train_losses, val_losses, track_tokens_seen