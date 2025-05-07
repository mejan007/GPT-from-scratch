import os
import torch


def save_model(model, model_save_name, optimizer = None, save_dir_name = 'models'):
    '''
    Saves the model's state dictionary to a directory name and file name 
    (creates the save_dir as an immediate sub-directory of project's root directory).
    model: The model to be saved.

    model_save_name: The name of the file where the model will be saved.

    optimizer: The optimizer to be saved. By default, it is None.
    
    save_dir_name: The name of the directory where the model will be saved. By default, it is 'models'.


    Optimizer is not saved by default, but it can be saved if needed (to resume training).

    Raises:
        FileExistsError: If the model file already exists.
    
    '''


    cwd_path = os.getcwd()
    # print(cwd_path)

    parent_path = os.path.join(cwd_path, '..')
    # This is a relative path that conceptually points to the 
    # parent directory of chap_5, which is LLM


    relative_save_dir_path = os.path.join(parent_path, save_dir_name) # This is still a relative path with .. in it.

    save_dir_path = os.path.abspath(relative_save_dir_path) # This resolves the relative path to an absolute path.
    # Ensure the directory exists
    os.makedirs(save_dir_path, exist_ok=True)

    save_path = os.path.join(save_dir_path, model_save_name)

    # Save the model state dictionary only if it doesn't already exist
    if os.path.exists(save_path):
        raise FileExistsError(f"Model already exists at {save_path}. \nPlease use a different name.")

    if optimizer is not None:
        # Save the model and optimizer state dictionaries
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, save_path)
        
    else:
        torch.save(model.state_dict(), save_path)



def load_model(model_obj, model_config, model_name, device, optimizer=None, load_dir_name='models'):
    '''
    Loads the model's state dictionary (and optionally the optimizer's) 
    from a given directory and file name.

    model_obj: The GPT model instance whose state will be loaded.
    
    model_config: The GPT configuration dictionary for the model. This is used to initialize the model.
    
    model_name: The name of the file where the model is saved.

    device: The device (CPU or GPU) on which the model will be loaded.

    optimizer: (Optional) The optimizer instance whose state will be loaded.
    
    load_dir_name: The directory name where the model is stored. Default is 'models'.

    Returns:
        model: The model with loaded state dict. (returns in evaluation mode if no optimizer is provided)
        optimizer: The optimizer with loaded state dict (if provided, model returns in train mode).
    '''

    cwd_path = os.getcwd()
    parent_path = os.path.join(cwd_path, '..')
    relative_load_dir_path = os.path.join(parent_path, load_dir_name)
    load_dir_path = os.path.abspath(relative_load_dir_path)
    load_path = os.path.join(load_dir_path, model_name)

    if not os.path.exists(load_path):
        raise FileNotFoundError(f"No model found at {load_path}")

    if optimizer is not None:
        # Load the model and optimizer state dictionaries
        checkpoint = torch.load(load_path, map_location = device)
        # Initialize the model with the provided configuration
        model = model_obj(model_config)

        # Load the model and optimizer state dictionaries
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        model.train() # Set the model to training mode
        return model, optimizer
    else:
        # Initialize the model with the provided configuration
        model = model_obj(model_config)
        # Load the model state dictionary
        model.load_state_dict(torch.load(load_path, map_location=device))

        model.eval() # Set the model to evaluation mode
        return model

