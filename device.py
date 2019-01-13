import torch

def get_device(device=None):
    # Set device to GPU if available and cpu not explicity specified
    if device == None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device