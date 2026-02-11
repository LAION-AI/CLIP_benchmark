"""
Utility functions shared across metrics modules.
"""
import torch


def to_device(x, device):
    """
    Recursively move data to device, handling tensors, dicts, lists, and tuples.
    
    Args:
        x: Data to move (tensor, dict, list, tuple, or other)
        device: Target device (e.g., 'cuda', 'cpu')
        
    Returns:
        Data moved to the specified device
    """
    if isinstance(x, torch.Tensor):
        return x.to(device)
    elif isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    elif isinstance(x, list):
        return [to_device(v, device) for v in x]
    elif isinstance(x, tuple):
        return tuple([to_device(v, device) for v in x])
    return x
