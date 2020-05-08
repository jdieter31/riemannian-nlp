from .config.config_loader import get_config
import torch

def get_device():
    """
    Returns the main device that experiments are to be ran on. Note that some
    components can and should still run on cpu for memory reasons so this
    should only be used to retrieve devices for things that would potentially
    run on the gpu.
    """
    general_config = get_config().general
    gpu = general_config.gpu
    device = torch.device(f'cuda:{gpu}' if gpu >= 0 else 'cpu')
    return device


    
