"""
Funkcje pomocnicze (utilities)
"""

import torch
import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Wczytuje plik konfiguracyjny YAML.
    
    Args:
        config_path: Ścieżka do pliku config.yaml
        
    Returns:
        Słownik z konfiguracją
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed: int = 42):
    """
    Ustawia seed dla reprodukowalności.
    
    Args:
        seed: Wartość seed
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import numpy as np
    np.random.seed(seed)
    import random
    random.seed(seed)
    

def save_checkpoint(model, optimizer, epoch, loss, path):
    """
    Zapisuje checkpoint modelu.
    
    Args:
        model: Model PyTorch
        optimizer: Optimizer
        epoch: Numer epoki
        loss: Wartość straty
        path: Ścieżka zapisu
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)
    

def load_checkpoint(model, optimizer, path):
    """
    Wczytuje checkpoint modelu.
    
    Args:
        model: Model PyTorch
        optimizer: Optimizer
        path: Ścieżka do checkpointa
        
    Returns:
        (epoch, loss): Numer epoki i wartość straty
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['loss']
