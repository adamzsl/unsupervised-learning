"""Inference helper for super-resolution."""
from __future__ import annotations

import torch

from .model import SimpleSRCNN


def super_resolve(model: SimpleSRCNN, image: torch.Tensor) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        output = model(image)
    return output
