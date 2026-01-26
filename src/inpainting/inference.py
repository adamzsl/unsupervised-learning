"""Inference helper for inpainting."""
from __future__ import annotations

import torch

from .model import SimpleUNet


def inpaint_image(model: SimpleUNet, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        output = model(image, mask)
    return output
