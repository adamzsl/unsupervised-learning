"""Narzędzia treningowe dla inpaintingu."""
from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from .model import SimpleUNet


def inpaint_step(model: SimpleUNet, batch, device: torch.device) -> Dict[str, float]:
    images, masks, masked, _ = batch
    images = images.to(device)
    masks = masks.to(device)
    masked = masked.to(device)
    output = model(masked, masks)
    loss = nn.functional.l1_loss(output * masks, images * masks)
    return {"loss": loss}
