"""Narzędzia treningowe dla super-resolution."""
from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from .model import SimpleSRCNN


def superres_step(model: SimpleSRCNN, batch, device: torch.device) -> Dict[str, float]:
    lr, hr = batch
    lr = lr.to(device)
    hr = hr.to(device)
    output = model(lr)
    loss = nn.functional.mse_loss(output, hr)
    return {"loss": loss}
