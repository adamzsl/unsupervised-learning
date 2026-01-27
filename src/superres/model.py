"""Prosty model super-resolution (SRCNN-lite)."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class SuperResConfig:
    input_channels: int = 3
    base_channels: int = 32


class SimpleSRCNN(nn.Module):
    def __init__(self, config: SuperResConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(config.input_channels, config.base_channels, 9, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(config.base_channels, config.base_channels // 2, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(config.base_channels // 2, config.input_channels, 5, padding=2),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
