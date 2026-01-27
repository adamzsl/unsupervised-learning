"""Prosty model inpainting oparty o U-Net."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class InpaintConfig:
    input_channels: int = 4  # obraz + maska
    base_channels: int = 32


class SimpleUNet(nn.Module):
    def __init__(self, config: InpaintConfig) -> None:
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(config.input_channels, config.base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(config.base_channels, config.base_channels * 2, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(config.base_channels * 2, config.base_channels * 4, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(config.base_channels * 4, config.base_channels * 2, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(config.base_channels * 2, config.base_channels, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.out = nn.Sequential(
            nn.Conv2d(config.base_channels, 3, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = torch.cat([image, mask], dim=1)
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        d2 = self.dec2(e3)
        d1 = self.dec1(d2)
        return self.out(d1)
