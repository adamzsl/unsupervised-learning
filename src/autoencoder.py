"""Prosty autoenkoder konwolucyjny do budowy reprezentacji."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import torch
from torch import nn


@dataclass
class AutoencoderConfig:
    input_channels: int = 3
    latent_dim: int = 256
    image_size: int = 256


class ConvAutoencoder(nn.Module):
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(config.input_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        with torch.no_grad():
            dummy = torch.zeros(1, config.input_channels, config.image_size, config.image_size)
            conv_out = self.encoder_conv(dummy)
        self._conv_shape = conv_out.shape[1:]
        self._conv_flat_dim = int(conv_out.numel())
        self.encoder_out = nn.Linear(self._conv_shape[0], config.latent_dim)
        self.decoder_in = nn.Linear(config.latent_dim, self._conv_flat_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self._conv_shape[0], 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, config.input_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder_conv(x)
        pooled = self.global_pool(features).flatten(1)
        return self.encoder_out(pooled)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        decoded = self.decoder_in(z).view(-1, *self._conv_shape)
        return self.decoder(decoded)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z


@torch.no_grad()
def extract_embeddings(
    model: ConvAutoencoder, dataloader: Iterable, device: torch.device
) -> torch.Tensor:
    """Ekstrahuje embeddingi dla całego dataloadera."""
    model.eval()
    embeddings = []
    for batch in dataloader:
        images = batch[0].to(device)
        _, z = model(images)
        embeddings.append(z.cpu())
    return torch.cat(embeddings, dim=0)
