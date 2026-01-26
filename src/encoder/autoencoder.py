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


class ConvAutoencoder(nn.Module):
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(config.input_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        self.encoder_out = nn.Linear(128 * 32 * 32, config.latent_dim)
        self.decoder_in = nn.Linear(config.latent_dim, 128 * 32 * 32)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, config.input_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return self.encoder_out(features)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        decoded = self.decoder_in(z).view(-1, 128, 32, 32)
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
