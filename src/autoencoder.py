"""Prosty autoenkoder konwolucyjny do budowy reprezentacji."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import torch
from torch import nn


@dataclass
class AutoencoderConfig:
    input_channels: int = 3
    latent_dim: int = 128
    image_size: int = 256
    base_channels: int | None = None
    use_dropout: bool = False


class ConvAutoencoder(nn.Module):
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        base = config.base_channels or max(8, config.latent_dim // 8)
        self.use_dropout = config.use_dropout

        self.encoder1 = self._conv_block(config.input_channels, base)
        self.encoder2 = self._conv_block(base, base * 2, pool=True)
        self.encoder3 = self._conv_block(base * 2, base * 4, pool=True)
        self.encoder4 = self._conv_block(base * 4, base * 8, pool=True)

        self.bottleneck = self._conv_block(base * 8, base * 16, pool=True)

        self.decoder4 = self._upconv_block(base * 16, base * 8)
        self.decoder3 = self._upconv_block(base * 8, base * 4)
        self.decoder2 = self._upconv_block(base * 4, base * 2)
        self.decoder1 = self._upconv_block(base * 2, base)

        self.final_conv = nn.Sequential(
            nn.Conv2d(base, config.input_channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self._last_skips: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None

    def _conv_block(self, in_channels: int, out_channels: int, pool: bool = False) -> nn.Sequential:
        layers = []
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=2))
        layers.extend(
            [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_channels),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_channels),
            ]
        )
        if self.use_dropout:
            layers.append(nn.Dropout(0.5))
        return nn.Sequential(*layers)

    def _upconv_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
        )

    def _encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        b = self.bottleneck(e4)
        return e1, e2, e3, e4, b

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        e1, e2, e3, e4, b = self._encode(x)
        self._last_skips = (e1, e2, e3, e4)
        return b

    def _decode(
        self,
        z: torch.Tensor,
        skips: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        e1, e2, e3, e4 = skips
        d4 = self.decoder4(z) + e4
        d3 = self.decoder3(d4) + e3
        d2 = self.decoder2(d3) + e2
        d1 = self.decoder1(d2) + e1
        return self.final_conv(d1)

    def decode(
        self,
        z: torch.Tensor,
        skips: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if skips is None:
            if self._last_skips is None:
                raise ValueError("Call encode() before decode() or provide skip connections.")
            skips = self._last_skips
        return self._decode(z, skips)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        e1, e2, e3, e4, z = self._encode(x)
        recon = self._decode(z, (e1, e2, e3, e4))
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
