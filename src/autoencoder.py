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
        base = config.base_channels or max(32, config.latent_dim // 2)
        self.use_dropout = config.use_dropout

        self.enc1 = self._conv_block(config.input_channels, base)
        self.enc2 = self._conv_block(base, base * 2, down=True)
        self.enc3 = self._conv_block(base * 2, base * 4, down=True)
        self.enc4 = self._conv_block(base * 4, config.latent_dim, down=True)

        self.dec3 = self._upconv_block(config.latent_dim, base * 4)
        self.dec2 = self._upconv_block(base * 4, base * 2)
        self.dec1 = self._upconv_block(base * 2, base)

        self.final_conv = nn.Sequential(
            nn.Conv2d(base, config.input_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def _conv_block(self, in_ch: int, out_ch: int, down: bool = False) -> nn.Sequential:
        layers = []
        if down:
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1))
        else:
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1))

        layers.extend([
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ])

        if self.use_dropout:
            layers.append(nn.Dropout2d(0.2))

        return nn.Sequential(*layers)

    def _upconv_block(self, in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Zwraca latent jako MAPĘ CECH (B, C, H, W)
        """
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        z = self.enc4(x)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = self.dec3(z)
        x = self.dec2(x)
        x = self.dec1(x)
        return self.final_conv(x)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z


@torch.no_grad()
def extract_embeddings(
    model: ConvAutoencoder, dataloader: Iterable, device: torch.device
) -> torch.Tensor:
    """
    Ekstrahuje embeddingi (B, latent_dim) z autoenkodera.
    Latent jest mapą cech (B, C, H, W) i jest redukowany
    przez Global Average Pooling.
    """
    model.eval()
    embeddings = []

    for batch in dataloader:
        images = batch[0].to(device)

        # forward
        _, z = model(images)          # z: (B, C, H, W)

        # global average pooling -> (B, C)
        z_vec = z.mean(dim=(2, 3))

        embeddings.append(z_vec.cpu())

    return torch.cat(embeddings, dim=0)

