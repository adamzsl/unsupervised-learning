"""Modele inpainting (U-Net oraz VAE)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch import nn
from PIL import Image
import torchvision.utils as vutils

def inpaint_image(model: nn.Module, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        output = model(image, mask)
        if isinstance(output, tuple):
            output = output[0]
    return output

@dataclass
class InpaintConfig:
    input_channels: int = 4  # obraz + maska
    base_channels: int = 32
    latent_dim: int = 128
    output_scale: float = 1.0
    use_dropout: bool = False


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimpleUNet(nn.Module):
    def __init__(self, config: InpaintConfig) -> None:
        super().__init__()
        base = config.base_channels
        self.use_dropout = config.use_dropout
        self.output_scale = config.output_scale

        self.enc1 = self._conv_block(config.input_channels, base)
        self.enc2 = self._conv_block(base, base * 2, pool=True)
        self.enc3 = self._conv_block(base * 2, base * 4, pool=True)
        self.enc4 = self._conv_block(base * 4, base * 8, pool=True)

        self.bottleneck = self._conv_block(base * 8, base * 16, pool=True)

        self.up4 = self._upconv_block(base * 16, base * 8)
        self.dec4 = DoubleConv(base * 8, base * 8)
        self.up3 = self._upconv_block(base * 8, base * 4)
        self.dec3 = DoubleConv(base * 4, base * 4)
        self.up2 = self._upconv_block(base * 4, base * 2)
        self.dec2 = DoubleConv(base * 2, base * 2)
        self.up1 = self._upconv_block(base * 2, base)
        self.dec1 = DoubleConv(base, base)

        self.out = nn.Sequential(
            nn.Conv2d(base, 3, kernel_size=1),
            nn.Sigmoid(),
        )

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

    def forward(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = image * (1.0 - mask)
        x = torch.cat([x, mask], dim=1)

        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        b = self.bottleneck(e4)

        d4 = self.up4(b) + e4
        d4 = self.dec4(d4)
        d3 = self.up3(d4) + e3
        d3 = self.dec3(d3)
        d2 = self.up2(d3) + e2
        d2 = self.dec2(d2)
        d1 = self.up1(d2) + e1
        d1 = self.dec1(d1)

        output = self.out(d1) * self.output_scale
        return output * mask + image * (1.0 - mask)


class VAEInpaint(nn.Module):
    def __init__(self, config: InpaintConfig) -> None:
        super().__init__()
        base = config.base_channels
        self.latent_dim = config.latent_dim
        self.output_scale = config.output_scale

        self.enc1 = self._conv_block(config.input_channels, base, down=True)
        self.enc2 = self._conv_block(base, base * 2, down=True)
        self.enc3 = self._conv_block(base * 2, base * 4, down=True)
        self.enc4 = self._conv_block(base * 4, base * 8, down=True)

        self.mu_conv = nn.Conv2d(base * 8, self.latent_dim, kernel_size=1)
        self.logvar_conv = nn.Conv2d(base * 8, self.latent_dim, kernel_size=1)

        self.dec3 = self._upconv_block(self.latent_dim, base * 8)
        self.dec2 = self._upconv_block(base * 8, base * 4)
        self.dec1 = self._upconv_block(base * 4, base * 2)
        self.dec0 = self._upconv_block(base * 2, base)

        self.out = nn.Sequential(
            nn.Conv2d(base, 3, kernel_size=1),
            nn.Sigmoid(),
        )

    def _conv_block(self, in_channels: int, out_channels: int, down: bool = False) -> nn.Sequential:
        layers = []
        if down:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1))
        else:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.extend(
            [
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
        )
        return nn.Sequential(*layers)

    def _upconv_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        return self.mu_conv(x), self.logvar_conv(x)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = self.dec3(z)
        x = self.dec2(x)
        x = self.dec1(x)
        x = self.dec0(x)
        return self.out(x)

    def forward(self, image: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        x = image * (1.0 - mask)
        x = torch.cat([x, mask], dim=1)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        output = self.decode(z) * self.output_scale
        composed = output * mask + image * (1.0 - mask)
        return composed, {"mu": mu, "logvar": logvar}


def vae_loss(
    output: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    mask: torch.Tensor,
    mask_weight: float = 1.0,
    background_weight: float = 0.1,
) -> torch.Tensor:
    loss_mask = nn.functional.l1_loss(output * mask, target * mask)
    inv_masks = 1.0 - mask
    loss_bg = nn.functional.l1_loss(output * inv_masks, target * inv_masks)
    recon = mask_weight * loss_mask + background_weight * loss_bg
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + kld
    
def inpaint_step(
    model: nn.Module,
    batch,
    device: torch.device,
    mask_weight: float = 1.0,
    background_weight: float = 0.1,
) -> Dict[str, float]:
    images, masks, masked, _ = batch
    images = images.to(device)
    masks = masks.to(device)
    masked = masked.to(device)
    outputs = model(masked, masks)
    if isinstance(outputs, tuple):
        output, latent = outputs
        loss = vae_loss(
            output,
            images,
            latent["mu"],
            latent["logvar"],
            masks,
            mask_weight=mask_weight,
            background_weight=background_weight,
        )
    else:
        output = outputs
        loss_mask = nn.functional.l1_loss(output * masks, images * masks)
        inv_masks = 1.0 - masks
        loss_bg = nn.functional.l1_loss(output * inv_masks, images * inv_masks)
        loss = mask_weight * loss_mask + background_weight * loss_bg
    return {"loss": loss}


def log_images_to_comet(logger, images: torch.Tensor, masks: torch.Tensor, output: torch.Tensor, epoch: int, step: int, name: str = "sample") -> None:
    if logger is None:
        return
    grid = vutils.make_grid(
        [
            images[0].cpu(),
            (images[0] * (1.0 - masks[0])).cpu(),
            output[0].cpu(),
        ],
        nrow=3,
        normalize=True,
    )
    grid_np = grid.permute(1, 2, 0).detach().numpy()
    grid_image = Image.fromarray((grid_np * 255).clip(0, 255).astype("uint8"))
    logger.log_image(grid_image, name=f"{name}_epoch_{epoch}_step_{step}.png")
