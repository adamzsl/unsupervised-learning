"""Modele inpainting (U-Net oraz VAE)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image
import torchvision.utils as vutils
from torchvision.models import vgg16, VGG16_Weights

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
    def __init__(self, in_channels: int, out_channels: int, use_dropout: bool = False) -> None:
        super().__init__()
        groups = min(8, out_channels)
        while groups > 1 and out_channels % groups != 0:
            groups -= 1
        norm1 = nn.GroupNorm(groups, out_channels)
        norm2 = nn.GroupNorm(groups, out_channels)
        layers = [
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            norm1,
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            norm2,
        ]
        if use_dropout:
            layers.append(nn.Dropout2d(0.2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimpleUNet(nn.Module):
    def __init__(self, config: InpaintConfig) -> None:
        super().__init__()
        base = config.base_channels
        self.use_dropout = config.use_dropout
        self.output_scale = config.output_scale

        self.enc1 = DoubleConv(config.input_channels, base, use_dropout=self.use_dropout)
        self.enc2 = self._conv_block(base, base * 2, pool=True)
        self.enc3 = self._conv_block(base * 2, base * 4, pool=True)
        self.enc4 = self._conv_block(base * 4, base * 8, pool=True)

        self.bottleneck = self._conv_block(base * 8, base * 16, pool=True)

        self.up4 = self._upconv_block(base * 16, base * 8)
        self.dec4 = DoubleConv(base * 16, base * 8, use_dropout=self.use_dropout)
        self.up3 = self._upconv_block(base * 8, base * 4)
        self.dec3 = DoubleConv(base * 8, base * 4, use_dropout=self.use_dropout)
        self.up2 = self._upconv_block(base * 4, base * 2)
        self.dec2 = DoubleConv(base * 4, base * 2, use_dropout=self.use_dropout)
        self.up1 = self._upconv_block(base * 2, base)
        self.dec1 = DoubleConv(base * 2, base, use_dropout=self.use_dropout)

        self.out = nn.Sequential(
            nn.Conv2d(base, 3, kernel_size=1),
            nn.Sigmoid(),
        )

    def _conv_block(self, in_channels: int, out_channels: int, pool: bool = False) -> nn.Sequential:
        layers = []
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=2))
        layers.append(DoubleConv(in_channels, out_channels, use_dropout=self.use_dropout))
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

        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))
        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

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
        groups = min(8, out_channels)
        while groups > 1 and out_channels % groups != 0:
            groups -= 1
        layers.extend(
            [
                nn.GroupNorm(groups, out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.GroupNorm(groups, out_channels),
                nn.ReLU(inplace=True),
            ]
        )
        return nn.Sequential(*layers)

    def _upconv_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(min(8, out_channels) if out_channels % min(8, out_channels) == 0 else 1, out_channels),
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


_VGG_FEATURES: Optional[nn.Module] = None
_VGG_DEVICE: Optional[torch.device] = None


def _get_vgg_features(device: torch.device) -> nn.Module:
    global _VGG_FEATURES, _VGG_DEVICE
    if _VGG_FEATURES is None or _VGG_DEVICE != device:
        weights = VGG16_Weights.DEFAULT
        vgg = vgg16(weights=weights).features[:16].eval()
        for param in vgg.parameters():
            param.requires_grad = False
        _VGG_FEATURES = vgg.to(device)
        _VGG_DEVICE = device
    return _VGG_FEATURES


def _normalize_for_vgg(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def perceptual_loss(output: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    vgg = _get_vgg_features(output.device)
    masked_out = output * mask
    masked_tgt = target * mask
    feat_out = vgg(_normalize_for_vgg(masked_out))
    feat_tgt = vgg(_normalize_for_vgg(masked_tgt))
    return F.l1_loss(feat_out, feat_tgt)


def vae_loss(
    output: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    mask: torch.Tensor,
    mask_weight: float = 1.0,
    perceptual_weight: float = 0.1,
    kld_weight: float = 1.0,
) -> torch.Tensor:
    loss_mask = F.l1_loss(output * mask, target * mask)
    loss = mask_weight * loss_mask
    if perceptual_weight > 0:
        loss = loss + perceptual_weight * perceptual_loss(output, target, mask)
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return loss + kld_weight * kld
    
def inpaint_step(
    model: nn.Module,
    batch,
    device: torch.device,
    mask_weight: float = 1.0,
    perceptual_weight: float = 0.1,
    kld_weight: float = 1.0,
) -> Dict[str, float]:
    images, masks, _, _ = batch
    images = images.to(device)
    masks = masks.to(device)
    # Note: Pass original 'images' to model, not 'masked'.
    # The model applies masking internally in forward().
    outputs = model(images, masks)
    if isinstance(outputs, tuple):
        output, latent = outputs
        loss = vae_loss(
            output,
            images,
            latent["mu"],
            latent["logvar"],
            masks,
            mask_weight=mask_weight,
            perceptual_weight=perceptual_weight,
            kld_weight=kld_weight,
        )
    else:
        output = outputs
        loss_mask = F.l1_loss(output * masks, images * masks)
        loss = mask_weight * loss_mask
        if perceptual_weight > 0:
            loss = loss + perceptual_weight * perceptual_loss(output, images, masks)
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
