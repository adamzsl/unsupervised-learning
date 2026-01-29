"""Prosty model inpainting oparty o U-Net."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import nn
from torchvision import models
from torchvision.models import VGG16_Weights

def inpaint_image(model: SimpleUNet, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        output = model(image, mask)
    return output

@dataclass
class InpaintConfig:
    input_channels: int = 4  # obraz + maska
    base_channels: int = 32


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VGGPerceptualLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        vgg = models.vgg16(weights=VGG16_Weights.DEFAULT).features[:16].eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        output_norm = (output - self.mean) / self.std
        target_norm = (target - self.mean) / self.std
        return nn.functional.l1_loss(self.vgg(output_norm), self.vgg(target_norm))


_PERCEPTUAL_LOSS: VGGPerceptualLoss | None = None


def _get_perceptual_loss(device: torch.device) -> VGGPerceptualLoss:
    global _PERCEPTUAL_LOSS
    if _PERCEPTUAL_LOSS is None:
        _PERCEPTUAL_LOSS = VGGPerceptualLoss()
    if next(_PERCEPTUAL_LOSS.parameters()).device != device:
        _PERCEPTUAL_LOSS = _PERCEPTUAL_LOSS.to(device)
    return _PERCEPTUAL_LOSS


class SimpleUNet(nn.Module):
    def __init__(self, config: InpaintConfig) -> None:
        super().__init__()
        base = config.base_channels
        self.enc1 = DoubleConv(config.input_channels, base)
        self.enc2 = nn.Sequential(
            nn.Conv2d(base, base * 2, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.enc2_conv = DoubleConv(base * 2, base * 2)
        self.enc3 = nn.Sequential(
            nn.Conv2d(base * 2, base * 4, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.enc3_conv = DoubleConv(base * 4, base * 4)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 4, stride=2, padding=1)
        self.dec2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 4, stride=2, padding=1)
        self.dec1 = DoubleConv(base * 2, base)

        self.out = nn.Sequential(
            nn.Conv2d(base, 3, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        masked = image * (1.0 - mask)
        x = torch.cat([masked, mask], dim=1)
        e1 = self.enc1(x)
        e2 = self.enc2_conv(self.enc2(e1))
        e3 = self.enc3_conv(self.enc3(e2))

        d2 = self.up2(e3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        output = self.out(d1)
        return output * mask + masked
    
def inpaint_step(
    model: SimpleUNet,
    batch,
    device: torch.device,
    mask_weight: float = 1.0,
    background_weight: float = 0.1,
    perceptual_weight: float = 0.1,
) -> Dict[str, float]:
    images, masks, masked, _ = batch
    images = images.to(device)
    masks = masks.to(device)
    masked = masked.to(device)
    output = model(masked, masks)
    loss_mask = nn.functional.l1_loss(output * masks, images * masks)
    inv_masks = 1.0 - masks
    loss_bg = nn.functional.l1_loss(output * inv_masks, images * inv_masks)
    if perceptual_weight > 0.0:
        perceptual = _get_perceptual_loss(device)
        loss_perceptual = perceptual(output, images)
    else:
        loss_perceptual = torch.tensor(0.0, device=device)
    loss = mask_weight * loss_mask + background_weight * loss_bg + perceptual_weight * loss_perceptual
    return {
        "loss": loss,
        "loss_mask": float(loss_mask.detach().cpu()),
        "loss_bg": float(loss_bg.detach().cpu()),
        "loss_perceptual": float(loss_perceptual.detach().cpu()),
    }
