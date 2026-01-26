"""Dataset dla super-resolution (HR -> LR)."""
from __future__ import annotations

from typing import Tuple

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class SuperResDataset(Dataset):
    def __init__(self, dataset, image_size: int, scale_factor: int = 2):
        self.dataset = dataset
        self.image_size = image_size
        self.scale_factor = scale_factor
        self.hr_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )
        lr_size = image_size // scale_factor
        self.lr_transform = transforms.Compose(
            [
                transforms.Resize((lr_size, lr_size)),
                transforms.ToTensor(),
                transforms.Resize((image_size, image_size)),
            ]
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.dataset[index]
        image = item["image"]
        hr = self.hr_transform(image)
        lr = self.lr_transform(image)
        return lr, hr
