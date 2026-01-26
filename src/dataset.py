"""Dataset utilities for WikiArt inpainting project."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .damage_generator.masks import apply_mask, generate_irregular_mask, generate_square_mask


@dataclass
class DatasetSplits:
    train: Dataset
    val: Dataset
    test: Dataset


class WikiArtMaskedDataset(Dataset):
    """Dataset zwracający obraz, maskę oraz obraz uszkodzony."""

    def __init__(
        self,
        dataset,
        image_size: int,
        mask_type: str = "square",
        max_damage_ratio: float = 0.0625,
        min_mask_size: int = 16,
        max_mask_size: int = 64,
        brush_width: int = 12,
        num_strokes: int = 3,
    ) -> None:
        self.dataset = dataset
        self.image_size = image_size
        self.mask_type = mask_type
        self.max_damage_ratio = max_damage_ratio
        self.min_mask_size = min_mask_size
        self.max_mask_size = max_mask_size
        self.brush_width = brush_width
        self.num_strokes = num_strokes
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def _make_mask(self) -> np.ndarray:
        if self.mask_type == "irregular":
            return generate_irregular_mask(
                self.image_size,
                max_damage_ratio=self.max_damage_ratio,
                brush_width=self.brush_width,
                num_strokes=self.num_strokes,
            )
        return generate_square_mask(
            self.image_size,
            max_damage_ratio=self.max_damage_ratio,
            min_mask_size=self.min_mask_size,
            max_mask_size=self.max_mask_size,
        )

    def __getitem__(self, index: int):
        item = self.dataset[index]
        image = self.transform(item["image"]).float()
        mask = self._make_mask()
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)
        masked = torch.from_numpy(apply_mask(image.numpy(), mask)).float()
        metadata = {
            "style": item.get("style", None),
            "artist": item.get("artist", None),
        }
        return image, mask_tensor, masked, metadata


def load_wikiart_splits(
    dataset_name: str,
    train_split: float,
    val_split: float,
    test_split: float,
    seed: int = 42,
) -> DatasetSplits:
    """Ładuje dataset i tworzy podziały train/val/test."""
    dataset = load_dataset(dataset_name, split="train")
    splits = dataset.train_test_split(test_size=test_split, seed=seed)
    remaining = splits["train"]
    val_ratio = val_split / (1.0 - test_split)
    train_val = remaining.train_test_split(test_size=val_ratio, seed=seed)
    return DatasetSplits(train=train_val["train"], val=train_val["test"], test=splits["test"])


def create_dataloaders(
    splits: DatasetSplits,
    image_size: int,
    mask_type: str,
    batch_size: int,
    num_workers: int,
    max_damage_ratio: float = 0.0625,
    min_mask_size: int = 16,
    max_mask_size: int = 64,
    brush_width: int = 12,
    num_strokes: int = 3,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Tworzy DataLoadery dla train/val/test."""
    train_ds = WikiArtMaskedDataset(
        splits.train,
        image_size=image_size,
        mask_type=mask_type,
        max_damage_ratio=max_damage_ratio,
        min_mask_size=min_mask_size,
        max_mask_size=max_mask_size,
        brush_width=brush_width,
        num_strokes=num_strokes,
    )
    val_ds = WikiArtMaskedDataset(
        splits.val,
        image_size=image_size,
        mask_type=mask_type,
        max_damage_ratio=max_damage_ratio,
        min_mask_size=min_mask_size,
        max_mask_size=max_mask_size,
        brush_width=brush_width,
        num_strokes=num_strokes,
    )
    test_ds = WikiArtMaskedDataset(
        splits.test,
        image_size=image_size,
        mask_type=mask_type,
        max_damage_ratio=max_damage_ratio,
        min_mask_size=min_mask_size,
        max_mask_size=max_mask_size,
        brush_width=brush_width,
        num_strokes=num_strokes,
    )
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    )


def save_split_indices(splits: DatasetSplits, output_dir: Path) -> None:
    """Zapisuje indeksy datasetu do plików npy (opcjonalne)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, split in {"train": splits.train, "val": splits.val, "test": splits.test}.items():
        if "__index_level_0__" in split.column_names:
            indices = split["__index_level_0__"]
        else:
            indices = list(range(len(split)))
        np.save(output_dir / f"{name}_indices.npy", np.array(indices))
