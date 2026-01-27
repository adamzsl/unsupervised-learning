"""
Generator masek uszkodzeń dla obrazów WikiArt.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from PIL import Image, ImageDraw


def _get_rng(rng: Optional[np.random.Generator] = None) -> np.random.Generator:
    return rng if rng is not None else np.random.default_rng()


def generate_square_mask(
    image_size: int,
    max_damage_ratio: float = 0.0625,
    min_mask_size: int = 16,
    max_mask_size: int = 64,
    num_squares: int = 1,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Generuje maskę z losowymi kwadratami (1 = uszkodzenie)."""
    rng = _get_rng(rng)
    mask = np.zeros((image_size, image_size), dtype=np.float32)
    max_area = max_damage_ratio * image_size * image_size

    for _ in range(num_squares):
        size = int(rng.integers(min_mask_size, max_mask_size + 1))
        size = min(size, image_size)
        if size * size > max_area:
            size = max(1, int(np.sqrt(max_area)))
        top = int(rng.integers(0, image_size - size + 1))
        left = int(rng.integers(0, image_size - size + 1))
        mask[top : top + size, left : left + size] = 1.0

    return mask


def generate_irregular_mask(
    image_size: int,
    max_damage_ratio: float = 0.0625,
    brush_width: int = 12,
    num_strokes: int = 3,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Generuje maskę o nieregularnym kształcie (pociągnięcia pędzla)."""
    rng = _get_rng(rng)
    mask_img = Image.new("L", (image_size, image_size), 0)
    draw = ImageDraw.Draw(mask_img)

    for _ in range(num_strokes):
        num_vertices = int(rng.integers(4, 8))
        points = [
            (
                int(rng.integers(0, image_size)),
                int(rng.integers(0, image_size)),
            )
        ]
        for _ in range(num_vertices):
            angle = float(rng.uniform(0, 2 * np.pi))
            length = int(rng.integers(max(4, image_size // 12), image_size // 4))
            last_x, last_y = points[-1]
            new_x = int(np.clip(last_x + length * np.cos(angle), 0, image_size - 1))
            new_y = int(np.clip(last_y + length * np.sin(angle), 0, image_size - 1))
            points.append((new_x, new_y))
        draw.line(points, fill=255, width=brush_width)
        radius = max(1, brush_width // 2)
        for x, y in points:
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=255)

    mask = np.array(mask_img, dtype=np.float32) / 255.0
    max_pixels = int(max_damage_ratio * image_size * image_size)
    if max_pixels > 0 and mask.sum() > max_pixels:
        coords = np.argwhere(mask > 0)
        rng.shuffle(coords)
        coords = coords[:max_pixels]
        mask = np.zeros_like(mask)
        mask[coords[:, 0], coords[:, 1]] = 1.0

    return mask


def apply_mask(image: np.ndarray, mask: np.ndarray, mask_value: float = 1.0) -> np.ndarray:
    """Nakłada maskę na obraz w formacie numpy (C,H,W) lub (H,W,C)."""
    if mask.ndim == 2:
        if image.ndim == 3 and image.shape[0] in {1, 3} and image.shape[1:] == mask.shape:
            mask_expanded = mask[None, ...]
        else:
            mask_expanded = mask[..., None]
    else:
        mask_expanded = mask
    return image * (1.0 - mask_expanded) + mask_value * mask_expanded
