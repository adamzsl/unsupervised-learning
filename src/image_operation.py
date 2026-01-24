import io
import random
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image
import torch
import torchvision.transforms as T


DATASET_DIR = Path("dataset/wiki_art")
IMG_SIZE = (256, 256)

MIN_HOLE = 10
MAX_HOLE_FRAC = 0.25


_resize = T.Resize(IMG_SIZE)
_to_tensor = T.ToTensor()


def _make_random_alpha_mask(h: int, w: int) -> torch.Tensor:
    """
    Zwraca maskę (1, H, W) z losową dziurą (0 = przezroczyste)
    """
    mask = torch.ones((1, h, w), dtype=torch.float32)

    size = random.randint(
        MIN_HOLE,
        max(int(min(h, w) * MAX_HOLE_FRAC), MIN_HOLE)
    )
    top = random.randint(0, h - size)
    left = random.randint(0, w - size)

    mask[:, top:top + size, left:left + size] = 0.0
    return mask


def _rgba_png_from_tensor(rgb: torch.Tensor, alpha: torch.Tensor) -> bytes:
    """
    rgb:   (3,H,W) float [0,1]
    alpha: (1,H,W) float [0,1]
    -> PNG RGBA (bytes)
    """
    rgb_u8 = (rgb.numpy() * 255).astype(np.uint8)
    alpha_u8 = (alpha.numpy() * 255).astype(np.uint8)

    rgba = np.concatenate([rgb_u8, alpha_u8], axis=0)  # (4,H,W)
    rgba = np.transpose(rgba, (1, 2, 0))               # (H,W,4)

    img = Image.fromarray(rgba, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def load_random_image_from_parquet() -> bytes:
    """
    Wybiera losowy plik Parquet + losowy rekord
    Zwraca PNG RGBA (bytes) z pełną alfą
    """
    parquet_files = list(DATASET_DIR.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError("Brak plików parquet w dataset/wiki_art")

    parquet_path = random.choice(parquet_files)
    pf = pq.ParquetFile(parquet_path)
    total_rows = sum(
        pf.metadata.row_group(i).num_rows
        for i in range(pf.num_row_groups)
    )
    idx = random.randint(0, total_rows - 1)

    for rg in range(pf.num_row_groups):
        rg_rows = pf.metadata.row_group(rg).num_rows
        if idx < rg_rows:
            row = pf.read_row_group(rg).to_pylist()[idx]
            img_bytes = row["image"]["bytes"]
            break
        idx -= rg_rows
    else:
        raise RuntimeError("Nie udało się wylosować obrazu")

    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = _resize(img)

    rgb = _to_tensor(img)  #(3,H,W)
    alpha = torch.ones((1, IMG_SIZE[0], IMG_SIZE[1]), dtype=torch.float32)

    return _rgba_png_from_tensor(rgb, alpha)

def damage_png_alpha(png_rgba: bytes) -> bytes:
    """
    Przyjmuje PNG RGBA
    Zwraca PNG RGBA z losową dziurą w kanale alfa
    """
    img = Image.open(io.BytesIO(png_rgba)).convert("RGBA")
    img = _resize(img)

    arr = np.array(img)
    rgb = torch.from_numpy(arr[..., :3]).permute(2, 0, 1).float() / 255.0

    h, w = IMG_SIZE
    alpha_mask = _make_random_alpha_mask(h, w)

    return _rgba_png_from_tensor(rgb, alpha_mask)
