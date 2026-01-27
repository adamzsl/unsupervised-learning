"""Proste GUI demo oparte o Streamlit."""
from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.damage_generator.masks import apply_mask, generate_irregular_mask, generate_square_mask

try:  # torch jest opcjonalny dla demo
    import torch

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    torch = None
    TORCH_AVAILABLE = False


st.set_page_config(page_title="WikiArt Inpainting Demo", layout="wide")

st.title("WikiArt Inpainting Demo")

image_dir = st.sidebar.text_input("Folder z obrazami testowymi", "data/processed/test_images")
mask_type = st.sidebar.selectbox("Typ maski", ["square", "irregular"])
image_size = st.sidebar.slider("Rozmiar obrazu", 128, 512, 256, step=32)

st.sidebar.markdown("---")
run_inpaint = st.sidebar.checkbox("Wykonaj inpainting", value=True)
run_superres = st.sidebar.checkbox("Wykonaj super-resolution", value=True)

if Path(image_dir).exists():
    images = list(Path(image_dir).glob("*.jpg")) + list(Path(image_dir).glob("*.png"))
else:
    images = []

if st.button("Wylosuj obraz") and images:
    selected = random.choice(images)
    st.session_state["selected_image"] = selected

selected = st.session_state.get("selected_image")
if selected:
    st.caption(f"Wybrany obraz: {Path(selected).name}")
    image = Image.open(selected).convert("RGB").resize((image_size, image_size))
    image_np = np.asarray(image) / 255.0
    if mask_type == "irregular":
        mask = generate_irregular_mask(image_size)
    else:
        mask = generate_square_mask(image_size)
    masked = apply_mask(image_np, mask)

    inpainted = None
    super_res = None
    if run_inpaint and TORCH_AVAILABLE:
        from src.inpainting.inference import inpaint_image
        from src.inpainting.model import InpaintConfig, SimpleUNet

        model = SimpleUNet(InpaintConfig())
        masked_tensor = torch.from_numpy(masked).permute(2, 0, 1).unsqueeze(0).float()
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float()
        inpainted = (
            inpaint_image(model, masked_tensor, mask_tensor)
            .squeeze(0)
            .permute(1, 2, 0)
            .numpy()
        )
    if run_superres and inpainted is not None and TORCH_AVAILABLE:
        from src.superres.inference import super_resolve
        from src.superres.model import SimpleSRCNN, SuperResConfig

        sr_model = SimpleSRCNN(SuperResConfig())
        input_tensor = torch.from_numpy(inpainted).permute(2, 0, 1).unsqueeze(0).float()
        super_res = (
            super_resolve(sr_model, input_tensor)
            .squeeze(0)
            .permute(1, 2, 0)
            .numpy()
        )
    if not TORCH_AVAILABLE:
        st.warning("Torch nie jest dostępny - pomijam inpainting i super-resolution.")

    columns = 3 + int(inpainted is not None) + int(super_res is not None)
    col_list = st.columns(columns)
    col_idx = 0
    col_list[col_idx].subheader("Oryginal")
    col_list[col_idx].image(image_np, clamp=True)
    col_idx += 1
    col_list[col_idx].subheader("Maska")
    col_list[col_idx].image(mask, clamp=True)
    col_idx += 1
    col_list[col_idx].subheader("Uszkodzony")
    col_list[col_idx].image(masked, clamp=True)
    col_idx += 1
    if inpainted is not None:
        col_list[col_idx].subheader("Inpainting")
        col_list[col_idx].image(inpainted, clamp=True)
        col_idx += 1
    if super_res is not None:
        col_list[col_idx].subheader("Super-resolution")
        col_list[col_idx].image(super_res, clamp=True)
else:
    st.info("Wybierz folder z obrazami testowymi i wylosuj obraz.")

st.sidebar.markdown("---")
if st.sidebar.button("Wgraj przykładowe obrazy demo"):
    demo_dir = Path(image_dir)
    demo_dir.mkdir(parents=True, exist_ok=True)
    for idx, color in enumerate([(220, 90, 90), (90, 220, 140), (90, 140, 220)]):
        img = Image.new("RGB", (image_size, image_size), color)
        img.save(demo_dir / f"demo_{idx}.png")
    st.sidebar.success(f"Zapisano przykładowe obrazy w {demo_dir}")
