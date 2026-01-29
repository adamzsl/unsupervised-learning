import io
import random
from pathlib import Path
import tkinter as tk
from tkinter import ttk

import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageTk, ImageDraw
import torch

from autoencoder import ConvAutoencoder, AutoencoderConfig
from inpainter import InpaintConfig, SimpleUNet, inpaint_image

from utils import load_config

BASE_DIR = Path(__file__).resolve().parents[1]
config = load_config(BASE_DIR / "config.yaml")
data_cfg = config['data']

DATASET_DIR = BASE_DIR / "dataset1" / "data"
AUTOENCODER_PATH = BASE_DIR / "models" / "autoencoder.pth"
INPAINTER_PATH = BASE_DIR / "models" / "inpaint_base.pth"


MIN_HOLE = 10
MAX_HOLE_FRAC = 0.25
RESPONSIVE_BREAKPOINT = 420
BLINK_MS = 400
BUTTON_MIN_WIDTH = 520

BG_COLOR = "#1e1e1e"
PRIMARY = "#2d89ef"
PRIMARY_DISABLED = "#3a3a3a"
TEXT_COLOR = "#ffffff"

FONT_MAIN = ("Segoe UI", 13, "bold")



def load_random_image():
    files = list(DATASET_DIR.glob("*.parquet"))
    pf = pq.ParquetFile(random.choice(files))
    total = sum(pf.metadata.row_group(i).num_rows for i in range(pf.num_row_groups))
    idx = random.randint(0, total - 1)

    for rg in range(pf.num_row_groups):
        rows = pf.metadata.row_group(rg).num_rows
        if idx < rows:
            row = pf.read_row_group(rg).to_pylist()[idx]
            return Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
        idx -= rows


def make_damage_mask(h, w):
    mask = torch.ones((h, w))
    size = random.randint(MIN_HOLE, int(min(h, w) * MAX_HOLE_FRAC))
    y = random.randint(0, h - size)
    x = random.randint(0, w - size)
    mask[y:y + size, x:x + size] = 0
    return mask


def apply_alpha(img, mask, visible):
    arr = np.array(img)
    alpha = np.ones(arr.shape[:2], dtype=np.uint8) * 255
    if visible:
        alpha[mask == 0] = 0
    return Image.fromarray(np.dstack([arr, alpha]), "RGBA")


def load_arrow_icon(size=28):
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.polygon(
        [
            (size * 0.65, size * 0.2),
            (size * 0.35, size * 0.5),
            (size * 0.65, size * 0.8),
        ],
        fill="white",
    )
    return ImageTk.PhotoImage(img)


class InpainterUI(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Inpainter")
        self.geometry("520x620")
        self.minsize(360, 480)
        self.configure(bg=BG_COLOR)

        self.mode = "idle"
        self.blink_on = False
        self.blink_job = None

        self.original_image = load_random_image()
        self.mask = None
        self.tk_img = None
        self.inpainted_image = None

        self._setup_style()
        self._build_ui()

        self.autoencoder = None
        self.ae_image = None
        self.inpainter_model = None

        self.bind("<Configure>", self._on_resize)

    def _setup_style(self):
        style = ttk.Style(self)
        style.theme_use("clam")

        style.configure(
            "Rounded.TButton",
            font=FONT_MAIN,
            foreground=TEXT_COLOR,
            background=PRIMARY,
            padding=(18, 12),
            borderwidth=0,
            focusthickness=0,
        )

        style.map(
            "Rounded.TButton",
            background=[
                ("disabled", PRIMARY_DISABLED),
                ("active", "#3ea0ff"),
            ],
            foreground=[("disabled", "#888888")],
        )

    def _build_ui(self):
        top = tk.Frame(self, bg=BG_COLOR)
        top.pack(fill="x", pady=6)

        self.arrow_icon = load_arrow_icon()

        self.back_btn = tk.Label(
            top,
            image=self.arrow_icon,
            bg=BG_COLOR,
            cursor="hand2",
            padx=10,
            pady=6,
        )
        self.back_btn.pack(side="left", padx=8)
        self.back_btn.bind("<Button-1>", lambda e: self.on_back())
        self.back_btn.configure(state="disabled")

        self.canvas = tk.Canvas(
            self,
            bg=BG_COLOR,
            highlightthickness=0,
            bd=0,
        )
        self.canvas.pack(expand=True, fill="both", padx=20, pady=10)

        self.btn_container = tk.Frame(self, bg=BG_COLOR)
        self.btn_container.pack(fill="x", pady=12)

        self.btn_inner = tk.Frame(
            self.btn_container, bg=BG_COLOR, width=BUTTON_MIN_WIDTH
        )
        self.btn_inner.grid(row=0, column=1)

        self.btn_container.columnconfigure(0, weight=1)
        self.btn_container.columnconfigure(2, weight=1)

        self._build_idle_buttons()
        self._render_image()

    def _clear_buttons(self):
        for w in self.btn_inner.winfo_children():
            w.destroy()

    def _build_idle_buttons(self):
        self._clear_buttons()
        self.buttons = [
            ttk.Button(
                self.btn_inner,
                text="INPAINTER",
                style="Rounded.TButton",
                command=self.on_inpainter,
            ),
            ttk.Button(
                self.btn_inner,
                text="SELECT DIFFERENT PICTURE",
                style="Rounded.TButton",
                command=self.on_new_image,
            ),
            ttk.Button(
                self.btn_inner,
                text="AUTOENCODER",
                style="Rounded.TButton",
                command=self.on_autoencoder,
            ),

        ]
        self._layout_buttons()

    def _build_inpaint_buttons(self):
        self._clear_buttons()
        self.buttons = [
            ttk.Button(
                self.btn_inner,
                text="RUN INPAINTER",
                style="Rounded.TButton",
                command=self.on_run_inpainter,
            ),
            ttk.Button(
                self.btn_inner,
                text="GENERATE DIFFERENT DAMAGE",
                style="Rounded.TButton",
                command=self.on_new_damage,
            ),
        ]
        self._layout_buttons()

    def _build_autoencoder_result_buttons(self):
        self._clear_buttons()
        self.buttons = [
            ttk.Button(
                self.btn_inner,
                text="SHOW ORIGINAL",
                style="Rounded.TButton",
                command=self.on_show_original_ae,
            ),
            ttk.Button(
                self.btn_inner,
                text="SHOW ENCODED",
                style="Rounded.TButton",
                command=self.on_show_encoded,
            ),
        ]
        self._layout_buttons()

    def _build_inpainter_result_buttons(self):
        self._clear_buttons()
        self.buttons = [
            ttk.Button(
                self.btn_inner, text="SHOW RESULT", style="Rounded.TButton", command=self.on_show_result_ip
            ),
            ttk.Button(
                self.btn_inner, text="SHOW MASK", style="Rounded.TButton", command=self.on_show_mask_ip
            ),
            ttk.Button(
                self.btn_inner, text="SHOW ORIGINAL", style="Rounded.TButton", command=self.on_show_original_ip
            ),
        ]
        self._layout_buttons()

    def _layout_buttons(self):
        for b in self.buttons:
            b.grid_forget()

        vertical = self.winfo_width() < RESPONSIVE_BREAKPOINT

        if vertical:
            for i, b in enumerate(self.buttons):
                b.grid(row=i, column=0, sticky="nsew", pady=6)
            self.btn_inner.columnconfigure(0, weight=1)
        else:
            for i, b in enumerate(self.buttons):
                b.grid(row=0, column=i, sticky="nsew", padx=8)
            for i in range(len(self.buttons)):
                self.btn_inner.columnconfigure(i, weight=1)

    def _render_image(self):
        img_src = self.original_image
        if self.mode == "autoencoder_result" and self.ae_image is not None:
            img_src = self.ae_image
        elif self.mode == "inpainting_result" and self.inpainted_image is not None:
            img_src = self.inpainted_image
        elif self.mode == "inpainting_mask" and self.mask is not None:
            mask_np = (self.mask.numpy() * 255).astype(np.uint8)
            img_src = Image.fromarray(mask_np).convert("RGB")

        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        if cw < 10 or ch < 10:
            return

        iw, ih = img_src.size
        scale = min(cw / iw, ch / ih)
        new_size = (int(iw * scale), int(ih * scale))

        img = img_src.resize(new_size, Image.NEAREST)

        if self.mode == "inpaint" and self.mask is not None:
            mask = torch.nn.functional.interpolate(
                self.mask.unsqueeze(0).unsqueeze(0),
                size=(new_size[1], new_size[0]),
                mode="nearest",
            )[0, 0]
            img = apply_alpha(img, mask.cpu().numpy(), self.blink_on)

        self.tk_img = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(
            cw // 2,
            ch // 2,
            image=self.tk_img,
            anchor="center",
        )
        
    def run_autoencoder(self):
        if self.autoencoder is None:
            print("Loading autoencoder")

            img_size = data_cfg['image_size'] 

            auto_cfg = AutoencoderConfig(
                input_channels=3,
                latent_dim=config['encoder']['latent_dim'],
                image_size=img_size,
            )

            self.autoencoder = ConvAutoencoder(auto_cfg).to("cpu")
            self.autoencoder.load_state_dict(
                torch.load(AUTOENCODER_PATH, map_location="cpu")
            )
            self.autoencoder.eval()

        img = self.original_image.resize(
            (data_cfg['image_size'], data_cfg['image_size']),
            Image.BILINEAR,
        )

        img = np.array(img).astype(np.float32) / 255.0
        tensor = (
            torch.from_numpy(img)
            .permute(2, 0, 1)
            .unsqueeze(0)
        )

        with torch.no_grad():
            recon, _ = self.autoencoder(tensor)

        out = (
            recon.squeeze(0)
            .permute(1, 2, 0)
            .clamp(0, 1)
            .numpy()
        )
        out = (out * 255).astype(np.uint8)

        self.ae_image = Image.fromarray(out)

    def run_inpainter_inference(self):
        if self.inpainter_model is None:
            print("Loading inpainter model")
            try:
                inpainter_cfg = InpaintConfig(input_channels=4, base_channels=config['inpainting']['base_channels'])
                self.inpainter_model = SimpleUNet(inpainter_cfg).to("cpu")
                self.inpainter_model.load_state_dict(
                    torch.load(INPAINTER_PATH, map_location="cpu")
                )
                self.inpainter_model.eval()
            except FileNotFoundError:
                print(f"Inpainter model not found at {INPAINTER_PATH}. Please train it first.")
                self.inpainter_model = None
                return
            except Exception as e:
                print(f"Error loading inpainter model: {e}")
                self.inpainter_model = None
                return

        if self.inpainter_model is None:
            return

        img_size = data_cfg['image_size']
        original_img_resized = self.original_image.resize(
            (img_size, img_size), Image.BILINEAR
        )
        image_np = np.array(original_img_resized).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).float()

        mask_tensor = self.mask.unsqueeze(0).unsqueeze(0).float()

        masked_input_tensor = image_tensor * (1-mask_tensor)

        with torch.no_grad():
            output_tensor = inpaint_image(self.inpainter_model, masked_input_tensor, mask_tensor)

        final_tensor = (
            (1-mask_tensor) * output_tensor +
             (mask_tensor) * image_tensor
        )

        out_np = (
            final_tensor.squeeze(0)
            .permute(1, 2, 0)
            .clamp(0, 1)
            .numpy()
        )

        self.inpainted_image = Image.fromarray((out_np * 255).astype(np.uint8))



    def on_new_image(self):
        self.stop_blink()
        self.original_image = load_random_image()
        self.ae_image = None
        self.inpainted_image = None
        self.mask = None
        self._render_image()

    def on_inpainter(self):
        self.mode = "inpaint"
        self.back_btn.configure(state="normal")
        img_size = data_cfg['image_size']
        self.mask = make_damage_mask(img_size, img_size)
        self._build_inpaint_buttons()
        self.start_blink()

    def on_new_damage(self):
        img_size = data_cfg['image_size']
        self.mask = make_damage_mask(img_size, img_size)
        self._render_image()

    def on_run_inpainter(self):
        self.stop_blink()
        self.run_inpainter_inference()
        self.mode = "inpainting_result"
        self._build_inpainter_result_buttons()
        self._render_image()

    def on_back(self):
        if self.mode == "idle":
            return
        self.stop_blink()
        self.mode = "idle"
        self.ae_image = None
        self.inpainted_image = None
        self.mask = None
        self.back_btn.configure(state="disabled")
        self._build_idle_buttons()
        self._render_image()

    def start_blink(self):
        self.blink_on = False
        self._blink()

    def stop_blink(self):
        if self.blink_job:
            self.after_cancel(self.blink_job)
            self.blink_job = None

    def _blink(self):
        self.blink_on = not self.blink_on
        self._render_image()
        self.blink_job = self.after(BLINK_MS, self._blink)

    def _on_resize(self, event):
        self._layout_buttons()
        self._render_image()

    def on_autoencoder(self):
        self.run_autoencoder()
        self.mode = "autoencoder_result"
        self.back_btn.configure(state="normal")
        self._build_autoencoder_result_buttons()
        self._render_image()

    def on_show_original_ae(self):
        self.mode = "autoencoder_original"
        self._render_image()

    def on_show_encoded(self):
        self.mode = "autoencoder_result"
        self._render_image()

    def on_show_result_ip(self):
        self.mode = "inpainting_result"
        self._render_image()

    def on_show_mask_ip(self):
        self.mode = "inpainting_mask"
        self._render_image()

    def on_show_original_ip(self):
        self.mode = "inpainting_original"
        self._render_image()


if __name__ == "__main__":
    InpainterUI().mainloop()
