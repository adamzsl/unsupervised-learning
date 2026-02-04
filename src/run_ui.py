import io
import random
from pathlib import Path
import tkinter as tk
from tkinter import ttk

import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageTk, ImageDraw
import torch



DATASET_DIR = Path("dataset/wiki_art")

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

        self._setup_style()
        self._build_ui()

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
            ),
            ttk.Button(
                self.btn_inner,
                text="GENERATE DIFFERENT DAMAGE",
                style="Rounded.TButton",
                command=self.on_new_damage,
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
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        if cw < 10 or ch < 10:
            return

        iw, ih = self.original_image.size
        scale = min(cw / iw, ch / ih)
        new_size = (int(iw * scale), int(ih * scale))

        img = self.original_image.resize(new_size, Image.BILINEAR)

        if self.mode == "inpaint" and self.mask is not None:
            mask = torch.nn.functional.interpolate(
                self.mask.unsqueeze(0).unsqueeze(0),
                size=(new_size[1], new_size[0]),
                mode="nearest",
            )[0, 0]
            img = apply_alpha(img, mask, self.blink_on)

        self.tk_img = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(
            cw // 2,
            ch // 2,
            image=self.tk_img,
            anchor="center",
        )

    def on_new_image(self):
        self.stop_blink()
        self.original_image = load_random_image()
        self._render_image()

    def on_inpainter(self):
        self.mode = "inpaint"
        self.back_btn.configure(state="normal")
        w, h = self.original_image.size
        self.mask = make_damage_mask(h, w)
        self._build_inpaint_buttons()
        self.start_blink()

    def on_new_damage(self):
        w, h = self.original_image.size
        self.mask = make_damage_mask(h, w)

    def on_back(self):
        if self.mode == "idle":
            return
        self.stop_blink()
        self.mode = "idle"
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


if __name__ == "__main__":
    InpainterUI().mainloop()
