import os
import sys
import io
import json
import base64
import traceback
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List, Set

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

from PySide6 import QtCore, QtGui, QtWidgets
from xml.sax.saxutils import escape as xml_escape

# ----------------------------
# Optional semantic deps
# ----------------------------
SEMANTIC_AVAILABLE = True
try:
    import torch
    from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
except Exception:
    SEMANTIC_AVAILABLE = False
    torch = None
    SegformerImageProcessor = None
    SegformerForSemanticSegmentation = None


# ----------------------------
# Config / presets
# ----------------------------
PRESET_CHARSETS = {
    "Standard": "@%#*+=-:. ",
    "Dense":   "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~<>i!lI;:,\"^`'. ",
    "Blocks":  "█▓▒░ "
}

# Added:
CONTENT_MODES = ["ASCII", "Semantic Letters", "Pixel Edges", "Edge Overlay"]
FILL_STYLES = ["Jumble", "Cycle"]
COLOR_MODES = ["Mono", "Color (Direct)", "Color (K-means)"]

EDGE_BG_MODES = ["Dark", "White"]
EDGE_FG_MODES = ["White", "Black"]

PROCESS_MAX_SIDE = 1600
FAST_DEBOUNCE_MS = 80
KMEANS_DEBOUNCE_MS = 220

SEG_MODEL_NAME = "nvidia/segformer-b0-finetuned-ade-512-512"
SEG_CACHE_SIZE = 256

DEFAULT_PREVIEW_CAP = 160


# ----------------------------
# Dark UI styling
# ----------------------------
def apply_dark_palette(app: QtWidgets.QApplication):
    palette = QtGui.QPalette()
    base = QtGui.QColor(20, 20, 22)
    alt = QtGui.QColor(28, 28, 32)
    text = QtGui.QColor(230, 230, 235)
    disabled = QtGui.QColor(150, 150, 160)

    palette.setColor(QtGui.QPalette.Window, base)
    palette.setColor(QtGui.QPalette.WindowText, text)
    palette.setColor(QtGui.QPalette.Base, QtGui.QColor(16, 16, 18))
    palette.setColor(QtGui.QPalette.AlternateBase, alt)
    palette.setColor(QtGui.QPalette.Text, text)
    palette.setColor(QtGui.QPalette.Button, alt)
    palette.setColor(QtGui.QPalette.ButtonText, text)
    palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(72, 122, 255))
    palette.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor(255, 255, 255))
    palette.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, disabled)
    palette.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.Text, disabled)
    palette.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.ButtonText, disabled)
    app.setPalette(palette)

    app.setStyleSheet("""
        QMainWindow { background: #141416; }
        QGroupBox {
            border: 1px solid #2B2B30;
            border-radius: 10px;
            margin-top: 10px;
            padding: 10px;
        }
        QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 6px; }
        QTabWidget::pane { border: 1px solid #2B2B30; border-radius: 10px; }
        QTabBar::tab {
            background: #1C1C20; padding: 8px 12px; margin-right: 4px;
            border-top-left-radius: 8px; border-top-right-radius: 8px;
        }
        QTabBar::tab:selected { background: #2A2A31; }
        QPushButton {
            background: #2A2A31; border: 1px solid #3A3A44;
            padding: 8px 10px; border-radius: 10px;
        }
        QPushButton:hover { background: #343440; }
        QPushButton:pressed { background: #202026; }
        QComboBox, QSpinBox, QLineEdit {
            background: #1A1A1E; border: 1px solid #3A3A44;
            padding: 6px; border-radius: 10px;
        }
        QTextEdit {
            background: #101012; border: 1px solid #2B2B30;
            border-radius: 10px;
        }
        QSlider::groove:horizontal { height: 6px; background: #2B2B30; border-radius: 3px; }
        QSlider::handle:horizontal {
            width: 16px; margin: -6px 0; border-radius: 8px; background: #4B8BFF;
        }
        QScrollArea { border: none; }
        QTableWidget {
            background: #101012;
            border: 1px solid #2B2B30;
            border-radius: 10px;
            gridline-color: #2B2B30;
        }
        QHeaderView::section {
            background: #1C1C20;
            border: 1px solid #2B2B30;
            padding: 6px;
        }
    """)


# ----------------------------
# Fonts
# ----------------------------
def list_windows_font_files() -> List[Tuple[str, str]]:
    fonts_dir = os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts")
    items = []
    try:
        for fn in os.listdir(fonts_dir):
            if fn.lower().endswith((".ttf", ".otf", ".ttc")):
                items.append((fn, os.path.join(fonts_dir, fn)))
    except Exception:
        pass
    items.sort(key=lambda x: x[0].lower())
    return items

_FONT_CACHE: Dict[Tuple[str, int, int], ImageFont.ImageFont] = {}

def safe_load_pil_font(path: str, size: int, ttc_index: int = 0):
    """
    Robust font loading:
    - Supports TTC via index.
    """
    key = (path or "", int(size), int(ttc_index))
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]

    try:
        if path and os.path.exists(path):
            f = ImageFont.truetype(path, size=int(size), index=int(ttc_index))
        else:
            f = ImageFont.load_default()
    except Exception:
        f = ImageFont.load_default()

    _FONT_CACHE[key] = f
    return f

def measure_char_cell(font) -> Tuple[int, int]:
    dummy = Image.new("RGB", (80, 80))
    d = ImageDraw.Draw(dummy)
    bbox = d.textbbox((0, 0), "M", font=font)
    return max(1, bbox[2] - bbox[0]), max(1, bbox[3] - bbox[1])

def downscale_for_processing(img: Image.Image, max_side: int) -> Image.Image:
    w, h = img.size
    m = max(w, h)
    if m <= max_side:
        return img
    scale = max_side / m
    return img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)


# ----------------------------
# Adjustments
# ----------------------------
def apply_adjustments(img_rgb: Image.Image,
                      brightness: int, contrast: int, ambiance: int,
                      saturation: int, highlights: int, shadows: int) -> Image.Image:
    x = np.asarray(img_rgb.convert("RGB"), dtype=np.float32) / 255.0

    b = 1.0 + (brightness / 100.0)
    x = np.clip(x * b, 0.0, 1.0)

    c = 1.0 + (contrast / 100.0)
    x = np.clip((x - 0.5) * c + 0.5, 0.0, 1.0)

    luma = (0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]).astype(np.float32)
    shadow_mask = np.clip((0.5 - luma) / 0.5, 0.0, 1.0)
    highlight_mask = np.clip((luma - 0.5) / 0.5, 0.0, 1.0)

    sh = shadows / 100.0
    hi = highlights / 100.0
    x = np.clip(x + (0.35 * sh) * shadow_mask[..., None], 0.0, 1.0)
    x = np.clip(x + (0.30 * hi) * highlight_mask[..., None], 0.0, 1.0)

    a = ambiance / 100.0
    if abs(a) > 1e-6:
        luma = (0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]).astype(np.float32)
        shadow_mask = np.clip((0.55 - luma) / 0.55, 0.0, 1.0)
        highlight_mask = np.clip((luma - 0.55) / 0.45, 0.0, 1.0)
        x = np.clip(x + (0.20 * a) * shadow_mask[..., None], 0.0, 1.0)
        x = np.clip(x - (0.12 * a) * highlight_mask[..., None], 0.0, 1.0)
        c2 = 1.0 + 0.10 * a
        x = np.clip((x - 0.5) * c2 + 0.5, 0.0, 1.0)

    s = 1.0 + (saturation / 100.0)
    luma = (0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]).astype(np.float32)
    x = np.clip(luma[..., None] + (x - luma[..., None]) * s, 0.0, 1.0)

    return Image.fromarray((x * 255.0).astype(np.uint8), mode="RGB")


# ----------------------------
# K-means color map
# ----------------------------
def kmeans_palette(pixels: np.ndarray, k: int = 8, iters: int = 10, seed: int = 42, sample: int = 6000):
    rng = np.random.default_rng(seed)
    n = pixels.shape[0]
    if n == 0:
        return np.zeros((k, 3), dtype=np.float32)
    data = pixels if n <= sample else pixels[rng.choice(n, size=sample, replace=False)]

    k = int(np.clip(k, 2, 32))
    if data.shape[0] < k:
        k = max(2, data.shape[0])

    centers = data[rng.choice(data.shape[0], size=k, replace=False)].copy()
    for _ in range(iters):
        d2 = ((data[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = d2.argmin(axis=1)
        new_centers = centers.copy()
        for j in range(k):
            mask = labels == j
            if mask.any():
                new_centers[j] = data[mask].mean(axis=0)
            else:
                new_centers[j] = data[rng.integers(0, data.shape[0])]
        if np.allclose(new_centers, centers, atol=1.0):
            centers = new_centers
            break
        centers = new_centers
    return centers.astype(np.float32)

def build_color_map(resized_rgb: Image.Image, mode: str, k: int):
    arr = np.asarray(resized_rgb, dtype=np.float32)
    if mode == "Color (Direct)":
        return np.clip(arr, 0, 255).astype(np.uint8)
    if mode == "Color (K-means)":
        h, w, _ = arr.shape
        flat = arr.reshape(-1, 3)
        centers = kmeans_palette(flat, k=int(k))
        d2 = ((flat[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = d2.argmin(axis=1)
        quant = centers[labels].reshape(h, w, 3)
        return np.clip(quant, 0, 255).astype(np.uint8)
    return None


# ----------------------------
# ASCII grid
# ----------------------------
def make_ascii_lines(img: Image.Image,
                     out_w: int,
                     charset: str,
                     invert: bool,
                     char_w: int,
                     char_h: int) -> Tuple[List[str], Image.Image, Tuple[int, int]]:
    out_w = max(20, int(out_w))
    rgb = img.convert("RGB")
    gray = rgb.convert("L")
    iw, ih = gray.size
    out_h = max(1, int(round((ih / iw) * out_w * (char_w / char_h))))

    rgb_r = rgb.resize((out_w, out_h), Image.Resampling.BILINEAR)
    gray_r = gray.resize((out_w, out_h), Image.Resampling.BILINEAR)

    if not charset or len(charset) < 2:
        charset = PRESET_CHARSETS["Standard"]

    chars = charset
    n = len(chars) - 1
    px = np.asarray(gray_r, dtype=np.float32)
    if invert:
        px = 255.0 - px
    idx = (px / 255.0 * n).astype(np.int32)
    lines = ["".join(chars[i] for i in row) for row in idx]
    return lines, rgb_r, (out_w, out_h)


# ----------------------------
# Pixelated edge detection (Pixel Edges + Edge Overlay)
# ----------------------------
def make_pixel_edge_mask(adjusted_img: Image.Image,
                         out_w: int,
                         out_h: int,
                         pixel_size: int = 6,
                         threshold: int = 70,
                         thickness: int = 2,
                         invert: bool = False) -> Image.Image:
    """
    Returns a pixelated, chunky edge mask (L mode), sized (out_w, out_h).
    """
    out_w = max(20, int(out_w))
    out_h = max(1, int(out_h))
    pixel_size = max(1, int(pixel_size))
    threshold = int(np.clip(threshold, 0, 255))
    thickness = max(0, int(thickness))

    gray = adjusted_img.convert("L").resize((out_w, out_h), Image.Resampling.BILINEAR)
    gray = gray.filter(ImageFilter.GaussianBlur(radius=1.0))
    arr = np.asarray(gray, dtype=np.float32)

    p = np.pad(arr, ((1, 1), (1, 1)), mode="edge")
    gx = (-p[:-2, :-2] + p[:-2, 2:]
          - 2.0 * p[1:-1, :-2] + 2.0 * p[1:-1, 2:]
          - p[2:, :-2] + p[2:, 2:])
    gy = (-p[:-2, :-2] - 2.0 * p[:-2, 1:-1] - p[:-2, 2:]
          + p[2:, :-2] + 2.0 * p[2:, 1:-1] + p[2:, 2:])

    mag = np.sqrt(gx * gx + gy * gy)
    mag = (mag / (mag.max() + 1e-6) * 255.0).astype(np.uint8)

    mask = (mag >= threshold).astype(np.uint8) * 255
    mask_img = Image.fromarray(mask).convert("L")

    # Pixelate blocks: downsample (BOX), binarize, upsample (NEAREST)
    if pixel_size > 1:
        sw = max(1, out_w // pixel_size)
        sh = max(1, out_h // pixel_size)
        small = mask_img.resize((sw, sh), Image.Resampling.BOX)
        small_arr = np.asarray(small, dtype=np.uint8)
        small_bin = (small_arr >= 32).astype(np.uint8) * 255
        small = Image.fromarray(small_bin).convert("L")
        mask_img = small.resize((out_w, out_h), Image.Resampling.NEAREST)

    # Thicken edges (dilation)
    if thickness > 0:
        size = 2 * thickness + 1
        mask_img = mask_img.filter(ImageFilter.MaxFilter(size=size))

    if invert:
        mask_img = Image.fromarray(255 - np.asarray(mask_img, dtype=np.uint8)).convert("L")

    return mask_img

def edge_mask_to_lines(mask_img: Image.Image, edge_char: str = "█") -> List[str]:
    edge_char = (edge_char or "█")[0]
    m = np.asarray(mask_img.convert("L"), dtype=np.uint8)
    return ["".join(edge_char if v > 0 else " " for v in row) for row in m]

def overlay_edges_on_lines(base_lines: List[str], mask_img: Image.Image, edge_char: str = "█") -> List[str]:
    if not base_lines:
        return base_lines
    edge_char = (edge_char or "█")[0]
    out_h = len(base_lines)
    out_w = max(len(l) for l in base_lines)
    m = np.asarray(mask_img.resize((out_w, out_h), Image.Resampling.NEAREST), dtype=np.uint8)

    out = []
    for y, line in enumerate(base_lines):
        row = list(line.ljust(out_w))
        for x in range(out_w):
            if m[y, x] > 0:
                row[x] = edge_char
        out.append("".join(row))
    return out


# ----------------------------
# Semantic segmentation
# ----------------------------
_SEG_PROC = None
_SEG_MODEL = None
_SEG_ID2LABEL = None

def _label_normalize(lbl: str) -> str:
    s = (lbl or "").lower().strip().replace("_", " ")
    s = " ".join([p for p in s.split() if p])
    return s or "object"

def ensure_seg_model():
    global _SEG_PROC, _SEG_MODEL, _SEG_ID2LABEL
    if not SEMANTIC_AVAILABLE:
        raise RuntimeError("Semantic mode requires torch + transformers.")
    if _SEG_MODEL is None:
        _SEG_PROC = SegformerImageProcessor.from_pretrained(SEG_MODEL_NAME)
        _SEG_MODEL = SegformerForSemanticSegmentation.from_pretrained(SEG_MODEL_NAME)
        _SEG_MODEL.eval()
        _SEG_ID2LABEL = _SEG_MODEL.config.id2label
    return _SEG_PROC, _SEG_MODEL, _SEG_ID2LABEL

def compute_seg_small(base_img_rgb: Image.Image) -> Tuple[np.ndarray, List[str]]:
    proc, model, id2label = ensure_seg_model()
    inputs = proc(images=base_img_rgb, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs)
        logits = out.logits
        up = torch.nn.functional.interpolate(
            logits, size=(SEG_CACHE_SIZE, SEG_CACHE_SIZE),
            mode="bilinear", align_corners=False
        )
        seg = up.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

    max_id = int(seg.max())
    lut = ["object"] * (max_id + 1)
    for i in range(max_id + 1):
        lut[i] = _label_normalize(id2label.get(i, "object"))
    return seg, lut

def seg_to_ids_grid(seg_small: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    seg_img = Image.fromarray(seg_small, mode="L")
    seg_resized = seg_img.resize((out_w, out_h), Image.Resampling.NEAREST)
    return np.asarray(seg_resized, dtype=np.uint8)


# ----------------------------
# Phrase -> char pool
# ----------------------------
def phrase_to_char_pool(phrase: str) -> List[str]:
    phrase = (phrase or "").strip()
    if not phrase:
        return ["o"]

    if "," in phrase:
        parts = [p.strip() for p in phrase.split(",") if p.strip()]
    else:
        parts = [p for p in phrase.split() if p]

    if len(parts) >= 2 and all(len(p) <= 3 for p in parts):
        seq = []
        for p in parts:
            ch = p[0]
            if ch == "_":
                ch = " "
            seq.append(ch)
        return seq if seq else ["o"]

    s = phrase.replace(" ", "")
    return [c for c in s] if s else ["o"]


# ----------------------------
# Deterministic jumble
# ----------------------------
def _splitmix64(z: int) -> int:
    z = (z + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9 & 0xFFFFFFFFFFFFFFFF
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB & 0xFFFFFFFFFFFFFFFF
    z = z ^ (z >> 31)
    return z & 0xFFFFFFFFFFFFFFFF

def pick_char_jumble(pool: List[str], x: int, y: int, cls_id: int, seed: int) -> str:
    if not pool:
        return "o"
    z = seed & 0xFFFFFFFFFFFFFFFF
    z ^= (x * 0x1F123BB5) & 0xFFFFFFFFFFFFFFFF
    z ^= (y * 0x9E3779B1) & 0xFFFFFFFFFFFFFFFF
    z ^= (cls_id * 0xC2B2AE35) & 0xFFFFFFFFFFFFFFFF
    h = _splitmix64(z)
    return pool[h % len(pool)]


# ----------------------------
# Render lines to RGBA (fast)
# ----------------------------
def render_lines_to_rgba(lines: List[str],
                         font: ImageFont.ImageFont,
                         color_map: Optional[np.ndarray],
                         transparent_bg: bool,
                         bg_rgb=(15, 15, 15),
                         fg_rgb=(230, 230, 230)) -> Image.Image:
    if not lines:
        raise ValueError("No lines to render.")
    char_w, char_h = measure_char_cell(font)
    w_chars = max(len(l) for l in lines)
    h_chars = len(lines)

    if transparent_bg:
        out = Image.new("RGBA", (w_chars * char_w, h_chars * char_h), (0, 0, 0, 0))
    else:
        out = Image.new("RGBA", (w_chars * char_w, h_chars * char_h), (*bg_rgb, 255))

    draw = ImageDraw.Draw(out, "RGBA")

    if color_map is None:
        fill = (*fg_rgb, 255)
        for y, line in enumerate(lines):
            draw.text((0, y * char_h), line, fill=fill, font=font)
        return out

    for y, line in enumerate(lines):
        if y >= color_map.shape[0]:
            draw.text((0, y * char_h), line, fill=(*fg_rgb, 255), font=font)
            continue
        x = 0
        while x < len(line):
            col = color_map[y, x] if x < color_map.shape[1] else np.array(fg_rgb, dtype=np.uint8)
            start = x
            x += 1
            while x < len(line):
                col2 = color_map[y, x] if x < color_map.shape[1] else np.array(fg_rgb, dtype=np.uint8)
                if not np.array_equal(col2, col):
                    break
                x += 1
            sub = line[start:x]
            fill = (int(col[0]), int(col[1]), int(col[2]), 255)
            draw.text((start * char_w, y * char_h), sub, fill=fill, font=font)

    return out


# ----------------------------
# SVG export helpers
# ----------------------------
def rgb_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))

def svg_embed_png(pil_img: Image.Image, transparent_bg: bool, bg_rgb=(15, 15, 15)) -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    w, h = pil_img.size
    rect = "" if transparent_bg else f'<rect width="100%" height="100%" fill="{rgb_hex(bg_rgb)}"/>'
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">
  {rect}
  <image x="0" y="0" width="{w}" height="{h}" href="data:image/png;base64,{b64}" />
</svg>'''

def svg_text_export(lines: List[str],
                    color_map: Optional[np.ndarray],
                    font_family: str,
                    font_size_px: int,
                    char_w: int,
                    line_h: int,
                    transparent_bg: bool,
                    bg_rgb=(15, 15, 15)) -> str:
    if not lines:
        raise ValueError("No text")
    pad = 2
    h_chars = len(lines)
    w_chars = max(len(l) for l in lines)
    width_px = w_chars * char_w + 2 * pad
    height_px = h_chars * line_h + 2 * pad

    bg_rect = "" if transparent_bg else f'<rect width="100%" height="100%" fill="{rgb_hex(bg_rgb)}"/>'
    fg_hex = rgb_hex((230, 230, 230))

    style = f"""
    <style>
      text {{
        font-family: {xml_escape(font_family)}, Consolas, "Cascadia Mono", "Courier New", monospace;
        font-size: {font_size_px}px;
        white-space: pre;
        font-variant-ligatures: none;
        font-feature-settings: "liga" 0, "calt" 0;
      }}
    </style>
    """

    out = []
    out.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width_px}" height="{height_px}" viewBox="0 0 {width_px} {height_px}">')
    out.append(style.strip())
    if bg_rect:
        out.append(bg_rect)

    for y, line in enumerate(lines):
        y_px = pad + y * line_h
        if color_map is None:
            out.append(f'<text x="{pad}" y="{y_px}" fill="{fg_hex}" dominant-baseline="hanging" xml:space="preserve">{xml_escape(line)}</text>')
        else:
            out.append(f'<text x="{pad}" y="{y_px}" dominant-baseline="hanging" xml:space="preserve">')
            x = 0
            while x < len(line):
                col = color_map[y, x] if (y < color_map.shape[0] and x < color_map.shape[1]) else np.array([230, 230, 230], dtype=np.uint8)
                run_color = rgb_hex(col)
                start = x
                x += 1
                while x < len(line):
                    col2 = color_map[y, x] if (y < color_map.shape[0] and x < color_map.shape[1]) else np.array([230, 230, 230], dtype=np.uint8)
                    if not np.array_equal(col2, col):
                        break
                    x += 1
                out.append(f'<tspan fill="{run_color}">{xml_escape(line[start:x])}</tspan>')
            out.append('</text>')
    out.append('</svg>')
    return "\n".join(out)


# ----------------------------
# PIL -> QImage
# ----------------------------
def pil_to_qimage(img: Image.Image) -> QtGui.QImage:
    rgba = img.convert("RGBA")
    data = rgba.tobytes("raw", "RGBA")
    qimg = QtGui.QImage(data, rgba.size[0], rgba.size[1], QtGui.QImage.Format_RGBA8888)
    return qimg.copy()


# ----------------------------
# UI Widgets
# ----------------------------
class ImageView(QtWidgets.QLabel):
    def __init__(self):
        super().__init__()
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setMinimumSize(320, 220)
        self._pixmap: Optional[QtGui.QPixmap] = None
        self.setStyleSheet("QLabel { background: #0F0F12; border-radius: 10px; border: 1px solid #2B2B30; }")

    def set_image(self, qimage: Optional[QtGui.QImage]):
        self._pixmap = QtGui.QPixmap.fromImage(qimage) if qimage is not None else None
        self._update_scaled()

    def resizeEvent(self, event: QtGui.QResizeEvent):
        super().resizeEvent(event)
        self._update_scaled()

    def _update_scaled(self):
        if self._pixmap is None:
            self.setPixmap(QtGui.QPixmap())
            return
        scaled = self._pixmap.scaled(self.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self.setPixmap(scaled)

class DoubleClickResetSlider(QtWidgets.QSlider):
    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent):
        self.setValue(0)
        event.accept()


# ----------------------------
# Render worker
# ----------------------------
@dataclass
class RenderParams:
    content_mode: str
    fill_style: str
    seed: int

    width_chars: int
    use_preview_cap: bool
    preview_cap: int

    invert: bool
    charset: str

    brightness: int
    contrast: int
    ambiance: int
    saturation: int
    highlights: int
    shadows: int

    font_path: str
    font_size: int
    ttc_index: int

    color_mode: str
    k: int
    allow_kmeans: bool

    transparent_bg: bool

    seg_small: Optional[np.ndarray]
    seg_lut: Optional[List[str]]

    label_map: Dict[str, str]
    hidden_labels: List[str]

    # Pixel edges / overlay
    edge_threshold: int
    edge_pixel_size: int
    edge_thickness: int
    edge_invert: bool
    edge_char: str
    edge_bg: str      # for Pixel Edges mode only
    edge_fg: str      # used in Pixel Edges + Overlay

@dataclass
class RenderResult:
    content_mode: str
    preview_qimage: QtGui.QImage
    render_qimage: QtGui.QImage

    lines: List[str]
    color_map: Optional[np.ndarray]

    font_path: str
    font_size: int
    ttc_index: int
    transparent_bg: bool

    seg_small: Optional[np.ndarray]
    seg_lut: Optional[List[str]]

class WorkerSignals(QtCore.QObject):
    result = QtCore.Signal(int, object)
    error = QtCore.Signal(int, str)

class RenderWorker(QtCore.QRunnable):
    def __init__(self, request_id: int, base_img: Image.Image, params: RenderParams):
        super().__init__()
        self.request_id = request_id
        self.base_img = base_img
        self.params = params
        self.signals = WorkerSignals()

    @QtCore.Slot()
    def run(self):
        try:
            font = safe_load_pil_font(self.params.font_path, self.params.font_size, self.params.ttc_index)
            char_w, char_h = measure_char_cell(font)

            adjusted = apply_adjustments(
                self.base_img,
                self.params.brightness, self.params.contrast, self.params.ambiance,
                self.params.saturation, self.params.highlights, self.params.shadows
            )

            out_w = int(self.params.width_chars)
            if self.params.use_preview_cap and self.params.preview_cap > 0:
                out_w = min(out_w, int(self.params.preview_cap))

            iw, ih = adjusted.size
            out_h = max(1, int(round((ih / iw) * out_w * (char_w / char_h))))
            resized_rgb = adjusted.resize((out_w, out_h), Image.Resampling.BILINEAR)

            # Base color map aligned to resized_rgb (used for ASCII/Overlay; Semantic does its own)
            base_color_map = None
            if self.params.color_mode == "Color (Direct)":
                base_color_map = build_color_map(resized_rgb, "Color (Direct)", self.params.k)
            elif self.params.color_mode == "Color (K-means)":
                if self.params.allow_kmeans:
                    base_color_map = build_color_map(resized_rgb, "Color (K-means)", self.params.k)
                else:
                    base_color_map = build_color_map(resized_rgb, "Color (Direct)", self.params.k)

            # Defaults for background/foreground
            bg_rgb = (15, 15, 15)
            fg_rgb = (230, 230, 230)

            mode = self.params.content_mode

            if mode == "ASCII":
                lines, rgb_r, _dims = make_ascii_lines(
                    adjusted,
                    out_w,
                    self.params.charset,
                    self.params.invert,
                    char_w, char_h
                )
                # re-align color map to rgb_r
                color_map = None
                if self.params.color_mode != "Mono":
                    if self.params.color_mode == "Color (Direct)":
                        color_map = build_color_map(rgb_r, "Color (Direct)", self.params.k)
                    elif self.params.color_mode == "Color (K-means)":
                        if self.params.allow_kmeans:
                            color_map = build_color_map(rgb_r, "Color (K-means)", self.params.k)
                        else:
                            color_map = build_color_map(rgb_r, "Color (Direct)", self.params.k)
                render_pil = render_lines_to_rgba(
                    lines=lines,
                    font=font,
                    color_map=color_map,
                    transparent_bg=self.params.transparent_bg,
                    bg_rgb=bg_rgb,
                    fg_rgb=fg_rgb
                )
                final_color_map = color_map

            elif mode == "Pixel Edges":
                # Pixelated edge mask -> lines
                mask = make_pixel_edge_mask(
                    adjusted_img=adjusted,
                    out_w=out_w,
                    out_h=out_h,
                    pixel_size=self.params.edge_pixel_size,
                    threshold=self.params.edge_threshold,
                    thickness=self.params.edge_thickness,
                    invert=self.params.edge_invert
                )
                lines = edge_mask_to_lines(mask, self.params.edge_char)

                # Pixel Edges specific fg/bg colors
                fg_rgb = (0, 0, 0) if self.params.edge_fg == "Black" else (235, 235, 235)
                bg_rgb = (255, 255, 255) if self.params.edge_bg == "White" else (15, 15, 15)

                render_pil = render_lines_to_rgba(
                    lines=lines,
                    font=font,
                    color_map=None,
                    transparent_bg=self.params.transparent_bg,
                    bg_rgb=bg_rgb,
                    fg_rgb=fg_rgb
                )
                final_color_map = None

            elif mode == "Edge Overlay":
                # Base ASCII
                base_lines, rgb_r, _ = make_ascii_lines(
                    adjusted,
                    out_w,
                    self.params.charset,
                    self.params.invert,
                    char_w, char_h
                )

                # Base colors (per-char) if selected; else fill with default fg
                if self.params.color_mode == "Mono":
                    base_cm = np.zeros((len(base_lines), max(len(l) for l in base_lines), 3), dtype=np.uint8)
                    base_cm[:, :, 0] = fg_rgb[0]
                    base_cm[:, :, 1] = fg_rgb[1]
                    base_cm[:, :, 2] = fg_rgb[2]
                else:
                    if self.params.color_mode == "Color (Direct)":
                        base_cm = build_color_map(rgb_r, "Color (Direct)", self.params.k)
                    else:
                        if self.params.allow_kmeans:
                            base_cm = build_color_map(rgb_r, "Color (K-means)", self.params.k)
                        else:
                            base_cm = build_color_map(rgb_r, "Color (Direct)", self.params.k)

                # Edge mask at same out_w/out_h
                mask = make_pixel_edge_mask(
                    adjusted_img=adjusted,
                    out_w=out_w,
                    out_h=len(base_lines),
                    pixel_size=self.params.edge_pixel_size,
                    threshold=self.params.edge_threshold,
                    thickness=self.params.edge_thickness,
                    invert=self.params.edge_invert
                )

                # Replace chars at edges + override their color
                lines = overlay_edges_on_lines(base_lines, mask, self.params.edge_char)

                edge_rgb = (0, 0, 0) if self.params.edge_fg == "Black" else (235, 235, 235)
                m = np.asarray(mask.resize((base_cm.shape[1], base_cm.shape[0]), Image.Resampling.NEAREST), dtype=np.uint8)
                cm = base_cm.copy()
                cm[m > 0] = np.array(edge_rgb, dtype=np.uint8)

                render_pil = render_lines_to_rgba(
                    lines=lines,
                    font=font,
                    color_map=cm,
                    transparent_bg=self.params.transparent_bg,
                    bg_rgb=bg_rgb,
                    fg_rgb=fg_rgb
                )
                final_color_map = cm

            else:
                # Semantic Letters
                if not SEMANTIC_AVAILABLE:
                    raise RuntimeError("Semantic mode requires torch + transformers.")

                seg_small = self.params.seg_small
                seg_lut = self.params.seg_lut
                if seg_small is None or seg_lut is None:
                    seg_small, seg_lut = compute_seg_small(self.base_img)

                ids_grid = seg_to_ids_grid(seg_small, out_w, out_h)
                max_id = int(ids_grid.max())
                if max_id >= len(seg_lut):
                    seg_lut = seg_lut + ["object"] * (max_id + 1 - len(seg_lut))

                hidden_set = set([s.strip().lower() for s in self.params.hidden_labels if s])

                id_to_pool: Dict[int, List[str]] = {}
                for cid in np.unique(ids_grid):
                    lbl = seg_lut[int(cid)].strip().lower()
                    if lbl in hidden_set:
                        id_to_pool[int(cid)] = [" "]
                    else:
                        phrase = (self.params.label_map.get(lbl, "") or "").strip() or lbl
                        id_to_pool[int(cid)] = phrase_to_char_pool(phrase)

                seed = int(self.params.seed) & 0xFFFFFFFFFFFFFFFF
                cycle = (self.params.fill_style == "Cycle")

                lines = []
                for y in range(out_h):
                    row = []
                    row_ids = ids_grid[y]
                    for x in range(out_w):
                        cid = int(row_ids[x])
                        pool = id_to_pool.get(cid) or ["o"]
                        if cycle:
                            idx = (x + y * 131 + cid * 17) % len(pool)
                            ch = pool[idx]
                        else:
                            ch = pick_char_jumble(pool, x, y, cid, seed)
                        row.append(ch[0] if ch else " ")
                    lines.append("".join(row))

                # Push cache back
                self.params.seg_small = seg_small
                self.params.seg_lut = seg_lut

                # Semantic uses normal color selection
                color_map = None
                if self.params.color_mode != "Mono":
                    if self.params.color_mode == "Color (Direct)":
                        color_map = build_color_map(resized_rgb, "Color (Direct)", self.params.k)
                    elif self.params.color_mode == "Color (K-means)":
                        if self.params.allow_kmeans:
                            color_map = build_color_map(resized_rgb, "Color (K-means)", self.params.k)
                        else:
                            color_map = build_color_map(resized_rgb, "Color (Direct)", self.params.k)

                render_pil = render_lines_to_rgba(
                    lines=lines,
                    font=font,
                    color_map=color_map,
                    transparent_bg=self.params.transparent_bg,
                    bg_rgb=bg_rgb,
                    fg_rgb=fg_rgb
                )
                final_color_map = color_map

            rr = RenderResult(
                content_mode=mode,
                preview_qimage=pil_to_qimage(adjusted),
                render_qimage=pil_to_qimage(render_pil),
                lines=lines,
                color_map=final_color_map,
                font_path=self.params.font_path,
                font_size=self.params.font_size,
                ttc_index=self.params.ttc_index,
                transparent_bg=self.params.transparent_bg,
                seg_small=self.params.seg_small,
                seg_lut=self.params.seg_lut
            )
            self.signals.result.emit(self.request_id, rr)

        except Exception:
            self.signals.error.emit(self.request_id, traceback.format_exc())


# ----------------------------
# Main Window
# ----------------------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FINAL: ASCII + Semantic + Pixel Edges + Edge Overlay")
        self.resize(1580, 950)

        self.thread_pool = QtCore.QThreadPool.globalInstance()
        self.thread_pool.setMaxThreadCount(1)
        self._active_workers: Dict[int, RenderWorker] = {}

        self.base_img: Optional[Image.Image] = None
        self.last_result: Optional[RenderResult] = None

        self.request_id = 0
        self.last_applied_id = 0

        self.seg_small: Optional[np.ndarray] = None
        self.seg_lut: Optional[List[str]] = None

        self.label_map: Dict[str, str] = {}
        self.hidden_labels: Set[str] = set()

        self.is_dragging = False

        self.font_files = list_windows_font_files()
        self.font_map: Dict[str, str] = {lbl: p for lbl, p in self.font_files}
        self._font_family_cache: Dict[str, str] = {}

        self.fast_timer = QtCore.QTimer(self)
        self.fast_timer.setSingleShot(True)
        self.fast_timer.timeout.connect(self.dispatch_fast)

        self.kmeans_timer = QtCore.QTimer(self)
        self.kmeans_timer.setSingleShot(True)
        self.kmeans_timer.timeout.connect(self.dispatch_kmeans)

        self._build_ui()
        self.set_controls_enabled(False)

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(12)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        root.addWidget(splitter)

        # LEFT previews
        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)

        tabs = QtWidgets.QTabWidget()
        self.image_view = ImageView()
        self.render_view = ImageView()

        t1 = QtWidgets.QWidget()
        l1 = QtWidgets.QVBoxLayout(t1)
        l1.addWidget(self.image_view)

        t2 = QtWidgets.QWidget()
        l2 = QtWidgets.QVBoxLayout(t2)
        l2.addWidget(self.render_view)

        tabs.addTab(t1, "Image Preview")
        tabs.addTab(t2, "Render Preview")
        left_layout.addWidget(tabs)
        splitter.addWidget(left)

        # RIGHT controls + optional text
        right = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)

        container = QtWidgets.QWidget()
        self.controls_layout = QtWidgets.QVBoxLayout(container)
        self.controls_layout.setContentsMargins(0, 0, 0, 0)
        self.controls_layout.setSpacing(10)
        scroll.setWidget(container)

        right_layout.addWidget(scroll, stretch=3)

        out_group = QtWidgets.QGroupBox("Text Output (optional)")
        out_layout = QtWidgets.QVBoxLayout(out_group)
        row = QtWidgets.QHBoxLayout()
        self.chk_live_text = QtWidgets.QCheckBox("Live text update (slower)")
        self.chk_live_text.setChecked(False)
        row.addWidget(self.chk_live_text)

        btn_copy = QtWidgets.QPushButton("Copy latest text")
        btn_copy.clicked.connect(self.copy_latest_text)
        row.addWidget(btn_copy)

        btn_save = QtWidgets.QPushButton("Save latest .txt")
        btn_save.clicked.connect(self.save_latest_txt)
        row.addWidget(btn_save)

        row.addStretch(1)
        out_layout.addLayout(row)

        self.output_text = QtWidgets.QTextEdit()
        self.output_text.setFont(QtGui.QFont("Consolas", 9))
        out_layout.addWidget(self.output_text)

        right_layout.addWidget(out_group, stretch=2)

        splitter.addWidget(right)
        splitter.setSizes([940, 640])

        self.status = QtWidgets.QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Open an image to begin.")

        self._build_groups()

    def _build_groups(self):
        # File
        g_file = QtWidgets.QGroupBox("File")
        h = QtWidgets.QHBoxLayout(g_file)

        btn_open = QtWidgets.QPushButton("Open")
        btn_open.clicked.connect(self.open_image)
        h.addWidget(btn_open)

        btn_export_png = QtWidgets.QPushButton("Export PNG")
        btn_export_png.clicked.connect(self.export_png)
        h.addWidget(btn_export_png)

        btn_export_svg = QtWidgets.QPushButton("Export SVG…")
        btn_export_svg.clicked.connect(self.export_svg)
        h.addWidget(btn_export_svg)

        h.addStretch(1)
        self.controls_layout.addWidget(g_file)

        # Render settings
        self.g_render = QtWidgets.QGroupBox("Render Settings")
        grid = QtWidgets.QGridLayout(self.g_render)

        grid.addWidget(QtWidgets.QLabel("Content"), 0, 0)
        self.cmb_content = QtWidgets.QComboBox()
        self.cmb_content.addItems(CONTENT_MODES)
        self.cmb_content.currentIndexChanged.connect(self.on_content_changed)
        grid.addWidget(self.cmb_content, 0, 1)

        if not SEMANTIC_AVAILABLE:
            model = self.cmb_content.model()
            it = model.item(1)  # Semantic Letters
            if it is not None:
                it.setEnabled(False)
            self.cmb_content.setToolTip("Semantic Letters requires: torch + transformers")

        grid.addWidget(QtWidgets.QLabel("Fill style"), 0, 2)
        self.cmb_fill = QtWidgets.QComboBox()
        self.cmb_fill.addItems(FILL_STYLES)
        self.cmb_fill.setCurrentText("Jumble")
        self.cmb_fill.currentIndexChanged.connect(self.schedule_fast)
        grid.addWidget(self.cmb_fill, 0, 3)

        grid.addWidget(QtWidgets.QLabel("Seed"), 1, 2)
        self.spin_seed = QtWidgets.QSpinBox()
        self.spin_seed.setRange(0, 2_000_000_000)
        self.spin_seed.setValue(1337)
        self.spin_seed.valueChanged.connect(self.schedule_fast)
        grid.addWidget(self.spin_seed, 1, 3)

        grid.addWidget(QtWidgets.QLabel("Width (chars)"), 1, 0)
        self.spin_width = QtWidgets.QSpinBox()
        self.spin_width.setRange(20, 700)
        self.spin_width.setValue(200)
        self.spin_width.valueChanged.connect(self.schedule_fast)
        grid.addWidget(self.spin_width, 1, 1)

        self.chk_invert = QtWidgets.QCheckBox("Invert (ASCII)")
        self.chk_invert.stateChanged.connect(self.schedule_fast)
        grid.addWidget(self.chk_invert, 2, 0, 1, 2)

        grid.addWidget(QtWidgets.QLabel("Color"), 2, 2)
        self.cmb_color = QtWidgets.QComboBox()
        self.cmb_color.addItems(COLOR_MODES)
        self.cmb_color.currentIndexChanged.connect(self.schedule_fast)
        grid.addWidget(self.cmb_color, 2, 3)

        grid.addWidget(QtWidgets.QLabel("K"), 3, 2)
        self.spin_k = QtWidgets.QSpinBox()
        self.spin_k.setRange(2, 32)
        self.spin_k.setValue(8)
        self.spin_k.valueChanged.connect(self.schedule_fast)
        grid.addWidget(self.spin_k, 3, 3)

        self.chk_transparent = QtWidgets.QCheckBox("Transparent background")
        self.chk_transparent.setChecked(False)
        self.chk_transparent.stateChanged.connect(self.schedule_fast)
        grid.addWidget(self.chk_transparent, 3, 0, 1, 2)

        self.chk_preview_cap = QtWidgets.QCheckBox("Preview cap while dragging")
        self.chk_preview_cap.setChecked(True)
        self.chk_preview_cap.stateChanged.connect(self.schedule_fast)
        grid.addWidget(self.chk_preview_cap, 4, 0, 1, 2)

        grid.addWidget(QtWidgets.QLabel("Cap (chars)"), 4, 2)
        self.spin_preview_cap = QtWidgets.QSpinBox()
        self.spin_preview_cap.setRange(40, 500)
        self.spin_preview_cap.setValue(DEFAULT_PREVIEW_CAP)
        self.spin_preview_cap.valueChanged.connect(self.schedule_fast)
        grid.addWidget(self.spin_preview_cap, 4, 3)

        self.controls_layout.addWidget(self.g_render)

        # Font
        self.g_font = QtWidgets.QGroupBox("Font (applies to ASCII + Semantic + Edges)")
        grid = QtWidgets.QGridLayout(self.g_font)

        grid.addWidget(QtWidgets.QLabel("Font (Windows)"), 0, 0)
        self.cmb_font = QtWidgets.QComboBox()
        if self.font_files:
            self.cmb_font.addItems([lbl for lbl, _ in self.font_files])
            prefer = ["consola.ttf", "cascadiamono.ttf", "lucon.ttf", "cour.ttf"]
            labels_lower = [lbl.lower() for lbl, _ in self.font_files]
            for p in prefer:
                if p in labels_lower:
                    self.cmb_font.setCurrentIndex(labels_lower.index(p))
                    break
        else:
            self.cmb_font.addItem("(No fonts found)")
        self.cmb_font.currentIndexChanged.connect(self.schedule_fast)
        grid.addWidget(self.cmb_font, 0, 1, 1, 2)

        grid.addWidget(QtWidgets.QLabel("Size"), 0, 3)
        self.spin_font_size = QtWidgets.QSpinBox()
        self.spin_font_size.setRange(8, 64)
        self.spin_font_size.setValue(12)
        self.spin_font_size.valueChanged.connect(self.schedule_fast)
        grid.addWidget(self.spin_font_size, 0, 4)

        grid.addWidget(QtWidgets.QLabel("TTC Index"), 1, 3)
        self.spin_ttc = QtWidgets.QSpinBox()
        self.spin_ttc.setRange(0, 16)
        self.spin_ttc.setValue(0)
        self.spin_ttc.valueChanged.connect(self.schedule_fast)
        grid.addWidget(self.spin_ttc, 1, 4)

        note = QtWidgets.QLabel("If your font is .ttc and doesn’t change, try TTC Index 0–3.")
        note.setWordWrap(True)
        grid.addWidget(note, 1, 0, 1, 3)

        self.controls_layout.addWidget(self.g_font)

        # ASCII charset
        self.g_charset = QtWidgets.QGroupBox("ASCII Charset (ASCII + Edge Overlay)")
        grid = QtWidgets.QGridLayout(self.g_charset)

        self.rb_preset = QtWidgets.QRadioButton("Preset")
        self.rb_custom = QtWidgets.QRadioButton("Custom")
        self.rb_preset.setChecked(True)
        self.rb_preset.toggled.connect(self.schedule_fast)
        grid.addWidget(self.rb_preset, 0, 0)
        grid.addWidget(self.rb_custom, 0, 1)

        self.cmb_charset = QtWidgets.QComboBox()
        self.cmb_charset.addItems(list(PRESET_CHARSETS.keys()))
        self.cmb_charset.currentIndexChanged.connect(self.schedule_fast)
        grid.addWidget(self.cmb_charset, 0, 2)

        self.edit_charset = QtWidgets.QLineEdit()
        self.edit_charset.setPlaceholderText("@%#*+=-:. ")
        self.edit_charset.textChanged.connect(self.schedule_fast)
        grid.addWidget(self.edit_charset, 1, 0, 1, 3)

        self.controls_layout.addWidget(self.g_charset)

        # Pixel Edges settings (Pixel Edges + Edge Overlay)
        self.g_edges = QtWidgets.QGroupBox("Pixel Edges (for Pixel Edges / Edge Overlay)")
        grid = QtWidgets.QGridLayout(self.g_edges)

        grid.addWidget(QtWidgets.QLabel("Threshold"), 0, 0)
        self.spin_edge_thresh = QtWidgets.QSpinBox()
        self.spin_edge_thresh.setRange(0, 255)
        self.spin_edge_thresh.setValue(70)
        self.spin_edge_thresh.valueChanged.connect(self.schedule_fast)
        grid.addWidget(self.spin_edge_thresh, 0, 1)

        grid.addWidget(QtWidgets.QLabel("Pixel size"), 0, 2)
        self.spin_edge_pix = QtWidgets.QSpinBox()
        self.spin_edge_pix.setRange(1, 40)
        self.spin_edge_pix.setValue(6)
        self.spin_edge_pix.valueChanged.connect(self.schedule_fast)
        grid.addWidget(self.spin_edge_pix, 0, 3)

        grid.addWidget(QtWidgets.QLabel("Thickness"), 1, 0)
        self.spin_edge_thick = QtWidgets.QSpinBox()
        self.spin_edge_thick.setRange(0, 10)
        self.spin_edge_thick.setValue(2)
        self.spin_edge_thick.valueChanged.connect(self.schedule_fast)
        grid.addWidget(self.spin_edge_thick, 1, 1)

        self.chk_edge_invert = QtWidgets.QCheckBox("Invert edges")
        self.chk_edge_invert.stateChanged.connect(self.schedule_fast)
        grid.addWidget(self.chk_edge_invert, 1, 2, 1, 2)

        grid.addWidget(QtWidgets.QLabel("Edge char"), 2, 0)
        self.edit_edge_char = QtWidgets.QLineEdit("█")
        self.edit_edge_char.setMaxLength(2)
        self.edit_edge_char.textChanged.connect(self.schedule_fast)
        grid.addWidget(self.edit_edge_char, 2, 1)

        grid.addWidget(QtWidgets.QLabel("Edge FG"), 2, 2)
        self.cmb_edge_fg = QtWidgets.QComboBox()
        self.cmb_edge_fg.addItems(EDGE_FG_MODES)
        self.cmb_edge_fg.setCurrentText("White")
        self.cmb_edge_fg.currentIndexChanged.connect(self.schedule_fast)
        grid.addWidget(self.cmb_edge_fg, 2, 3)

        grid.addWidget(QtWidgets.QLabel("Edge BG (Pixel Edges only)"), 3, 0, 1, 2)
        self.cmb_edge_bg = QtWidgets.QComboBox()
        self.cmb_edge_bg.addItems(EDGE_BG_MODES)
        self.cmb_edge_bg.setCurrentText("White")
        self.cmb_edge_bg.currentIndexChanged.connect(self.schedule_fast)
        grid.addWidget(self.cmb_edge_bg, 3, 2, 1, 2)

        hint = QtWidgets.QLabel("Reference look: Threshold=60–90, Pixel size=4–7, Thickness=2–3. Overlay uses your ASCII base + edges on top.")
        hint.setWordWrap(True)
        grid.addWidget(hint, 4, 0, 1, 4)

        self.controls_layout.addWidget(self.g_edges)

        # Semantics management
        self.g_sem = QtWidgets.QGroupBox("Semantics: manage labels (replace / delete / hide all except)")
        v = QtWidgets.QVBoxLayout(self.g_sem)

        row = QtWidgets.QHBoxLayout()
        self.btn_detect = QtWidgets.QPushButton("Detect labels")
        self.btn_detect.clicked.connect(self.detect_labels)
        row.addWidget(self.btn_detect)

        self.btn_hide_all_except = QtWidgets.QPushButton("Hide all except selected")
        self.btn_hide_all_except.clicked.connect(self.hide_all_except_selected)
        row.addWidget(self.btn_hide_all_except)

        self.btn_unhide_all = QtWidgets.QPushButton("Unhide all")
        self.btn_unhide_all.clicked.connect(self.unhide_all)
        row.addWidget(self.btn_unhide_all)

        self.btn_import = QtWidgets.QPushButton("Import JSON")
        self.btn_import.clicked.connect(self.import_json)
        row.addWidget(self.btn_import)

        self.btn_export = QtWidgets.QPushButton("Export JSON")
        self.btn_export.clicked.connect(self.export_json)
        row.addWidget(self.btn_export)

        row.addStretch(1)
        v.addLayout(row)

        row2 = QtWidgets.QHBoxLayout()
        self.edit_add_label = QtWidgets.QLineEdit()
        self.edit_add_label.setPlaceholderText("label (e.g., tree)")
        row2.addWidget(self.edit_add_label)

        self.edit_add_phrase = QtWidgets.QLineEdit()
        self.edit_add_phrase.setPlaceholderText("replacement (e.g., cherry blossom) OR sequence: t r e T r e")
        row2.addWidget(self.edit_add_phrase)

        self.chk_add_hide = QtWidgets.QCheckBox("Hide")
        row2.addWidget(self.chk_add_hide)

        self.btn_add_row = QtWidgets.QPushButton("Add/Update")
        self.btn_add_row.clicked.connect(self.add_update_mapping)
        row2.addWidget(self.btn_add_row)

        self.btn_remove = QtWidgets.QPushButton("Remove selected")
        self.btn_remove.clicked.connect(self.remove_selected_mappings)
        row2.addWidget(self.btn_remove)

        v.addLayout(row2)

        self.lbl_sem_status = QtWidgets.QLabel("—")
        self.lbl_sem_status.setWordWrap(True)
        v.addWidget(self.lbl_sem_status)

        self.table_sem = QtWidgets.QTableWidget(0, 3)
        self.table_sem.setHorizontalHeaderLabels(["Label", "Replacement phrase", "Hide"])
        self.table_sem.horizontalHeader().setStretchLastSection(False)
        self.table_sem.setColumnWidth(2, 70)
        self.table_sem.horizontalHeader().setStretchLastSection(True)
        self.table_sem.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table_sem.setEditTriggers(QtWidgets.QAbstractItemView.DoubleClicked | QtWidgets.QAbstractItemView.EditKeyPressed)
        self.table_sem.itemChanged.connect(self.on_table_changed)
        v.addWidget(self.table_sem)

        hint = QtWidgets.QLabel(
            "Delete vs Replace:\n"
            "- Replace: set a phrase; region uses characters from phrase.\n"
            "- Delete: check Hide => region becomes spaces (and transparent if background is transparent).\n\n"
            "Tip: blank replacement means 'use label itself'."
        )
        hint.setWordWrap(True)
        v.addWidget(hint)

        self.controls_layout.addWidget(self.g_sem)

        # Adjustments
        self.g_adj = QtWidgets.QGroupBox("Adjustments (double-click slider = reset)")
        grid = QtWidgets.QGridLayout(self.g_adj)
        self.sliders: Dict[str, QtWidgets.QSlider] = {}

        def add_slider(row, name):
            lbl = QtWidgets.QLabel(name)
            val = QtWidgets.QLabel("0")
            val.setFixedWidth(30)
            val.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)

            s = DoubleClickResetSlider(QtCore.Qt.Horizontal)
            s.setRange(-100, 100)
            s.setValue(0)

            s.sliderPressed.connect(self.on_slider_pressed)
            s.sliderReleased.connect(self.on_slider_released)

            def on_change(v):
                val.setText(str(v))
                self.schedule_fast()

            s.valueChanged.connect(on_change)

            grid.addWidget(lbl, row, 0)
            grid.addWidget(s, row, 1, 1, 3)
            grid.addWidget(val, row, 4)
            self.sliders[name] = s

        for i, nm in enumerate(["Brightness", "Contrast", "Ambiance", "Saturation", "Highlights", "Shadows"]):
            add_slider(i, nm)

        btn_reset_all = QtWidgets.QPushButton("Reset All Adjustments")
        btn_reset_all.clicked.connect(self.reset_all_adjustments)
        grid.addWidget(btn_reset_all, 6, 0, 1, 5)

        self.controls_layout.addWidget(self.g_adj)
        self.controls_layout.addStretch(1)

        # Set initial toggles
        self.on_content_changed()

    def set_controls_enabled(self, enabled: bool):
        for g in [self.g_render, self.g_font, self.g_charset, self.g_edges, self.g_sem, self.g_adj]:
            g.setEnabled(enabled)

    # ----------------------------
    # Slider drag
    # ----------------------------
    def on_slider_pressed(self):
        self.is_dragging = True

    def on_slider_released(self):
        self.is_dragging = False
        self.dispatch_render(allow_kmeans=True)

    def reset_all_adjustments(self):
        for s in self.sliders.values():
            s.setValue(0)
        self.schedule_fast()

    # ----------------------------
    # Content changed
    # ----------------------------
    def on_content_changed(self):
        mode = self.cmb_content.currentText()

        is_ascii_like = mode in ("ASCII", "Edge Overlay")
        is_sem = (mode == "Semantic Letters")
        is_edges = mode in ("Pixel Edges", "Edge Overlay")
        is_edges_only = (mode == "Pixel Edges")

        self.g_charset.setEnabled(is_ascii_like)
        self.chk_invert.setEnabled(is_ascii_like)

        self.g_edges.setEnabled(is_edges)

        self.g_sem.setEnabled(is_sem and SEMANTIC_AVAILABLE)

        # Color controls: not used in Pixel Edges (edges-only)
        self.cmb_color.setEnabled(not is_edges_only)
        self.spin_k.setEnabled((not is_edges_only) and (self.cmb_color.currentText() == "Color (K-means)"))

        # Fill/seed only used in Semantic Letters (but harmless elsewhere)
        self.cmb_fill.setEnabled(is_sem and SEMANTIC_AVAILABLE)
        self.spin_seed.setEnabled(is_sem and SEMANTIC_AVAILABLE)

        self.schedule_fast()

    # ----------------------------
    # Semantics table helpers
    # ----------------------------
    def _make_hide_item(self, hidden: bool) -> QtWidgets.QTableWidgetItem:
        it = QtWidgets.QTableWidgetItem("")
        it.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsSelectable)
        it.setCheckState(QtCore.Qt.Checked if hidden else QtCore.Qt.Unchecked)
        return it

    def _find_label_row(self, lbl: str) -> Optional[int]:
        lbl = lbl.strip().lower()
        for r in range(self.table_sem.rowCount()):
            it = self.table_sem.item(r, 0)
            if it and it.text().strip().lower() == lbl:
                return r
        return None

    def _refresh_table_from_maps(self):
        self.table_sem.blockSignals(True)
        self.table_sem.setRowCount(0)
        keys = sorted(set(self.label_map.keys()) | set(self.hidden_labels))
        for lbl in keys:
            r = self.table_sem.rowCount()
            self.table_sem.insertRow(r)

            item_lbl = QtWidgets.QTableWidgetItem(lbl)
            item_lbl.setFlags(item_lbl.flags() & ~QtCore.Qt.ItemIsEditable)
            self.table_sem.setItem(r, 0, item_lbl)

            self.table_sem.setItem(r, 1, QtWidgets.QTableWidgetItem(self.label_map.get(lbl, "")))
            self.table_sem.setItem(r, 2, self._make_hide_item(lbl in self.hidden_labels))
        self.table_sem.blockSignals(False)

    # ----------------------------
    # Detect labels
    # ----------------------------
    def detect_labels(self):
        if not SEMANTIC_AVAILABLE:
            QtWidgets.QMessageBox.information(
                self, "Semantic not available",
                "Install:\n\npip install transformers\npip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
            )
            return
        if self.base_img is None:
            return

        self.lbl_sem_status.setText("Detecting labels… (first time may download model)")
        QtWidgets.QApplication.processEvents()

        try:
            if self.seg_small is None or self.seg_lut is None:
                self.seg_small, self.seg_lut = compute_seg_small(self.base_img)

            flat = self.seg_small.reshape(-1)
            counts = np.bincount(flat, minlength=max(int(flat.max()) + 1, len(self.seg_lut)))
            pairs = []
            for cid, cnt in enumerate(counts):
                if cnt <= 0:
                    continue
                if cid >= len(self.seg_lut):
                    continue
                pairs.append((int(cnt), self.seg_lut[cid]))
            pairs.sort(reverse=True)

            top = []
            seen = set()
            for cnt, lbl in pairs:
                if lbl in seen:
                    continue
                seen.add(lbl)
                top.append(lbl.strip().lower())
                if len(top) >= 30:
                    break

            self.table_sem.blockSignals(True)
            self.table_sem.setRowCount(0)
            for lbl in top:
                r = self.table_sem.rowCount()
                self.table_sem.insertRow(r)

                item_lbl = QtWidgets.QTableWidgetItem(lbl)
                item_lbl.setFlags(item_lbl.flags() & ~QtCore.Qt.ItemIsEditable)
                self.table_sem.setItem(r, 0, item_lbl)

                self.table_sem.setItem(r, 1, QtWidgets.QTableWidgetItem(self.label_map.get(lbl, "")))
                self.table_sem.setItem(r, 2, self._make_hide_item(lbl in self.hidden_labels))
            self.table_sem.blockSignals(False)

            self.lbl_sem_status.setText("Labels detected. Edit replacement or tick Hide to delete regions.")
            self.schedule_fast()

        except Exception:
            print(traceback.format_exc())
            self.lbl_sem_status.setText("Error detecting labels (see console).")

    # ----------------------------
    # Hide all except selected
    # ----------------------------
    def hide_all_except_selected(self):
        if self.base_img is None:
            return

        selected_rows = {idx.row() for idx in self.table_sem.selectionModel().selectedRows()}
        if not selected_rows:
            QtWidgets.QMessageBox.information(self, "Select labels", "Select one or more labels (rows) first.")
            return

        keep = set()
        for r in selected_rows:
            it = self.table_sem.item(r, 0)
            if it:
                keep.add(it.text().strip().lower())

        all_labels = set()
        try:
            if SEMANTIC_AVAILABLE:
                if self.seg_small is None or self.seg_lut is None:
                    self.seg_small, self.seg_lut = compute_seg_small(self.base_img)
                ids = np.unique(self.seg_small.reshape(-1))
                for cid in ids:
                    if cid < len(self.seg_lut):
                        all_labels.add(self.seg_lut[int(cid)].strip().lower())
        except Exception:
            pass

        if not all_labels:
            for r in range(self.table_sem.rowCount()):
                it = self.table_sem.item(r, 0)
                if it:
                    all_labels.add(it.text().strip().lower())

        self.hidden_labels = set(all_labels) - set(keep)

        self.table_sem.blockSignals(True)
        for r in range(self.table_sem.rowCount()):
            it = self.table_sem.item(r, 0)
            if not it:
                continue
            lbl = it.text().strip().lower()
            self.table_sem.setItem(r, 2, self._make_hide_item(lbl in self.hidden_labels))
        self.table_sem.blockSignals(False)

        self.lbl_sem_status.setText(f"Kept visible: {', '.join(sorted(keep))}")
        self.schedule_fast()

    def unhide_all(self):
        self.hidden_labels = set()
        self.table_sem.blockSignals(True)
        for r in range(self.table_sem.rowCount()):
            self.table_sem.setItem(r, 2, self._make_hide_item(False))
        self.table_sem.blockSignals(False)
        self.lbl_sem_status.setText("All labels unhidden.")
        self.schedule_fast()

    # ----------------------------
    # Add / Update mapping (with Hide)
    # ----------------------------
    def add_update_mapping(self):
        lbl = (self.edit_add_label.text() or "").strip().lower()
        phrase = (self.edit_add_phrase.text() or "").strip()
        hide = bool(self.chk_add_hide.isChecked())
        if not lbl:
            return

        if phrase == "":
            self.label_map.pop(lbl, None)
        else:
            self.label_map[lbl] = phrase

        if hide:
            self.hidden_labels.add(lbl)
        else:
            self.hidden_labels.discard(lbl)

        self.table_sem.blockSignals(True)
        row = self._find_label_row(lbl)
        if row is None:
            r = self.table_sem.rowCount()
            self.table_sem.insertRow(r)

            item_lbl = QtWidgets.QTableWidgetItem(lbl)
            item_lbl.setFlags(item_lbl.flags() & ~QtCore.Qt.ItemIsEditable)
            self.table_sem.setItem(r, 0, item_lbl)

            self.table_sem.setItem(r, 1, QtWidgets.QTableWidgetItem(self.label_map.get(lbl, "")))
            self.table_sem.setItem(r, 2, self._make_hide_item(lbl in self.hidden_labels))
        else:
            self.table_sem.item(row, 1).setText(self.label_map.get(lbl, ""))
            self.table_sem.setItem(row, 2, self._make_hide_item(lbl in self.hidden_labels))
        self.table_sem.blockSignals(False)

        self.edit_add_label.setText("")
        self.edit_add_phrase.setText("")
        self.chk_add_hide.setChecked(False)

        self.schedule_fast()

    def remove_selected_mappings(self):
        rows = sorted({idx.row() for idx in self.table_sem.selectionModel().selectedRows()}, reverse=True)
        if not rows:
            return
        self.table_sem.blockSignals(True)
        for r in rows:
            lbl_item = self.table_sem.item(r, 0)
            if lbl_item:
                lbl = lbl_item.text().strip().lower()
                self.label_map.pop(lbl, None)
                self.hidden_labels.discard(lbl)
            self.table_sem.removeRow(r)
        self.table_sem.blockSignals(False)
        self.schedule_fast()

    def on_table_changed(self, item: QtWidgets.QTableWidgetItem):
        row = item.row()
        lbl_item = self.table_sem.item(row, 0)
        if not lbl_item:
            return
        lbl = lbl_item.text().strip().lower()

        if item.column() == 1:
            phrase = (item.text() or "").strip()
            if phrase == "":
                self.label_map.pop(lbl, None)
            else:
                self.label_map[lbl] = phrase
            self.schedule_fast()
            return

        if item.column() == 2:
            hidden = (item.checkState() == QtCore.Qt.Checked)
            if hidden:
                self.hidden_labels.add(lbl)
            else:
                self.hidden_labels.discard(lbl)
            self.schedule_fast()
            return

    # ----------------------------
    # Import / Export JSON
    # ----------------------------
    def import_json(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Import mappings", "", "JSON (*.json)")
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.label_map = {}
            self.hidden_labels = set()

            if isinstance(data, dict):
                for k, v in data.items():
                    lbl = str(k).strip().lower()
                    if isinstance(v, dict):
                        phrase = str(v.get("phrase", "") or "").strip()
                        hide = bool(v.get("hide", False))
                    else:
                        phrase = str(v or "").strip()
                        hide = False

                    if phrase != "":
                        self.label_map[lbl] = phrase
                    if hide:
                        self.hidden_labels.add(lbl)

            self._refresh_table_from_maps()
            self.lbl_sem_status.setText(f"Imported {len(set(self.label_map) | self.hidden_labels)} entries.")
            self.schedule_fast()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Import error", str(e))

    def export_json(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export mappings", "mappings.json", "JSON (*.json)")
        if not path:
            return
        if not path.lower().endswith(".json"):
            path += ".json"
        try:
            keys = set(self.label_map.keys()) | set(self.hidden_labels)
            data = {}
            for lbl in sorted(keys):
                data[lbl] = {
                    "phrase": self.label_map.get(lbl, ""),
                    "hide": (lbl in self.hidden_labels)
                }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            self.lbl_sem_status.setText(f"Exported {len(data)} entries.")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Export error", str(e))

    # ----------------------------
    # File operations
    # ----------------------------
    def open_image(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Image", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.webp *.tiff);;All Files (*.*)"
        )
        if not path:
            return
        try:
            img = Image.open(path).convert("RGB")
            self.base_img = downscale_for_processing(img, PROCESS_MAX_SIDE)

            self.seg_small = None
            self.seg_lut = None
            self.label_map = {}
            self.hidden_labels = set()

            self.table_sem.blockSignals(True)
            self.table_sem.setRowCount(0)
            self.table_sem.blockSignals(False)
            self.lbl_sem_status.setText("—")

            self.image_view.set_image(pil_to_qimage(self.base_img))
            self.render_view.set_image(None)
            self.output_text.setPlainText("")

            self.set_controls_enabled(True)
            self.status.showMessage(f"Loaded: {os.path.basename(path)} | processing={self.base_img.size[0]}×{self.base_img.size[1]}")
            self.schedule_fast()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Open error", str(e))

    def copy_latest_text(self):
        if not self.last_result:
            return
        QtWidgets.QApplication.clipboard().setText("\n".join(self.last_result.lines))
        self.status.showMessage("Copied latest text.")

    def save_latest_txt(self):
        if not self.last_result:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save .txt", "output.txt", "Text (*.txt)")
        if not path:
            return
        if not path.lower().endswith(".txt"):
            path += ".txt"
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.last_result.lines))
        self.status.showMessage(f"Saved: {os.path.basename(path)}")

    # ----------------------------
    # Scheduling / render pipeline
    # ----------------------------
    def schedule_fast(self):
        if self.base_img is None:
            return
        self.fast_timer.start(FAST_DEBOUNCE_MS)

    def dispatch_fast(self):
        allow_kmeans = (self.cmb_color.currentText() != "Color (K-means)")
        self.dispatch_render(allow_kmeans=allow_kmeans)
        if self.cmb_color.currentText() == "Color (K-means)":
            self.kmeans_timer.start(KMEANS_DEBOUNCE_MS)

    def dispatch_kmeans(self):
        self.dispatch_render(allow_kmeans=True)

    def current_charset(self) -> str:
        if self.rb_custom.isChecked():
            cs = self.edit_charset.text()
            return cs if len(cs) >= 2 else PRESET_CHARSETS["Standard"]
        return PRESET_CHARSETS[self.cmb_charset.currentText()]

    def current_font_path(self) -> str:
        return self.font_map.get(self.cmb_font.currentText(), "")

    def selected_font_family(self) -> str:
        path = self.current_font_path()
        if not path:
            return "Consolas"
        if path in self._font_family_cache:
            return self._font_family_cache[path]
        fam = "Consolas"
        try:
            fid = QtGui.QFontDatabase.addApplicationFont(path)
            families = QtGui.QFontDatabase.applicationFontFamilies(fid)
            if families:
                fam = families[0]
        except Exception:
            pass
        self._font_family_cache[path] = fam
        return fam

    def dispatch_render(self, allow_kmeans: bool):
        if self.base_img is None:
            return

        mode = self.cmb_content.currentText()
        if mode == "Semantic Letters" and not SEMANTIC_AVAILABLE:
            self.cmb_content.setCurrentText("ASCII")
            mode = "ASCII"

        self.request_id += 1
        rid = self.request_id

        use_cap = bool(self.chk_preview_cap.isChecked()) and self.is_dragging

        params = RenderParams(
            content_mode=mode,
            fill_style=self.cmb_fill.currentText(),
            seed=int(self.spin_seed.value()),

            width_chars=int(self.spin_width.value()),
            use_preview_cap=use_cap,
            preview_cap=int(self.spin_preview_cap.value()),

            invert=bool(self.chk_invert.isChecked()),
            charset=self.current_charset(),

            brightness=int(self.sliders["Brightness"].value()),
            contrast=int(self.sliders["Contrast"].value()),
            ambiance=int(self.sliders["Ambiance"].value()),
            saturation=int(self.sliders["Saturation"].value()),
            highlights=int(self.sliders["Highlights"].value()),
            shadows=int(self.sliders["Shadows"].value()),

            font_path=self.current_font_path(),
            font_size=int(self.spin_font_size.value()),
            ttc_index=int(self.spin_ttc.value()),

            color_mode=self.cmb_color.currentText(),
            k=int(self.spin_k.value()),
            allow_kmeans=allow_kmeans,

            transparent_bg=bool(self.chk_transparent.isChecked()),

            seg_small=self.seg_small,
            seg_lut=self.seg_lut,

            label_map=dict(self.label_map),
            hidden_labels=sorted(self.hidden_labels),

            edge_threshold=int(self.spin_edge_thresh.value()),
            edge_pixel_size=int(self.spin_edge_pix.value()),
            edge_thickness=int(self.spin_edge_thick.value()),
            edge_invert=bool(self.chk_edge_invert.isChecked()),
            edge_char=str(self.edit_edge_char.text() or "█"),
            edge_bg=str(self.cmb_edge_bg.currentText()),
            edge_fg=str(self.cmb_edge_fg.currentText()),
        )

        worker = RenderWorker(rid, self.base_img, params)
        worker.signals.result.connect(self.on_worker_result)
        worker.signals.error.connect(self.on_worker_error)
        self._active_workers[rid] = worker

        self.thread_pool.start(worker)
        self.status.showMessage("Rendering…")

    @QtCore.Slot(int, object)
    def on_worker_result(self, rid: int, result: RenderResult):
        self._active_workers.pop(rid, None)
        if rid < self.last_applied_id:
            return
        self.last_applied_id = rid
        self.last_result = result

        if result.seg_small is not None and result.seg_lut is not None:
            self.seg_small = result.seg_small
            self.seg_lut = result.seg_lut

        self.image_view.set_image(result.preview_qimage)
        self.render_view.set_image(result.render_qimage)

        if self.chk_live_text.isChecked():
            self.output_text.blockSignals(True)
            self.output_text.setPlainText("\n".join(result.lines))
            self.output_text.blockSignals(False)

        fontfile = os.path.basename(result.font_path) if result.font_path else "default"
        self.status.showMessage(f"{result.content_mode} | font={fontfile} size={result.font_size} idx={result.ttc_index}")

    @QtCore.Slot(int, str)
    def on_worker_error(self, rid: int, msg: str):
        self._active_workers.pop(rid, None)
        print(msg)
        self.status.showMessage("Render error (see console).")
        if "Semantic" in msg or "torch" in msg or "transformers" in msg:
            QtWidgets.QMessageBox.information(self, "Semantic error", msg)

    # ----------------------------
    # Export
    # ----------------------------
    def _build_export_render_full(self, force_kmeans: bool = True) -> Tuple[Image.Image, List[str], Optional[np.ndarray], Tuple[int, int, int]]:
        """
        Returns:
          pil_img (RGBA),
          lines,
          color_map (or None),
          bg_rgb tuple used for SVG background if needed.
        """
        if self.base_img is None:
            raise RuntimeError("No image loaded.")

        mode = self.cmb_content.currentText()
        if mode == "Semantic Letters" and not SEMANTIC_AVAILABLE:
            mode = "ASCII"

        font = safe_load_pil_font(self.current_font_path(), int(self.spin_font_size.value()), int(self.spin_ttc.value()))
        char_w, char_h = measure_char_cell(font)

        adjusted = apply_adjustments(
            self.base_img,
            int(self.sliders["Brightness"].value()),
            int(self.sliders["Contrast"].value()),
            int(self.sliders["Ambiance"].value()),
            int(self.sliders["Saturation"].value()),
            int(self.sliders["Highlights"].value()),
            int(self.sliders["Shadows"].value())
        )

        out_w = int(self.spin_width.value())
        iw, ih = adjusted.size
        out_h = max(1, int(round((ih / iw) * out_w * (char_w / char_h))))
        resized_rgb = adjusted.resize((out_w, out_h), Image.Resampling.BILINEAR)

        cmode = self.cmb_color.currentText()
        bg_rgb = (15, 15, 15)
        fg_rgb = (230, 230, 230)

        color_map = None
        if cmode == "Color (Direct)":
            color_map = build_color_map(resized_rgb, "Color (Direct)", int(self.spin_k.value()))
        elif cmode == "Color (K-means)":
            if force_kmeans:
                color_map = build_color_map(resized_rgb, "Color (K-means)", int(self.spin_k.value()))
            else:
                color_map = build_color_map(resized_rgb, "Color (Direct)", int(self.spin_k.value()))
        if cmode == "Mono":
            color_map = None

        if mode == "ASCII":
            lines, rgb_r, _ = make_ascii_lines(
                adjusted,
                out_w,
                self.current_charset(),
                bool(self.chk_invert.isChecked()),
                char_w, char_h
            )
            if cmode == "Color (Direct)":
                color_map = build_color_map(rgb_r, "Color (Direct)", int(self.spin_k.value()))
            elif cmode == "Color (K-means)":
                color_map = build_color_map(rgb_r, "Color (K-means)", int(self.spin_k.value()))
            elif cmode == "Mono":
                color_map = None

        elif mode == "Pixel Edges":
            mask = make_pixel_edge_mask(
                adjusted_img=adjusted,
                out_w=out_w,
                out_h=out_h,
                pixel_size=int(self.spin_edge_pix.value()),
                threshold=int(self.spin_edge_thresh.value()),
                thickness=int(self.spin_edge_thick.value()),
                invert=bool(self.chk_edge_invert.isChecked()),
            )
            lines = edge_mask_to_lines(mask, str(self.edit_edge_char.text() or "█"))

            fg_rgb = (0, 0, 0) if self.cmb_edge_fg.currentText() == "Black" else (235, 235, 235)
            bg_rgb = (255, 255, 255) if self.cmb_edge_bg.currentText() == "White" else (15, 15, 15)
            color_map = None

        elif mode == "Edge Overlay":
            base_lines, rgb_r, _ = make_ascii_lines(
                adjusted,
                out_w,
                self.current_charset(),
                bool(self.chk_invert.isChecked()),
                char_w, char_h
            )

            if cmode == "Mono":
                base_cm = np.zeros((len(base_lines), max(len(l) for l in base_lines), 3), dtype=np.uint8)
                base_cm[:, :, 0] = fg_rgb[0]
                base_cm[:, :, 1] = fg_rgb[1]
                base_cm[:, :, 2] = fg_rgb[2]
            else:
                if cmode == "Color (Direct)":
                    base_cm = build_color_map(rgb_r, "Color (Direct)", int(self.spin_k.value()))
                else:
                    base_cm = build_color_map(rgb_r, "Color (K-means)", int(self.spin_k.value()))

            mask = make_pixel_edge_mask(
                adjusted_img=adjusted,
                out_w=out_w,
                out_h=len(base_lines),
                pixel_size=int(self.spin_edge_pix.value()),
                threshold=int(self.spin_edge_thresh.value()),
                thickness=int(self.spin_edge_thick.value()),
                invert=bool(self.chk_edge_invert.isChecked()),
            )
            lines = overlay_edges_on_lines(base_lines, mask, str(self.edit_edge_char.text() or "█"))

            edge_rgb = (0, 0, 0) if self.cmb_edge_fg.currentText() == "Black" else (235, 235, 235)
            m = np.asarray(mask.resize((base_cm.shape[1], base_cm.shape[0]), Image.Resampling.NEAREST), dtype=np.uint8)
            color_map = base_cm.copy()
            color_map[m > 0] = np.array(edge_rgb, dtype=np.uint8)

        else:
            # Semantic Letters
            if self.seg_small is None or self.seg_lut is None:
                self.seg_small, self.seg_lut = compute_seg_small(self.base_img)

            ids_grid = seg_to_ids_grid(self.seg_small, out_w, out_h)
            lut = list(self.seg_lut)
            max_id = int(ids_grid.max())
            if max_id >= len(lut):
                lut = lut + ["object"] * (max_id + 1 - len(lut))

            hidden_set = set(self.hidden_labels)

            id_to_pool: Dict[int, List[str]] = {}
            for cid in np.unique(ids_grid):
                lbl = lut[int(cid)].strip().lower()
                if lbl in hidden_set:
                    id_to_pool[int(cid)] = [" "]
                else:
                    phrase = (self.label_map.get(lbl, "") or "").strip() or lbl
                    id_to_pool[int(cid)] = phrase_to_char_pool(phrase)

            seed = int(self.spin_seed.value()) & 0xFFFFFFFFFFFFFFFF
            cycle = (self.cmb_fill.currentText() == "Cycle")
            lines = []
            for y in range(out_h):
                row = []
                row_ids = ids_grid[y]
                for x in range(out_w):
                    cid = int(row_ids[x])
                    pool = id_to_pool.get(cid) or ["o"]
                    if cycle:
                        idx = (x + y * 131 + cid * 17) % len(pool)
                        ch = pool[idx]
                    else:
                        ch = pick_char_jumble(pool, x, y, cid, seed)
                    row.append(ch[0] if ch else " ")
                lines.append("".join(row))

        img = render_lines_to_rgba(
            lines=lines,
            font=font,
            color_map=color_map,
            transparent_bg=bool(self.chk_transparent.isChecked()),
            bg_rgb=bg_rgb,
            fg_rgb=fg_rgb
        )
        return img, lines, color_map, bg_rgb

    def export_png(self):
        if self.base_img is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export PNG", "render.png", "PNG (*.png)")
        if not path:
            return
        if not path.lower().endswith(".png"):
            path += ".png"

        try:
            pil_img, _, _, _ = self._build_export_render_full(force_kmeans=True)
            if max(pil_img.size) > 16000:
                QtWidgets.QMessageBox.information(
                    self, "Large export",
                    f"This PNG is very large ({pil_img.size[0]}×{pil_img.size[1]}). "
                    f"Reduce Width(chars) or Font Size if needed."
                )
            pil_img.save(path, format="PNG")
            self.status.showMessage(f"Exported: {os.path.basename(path)}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Export error", str(e))

    def export_svg(self):
        if self.base_img is None:
            return

        msg = QtWidgets.QMessageBox(self)
        msg.setWindowTitle("Export SVG")
        msg.setText("Choose SVG export mode:")
        btn_text = msg.addButton("SVG (Text, editable)", QtWidgets.QMessageBox.AcceptRole)
        btn_embed = msg.addButton("SVG (Embedded PNG, reliable)", QtWidgets.QMessageBox.AcceptRole)
        msg.addButton("Cancel", QtWidgets.QMessageBox.RejectRole)
        msg.exec()

        clicked = msg.clickedButton()
        if clicked is None or clicked.text() == "Cancel":
            return
        as_text = (clicked == btn_text)

        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export SVG", "render.svg", "SVG (*.svg)")
        if not path:
            return
        if not path.lower().endswith(".svg"):
            path += ".svg"

        transparent_bg = bool(self.chk_transparent.isChecked())

        try:
            pil_img, lines, color_map, bg_rgb = self._build_export_render_full(force_kmeans=True)

            if not as_text:
                svg = svg_embed_png(pil_img, transparent_bg=transparent_bg, bg_rgb=bg_rgb)
                with open(path, "w", encoding="utf-8") as f:
                    f.write(svg)
                self.status.showMessage(f"Exported: {os.path.basename(path)}")
                return

            font_family = self.selected_font_family()
            font_size = int(self.spin_font_size.value())
            qfont = QtGui.QFont(font_family, font_size)
            metrics = QtGui.QFontMetrics(qfont)
            char_w = max(1, metrics.horizontalAdvance("M"))
            line_h = max(1, metrics.height())

            svg = svg_text_export(
                lines=lines,
                color_map=color_map,
                font_family=font_family,
                font_size_px=font_size,
                char_w=char_w,
                line_h=line_h,
                transparent_bg=transparent_bg,
                bg_rgb=bg_rgb
            )
            with open(path, "w", encoding="utf-8") as f:
                f.write(svg)
            self.status.showMessage(f"Exported: {os.path.basename(path)}")

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Export error", str(e))


def main():
    app = QtWidgets.QApplication(sys.argv)
    apply_dark_palette(app)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()