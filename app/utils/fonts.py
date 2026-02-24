# app/utils/fonts.py
import os
from typing import Dict, List, Tuple
from PIL import Image, ImageDraw, ImageFont

_FONT_CACHE: Dict[Tuple[str, int, int], ImageFont.ImageFont] = {}

def list_windows_font_files() -> List[Tuple[str, str]]:
    if os.name != "nt":
        return []
    fonts_dir = os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts")
    items: List[Tuple[str, str]] = []
    try:
        for fn in os.listdir(fonts_dir):
            if fn.lower().endswith((".ttf", ".otf", ".ttc")):
                items.append((fn, os.path.join(fonts_dir, fn)))
    except Exception:
        pass
    items.sort(key=lambda x: x[0].lower())
    return items

def safe_load_pil_font(path: str, size: int, ttc_index: int = 0):
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
    """
    More stable than textbbox('M') alone:
    - width from getlength('M') when available
    - height from ascent+descent when available
    """
    # width
    try:
        if hasattr(font, "getlength"):
            char_w = int(round(font.getlength("M")))
        else:
            dummy = Image.new("RGB", (80, 80))
            d = ImageDraw.Draw(dummy)
            bbox = d.textbbox((0, 0), "M", font=font)
            char_w = max(1, bbox[2] - bbox[0])
    except Exception:
        char_w = 10

    # height
    try:
        if hasattr(font, "getmetrics"):
            ascent, descent = font.getmetrics()
            char_h = max(1, int(ascent + descent))
        else:
            dummy = Image.new("RGB", (80, 80))
            d = ImageDraw.Draw(dummy)
            bbox = d.textbbox((0, 0), "Hg", font=font)  # includes descenders better
            char_h = max(1, bbox[3] - bbox[1])
    except Exception:
        char_h = 16

    return max(1, char_w), max(1, char_h)

def is_monospace_font_file(path: str, size: int = 16, ttc_index: int = 0) -> bool:
    """
    Heuristic check: monospaced fonts have equal advances for narrow/wide glyphs.
    Works well for filtering Windows Fonts folder.
    """
    try:
        f = ImageFont.truetype(path, size=int(size), index=int(ttc_index))
        if hasattr(f, "getlength"):
            wi = f.getlength("i")
            wW = f.getlength("W")
            wM = f.getlength("M")
            ws = f.getlength(" ")
            # allow small tolerance (hinting / rounding)
            return (abs(wi - wW) < 0.5) and (abs(wi - wM) < 0.5) and (abs(wi - ws) < 0.5)
        else:
            # fallback bbox widths
            dummy = Image.new("RGB", (200, 80))
            d = ImageDraw.Draw(dummy)
            def w(ch):
                b = d.textbbox((0, 0), ch, font=f)
                return b[2] - b[0]
            wi, wW, wM, ws = w("i"), w("W"), w("M"), w(" ")
            return (abs(wi - wW) <= 1) and (abs(wi - wM) <= 1) and (abs(wi - ws) <= 1)
    except Exception:
        return False