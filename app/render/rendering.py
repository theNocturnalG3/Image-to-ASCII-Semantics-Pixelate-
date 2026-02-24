from typing import List, Optional
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from PySide6 import QtGui
from app.utils.fonts import measure_char_cell

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

def pil_to_qimage(img: Image.Image) -> QtGui.QImage:
    rgba = img.convert("RGBA")
    data = rgba.tobytes("raw", "RGBA")
    qimg = QtGui.QImage(data, rgba.size[0], rgba.size[1], QtGui.QImage.Format_RGBA8888)
    return qimg.copy()