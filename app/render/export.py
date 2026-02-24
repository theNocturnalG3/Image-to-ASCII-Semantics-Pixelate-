import io
import base64
import numpy as np
from typing import List, Optional, Tuple
from PIL import Image
from xml.sax.saxutils import escape as xml_escape

def rgb_hex(rgb: Tuple[int, int, int]) -> str:
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
                    bg_rgb=(15, 15, 15),
                    default_fg_rgb=(230, 230, 230)) -> str:
    if not lines:
        raise ValueError("No text")
    pad = 2
    h_chars = len(lines)
    w_chars = max(len(l) for l in lines)
    width_px = w_chars * char_w + 2 * pad
    height_px = h_chars * line_h + 2 * pad

    bg_rect = "" if transparent_bg else f'<rect width="100%" height="100%" fill="{rgb_hex(bg_rgb)}"/>'
    fg_hex = rgb_hex(default_fg_rgb)

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
                col = color_map[y, x] if (y < color_map.shape[0] and x < color_map.shape[1]) else np.array(default_fg_rgb, dtype=np.uint8)
                run_color = rgb_hex((int(col[0]), int(col[1]), int(col[2])))
                start = x
                x += 1
                while x < len(line):
                    col2 = color_map[y, x] if (y < color_map.shape[0] and x < color_map.shape[1]) else np.array(default_fg_rgb, dtype=np.uint8)
                    if not np.array_equal(col2, col):
                        break
                    x += 1
                out.append(f'<tspan fill="{run_color}">{xml_escape(line[start:x])}</tspan>')
            out.append('</text>')
    out.append('</svg>')
    return "\n".join(out)