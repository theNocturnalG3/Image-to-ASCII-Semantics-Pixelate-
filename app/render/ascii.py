import numpy as np
from typing import List, Tuple
from PIL import Image

PRESET_CHARSETS = {
    "Standard": "@%#*+=-:. ",
    "Dense":   "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~<>i!lI;:,\"^`'. ",
    "Blocks":  "█▓▒░ "
}

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