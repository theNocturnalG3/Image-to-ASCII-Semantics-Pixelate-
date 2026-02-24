import numpy as np
from typing import List
from PIL import Image, ImageFilter

def make_pixel_edge_mask(adjusted_img: Image.Image,
                         out_w: int,
                         out_h: int,
                         pixel_size: int = 6,
                         threshold: int = 70,
                         thickness: int = 2,
                         invert: bool = False) -> Image.Image:
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

    if pixel_size > 1:
        sw = max(1, out_w // pixel_size)
        sh = max(1, out_h // pixel_size)
        small = mask_img.resize((sw, sh), Image.Resampling.BOX)
        small_arr = np.asarray(small, dtype=np.uint8)
        small_bin = (small_arr >= 32).astype(np.uint8) * 255
        small = Image.fromarray(small_bin).convert("L")
        mask_img = small.resize((out_w, out_h), Image.Resampling.NEAREST)

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