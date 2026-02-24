import numpy as np
from PIL import Image

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