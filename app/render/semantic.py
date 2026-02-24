from __future__ import annotations

from typing import List, Tuple, Optional, TYPE_CHECKING
import numpy as np
from PIL import Image

import os
from app.utils.resources import resource_path

LOCAL_MODEL_DIR = resource_path(os.path.join("models", "segformer-b0-ade-512-512"))

SEMANTIC_AVAILABLE = True
try:
    import torch
    from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
except Exception:
    SEMANTIC_AVAILABLE = False
    torch = None
    SegformerImageProcessor = None
    SegformerForSemanticSegmentation = None

SEG_MODEL_NAME = "nvidia/segformer-b0-finetuned-ade-512-512"
SEG_CACHE_SIZE = 256

if TYPE_CHECKING:
    from transformers import SegformerImageProcessor as _SegformerImageProcessor
    from transformers import SegformerForSemanticSegmentation as _SegformerForSemanticSegmentation

_SEG_PROC: Optional["_SegformerImageProcessor"] = None
_SEG_MODEL: Optional["_SegformerForSemanticSegmentation"] = None
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
        # Prefer local bundled model (offline)
        model_src = LOCAL_MODEL_DIR if os.path.isdir(LOCAL_MODEL_DIR) else SEG_MODEL_NAME

        _SEG_PROC = SegformerImageProcessor.from_pretrained(model_src, local_files_only=True)
        _SEG_MODEL = SegformerForSemanticSegmentation.from_pretrained(model_src, local_files_only=True)
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