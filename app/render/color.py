import numpy as np
from PIL import Image

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