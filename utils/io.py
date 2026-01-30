"""I/O utilities: image loading and normalization."""
from typing import Tuple
import numpy as np
from skimage import io


def load_normal_map(path: str) -> np.ndarray:
    """Load an RGBA/RGB normal map from disk and convert to [-1, 1].

    Returns a normalized float32 array of shape (H,W,3) in [-1,1].
    """
    n = io.imread(path).astype(np.float32)
    if n.ndim == 3 and n.shape[2] == 4:
        n = n[..., :3]
    n = n / 255.0 * 2.0 - 1.0
    # ensure per-pixel normalization (match legacy behavior)
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    n = n / (norm + 1e-8)
    return n
