"""Geometry primitives: normals, gradients, divergence.

Preserves original math; extracted from legacy script.
"""
from typing import Tuple
import numpy as np


def normalize(v: np.ndarray) -> np.ndarray:
    """Normalize along the last axis, stable to small norms.

    Args:
        v: (..., C) array
    Returns:
        normalized array of same shape
    """
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8)


def compute_gradients(normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute p = dz/dx and q = dz/dy from a normal map.

    Inputs/Outputs match legacy behavior:
      normal: (H, W, 3) in [-1,1]
      returns: p,q each (H,W)
    """
    nx, ny, nz = normal[..., 0], normal[..., 1], normal[..., 2]
    p = -nx / (nz + 1e-8)
    q = -ny / (nz + 1e-8)
    return p, q


def compute_divergence(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Finite-difference divergence used as Poisson source f = d p / d x + d q / d y.

    This function is a direct extraction from the original script; boundary
    handling is identical.
    """
    dp = np.zeros_like(p)
    dq = np.zeros_like(q)
    dp[:, 1:-1] = 0.5 * (p[:, 2:] - p[:, : -2])
    dp[:, 0] = p[:, 1] - p[:, 0]  # forward
    dp[:, -1] = p[:, -1] - p[:, -2]  # backward
    dq[1:-1, :] = 0.5 * (q[2:, :] - q[:-2, :])
    dq[0, :] = q[1, :] - q[0, :]
    dq[-1, :] = q[-1, :] - q[-2, :]
    return dp + dq


def normals_from_height(z: np.ndarray) -> np.ndarray:
    """Compute normal map from height field z.

    Returns normal map as uint8 in [0,255], preserving original scaling.
    """
    dzdx = np.gradient(z, axis=1)
    dzdy = np.gradient(z, axis=0)
    n = np.stack([dzdx, dzdy, np.ones_like(z)], axis=-1)
    n = normalize(n)
    return ((n + 1) * 0.5 * 255).astype(np.uint8)
