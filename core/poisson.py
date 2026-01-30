"""Poisson solver with Neumann boundary conditions.

Algorithm preserved exactly from original script (ghost-point elimination).
"""
from typing import Tuple
import numpy as np
import scipy.sparse
import scipy.sparse.linalg


def solve_poisson_neumann(f: np.ndarray, p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Solve discrete Poisson equation with Neumann BCs and mean-zero fix.

    Args:
        f: (H,W) source divergence
        p,q: (H,W) boundary gradient fields
    Returns:
        z: (H,W) mean-zero height field
    """
    H, W = f.shape
    N = H * W
    idx = lambda x, y: y * W + x

    A = scipy.sparse.lil_matrix((N, N), dtype=np.float64)
    b = np.zeros(N, dtype=np.float64)

    for y in range(H):
        for x in range(W):
            i = idx(x, y)
            neighbors = 0

            # left neighbor
            if x > 0:
                A[i, idx(x - 1, y)] = 1.0
                neighbors += 1
            else:
                if x < W - 1:
                    A[i, idx(x + 1, y)] = A[i, idx(x + 1, y)] + 1.0
                    neighbors += 1
                b[i] -= 2.0 * p[y, x]

            # right neighbor
            if x < W - 1:
                A[i, idx(x + 1, y)] = A[i, idx(x + 1, y)] + 1.0
                neighbors += 1
            else:
                if x > 0:
                    A[i, idx(x - 1, y)] = A[i, idx(x - 1, y)] + 1.0
                    neighbors += 1
                b[i] += 2.0 * p[y, x]

            # top neighbor
            if y > 0:
                A[i, idx(x, y - 1)] = 1.0
                neighbors += 1
            else:
                if y < H - 1:
                    A[i, idx(x, y + 1)] = A[i, idx(x, y + 1)] + 1.0
                    neighbors += 1
                b[i] -= 2.0 * q[y, x]

            # bottom neighbor
            if y < H - 1:
                A[i, idx(x, y + 1)] = A[i, idx(x, y + 1)] + 1.0
                neighbors += 1
            else:
                if y > 0:
                    A[i, idx(x, y - 1)] = A[i, idx(x, y - 1)] + 1.0
                    neighbors += 1
                b[i] += 2.0 * q[y, x]

            A[i, i] = -float(neighbors)
            b[i] -= float(f[y, x])

    # fix null-space
    A[0, :] = 0.0
    A[0, 0] = 1.0
    b[0] = 0.0

    z = scipy.sparse.linalg.spsolve(A.tocsr(), b)
    z = z.reshape(H, W).astype(np.float64)
    z -= z.mean()
    return z
