"""Export and visualization helpers for heights and meshes."""
from typing import Tuple
import numpy as np


def export_obj(filename: str, z: np.ndarray, xy_scale: float = 1.0, z_scale: float = 1.0, center: bool = True) -> None:
    """Export a height field z (H x W) to an OBJ mesh.

    Behavior is unchanged from the prototype.
    """
    H, W = z.shape
    z = z.astype(np.float64) * z_scale

    xs = np.arange(W) * xy_scale
    ys = np.arange(H) * xy_scale
    xx, yy = np.meshgrid(xs, ys)

    if center:
        xx -= xx.mean()
        yy -= yy.mean()
        zc = z - z.mean()
    else:
        zc = z

    vertices = np.stack([xx, yy, zc], axis=-1).reshape(-1, 3)

    dzdy, dzdx = np.gradient(zc, xy_scale, xy_scale)
    normals = np.stack([-dzdx, -dzdy, np.ones_like(zc)], axis=-1)
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-8
    normals = normals.reshape(-1, 3)

    faces = []
    for y in range(H - 1):
        for x in range(W - 1):
            v0 = y * W + x
            v1 = y * W + (x + 1)
            v2 = (y + 1) * W + x
            v3 = (y + 1) * W + (x + 1)

            faces.append((v0, v1, v2))
            faces.append((v2, v1, v3))

    with open(filename, "w") as f:
        f.write("# Height field exported from Python\n")
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for n in normals:
            f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
        for (a, b, c) in faces:
            a += 1
            b += 1
            c += 1
            f.write(f"f {a}//{a} {b}//{b} {c}//{c}\n")


def height_to_vis(z: np.ndarray) -> np.ndarray:
    z = z - z.min()
    z = z / (z.max() + 1e-8)
    return (z * 255).astype(np.uint8)
