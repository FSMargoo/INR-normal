"""Visualization helpers and small utilities."""
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt

from core.export import height_to_vis


def angle_error(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    gt = (gt / 255.0) * 2 - 1
    pred = (pred / 255.0) * 2 - 1
    gt = gt / (np.linalg.norm(gt, axis=-1, keepdims=True) + 1e-8)
    pred = pred / (np.linalg.norm(pred, axis=-1, keepdims=True) + 1e-8)
    dot = np.clip(np.sum(gt * pred, axis=-1), -1, 1)
    return np.arccos(dot)


def plot_all(gt, z_p, n_p, z_s, n_s, diff, name: str) -> None:
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["CMU Serif"],
    })

    axs[0, 0].imshow(gt);
    axs[0, 0].set_title("GT Normal");
    axs[0, 0].axis("off")

    axs[0, 1].imshow(height_to_vis(z_p), cmap="gray");
    axs[0, 1].set_title("Poisson Height");
    axs[0, 1].axis("off")

    axs[0, 2].imshow(n_p);
    axs[0, 2].set_title("Poisson Normal");
    axs[0, 2].axis("off")

    axs[1, 0].imshow(height_to_vis(z_s), cmap="gray");
    axs[1, 0].set_title("SIREN Height");
    axs[1, 0].axis("off")

    axs[1, 1].imshow(n_s);
    axs[1, 1].set_title("SIREN Normal");
    axs[1, 1].axis("off")

    im = axs[1, 2].imshow(diff, cmap="inferno");
    axs[1, 2].set_title("Angle Error (rad)");
    axs[1, 2].axis("off")
    fig.colorbar(im, ax=axs[1, 2])

    plt.tight_layout()
    plt.savefig(name + ".png", dpi=300)
    plt.close()
