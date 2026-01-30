import os
import numpy as np
import torch
from core.siren import fit_siren_height


def test_resume_from_checkpoint(tmp_path):
    H, W = 16, 16
    yy, xx = np.mgrid[0:H, 0:W]
    z = 1e-3 * ((xx - W/2) ** 2 + (yy - H/2) ** 2)
    valid = np.ones_like(z, dtype=bool)

    checkpoint = str(tmp_path / "resume_checkpoint.pth")

    # initial short training with checkpoint saving enabled
    _ = fit_siren_height(z, valid, iters=20, device="cpu", val_split=0.2, checkpoint_path=checkpoint, checkpoint_interval=10, early_stopping_patience=100)

    assert os.path.exists(checkpoint), "Initial checkpoint not created"
    ck1 = torch.load(checkpoint, map_location="cpu")
    iter1 = ck1.get("iter", 0)

    # resume training from the checkpoint and run more iterations (total iters higher)
    _ = fit_siren_height(z, valid, iters=40, device="cpu", val_split=0.2, resume_from=checkpoint, checkpoint_path=checkpoint, checkpoint_interval=0, early_stopping_patience=2)

    ck2 = torch.load(checkpoint, map_location="cpu")
    iter2 = ck2.get("iter", 0)

    assert iter2 >= iter1, f"Checkpoint iter did not increase after resume: {iter1} -> {iter2}"

    # cleanup
    os.remove(checkpoint)
