import os
import numpy as np
import torch
from core.siren import fit_siren_height


def test_checkpointing_creates_file(tmp_path):
    H, W = 16, 16
    yy, xx = np.mgrid[0:H, 0:W]
    z = 1e-3 * ((xx - W/2) ** 2 + (yy - H/2) ** 2)
    valid = np.ones_like(z, dtype=bool)

    checkpoint = str(tmp_path / "best_checkpoint.pth")

    model = fit_siren_height(z, valid, iters=50, device="cpu", val_split=0.2, checkpoint_path=checkpoint, checkpoint_interval=10, early_stopping_patience=2)

    assert os.path.exists(checkpoint), "Checkpoint file was not created"
    ck = torch.load(checkpoint, map_location="cpu")
    assert isinstance(ck, dict) and "optim" in ck and "model" in ck, "Checkpoint does not contain optimizer+model state"
    # cleanup
    os.remove(checkpoint)
