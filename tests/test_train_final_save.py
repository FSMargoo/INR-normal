import os
from scripts.train_siren import train_on_file


def test_final_save_uses_checkpoint(tmp_path):
    # create small fake normal map in memory by writing a valid PNG using numpy
    import numpy as np
    from skimage.io import imsave

    H, W = 16, 16
    # normal map in uint8 [0,255], store as neutral normal pointing Z+ (0.5->255/2)
    n = np.zeros((H, W, 3), dtype=np.uint8)
    n[..., 0] = 127
    n[..., 1] = 127
    n[..., 2] = 255

    inp = tmp_path / "n_test.png"
    out = str(tmp_path / "out.pth")
    ckpt = str(tmp_path / "best.pth")

    imsave(str(inp), n)

    # run training to create a checkpoint
    train_on_file(str(inp), iters=10, device="cpu", out=out, val_split=0.2, checkpoint=ckpt, checkpoint_interval=5, early_stopping_patience=100)

    assert os.path.exists(ckpt), "Checkpoint should have been created"

    # run again and make sure final save loads the checkpoint and writes the requested out path
    train_on_file(str(inp), iters=2, device="cpu", out=out, val_split=0.2, checkpoint=ckpt, checkpoint_interval=0, early_stopping_patience=0, resume=ckpt)
    assert os.path.exists(out), "Final output file was not written"