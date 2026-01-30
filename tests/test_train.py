import numpy as np
from core.siren import fit_siren_height


def test_fit_siren_small():
    # create a tiny synthetic height and mask
    H, W = 16, 16
    yy, xx = np.mgrid[0:H, 0:W]
    z = 1e-3 * ((xx - W/2) ** 2 + (yy - H/2) ** 2)
    valid = np.ones_like(z, dtype=bool)

    # train for very few iterations to smoke-test the pipeline
    model = fit_siren_height(z, valid, iters=10, device="cpu")
    # ensure the model returns a forward pass of expected shape
    import torch
    H, W = z.shape
    yy = torch.linspace(-1, 1, H)
    xx = torch.linspace(-1, 1, W)
    yy, xx = torch.meshgrid(yy, xx, indexing="ij")
    coords = torch.stack([xx, yy], dim=-1).reshape(-1, 2)
    z_pred = model(coords).detach().cpu().numpy()
    assert z_pred.shape[0] == H * W
