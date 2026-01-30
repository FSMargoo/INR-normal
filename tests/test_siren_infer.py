import torch
from core.siren import SirenNet, siren_normals_and_height


def test_siren_inference_shapes():
    H, W = 8, 8
    device = "cpu"
    model = SirenNet(hidden=32, layers=1).to(device)

    n, z = siren_normals_and_height(model, H, W, device=device)
    assert n.shape == (H, W, 3)
    assert z.shape == (H, W)
    assert n.dtype == "uint8"
