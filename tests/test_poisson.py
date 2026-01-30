import numpy as np
from core.poisson import solve_poisson_neumann
from core.geometry import compute_gradients


def test_poisson_reconstruction_small():
    # create simple synthetic height (quadratic bowl)
    H, W = 16, 16
    yy, xx = np.mgrid[0:H, 0:W]
    z = 0.001 * (xx - W/2)**2 + 0.001 * (yy - H/2)**2

    # compute gradients and divergence
    p, q = compute_gradients(((np.stack([ - (z - z.mean()), - (z - z.mean()), np.ones_like(z)], axis=-1))))
    # For test purpose, compute p,q numerically
    p = np.gradient(z, axis=1)
    q = np.gradient(z, axis=0)
    f = np.gradient(p, axis=1) + np.gradient(q, axis=0)

    z_rec = solve_poisson_neumann(f, p, q)
    assert z_rec.shape == z.shape
    # mean-zero enforced; allow a modest numeric tolerance comparing to the mean-centered ground truth
    mse = np.mean((z_rec - (z - z.mean()))**2)
    assert mse < 1e-2, f"MSE too large: {mse}"
