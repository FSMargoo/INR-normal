"""SIREN model and inference helpers.

This module keeps all SIREN-related code unchanged mathematically but organized
with clear docstrings and types.
"""
from typing import Tuple
import os
import numpy as np
import torch


class SirenLayer(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int, w0: float = 30.0, first: bool = False):
        super().__init__()
        self.linear = torch.nn.Linear(in_dim, out_dim)
        self.w0 = w0

        with torch.no_grad():
            if first:
                self.linear.weight.uniform_(-1 / in_dim, 1 / in_dim)
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6 / in_dim) / w0, np.sqrt(6 / in_dim) / w0
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # -> sin(w0 * (Wx + b))
        return torch.sin(self.w0 * self.linear(x))


class SirenNet(torch.nn.Module):
    def __init__(self, hidden: int = 128, layers: int = 3):
        super().__init__()
        net = [SirenLayer(2, hidden, first=True)]
        for _ in range(layers):
            net.append(SirenLayer(hidden, hidden))
        net.append(torch.nn.Linear(hidden, 1))
        self.net = torch.nn.Sequential(*net)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def fit_siren_height(
    z_gt: np.ndarray,
    valid: np.ndarray,
    iters: int = 6000,
    device: str = None,
    val_split: float = 0.0,
    checkpoint_path: str = None,
    checkpoint_interval: int = 0,
    early_stopping_patience: int = 0,
    resume_from: str = None,
) -> SirenNet:
    """Fit a SIREN to a ground-truth height map using masked MSE.

    Extended with optional validation split, checkpointing, early stopping and
    resume-from-checkpoint support.

    Args:
        z_gt: (H,W) target heights
        valid: (H,W) boolean mask
        iters: number of Adam iterations
        device: optional device (defaults to CUDA if available)
        val_split: fraction of valid pixels to hold out for validation (0 disables)
        checkpoint_path: path to write best model (if provided)
        checkpoint_interval: save an intermediate checkpoint every N iterations (0 disables)
        early_stopping_patience: number of evaluations (on validation) with no
            improvement before stopping early (0 disables)
        resume_from: path to a checkpoint file (state_dict) to initialize model from
    Returns:
        trained SirenNet
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    H, W = z_gt.shape

    yy, xx = np.mgrid[0:H, 0:W]
    coords = np.stack([xx, yy], axis=-1).astype(np.float32)
    coords[..., 0] = coords[..., 0] / (W - 1) * 2 - 1
    coords[..., 1] = coords[..., 1] / (H - 1) * 2 - 1

    coords = torch.from_numpy(coords).to(device).reshape(-1, 2)
    z_gt = torch.from_numpy(z_gt).to(device).reshape(-1, 1)
    valid_mask = torch.from_numpy(valid).to(device).reshape(-1).bool()

    # prepare optional validation split
    if val_split and val_split > 0.0:
        valid_indices = torch.nonzero(valid_mask).view(-1)
        n_val = max(1, int(len(valid_indices) * val_split))
        perm = torch.randperm(len(valid_indices), device=device)
        val_idx = valid_indices[perm[:n_val]]
        train_idx = valid_indices[perm[n_val:]]
        train_mask = torch.zeros_like(valid_mask)
        train_mask[train_idx] = True
        val_mask = torch.zeros_like(valid_mask)
        val_mask[val_idx] = True
    else:
        train_mask = valid_mask
        val_mask = None

    import time
    import json

    model = SirenNet().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)

    # optional console printer
    try:
        from utils.console import print_simple
    except Exception:
        def print_simple(a, b):
            print(f"{a}: {b}")

    # Prepare resume bookkeeping
    start_iter = 0
    best_state = None

    # Handle resume from checkpoint (supports dict checkpoints with optimizer state)
    if resume_from is not None:
        if not os.path.exists(resume_from):
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_from}")
        ckpt = torch.load(resume_from, map_location=device)
        if isinstance(ckpt, dict) and "model" in ckpt:
            model.load_state_dict(ckpt["model"])
            if "optim" in ckpt:
                optim.load_state_dict(ckpt["optim"])
            start_iter = ckpt.get("iter", 0)
            # initialize best_val and best_state from checkpoint
            best_val = ckpt.get("best_val", float("inf"))
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            print_simple("Resumed", f"from {resume_from} (iter={start_iter})")
        else:
            # legacy checkpoint (just state_dict)
            model.load_state_dict(ckpt)
            print_simple("Resumed", resume_from)

    best_val = float("inf") if best_state is None else best_val
    no_imp_counter = 0

    # optionally create an initial checkpoint (untrained) to make resume-trivial
    if checkpoint_path and checkpoint_interval == 0 and checkpoint_path and os.path.exists(os.path.dirname(checkpoint_path) or ".") and checkpoint_path and checkpoint_interval == 0 and False:
        # placeholder for future logic
        pass

    # bookkeeping for summary
    t0 = time.perf_counter()
    final_train_loss = None
    final_val_loss = None

    for it in range(start_iter, iters):
        model.train()
        z_pred = model(coords)
        loss = ((z_pred - z_gt) ** 2)[train_mask].mean()

        optim.zero_grad()
        loss.backward()
        optim.step()

        final_train_loss = loss.item()

        # periodic reporting
        if it % 500 == 0:
            print_simple("[SIREN]", f"iter {it:04d} | train loss = {loss.item():.6f}")

        # optional checkpointing (intermediate)
        if checkpoint_interval and checkpoint_interval > 0 and it % checkpoint_interval == 0 and it > 0 and checkpoint_path:
            cp_path = f"{checkpoint_path}.iter{it}.pth"
            torch.save({"model": model.state_dict(), "optim": optim.state_dict(), "iter": it, "best_val": best_val}, cp_path)
            print_simple("Checkpoint", cp_path)

        # validation / early stopping logic
        if val_mask is not None and (it % 50 == 0 or it == iters - 1):
            model.eval()
            with torch.no_grad():
                z_pred = model(coords)
                val_loss = ((z_pred - z_gt) ** 2)[val_mask].mean().item()

            final_val_loss = val_loss
            print_simple("Validation", f"iter {it:04d} | val loss = {val_loss:.6f}")

            if val_loss < best_val - 1e-12:
                best_val = val_loss
                no_imp_counter = 0
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                if checkpoint_path:
                    torch.save({"model": best_state, "optim": optim.state_dict(), "iter": it, "best_val": best_val}, checkpoint_path)
                    print_simple("Saved best", checkpoint_path)
            else:
                no_imp_counter += 1

            if early_stopping_patience and no_imp_counter >= early_stopping_patience:
                print_simple("EarlyStopping", f"no improvement for {no_imp_counter} checks, stopping at iter {it}")
                break

    # load best state if available
    if best_state is not None:
        model.load_state_dict(best_state)

    # save training summary if requested via checkpoint_path or always alongside checkpoint
    iterations_run = (it + 1) if "it" in locals() else start_iter
    summary = {
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        "best_val": best_val,
        "iterations_run": iterations_run,
    }
    summary_path = None
    if checkpoint_path:
        summary_path = checkpoint_path + ".summary.json"
    else:
        summary_path = "training.summary.json"

    try:
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print_simple("Saved summary", summary_path)
    except Exception:
        pass

    return model


def siren_normals_and_height(model: SirenNet, H: int, W: int, device: str = None) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate height and normals from SIREN using autograd.

    Returns a uint8 normal map and the mean-centered height image.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    y_range = torch.linspace(-1, 1, H, device=device)
    x_range = torch.linspace(-1, 1, W, device=device)
    yy, xx = torch.meshgrid(y_range, x_range, indexing="ij")

    coords = torch.stack([xx, yy], dim=-1).reshape(-1, 2)
    coords.requires_grad_(True)

    z = model(coords)  # (N, 1)

    grads = torch.autograd.grad(
        outputs=z.sum(), inputs=coords, create_graph=False, retain_graph=False
    )[0]

    dzdx_norm = grads[:, 0].reshape(H, W)
    dzdy_norm = grads[:, 1].reshape(H, W)

    dzdx = dzdx_norm * (2.0 / (W - 1))
    dzdy = dzdy_norm * (2.0 / (H - 1))

    n = torch.stack([-dzdx, -dzdy, torch.ones_like(dzdx)], dim=-1)
    n = torch.nn.functional.normalize(n, p=2, dim=-1)

    normal_map = (n + 1.0) * 0.5
    normal_map = (normal_map.clamp(0, 1) * 255).to(torch.uint8)

    return normal_map.cpu().numpy(), z.reshape(H, W).detach().cpu().numpy()


def siren_normals_and_height_supersample(model: SirenNet, H: int, W: int, device: str = None, scale: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """Supersample SIREN on a finer grid, compute normals, then downsample.

    Algorithm preserved exactly. Returns (normal_uint8, z_low).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    Hh = H * scale
    Wh = W * scale

    y_range = torch.linspace(-1, 1, Hh, device=device)
    x_range = torch.linspace(-1, 1, Wh, device=device)
    yy, xx = torch.meshgrid(y_range, x_range, indexing="ij")

    coords = torch.stack([xx, yy], dim=-1).reshape(-1, 2)
    coords.requires_grad_(True)

    z = model(coords)
    z = z - z.mean()
    z_img = z.reshape(Hh, Wh)

    # Compute gradients using the finite-difference normals_from_height logic
    # but keep original pipeline behavior (we reuse grads via autograd)
    grads = torch.autograd.grad(outputs=z.sum(), inputs=coords)[0]

    dzdx_norm = grads[:, 0].reshape(Hh, Wh)
    dzdy_norm = grads[:, 1].reshape(Hh, Wh)

    dzdx = dzdx_norm * (2.0 / (Wh - 1))
    dzdy = dzdy_norm * (2.0 / (Hh - 1))

    n_hi = torch.stack([-dzdx, -dzdy, torch.ones_like(dzdx)], dim=-1)
    n_hi = torch.nn.functional.normalize(n_hi, dim=-1)

    n_hi = n_hi.permute(2, 0, 1).unsqueeze(0)  # (1,3,Hh,Wh)

    n_lo = torch.nn.functional.avg_pool2d(n_hi, kernel_size=scale, stride=scale)
    n_lo = n_lo.squeeze(0).permute(1, 2, 0)
    n_lo = torch.nn.functional.normalize(n_lo, dim=-1)

    z_lo = torch.nn.functional.avg_pool2d(z_img.unsqueeze(0).unsqueeze(0), kernel_size=scale, stride=scale).squeeze()

    normal_uint8 = ((n_lo + 1.0) * 0.5 * 255.0).clamp(0, 255).byte()

    return normal_uint8.cpu().numpy(), z_lo.cpu().numpy()


def save_siren_model(model: SirenNet, path: str = "siren_weights.pth") -> None:
    torch.save(model.state_dict(), path)


def load_siren_model(model: SirenNet, path: str = "siren_weights.pth", device: str = None) -> bool:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if os.path.exists(path := path):
        ckpt = torch.load(path, map_location=device)
        if isinstance(ckpt, dict) and "model" in ckpt:
            model.load_state_dict(ckpt["model"])
        else:
            model.load_state_dict(ckpt)
        model.to(device)
        model.eval()
        return True
    return False
