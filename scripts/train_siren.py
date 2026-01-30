"""Train SIREN on a single normal map (Poisson-derived height) or a folder of normal maps.

Usage example:
    python -m scripts.train_siren --input ./exper/sample.png --iters 1000 --device cpu --out weights.pth

This script is safe to run both as a module (`python -m scripts.train_siren`)
and directly (`python scripts/train_siren.py`). When run directly we add the
project root to `sys.path` so package-style imports like `utils.*` work.
"""
import sys
import argparse
import logging
import os

# Allow running the script directly (not as a package) by ensuring the
# repository root is on sys.path so `import utils` and `import core` succeed.
try:
    import utils  # type: ignore
except Exception:
    project_root = os.path.dirname(os.path.dirname(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)



import numpy as np

from utils.io import load_normal_map
from core.geometry import compute_gradients, compute_divergence
from core.poisson import solve_poisson_neumann
from core.siren import fit_siren_height, save_siren_model, SirenNet, load_siren_model
from utils.console import print_simple


def train_on_file(path: str, iters: int, device: str, out: str, val_split: float = 0.0, checkpoint: str = None, checkpoint_interval: int = 0, early_stopping_patience: int = 0, resume: str = None, force_init_checkpoint: bool = False) -> None:
    normal = load_normal_map(path)
    valid = normal[..., 2] > 0.1

    # Create an initial checkpoint (untrained model + optim state) if requested
    if force_init_checkpoint and checkpoint is not None:
        tmp = SirenNet()
        torch.save({"model": tmp.state_dict(), "optim": torch.optim.Adam(tmp.parameters()).state_dict(), "iter": 0, "best_val": float("inf")}, checkpoint)
        print_simple("Force checkpoint", f"wrote initial checkpoint to {checkpoint}")
    p, q = compute_gradients(normal)
    p[~valid] = 0
    q[~valid] = 0

    f = compute_divergence(p, q)
    z_poisson = solve_poisson_neumann(f, p, q)

    # Inform about checkpoint/resume arguments (helpful for debugging typos)
    if checkpoint is None:
        print_simple("Checkpoint", "disabled (pass --checkpoint <path> to enable)")
    else:
        print_simple("Checkpoint", f"will save to {checkpoint}")

    if resume is not None and not os.path.exists(resume):
        print_simple("Resume", f"requested resume file not found: {resume}")

    print_simple("Training", f"on {os.path.basename(path)} (iters={iters})")
    model = fit_siren_height(
        z_poisson,
        valid,
        iters=iters,
        device=device,
        val_split=val_split,
        checkpoint_path=checkpoint,
        checkpoint_interval=checkpoint_interval,
        early_stopping_patience=early_stopping_patience,
        resume_from=resume,
    )

    # If checkpoint exists (best), prefer to save it as final output
    if checkpoint and os.path.exists(checkpoint):
        # load best and save to requested out
        from core.siren import load_siren_model
        tmp_model = SirenNet().to(device)
        load_siren_model(tmp_model, checkpoint, device=device)
        save_siren_model(tmp_model, out)
        print_simple("Saved best", out)
    else:
        save_siren_model(model, out)
        print_simple("Saved", out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=None, help="single normal PNG input")
    parser.add_argument("--root", type=str, default=None, help="folder of normal PNGs to train on (will train separately per-file)")
    parser.add_argument("--iters", type=int, default=1000, help="training iterations (default 1000)")
    parser.add_argument("--device", type=str, default=("cuda" if __import__("torch").cuda.is_available() else "cpu"))
    parser.add_argument("--out", type=str, default="siren_weights.pth", help="output weights path")
    parser.add_argument("--val-split", type=float, default=0.0, help="fraction of valid pixels used for validation (0 disables)")
    parser.add_argument("--checkpoint", type=str, default=None, help="path to save best checkpoint (e.g., outputs/best.pth)")
    parser.add_argument("--checkpoint-interval", type=int, default=0, help="save intermediate checkpoint every N iterations (0 disables)")
    parser.add_argument("--force-init-checkpoint", action="store_true", help="create initial checkpoint before training (useful to enable resume even without validation)")
    parser.add_argument("--early-stopping-patience", type=int, default=0, help="number of val checks with no improvement before early stopping (0 disables)")
    parser.add_argument("--resume", type=str, default=None, help="path to a checkpoint to resume from")
    args = parser.parse_args()

    # strict resume check: fail fast if resume requested but file not present
    if args.resume and not os.path.exists(args.resume):
        raise SystemExit(f"Requested resume checkpoint not found: {args.resume}")
    logging.basicConfig(level=logging.INFO)

    if args.input is None and args.root is None:
        raise SystemExit("Provide either --input <file.png> or --root <folder>")

    if args.input:
        train_on_file(
            args.input,
            args.iters,
            args.device,
            args.out,
            val_split=args.val_split,
            checkpoint=args.checkpoint,
            checkpoint_interval=args.checkpoint_interval,
            early_stopping_patience=args.early_stopping_patience,
            resume=args.resume,
            force_init_checkpoint=args.force_init_checkpoint,
        )

    if args.root:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        for file in os.listdir(args.root):
            if not file.lower().endswith('.png'):
                continue
            base_out = args.out
            # name output per-file if a folder is given
            out_name = os.path.splitext(base_out)[0] + f"_{os.path.splitext(file)[0]}.pth"
            train_on_file(
                os.path.join(args.root, file),
                args.iters,
                args.device,
                out_name,
                val_split=args.val_split,
                checkpoint=args.checkpoint,
                checkpoint_interval=args.checkpoint_interval,
                early_stopping_patience=args.early_stopping_patience,
                resume=args.resume,
                force_init_checkpoint=args.force_init_checkpoint,
            )


if __name__ == "__main__":
    main()
