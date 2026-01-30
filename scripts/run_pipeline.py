"""Run the full pipeline over a directory of normal maps.

Usage example:
    python -m scripts.run_pipeline --root ./exper --device cpu --weights high_res/weights.pth

This script supports direct execution as well. If you run it via
`python scripts/run_pipeline.py` we ensure the project root is added to
`sys.path` to allow package-style imports.
"""
import sys
import argparse
import logging
import os
import time

# Make script runnable directly by adding repo root to sys.path if needed
try:
    import utils  # type: ignore
except Exception:
    project_root = os.path.dirname(os.path.dirname(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


from utils.io import load_normal_map
from core.geometry import compute_gradients, compute_divergence, normals_from_height
from core.poisson import solve_poisson_neumann
from core.siren import SirenNet, load_siren_model, siren_normals_and_height
from utils.visualization import plot_all, angle_error
from evaluation.metrics import evaluate_normal_performance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="./exper", help="input folder with normal PNGs")
    parser.add_argument("--device", type=str, default=("cuda" if __import__("torch").cuda.is_available() else "cpu"))
    parser.add_argument("--weights", type=str, default="high_res/weights_group3_small.png.pth")
    parser.add_argument("--out", type=str, default="outputs")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("pipeline")

    for file in os.listdir(args.root):
        if not file.lower().endswith(".png"):
            continue

        logger.info("Processing %s", file)
        start = time.perf_counter()

        path = os.path.join(args.root, file)
        normal = load_normal_map(path)
        valid = normal[..., 2] > 0.1

        p, q = compute_gradients(normal)
        p[~valid] = 0
        q[~valid] = 0

        f = compute_divergence(p, q)
        z_poisson = solve_poisson_neumann(f, p, q)
        n_poisson = normals_from_height(z_poisson)

        model = SirenNet().to(args.device)
        loaded = load_siren_model(model, args.weights, device=args.device)
        if not loaded:
            logger.warning("Weight file %s not found. Using untrained SIREN.", args.weights)

        n_siren, z_siren = siren_normals_and_height(model, *z_poisson.shape, device=args.device)

        gt_uint8 = ((normal + 1) * 0.5 * 255).astype("uint8")
        diff = angle_error(gt_uint8, n_siren)

        n_siren_from_z = normals_from_height(z_siren)

        outname = os.path.join(args.out, file.replace('.png', '_siren'))
        plot_all(gt_uint8, z_poisson, n_poisson, z_siren, n_siren_from_z, diff, outname)

        evaluate_normal_performance(gt_uint8, n_siren_from_z, name=file)

        end = time.perf_counter()
        logger.info("Elapsed time: %.4f s", end - start)


if __name__ == "__main__":
    main()
