"""Evaluation metrics: PSNR, SSIM, LPIPS, MS-SSIM, MAE and curl."""
from typing import Dict, Tuple
import numpy as np
import torch
import lpips
from pytorch_msssim import ms_ssim


def compute_detailed_metrics(pred_np: np.ndarray, gt_np: np.ndarray) -> Tuple[float, float, float, float]:
    """Compute PSNR, SSIM, MAE (deg) and curl error.

    Inputs are expected in [0,1].
    """
    p_val = ms_ssim  # placeholder to keep API similar; main eval below
    p_val = None
    p_val = None
    # For compatibility we reuse original PSNR/SSIM usage at the script level
    raise NotImplementedError("Use evaluate_normal_performance for full evaluation")


def evaluate_normal_performance(gt_normal: np.ndarray, pred_normal: np.ndarray, name: str = "Model", device: str = "cuda") -> Dict[str, float]:
    """Perceptual and geometric evaluation report.

    Returns dict with keys: lpips, ms_ssim, mae, curl
    """
    gt = gt_normal.astype(np.float32)
    pred = pred_normal.astype(np.float32)

    if gt.max() > 1.1:
        gt /= 255.0
    if pred.max() > 1.1:
        pred /= 255.0

    gt = np.clip(gt, 0.0, 1.0)
    pred = np.clip(pred, 0.0, 1.0)

    gt_t = torch.from_numpy(gt).permute(2, 0, 1).unsqueeze(0).to(device)
    pred_t = torch.from_numpy(pred).permute(2, 0, 1).unsqueeze(0).to(device)

    gt_lp = gt_t * 2.0 - 1.0
    pred_lp = pred_t * 2.0 - 1.0

    lpips_fn = lpips.LPIPS(net="alex").to(device)
    with torch.no_grad():
        lpips_val = lpips_fn(gt_lp, pred_lp).item()
        ms_ssim_val = ms_ssim(gt_t, pred_t, data_range=1.0, size_average=True).item()

    # geometric MAE
    v_gt = gt * 2.0 - 1.0
    v_pred = pred * 2.0 - 1.0
    v_gt /= (np.linalg.norm(v_gt, axis=-1, keepdims=True) + 1e-8)
    v_pred /= (np.linalg.norm(v_pred, axis=-1, keepdims=True) + 1e-8)
    dot = np.sum(v_gt * v_pred, axis=-1).clip(-1.0, 1.0)
    angles = np.arccos(dot) * (180.0 / np.pi)
    mae = np.mean(angles)

    nx, ny = v_pred[..., 0], v_pred[..., 1]
    dnx_dy = np.diff(nx, axis=0)[:, :-1]
    dny_dx = np.diff(ny, axis=1)[:-1, :]
    curl_err = np.mean(np.abs(dnx_dy - dny_dx))

    # Attempt to use the console helper for a colorful, unified report
    try:
        from utils.console import print_metrics_report
    except Exception:
        print("\n" + "=" * 55)
        print(f" EVALUATION REPORT: {name}")
        print("=" * 55)
        print(f" [Perceptual Quality]  LPIPS : {lpips_val:.6f} (lower is better)")
        print(f" [Multi-Scale Structure] MS-SSIM: {ms_ssim_val:.6f}")
        print(f" [Geometric Accuracy]  MAE   : {mae:.4f}°")
        print(f" [Physical Consistency] Curl : {curl_err:.8f}")
        print("=" * 55 + "\n")

        return {"lpips": lpips_val, "ms_ssim": ms_ssim_val, "mae": mae, "curl": curl_err}

    metrics = {
        "LPIPS": lpips_val,
        "MS-SSIM": ms_ssim_val,
        "MAE (deg)": mae,
        "Curl": curl_err,
    }
    units = {"MAE (deg)": "°"}

    print_metrics_report(f"EVALUATION REPORT: {name}", metrics, units=units)

    return {"lpips": lpips_val, "ms_ssim": ms_ssim_val, "mae": mae, "curl": curl_err}
