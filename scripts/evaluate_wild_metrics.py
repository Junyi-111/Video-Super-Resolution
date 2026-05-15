"""Evaluate Wild synthetic LR outputs against the original captured video."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from metrics_core import evaluate_sequence, evaluate_wild_jobs_unified, format_metric, load_sequence  # noqa: E402


def _part_for_method(name: str) -> str:
    if name == "LR input":
        return "input"
    if name.startswith("Part3"):
        return "part3"
    if name in ("Real-ESRGAN", "BasicVSR"):
        return "part2"
    return "part1"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Wild SR outputs with pseudo-GT.")
    parser.add_argument("--gt", default="dataset/wild_real.mp4")
    parser.add_argument("--lr", default="dataset/wild_real_lr.mp4")
    parser.add_argument("--out_csv", default="output/tables/wild_metrics.csv")
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--short_side", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--unified-eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Same streaming-eval ROI + min-T as Part3 / evaluate_project (default on).",
    )
    args = parser.parse_args()

    methods = [
        ("LR input", "dataset/wild_real_lr.mp4"),
        ("Bicubic", "part1_baseline/outputs/wild_real_lr/wild_real_lr_part1_bicubic_x4.mp4"),
        ("Lanczos", "part1_baseline/outputs/wild_real_lr/wild_real_lr_part1_lanczos_x4.mp4"),
        (
            "Temporal+Unsharp",
            "part1_baseline/outputs/wild_real_lr/wild_real_lr_part1_temporal_w5_unsharp_x4.mp4",
        ),
        ("Real-ESRGAN", "part2_sota/outputs/wild_real_lr/realesrgan_x4.mp4"),
        ("BasicVSR", "part2_sota/outputs/basicvsr/wild_real_lr.mp4"),
        ("Part3 Hybrid", "part3_exploration/outputs/wild_real_lr_hybrid_x4.mp4"),
        ("Part3 Flow-Stabilized", "part3_exploration/outputs/wild_real_lr_flow_stabilized_x4.mp4"),
        ("Part3 Uncertainty-Adaptive", "part3_exploration/outputs/wild_real_lr_uncertainty_adaptive_x4.mp4"),
    ]

    device = torch.device(args.device)
    metrics = {"psnr", "ssim", "temporal_mae"}
    rows: list[dict] = []

    if args.unified_eval:
        jobs = []
        for name, path in methods:
            if not (REPO_ROOT / path).is_file():
                continue
            jobs.append(
                {
                    "dataset": "wild",
                    "clip": "wild_real_lr",
                    "method": name,
                    "part": _part_for_method(name),
                    "pred": path.replace("\\", "/"),
                }
            )
        proj_rows = evaluate_wild_jobs_unified(
            REPO_ROOT,
            jobs,
            wild_gt_rel=args.gt,
            wild_lr_rel=args.lr,
            metrics=metrics,
            device=device,
            stride=args.stride,
            short_side=args.short_side,
            loss_fn=None,
        )
        for r in proj_rows:
            rows.append(
                {
                    "method": r["method"],
                    "frames": int(r["frames"]),
                    "psnr": r["psnr"],
                    "ssim": r["ssim"],
                    "temporal_mae": r["temporal_mae"],
                    "video": next(p for n, p in methods if n == r["method"]),
                    "eval_policy": r.get("eval_policy", ""),
                    "crop_twh": r.get("crop_twh", ""),
                }
            )
            print(f"{r['method']}: PSNR={r['psnr']}, SSIM={r['ssim']}")
        fieldnames = ["method", "frames", "psnr", "ssim", "temporal_mae", "video", "eval_policy", "crop_twh"]
    else:
        gt_frames = load_sequence(REPO_ROOT / args.gt)
        for name, path in methods:
            pred_path = REPO_ROOT / path
            if not pred_path.is_file():
                continue
            pred_frames = load_sequence(pred_path)
            vals = evaluate_sequence(
                pred_frames,
                gt_frames,
                metrics=metrics,
                device=device,
                stride=args.stride,
                short_side=args.short_side,
            )
            rows.append(
                {
                    "method": name,
                    "frames": int(vals["frames"]),
                    "psnr": format_metric(vals.get("psnr")),
                    "ssim": format_metric(vals.get("ssim")),
                    "temporal_mae": format_metric(vals.get("temporal_mae")),
                    "video": path,
                    "eval_policy": "legacy",
                    "crop_twh": "",
                }
            )
            print(f"{name}: PSNR={rows[-1]['psnr']}, SSIM={rows[-1]['ssim']}")
        fieldnames = ["method", "frames", "psnr", "ssim", "temporal_mae", "video", "eval_policy", "crop_twh"]

    out_csv = Path(args.out_csv)
    if not out_csv.is_absolute():
        out_csv = REPO_ROOT / out_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
