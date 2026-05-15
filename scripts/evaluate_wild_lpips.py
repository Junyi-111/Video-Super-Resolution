"""Compute LPIPS and temporal LPIPS for Wild outputs."""

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
    parser = argparse.ArgumentParser(description="Evaluate Wild LPIPS metrics.")
    parser.add_argument("--gt", default="dataset/wild_real.mp4")
    parser.add_argument("--lr", default="dataset/wild_real_lr.mp4")
    parser.add_argument("--out_csv", default="output/tables/wild_lpips_metrics.csv")
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
    metrics = {"lpips", "tlpips", "tlpips_noref"}
    import lpips

    loss_fn = lpips.LPIPS(net="alex").to(device).eval()

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
            loss_fn=loss_fn,
        )
        for r in proj_rows:
            path = next(p for n, p in methods if n == r["method"])
            rows.append(
                {
                    "method": r["method"],
                    "lpips": r["lpips"],
                    "tlpips": r["tlpips"],
                    "tlpips_proxy": r.get("tlpips_noref", ""),
                    "sampled_frames": int(r["frames"]),
                    "temporal_pairs": max(0, int(r["frames"]) // args.stride),
                    "video": path,
                    "eval_policy": r.get("eval_policy", ""),
                    "crop_twh": r.get("crop_twh", ""),
                }
            )
            if not rows[-1]["tlpips_proxy"]:
                rows[-1]["tlpips_proxy"] = ""
            print(f"{r['method']}: LPIPS={r['lpips']}, tLPIPS={r['tlpips']}")
        fieldnames = [
            "method",
            "lpips",
            "tlpips",
            "tlpips_proxy",
            "sampled_frames",
            "temporal_pairs",
            "video",
            "eval_policy",
            "crop_twh",
        ]
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
                loss_fn=loss_fn,
                reuse_loss_fn=True,
            )
            rows.append(
                {
                    "method": name,
                    "lpips": format_metric(vals.get("lpips")),
                    "tlpips": format_metric(vals.get("tlpips")),
                    "tlpips_proxy": format_metric(vals.get("tlpips_noref")),
                    "sampled_frames": int(vals.get("frames", 0)),
                    "temporal_pairs": max(0, int(vals.get("frames", 0)) // args.stride),
                    "video": path,
                    "eval_policy": "legacy",
                    "crop_twh": "",
                }
            )
            print(f"{name}: LPIPS={rows[-1]['lpips']}, tLPIPS={rows[-1]['tlpips']}")
        fieldnames = [
            "method",
            "lpips",
            "tlpips",
            "tlpips_proxy",
            "sampled_frames",
            "temporal_pairs",
            "video",
            "eval_policy",
            "crop_twh",
        ]

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
