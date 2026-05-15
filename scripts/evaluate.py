"""Evaluate restored frames/videos against GT with PSNR, SSIM, LPIPS, FID, and tLPIPS."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from metrics_core import (  # noqa: E402
    evaluate_sequence,
    format_metric,
    load_sequence,
)

ALL_METRICS = ("psnr", "ssim", "lpips", "tlpips", "tlpips_noref", "fid", "temporal_mae")


def parse_metrics(raw: str) -> set[str]:
    names = {m.strip().lower() for m in raw.split(",") if m.strip()}
    unknown = names - set(ALL_METRICS)
    if unknown:
        raise ValueError(f"Unknown metrics: {sorted(unknown)}. Supported: {', '.join(ALL_METRICS)}")
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate pred vs GT frames or videos.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--pred_dir", type=str, help="Predicted frames directory.")
    src.add_argument("--pred_video", type=str, help="Predicted video path.")
    gt = parser.add_mutually_exclusive_group(required=True)
    gt.add_argument("--gt_dir", type=str, help="Ground-truth frames directory.")
    gt.add_argument("--gt_video", type=str, help="Ground-truth video path.")
    parser.add_argument(
        "--metrics",
        default="psnr,ssim,lpips,tlpips",
        help=f"Comma-separated metrics: {', '.join(ALL_METRICS)}",
    )
    parser.add_argument("--csv", default="output/tables/metrics.csv", help="Output CSV path.")
    parser.add_argument("--metric_csv", default="", help="Alias output path (e.g. metric.csv).")
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--short_side", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--label", default="pred", help="Row label written to CSV.")
    args = parser.parse_args()

    metrics = parse_metrics(args.metrics)
    device = torch.device(args.device)

    pred_path = args.pred_video or args.pred_dir
    gt_path = args.gt_video or args.gt_dir
    pred_frames = load_sequence(pred_path)
    gt_frames = load_sequence(gt_path)

    vals = evaluate_sequence(
        pred_frames,
        gt_frames,
        metrics=metrics,
        device=device,
        stride=args.stride,
        short_side=args.short_side,
    )

    row = {
        "dataset": "",
        "clip": Path(pred_path).stem,
        "method": args.label,
        "part": "",
        "frames": int(vals.get("frames", 0)),
        "psnr": format_metric(vals.get("psnr")),
        "ssim": format_metric(vals.get("ssim")),
        "lpips": format_metric(vals.get("lpips")),
        "tlpips": format_metric(vals.get("tlpips", vals.get("tlpips_noref"))),
        "fid": format_metric(vals.get("fid")),
        "temporal_mae": format_metric(vals.get("temporal_mae")),
        "has_gt": "true",
        "pred_path": str(pred_path),
        "gt_path": str(gt_path),
    }

    fieldnames = list(row.keys())
    for out in {args.csv, args.metric_csv}:
        if not out:
            continue
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(row)
        print(f"Wrote {out_path}")

    summary = ", ".join(f"{k}={row[k]}" for k in ("psnr", "ssim", "lpips", "tlpips", "fid") if row.get(k))
    print(f"{args.label}: {summary}")


if __name__ == "__main__":
    main()
