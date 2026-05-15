"""Batch evaluation for mandatory Wild + sample datasets; writes metrics.csv and metric.csv."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from metrics_core import (  # noqa: E402
    evaluate_sample_stem_unified,
    evaluate_sequence,
    evaluate_wild_jobs_unified,
    format_metric,
    load_sequence,
)

DEFAULT_METRICS = ("psnr", "ssim", "lpips", "tlpips", "fid", "temporal_mae")


def discover_wild_jobs() -> list[dict[str, str]]:
    return [
        {"dataset": "wild", "clip": "wild_real_lr", "method": "LR input", "part": "input", "pred": "dataset/wild_real_lr.mp4"},
        {
            "dataset": "wild",
            "clip": "wild_real_lr",
            "method": "Bicubic",
            "part": "part1",
            "pred": "part1_baseline/outputs/wild_real_lr/wild_real_lr_part1_bicubic_x4.mp4",
        },
        {
            "dataset": "wild",
            "clip": "wild_real_lr",
            "method": "Lanczos",
            "part": "part1",
            "pred": "part1_baseline/outputs/wild_real_lr/wild_real_lr_part1_lanczos_x4.mp4",
        },
        {
            "dataset": "wild",
            "clip": "wild_real_lr",
            "method": "Temporal+Unsharp",
            "part": "part1",
            "pred": "part1_baseline/outputs/wild_real_lr/wild_real_lr_part1_temporal_w5_unsharp_x4.mp4",
        },
        {
            "dataset": "wild",
            "clip": "wild_real_lr",
            "method": "Real-ESRGAN",
            "part": "part2",
            "pred": "part2_sota/outputs/wild_real_lr/realesrgan_x4.mp4",
        },
        {
            "dataset": "wild",
            "clip": "wild_real_lr",
            "method": "BasicVSR",
            "part": "part2",
            "pred": "part2_sota/outputs/basicvsr/wild_real_lr.mp4",
        },
        {
            "dataset": "wild",
            "clip": "wild_real_lr",
            "method": "Part3 Hybrid",
            "part": "part3",
            "pred": "part3_exploration/outputs/wild_real_lr_hybrid_x4.mp4",
        },
        {
            "dataset": "wild",
            "clip": "wild_real_lr",
            "method": "Part3 Flow-Stabilized",
            "part": "part3",
            "pred": "part3_exploration/outputs/wild_real_lr_flow_stabilized_x4.mp4",
        },
        {
            "dataset": "wild",
            "clip": "wild_real_lr",
            "method": "Part3 Uncertainty-Adaptive",
            "part": "part3",
            "pred": "part3_exploration/outputs/wild_real_lr_uncertainty_adaptive_x4.mp4",
        },
    ]


def resolve_gt_video(stem: str, gt_roots: list[Path]) -> Path | None:
    for root in gt_roots:
        for candidate in (
            root / f"{stem}.mp4",
            root / stem / "gt.mp4",
            root / stem / f"{stem}.mp4",
        ):
            if candidate.is_file():
                return candidate
    return None


def discover_sample_jobs(
    inputs_dir: Path,
    part1_root: Path,
    part2_basicvsr_root: Path,
    part2_realesrgan_root: Path,
    gt_roots: list[Path],
) -> list[dict[str, str]]:
    jobs: list[dict[str, str]] = []
    if not inputs_dir.is_dir():
        return jobs

    part1_outputs: dict[str, tuple[str, Path]] = {}
    if part1_root.is_dir():
        for mp4 in part1_root.rglob("*.mp4"):
            stem = mp4.stem
            if "_part1_bicubic_" in stem:
                part1_outputs.setdefault(stem.split("_part1_bicubic_")[0], ("Bicubic", mp4))
            elif "_part1_temporal_" in stem:
                part1_outputs.setdefault(stem.split("_part1_temporal_")[0], ("Temporal+Unsharp", mp4))

    part2_outputs: dict[str, tuple[str, Path]] = {}
    for root, method in ((part2_basicvsr_root, "BasicVSR"), (part2_realesrgan_root, "Real-ESRGAN")):
        if not root.is_dir():
            continue
        for mp4 in root.glob("*.mp4"):
            if mp4.name.startswith("wild"):
                continue
            part2_outputs.setdefault(mp4.stem, (method, mp4))

    for inp in sorted(inputs_dir.glob("*.mp4")):
        if inp.name.startswith("wild"):
            continue
        stem = inp.stem
        dataset = "reds-sample" if stem.startswith("REDS-sample") else "vimeo-rl"
        gt = resolve_gt_video(stem, gt_roots)
        gt_str = str(gt.relative_to(REPO_ROOT)).replace("\\", "/") if gt else ""

        method_paths: list[tuple[str, str, Path]] = []
        if stem in part1_outputs:
            method, path = part1_outputs[stem]
            method_paths.append((method, "part1", path))
        if stem in part2_outputs:
            method, path = part2_outputs[stem]
            method_paths.append((method, "part2", path))

        for method, part, pred in method_paths:
            jobs.append(
                {
                    "dataset": dataset,
                    "clip": stem,
                    "method": method,
                    "part": part,
                    "pred": str(pred.relative_to(REPO_ROOT)).replace("\\", "/"),
                    "gt": gt_str,
                }
            )
    return jobs


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate all project outputs and write metrics.csv.")
    parser.add_argument("--wild_gt", default="dataset/wild_real.mp4")
    parser.add_argument("--wild_lr", default="dataset/wild_real_lr.mp4")
    parser.add_argument("--inputs_dir", default="dataset/inputs_mp4")
    parser.add_argument("--gt_dir", default="dataset/gt_mp4", help="Optional GT mp4 root for sample clips.")
    parser.add_argument("--out_csv", default="output/tables/metrics.csv")
    parser.add_argument("--metric_csv", default="output/tables/metric.csv", help="Course alias output.")
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--short_side", type=int, default=256)
    parser.add_argument("--skip_fid", action="store_true", help="Skip FID (faster).")
    parser.add_argument("--skip_sample", action="store_true")
    parser.add_argument("--skip_wild", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--unified-eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="128-multiple crop on 4× LR canvas + min temporal length (same as Part3).",
    )
    args = parser.parse_args()

    metrics = set(DEFAULT_METRICS)
    if args.skip_fid:
        metrics.discard("fid")

    device = torch.device(args.device)
    rows: list[dict[str, str]] = []
    fieldnames = [
        "dataset",
        "clip",
        "method",
        "part",
        "frames",
        "psnr",
        "ssim",
        "lpips",
        "tlpips",
        "tlpips_noref",
        "fid",
        "temporal_mae",
        "has_gt",
        "pred_path",
        "gt_path",
        "eval_policy",
        "crop_twh",
    ]

    loss_fn = None
    if metrics & {"lpips", "tlpips", "tlpips_noref"}:
        import lpips

        loss_fn = lpips.LPIPS(net="alex").to(device).eval()

    if not args.skip_wild:
        wild_gt_path = REPO_ROOT / args.wild_gt
        if not wild_gt_path.is_file():
            print(f"Warning: wild GT not found: {wild_gt_path}")
        elif args.unified_eval:
            urows = evaluate_wild_jobs_unified(
                REPO_ROOT,
                discover_wild_jobs(),
                wild_gt_rel=args.wild_gt,
                wild_lr_rel=args.wild_lr,
                metrics=metrics,
                device=device,
                stride=args.stride,
                short_side=args.short_side,
                loss_fn=loss_fn,
            )
            rows.extend(urows)
            for r in urows:
                print(
                    f"[wild] {r['method']}: PSNR={r['psnr']} SSIM={r['ssim']} "
                    f"LPIPS={r['lpips']} tLPIPS={r['tlpips']} FID={r['fid']} frames={r['frames']}"
                )
        else:
            wild_gt = load_sequence(wild_gt_path)
            for job in discover_wild_jobs():
                pred_path = REPO_ROOT / job["pred"]
                if not pred_path.is_file():
                    print(f"Skip missing: {pred_path}")
                    continue
                pred_frames = load_sequence(pred_path)
                vals = evaluate_sequence(
                    pred_frames,
                    wild_gt,
                    metrics=metrics,
                    device=device,
                    stride=args.stride,
                    short_side=args.short_side,
                    loss_fn=loss_fn,
                    reuse_loss_fn=True,
                )
                row = {
                    "dataset": job["dataset"],
                    "clip": job["clip"],
                    "method": job["method"],
                    "part": job["part"],
                    "frames": str(int(vals.get("frames", 0))),
                    "psnr": format_metric(vals.get("psnr")),
                    "ssim": format_metric(vals.get("ssim")),
                    "lpips": format_metric(vals.get("lpips")),
                    "tlpips": format_metric(vals.get("tlpips")),
                    "tlpips_noref": format_metric(vals.get("tlpips_noref")),
                    "fid": format_metric(vals.get("fid")),
                    "temporal_mae": format_metric(vals.get("temporal_mae")),
                    "has_gt": "true",
                    "pred_path": job["pred"],
                    "gt_path": args.wild_gt,
                    "eval_policy": "legacy",
                    "crop_twh": "",
                }
                rows.append(row)
                print(
                    f"[wild] {job['method']}: PSNR={row['psnr']} SSIM={row['ssim']} "
                    f"LPIPS={row['lpips']} tLPIPS={row['tlpips']} FID={row['fid']}"
                )

    if not args.skip_sample:
        gt_roots = [REPO_ROOT / args.gt_dir]
        sample_jobs = discover_sample_jobs(
            REPO_ROOT / args.inputs_dir,
            REPO_ROOT / "part1_baseline/outputs",
            REPO_ROOT / "part2_sota/outputs/basicvsr",
            REPO_ROOT / "part2_sota/outputs/realesrgan",
            gt_roots,
        )
        if args.unified_eval:
            by_clip: dict[str, list[dict[str, str]]] = defaultdict(list)
            for job in sample_jobs:
                by_clip[job["clip"]].append(job)
            for clip in sorted(by_clip.keys()):
                jlist = by_clip[clip]
                lr_path = REPO_ROOT / args.inputs_dir / f"{clip}.mp4"
                gt_str = jlist[0].get("gt", "")
                gt_path = REPO_ROOT / gt_str if gt_str else None
                srows = evaluate_sample_stem_unified(
                    REPO_ROOT,
                    lr_path,
                    gt_path,
                    jlist,
                    metrics=metrics,
                    device=device,
                    stride=args.stride,
                    short_side=args.short_side,
                    loss_fn=loss_fn,
                )
                rows.extend(srows)
                for r in srows:
                    print(f"[{r['dataset']}/{r['clip']}] {r['method']}: tLPIPS={r['tlpips']}")
        else:
            for job in sample_jobs:
                pred_path = REPO_ROOT / job["pred"]
                gt_frames = None
                has_gt = "false"
                gt_path_str = job.get("gt", "")
                if gt_path_str:
                    gt_frames = load_sequence(REPO_ROOT / gt_path_str)
                    has_gt = "true"
                pred_frames = load_sequence(pred_path)
                sample_metrics = set(metrics)
                if gt_frames is None:
                    sample_metrics -= {"psnr", "ssim", "lpips", "fid", "temporal_mae"}
                    sample_metrics.add("tlpips_noref")
                vals = evaluate_sequence(
                    pred_frames,
                    gt_frames,
                    metrics=sample_metrics,
                    device=device,
                    stride=args.stride,
                    short_side=args.short_side,
                    loss_fn=loss_fn,
                    reuse_loss_fn=True,
                )
                row = {
                    "dataset": job["dataset"],
                    "clip": job["clip"],
                    "method": job["method"],
                    "part": job["part"],
                    "frames": str(int(vals.get("frames", 0))),
                    "psnr": format_metric(vals.get("psnr")),
                    "ssim": format_metric(vals.get("ssim")),
                    "lpips": format_metric(vals.get("lpips")),
                    "tlpips": format_metric(vals.get("tlpips", vals.get("tlpips_noref"))),
                    "tlpips_noref": format_metric(vals.get("tlpips_noref")),
                    "fid": format_metric(vals.get("fid")),
                    "temporal_mae": format_metric(vals.get("temporal_mae")),
                    "has_gt": has_gt,
                    "pred_path": job["pred"],
                    "gt_path": gt_path_str,
                    "eval_policy": "legacy",
                    "crop_twh": "",
                }
                rows.append(row)
                print(f"[{job['dataset']}/{job['clip']}] {job['method']}: tLPIPS={row['tlpips']}")

    if not rows:
        raise RuntimeError("No evaluation rows produced. Check that output videos exist.")

    write_csv(REPO_ROOT / args.out_csv, rows, fieldnames)
    write_csv(REPO_ROOT / args.metric_csv, rows, fieldnames)
    print(f"Wrote {args.out_csv} and {args.metric_csv} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
