"""Aggregate Part 3 direction metrics (per clip + summary) for the report."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch

# This file lives in part3_exploration/; dataset + output paths are relative to CV repo root.
_PKG_ROOT = Path(__file__).resolve().parent
_CV_ROOT = _PKG_ROOT.parent
_SCRIPTS_CV = _CV_ROOT / "scripts"
if _SCRIPTS_CV.is_dir() and str(_SCRIPTS_CV) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_CV))

from metrics_core import (  # noqa: E402
    align_sequences_streaming_vsr_rule,
    count_video_frames,
    evaluate_sequence,
    format_metric,
    load_sequence,
    read_lr_whn,
    streaming_vsr_canvas_dims,
)

VARIANTS: list[tuple[str, str, str]] = [
    ("part3_A_rectified_flow", "part3_exploration/outputs/direction_a_flow_matching", "_rectified_flow_x4.mp4"),
    ("part3_B_flow_stabilized", "part3_exploration/outputs/direction_b_sd_controlnet", "_flow_stabilized_x4.mp4"),
    ("part3_B_temporal_gen_blend", "part3_exploration/outputs/direction_b_sd_controlnet", "_temporal_gen_blend_x4.mp4"),
    ("part3_C_uncertainty_adaptive", "part3_exploration/outputs/direction_c_uncertainty", "_uncertainty_adaptive_x4.mp4"),
    ("part3_D_streaming_distilled", "part3_exploration/outputs/direction_d_distilled_streaming", "_streaming_distilled_x4.mp4"),
]

LEGACY_WILD: dict[str, str] = {
    "part3_B_temporal_gen_blend": "part3_exploration/outputs/wild_real_lr_hybrid_x4.mp4",
    "part3_B_flow_stabilized": "part3_exploration/outputs/wild_real_lr_flow_stabilized_x4.mp4",
    "part3_C_uncertainty_adaptive": "part3_exploration/outputs/wild_real_lr_uncertainty_adaptive_x4.mp4",
}


def collect_stems(inputs_dir: Path, wild_lr: Path) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    if wild_lr.is_file():
        out.append((wild_lr.stem, "wild"))
    if inputs_dir.is_dir():
        for p in sorted(inputs_dir.glob("*.mp4")):
            if p.name.startswith("wild"):
                continue
            out.append((p.stem, "reds-sample" if p.stem.startswith("REDS-sample") else "vimeo-rl"))
    return out


def resolve_pred(method: str, rel_out_dir: str, suffix: str, stem: str) -> Path:
    return _CV_ROOT / rel_out_dir / f"{stem}{suffix}"


def resolve_pred_with_legacy(method: str, rel_dir: str, suffix: str, stem: str) -> Path | None:
    pred = resolve_pred(method, rel_dir, suffix, stem)
    if pred.is_file():
        return pred
    if stem == "wild_real_lr" and method in LEGACY_WILD:
        cand = _CV_ROOT / LEGACY_WILD[method]
        if cand.is_file():
            return cand
    return None


def lr_path_for_stem(stem: str, wild_lr: Path, inputs_dir: Path) -> Path:
    if stem == wild_lr.stem:
        return wild_lr
    return inputs_dir / f"{stem}.mp4"


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate all Part 3 direction outputs.")
    ap.add_argument("--inputs_dir", default="dataset/inputs_mp4")
    ap.add_argument("--wild_lr", default="dataset/wild_real_lr.mp4")
    ap.add_argument("--wild_gt", default="dataset/wild_real.mp4")
    ap.add_argument("--gt_mp4_dir", default="dataset/gt_mp4")
    ap.add_argument("--per_clip_csv", default="output/tables/part3_metrics_per_clip.csv")
    ap.add_argument("--summary_csv", default="output/tables/part3_metrics_summary.csv")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--short_side", type=int, default=256)
    ap.add_argument("--skip_fid", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument(
        "--unified-eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Align all directions: 128-multiple center crop on 4脳 LR canvas + same temporal length (min over GT and existing preds). Use --no-unified-eval for legacy per-method min(T,GT).",
    )
    args = ap.parse_args()

    device = torch.device(args.device)
    metrics = {"psnr", "ssim", "lpips", "tlpips", "temporal_mae"}
    if not args.skip_fid:
        metrics.add("fid")

    import lpips

    loss_fn = lpips.LPIPS(net="alex").to(device).eval()

    wild_gt_path = _CV_ROOT / args.wild_gt
    gt_dir = _CV_ROOT / args.gt_mp4_dir
    inputs_dir = _CV_ROOT / args.inputs_dir
    wild_lr = _CV_ROOT / args.wild_lr

    per_clip_rows: list[dict[str, str]] = []

    if args.unified_eval:
        for stem, dataset in collect_stems(inputs_dir, wild_lr):
            lr_p = lr_path_for_stem(stem, wild_lr, inputs_dir)
            if not lr_p.is_file():
                continue
            try:
                w0, h0, _n_lr = read_lr_whn(lr_p)
            except (RuntimeError, OSError) as e:
                print(f"[skip {stem}] LR read failed: {e}", file=sys.stderr)
                continue
            s_w, s_h, t_w, t_h = streaming_vsr_canvas_dims(w0, h0, scale=4, multiple=128)

            gt_path: Path | None
            if dataset == "wild":
                gt_path = wild_gt_path if wild_gt_path.is_file() else None
            else:
                cand = gt_dir / f"{stem}.mp4"
                gt_path = cand if cand.is_file() else None

            pred_by_method: dict[str, Path] = {}
            for method, rel_dir, suffix in VARIANTS:
                rp = resolve_pred_with_legacy(method, rel_dir, suffix, stem)
                if rp is not None:
                    pred_by_method[method] = rp

            if not pred_by_method:
                continue

            counts: list[int] = [count_video_frames(p) for p in pred_by_method.values()]
            if gt_path is not None and gt_path.is_file():
                counts.append(count_video_frames(gt_path))
            if min(counts) <= 0:
                continue

            loaded: dict[str, list] = {}
            for method, pth in pred_by_method.items():
                try:
                    loaded[method] = load_sequence(pth)
                except (RuntimeError, OSError) as e:
                    print(f"[skip {stem} {method}] pred load: {e}", file=sys.stderr)
                    loaded[method] = []
            loaded = {m: fr for m, fr in loaded.items() if fr}
            if not loaded:
                continue

            gt_list: list | None = None
            if gt_path is not None and gt_path.is_file():
                try:
                    gt_list = load_sequence(gt_path)
                except (RuntimeError, OSError) as e:
                    print(f"[warn {stem}] GT load failed: {e}", file=sys.stderr)
                    gt_list = None

            lens = [len(fr) for fr in loaded.values()]
            if gt_list is not None:
                lens.append(len(gt_list))
            n_eval = min(lens)

            for method, rel_dir, suffix in VARIANTS:
                pred = pred_by_method.get(method)
                if pred is None or method not in loaded:
                    continue
                pred_frames = loaded[method][:n_eval]

                pred_a, gt_a = align_sequences_streaming_vsr_rule(
                    pred_frames,
                    gt_list,
                    s_h=s_h,
                    s_w=s_w,
                    t_h=t_h,
                    t_w=t_w,
                    n_eval=n_eval,
                )

                mset = set(metrics)
                if gt_a is None:
                    mset = {"tlpips_noref"}
                vals = evaluate_sequence(
                    pred_a,
                    gt_a,
                    metrics=mset,
                    device=device,
                    stride=args.stride,
                    short_side=args.short_side,
                    loss_fn=loss_fn,
                    reuse_loss_fn=True,
                )
                row = {
                    "dataset": dataset,
                    "clip": stem,
                    "method": method,
                    "frames": str(int(vals.get("frames", 0))),
                    "psnr": format_metric(vals.get("psnr")),
                    "ssim": format_metric(vals.get("ssim")),
                    "lpips": format_metric(vals.get("lpips")),
                    "tlpips": format_metric(vals.get("tlpips", vals.get("tlpips_noref"))),
                    "fid": format_metric(vals.get("fid")),
                    "temporal_mae": format_metric(vals.get("temporal_mae")),
                    "has_gt": "true" if gt_a is not None else "false",
                    "pred_path": str(pred.relative_to(_CV_ROOT)),
                    "gt_path": str(gt_path.relative_to(_CV_ROOT)) if gt_path and gt_path.is_file() else "",
                    "eval_policy": "unified_streaming_vsr_crop_minT",
                    "crop_twh": f"{t_w}x{t_h}",
                }
                per_clip_rows.append(row)
    else:
        for stem, dataset in collect_stems(inputs_dir, wild_lr):
            gt_path: Path | None
            if dataset == "wild":
                gt_path = wild_gt_path if wild_gt_path.is_file() else None
            else:
                cand = gt_dir / f"{stem}.mp4"
                gt_path = cand if cand.is_file() else None

            for method, rel_dir, suffix in VARIANTS:
                pred = resolve_pred_with_legacy(method, rel_dir, suffix, stem)
                if pred is None:
                    continue

                gt_frames = load_sequence(gt_path) if gt_path and gt_path.is_file() else None
                pred_frames = load_sequence(pred)
                mset = set(metrics)
                if gt_frames is None:
                    mset = {"tlpips_noref"}
                vals = evaluate_sequence(
                    pred_frames,
                    gt_frames,
                    metrics=mset,
                    device=device,
                    stride=args.stride,
                    short_side=args.short_side,
                    loss_fn=loss_fn,
                    reuse_loss_fn=True,
                )
                row = {
                    "dataset": dataset,
                    "clip": stem,
                    "method": method,
                    "frames": str(int(vals.get("frames", 0))),
                    "psnr": format_metric(vals.get("psnr")),
                    "ssim": format_metric(vals.get("ssim")),
                    "lpips": format_metric(vals.get("lpips")),
                    "tlpips": format_metric(vals.get("tlpips", vals.get("tlpips_noref"))),
                    "fid": format_metric(vals.get("fid")),
                    "temporal_mae": format_metric(vals.get("temporal_mae")),
                    "has_gt": "true" if gt_frames is not None else "false",
                    "pred_path": str(pred.relative_to(_CV_ROOT)),
                    "gt_path": str(gt_path.relative_to(_CV_ROOT)) if gt_path and gt_path.is_file() else "",
                    "eval_policy": "legacy",
                    "crop_twh": "",
                }
                per_clip_rows.append(row)

    fieldnames = [
        "dataset",
        "clip",
        "method",
        "frames",
        "psnr",
        "ssim",
        "lpips",
        "tlpips",
        "fid",
        "temporal_mae",
        "has_gt",
        "pred_path",
        "gt_path",
        "eval_policy",
        "crop_twh",
    ]
    out_pc = _CV_ROOT / args.per_clip_csv
    out_pc.parent.mkdir(parents=True, exist_ok=True)
    with out_pc.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(per_clip_rows)
    print(f"Wrote {out_pc} ({len(per_clip_rows)} rows)")

    summary_rows = []
    for method, _rel, _suf in VARIANTS:
        psnrs = [float(r["psnr"]) for r in per_clip_rows if r["method"] == method and r["psnr"]]
        summary_rows.append(
            {
                "method": method,
                "clips_evaluated": str(len(psnrs)),
                "mean_psnr": f"{sum(psnrs) / len(psnrs):.4f}" if psnrs else "",
            }
        )
    out_s = _CV_ROOT / args.summary_csv
    with out_s.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["method", "clips_evaluated", "mean_psnr"])
        w.writeheader()
        w.writerows(summary_rows)
    print(f"Wrote {out_s}")


if __name__ == "__main__":
    main()
