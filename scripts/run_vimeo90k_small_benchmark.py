"""Run a small Vimeo-90K GT benchmark on 10 triplets."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from time import perf_counter

import cv2
import lpips
import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from part1_baseline.video_io import write_video
from part2_sota.infer_basicvsr_standalone import infer_chunks, load_basicvsr
from part2_sota.infer_realesrgan import build_realesrgan_upsampler, enhance_rgb_frames
from part3_exploration.direction_b_sd_controlnet.run_flow_stabilize import stabilize_frames
from scripts.metrics_core import psnr, ssim_rgb


def read_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def resize_frame(frame: np.ndarray, size: tuple[int, int], interpolation: int) -> np.ndarray:
    return cv2.resize(frame, size, interpolation=interpolation)


def make_lr_clip(hr_frames: list[np.ndarray], scale: int) -> list[np.ndarray]:
    h, w = hr_frames[0].shape[:2]
    return [resize_frame(f, (w // scale, h // scale), cv2.INTER_CUBIC) for f in hr_frames]


def bicubic_upscale(lr_frames: list[np.ndarray], hr_size: tuple[int, int]) -> list[np.ndarray]:
    return [resize_frame(f, hr_size, cv2.INTER_CUBIC) for f in lr_frames]


def bgr_to_rgb_frames(frames: list[np.ndarray]) -> list[np.ndarray]:
    return [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]


def rgb_to_bgr_frames(frames: list[np.ndarray]) -> list[np.ndarray]:
    return [cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in frames]


def lpips_tensor(frame: np.ndarray, device: torch.device) -> torch.Tensor:
    arr = frame.astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


def temporal_mae(frames: list[np.ndarray]) -> float:
    if len(frames) < 2:
        return 0.0
    return float(
        np.mean([
            np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32)))
            for a, b in zip(frames[:-1], frames[1:])
        ])
    )


def evaluate_clip(
    method: str,
    clip_id: str,
    pred_frames: list[np.ndarray],
    gt_frames: list[np.ndarray],
    loss_fn: torch.nn.Module,
    device: torch.device,
) -> dict[str, str]:
    psnr_vals: list[float] = []
    ssim_vals: list[float] = []
    lpips_vals: list[float] = []
    with torch.no_grad():
        for pred, gt in zip(pred_frames, gt_frames):
            if pred.shape[:2] != gt.shape[:2]:
                pred = resize_frame(pred, (gt.shape[1], gt.shape[0]), cv2.INTER_AREA)
            psnr_vals.append(psnr(pred, gt))
            ssim_vals.append(ssim_rgb(pred, gt))
            lpips_vals.append(float(loss_fn(lpips_tensor(pred, device), lpips_tensor(gt, device)).item()))
    return {
        "clip": clip_id,
        "method": method,
        "frames": str(min(len(pred_frames), len(gt_frames))),
        "psnr": f"{np.mean(psnr_vals):.4f}",
        "ssim": f"{np.mean(ssim_vals):.4f}",
        "lpips": f"{np.mean(lpips_vals):.4f}",
        "temporal_mae": f"{temporal_mae(pred_frames):.4f}",
    }


def collect_clips(root: Path, limit: int) -> list[str]:
    list_path = root / "tri_testlist.txt"
    clips = [line.strip().replace("\\", "/") for line in list_path.read_text().splitlines() if line.strip()]
    return clips[:limit]


def load_hr_triplet(root: Path, clip: str) -> list[np.ndarray]:
    input_dir = root / "input" / clip
    target_dir = root / "target" / clip
    return [read_rgb(input_dir / "im1.png"), read_rgb(target_dir / "im2.png"), read_rgb(input_dir / "im3.png")]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run small Vimeo-90K benchmark.")
    parser.add_argument("--root", default="dataset/vimeo_interp_test")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--fps", type=float, default=3.0)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--out_dir", default="output/vimeo90k_small")
    parser.add_argument("--table_dir", default="output/tables")
    parser.add_argument("--realesrgan_weight", default="weights/realesrgan/RealESRGAN_x4plus.pth")
    parser.add_argument("--basicvsr_weight", default="weights/basicvsr/basicvsr_vimeo90k_bi.pth")
    parser.add_argument("--tile", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    table_dir = Path(args.table_dir)
    table_dir.mkdir(parents=True, exist_ok=True)
    clips = collect_clips(root, args.limit)
    device = torch.device(args.device)

    loss_fn = lpips.LPIPS(net="alex").to(device).eval()
    realesrgan = build_realesrgan_upsampler(args.realesrgan_weight, tile=args.tile)
    basicvsr = load_basicvsr(args.basicvsr_weight, device)

    rows: list[dict[str, str]] = []
    manifest: list[dict[str, str]] = []
    start_all = perf_counter()

    for idx, clip in enumerate(clips, start=1):
        safe_id = clip.replace("/", "_")
        clip_dir = out_dir / safe_id
        clip_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{idx}/{len(clips)}] {clip}")

        hr = load_hr_triplet(root, clip)
        h, w = hr[0].shape[:2]
        hr_size = (w, h)
        lr = make_lr_clip(hr, args.scale)
        bicubic = bicubic_upscale(lr, hr_size)
        real = enhance_rgb_frames(realesrgan, lr, desc=None)
        basic = bgr_to_rgb_frames(
            infer_chunks(basicvsr, rgb_to_bgr_frames(lr), device, chunk_size=3, overlap=0)
        )
        flow = bgr_to_rgb_frames(
            stabilize_frames(
                rgb_to_bgr_frames(lr),
                rgb_to_bgr_frames(real),
                scale=float(args.scale),
                max_alpha=0.28,
                error_sigma=18.0,
                motion_sigma=10.0,
                recursive=False,
            )
        )

        videos = {
            "gt": hr,
            "lr_x4_bicubic": bicubic,
            "realesrgan": real,
            "basicvsr": basic,
            "flow_stabilized": flow,
        }
        for name, frames in videos.items():
            path = clip_dir / f"{name}.mp4"
            write_video(str(path), frames, args.fps)
            manifest.append({"clip": clip, "name": name, "video": str(path)})

        for method, frames in [
            ("Bicubic", bicubic),
            ("Real-ESRGAN", real),
            ("BasicVSR", basic),
            ("Flow-Stabilized", flow),
        ]:
            rows.append(evaluate_clip(method, clip, frames, hr, loss_fn, device))

    metrics_csv = table_dir / "vimeo90k_small_metrics.csv"
    with metrics_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["clip", "method", "frames", "psnr", "ssim", "lpips", "temporal_mae"])
        writer.writeheader()
        writer.writerows(rows)

    summary: list[dict[str, str]] = []
    for method in ["Bicubic", "Real-ESRGAN", "BasicVSR", "Flow-Stabilized"]:
        method_rows = [r for r in rows if r["method"] == method]
        summary.append(
            {
                "method": method,
                "clips": str(len(method_rows)),
                "psnr": f"{np.mean([float(r['psnr']) for r in method_rows]):.4f}",
                "ssim": f"{np.mean([float(r['ssim']) for r in method_rows]):.4f}",
                "lpips": f"{np.mean([float(r['lpips']) for r in method_rows]):.4f}",
                "temporal_mae": f"{np.mean([float(r['temporal_mae']) for r in method_rows]):.4f}",
            }
        )

    summary_csv = table_dir / "vimeo90k_small_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "clips", "psnr", "ssim", "lpips", "temporal_mae"])
        writer.writeheader()
        writer.writerows(summary)

    summary_md = table_dir / "vimeo90k_small_summary.md"
    lines = [
        "| Method | Clips | PSNR (higher) | SSIM (higher) | LPIPS (lower) | Temporal MAE (lower) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in summary:
        lines.append(f"| {r['method']} | {r['clips']} | {r['psnr']} | {r['ssim']} | {r['lpips']} | {r['temporal_mae']} |")
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    manifest_csv = table_dir / "vimeo90k_small_manifest.csv"
    with manifest_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["clip", "name", "video"])
        writer.writeheader()
        writer.writerows(manifest)

    print(summary_md.read_text(encoding="utf-8"))
    print(f"Wrote {metrics_csv}, {summary_csv}, {summary_md}, {manifest_csv}")
    print(f"Total time: {perf_counter() - start_all:.1f}s")


if __name__ == "__main__":
    main()
