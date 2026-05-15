"""Evaluate diffusion-tile keyframe experiments against pseudo-GT frames."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import lpips
import numpy as np
import torch
from PIL import Image

import sys

sys.path.append(str(Path(__file__).resolve().parents[0] / "scripts"))
from metrics_core import psnr, ssim_rgb


def read_video_frame(path: str, frame_index: int) -> np.ndarray:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_index} from {path}")
    return frame


def read_image_bgr(path: Path) -> np.ndarray:
    rgb = np.array(Image.open(path).convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def to_lpips_tensor(frame_bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    arr = rgb.astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


def evaluate_pair(loss_fn: torch.nn.Module, pred: np.ndarray, gt: np.ndarray, device: torch.device) -> dict[str, float]:
    gt_r = cv2.resize(gt, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_AREA)
    with torch.no_grad():
        lp = float(loss_fn(to_lpips_tensor(pred, device), to_lpips_tensor(gt_r, device)).item())
    return {
        "psnr": psnr(pred, gt_r),
        "ssim": ssim_rgb(pred, gt_r),
        "lpips": lp,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate diffusion keyframes.")
    parser.add_argument("--gt_video", default="dataset/wild_real.mp4")
    parser.add_argument("--realesrgan_video", default="part2_sota/outputs/wild_real_lr/realesrgan_x4.mp4")
    parser.add_argument("--out_csv", default="output/tables/diffusion_keyframe_metrics.csv")
    parser.add_argument("--frames", nargs="*", type=int, default=[80, 120, 160])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    methods = [
        ("Real-ESRGAN input", None),
        ("Diffusion conservative", Path("part3_exploration/outputs/diffusion_tile_conservative")),
        ("Diffusion over-strong", Path("part3_exploration/outputs/diffusion_tile_report")),
    ]

    device = torch.device(args.device)
    loss_fn = lpips.LPIPS(net="alex").to(device).eval()

    rows = []
    for frame_index in args.frames:
        gt = read_video_frame(args.gt_video, frame_index)
        real = read_video_frame(args.realesrgan_video, frame_index)

        for method, directory in methods:
            if directory is None:
                pred = cv2.resize(real, (432, 768), interpolation=cv2.INTER_AREA)
            else:
                pred = read_image_bgr(directory / f"frame_{frame_index:04d}_diffusion_tile.png")

            vals = evaluate_pair(loss_fn, pred, gt, device)
            rows.append(
                {
                    "frame": frame_index,
                    "method": method,
                    "psnr": f"{vals['psnr']:.4f}",
                    "ssim": f"{vals['ssim']:.4f}",
                    "lpips": f"{vals['lpips']:.4f}",
                }
            )
            print(
                f"frame {frame_index} {method}: "
                f"PSNR={vals['psnr']:.4f}, SSIM={vals['ssim']:.4f}, LPIPS={vals['lpips']:.4f}"
            )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["frame", "method", "psnr", "ssim", "lpips"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
