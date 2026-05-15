"""Temporal baseline vs Real-ESRGAN blend (Direction B lightweight consistency)."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from part3_exploration.common_video import read_video, write_video


def texture_mask(frame_bgr: np.ndarray, blur_sigma: float, edge_strength: float) -> np.ndarray:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    mag = np.abs(lap)
    mag = cv2.GaussianBlur(mag, (0, 0), blur_sigma)
    p90 = np.percentile(mag, 90)
    scale = max(p90, 1.0)
    mask = np.clip((mag / scale) * edge_strength, 0.0, 1.0)
    mask = cv2.GaussianBlur(mask, (0, 0), blur_sigma)
    return mask[..., None].astype(np.float32)


def fuse_frame(
    temporal: np.ndarray,
    realesrgan: np.ndarray,
    real_min: float,
    real_max: float,
    blur_sigma: float,
    edge_strength: float,
) -> np.ndarray:
    if temporal.shape != realesrgan.shape:
        realesrgan = cv2.resize(realesrgan, (temporal.shape[1], temporal.shape[0]), interpolation=cv2.INTER_AREA)
    mask = texture_mask(realesrgan, blur_sigma=blur_sigma, edge_strength=edge_strength)
    alpha = real_min + (real_max - real_min) * mask
    fused = temporal.astype(np.float32) * (1.0 - alpha) + realesrgan.astype(np.float32) * alpha
    return np.clip(fused, 0, 255).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Temporal + Real-ESRGAN hybrid fusion.")
    parser.add_argument("--temporal", required=True)
    parser.add_argument("--realesrgan", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--real_min", type=float, default=0.35)
    parser.add_argument("--real_max", type=float, default=0.82)
    parser.add_argument("--blur_sigma", type=float, default=3.0)
    parser.add_argument("--edge_strength", type=float, default=1.25)
    args = parser.parse_args()

    t_frames, fps = read_video(args.temporal)
    r_frames, _ = read_video(args.realesrgan)
    n = min(len(t_frames), len(r_frames))
    out = [
        fuse_frame(t_frames[i], r_frames[i], args.real_min, args.real_max, args.blur_sigma, args.edge_strength)
        for i in range(n)
    ]
    write_video(args.out, out, fps)
    print(f"Wrote {args.out} ({len(out)} frames)")


if __name__ == "__main__":
    main()
