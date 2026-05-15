"""Create a synthetic low-quality wild video from a normal MP4.

The course requires a real-world wild video. When a legacy low-resolution
camera is unavailable, this script degrades a casually captured video with
downsampling, blur, darkening, sensor-like noise, and low-bitrate encoding.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def _even(value: int) -> int:
    return max(2, value - (value % 2))


def degrade_frame(
    frame: np.ndarray,
    lr_width: int,
    darken: float,
    blur_sigma: float,
    noise_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    h, w = frame.shape[:2]
    lr_w = min(_even(lr_width), _even(w))
    lr_h = _even(round(h * lr_w / w))

    small = cv2.resize(frame, (lr_w, lr_h), interpolation=cv2.INTER_AREA)
    if blur_sigma > 0:
        small = cv2.GaussianBlur(small, (0, 0), blur_sigma)

    degraded = small.astype(np.float32) * darken
    if noise_std > 0:
        degraded += rng.normal(0.0, noise_std, degraded.shape).astype(np.float32)

    return np.clip(degraded, 0, 255).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synthesize a mandatory Wild LR video from a normal MP4."
    )
    parser.add_argument("--input", required=True, help="Path to a normal source MP4.")
    parser.add_argument(
        "--out",
        default="part2_sota/inputs_mp4/wild_real_lr.mp4",
        help="Output low-resolution MP4 path.",
    )
    parser.add_argument("--lr_width", type=int, default=320, help="Output LR width.")
    parser.add_argument("--darken", type=float, default=0.65, help="Brightness multiplier.")
    parser.add_argument("--blur_sigma", type=float, default=0.7, help="Gaussian blur sigma.")
    parser.add_argument("--noise_std", type=float, default=6.0, help="Gaussian noise std.")
    parser.add_argument("--max_frames", type=int, default=240, help="Optional frame cap.")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    inp = Path(args.input)
    if not inp.is_file():
        raise FileNotFoundError(f"Input video not found: {inp}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(inp))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {inp}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1e-3:
        fps = 24.0

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    lr_w = min(_even(args.lr_width), _even(src_w))
    lr_h = _even(round(src_h * lr_w / src_w))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out), fourcc, fps, (lr_w, lr_h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create output video: {out}")

    rng = np.random.default_rng(args.seed)
    count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(
            degrade_frame(
                frame,
                lr_width=args.lr_width,
                darken=args.darken,
                blur_sigma=args.blur_sigma,
                noise_std=args.noise_std,
                rng=rng,
            )
        )
        count += 1
        if args.max_frames > 0 and count >= args.max_frames:
            break

    cap.release()
    writer.release()
    print(f"Wrote {out} ({count} frames, {lr_w}x{lr_h}, {fps:.2f} fps)")


if __name__ == "__main__":
    main()
