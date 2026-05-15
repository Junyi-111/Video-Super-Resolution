"""Create a report panel for Direction C uncertainty-aware fusion."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def read_frame(path: str, index: int) -> np.ndarray:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {index} from {path}")
    return frame


def resize_like(frame: np.ndarray, ref: np.ndarray) -> np.ndarray:
    h, w = ref.shape[:2]
    if frame.shape[:2] == (h, w):
        return frame
    return cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)


def add_label(frame: np.ndarray, label: str) -> np.ndarray:
    out = frame.copy()
    pad = 12
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    thickness = 2
    (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
    cv2.rectangle(out, (0, 0), (tw + pad * 2, th + pad * 2), (0, 0, 0), -1)
    cv2.putText(out, label, (pad, th + pad), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return out


def center_crop(frame: np.ndarray, cx: float, cy: float, size: int) -> np.ndarray:
    h, w = frame.shape[:2]
    half = size // 2
    x = int(round(cx * w))
    y = int(round(cy * h))
    left = max(0, min(w - size, x - half))
    top = max(0, min(h - size, y - half))
    return frame[top : top + size, left : left + size]


def hstack(frames: list[np.ndarray], gap: int = 12) -> np.ndarray:
    h = max(f.shape[0] for f in frames)
    padded = []
    for frame in frames:
        padded.append(cv2.copyMakeBorder(frame, 0, h - frame.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(245, 245, 245)))
    spacer = np.full((h, gap, 3), 245, dtype=np.uint8)
    row = padded[0]
    for frame in padded[1:]:
        row = np.hstack([row, spacer, frame])
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Make Direction C uncertainty panel.")
    parser.add_argument("--frame", type=int, default=80)
    parser.add_argument("--out_dir", default="output/figures/wild_uncertainty_adaptive")
    parser.add_argument("--zoom_size", type=int, default=520)
    parser.add_argument("--zoom_x", type=float, default=0.50)
    parser.add_argument("--zoom_y", type=float, default=0.50)
    args = parser.parse_args()

    videos = [
        ("GT", "dataset/wild_real.mp4"),
        ("Real-ESRGAN", "part2_sota/outputs/wild_real_lr/realesrgan_x4.mp4"),
        ("BasicVSR", "part2_sota/outputs/basicvsr/wild_real_lr.mp4"),
        ("Uncertainty-Adaptive", "part3_exploration/outputs/wild_real_lr_uncertainty_adaptive_x4.mp4"),
        ("Alpha map", "part3_exploration/outputs/wild_real_lr_uncertainty_alpha.mp4"),
    ]

    gt = read_frame(videos[0][1], args.frame)
    full_panels = []
    zoom_panels = []
    for label, path in videos:
        frame = resize_like(read_frame(path, args.frame), gt)
        full_panels.append(add_label(cv2.resize(frame, (360, 640), interpolation=cv2.INTER_AREA), label))
        zoom_panels.append(add_label(center_crop(frame, args.zoom_x, args.zoom_y, args.zoom_size), label))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    full_path = out_dir / f"wild_frame_{args.frame:04d}_full.png"
    zoom_path = out_dir / f"wild_frame_{args.frame:04d}_zoom.png"
    cv2.imwrite(str(full_path), hstack(full_panels))
    cv2.imwrite(str(zoom_path), hstack(zoom_panels))
    print(f"Wrote {full_path}")
    print(f"Wrote {zoom_path}")


if __name__ == "__main__":
    main()
