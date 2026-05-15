"""Build qualitative comparison figures from aligned SR videos."""

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


def resize_to_height(frame: np.ndarray, height: int) -> np.ndarray:
    h, w = frame.shape[:2]
    width = max(1, round(w * height / h))
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def center_crop(frame: np.ndarray, cx: float, cy: float, size: int) -> np.ndarray:
    h, w = frame.shape[:2]
    x = int(round(cx * w))
    y = int(round(cy * h))
    half = size // 2
    left = max(0, min(w - size, x - half))
    top = max(0, min(h - size, y - half))
    return frame[top : top + size, left : left + size]


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


def pad_to_same_height(frames: list[np.ndarray], color: int = 245) -> list[np.ndarray]:
    max_h = max(f.shape[0] for f in frames)
    out = []
    for f in frames:
        if f.shape[0] == max_h:
            out.append(f)
            continue
        pad_h = max_h - f.shape[0]
        out.append(cv2.copyMakeBorder(f, 0, pad_h, 0, 0, cv2.BORDER_CONSTANT, value=(color, color, color)))
    return out


def hstack_with_gap(frames: list[np.ndarray], gap: int = 12) -> np.ndarray:
    frames = pad_to_same_height(frames)
    spacer = np.full((frames[0].shape[0], gap, 3), 245, dtype=np.uint8)
    row = frames[0]
    for f in frames[1:]:
        row = np.hstack([row, spacer, f])
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Create qualitative SR comparison figures.")
    parser.add_argument("--frame", type=int, default=80)
    parser.add_argument("--out_dir", default="output/figures/wild_real_lr")
    parser.add_argument("--thumb_height", type=int, default=700)
    parser.add_argument("--zoom_size", type=int, default=420)
    parser.add_argument("--zoom_x", type=float, default=0.50, help="Relative x center in [0,1].")
    parser.add_argument("--zoom_y", type=float, default=0.50, help="Relative y center in [0,1].")
    args = parser.parse_args()

    videos = [
        ("LR input", "dataset/wild_real_lr.mp4"),
        ("Bicubic", "part1_baseline/outputs/wild_real_lr/wild_real_lr_part1_bicubic_x4.mp4"),
        ("Lanczos", "part1_baseline/outputs/wild_real_lr/wild_real_lr_part1_lanczos_x4.mp4"),
        (
            "Temporal+Unsharp",
            "part1_baseline/outputs/wild_real_lr/wild_real_lr_part1_temporal_w5_unsharp_x4.mp4",
        ),
        ("Real-ESRGAN", "part2_sota/outputs/wild_real_lr/realesrgan_x4.mp4"),
        ("Part3 Hybrid", "part3_exploration/outputs/wild_real_lr_hybrid_x4.mp4"),
        ("Flow-Stabilized", "part3_exploration/outputs/wild_real_lr_flow_stabilized_x4.mp4"),
    ]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = [(label, read_frame(path, args.frame)) for label, path in videos]
    target_h, target_w = frames[-1][1].shape[:2]

    full_panels = []
    zoom_panels = []
    for label, frame in frames:
        aligned = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        full_panels.append(add_label(resize_to_height(aligned, args.thumb_height), label))
        crop = center_crop(aligned, args.zoom_x, args.zoom_y, args.zoom_size)
        zoom_panels.append(add_label(crop, label))

    full = hstack_with_gap(full_panels)
    zoom = hstack_with_gap(zoom_panels)

    full_path = out_dir / f"wild_frame_{args.frame:04d}_full.png"
    zoom_path = out_dir / f"wild_frame_{args.frame:04d}_zoom.png"
    cv2.imwrite(str(full_path), full)
    cv2.imwrite(str(zoom_path), zoom)
    print(f"Wrote {full_path}")
    print(f"Wrote {zoom_path}")


if __name__ == "__main__":
    main()
