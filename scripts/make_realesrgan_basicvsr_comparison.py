"""Create a focused Wild comparison figure for Real-ESRGAN vs BasicVSR."""

from __future__ import annotations

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
    return cv2.resize(frame, (w, h), interpolation=cv2.INTER_CUBIC)


def center_crop(frame: np.ndarray, cx: float, cy: float, size: int) -> np.ndarray:
    h, w = frame.shape[:2]
    half = size // 2
    x = int(round(cx * w))
    y = int(round(cy * h))
    left = max(0, min(w - size, x - half))
    top = max(0, min(h - size, y - half))
    return frame[top : top + size, left : left + size]


def add_label(frame: np.ndarray, label: str) -> np.ndarray:
    out = frame.copy()
    pad = 12
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.75
    thickness = 2
    (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
    cv2.rectangle(out, (0, 0), (tw + pad * 2, th + pad * 2), (0, 0, 0), -1)
    cv2.putText(out, label, (pad, th + pad), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return out


def hstack(frames: list[np.ndarray], gap: int = 12) -> np.ndarray:
    h = max(frame.shape[0] for frame in frames)
    padded = []
    for frame in frames:
        pad = h - frame.shape[0]
        padded.append(cv2.copyMakeBorder(frame, 0, pad, 0, 0, cv2.BORDER_CONSTANT, value=(245, 245, 245)))
    spacer = np.full((h, gap, 3), 245, dtype=np.uint8)
    row = padded[0]
    for frame in padded[1:]:
        row = np.hstack([row, spacer, frame])
    return row


def main() -> None:
    frame_index = 80
    zoom_size = 520
    zoom_x = 0.50
    zoom_y = 0.50
    out_dir = Path("output/figures/wild_realesrgan_basicvsr")
    out_dir.mkdir(parents=True, exist_ok=True)

    videos = [
        ("GT", "dataset/wild_real.mp4"),
        ("LR input", "dataset/wild_real_lr.mp4"),
        ("Real-ESRGAN", "part2_sota/outputs/wild_real_lr/realesrgan_x4.mp4"),
        ("BasicVSR", "part2_sota/outputs/basicvsr/wild_real_lr.mp4"),
    ]
    gt = read_frame(videos[0][1], frame_index)
    panels = []
    zooms = []
    for label, path in videos:
        frame = resize_like(read_frame(path, frame_index), gt)
        preview = cv2.resize(frame, (360, 640), interpolation=cv2.INTER_AREA)
        panels.append(add_label(preview, label))
        zooms.append(add_label(center_crop(frame, zoom_x, zoom_y, zoom_size), label))

    full_path = out_dir / f"wild_frame_{frame_index:04d}_full.png"
    zoom_path = out_dir / f"wild_frame_{frame_index:04d}_zoom.png"
    cv2.imwrite(str(full_path), hstack(panels))
    cv2.imwrite(str(zoom_path), hstack(zooms))
    print(f"Wrote {full_path}")
    print(f"Wrote {zoom_path}")


if __name__ == "__main__":
    main()
