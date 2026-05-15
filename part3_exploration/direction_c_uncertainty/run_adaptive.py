"""Direction C: uncertainty-aware adaptive fusion (single clip CLI)."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from part3_exploration.common_video import read_video, write_video


def normalize_percentile(x: np.ndarray, p_low: float = 5.0, p_high: float = 95.0) -> np.ndarray:
    lo = float(np.percentile(x, p_low))
    hi = float(np.percentile(x, p_high))
    if hi <= lo + 1e-6:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)


def texture_strength(frame: np.ndarray, sigma: float) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(grad_x * grad_x + grad_y * grad_y)
    score = 0.55 * normalize_percentile(lap, 50, 98) + 0.45 * normalize_percentile(grad, 50, 98)
    return cv2.GaussianBlur(score, (0, 0), sigma).astype(np.float32)


def branch_disagreement(real: np.ndarray, basic: np.ndarray, sigma: float) -> np.ndarray:
    diff = np.mean(np.abs(real.astype(np.float32) - basic.astype(np.float32)), axis=2)
    score = normalize_percentile(diff, 20, 97)
    return cv2.GaussianBlur(score, (0, 0), sigma).astype(np.float32)


def text_like_mask(frame: np.ndarray, sigma: float) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3)))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3)))
    contrast = np.maximum(blackhat, tophat).astype(np.float32)
    edges = cv2.Canny(gray, 80, 160).astype(np.float32) / 255.0
    score = normalize_percentile(contrast, 65, 99) * 0.65 + edges * 0.35
    score = cv2.dilate(score, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5)), iterations=1)
    return cv2.GaussianBlur(np.clip(score, 0.0, 1.0), (0, 0), sigma).astype(np.float32)


def load_face_detector() -> cv2.CascadeClassifier | None:
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    if not cascade_path.exists():
        return None
    detector = cv2.CascadeClassifier(str(cascade_path))
    return detector if not detector.empty() else None


def face_mask(frame: np.ndarray, detector: cv2.CascadeClassifier | None, sigma: float) -> np.ndarray:
    mask = np.zeros(frame.shape[:2], dtype=np.float32)
    if detector is None:
        return mask
    h, w = frame.shape[:2]
    small_w = 480
    scale = small_w / max(w, 1)
    small_h = max(1, round(h * scale))
    small = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(24, 24))
    for x, y, fw, fh in faces:
        x0 = max(0, int((x - 0.15 * fw) / scale))
        y0 = max(0, int((y - 0.20 * fh) / scale))
        x1 = min(w, int((x + 1.15 * fw) / scale))
        y1 = min(h, int((y + 1.25 * fh) / scale))
        mask[y0:y1, x0:x1] = 1.0
    return cv2.GaussianBlur(mask, (0, 0), sigma).astype(np.float32)


def farneback_flow(prev_bgr: np.ndarray, curr_bgr: np.ndarray) -> np.ndarray:
    prev = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
    curr = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.calcOpticalFlowFarneback(
        curr, prev, None, pyr_scale=0.5, levels=4, winsize=25, iterations=4, poly_n=7, poly_sigma=1.5, flags=0
    )


def warp_frame(frame: np.ndarray, flow: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(
        frame,
        grid_x + flow[..., 0].astype(np.float32),
        grid_y + flow[..., 1].astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )


def temporal_artifact_score(
    prev_real: np.ndarray | None, curr_real: np.ndarray, prev_lr: np.ndarray | None, curr_lr: np.ndarray
) -> np.ndarray:
    if prev_real is None or prev_lr is None:
        return np.zeros(curr_real.shape[:2], dtype=np.float32)
    flow_lr = farneback_flow(prev_lr, curr_lr)
    hr_h, hr_w = curr_real.shape[:2]
    flow_hr = cv2.resize(flow_lr, (hr_w, hr_h), interpolation=cv2.INTER_LINEAR) * 4.0
    warped_prev = warp_frame(prev_real, flow_hr)
    err = np.mean(np.abs(curr_real.astype(np.float32) - warped_prev.astype(np.float32)), axis=2)
    return cv2.GaussianBlur(normalize_percentile(err, 35, 98), (0, 0), 3.0).astype(np.float32)


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    u8 = np.clip(mask * 255.0, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)


def fuse_adaptive(
    lr_frames: list[np.ndarray],
    real_frames: list[np.ndarray],
    basic_frames: list[np.ndarray],
    *,
    real_min: float,
    real_max: float,
    texture_weight: float,
    protect_weight: float,
    artifact_weight: float,
    mask_sigma: float,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    n = min(len(lr_frames), len(real_frames), len(basic_frames))
    detector = load_face_detector()
    fused_frames: list[np.ndarray] = []
    mask_frames: list[np.ndarray] = []

    prev_real: np.ndarray | None = None
    prev_lr: np.ndarray | None = None
    for i in range(n):
        real = real_frames[i]
        basic = basic_frames[i]
        if basic.shape != real.shape:
            basic = cv2.resize(basic, (real.shape[1], real.shape[0]), interpolation=cv2.INTER_AREA)

        texture = texture_strength(real, mask_sigma)
        disagree = branch_disagreement(real, basic, mask_sigma)
        protected = np.maximum(text_like_mask(real, mask_sigma), face_mask(real, detector, mask_sigma))
        artifact = temporal_artifact_score(prev_real, real, prev_lr, lr_frames[i])

        texture_uncertainty = np.clip(0.55 * texture + 0.45 * disagree, 0.0, 1.0)
        alpha = (
            real_min + texture_weight * texture_uncertainty - protect_weight * protected - artifact_weight * artifact
        )
        alpha = cv2.GaussianBlur(np.clip(alpha, real_min, real_max), (0, 0), mask_sigma)
        alpha3 = alpha[..., None].astype(np.float32)

        fused = basic.astype(np.float32) * (1.0 - alpha3) + real.astype(np.float32) * alpha3
        fused_frames.append(np.clip(fused, 0, 255).astype(np.uint8))
        mask_frames.append(colorize_mask(alpha))
        prev_real = real
        prev_lr = lr_frames[i]
    return fused_frames, mask_frames


def main() -> None:
    parser = argparse.ArgumentParser(description="Direction C adaptive hybrid.")
    parser.add_argument("--lr", required=True)
    parser.add_argument("--realesrgan", required=True)
    parser.add_argument("--basicvsr", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--mask_out", required=True)
    parser.add_argument("--real_min", type=float, default=0.82)
    parser.add_argument("--real_max", type=float, default=0.98)
    parser.add_argument("--texture_weight", type=float, default=0.18)
    parser.add_argument("--protect_weight", type=float, default=0.18)
    parser.add_argument("--artifact_weight", type=float, default=0.08)
    parser.add_argument("--mask_sigma", type=float, default=2.5)
    args = parser.parse_args()

    lr_frames, fps = read_video(args.lr)
    real_frames, _ = read_video(args.realesrgan)
    basic_frames, _ = read_video(args.basicvsr)
    fused, masks = fuse_adaptive(
        lr_frames,
        real_frames,
        basic_frames,
        real_min=args.real_min,
        real_max=args.real_max,
        texture_weight=args.texture_weight,
        protect_weight=args.protect_weight,
        artifact_weight=args.artifact_weight,
        mask_sigma=args.mask_sigma,
    )
    write_video(args.out, fused, fps)
    write_video(args.mask_out, masks, fps)
    print(f"Wrote {args.out} ({len(fused)} frames)")
    print(f"Wrote {args.mask_out} ({len(masks)} frames)")


if __name__ == "__main__":
    main()
