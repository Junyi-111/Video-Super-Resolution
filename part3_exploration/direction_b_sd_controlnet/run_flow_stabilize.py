"""Optical-flow temporal stabilization (Direction B post-processing)."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from part3_exploration.common_video import read_video, write_video


def farneback_flow(prev_bgr: np.ndarray, curr_bgr: np.ndarray) -> np.ndarray:
    prev = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
    curr = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.calcOpticalFlowFarneback(
        curr, prev, None, pyr_scale=0.5, levels=4, winsize=25, iterations=4, poly_n=7, poly_sigma=1.5, flags=0
    )


def upscale_flow(flow_lr: np.ndarray, hr_size: tuple[int, int], scale: float) -> np.ndarray:
    hr_w, hr_h = hr_size
    flow_hr = cv2.resize(flow_lr, (hr_w, hr_h), interpolation=cv2.INTER_LINEAR)
    flow_hr *= scale
    return flow_hr


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


def reliability_alpha(
    current: np.ndarray,
    warped_prev: np.ndarray,
    flow_hr: np.ndarray,
    max_alpha: float,
    error_sigma: float,
    motion_sigma: float,
) -> np.ndarray:
    diff = np.mean(np.abs(current.astype(np.float32) - warped_prev.astype(np.float32)), axis=2)
    motion = np.linalg.norm(flow_hr.astype(np.float32), axis=2)
    err_weight = np.exp(-diff / max(error_sigma, 1e-6))
    motion_weight = np.exp(-motion / max(motion_sigma, 1e-6))
    alpha = max_alpha * err_weight * motion_weight
    alpha = cv2.GaussianBlur(alpha, (0, 0), 3.0)
    return np.clip(alpha[..., None], 0.0, max_alpha).astype(np.float32)


def stabilize_frames(
    lr_frames: list[np.ndarray],
    real_frames: list[np.ndarray],
    scale: float,
    max_alpha: float,
    error_sigma: float,
    motion_sigma: float,
    recursive: bool,
) -> list[np.ndarray]:
    n = min(len(lr_frames), len(real_frames))
    out = [real_frames[0]]
    hr_h, hr_w = real_frames[0].shape[:2]
    for i in range(1, n):
        flow_lr = farneback_flow(lr_frames[i - 1], lr_frames[i])
        flow_hr = upscale_flow(flow_lr, (hr_w, hr_h), scale)
        prev_source = out[-1] if recursive else real_frames[i - 1]
        warped_prev = warp_frame(prev_source, flow_hr)
        alpha = reliability_alpha(
            real_frames[i], warped_prev, flow_hr, max_alpha, error_sigma, motion_sigma
        )
        fused = real_frames[i].astype(np.float32) * (1.0 - alpha) + warped_prev.astype(np.float32) * alpha
        out.append(np.clip(fused, 0, 255).astype(np.uint8))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Flow-guided temporal stabilization for Real-ESRGAN.")
    parser.add_argument("--lr", required=True)
    parser.add_argument("--realesrgan", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--scale", type=float, default=4.0)
    parser.add_argument("--max_alpha", type=float, default=0.28)
    parser.add_argument("--error_sigma", type=float, default=18.0)
    parser.add_argument("--motion_sigma", type=float, default=10.0)
    parser.add_argument("--recursive", action="store_true")
    args = parser.parse_args()

    lr_frames, fps = read_video(args.lr)
    real_frames, _ = read_video(args.realesrgan)
    out_frames = stabilize_frames(
        lr_frames,
        real_frames,
        scale=args.scale,
        max_alpha=args.max_alpha,
        error_sigma=args.error_sigma,
        motion_sigma=args.motion_sigma,
        recursive=args.recursive,
    )
    write_video(args.out, out_frames, fps)
    print(f"Wrote {args.out} ({len(out_frames)} frames)")


if __name__ == "__main__":
    main()
