"""Bicubic / Lanczos upsampling via OpenCV."""

from __future__ import annotations

import cv2
import numpy as np


def upscale_frame(frame: np.ndarray, scale: int, mode: str = "bicubic") -> np.ndarray:
    """frame: HxWx3 uint8 RGB. mode: bicubic | lanczos."""
    h, w = frame.shape[:2]
    nh, nw = h * scale, w * scale
    interp = cv2.INTER_CUBIC if mode == "bicubic" else cv2.INTER_LANCZOS4
    return cv2.resize(frame, (nw, nh), interpolation=interp)


def unsharp_mask(frame: np.ndarray, sigma: float = 1.0, amount: float = 1.0) -> np.ndarray:
    """Unsharp masking on float [0,255] in BGR-style per-channel blur for stability."""
    blur = cv2.GaussianBlur(frame, (0, 0), sigma)
    sharp = cv2.addWeighted(frame, 1.0 + amount, blur, -amount, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)
