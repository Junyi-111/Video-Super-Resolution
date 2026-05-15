"""Direction A: few-step rectified refinement from bicubic HR toward Real-ESRGAN HR."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from part3_exploration.common_video import read_video, write_video


def refine_frames(bicubic: list[np.ndarray], generative: list[np.ndarray], steps: int) -> list[np.ndarray]:
    n = min(len(bicubic), len(generative))
    out: list[np.ndarray] = []
    k = max(1, steps)
    for i in range(n):
        b = bicubic[i].astype(np.float32)
        g = generative[i].astype(np.float32)
        if b.shape != g.shape:
            g = cv2.resize(g, (b.shape[1], b.shape[0]), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        z = b.copy()
        for _ in range(k):
            z = z + (1.0 / k) * (g - z)
        out.append(np.clip(z, 0, 255).astype(np.uint8))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Rectified-flow-style few-step HR refinement.")
    parser.add_argument("--bicubic", required=True, help="Bicubic 4x HR video (Part1 output).")
    parser.add_argument("--realesrgan", required=True, help="Real-ESRGAN 4x HR video.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=4, help="Euler substeps K (>=1).")
    args = parser.parse_args()

    bic, fps = read_video(args.bicubic)
    gen, _ = read_video(args.realesrgan)
    refined = refine_frames(bic, gen, args.steps)
    write_video(args.out, refined, fps)
    print(f"Wrote {args.out} ({len(refined)} frames, steps={args.steps})")


if __name__ == "__main__":
    main()
