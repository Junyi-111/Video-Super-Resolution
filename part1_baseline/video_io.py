"""Read/write video with imageio; frame iteration helpers."""

from __future__ import annotations

import os
from typing import Generator, Iterator, Tuple

import imageio.v2 as imageio
import numpy as np
import torch


def read_video(path: str) -> Tuple[list[np.ndarray], float]:
    """Returns list of HxWx3 uint8 RGB frames and fps (default 24 if unknown)."""
    r = imageio.get_reader(path, "ffmpeg")
    meta = {}
    try:
        meta = r.get_meta_data()
    except Exception:
        pass
    fps = float(meta.get("fps", 24.0))
    frames: list[np.ndarray] = []
    for frame in r:
        if frame.ndim == 2:
            frame = np.stack([frame] * 3, axis=-1)
        if frame.shape[-1] == 4:
            frame = frame[..., :3]
        frames.append(np.ascontiguousarray(frame))
    r.close()
    return frames, fps


def write_video(path: str, frames: list[np.ndarray] | Iterator[np.ndarray], fps: float) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    w = imageio.get_writer(
        path,
        fps=fps,
        codec="libx264",
        ffmpeg_params=["-pix_fmt", "yuv420p"],
        macro_block_size=16,
    )
    for f in frames:
        if f.dtype != np.uint8:
            f = np.clip(np.round(f), 0, 255).astype(np.uint8)
        w.append_data(f)
    w.close()


def frames_to_tensor_batch(frames: list[np.ndarray], device: torch.device) -> torch.Tensor:
    """N x 3 x H x W float [0,1]."""
    arr = np.stack([f.astype(np.float32) / 255.0 for f in frames], axis=0)
    t = torch.from_numpy(arr).permute(0, 3, 1, 2).to(device)
    return t


def tensor_to_uint8_frame(t: torch.Tensor) -> np.ndarray:
    """1 x 3 x H x W -> H W 3 uint8."""
    x = t.squeeze(0).detach().cpu().clamp(0, 1).numpy().transpose(1, 2, 0)
    return (x * 255.0).round().astype(np.uint8)


def iter_sliding_window(length: int, radius: int) -> Generator[Tuple[list[int], int], None, None]:
    """For each center index, yields (neighbor_indices_inclusive, center)."""
    for c in range(length):
        idx = [max(0, min(length - 1, c + d)) for d in range(-radius, radius + 1)]
        yield idx, c
