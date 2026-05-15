"""Synthetic LR–HR pairs from HR frames (official PNG, GT mp4, or Part2 pseudo-HR)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from part1_baseline.clip_io import (
    default_dataset_dir,
    iter_reds_sample_clips,
    iter_vimeo_rl_subclips,
    list_reds_hr_frame_paths,
    repo_root,
)


@dataclass(frozen=True)
class HrSample:
    """One HR training frame: PNG on disk or a frame index inside an mp4."""

    kind: Literal["image", "video_frame"]
    path: Path
    frame_index: int = 0
    tag: str = ""  # e.g. official / wild_gt / part2_realesrgan


def collect_hr_paths(dataset_root: Path | None, source: str = "both") -> list[Path]:
    root = dataset_root or default_dataset_dir()
    paths: list[Path] = []
    if source in ("reds", "both"):
        for clip in iter_reds_sample_clips(root):
            paths.extend(list_reds_hr_frame_paths(clip))
    if source in ("vimeo", "both"):
        for sub in iter_vimeo_rl_subclips(root):

            def _im_index(p: Path) -> int:
                m = re.search(r"(\d+)", p.stem)
                return int(m.group(1)) if m else 0

            paths.extend(sorted(sub.glob("im*.png"), key=_im_index))
    return paths


def video_frame_count(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return 0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if n <= 0:
        n = 0
        while cap.read()[0]:
            n += 1
    cap.release()
    return n


def _frame_indices(n: int, max_frames: int | None, stride: int) -> list[int]:
    if n <= 0:
        return []
    if max_frames is not None and max_frames > 0 and n > max_frames:
        return [int(x) for x in np.linspace(0, n - 1, max_frames, dtype=int)]
    return list(range(0, n, max(stride, 1)))


def _samples_from_video(
    path: Path,
    *,
    tag: str,
    max_frames: int | None = None,
    stride: int = 1,
) -> list[HrSample]:
    n = video_frame_count(path)
    return [
        HrSample("video_frame", path, i, tag=tag)
        for i in _frame_indices(n, max_frames, stride)
    ]


def collect_hr_samples(
    dataset_root: Path | None = None,
    official_source: str = "both",
    hr_source: str = "auto",
    part2_realesrgan_dir: Path | None = None,
    max_frames_per_video: int | None = None,
    video_stride: int = 1,
) -> list[HrSample]:
    """
    Collect HR training samples.

    hr_source:
      - official: REDS/Vimeo sharp PNG only
      - videos: dataset/wild_real.mp4 + dataset/gt_mp4/*.mp4
      - part2: part2_sota/outputs/realesrgan/*.mp4 (pseudo-HR teacher)
      - all: merge all of the above
      - auto: official if any, else videos + part2 (matches course zip with only mp4)
    """
    root = dataset_root or default_dataset_dir()
    repo = repo_root()
    samples: list[HrSample] = []

    def add_official() -> None:
        for p in collect_hr_paths(root, official_source):
            samples.append(HrSample("image", p, tag="official"))

    def add_videos() -> None:
        wild = root / "wild_real.mp4"
        if wild.is_file():
            samples.extend(
                _samples_from_video(
                    wild,
                    tag="wild_gt",
                    max_frames=max_frames_per_video,
                    stride=video_stride,
                )
            )
        gt_dir = root / "gt_mp4"
        if gt_dir.is_dir():
            for v in sorted(gt_dir.glob("*.mp4")):
                samples.extend(
                    _samples_from_video(
                        v,
                        tag="gt_mp4",
                        max_frames=max_frames_per_video,
                        stride=video_stride,
                    )
                )

    def add_part2() -> None:
        rs_dir = part2_realesrgan_dir or (repo / "part2_sota" / "outputs" / "realesrgan")
        if not rs_dir.is_dir():
            return
        for v in sorted(rs_dir.glob("*.mp4")):
            samples.extend(
                _samples_from_video(
                    v,
                    tag="part2_realesrgan",
                    max_frames=max_frames_per_video,
                    stride=video_stride,
                )
            )

    mode = hr_source.lower()
    if mode == "official":
        add_official()
    elif mode == "videos":
        add_videos()
    elif mode == "part2":
        add_part2()
    elif mode == "all":
        add_official()
        add_videos()
        add_part2()
    elif mode == "auto":
        add_official()
        if not samples:
            add_videos()
            add_part2()
    else:
        raise ValueError(f"Unknown hr_source={hr_source!r}")

    return samples


def summarize_hr_samples(samples: list[HrSample]) -> str:
    counts: dict[str, int] = {}
    for s in samples:
        key = s.tag or s.kind
        counts[key] = counts.get(key, 0) + 1
    parts = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    return f"total={len(samples)} ({parts})" if parts else f"total={len(samples)}"


def read_hr_rgb(sample: HrSample) -> np.ndarray:
    if sample.kind == "image":
        return np.array(Image.open(sample.path).convert("RGB"))
    cap = cv2.VideoCapture(str(sample.path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {sample.path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(sample.frame_index))
    ok, bgr = cap.read()
    cap.release()
    if not ok or bgr is None:
        raise RuntimeError(f"Could not read frame {sample.frame_index} from {sample.path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def degrade_hr_to_lr(
    hr_rgb: np.ndarray,
    scale: int,
    rng: np.random.Generator,
    *,
    blur_sigma_range: tuple[float, float] = (0.5, 1.5),
    noise_std_range: tuple[float, float] = (3.0, 8.0),
    jpeg_prob: float = 0.35,
    jpeg_quality_range: tuple[int, int] = (60, 95),
) -> np.ndarray:
    """HR uint8 RGB -> LR uint8 RGB (spatial size H/scale x W/scale)."""
    h, w = hr_rgb.shape[:2]
    h2, w2 = (h // scale) * scale, (w // scale) * scale
    if h2 < scale or w2 < scale:
        raise ValueError(f"HR too small for scale {scale}: {hr_rgb.shape}")
    if (h2, w2) != (h, w):
        y0, x0 = (h - h2) // 2, (w - w2) // 2
        hr_rgb = hr_rgb[y0 : y0 + h2, x0 : x0 + w2]

    sigma = float(rng.uniform(*blur_sigma_range))
    if sigma > 0:
        hr_blur = cv2.GaussianBlur(hr_rgb, (0, 0), sigma)
    else:
        hr_blur = hr_rgb

    lr = cv2.resize(hr_blur, (w2 // scale, h2 // scale), interpolation=cv2.INTER_AREA)
    noise = float(rng.uniform(*noise_std_range))
    if noise > 0:
        lr = np.clip(lr.astype(np.float32) + rng.normal(0, noise, lr.shape), 0, 255).astype(np.uint8)

    if jpeg_prob > 0 and rng.random() < jpeg_prob:
        q = int(rng.integers(jpeg_quality_range[0], jpeg_quality_range[1] + 1))
        ok, enc = cv2.imencode(".jpg", cv2.cvtColor(lr, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if ok:
            lr = cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

    return lr


def bicubic_up_lr(lr_rgb: np.ndarray, scale: int) -> np.ndarray:
    h, w = lr_rgb.shape[:2]
    return cv2.resize(lr_rgb, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)


def uint8_rgb_to_tensor(img: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)


def tensor_to_uint8_rgb(t: torch.Tensor) -> np.ndarray:
    x = t.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return (x * 255.0).round().astype(np.uint8)


class DegradedPatchDataset(Dataset):
    """Returns (bicubic_hr, hr) tensors in [0,1], CHW."""

    def __init__(
        self,
        hr_samples: list[HrSample],
        scale: int = 4,
        patch_hr: int = 128,
        seed: int = 0,
    ) -> None:
        if not hr_samples:
            raise FileNotFoundError(
                "No HR frames for training. Your dataset/ has no REDS/Vimeo PNG sharp frames.\n"
                "Use --hr_source auto (default) to train from wild_real.mp4 + part2 Real-ESRGAN outputs,\n"
                "or place official sharp frames under dataset/REDS-sample/ etc."
            )
        self.hr_samples = hr_samples
        self.scale = scale
        self.patch_hr = max(patch_hr, scale * 8)
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.hr_samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        rng = np.random.default_rng(self.rng.integers(0, 2**31 - 1))
        hr_rgb = read_hr_rgb(self.hr_samples[index])
        lr_rgb = degrade_hr_to_lr(hr_rgb, self.scale, rng)
        bic_rgb = bicubic_up_lr(lr_rgb, self.scale)

        ph = min(self.patch_hr, bic_rgb.shape[0], bic_rgb.shape[1])
        ph = (ph // self.scale) * self.scale
        if ph < self.scale:
            raise ValueError(f"Patch too small at {self.hr_samples[index]}")

        if ph < bic_rgb.shape[0] or ph < bic_rgb.shape[1]:
            max_t = bic_rgb.shape[0] - ph
            max_l = bic_rgb.shape[1] - ph
            top = int(rng.integers(0, max_t + 1)) if max_t > 0 else 0
            left = int(rng.integers(0, max_l + 1)) if max_l > 0 else 0
            bic_rgb = bic_rgb[top : top + ph, left : left + ph]
            hr_rgb = hr_rgb[top : top + ph, left : left + ph]

        if rng.random() < 0.5:
            bic_rgb = np.ascontiguousarray(bic_rgb[:, ::-1, :])
            hr_rgb = np.ascontiguousarray(hr_rgb[:, ::-1, :])

        return uint8_rgb_to_tensor(bic_rgb), uint8_rgb_to_tensor(hr_rgb)


class FusionPatchDataset(Dataset):
    """Returns (hr, bicubic_hr, lr) for fusion-head training; gen branch in train loop."""

    def __init__(
        self,
        hr_samples: list[HrSample],
        scale: int = 4,
        patch_hr: int = 128,
        seed: int = 0,
    ) -> None:
        self.inner = DegradedPatchDataset(hr_samples, scale=scale, patch_hr=patch_hr, seed=seed)

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bic_t, hr_t = self.inner[index]
        s = self.inner.scale
        bic_np = tensor_to_uint8_rgb(bic_t)
        lr_np = cv2.resize(bic_np, (bic_np.shape[1] // s, bic_np.shape[0] // s), interpolation=cv2.INTER_AREA)
        return hr_t, bic_t, uint8_rgb_to_tensor(lr_np)
