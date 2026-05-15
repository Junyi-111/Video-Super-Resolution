"""Real-ESRGAN x4 inference (frame-wise) using official pip stack."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from tqdm import tqdm


def _default_out_path() -> str:
    return str(Path(__file__).resolve().parent / "outputs" / "realesrgan_out.mp4")


def build_realesrgan_upsampler(
    model_path: str,
    tile: int = 0,
    tile_pad: int = 10,
) -> Any:
    """构造 RealESRGANer；依赖 basicsr / realesrgan。"""
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    return RealESRGANer(
        scale=4,
        model_path=model_path,
        model=model,
        tile=tile,
        tile_pad=tile_pad,
        pre_pad=0,
        half=torch.cuda.is_available(),
    )


def enhance_rgb_frames(
    upsampler: Any,
    frames: list[np.ndarray],
    desc: str | None = "realesrgan",
) -> list[np.ndarray]:
    """RGB uint8 帧列表 → 4× 超分后的 RGB 帧列表。`desc=None` 时不显示帧级 tqdm。"""
    out_frames: list[np.ndarray] = []
    it = tqdm(frames, desc=desc) if desc is not None else frames
    for f in it:
        bgr = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
        out, _ = upsampler.enhance(bgr, outscale=4)
        out_frames.append(np.ascontiguousarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB)))
    return out_frames


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True)
    ap.add_argument(
        "--out",
        type=str,
        default=_default_out_path(),
        help="输出 mp4 路径，默认 part2_sota/outputs/realesrgan_out.mp4",
    )
    ap.add_argument(
        "--model_path",
        type=str,
        default=os.path.join("weights", "realesrgan", "RealESRGAN_x4plus.pth"),
    )
    ap.add_argument("--tile", type=int, default=0, help="Tile size for GPU memory; 0 = no tiling")
    ap.add_argument("--tile_pad", type=int, default=10)
    args = ap.parse_args()

    try:
        from basicsr.archs.rrdbnet_arch import RRDBNet  # noqa: F401
        from realesrgan import RealESRGANer  # noqa: F401
    except ImportError as e:
        print(
            "Missing dependencies. Install: pip install basicsr facexlib gfpgan realesrgan",
            file=sys.stderr,
        )
        raise e

    from part1_baseline.video_io import read_video, write_video

    if not os.path.isfile(args.model_path):
        print(f"Weight not found: {args.model_path}", file=sys.stderr)
        sys.exit(1)

    inp = Path(args.input)
    if not inp.exists():
        print(f"输入不存在: {inp.resolve()}", file=sys.stderr)
        try:
            from part1_baseline.clip_io import format_dataset_hints

            print(format_dataset_hints(), file=sys.stderr)
        except Exception:
            pass
        sys.exit(1)
    if inp.is_dir():
        print(
            "Real-ESRGAN 当前脚本仅接受视频文件路径；帧目录请先 python scripts/frames_to_video.py 封装为 mp4，或使用 Wild 等目录下的 .mp4 作为 --input。",
            file=sys.stderr,
        )
        sys.exit(1)

    upsampler = build_realesrgan_upsampler(args.model_path, args.tile, args.tile_pad)
    frames, fps = read_video(str(inp))
    out_frames = enhance_rgb_frames(upsampler, frames, desc="realesrgan")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    write_video(args.out, out_frames, fps)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
