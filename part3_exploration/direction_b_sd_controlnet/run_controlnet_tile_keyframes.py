"""Stable Diffusion 1.5 + ControlNet-Tile on keyframes (Direction B)."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


def read_frame(video_path: str, frame_index: int) -> Image.Image:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_index} from {video_path}")
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def fit_to_sd_size(image: Image.Image, max_side: int) -> Image.Image:
    w, h = image.size
    scale = min(max_side / max(w, h), 1.0)
    nw = max(64, int(round(w * scale / 8) * 8))
    nh = max(64, int(round(h * scale / 8) * 8))
    if (nw, nh) == (w, h):
        return image
    return image.resize((nw, nh), Image.Resampling.LANCZOS)


def save_side_by_side(left: Image.Image, right: Image.Image, out_path: Path) -> None:
    if left.size != right.size:
        right = right.resize(left.size, Image.Resampling.LANCZOS)
    gap = 16
    canvas = Image.new("RGB", (left.width * 2 + gap, left.height), (245, 245, 245))
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width + gap, 0))
    canvas.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="ControlNet-Tile diffusion on sampled keyframes.")
    parser.add_argument("--input", default="part2_sota/outputs/wild_real_lr/realesrgan_x4.mp4")
    parser.add_argument(
        "--out_dir",
        default="part3_exploration/outputs/direction_b_sd_controlnet/diffusion_keyframes",
    )
    parser.add_argument("--frames", nargs="*", type=int, default=[80, 120, 160])
    parser.add_argument("--max_side", type=int, default=768)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--strength", type=float, default=0.22)
    parser.add_argument("--guidance", type=float, default=4.5)
    parser.add_argument("--control_scale", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--base_model", default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--controlnet", default="lllyasviel/control_v11f1e_sd15_tile")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument(
        "--prompt",
        default=(
            "high quality realistic video frame, natural sharp details, clean texture, "
            "low noise, faithful structure"
        ),
    )
    parser.add_argument(
        "--negative_prompt",
        default="cartoon, painting, over-sharpened, distorted geometry, text artifacts, hallucinated objects",
    )
    args = parser.parse_args()

    from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    controlnet = ControlNetModel.from_pretrained(
        args.controlnet, torch_dtype=dtype, local_files_only=args.local_files_only
    )
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        args.base_model,
        controlnet=controlnet,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
        local_files_only=args.local_files_only,
    )
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()

    generator = torch.Generator(device=device).manual_seed(args.seed)

    for frame_index in args.frames:
        source = fit_to_sd_size(read_frame(args.input, frame_index), args.max_side)
        result = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            image=source,
            control_image=source,
            strength=args.strength,
            guidance_scale=args.guidance,
            controlnet_conditioning_scale=args.control_scale,
            num_inference_steps=args.steps,
            generator=generator,
        ).images[0]

        source_path = out_dir / f"frame_{frame_index:04d}_realesrgan_input.png"
        out_path = out_dir / f"frame_{frame_index:04d}_diffusion_tile.png"
        compare_path = out_dir / f"frame_{frame_index:04d}_compare.png"
        source.save(source_path)
        result.save(out_path)
        save_side_by_side(source, result, compare_path)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
