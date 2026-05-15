#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Part 3–D inference: single-step distilled streaming diffusion VSR (4×).
Runtime lives under streaming_distillation_upstream/; install that tree (pip install -e) or rely on PYTHONPATH below.

Upstream research implementation (Apache-2.0) is vendored unmodified aside from filesystem layout expectations.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import imageio
from tqdm import tqdm
import torch
from einops import rearrange

_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _DIR.parents[1]  # CV repo root (…/direction_d → part3_exploration → CV)
_UPSTREAM_ROOT = _DIR / "streaming_distillation_upstream"
_WAN_EXAMPLES = _UPSTREAM_ROOT / "examples" / "WanVSR"


def resolve_repo_path(p: str | Path) -> Path:
    """Resolve relative paths against CV repo root (parent of part3_exploration/)."""
    path = Path(p).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (_REPO_ROOT / path).resolve()


def resolve_weights_dir(arg: str) -> Path:
    """Weights live in ``weights/part3_d/`` (flat) or ``weights/part3_d/streaming_ckpt_v11/``."""
    wdir = resolve_repo_path(arg)
    if wdir.is_dir():
        return wdir
    if "streaming_ckpt_v11" in arg.replace("\\", "/"):
        flat = resolve_repo_path("weights/part3_d")
        if flat.is_dir() and (flat / "diffusion_pytorch_model_streaming_dmd.safetensors").is_file():
            return flat
    return wdir


for _p in (_UPSTREAM_ROOT, _WAN_EXAMPLES):
    ps = str(_p)
    if ps not in sys.path:
        sys.path.insert(0, ps)


def tensor2video(frames: torch.Tensor):
    frames = rearrange(frames, "C T H W -> T H W C")
    frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
    frames = [Image.fromarray(frame) for frame in frames]
    return frames


def natural_key(name: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"([0-9]+)", os.path.basename(name))]


def list_images_natural(folder: str):
    exts = (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG")
    fs = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(exts)]
    fs.sort(key=natural_key)
    return fs


def largest_8n1_leq(n):  # 8n+1
    return 0 if n < 1 else ((n - 1) // 8) * 8 + 1


def is_video(path):
    return os.path.isfile(path) and path.lower().endswith((".mp4", ".mov", ".avi", ".mkv"))


def pil_to_tensor_neg1_1(img: Image.Image, dtype=torch.bfloat16, device="cuda"):
    t = torch.from_numpy(np.asarray(img, np.uint8)).to(device=device, dtype=torch.float32)
    t = t.permute(2, 0, 1) / 255.0 * 2.0 - 1.0
    return t.to(dtype)


def save_video(frames, save_path, fps=30, quality=5):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    w = imageio.get_writer(str(save_path), fps=fps, quality=quality)
    for f in tqdm(frames, desc=f"Saving {save_path.name}"):
        w.append_data(np.array(f))
    w.close()


def compute_scaled_and_target_dims(w0: int, h0: int, scale=4.0, multiple: int = 128):
    if w0 <= 0 or h0 <= 0:
        raise ValueError("invalid original size")
    if isinstance(scale, int):
        sW, sH = w0 * scale, h0 * scale
    else:
        sW = int(round(w0 * scale))
        sH = int(round(h0 * scale))
    tW = max(multiple, (sW // multiple) * multiple)
    tH = max(multiple, (sH // multiple) * multiple)
    return sW, sH, tW, tH


def upscale_then_center_crop(img: Image.Image, scale, tW: int, tH: int) -> Image.Image:
    w0, h0 = img.size
    if isinstance(scale, int):
        sW, sH = w0 * scale, h0 * scale
    else:
        sW = int(round(w0 * scale))
        sH = int(round(h0 * scale))
    up = img.resize((sW, sH), Image.BICUBIC)
    l = max(0, (sW - tW) // 2)
    t = max(0, (sH - tH) // 2)
    return up.crop((l, t, l + tW, t + tH))


def prepare_input_tensor(path: str, scale=4, dtype=torch.bfloat16, device="cuda"):
    if os.path.isdir(path):
        paths0 = list_images_natural(path)
        if not paths0:
            raise FileNotFoundError(f"No images in {path}")
        with Image.open(paths0[0]) as _img0:
            w0, h0 = _img0.size
        N0 = len(paths0)
        print(f"[{os.path.basename(path)}] Original Resolution: {w0}x{h0} | Original Frames: {N0}")

        sW, sH, tW, tH = compute_scaled_and_target_dims(w0, h0, scale=scale, multiple=128)
        print(f"[{os.path.basename(path)}] Scaled: {sW}x{sH} -> Target (128-multiple): {tW}x{tH}")

        paths = paths0 + [paths0[-1]] * 4
        F = largest_8n1_leq(len(paths))
        if F == 0:
            raise RuntimeError(f"Not enough frames after padding in {path}. Got {len(paths)}.")
        paths = paths[:F]
        print(f"[{os.path.basename(path)}] Target Frames (8n-3): {F - 4}")

        frames = []
        for p in paths:
            with Image.open(p).convert("RGB") as img:
                img_out = upscale_then_center_crop(img, scale=scale, tW=tW, tH=tH)
            frames.append(pil_to_tensor_neg1_1(img_out, dtype, device))
        vid = torch.stack(frames, 0).permute(1, 0, 2, 3).unsqueeze(0)
        fps = 30
        return vid, tH, tW, F, fps

    if is_video(path):
        rdr = imageio.get_reader(path)
        first = Image.fromarray(rdr.get_data(0)).convert("RGB")
        w0, h0 = first.size

        meta = {}
        try:
            meta = rdr.get_meta_data()
        except Exception:
            pass
        fps_val = meta.get("fps", 30)
        fps = int(round(fps_val)) if isinstance(fps_val, (int, float)) else 30

        def count_frames(r):
            try:
                nf = meta.get("nframes", None)
                if isinstance(nf, int) and nf > 0:
                    return nf
            except Exception:
                pass
            try:
                return r.count_frames()
            except Exception:
                n = 0
                try:
                    while True:
                        r.get_data(n)
                        n += 1
                except Exception:
                    return n

        total = count_frames(rdr)
        if total <= 0:
            rdr.close()
            raise RuntimeError(f"Cannot read frames from {path}")

        print(f"[{os.path.basename(path)}] Original Resolution: {w0}x{h0} | Original Frames: {total} | FPS: {fps}")

        sW, sH, tW, tH = compute_scaled_and_target_dims(w0, h0, scale=scale, multiple=128)
        print(f"[{os.path.basename(path)}] Scaled: {sW}x{sH} -> Target (128-multiple): {tW}x{tH}")

        idx = list(range(total)) + [total - 1] * 4
        F = largest_8n1_leq(len(idx))
        if F == 0:
            rdr.close()
            raise RuntimeError(f"Not enough frames after padding in {path}. Got {len(idx)}.")
        idx = idx[:F]
        print(f"[{os.path.basename(path)}] Target Frames (8n-3): {F - 4}")

        frames = []
        try:
            for i in idx:
                img = Image.fromarray(rdr.get_data(i)).convert("RGB")
                img_out = upscale_then_center_crop(img, scale=scale, tW=tW, tH=tH)
                frames.append(pil_to_tensor_neg1_1(img_out, dtype, device))
        finally:
            try:
                rdr.close()
            except Exception:
                pass

        vid = torch.stack(frames, 0).permute(1, 0, 2, 3).unsqueeze(0)
        return vid, tH, tW, F, fps

    raise ValueError(f"Unsupported input: {path}")


def init_pipeline_full(weights_dir: Path):
    from diffsynth import ModelManager, FlashVSRFullPipeline
    from utils.utils import Causal_LQ4x_Proj

    d = weights_dir.resolve()
    w_dit = d / "diffusion_pytorch_model_streaming_dmd.safetensors"
    w_vae = d / "Wan2.1_VAE.pth"
    w_proj = d / "LQ_proj_in.ckpt"
    for req in (w_dit, w_vae):
        if not req.is_file():
            raise FileNotFoundError(f"Missing weight file: {req}")
    mm = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
    mm.load_models([str(w_dit), str(w_vae)])
    pipe = FlashVSRFullPipeline.from_model_manager(mm, device="cuda")
    pipe.denoising_model().LQ_proj_in = Causal_LQ4x_Proj(in_dim=3, out_dim=1536, layer_num=1).to("cuda", dtype=torch.bfloat16)
    if w_proj.is_file():
        pipe.denoising_model().LQ_proj_in.load_state_dict(torch.load(str(w_proj), map_location="cpu"), strict=True)
    pipe.denoising_model().LQ_proj_in.to("cuda")
    pipe.vae.model.encoder = None
    pipe.vae.model.conv1 = None
    pipe.to("cuda")
    pipe.enable_vram_management(num_persistent_param_in_dit=None)
    pipe.init_cross_kv()
    pipe.load_models_to_device(["dit", "vae"])
    return pipe


def init_pipeline_tiny(weights_dir: Path):
    from diffsynth import ModelManager, FlashVSRTinyPipeline
    from utils.utils import Causal_LQ4x_Proj
    from utils.TCDecoder import build_tcdecoder

    d = weights_dir.resolve()
    w_dit = d / "diffusion_pytorch_model_streaming_dmd.safetensors"
    w_proj = d / "LQ_proj_in.ckpt"
    w_tc = d / "TCDecoder.ckpt"
    if not w_dit.is_file():
        raise FileNotFoundError(f"Missing weight file: {w_dit}")
    mm = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
    mm.load_models([str(w_dit)])
    pipe = FlashVSRTinyPipeline.from_model_manager(mm, device="cuda")
    pipe.denoising_model().LQ_proj_in = Causal_LQ4x_Proj(in_dim=3, out_dim=1536, layer_num=1).to("cuda", dtype=torch.bfloat16)
    if w_proj.is_file():
        pipe.denoising_model().LQ_proj_in.load_state_dict(torch.load(str(w_proj), map_location="cpu"), strict=True)
    pipe.denoising_model().LQ_proj_in.to("cuda")

    multi_scale_channels = [512, 256, 128, 128]
    pipe.TCDecoder = build_tcdecoder(new_channels=multi_scale_channels, new_latent_channels=16 + 768)
    if w_tc.is_file():
        pipe.TCDecoder.load_state_dict(torch.load(str(w_tc), map_location="cpu"), strict=False)
    pipe.to("cuda")
    pipe.enable_vram_management(num_persistent_param_in_dit=None)
    pipe.init_cross_kv()
    pipe.load_models_to_device(["dit", "vae"])
    return pipe


def main():
    ap = argparse.ArgumentParser(description="Part 3-D distilled streaming 4× VSR inference.")
    ap.add_argument("--input", required=True, help="LR mp4 path or RGB frame folder.")
    ap.add_argument("--out", required=True, help="Output HR mp4 path.")
    ap.add_argument(
        "--weights_dir",
        type=str,
        default="weights/part3_d",
        help="Folder with checkpoints (flat weights/part3_d/ or weights/part3_d/streaming_ckpt_v11/)",
    )
    ap.add_argument("--variant", choices=("full", "tiny"), default="full")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scale", type=float, default=4.0)
    ap.add_argument("--sparse_ratio", type=float, default=2.0, help="Sparse attention tradeoff for full variant.")
    ap.add_argument("--local_range", type=int, default=11)
    ap.add_argument("--tiled", action="store_true", help="Lower VRAM, slower (full variant).")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this pipeline (torch.cuda.is_available() is False).")

    inp = resolve_repo_path(args.input)
    out = resolve_repo_path(args.out)
    wdir = resolve_weights_dir(args.weights_dir)
    if not wdir.is_dir():
        raise FileNotFoundError(
            f"Weights dir not found: {wdir}\n"
            "Put the four checkpoint files under weights/part3_d/ (repo root), "
            "or pass --weights_dir with an absolute path."
        )

    dtype, device = torch.bfloat16, "cuda"

    if args.variant == "full":
        pipe = init_pipeline_full(wdir)
    else:
        pipe = init_pipeline_tiny(wdir)

    torch.cuda.empty_cache()

    ip = str(inp)
    LQ, th, tw, F, fps = prepare_input_tensor(ip, scale=args.scale, dtype=dtype, device=device)

    sparse_ratio = float(args.sparse_ratio)
    topk = sparse_ratio * 768 * 1280 / (th * tw)

    if args.variant == "full":
        video = pipe(
            prompt="",
            negative_prompt="",
            cfg_scale=1.0,
            num_inference_steps=1,
            seed=args.seed,
            tiled=args.tiled,
            LQ_video=LQ,
            num_frames=F,
            height=th,
            width=tw,
            is_full_block=False,
            if_buffer=True,
            topk_ratio=topk,
            kv_ratio=3.0,
            local_range=args.local_range,
            color_fix=True,
        )
    else:
        video = pipe(
            prompt="",
            negative_prompt="",
            cfg_scale=1.0,
            num_inference_steps=1,
            seed=args.seed,
            LQ_video=LQ,
            num_frames=F,
            height=th,
            width=tw,
            is_full_block=False,
            if_buffer=True,
            topk_ratio=topk,
            kv_ratio=3.0,
            local_range=args.local_range,
            color_fix=True,
        )

    video = tensor2video(video)
    save_video(video, str(out), fps=fps, quality=6)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
