"""Part 1 pipeline: bicubic, lanczos, SRCNN, temporal weighted average (+ optional unsharp)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from part1_baseline.spatial import unsharp_mask, upscale_frame
from part1_baseline.srcnn import SRCNN
from part1_baseline.clip_io import (
    batch_job_out_dir,
    collect_lr_dataset_jobs,
    default_scan_roots,
    describe_clip,
    format_dataset_hints,
    is_probably_frame_clip,
    read_clip_frames,
    repo_root,
)
from part1_baseline.video_io import read_video, tensor_to_uint8_frame, write_video


def weighted_temporal_stack(frames_up: list[np.ndarray], center: int, window: int) -> np.ndarray:
    """Gaussian-like weights centered at `center`."""
    n = len(frames_up)
    r = window // 2
    weights = []
    indices = []
    for d in range(-r, r + 1):
        idx = max(0, min(n - 1, center + d))
        indices.append(idx)
        sigma = max(r / 2.0, 1.0)
        w = np.exp(-(d**2) / (2 * sigma**2))
        weights.append(w)
    wsum = sum(weights)
    weights = [w / wsum for w in weights]
    acc = np.zeros_like(frames_up[0], dtype=np.float64)
    for wi, ii in zip(weights, indices):
        acc += wi * frames_up[ii].astype(np.float64)
    return np.clip(acc, 0, 255).astype(np.uint8)


def _default_out_dir() -> str:
    return str(Path(__file__).resolve().parent / "outputs")


def run_part1_pipeline(inp: Path, out_dir: str, args: argparse.Namespace) -> None:
    """对单个 LR 输入（视频或帧目录）跑完整 Part1 管线，写出一条 mp4。"""
    os.makedirs(out_dir, exist_ok=True)
    if not inp.exists():
        print(f"输入路径不存在: {inp.resolve()}", file=sys.stderr)
        print(format_dataset_hints(), file=sys.stderr)
        sys.exit(1)
    if inp.is_dir() and is_probably_frame_clip(inp):
        base = describe_clip(inp)
        frames = read_clip_frames(inp)
        fps = float(args.clip_fps)
    elif inp.is_dir():
        print(
            "该目录不是可识别的帧序列：REDS 需「当前目录下」帧文件名为纯数字；Vimeo 需 im1.png … im*.png。",
            file=sys.stderr,
        )
        print(format_dataset_hints(), file=sys.stderr)
        sys.exit(1)
    else:
        base = os.path.splitext(os.path.basename(str(inp)))[0]
        frames, fps = read_video(str(inp))
    out_mp4 = os.path.join(out_dir, f"{base}_part1_{args.mode}_x{args.scale}.mp4")
    if not frames:
        print("No frames read.", file=sys.stderr)
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode in ("bicubic", "lanczos"):
        out_frames = []
        interp = "bicubic" if args.mode == "bicubic" else "lanczos"
        for f in tqdm(frames, desc=args.mode):
            out_frames.append(upscale_frame(f, args.scale, interp))
        write_video(out_mp4, out_frames, fps)
        print(f"Wrote {out_mp4}")
        return

    if args.mode == "srcnn":
        if not args.srcnn_ckpt or not os.path.isfile(args.srcnn_ckpt):
            print("For srcnn mode, provide --srcnn_ckpt to a trained .pth", file=sys.stderr)
            sys.exit(1)
        ckpt = torch.load(args.srcnn_ckpt, map_location=device)
        model = SRCNN().to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        out_frames = []
        with torch.no_grad():
            for f in tqdm(frames, desc="srcnn"):
                up = upscale_frame(f, args.scale, "bicubic")
                t = torch.from_numpy(up.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
                pred = model(t)
                out_frames.append(tensor_to_uint8_frame(pred))
        write_video(out_mp4, out_frames, fps)
        print(f"Wrote {out_mp4}")
        return

    if args.mode == "temporal":
        w = max(3, args.temporal_window | 1)
        bic_up = [upscale_frame(f, args.scale, "bicubic") for f in frames]
        out_frames = []
        for c in tqdm(range(len(bic_up)), desc="temporal"):
            fused = weighted_temporal_stack(bic_up, c, w)
            if args.unsharp:
                fused = unsharp_mask(fused, sigma=1.0, amount=0.8)
            out_frames.append(fused)
        suffix = "_unsharp" if args.unsharp else ""
        out_mp4 = os.path.join(out_dir, f"{base}_part1_temporal_w{w}{suffix}_x{args.scale}.mp4")
        write_video(out_mp4, out_frames, fps)
        print(f"Wrote {out_mp4}")
        return


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Part1：单输入或批量扫描 dataset/ 与 data/raw/ 下全部 LR 视频与帧 clip。",
    )
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--input",
        type=str,
        help="单个 LR 视频路径，或 REDS/Vimeo 帧目录",
    )
    g.add_argument(
        "--input_all",
        action="store_true",
        help="不指定单个文件：扫描 --scan_roots（默认 dataset/ 与 data/raw/）下全部低码率项并逐项处理",
    )
    ap.add_argument(
        "--scan_roots",
        nargs="*",
        default=[],
        help="与 --input_all 连用：相对仓库根的扫描目录列表；省略则默认 dataset 与 data/raw（存在者）",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default=_default_out_dir(),
        help="单输入时：输出目录。--input_all 时：输出根目录，其下按数据来源分子目录",
    )
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument(
        "--mode",
        type=str,
        choices=("bicubic", "lanczos", "srcnn", "temporal"),
        default="bicubic",
    )
    ap.add_argument("--srcnn_ckpt", type=str, default="", help="Required for mode=srcnn")
    ap.add_argument("--temporal_window", type=int, default=5, help="Odd window for temporal mode")
    ap.add_argument("--unsharp", action="store_true", help="Apply unsharp after temporal fusion")
    ap.add_argument(
        "--clip_fps",
        type=float,
        default=24.0,
        help="当输入为帧目录时写入 MP4 的帧率（默认 24）",
    )
    args = ap.parse_args()

    if args.input_all:
        base_repo = repo_root()
        if args.scan_roots:
            roots = [(base_repo / r).resolve() for r in args.scan_roots]
        else:
            roots = default_scan_roots(base_repo)
        roots = [p for p in roots if p.is_dir()]
        if not roots:
            print("未找到可扫描目录：请创建 dataset/ 或 data/raw/，或使用 --scan_roots", file=sys.stderr)
            sys.exit(1)
        jobs = collect_lr_dataset_jobs(roots)
        if not jobs:
            print(f"在 {[str(r) for r in roots]} 下未发现视频或 REDS/Vimeo 帧目录。", file=sys.stderr)
            print(format_dataset_hints(), file=sys.stderr)
            sys.exit(1)
        batch_root = Path(args.out_dir)
        print(f"批量处理共 {len(jobs)} 个输入，输出根目录: {batch_root.resolve()}")
        for i, (data_root, inp) in enumerate(jobs, 1):
            job_out = batch_job_out_dir(batch_root, data_root, inp)
            job_out.mkdir(parents=True, exist_ok=True)
            print(f"[{i}/{len(jobs)}] {inp}")
            run_part1_pipeline(inp, str(job_out), args)
        return

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"输入路径不存在: {in_path.resolve()}", file=sys.stderr)
        print("", file=sys.stderr)
        print("请使用本机 dataset 下真实路径，或改用 --input_all 批量扫描。当前扫描结果：", file=sys.stderr)
        print(format_dataset_hints(), file=sys.stderr)
        print("", file=sys.stderr)
        print('提示: python scripts/print_dataset_layout.py', file=sys.stderr)
        sys.exit(1)
    run_part1_pipeline(in_path, args.out_dir, args)


if __name__ == "__main__":
    main()
