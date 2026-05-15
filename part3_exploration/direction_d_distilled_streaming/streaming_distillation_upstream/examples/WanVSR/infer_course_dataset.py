#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Course dataset inference — same layout as upstream ``examples/WanVSR`` drivers
(``prepare_input_tensor`` + ``init_pipeline`` + ``pipe(...)``).

Run **from this directory** (or use ``python -m part3_exploration.direction_d_distilled_streaming.run_course_infer`` from repo root).

Creates a symlink ``./part3_streaming_weights`` → your weights folder (default: repo ``weights/part3_d``).
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

# __file__ is .../CV/part3_exploration/.../examples/WanVSR/infer_course_dataset.py
# _HERE = WanVSR; parents[4] = CV (parents[5] would be the parent of CV).
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[4]
_WEIGHTS_LINK = _HERE / "part3_streaming_weights"

# Vendored streaming-VSR diffsynth (under streaming_distillation_upstream/) must shadow PyPI ``diffsynth``.
_UPSTREAM_ROOT = _HERE.parents[1]
for _p in (_UPSTREAM_ROOT, _HERE):
    _ps = str(_p.resolve())
    if _ps not in sys.path:
        sys.path.insert(0, _ps)

_VARIANT_FILES = {
    "full": "infer_streaming_v11_full.py",
    "tiny": "infer_streaming_v11_tiny.py",
    "tiny-long": "infer_streaming_v11_tiny_long_video.py",
}


def _load_infer_module(name: str):
    path = _HERE / name
    spec = importlib.util.spec_from_file_location(name.replace(".", "_"), path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ensure_weights_link(weights_dir: Path, variant: str) -> None:
    weights_dir = weights_dir.resolve()
    if not weights_dir.is_dir():
        raise FileNotFoundError(f"Weights directory not found: {weights_dir}")
    need = [
        "diffusion_pytorch_model_streaming_dmd.safetensors",
        "LQ_proj_in.ckpt",
    ]
    if variant == "full":
        need.append("Wan2.1_VAE.pth")
    if variant in ("tiny", "tiny-long"):
        need.append("TCDecoder.ckpt")
    for n in need:
        if not (weights_dir / n).is_file():
            raise FileNotFoundError(f"Missing {n} under {weights_dir}")
    if _WEIGHTS_LINK.is_symlink() or _WEIGHTS_LINK.is_file():
        _WEIGHTS_LINK.unlink()
    elif _WEIGHTS_LINK.is_dir():
        raise RuntimeError(f"Remove or rename existing directory: {_WEIGHTS_LINK}")
    _WEIGHTS_LINK.symlink_to(weights_dir, target_is_directory=True)


def _collect_inputs(
    repo: Path,
    inputs_dir: Path | None,
    wild_lr: Path | None,
    *,
    skip_wild: bool,
    skip_inputs: bool,
    limit: int,
) -> list[Path]:
    jobs: list[Path] = []
    if not skip_wild and wild_lr is not None and wild_lr.is_file():
        jobs.append(wild_lr.resolve())
    if not skip_inputs and inputs_dir is not None and inputs_dir.is_dir():
        for p in sorted(inputs_dir.glob("*.mp4")):
            if p.name.startswith("wild"):
                continue
            jobs.append(p.resolve())
    if limit > 0:
        jobs = jobs[:limit]
    return jobs


def _run_pipe(pipe, variant: str, tiled: bool, **common):
    kwargs = {
        "prompt": "",
        "negative_prompt": "",
        "cfg_scale": 1.0,
        "num_inference_steps": 1,
        "seed": common["seed"],
        "LQ_video": common["LQ"],
        "num_frames": common["F"],
        "height": common["th"],
        "width": common["tw"],
        "is_full_block": False,
        "if_buffer": True,
        "topk_ratio": common["sparse_ratio"] * 768 * 1280 / (common["th"] * common["tw"]),
        "kv_ratio": 3.0,
        "local_range": 11,
        "color_fix": True,
    }
    if variant == "full":
        kwargs["tiled"] = tiled
        if common.get("low_vram"):
            kwargs["tile_size"] = (34, 34)
            kwargs["tile_stride"] = (18, 16)
    return pipe(**kwargs)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run v1.1-style streaming VSR on course LR mp4s (Wild + dataset/inputs_mp4)."
    )
    ap.add_argument(
        "--weights-dir",
        type=str,
        default=str(_REPO / "weights" / "part3_d"),
        help="Folder containing diffusion + VAE/LQ weights (same files as upstream v1.1 bundle).",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default=str(_REPO / "part3_exploration" / "direction_d_distilled_streaming" / "outputs"),
    )
    ap.add_argument("--inputs-dir", type=str, default=str(_REPO / "dataset" / "inputs_mp4"))
    ap.add_argument("--wild-lr", type=str, default=str(_REPO / "dataset" / "wild_real_lr.mp4"))
    ap.add_argument("--skip-wild", action="store_true")
    ap.add_argument("--skip-inputs-dir", action="store_true")
    ap.add_argument("--variant", choices=sorted(_VARIANT_FILES.keys()), default="full")
    ap.add_argument("--limit", type=int, default=0, help="Max number of clips (0 = all collected).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sparse-ratio", type=float, default=2.0)
    ap.add_argument(
        "--tiled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Full variant only: tiled VAE encode/decode. Default on (saves VRAM on long / large clips). Use --no-tiled if you have headroom and want max speed.",
    )
    ap.add_argument(
        "--low-vram",
        action="store_true",
        help="Full only: smaller VAE tiles (slower, lower peak VRAM during decode). Use if you still OOM with default tiling.",
    )
    ap.add_argument("--force", action="store_true", help="Overwrite existing output mp4s.")
    ap.add_argument(
        "--cuda-id",
        type=int,
        default=None,
        metavar="N",
        help="Use physical GPU N only (sets CUDA_VISIBLE_DEVICES before torch loads). Example: --cuda-id 6",
    )
    args = ap.parse_args()

    if args.cuda_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_id)
        print(f"[infer_course_dataset] CUDA_VISIBLE_DEVICES={args.cuda_id}")

    # Reduces peak VRAM fragmentation on long streaming runs (safe default if unset).
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    os.chdir(_HERE)

    weights = Path(args.weights_dir).expanduser()
    if not weights.is_absolute():
        weights = (_REPO / weights).resolve()
    else:
        weights = weights.resolve()

    def _resolve_repo_path(p: str) -> Path:
        path = Path(p).expanduser()
        return path.resolve() if path.is_absolute() else (_REPO / path).resolve()

    inputs_dir = _resolve_repo_path(args.inputs_dir)
    wild_lr = _resolve_repo_path(args.wild_lr)
    jobs = _collect_inputs(
        _REPO,
        inputs_dir if not args.skip_inputs_dir else None,
        wild_lr if not args.skip_wild else None,
        skip_wild=args.skip_wild,
        skip_inputs=args.skip_inputs_dir,
        limit=args.limit,
    )
    if not jobs:
        print("No input videos found. Check --inputs-dir / --wild-lr / --skip-* flags.")
        return

    _ensure_weights_link(weights, args.variant)

    mod_name = _VARIANT_FILES[args.variant]
    m = _load_infer_module(mod_name)
    prepare_input_tensor = m.prepare_input_tensor
    tensor2video = m.tensor2video
    save_video = m.save_video
    init_pipeline = m.init_pipeline

    out_dir = Path(args.out_dir).expanduser()
    if not out_dir.is_absolute():
        out_dir = (_REPO / out_dir).resolve()
    else:
        out_dir = out_dir.resolve()

    out_dir.mkdir(parents=True, exist_ok=True)

    scale = 4.0 if args.variant != "full" else 4
    import torch

    dtype = torch.bfloat16
    device = "cuda"

    print(
        f"[infer_course_dataset] variant={args.variant} jobs={len(jobs)} "
        f"tiled={args.tiled} low_vram={args.low_vram} weights->{weights}"
    )
    pipe = init_pipeline()

    for inp in jobs:
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        name = inp.name
        if name.startswith("."):
            continue
        stem = inp.stem
        out_path = out_dir / f"{stem}_streaming_distilled_x4.mp4"
        if not args.force and out_path.is_file() and out_path.stat().st_size > 0:
            print(f"skip existing {out_path.name}")
            continue
        try:
            LQ, th, tw, F, fps = prepare_input_tensor(str(inp), scale=scale, dtype=dtype, device=device)
        except Exception as e:
            print(f"[Error] {name}: {e}")
            continue

        try:
            video = _run_pipe(
                pipe,
                args.variant,
                args.tiled,
                seed=args.seed,
                LQ=LQ,
                F=F,
                th=th,
                tw=tw,
                sparse_ratio=args.sparse_ratio,
                low_vram=args.low_vram,
            )
        except Exception as e:
            print(f"[Error] pipe {name}: {e}")
            continue

        video = tensor2video(video)
        save_video(video, str(out_path), fps=fps, quality=6 if args.variant == "full" else 5)
        print(f"Wrote {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
