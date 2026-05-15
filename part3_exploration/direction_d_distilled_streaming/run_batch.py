"""Batch Part 3鈥揇 distilled streaming SR on LR mp4 clips (Wild + REDS/Vimeo inputs).

Delegates to ``examples/WanVSR/infer_course_dataset.py`` (official-style
``prepare_input_tensor`` + ``init_pipeline`` + ``pipe``), one subprocess for the whole batch.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]  # CV repo root
_INFER = (
    REPO
    / "part3_exploration"
    / "direction_d_distilled_streaming"
    / "streaming_distillation_upstream"
    / "examples"
    / "WanVSR"
    / "infer_course_dataset.py"
)


def _weights_dir_resolved(weights_dir: str) -> Path:
    from part3_exploration.direction_d_distilled_streaming.streaming_one_step_infer import (
        resolve_weights_dir,
    )

    return resolve_weights_dir(weights_dir)


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch Direction D distilled streaming SR.")
    ap.add_argument("--inputs_dir", default="dataset/inputs_mp4")
    ap.add_argument("--wild_lr", default="dataset/wild_real_lr.mp4")
    ap.add_argument("--out_dir", default="part3_exploration/outputs/direction_d_distilled_streaming")
    ap.add_argument("--weights_dir", default="weights/part3_d")
    ap.add_argument("--variant", choices=("full", "tiny", "tiny-long"), default="full")
    ap.add_argument(
        "--tiled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Full variant: tiled VAE (default on). Pass --no-tiled for max speed if VRAM allows.",
    )
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip_wild", action="store_true")
    ap.add_argument("--force", action="store_true", help="Overwrite existing outputs.")
    ap.add_argument(
        "--cuda-id",
        type=int,
        default=None,
        help="Physical GPU index (passed to infer_course_dataset as CUDA_VISIBLE_DEVICES).",
    )
    ap.add_argument(
        "--low-vram",
        action="store_true",
        help="Forward --low-vram to infer_course_dataset (smaller VAE tiles, full only).",
    )
    args = ap.parse_args()

    resolved = _weights_dir_resolved(args.weights_dir)
    if not resolved.is_dir():
        print(
            f"Skipping Direction D batch: weights not found at {resolved}.\n"
            "Populate that directory (see direction_d README) then re-run.",
            file=sys.stderr,
        )
        return

    cmd = [
        sys.executable,
        str(_INFER),
        "--weights-dir",
        str(resolved),
        "--out-dir",
        str(REPO / args.out_dir),
        "--inputs-dir",
        str(REPO / args.inputs_dir),
        "--wild-lr",
        str(REPO / args.wild_lr),
        "--variant",
        args.variant,
    ]
    if args.skip_wild:
        cmd.append("--skip-wild")
    if args.limit > 0:
        cmd.extend(["--limit", str(args.limit)])
    cmd.append("--tiled" if args.tiled else "--no-tiled")
    if args.force:
        cmd.append("--force")
    if args.cuda_id is not None:
        cmd.extend(["--cuda-id", str(args.cuda_id)])
    if args.low_vram:
        cmd.append("--low-vram")
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=str(REPO), check=True)


if __name__ == "__main__":
    main()
