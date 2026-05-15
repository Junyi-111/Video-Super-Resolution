"""One-shot Part 3 training pipeline (~5h): RefineNet (A) then FusionHead (C)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(REPO), check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="5h Part 3 training: A then C.")
    ap.add_argument("--epochs_a", type=int, default=8)
    ap.add_argument("--epochs_c", type=int, default=15)
    ap.add_argument("--batch_a", type=int, default=16)
    ap.add_argument("--batch_c", type=int, default=8)
    ap.add_argument("--skip_a", action="store_true")
    ap.add_argument("--skip_c", action="store_true")
    ap.add_argument("--eval_wild", action="store_true", help="Run Wild inference + metrics after training.")
    args = ap.parse_args()

    py = sys.executable
    ckpt_a = "weights/part3_a/refinenet_x4.pth"
    ckpt_c = "weights/part3_c/fusion_head.pth"

    if not args.skip_a:
        run(
            [
                py,
                "-m",
                "part3_exploration.direction_a_flow_matching.train_refinenet",
                "--epochs",
                str(args.epochs_a),
                "--batch_size",
                str(args.batch_a),
                "--out_ckpt",
                ckpt_a,
            ]
        )

    if not args.skip_c:
        run(
            [
                py,
                "-m",
                "part3_exploration.direction_c_uncertainty.train_fusion",
                "--epochs",
                str(args.epochs_c),
                "--batch_size",
                str(args.batch_c),
                "--refinenet_ckpt",
                ckpt_a,
                "--freeze_refinenet",
                "--out_ckpt",
                ckpt_c,
            ]
        )

    if args.eval_wild:
        out_a = "part3_exploration/outputs/direction_a_flow_matching/wild_real_lr_refinenet_x4.mp4"
        out_c = "part3_exploration/outputs/direction_c_uncertainty/wild_real_lr_fusion_x4.mp4"
        run(
            [
                py,
                "-m",
                "part3_exploration.direction_a_flow_matching.infer_refinenet",
                "--input",
                "dataset/wild_real_lr.mp4",
                "--out",
                out_a,
                "--ckpt",
                ckpt_a,
            ]
        )
        run(
            [
                py,
                "-m",
                "part3_exploration.direction_c_uncertainty.infer_fusion",
                "--basic",
                "part2_sota/outputs/basicvsr/wild_real_lr.mp4",
                "--gen",
                "part2_sota/outputs/wild_real_lr/realesrgan_x4.mp4",
                "--out",
                out_c,
                "--fusion_ckpt",
                ckpt_c,
            ]
        )
        run([py, "scripts/evaluate_project.py", "--skip_fid", "--skip_sample"])


if __name__ == "__main__":
    main()
