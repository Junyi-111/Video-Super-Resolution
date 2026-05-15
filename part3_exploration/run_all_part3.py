"""Run all Part 3 directions (A/B/C) on the full dataset, then evaluate metrics."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def run_mod(module: str, extra: list[str] | None = None) -> None:
    cmd = [sys.executable, "-m", module] + (extra or [])
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=str(REPO), check=True)


def sync_legacy_wild() -> None:
    """Copy wild outputs to legacy paths for older figures / scripts."""
    legacy_dir = REPO / "part3_exploration/outputs"
    legacy_dir.mkdir(parents=True, exist_ok=True)
    pairs = [
        (
            REPO / "part3_exploration/outputs/direction_b_sd_controlnet/wild_real_lr_temporal_gen_blend_x4.mp4",
            legacy_dir / "wild_real_lr_hybrid_x4.mp4",
        ),
        (
            REPO / "part3_exploration/outputs/direction_b_sd_controlnet/wild_real_lr_flow_stabilized_x4.mp4",
            legacy_dir / "wild_real_lr_flow_stabilized_x4.mp4",
        ),
        (
            REPO / "part3_exploration/outputs/direction_c_uncertainty/wild_real_lr_uncertainty_adaptive_x4.mp4",
            legacy_dir / "wild_real_lr_uncertainty_adaptive_x4.mp4",
        ),
        (
            REPO / "part3_exploration/outputs/direction_c_uncertainty/wild_real_lr_uncertainty_alpha.mp4",
            legacy_dir / "wild_real_lr_uncertainty_alpha.mp4",
        ),
    ]
    for src, dst in pairs:
        if src.is_file():
            shutil.copy2(src, dst)
            print(f"legacy sync: {dst.relative_to(REPO)}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Full Part 3 batch + metrics.")
    ap.add_argument("--limit", type=int, default=0, help="Limit clips per direction (0 = all).")
    ap.add_argument("--skip_a", action="store_true")
    ap.add_argument("--skip_b", action="store_true")
    ap.add_argument("--skip_c", action="store_true")
    ap.add_argument("--skip_d", action="store_true")
    ap.add_argument("--skip_eval", action="store_true")
    ap.add_argument("--skip_legacy_sync", action="store_true")
    ap.add_argument("--eval_skip_fid", action="store_true")
    args = ap.parse_args()

    lim = ["--limit", str(args.limit)] if args.limit > 0 else []

    if not args.skip_a:
        run_mod("part3_exploration.direction_a_flow_matching.run_batch", lim)
    if not args.skip_b:
        run_mod("part3_exploration.direction_b_sd_controlnet.run_batch", lim)
    if not args.skip_c:
        run_mod("part3_exploration.direction_c_uncertainty.run_batch", lim)
    if not args.skip_d:
        run_mod("part3_exploration.direction_d_distilled_streaming.run_batch", lim)

    if not args.skip_legacy_sync:
        sync_legacy_wild()

    if not args.skip_eval:
        ev_extra = []
        if args.eval_skip_fid:
            ev_extra.append("--skip_fid")
        run_mod("part3_exploration.evaluate_part3_metrics", ev_extra)


if __name__ == "__main__":
    main()
