"""Batch Direction B: flow stabilization + temporal/generative blend per clip."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def find_temporal(stem: str, part1_root: Path) -> Path | None:
    for p in part1_root.rglob("*.mp4"):
        if p.stem.startswith(f"{stem}_part1_temporal"):
            return p
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch Direction B (flow + blend).")
    ap.add_argument("--inputs_dir", default="dataset/inputs_mp4")
    ap.add_argument("--wild_lr", default="dataset/wild_real_lr.mp4")
    ap.add_argument("--part1_root", default="part1_baseline/outputs")
    ap.add_argument("--realesrgan_dir", default="part2_sota/outputs/realesrgan")
    ap.add_argument("--out_dir", default="part3_exploration/outputs/direction_b_sd_controlnet")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip_wild", action="store_true")
    ap.add_argument("--skip_blend", action="store_true")
    ap.add_argument("--skip_flow", action="store_true")
    args = ap.parse_args()

    out_dir = REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    part1 = REPO / args.part1_root
    rs_dir = REPO / args.realesrgan_dir
    inp_dir = REPO / args.inputs_dir

    jobs: list[tuple[str, Path, Path, Path | None]] = []

    def add_job(stem: str, lr: Path, rs: Path) -> None:
        tmp = find_temporal(stem, part1)
        jobs.append((stem, lr, rs, tmp))

    if not args.skip_wild:
        wstem = Path(args.wild_lr).stem
        lr = REPO / args.wild_lr
        rs = REPO / "part2_sota/outputs/wild_real_lr/realesrgan_x4.mp4"
        tmp = REPO / f"part1_baseline/outputs/{wstem}/{wstem}_part1_temporal_w5_unsharp_x4.mp4"
        if lr.is_file() and rs.is_file():
            jobs.append((wstem, lr, rs, tmp if tmp.is_file() else None))

    if inp_dir.is_dir():
        for mp4 in sorted(inp_dir.glob("*.mp4")):
            if mp4.name.startswith("wild"):
                continue
            stem = mp4.stem
            rs = rs_dir / f"{stem}.mp4"
            if not rs.is_file():
                continue
            add_job(stem, mp4, rs)

    if args.limit > 0:
        jobs = jobs[: args.limit]

    mod_flow = "part3_exploration.direction_b_sd_controlnet.run_flow_stabilize"
    mod_blend = "part3_exploration.direction_b_sd_controlnet.run_temporal_generative_blend"

    for stem, lr, rs, temporal in jobs:
        if not args.skip_flow:
            out_flow = out_dir / f"{stem}_flow_stabilized_x4.mp4"
            if not (out_flow.exists() and out_flow.stat().st_size > 0):
                cmd = [
                    sys.executable,
                    "-m",
                    mod_flow,
                    "--lr",
                    str(lr.relative_to(REPO)),
                    "--realesrgan",
                    str(rs.relative_to(REPO)),
                    "--out",
                    str(out_flow.relative_to(REPO)),
                ]
                print(" ".join(cmd))
                subprocess.run(cmd, cwd=str(REPO), check=True)
            else:
                print(f"skip flow {out_flow.name}")

        if not args.skip_blend and temporal is not None and temporal.is_file():
            out_blend = out_dir / f"{stem}_temporal_gen_blend_x4.mp4"
            if not (out_blend.exists() and out_blend.stat().st_size > 0):
                cmd = [
                    sys.executable,
                    "-m",
                    mod_blend,
                    "--temporal",
                    str(temporal.relative_to(REPO)),
                    "--realesrgan",
                    str(rs.relative_to(REPO)),
                    "--out",
                    str(out_blend.relative_to(REPO)),
                ]
                print(" ".join(cmd))
                subprocess.run(cmd, cwd=str(REPO), check=True)
            else:
                print(f"skip blend {out_blend.name}")
        elif not args.skip_blend:
            print(f"skip blend for {stem}: no temporal Part1 output")


if __name__ == "__main__":
    main()
