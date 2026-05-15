"""Batch Direction A on all clips that have bicubic + Real-ESRGAN outputs."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def find_bicubic(stem: str, part1_root: Path) -> Path | None:
    for p in part1_root.rglob("*.mp4"):
        if p.stem == f"{stem}_part1_bicubic_x4" or p.stem.startswith(f"{stem}_part1_bicubic_"):
            return p
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch rectified-flow refinement (Direction A).")
    ap.add_argument("--inputs_dir", default="dataset/inputs_mp4")
    ap.add_argument("--wild_lr", default="dataset/wild_real_lr.mp4")
    ap.add_argument("--part1_root", default="part1_baseline/outputs")
    ap.add_argument("--realesrgan_dir", default="part2_sota/outputs/realesrgan")
    ap.add_argument("--out_dir", default="part3_exploration/outputs/direction_a_flow_matching")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip_wild", action="store_true")
    args = ap.parse_args()

    out_dir = REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    part1 = REPO / args.part1_root
    rs_dir = REPO / args.realesrgan_dir
    inp_dir = REPO / args.inputs_dir

    jobs: list[tuple[str, Path, Path]] = []

    if not args.skip_wild:
        wild_stem = Path(args.wild_lr).stem
        bic = REPO / f"part1_baseline/outputs/{wild_stem}/{wild_stem}_part1_bicubic_x4.mp4"
        rs = REPO / "part2_sota/outputs/wild_real_lr/realesrgan_x4.mp4"
        if bic.is_file() and rs.is_file():
            jobs.append((wild_stem, bic, rs))

    if inp_dir.is_dir():
        for mp4 in sorted(inp_dir.glob("*.mp4")):
            if mp4.name.startswith("wild"):
                continue
            stem = mp4.stem
            rs = rs_dir / f"{stem}.mp4"
            if not rs.is_file():
                continue
            bic = find_bicubic(stem, part1)
            if bic is None:
                print(f"Skip {stem}: no bicubic Part1 output found under {part1}")
                continue
            jobs.append((stem, bic, rs))

    if args.limit > 0:
        jobs = jobs[: args.limit]

    if not jobs:
        raise RuntimeError("No jobs. Run Part1 bicubic and Part2 Real-ESRGAN first.")

    mod = "part3_exploration.direction_a_flow_matching.rectified_refinement"
    for stem, bic, rs in jobs:
        out = out_dir / f"{stem}_rectified_flow_x4.mp4"
        if out.exists() and out.stat().st_size > 0:
            print(f"skip existing {out.name}")
            continue
        cmd = [
            sys.executable,
            "-m",
            mod,
            "--bicubic",
            str(bic.relative_to(REPO)),
            "--realesrgan",
            str(rs.relative_to(REPO)),
            "--out",
            str(out.relative_to(REPO)),
            "--steps",
            str(args.steps),
        ]
        print(" ".join(cmd))
        subprocess.run(cmd, cwd=str(REPO), check=True)


if __name__ == "__main__":
    main()
