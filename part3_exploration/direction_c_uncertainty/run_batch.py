"""Batch Direction C on all clips with LR + Real-ESRGAN + BasicVSR."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch Direction C uncertainty fusion.")
    ap.add_argument("--inputs_dir", default="dataset/inputs_mp4")
    ap.add_argument("--wild_lr", default="dataset/wild_real_lr.mp4")
    ap.add_argument("--realesrgan_dir", default="part2_sota/outputs/realesrgan")
    ap.add_argument("--basicvsr_dir", default="part2_sota/outputs/basicvsr")
    ap.add_argument("--out_dir", default="part3_exploration/outputs/direction_c_uncertainty")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip_wild", action="store_true")
    args = ap.parse_args()

    out_dir = REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rs_dir = REPO / args.realesrgan_dir
    bv_dir = REPO / args.basicvsr_dir
    inp_dir = REPO / args.inputs_dir

    jobs: list[tuple[str, Path, Path, Path]] = []

    if not args.skip_wild:
        w = Path(args.wild_lr).stem
        lr = REPO / args.wild_lr
        rs = REPO / "part2_sota/outputs/wild_real_lr/realesrgan_x4.mp4"
        bv = REPO / "part2_sota/outputs/basicvsr/wild_real_lr.mp4"
        if lr.is_file() and rs.is_file() and bv.is_file():
            jobs.append((w, lr, rs, bv))

    if inp_dir.is_dir():
        for mp4 in sorted(inp_dir.glob("*.mp4")):
            if mp4.name.startswith("wild"):
                continue
            stem = mp4.stem
            rs = rs_dir / f"{stem}.mp4"
            bv = bv_dir / f"{stem}.mp4"
            if rs.is_file() and bv.is_file():
                jobs.append((stem, mp4, rs, bv))

    if args.limit > 0:
        jobs = jobs[: args.limit]

    mod = "part3_exploration.direction_c_uncertainty.run_adaptive"
    for stem, lr, rs, bv in jobs:
        out = out_dir / f"{stem}_uncertainty_adaptive_x4.mp4"
        mask = out_dir / f"{stem}_uncertainty_alpha.mp4"
        if out.exists() and out.stat().st_size > 0:
            print(f"skip existing {out.name}")
            continue
        cmd = [
            sys.executable,
            "-m",
            mod,
            "--lr",
            str(lr.relative_to(REPO)),
            "--realesrgan",
            str(rs.relative_to(REPO)),
            "--basicvsr",
            str(bv.relative_to(REPO)),
            "--out",
            str(out.relative_to(REPO)),
            "--mask_out",
            str(mask.relative_to(REPO)),
        ]
        print(" ".join(cmd))
        subprocess.run(cmd, cwd=str(REPO), check=True)


if __name__ == "__main__":
    main()
