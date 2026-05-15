"""Run standalone BasicVSR on the project video dataset.

This script is intentionally resumable: by default it skips outputs that already
exist, so long BasicVSR runs can be continued after an interruption.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from time import perf_counter

import torch

from part2_sota.infer_basicvsr_standalone import (
    infer_chunks,
    load_basicvsr,
    read_video,
    write_video,
)


def collect_inputs(input_dir: Path, include_wild: bool, wild_path: Path) -> list[Path]:
    videos = sorted(input_dir.glob("*.mp4"))
    if include_wild and wild_path.exists():
        videos.append(wild_path)
    return videos


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch BasicVSR inference for all existing project videos.")
    parser.add_argument("--input_dir", default="dataset/inputs_mp4")
    parser.add_argument("--wild", default="dataset/wild_real_lr.mp4")
    parser.add_argument("--no_wild", action="store_true")
    parser.add_argument("--out_dir", default="part2_sota/outputs/basicvsr")
    parser.add_argument("--ckpt", default="weights/basicvsr/basicvsr_vimeo90k_bi.pth")
    parser.add_argument("--chunk_size", type=int, default=10)
    parser.add_argument("--overlap", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit for smoke tests.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--manifest", default="part2_sota/outputs/basicvsr_manifest.csv")
    parser.add_argument("--evaluate", action="store_true", help="Run metrics evaluation after inference.")
    parser.add_argument("--metric_csv", default="output/tables/metric.csv")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    videos = collect_inputs(input_dir, include_wild=not args.no_wild, wild_path=Path(args.wild))
    if args.limit > 0:
        videos = videos[: args.limit]
    if not videos:
        raise RuntimeError(f"No mp4 files found in {input_dir}")

    device = torch.device(args.device)
    model = load_basicvsr(args.ckpt, device)
    rows: list[dict[str, str]] = []

    for idx, inp in enumerate(videos, start=1):
        out = out_dir / inp.name
        if out.exists() and out.stat().st_size > 0 and not args.overwrite:
            print(f"[{idx}/{len(videos)}] skip existing {out.name}")
            rows.append({"input": str(inp), "output": str(out), "status": "skipped", "seconds": "0.00"})
            continue

        print(f"[{idx}/{len(videos)}] BasicVSR {inp} -> {out}")
        start = perf_counter()
        try:
            frames, fps = read_video(str(inp))
            out_frames = infer_chunks(model, frames, device, args.chunk_size, args.overlap)
            write_video(str(out), out_frames, fps)
            seconds = perf_counter() - start
            rows.append({"input": str(inp), "output": str(out), "status": "ok", "seconds": f"{seconds:.2f}"})
            print(f"  wrote {out.name}: {len(out_frames)} frames in {seconds:.1f}s")
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() and device.type == "cuda":
                torch.cuda.empty_cache()
            rows.append({"input": str(inp), "output": str(out), "status": f"failed: {exc}", "seconds": "0.00"})
            print(f"  failed: {exc}")

    manifest = Path(args.manifest)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["input", "output", "status", "seconds"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote manifest: {manifest}")

    if args.evaluate:
        import subprocess
        import sys

        cmd = [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "scripts" / "evaluate_project.py"),
            "--metric_csv",
            args.metric_csv,
            "--skip_fid",
        ]
        print("Running evaluation:", " ".join(cmd))
        subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()
