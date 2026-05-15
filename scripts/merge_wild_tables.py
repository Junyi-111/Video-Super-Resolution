"""Merge Wild fidelity and perceptual metrics into a report-ready Markdown table."""

from __future__ import annotations

import csv
from pathlib import Path


def read_csv(path: str) -> dict[str, dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return {row["method"]: row for row in csv.DictReader(f)}


def main() -> None:
    fidelity = read_csv("output/tables/wild_metrics.csv")
    perceptual = read_csv("output/tables/wild_lpips_metrics.csv")
    methods = [
        "LR input",
        "Bicubic",
        "Lanczos",
        "Temporal+Unsharp",
        "Real-ESRGAN",
        "BasicVSR",
        "Part3 Hybrid",
        "Part3 Flow-Stabilized",
        "Part3 Uncertainty-Adaptive",
    ]

    lines = [
        "| Method | PSNR (higher) | SSIM (higher) | LPIPS (lower) | tLPIPS (lower) | tLPIPS proxy (lower) | Temporal MAE (lower) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in methods:
        f = fidelity[method]
        p = perceptual[method]
        lines.append(
            f"| {method} | {f['psnr']} | {f['ssim']} | {p['lpips']} | "
            f"{p.get('tlpips', '')} | {p['tlpips_proxy']} | {f['temporal_mae']} |"
        )

    out = Path("output/tables/wild_metrics_table.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
