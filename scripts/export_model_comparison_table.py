"""Merge metric CSVs into one report-ready model comparison table (Markdown + CSV)."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

PART3_LEGACY = {"Part3 Hybrid", "Part3 Flow-Stabilized", "Part3 Uncertainty-Adaptive"}

PART3_DISPLAY = {
    "part3_A_rectified_flow": "Part 3-A (Rectified-flow)",
    "part3_B_flow_stabilized": "Part 3-B (Flow-stabilized)",
    "part3_B_temporal_gen_blend": "Part 3-B (Temporal + Real-ESRGAN blend)",
    "part3_C_uncertainty_adaptive": "Part 3-C (Uncertainty-adaptive)",
    "part3_D_streaming_distilled": "Part 3-D (Streaming distilled diffusion)",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def ffloat(s: str) -> float | None:
    s = (s or "").strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def mean_vals(values: list[float]) -> str:
    return f"{sum(values) / len(values):.4f}" if values else ""


def md_row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def main() -> None:
    parser = argparse.ArgumentParser(description="Export unified model metric comparison.")
    parser.add_argument("--metric_csv", default="output/tables/metric.csv")
    parser.add_argument("--part3_csv", default="output/tables/part3_metrics_per_clip.csv")
    parser.add_argument("--out_md", default="output/tables/model_comparison_table.md")
    parser.add_argument("--out_csv", default="output/tables/model_comparison_summary.csv")
    args = parser.parse_args()

    metric_path = REPO / args.metric_csv
    part3_path = REPO / args.part3_csv
    if not metric_path.is_file():
        raise FileNotFoundError(f"Missing {metric_path}")
    if not part3_path.is_file():
        raise FileNotFoundError(f"Missing {part3_path}")

    base_rows = read_csv(metric_path)
    wild_baseline = [
        r for r in base_rows if r.get("dataset") == "wild" and r.get("method") not in PART3_LEGACY
    ]

    p3_all = read_csv(part3_path)
    wild_part3 = [r for r in p3_all if r.get("clip") == "wild_real_lr" and r.get("method") in PART3_DISPLAY]

    order_base = ["LR input", "Bicubic", "Lanczos", "Temporal+Unsharp", "Real-ESRGAN", "BasicVSR"]
    by_method = {r["method"]: r for r in wild_baseline}

    table_rows: list[dict[str, str]] = []
    for m in order_base:
        if m in by_method:
            r = by_method[m]
            table_rows.append(
                {
                    "group": "Wild (GT: wild_real.mp4)",
                    "model": m,
                    "psnr": r.get("psnr", ""),
                    "ssim": r.get("ssim", ""),
                    "lpips": r.get("lpips", ""),
                    "tlpips": r.get("tlpips", ""),
                    "fid": r.get("fid", ""),
                    "temporal_mae": r.get("temporal_mae", ""),
                    "frames": r.get("frames", ""),
                }
            )

    for key in [
        "part3_A_rectified_flow",
        "part3_B_flow_stabilized",
        "part3_B_temporal_gen_blend",
        "part3_C_uncertainty_adaptive",
        "part3_D_streaming_distilled",
    ]:
        r = next((x for x in wild_part3 if x.get("method") == key), None)
        if r:
            table_rows.append(
                {
                    "group": "Wild (GT: wild_real.mp4)",
                    "model": PART3_DISPLAY[key],
                    "psnr": r.get("psnr", ""),
                    "ssim": r.get("ssim", ""),
                    "lpips": r.get("lpips", ""),
                    "tlpips": r.get("tlpips", ""),
                    "fid": r.get("fid", ""),
                    "temporal_mae": r.get("temporal_mae", ""),
                    "frames": r.get("frames", ""),
                }
            )

    agg_gt: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    agg_nogt_tlpips: dict[str, list[float]] = defaultdict(list)

    for r in p3_all:
        m = r.get("method", "")
        if not m:
            continue
        gt_ok = str(r.get("has_gt", "")).lower() == "true"
        if gt_ok:
            for field in ("psnr", "ssim", "lpips", "tlpips", "fid", "temporal_mae"):
                v = ffloat(r.get(field, ""))
                if v is not None:
                    agg_gt[m][field].append(v)
        else:
            v = ffloat(r.get("tlpips", ""))
            if v is not None:
                agg_nogt_tlpips[m].append(v)

    agg_summary: list[dict[str, str]] = []
    for key in [
        "part3_A_rectified_flow",
        "part3_B_flow_stabilized",
        "part3_B_temporal_gen_blend",
        "part3_C_uncertainty_adaptive",
        "part3_D_streaming_distilled",
    ]:
        a = agg_gt.get(key, {})
        nr = agg_nogt_tlpips.get(key, [])
        n_gt_clip = sum(
            1 for row in p3_all if row.get("method") == key and str(row.get("has_gt", "")).lower() == "true"
        )
        agg_summary.append(
            {
                "model": PART3_DISPLAY[key],
                "clips_with_gt": str(n_gt_clip),
                "mean_psnr_gt": mean_vals(a.get("psnr", [])),
                "mean_ssim_gt": mean_vals(a.get("ssim", [])),
                "mean_lpips_gt": mean_vals(a.get("lpips", [])),
                "mean_tlpips_gt": mean_vals(a.get("tlpips", [])),
                "mean_fid_gt": mean_vals(a.get("fid", [])),
                "mean_temporal_mae_gt": mean_vals(a.get("temporal_mae", [])),
                "clips_without_gt": str(len(nr)),
                "mean_tlpips_noref": mean_vals(nr),
            }
        )

    out_md = REPO / args.out_md
    out_csv = REPO / args.out_csv
    out_md.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# 模型指标对比（自动生成）",
        "",
        "说明：**PSNR / SSIM 越高越好**；**LPIPS / tLPIPS / FID / Temporal MAE 越低越好**。",
        "Part 3 数据集汇总：**有 GT** 片段上对各指标取均值（常为 Wild）；**无 GT** 片段单独报告 **tLPIPS（无参考）** clip 均值，勿与有参考 tLPIPS 混比。",
        "",
        "## 1. Wild 视频（伪 GT：wild_real.mp4）：Part1 / Part2 / Part3 同窗对比",
        "",
        md_row(["Model", "PSNR↑", "SSIM↑", "LPIPS↓", "tLPIPS↓", "FID↓", "Temp. MAE↓", "Frames"]),
        md_row(["---"] * 8),
    ]
    for tr in table_rows:
        lines.append(
            md_row(
                [
                    tr["model"],
                    tr["psnr"] or "—",
                    tr["ssim"] or "—",
                    tr["lpips"] or "—",
                    tr["tlpips"] or "—",
                    tr["fid"] or "—",
                    tr["temporal_mae"] or "—",
                    tr["frames"] or "—",
                ]
            )
        )

    lines.extend(
        [
            "",
            "## 2. Part 3 各方向 — 全数据集汇总（有 GT / 无 GT 分列）",
            "",
            md_row(
                [
                    "Model",
                    "# clips 有 GT",
                    "mean PSNR↑",
                    "mean SSIM↑",
                    "mean LPIPS↓",
                    "mean tLPIPS↓(有 GT)",
                    "mean FID↓",
                    "mean Temp.MAE↓",
                    "# clips 无 GT",
                    "mean tLPIPS(nr)↓",
                ]
            ),
            md_row(["---"] * 10),
        ]
    )
    for ar in agg_summary:
        lines.append(
            md_row(
                [
                    ar["model"],
                    ar["clips_with_gt"],
                    ar["mean_psnr_gt"] or "—",
                    ar["mean_ssim_gt"] or "—",
                    ar["mean_lpips_gt"] or "—",
                    ar["mean_tlpips_gt"] or "—",
                    ar["mean_fid_gt"] or "—",
                    ar["mean_temporal_mae_gt"] or "—",
                    ar["clips_without_gt"],
                    ar["mean_tlpips_noref"] or "—",
                ]
            )
        )

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "scope",
                "model",
                "psnr",
                "ssim",
                "lpips",
                "tlpips",
                "fid",
                "temporal_mae",
                "frames",
                "clips_with_gt",
                "mean_psnr_gt",
                "mean_ssim_gt",
                "mean_lpips_gt",
                "mean_tlpips_gt",
                "mean_fid_gt",
                "mean_temporal_mae_gt",
                "clips_without_gt",
                "mean_tlpips_noref",
            ],
        )
        w.writeheader()
        for tr in table_rows:
            w.writerow(
                {
                    "scope": "wild",
                    "model": tr["model"],
                    "psnr": tr["psnr"],
                    "ssim": tr["ssim"],
                    "lpips": tr["lpips"],
                    "tlpips": tr["tlpips"],
                    "fid": tr["fid"],
                    "temporal_mae": tr["temporal_mae"],
                    "frames": tr["frames"],
                    "clips_with_gt": "",
                    "mean_psnr_gt": "",
                    "mean_ssim_gt": "",
                    "mean_lpips_gt": "",
                    "mean_tlpips_gt": "",
                    "mean_fid_gt": "",
                    "mean_temporal_mae_gt": "",
                    "clips_without_gt": "",
                    "mean_tlpips_noref": "",
                }
            )
        for ar in agg_summary:
            w.writerow(
                {
                    "scope": "part3_agg",
                    "model": ar["model"],
                    "psnr": "",
                    "ssim": "",
                    "lpips": "",
                    "tlpips": "",
                    "fid": "",
                    "temporal_mae": "",
                    "frames": "",
                    "clips_with_gt": ar["clips_with_gt"],
                    "mean_psnr_gt": ar["mean_psnr_gt"],
                    "mean_ssim_gt": ar["mean_ssim_gt"],
                    "mean_lpips_gt": ar["mean_lpips_gt"],
                    "mean_tlpips_gt": ar["mean_tlpips_gt"],
                    "mean_fid_gt": ar["mean_fid_gt"],
                    "mean_temporal_mae_gt": ar["mean_temporal_mae_gt"],
                    "clips_without_gt": ar["clips_without_gt"],
                    "mean_tlpips_noref": ar["mean_tlpips_noref"],
                }
            )

    print(f"Wrote {out_md}")
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
