"""BasicVSR / BasicVSR++ inference via MMagic (optional dependency)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

if not hasattr(np, "bool8"):
    # MMagic 1.2.0 still references np.bool8, which was removed in NumPy 2.x.
    np.bool8 = np.bool_


def _default_out_dir() -> str:
    return str(Path(__file__).resolve().parent / "outputs" / "basicvsr")


def build_mmagic_inferencer(model_name: str, model_setting: int | None = 4) -> Any:
    """构造 MMagicInferencer；兼容部分版本无 ``model_setting`` 参数。"""
    from mmagic.apis import MMagicInferencer

    try:
        if model_setting is not None:
            return MMagicInferencer(model_name=model_name, model_setting=model_setting)
    except TypeError:
        pass
    return MMagicInferencer(model_name=model_name)


def infer_basicvsr_video(editor: Any, video_path: str, out_dir: str) -> Any:
    """
    对单个视频做 BasicVSR 类推理。
    优先 ``.infer(video=..., result_out_dir=...)``（MMagic 文档），
    再尝试 ``.infer(inputs=..., result_out_dir=...)``，
    最后尝试 ``editor(inputs=..., out_dir=...)``（与历史示例一致的可调用形式）。
    """
    os.makedirs(out_dir, exist_ok=True)
    vp = str(Path(video_path).resolve())
    od = str(Path(out_dir).resolve())
    infer = getattr(editor, "infer", None)
    if callable(infer):
        try:
            return infer(video=vp, result_out_dir=od)
        except TypeError:
            try:
                return infer(inputs=vp, result_out_dir=od)
            except TypeError:
                pass
    try:
        return editor(inputs=vp, out_dir=od)
    except TypeError as e:
        raise RuntimeError(
            "无法调用 MMagicInferencer：请检查 mmagic / mmcv 版本与 part2_sota/README.md 是否一致。"
        ) from e


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True)
    ap.add_argument(
        "--out_dir",
        type=str,
        default=_default_out_dir(),
        help="MMagic 输出目录，默认 part2_sota/outputs/basicvsr/",
    )
    ap.add_argument(
        "--model_name",
        type=str,
        default="basicvsr_pp",
        help="MMagic model name, e.g. basicvsr_pp or basicvsr",
    )
    args = ap.parse_args()

    try:
        inferencer = build_mmagic_inferencer(args.model_name)
    except ImportError:
        print(
            "MMagic is not installed. Install with: pip install mmagic mmengine\n"
            "See part2_sota/README.md for MMCV wheel.",
            file=sys.stderr,
        )
        sys.exit(1)
    result = infer_basicvsr_video(inferencer, args.input, args.out_dir)
    print(f"Done. Output dir: {args.out_dir}, result: {result}")


if __name__ == "__main__":
    main()
