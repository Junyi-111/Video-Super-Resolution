"""课程数据 `dataset/`：兼容多种目录命名与嵌套结构，递归发现 REDS / Vimeo 帧目录与 wild 视频。"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from PIL import Image

# 常见课程包 / NTIRE 解压后的顶层目录名（不假定固定为 REDS-sample/000）
REDS_ROOT_NAMES = (
    "REDS-sample",
    "reds_sample",
    "REDS_sample",
    "REDS",
    "reds",
    "REDS_sample_val",
    "REDS4K",
)
# 在 REDS 顶层目录下常见的「帧再分子目录」（NTIRE / 官方包常见）
REDS_INNER_SHARP_DIRS = (
    "train/train_sharp",
    "train_sharp",
    "val/sharp",
    "sharp",
    "train/train_sharp_blur",
)
VIMEO_ROOT_NAMES = (
    "vimeo-RL",
    "vimeo_rl",
    "Vimeo-90k",
    "vimeo_septuplet",
    "vimeo90k",
    "vimeo_90k",
)
WILD_DIR_NAMES = ("wild", "Wild", "wild_video", "Wild_video")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_dataset_dir() -> Path:
    return repo_root() / "dataset"


def _sorted_image_files(d: Path) -> list[Path]:
    """目录 d 下直接子级中的 .png/.jpg/.jpeg，按文件名排序。"""
    if not d.is_dir():
        return []
    exts = {".png", ".jpg", ".jpeg"}
    out = [p for p in d.iterdir() if p.suffix.lower() in exts and p.is_file()]
    return sorted(out, key=lambda p: p.name)


def _sorted_pngs(d: Path) -> list[Path]:
    """仅 PNG（与旧逻辑兼容）。"""
    if not d.is_dir():
        return []
    out = [p for p in d.iterdir() if p.suffix.lower() == ".png" and p.is_file()]
    return sorted(out, key=lambda p: p.name)


def read_png_rgb(path: Path) -> np.ndarray:
    return np.ascontiguousarray(np.array(Image.open(path).convert("RGB")))


def list_reds_hr_frame_paths(clip_dir: Path) -> list[Path]:
    """REDS clip 目录内、数字文件名的帧路径列表（png/jpg/jpeg）。"""
    return [p for p in _sorted_image_files(clip_dir) if p.stem.isdigit()]


def read_reds_clip_frames(clip_dir: Path) -> list[np.ndarray]:
    """REDS 风格：目录内帧文件名为纯数字（可 png/jpg）。"""
    return [read_png_rgb(p) for p in list_reds_hr_frame_paths(clip_dir)]


def read_vimeo_subclip_frames(subclip_dir: Path) -> list[np.ndarray]:
    """Vimeo 七帧：im1.png … im7.png（大小写不敏感）。"""
    paths = [p for p in subclip_dir.iterdir() if p.suffix.lower() == ".png" and p.is_file()]

    def sort_key(p: Path) -> tuple[int, str]:
        m = re.match(r"^im(\d+)\.png$", p.name, re.I)
        if m:
            return (int(m.group(1)), p.name.lower())
        return (999, p.name.lower())

    paths = sorted(paths, key=sort_key)
    return [read_png_rgb(p) for p in paths]


def _is_reds_numeric_clip_dir(d: Path) -> bool:
    """是否为「一层内全是数字文件名的帧」的 REDS clip 目录。"""
    imgs = _sorted_image_files(d)
    if len(imgs) < 1:
        return False
    return all(p.stem.isdigit() for p in imgs)


def _is_vimeo_im_clip_dir(d: Path) -> bool:
    if not d.is_dir():
        return False
    return bool(list(d.glob("im*.png"))) or bool(list(d.glob("im*.PNG")))


def _discover_reds_clips_under(reds_root: Path) -> list[Path]:
    """从 REDS 数据根目录递归找出所有「叶子」数字帧目录。"""
    found: list[Path] = []

    def walk(d: Path) -> None:
        if not d.is_dir():
            return
        if _is_reds_numeric_clip_dir(d):
            found.append(d)
            return
        for sub in sorted(d.iterdir(), key=lambda x: x.name):
            if sub.is_dir():
                walk(sub)

    walk(reds_root)
    return found


def _discover_vimeo_subclips_under(v_root: Path) -> list[Path]:
    found: list[Path] = []

    def walk(d: Path) -> None:
        if not d.is_dir():
            return
        if _is_vimeo_im_clip_dir(d):
            found.append(d)
            return
        for sub in sorted(d.iterdir(), key=lambda x: x.name):
            if sub.is_dir():
                walk(sub)

    walk(v_root)
    return found


def _reds_search_roots(dataset_dir: Path) -> list[Path]:
    """可能含 REDS 帧的目录列表（去重）。"""
    roots: list[Path] = []
    seen: set[str] = set()
    for name in REDS_ROOT_NAMES:
        p = dataset_dir / name
        if not p.is_dir():
            continue
        for cand in (p, *(p / sub for sub in REDS_INNER_SHARP_DIRS if (p / sub).is_dir())):
            key = str(cand.resolve())
            if key not in seen:
                seen.add(key)
                roots.append(cand)
    return roots


def _first_existing_vimeo_root(dataset_dir: Path) -> Path | None:
    for name in VIMEO_ROOT_NAMES:
        p = dataset_dir / name
        if p.is_dir():
            return p
    return None


def iter_reds_sample_clips(dataset_dir: Path | None = None) -> list[Path]:
    root = dataset_dir or default_dataset_dir()
    clips: list[Path] = []
    seen: set[str] = set()
    for reds_root in _reds_search_roots(root):
        for c in _discover_reds_clips_under(reds_root):
            k = str(c.resolve())
            if k not in seen:
                seen.add(k)
                clips.append(c)
    return clips


def iter_vimeo_rl_subclips(dataset_dir: Path | None = None) -> list[Path]:
    root = dataset_dir or default_dataset_dir()
    v_root = _first_existing_vimeo_root(root)
    if v_root is None:
        return []
    return _discover_vimeo_subclips_under(v_root)


def iter_wild_videos(dataset_dir: Path | None = None) -> list[Path]:
    """Wild：dataset/wild/（或 Wild 等）下的 mp4/avi/mov。"""
    root = dataset_dir or default_dataset_dir()
    vids: list[Path] = []
    for name in WILD_DIR_NAMES:
        w = root / name
        if not w.is_dir():
            continue
        for p in sorted(w.iterdir(), key=lambda x: x.name):
            if p.is_file() and p.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv"):
                vids.append(p)
    return vids


def collect_lr_dataset_jobs(roots: list[Path]) -> list[tuple[Path, Path]]:
    """
    收集低码率批量任务列表 ``(data_root, input_path)``。

    - ``input_path`` 为 **视频文件**（在各 root 下递归匹配 mp4/avi/mov/mkv），或
    - **REDS / Vimeo 帧目录**（通过 ``iter_reds_sample_clips`` / ``iter_vimeo_rl_subclips`` 发现）。

    ``data_root`` 用于输出目录镜像（见 ``batch_job_out_dir``）。同一文件路径只出现一次。
    """
    jobs: list[tuple[Path, Path]] = []
    seen: set[str] = set()
    for root in roots:
        root = root.resolve()
        if not root.is_dir():
            continue
        for pat in ("**/*.mp4", "**/*.avi", "**/*.mov", "**/*.mkv"):
            for v in root.glob(pat):
                if not v.is_file():
                    continue
                k = str(v.resolve())
                if k in seen:
                    continue
                seen.add(k)
                jobs.append((root, v))
        for clip in iter_reds_sample_clips(root):
            k = str(clip.resolve())
            if k in seen:
                continue
            seen.add(k)
            jobs.append((root, clip))
        for sub in iter_vimeo_rl_subclips(root):
            k = str(sub.resolve())
            if k in seen:
                continue
            seen.add(k)
            jobs.append((root, sub))
    return jobs


def default_scan_roots(root: Path | None = None) -> list[Path]:
    """默认扫描 ``dataset/`` 与 ``data/raw/``（存在则加入）。``root`` 为仓库根，默认自动推断。"""
    base = root if root is not None else repo_root()
    out: list[Path] = []
    for rel in ("dataset", Path("data") / "raw"):
        p = (base / rel).resolve()
        if p.is_dir():
            out.append(p)
    return out


def batch_job_out_dir(batch_root: Path, data_root: Path, inp: Path) -> Path:
    """与 ``process_mandatory`` 一致：每个任务单独子目录，避免不同来源同名冲突。"""
    rel = inp.relative_to(data_root)
    if inp.is_file():
        return batch_root / rel.parent / inp.stem
    return batch_root / rel


def is_probably_frame_clip(path: Path) -> bool:
    """路径是否为可直接送入 `read_clip_frames` 的帧目录。"""
    if not path.is_dir():
        return False
    if _is_reds_numeric_clip_dir(path):
        return True
    if _is_vimeo_im_clip_dir(path):
        return True
    imgs = _sorted_image_files(path)
    return len(imgs) >= 1


def read_clip_frames(clip_dir: Path) -> list[np.ndarray]:
    """REDS 数字帧名、Vimeo im*、或其它已排序图像序列。"""
    if _is_reds_numeric_clip_dir(clip_dir):
        return read_reds_clip_frames(clip_dir)
    if _is_vimeo_im_clip_dir(clip_dir):
        return read_vimeo_subclip_frames(clip_dir)
    imgs = _sorted_image_files(clip_dir)
    if imgs:
        return [read_png_rgb(p) for p in imgs]
    return []


def describe_clip(clip_dir: Path) -> str:
    try:
        rel = clip_dir.resolve().relative_to(default_dataset_dir().resolve())
        return str(rel).replace("\\", "/").replace("/", "_")
    except ValueError:
        return clip_dir.name


def dataset_inventory(
    dataset_dir: Path | None = None,
) -> tuple[Path, list[Path], list[Path], list[Path]]:
    """(dataset_root, reds_clips, vimeo_subclips, wild_videos)。"""
    root = (dataset_dir or default_dataset_dir()).resolve()
    return (
        root,
        iter_reds_sample_clips(root),
        iter_vimeo_rl_subclips(root),
        iter_wild_videos(root),
    )


def format_dataset_hints(dataset_dir: Path | None = None) -> str:
    """人类可读：实际发现了哪些输入路径。"""
    root, reds, vim, wild = dataset_inventory(dataset_dir)
    lines = [f"dataset 根目录: {root}", ""]
    if not root.is_dir():
        lines.append("（目录不存在，请先创建 dataset/ 并放入数据）")
        return "\n".join(lines)

    lines.append(f"REDS 帧目录（共 {len(reds)} 个，取其一作为 --input）:")
    for i, p in enumerate(reds[:20]):
        lines.append(f"  [{i}] {p}")
    if len(reds) > 20:
        lines.append(f"  ... 另有 {len(reds) - 20} 个")
    if not reds:
        lines.append("  （未发现：请检查是否在以下目录之一内存放 REDS，且帧文件名为纯数字：")
        lines.append(f"    {', '.join(REDS_ROOT_NAMES)}）")

    lines.append("")
    lines.append(f"Vimeo 子片段（共 {len(vim)} 个）:")
    for i, p in enumerate(vim[:20]):
        lines.append(f"  [{i}] {p}")
    if len(vim) > 20:
        lines.append(f"  ... 另有 {len(vim) - 20} 个")
    if not vim:
        lines.append("  （未发现：目录名可为 " + ", ".join(f'"{n}"' for n in VIMEO_ROOT_NAMES[:4]) + " 等）")

    lines.append("")
    lines.append(f"Wild 视频（共 {len(wild)} 个）:")
    for i, p in enumerate(wild):
        lines.append(f"  [{i}] {p}")
    if not wild:
        lines.append(f"  （未发现：请将 mp4 等放入 {root / 'wild'} 或下列之一：{', '.join(WILD_DIR_NAMES)}）")

    lines.append("")
    if wild:
        ex = wild[0]
        lines.append("示例（Wild）:")
        lines.append(f'  python -m part1_baseline.run_part1 --input "{ex}" --scale 4 --mode bicubic')
    if reds:
        ex = reds[0]
        lines.append("示例（REDS 某一 clip 目录）:")
        lines.append(f'  python -m part1_baseline.run_part1 --input "{ex}" --scale 4 --mode bicubic --clip_fps 24')
    if vim:
        ex = vim[0]
        lines.append("示例（Vimeo 某一子目录）:")
        lines.append(f'  python -m part1_baseline.run_part1 --input "{ex}" --scale 4 --mode bicubic --clip_fps 24')
    return "\n".join(lines)
