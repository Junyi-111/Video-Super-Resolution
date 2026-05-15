"""Train SRCNN：支持 LR/HR 成对文件夹，或官方 `dataset/`（REDS-sample + vimeo-RL）合成下采样对。"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from part1_baseline.clip_io import (
    default_dataset_dir,
    iter_reds_sample_clips,
    iter_vimeo_rl_subclips,
    list_reds_hr_frame_paths,
)
from part1_baseline.srcnn import SRCNN


class PairDataset(Dataset):
    """LR/HR pairs: HR must match bicubic(LR) spatial size (typical SRCNN setup)."""

    def __init__(self, lr_dir: str, hr_dir: str, scale: int) -> None:
        self.lr_dir = Path(lr_dir)
        self.hr_dir = Path(hr_dir)
        self.scale = scale
        self.names = sorted(
            [p.name for p in self.lr_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")]
        )
        if not self.names:
            raise FileNotFoundError(f"No images in {lr_dir}")

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, i: int):
        name = self.names[i]
        lr = Image.open(self.lr_dir / name).convert("RGB")
        hr = Image.open(self.hr_dir / name).convert("RGB")
        w, h = lr.size
        bic = lr.resize((w * self.scale, h * self.scale), Image.BICUBIC)
        if bic.size != hr.size:
            hr = hr.resize(bic.size, Image.BICUBIC)
        bic_t = torch.from_numpy(np.array(bic).astype("float32") / 255.0).permute(2, 0, 1)
        hr_t = torch.from_numpy(np.array(hr).astype("float32") / 255.0).permute(2, 0, 1)
        return bic_t, hr_t


def _collect_official_hr_paths(dataset_root: Path, source: str) -> list[Path]:
    paths: list[Path] = []
    if source in ("reds", "both"):
        for clip in iter_reds_sample_clips(dataset_root):
            paths.extend(list_reds_hr_frame_paths(clip))
    if source in ("vimeo", "both"):
        for sub in iter_vimeo_rl_subclips(dataset_root):

            def _im_index(p: Path) -> int:
                m = re.search(r"(\d+)", p.stem)
                return int(m.group(1)) if m else 0

            paths.extend(sorted(sub.glob("im*.png"), key=_im_index))
    return paths


class SyntheticBicubicDownscaleDataset(Dataset):
    """将官方帧视作 HR，双三次下采样得到 LR，再双三次上采样得到 SRCNN 输入（标准合成对）。"""

    def __init__(self, hr_paths: list[Path], scale: int, patch_hr: int) -> None:
        if not hr_paths:
            raise FileNotFoundError("未找到任何训练帧，请检查 --dataset_root 与 --official_source。")
        self.hr_paths = hr_paths
        self.scale = scale
        self.patch_hr = max(patch_hr, scale * 8)

    def __len__(self) -> int:
        return len(self.hr_paths)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        s = self.scale
        hr = Image.open(self.hr_paths[i]).convert("RGB")
        w, h = hr.size
        w2, h2 = (w // s) * s, (h // s) * s
        if w2 < s or h2 < s:
            raise ValueError(f"图像过小: {self.hr_paths[i]} ({w}x{h}), scale={s}")
        if (w2, h2) != (w, h):
            left = (w - w2) // 2
            top = (h - h2) // 2
            hr = hr.crop((left, top, left + w2, top + h2))

        ph = min(self.patch_hr, w2, h2)
        ph = (ph // s) * s
        if ph < s:
            raise ValueError(f"patch 过小: {self.hr_paths[i]} 在 scale={s} 下无法裁块")
        if ph < w2 or ph < h2:
            max_t = h2 - ph
            max_l = w2 - ph
            top = int(np.random.randint(0, max_t + 1)) if max_t > 0 else 0
            left = int(np.random.randint(0, max_l + 1)) if max_l > 0 else 0
            hr = hr.crop((left, top, left + ph, top + ph))

        lr = hr.resize((ph // s, ph // s), Image.BICUBIC)
        bic = lr.resize((ph, ph), Image.BICUBIC)
        bic_t = torch.from_numpy(np.array(bic).astype("float32") / 255.0).permute(2, 0, 1)
        hr_t = torch.from_numpy(np.array(hr).astype("float32") / 255.0).permute(2, 0, 1)
        return bic_t, hr_t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--train_mode",
        type=str,
        choices=("folders", "official"),
        default="official",
        help="folders: 显式 LR/HR 目录；official: 使用 dataset/ 下 REDS-sample 与 vimeo-RL 合成对",
    )
    ap.add_argument(
        "--lr_dir",
        type=str,
        default=str(default_dataset_dir() / "pairs" / "train_lr"),
        help="train_mode=folders：LR 图像目录，默认 dataset/pairs/train_lr",
    )
    ap.add_argument(
        "--hr_dir",
        type=str,
        default=str(default_dataset_dir() / "pairs" / "train_hr"),
        help="train_mode=folders：HR 图像目录，默认 dataset/pairs/train_hr",
    )
    ap.add_argument(
        "--dataset_root",
        type=str,
        default="",
        help="train_mode=official：课程数据根目录，省略时默认仓库下 dataset/",
    )
    ap.add_argument(
        "--official_source",
        type=str,
        choices=("reds", "vimeo", "both"),
        default="both",
        help="official 模式使用的子集",
    )
    ap.add_argument("--out_ckpt", type=str, default="weights/srcnn/srcnn_x4.pth")
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument(
        "--patch_hr",
        type=int,
        default=64,
        help="official 模式随机裁剪的 HR 正方形边长（需可被 scale 整除，且不超过图像短边）",
    )
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.train_mode == "folders":
        ds = PairDataset(args.lr_dir, args.hr_dir, args.scale)
    else:
        root = Path(args.dataset_root) if args.dataset_root else default_dataset_dir()
        hr_paths = _collect_official_hr_paths(root, args.official_source)
        ds = SyntheticBicubicDownscaleDataset(hr_paths, args.scale, args.patch_hr)

    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)

    model = SRCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.L1Loss()

    out_path = Path(args.out_ckpt)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(dl, desc=f"epoch {epoch+1}/{args.epochs}")
        total = 0.0
        for bic_b, hr_b in pbar:
            bic_b = bic_b.to(device)
            hr_b = hr_b.to(device)
            pred = model(bic_b)
            loss = loss_fn(pred, hr_b)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        print(f"epoch {epoch+1} avg L1: {total / len(dl):.6f}")

    torch.save({"model": model.state_dict(), "scale": args.scale}, str(out_path))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
