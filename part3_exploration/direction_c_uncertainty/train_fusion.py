"""Train fusion head: basic branch = bicubic HR, gen branch = RefineNet(bicubic)."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from part1_baseline.clip_io import default_dataset_dir
from part3_exploration.data.degrade_dataset import (
    FusionPatchDataset,
    collect_hr_samples,
    summarize_hr_samples,
)
from part3_exploration.direction_a_flow_matching.refinenet import RefineNet
from part3_exploration.direction_c_uncertainty.fusion_head import FusionHead


def main() -> None:
    ap = argparse.ArgumentParser(description="Train fusion head (Part 3-C).")
    ap.add_argument("--dataset_root", type=str, default="")
    ap.add_argument("--official_source", choices=("reds", "vimeo", "both"), default="both")
    ap.add_argument("--hr_source", choices=("auto", "official", "videos", "part2", "all"), default="auto")
    ap.add_argument("--max_frames_per_video", type=int, default=0)
    ap.add_argument("--part2_realesrgan_dir", type=str, default="")
    ap.add_argument("--refinenet_ckpt", type=str, default="weights/part3_a/refinenet_x4.pth")
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--patch_hr", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--out_ckpt", type=str, default="weights/part3_c/fusion_head.pth")
    ap.add_argument("--freeze_refinenet", action="store_true", help="Do not backprop into RefineNet.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = Path(args.dataset_root) if args.dataset_root else default_dataset_dir()
    max_fpv = args.max_frames_per_video if args.max_frames_per_video > 0 else None
    part2_dir = Path(args.part2_realesrgan_dir) if args.part2_realesrgan_dir else None
    hr_samples = collect_hr_samples(
        root,
        official_source=args.official_source,
        hr_source=args.hr_source,
        part2_realesrgan_dir=part2_dir,
        max_frames_per_video=max_fpv,
    )
    print(f"Training HR samples: {summarize_hr_samples(hr_samples)}")

    ds = FusionPatchDataset(hr_samples, scale=args.scale, patch_hr=args.patch_hr, seed=args.seed)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    refinenet = RefineNet().to(device)
    rn_ckpt = torch.load(args.refinenet_ckpt, map_location=device)
    refinenet.load_state_dict(rn_ckpt["model"])
    if args.freeze_refinenet:
        refinenet.eval()
        for p in refinenet.parameters():
            p.requires_grad = False

    fusion = FusionHead().to(device)
    params = list(fusion.parameters())
    if not args.freeze_refinenet:
        params += list(refinenet.parameters())
    opt = torch.optim.Adam(params, lr=args.lr)
    loss_fn = nn.L1Loss()

    out_path = Path(args.out_ckpt)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        fusion.train()
        refinenet.train(not args.freeze_refinenet)
        total = 0.0
        pbar = tqdm(dl, desc=f"C epoch {epoch + 1}/{args.epochs}")
        for hr, bic, _lr in pbar:
            hr = hr.to(device)
            bic = bic.to(device)
            if args.freeze_refinenet:
                with torch.no_grad():
                    gen = refinenet(bic)
            else:
                gen = refinenet(bic)
            fused, _alpha = fusion.fuse(bic, gen)
            loss = loss_fn(fused, hr)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item()
            pbar.set_postfix(l1=f"{loss.item():.4f}")
        print(f"epoch {epoch + 1} avg L1: {total / max(len(dl), 1):.6f}")

    torch.save(
        {
            "fusion": fusion.state_dict(),
            "refinenet": refinenet.state_dict() if not args.freeze_refinenet else None,
            "scale": args.scale,
            "branch_basic": "bicubic",
            "branch_gen": "refinenet",
        },
        str(out_path),
    )
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
