"""Inference: LR video -> bicubic x4 -> RefineNet -> HR video."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from part3_exploration.common_video import read_video, write_video
from part3_exploration.data.degrade_dataset import bicubic_up_lr, uint8_rgb_to_tensor, tensor_to_uint8_rgb
from part3_exploration.direction_a_flow_matching.refinenet import RefineNet


def main() -> None:
    ap = argparse.ArgumentParser(description="RefineNet video inference (Part 3-A).")
    ap.add_argument("--input", required=True, help="LR video path.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default="weights/part3_a/refinenet_x4.pth")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    ckpt = torch.load(args.ckpt, map_location=device)
    scale = int(ckpt.get("scale", 4))
    model = RefineNet().to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    frames_bgr, fps = read_video(args.input)
    out_frames: list[np.ndarray] = []
    with torch.no_grad():
        for frame in frames_bgr:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            bic = bicubic_up_lr(rgb, scale)
            t = uint8_rgb_to_tensor(bic).unsqueeze(0).to(device)
            pred = model(t).squeeze(0)
            out_rgb = tensor_to_uint8_rgb(pred)
            out_frames.append(cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_video(str(out_path), out_frames, fps)
    print(f"Wrote {out_path} ({len(out_frames)} frames)")


if __name__ == "__main__":
    main()
