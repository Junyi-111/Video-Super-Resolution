"""Inference: fuse BasicVSR + generative (Real-ESRGAN or RefineNet) with trained FusionHead."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from part3_exploration.common_video import read_video, write_video
from part3_exploration.data.degrade_dataset import tensor_to_uint8_rgb, uint8_rgb_to_tensor
from part3_exploration.direction_c_uncertainty.fusion_head import FusionHead


def load_fusion(ckpt_path: str, device: torch.device) -> FusionHead:
    ckpt = torch.load(ckpt_path, map_location=device)
    model = FusionHead().to(device)
    model.load_state_dict(ckpt["fusion"])
    model.eval()
    return model


def main() -> None:
    ap = argparse.ArgumentParser(description="FusionHead video inference (Part 3-C).")
    ap.add_argument("--basic", required=True, help="Conservative HR video (e.g. BasicVSR).")
    ap.add_argument("--gen", required=True, help="Generative HR video (e.g. Real-ESRGAN or RefineNet).")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mask_out", default="", help="Optional alpha visualization mp4.")
    ap.add_argument("--fusion_ckpt", default="weights/part3_c/fusion_head.pth")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    fusion = load_fusion(args.fusion_ckpt, device)

    basic_frames, fps = read_video(args.basic)
    gen_frames, _ = read_video(args.gen)
    n = min(len(basic_frames), len(gen_frames))
    if n == 0:
        raise RuntimeError("No frames to fuse")

    out_frames: list[np.ndarray] = []
    mask_frames: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(n):
            bgr_b = basic_frames[i]
            bgr_g = gen_frames[i]
            if bgr_b.shape[:2] != bgr_g.shape[:2]:
                bgr_g = cv2.resize(bgr_g, (bgr_b.shape[1], bgr_b.shape[0]), interpolation=cv2.INTER_AREA)
            rgb_b = cv2.cvtColor(bgr_b, cv2.COLOR_BGR2RGB)
            rgb_g = cv2.cvtColor(bgr_g, cv2.COLOR_BGR2RGB)
            tb = uint8_rgb_to_tensor(rgb_b).unsqueeze(0).to(device)
            tg = uint8_rgb_to_tensor(rgb_g).unsqueeze(0).to(device)
            fused, alpha = fusion.fuse(tb, tg)
            out_rgb = tensor_to_uint8_rgb(fused.squeeze(0))
            out_frames.append(cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR))
            if args.mask_out:
                a = alpha.squeeze().cpu().numpy()
                u8 = np.clip(a * 255.0, 0, 255).astype(np.uint8)
                mask_frames.append(cv2.applyColorMap(u8, cv2.COLORMAP_TURBO))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_video(str(out_path), out_frames, fps)
    print(f"Wrote {out_path} ({len(out_frames)} frames)")

    if args.mask_out:
        write_video(args.mask_out, mask_frames, fps)
        print(f"Wrote {args.mask_out}")


if __name__ == "__main__":
    main()
