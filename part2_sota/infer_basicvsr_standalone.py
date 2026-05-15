"""BasicVSR x4 inference using a standalone network and official weights."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from part2_sota.basicvsr_standalone import BasicVSRNet


def read_video(path: str) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames read from {path}")
    return frames, fps


def write_video(path: str, frames: list[np.ndarray], fps: float) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {out}")
    for frame in frames:
        writer.write(frame)
    writer.release()


def load_basicvsr(ckpt_path: str, device: torch.device) -> BasicVSRNet:
    model = BasicVSRNet(mid_channels=64, num_blocks=30).to(device)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    state_dict = {k.removeprefix("generator."): v for k, v in state_dict.items() if k.startswith("generator.")}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"Warning: missing={len(missing)}, unexpected={len(unexpected)}")
        if missing:
            print("  first missing:", missing[:5])
        if unexpected:
            print("  first unexpected:", unexpected[:5])
    model.eval()
    return model


def frames_to_tensor(frames: list[np.ndarray], device: torch.device) -> torch.Tensor:
    rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
    arr = np.stack(rgb).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(0, 3, 1, 2).unsqueeze(0).to(device)


def tensor_to_frames(tensor: torch.Tensor) -> list[np.ndarray]:
    arr = tensor.squeeze(0).detach().cpu().clamp(0, 1).permute(0, 2, 3, 1).numpy()
    frames = []
    for rgb in arr:
        u8 = (rgb * 255.0).round().astype(np.uint8)
        frames.append(cv2.cvtColor(u8, cv2.COLOR_RGB2BGR))
    return frames


def infer_chunks(
    model: BasicVSRNet,
    frames: list[np.ndarray],
    device: torch.device,
    chunk_size: int,
    overlap: int,
) -> list[np.ndarray]:
    if len(frames) <= chunk_size:
        with torch.no_grad():
            return tensor_to_frames(model(frames_to_tensor(frames, device)))

    out_frames: list[np.ndarray] = []
    step = max(1, chunk_size - overlap)
    for start in tqdm(range(0, len(frames), step), desc="basicvsr chunks"):
        end = min(len(frames), start + chunk_size)
        chunk = frames[start:end]
        with torch.no_grad():
            sr = tensor_to_frames(model(frames_to_tensor(chunk, device)))
        keep_from = 0 if start == 0 else overlap
        out_frames.extend(sr[keep_from:])
        if end == len(frames):
            break
    return out_frames[: len(frames)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone BasicVSR inference.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", default="")
    parser.add_argument("--out_dir", default="part2_sota/outputs/basicvsr")
    parser.add_argument("--ckpt", default="weights/basicvsr/basicvsr_vimeo90k_bi.pth")
    parser.add_argument("--chunk_size", type=int, default=30)
    parser.add_argument("--overlap", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    inp = Path(args.input)
    out = Path(args.out) if args.out else Path(args.out_dir) / inp.name

    device = torch.device(args.device)
    model = load_basicvsr(args.ckpt, device)
    frames, fps = read_video(str(inp))
    out_frames = infer_chunks(model, frames, device, args.chunk_size, args.overlap)
    write_video(str(out), out_frames, fps)
    print(f"Wrote {out} ({len(out_frames)} frames)")


if __name__ == "__main__":
    main()
