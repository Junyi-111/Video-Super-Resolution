"""Shared video / frame-sequence metrics for course evaluation (PSNR, SSIM, LPIPS, FID, tLPIPS)."""

from __future__ import annotations

import contextlib
import os
import shutil
import tempfile
import warnings
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch

SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def read_video(path: str | Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames read from {path}")
    return frames


def list_frame_paths(frame_dir: str | Path) -> list[Path]:
    d = Path(frame_dir)
    if not d.is_dir():
        raise FileNotFoundError(f"Frame directory not found: {d}")
    paths = [p for p in d.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTS]
    return sorted(paths, key=lambda p: p.name)


def read_frames_dir(frame_dir: str | Path) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    for path in list_frame_paths(frame_dir):
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Could not read image: {path}")
        frames.append(img)
    if not frames:
        raise RuntimeError(f"No frames found in {frame_dir}")
    return frames


def load_sequence(path: str | Path) -> list[np.ndarray]:
    p = Path(path)
    if p.is_dir():
        return read_frames_dir(p)
    if p.is_file():
        return read_video(p)
    raise FileNotFoundError(f"Path not found: {p}")


def psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_f = pred.astype(np.float64)
    gt_f = gt.astype(np.float64)
    mse = np.mean((pred_f - gt_f) ** 2)
    if mse <= 1e-12:
        return float("inf")
    return 20.0 * np.log10(255.0 / np.sqrt(mse))


def ssim_channel(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_f = pred.astype(np.float64)
    gt_f = gt.astype(np.float64)
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    kernel = (11, 11)
    sigma = 1.5

    mu_x = cv2.GaussianBlur(pred_f, kernel, sigma)
    mu_y = cv2.GaussianBlur(gt_f, kernel, sigma)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = cv2.GaussianBlur(pred_f * pred_f, kernel, sigma) - mu_x2
    sigma_y2 = cv2.GaussianBlur(gt_f * gt_f, kernel, sigma) - mu_y2
    sigma_xy = cv2.GaussianBlur(pred_f * gt_f, kernel, sigma) - mu_xy

    ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    )
    return float(np.mean(ssim_map))


def ssim_rgb(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.mean([ssim_channel(pred[..., c], gt[..., c]) for c in range(3)]))


def resize_like(frame: np.ndarray, ref: np.ndarray) -> np.ndarray:
    h, w = ref.shape[:2]
    if frame.shape[:2] == (h, w):
        return frame
    return cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)


def resize_for_lpips(frame: np.ndarray, short_side: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if min(h, w) == short_side:
        return frame
    if h < w:
        nh = short_side
        nw = round(w * short_side / h)
    else:
        nw = short_side
        nh = round(h * short_side / w)
    return cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)


def streaming_vsr_canvas_dims(w0: int, h0: int, scale: int = 4, multiple: int = 128) -> tuple[int, int, int, int]:
    """LR → 4× canvas size and 128-multiple target crop (matches course WanVSR driver)."""
    if w0 <= 0 or h0 <= 0:
        raise ValueError("invalid LR size")
    s_w = w0 * scale
    s_h = h0 * scale
    t_w = max(multiple, (s_w // multiple) * multiple)
    t_h = max(multiple, (s_h // multiple) * multiple)
    return s_w, s_h, t_w, t_h


def read_lr_whn(path: str | Path) -> tuple[int, int, int]:
    """First-frame LR width/height and frame count (OpenCV: shape rows=H, cols=W)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open LR video: {path}")
    ok, fr = cap.read()
    if not ok or fr is None:
        cap.release()
        raise RuntimeError(f"No frames in LR video: {path}")
    h_lr, w_lr = fr.shape[:2]
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if n <= 0:
        n = len(read_video(path))
    return int(w_lr), int(h_lr), int(n)


def count_video_frames(path: str | Path) -> int:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return 0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if n > 0:
        return n
    return len(read_video(path))


def center_crop_pred_like_streaming_canvas(
    frame: np.ndarray,
    *,
    s_h: int,
    s_w: int,
    t_h: int,
    t_w: int,
) -> np.ndarray:
    """Center-crop prediction from full 4× canvas (s_h×s_w) to t_h×t_w (OpenCV H×W)."""
    hp, wp = frame.shape[:2]
    if hp == t_h and wp == t_w:
        return frame.copy()
    if hp == s_h and wp == s_w:
        y0 = (s_h - t_h) // 2
        x0 = (s_w - t_w) // 2
        return frame[y0 : y0 + t_h, x0 : x0 + t_w].copy()
    adj = frame
    if hp != s_h or wp != s_w:
        adj = cv2.resize(frame, (s_w, s_h), interpolation=cv2.INTER_AREA)
    y0 = (s_h - t_h) // 2
    x0 = (s_w - t_w) // 2
    return adj[y0 : y0 + t_h, x0 : x0 + t_w].copy()


def center_crop_gt_like_streaming_canvas(
    frame: np.ndarray,
    *,
    s_h: int,
    s_w: int,
    t_h: int,
    t_w: int,
) -> np.ndarray:
    """Apply the same *relative* center crop as on the 4× canvas, mapped onto GT resolution."""
    hg, wg = frame.shape[:2]
    y0_4 = (s_h - t_h) / 2.0
    x0_4 = (s_w - t_w) / 2.0
    y0_g = int(round(y0_4 * hg / s_h))
    x0_g = int(round(x0_4 * wg / s_w))
    h_g = int(round(t_h * hg / s_h))
    w_g = int(round(t_w * wg / s_w))
    y0_g = max(0, min(y0_g, max(0, hg - 1)))
    x0_g = max(0, min(x0_g, max(0, wg - 1)))
    h_g = max(1, min(h_g, hg - y0_g))
    w_g = max(1, min(w_g, wg - x0_g))
    return frame[y0_g : y0_g + h_g, x0_g : x0_g + w_g].copy()


def align_sequences_streaming_vsr_rule(
    pred_frames: list[np.ndarray],
    gt_frames: list[np.ndarray] | None,
    *,
    s_h: int,
    s_w: int,
    t_h: int,
    t_w: int,
    n_eval: int,
) -> tuple[list[np.ndarray], list[np.ndarray] | None]:
    """Spatial crop + truncate to ``n_eval`` frames (same rule for pred; GT crop when present)."""
    pred_frames = pred_frames[:n_eval]
    pred_out = [center_crop_pred_like_streaming_canvas(f, s_h=s_h, s_w=s_w, t_h=t_h, t_w=t_w) for f in pred_frames]
    if gt_frames is None:
        return pred_out, None
    gt_cut = gt_frames[:n_eval]
    gt_out = [center_crop_gt_like_streaming_canvas(f, s_h=s_h, s_w=s_w, t_h=t_h, t_w=t_w) for f in gt_cut]
    return pred_out, gt_out


def upscale_lr_frame_to_streaming_canvas_roi(
    frame_bgr: np.ndarray,
    *,
    s_h: int,
    s_w: int,
    t_h: int,
    t_w: int,
) -> np.ndarray:
    """Bicubic upscale LR to full 4× canvas then center-crop to the streaming-eval ROI."""
    up = cv2.resize(frame_bgr, (s_w, s_h), interpolation=cv2.INTER_CUBIC)
    y0 = (s_h - t_h) // 2
    x0 = (s_w - t_w) // 2
    return up[y0 : y0 + t_h, x0 : x0 + t_w].copy()


def prepare_pred_frames_unified(
    pred_frames: list[np.ndarray],
    *,
    s_h: int,
    s_w: int,
    t_h: int,
    t_w: int,
    n_eval: int,
    native_lr: bool,
) -> list[np.ndarray]:
    seq = pred_frames[:n_eval]
    if native_lr:
        return [upscale_lr_frame_to_streaming_canvas_roi(f, s_h=s_h, s_w=s_w, t_h=t_h, t_w=t_w) for f in seq]
    return [center_crop_pred_like_streaming_canvas(f, s_h=s_h, s_w=s_w, t_h=t_h, t_w=t_w) for f in seq]


def crop_gt_sequence_unified(
    gt_frames: list[np.ndarray],
    *,
    s_h: int,
    s_w: int,
    t_h: int,
    t_w: int,
    n_eval: int,
) -> list[np.ndarray]:
    return [
        center_crop_gt_like_streaming_canvas(f, s_h=s_h, s_w=s_w, t_h=t_h, t_w=t_w) for f in gt_frames[:n_eval]
    ]


def evaluate_wild_jobs_unified(
    repo: Path,
    jobs: list[dict],
    *,
    wild_gt_rel: str,
    wild_lr_rel: str,
    metrics: set[str],
    device: torch.device,
    stride: int,
    short_side: int,
    loss_fn: torch.nn.Module | None,
) -> list[dict[str, str]]:
    """Same spatial/temporal alignment as Part3 ``evaluate_part3_metrics`` (128-multiple crop on 4× canvas + min length)."""
    lr_path = (repo / wild_lr_rel.replace("\\", "/")).resolve()
    gt_path = (repo / wild_gt_rel.replace("\\", "/")).resolve()
    if not lr_path.is_file() or not gt_path.is_file():
        return []

    w0, h0, _ = read_lr_whn(lr_path)
    s_w, s_h, t_w, t_h = streaming_vsr_canvas_dims(w0, h0)
    twh = f"{t_w}x{t_h}"

    loaded: list[tuple[dict, list]] = []
    for job in jobs:
        pred_p = (repo / job["pred"].replace("\\", "/")).resolve()
        if not pred_p.is_file():
            continue
        try:
            fr = load_sequence(pred_p)
        except (RuntimeError, OSError):
            continue
        if not fr:
            continue
        loaded.append((job, fr))

    if not loaded:
        return []

    try:
        gt_raw = load_sequence(gt_path)
    except (RuntimeError, OSError):
        return []

    lens = [len(fr) for _, fr in loaded]
    lens.append(len(gt_raw))
    n_eval = min(lens)
    if n_eval <= 0:
        return []

    gt_crop = crop_gt_sequence_unified(gt_raw, s_h=s_h, s_w=s_w, t_h=t_h, t_w=t_w, n_eval=n_eval)

    rows: list[dict[str, str]] = []
    for job, fr in loaded:
        native = job.get("part") == "input"
        pred_crop = prepare_pred_frames_unified(
            fr, s_h=s_h, s_w=s_w, t_h=t_h, t_w=t_w, n_eval=n_eval, native_lr=native
        )
        vals = evaluate_sequence(
            pred_crop,
            gt_crop,
            metrics=metrics,
            device=device,
            stride=stride,
            short_side=short_side,
            loss_fn=loss_fn,
            reuse_loss_fn=True,
        )
        pred_rel = job["pred"].replace("\\", "/")
        rows.append(
            {
                "dataset": job["dataset"],
                "clip": job["clip"],
                "method": job["method"],
                "part": job["part"],
                "frames": str(int(vals.get("frames", 0))),
                "psnr": format_metric(vals.get("psnr")),
                "ssim": format_metric(vals.get("ssim")),
                "lpips": format_metric(vals.get("lpips")),
                "tlpips": format_metric(vals.get("tlpips")),
                "tlpips_noref": format_metric(vals.get("tlpips_noref")),
                "fid": format_metric(vals.get("fid")),
                "temporal_mae": format_metric(vals.get("temporal_mae")),
                "has_gt": "true",
                "pred_path": pred_rel,
                "gt_path": wild_gt_rel.replace("\\", "/"),
                "eval_policy": "unified_streaming_vsr_crop_minT",
                "crop_twh": twh,
            }
        )
    return rows


def evaluate_sample_stem_unified(
    repo: Path,
    lr_path: Path,
    gt_path: Path | None,
    jobs: list[dict],
    *,
    metrics: set[str],
    device: torch.device,
    stride: int,
    short_side: int,
    loss_fn: torch.nn.Module | None,
) -> list[dict[str, str]]:
    """Part1/Part2 sample clips: same streaming-eval ROI + ``n_eval`` = min(pred lengths, GT if any)."""
    if not lr_path.is_file():
        return []
    w0, h0, _ = read_lr_whn(lr_path)
    s_w, s_h, t_w, t_h = streaming_vsr_canvas_dims(w0, h0)
    twh = f"{t_w}x{t_h}"

    loaded: list[tuple[dict, list]] = []
    for job in jobs:
        pred_p = (repo / job["pred"].replace("\\", "/")).resolve()
        if not pred_p.is_file():
            continue
        try:
            fr = load_sequence(pred_p)
        except (RuntimeError, OSError):
            continue
        if not fr:
            continue
        loaded.append((job, fr))

    if not loaded:
        return []

    gt_raw: list | None = None
    if gt_path is not None and gt_path.is_file():
        try:
            gt_raw = load_sequence(gt_path)
        except (RuntimeError, OSError):
            gt_raw = None

    lens = [len(fr) for _, fr in loaded]
    if gt_raw is not None:
        lens.append(len(gt_raw))
    n_eval = min(lens)
    if n_eval <= 0:
        return []

    gt_crop = (
        crop_gt_sequence_unified(gt_raw, s_h=s_h, s_w=s_w, t_h=t_h, t_w=t_w, n_eval=n_eval)
        if gt_raw is not None
        else None
    )

    rows: list[dict[str, str]] = []
    for job, fr in loaded:
        pred_crop = prepare_pred_frames_unified(
            fr, s_h=s_h, s_w=s_w, t_h=t_h, t_w=t_w, n_eval=n_eval, native_lr=False
        )
        mset = set(metrics)
        if gt_crop is None:
            mset = {"tlpips_noref"}
        vals = evaluate_sequence(
            pred_crop,
            gt_crop,
            metrics=mset,
            device=device,
            stride=stride,
            short_side=short_side,
            loss_fn=loss_fn,
            reuse_loss_fn=True,
        )
        gt_rel = job.get("gt", "")
        rows.append(
            {
                "dataset": job["dataset"],
                "clip": job["clip"],
                "method": job["method"],
                "part": job["part"],
                "frames": str(int(vals.get("frames", 0))),
                "psnr": format_metric(vals.get("psnr")),
                "ssim": format_metric(vals.get("ssim")),
                "lpips": format_metric(vals.get("lpips")),
                "tlpips": format_metric(vals.get("tlpips")),
                "tlpips_noref": format_metric(vals.get("tlpips_noref")),
                "fid": format_metric(vals.get("fid")),
                "temporal_mae": format_metric(vals.get("temporal_mae")),
                "has_gt": "true" if gt_crop is not None else "false",
                "pred_path": job["pred"].replace("\\", "/"),
                "gt_path": gt_rel,
                "eval_policy": "unified_streaming_vsr_crop_minT",
                "crop_twh": twh,
            }
        )
    return rows


def to_lpips_tensor_bgr(frame: np.ndarray, device: torch.device) -> torch.Tensor:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    arr = rgb.astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


def frame_diff_mae(frames: list[np.ndarray]) -> float:
    if len(frames) < 2:
        return 0.0
    vals = [
        float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))))
        for a, b in zip(frames[:-1], frames[1:])
    ]
    return float(np.mean(vals))


def _pair_lpips(
    loss_fn: torch.nn.Module,
    a: np.ndarray,
    b: np.ndarray,
    device: torch.device,
    short_side: int,
) -> float:
    a_r = resize_for_lpips(a, short_side)
    b_r = resize_for_lpips(b, short_side)
    with torch.no_grad():
        return float(loss_fn(to_lpips_tensor_bgr(a_r, device), to_lpips_tensor_bgr(b_r, device)).item())


def mean_lpips(
    loss_fn: torch.nn.Module,
    gt_frames: list[np.ndarray],
    pred_frames: list[np.ndarray],
    stride: int,
    short_side: int,
    device: torch.device,
) -> tuple[float, int]:
    vals: list[float] = []
    n = min(len(gt_frames), len(pred_frames))
    for i in range(0, n, stride):
        pred = resize_like(pred_frames[i], gt_frames[i])
        vals.append(_pair_lpips(loss_fn, pred, gt_frames[i], device, short_side))
    return float(np.mean(vals)), len(vals)


def temporal_lpips(
    loss_fn: torch.nn.Module,
    gt_frames: list[np.ndarray],
    pred_frames: list[np.ndarray],
    stride: int,
    short_side: int,
    device: torch.device,
) -> tuple[float, int]:
    """tLPIPS: perceptual distance between temporal differences (pred vs GT)."""
    vals: list[float] = []
    n = min(len(gt_frames), len(pred_frames))
    for i in range(stride, n, stride):
        gt_diff = cv2.absdiff(gt_frames[i], gt_frames[i - stride])
        pred_i = resize_like(pred_frames[i], gt_frames[i])
        pred_prev = resize_like(pred_frames[i - stride], gt_frames[i - stride])
        pred_diff = cv2.absdiff(pred_i, pred_prev)
        vals.append(_pair_lpips(loss_fn, pred_diff, gt_diff, device, short_side))
    return float(np.mean(vals)) if vals else 0.0, len(vals)


def temporal_lpips_noref(
    loss_fn: torch.nn.Module,
    pred_frames: list[np.ndarray],
    stride: int,
    short_side: int,
    device: torch.device,
) -> tuple[float, int]:
    """No-reference temporal proxy: LPIPS between consecutive predicted frames."""
    vals: list[float] = []
    for i in range(0, len(pred_frames) - stride, stride):
        vals.append(_pair_lpips(loss_fn, pred_frames[i], pred_frames[i + stride], device, short_side))
    return float(np.mean(vals)) if vals else 0.0, len(vals)


def _write_sampled_frames(frames: Iterable[np.ndarray], out_dir: Path, stride: int) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for i, frame in enumerate(frames):
        if i % stride != 0:
            continue
        cv2.imwrite(str(out_dir / f"{count:06d}.png"), frame)
        count += 1
    return count


@contextlib.contextmanager
def _writable_torch_hub_env():
    """Force Torch Hub / FID Inception weights into a user-writable cache.

    Cluster images often set ``TORCH_HOME=/opt/torch_models`` (read-only), which
    breaks ``pytorch-fid``'s first-time weight download with PermissionError.
    """
    previous = os.environ.get("TORCH_HOME")
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        hub_root = Path(xdg) / "cv_torch_hub"
    else:
        hub_root = Path.home() / ".cache" / "cv_torch_hub"
    try:
        hub_root.mkdir(parents=True, exist_ok=True)
    except OSError:
        hub_root = Path(tempfile.gettempdir()) / "cv_torch_hub"
        hub_root.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_HOME"] = str(hub_root)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("TORCH_HOME", None)
        else:
            os.environ["TORCH_HOME"] = previous


def compute_fid(
    gt_frames: list[np.ndarray],
    pred_frames: list[np.ndarray],
    device: torch.device,
    stride: int = 5,
    short_side: int = 256,
) -> float:
    from pytorch_fid import fid_score

    n = min(len(gt_frames), len(pred_frames))
    if n < 2:
        return float("nan")

    with tempfile.TemporaryDirectory(prefix="cv_fid_gt_") as gt_tmp, tempfile.TemporaryDirectory(
        prefix="cv_fid_pred_"
    ) as pred_tmp:
        gt_dir = Path(gt_tmp)
        pred_dir = Path(pred_tmp)
        gt_count = 0
        pred_count = 0
        for i in range(0, n, stride):
            gt = gt_frames[i]
            pred = resize_like(pred_frames[i], gt)
            if short_side > 0:
                gt = resize_for_lpips(gt, short_side)
                pred = resize_for_lpips(pred, short_side)
            cv2.imwrite(str(gt_dir / f"{gt_count:06d}.png"), gt)
            cv2.imwrite(str(pred_dir / f"{pred_count:06d}.png"), pred)
            gt_count += 1
            pred_count += 1
        if gt_count < 2 or pred_count < 2:
            return float("nan")
        try:
            with _writable_torch_hub_env():
                return float(
                    fid_score.calculate_fid_given_paths(
                        [str(gt_dir), str(pred_dir)],
                        batch_size=min(8, gt_count),
                        device=device,
                        dims=2048,
                        num_workers=0,
                    )
                )
        except (PermissionError, OSError, RuntimeError, ValueError) as exc:
            warnings.warn(f"FID skipped or failed ({exc}).", stacklevel=2)
            return float("nan")


def evaluate_sequence(
    pred_frames: list[np.ndarray],
    gt_frames: list[np.ndarray] | None,
    *,
    metrics: set[str],
    device: torch.device,
    stride: int = 5,
    short_side: int = 256,
    loss_fn: torch.nn.Module | None = None,
    reuse_loss_fn: bool = False,
) -> dict[str, float | int]:
    """Evaluate one predicted sequence. GT may be None for no-reference metrics only."""
    result: dict[str, float | int] = {"frames": min(len(pred_frames), len(gt_frames) if gt_frames else len(pred_frames))}

    aligned_pred: list[np.ndarray] = []
    if gt_frames is not None:
        n = min(len(gt_frames), len(pred_frames))
        psnr_vals: list[float] = []
        ssim_vals: list[float] = []
        for gt, pred in zip(gt_frames[:n], pred_frames[:n]):
            pred_r = resize_like(pred, gt)
            aligned_pred.append(pred_r)
            if "psnr" in metrics:
                psnr_vals.append(psnr(pred_r, gt))
            if "ssim" in metrics:
                ssim_vals.append(ssim_rgb(pred_r, gt))
        if "psnr" in metrics:
            result["psnr"] = float(np.mean(psnr_vals))
        if "ssim" in metrics:
            result["ssim"] = float(np.mean(ssim_vals))
        if "temporal_mae" in metrics:
            result["temporal_mae"] = frame_diff_mae(aligned_pred)
    else:
        aligned_pred = list(pred_frames)

    needs_lpips = bool(metrics & {"lpips", "tlpips", "tlpips_noref"})
    created_loss_fn = False
    if needs_lpips:
        if loss_fn is None:
            import lpips

            loss_fn = lpips.LPIPS(net="alex").to(device).eval()
            created_loss_fn = True

        if gt_frames is not None and "lpips" in metrics:
            lp, _ = mean_lpips(loss_fn, gt_frames, pred_frames, stride, short_side, device)
            result["lpips"] = lp
        if gt_frames is not None and "tlpips" in metrics:
            tlp, _ = temporal_lpips(loss_fn, gt_frames, pred_frames, stride, short_side, device)
            result["tlpips"] = tlp
        if "tlpips_noref" in metrics or (gt_frames is None and "tlpips" in metrics):
            tlp_nr, _ = temporal_lpips_noref(loss_fn, pred_frames, stride, short_side, device)
            result["tlpips_noref"] = tlp_nr

    if gt_frames is not None and "fid" in metrics:
        result["fid"] = compute_fid(gt_frames, pred_frames, device, stride=stride, short_side=short_side)

    if created_loss_fn and not reuse_loss_fn:
        del loss_fn

    return result


def format_metric(value: float | int | None, digits: int = 4) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def cleanup_temp_dir(path: Path | None) -> None:
    if path is not None and path.exists():
        shutil.rmtree(path, ignore_errors=True)
