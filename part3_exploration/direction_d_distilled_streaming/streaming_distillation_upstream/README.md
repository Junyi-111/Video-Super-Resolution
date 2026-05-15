# Vendored streaming diffusion VSR runtime (course)

This tree is a **vendored inference / library snapshot** used by Part 3 Direction D. It is installed in editable mode from the CV repo root:

```bash
pip install -e part3_exploration/direction_d_distilled_streaming/streaming_distillation_upstream
```

**Block-sparse attention:** optional compiled extension for full performance; see `part3_exploration/direction_d_distilled_streaming/README.md` (environment and troubleshooting).

**Weights:** Direction D checkpoints are expected under `weights/part3_d/`, produced by the **course training / distillation pipeline on your data** (see `weights/part3_d/README.md` and the Direction D README). This README does not redistribute upstream project branding or download instructions.

**Packaging:** to ship **code only** (no videos, no `.pth` / `.pt` / `.safetensors` / `.ckpt` checkpoints), run `bash scripts/package_cv_sources_no_video_zip.sh` or `bash scripts/package_cv_sources_no_video.sh` from the repo root (`part3_exploration/README.md` documents options).
