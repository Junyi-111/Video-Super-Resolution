# Video Super-Resolution Project

Repository: https://github.com/Junyi-111/Video-Super-Resolution.git

This repository contains a three-part video super-resolution pipeline for the course project.

- **Part 1: Classical / baseline VSR** in `part1_baseline/`
- **Part 2: SOTA models** in `part2_sota/`
- **Part 3: Exploratory extensions** in `part3_exploration/`
- **Overall comparison artifacts** in root `output/`
- **Submission videos** in root `videos.zip` after packaging

## Repository Layout

```text
CV/
  dataset/
    inputs_mp4.zip                         # mandatory input clips archive

  part1_baseline/
    run_part1.py
    outputs/                               # Part 1 generated videos / archives

  part2_sota/
    infer_realesrgan.py
    infer_basicvsr.py
    infer_basicvsr_standalone.py
    run_basicvsr_all.py
    outputs/                               # Part 2 generated videos / archives

  part3_exploration/
    direction_a_flow_matching/             # rectified / flow-matching style refinement
    direction_b_sd_controlnet/             # ControlNet-Tile, flow stabilization, hybrid blend
    direction_c_uncertainty/               # uncertainty-aware adaptive fusion
    direction_d_distilled_streaming/        # distilled streaming diffusion VSR
    outputs/                               # all Part 3 generated outputs
      diffusion_tile*/                     # diffusion keyframe image experiments
      direction_a_flow_matching/
      direction_b_sd_controlnet/
      direction_c_uncertainty/
      direction_d_distilled_streaming/

  output/
    tables/                                # overall metrics tables
    figures/                               # overall comparison figures

  docs/
    requirements/                          # project PDF
    report/                                # report drafts / notes

  weights/
    part3_d/                               # Direction D weights placeholder / instructions

  Block-Sparse-Attention/                  # optional dependency for Direction D
  scripts/                                 # evaluation, figure, packaging helpers
```

Part-specific generated files should stay under each part's `outputs/` directory. Cross-method comparison tables and figures should stay under root `output/`.

## Environment

The lightweight parts use Python with OpenCV, NumPy, PyTorch, and image/video metric packages. Recommended setup:

```powershell
conda activate assignment
pip install opencv-python numpy torch torchvision pandas scikit-image lpips
```

Real-ESRGAN, BasicVSR, diffusion, and Direction D may require extra model-specific packages. See the README files inside `part2_sota/` and `part3_exploration/`.

## Data

Mandatory inputs are expected under:

```text
dataset/inputs_mp4/
dataset/wild_real_lr.mp4
dataset/wild_real.mp4      # optional GT-style wild source, if available
```

If only `dataset/inputs_mp4.zip` is present, extract it first:

```powershell
Expand-Archive dataset/inputs_mp4.zip dataset/inputs_mp4
```

For GT-based metrics, place GT videos under:

```text
dataset/gt_mp4/<clip_stem>.mp4
```

Without GT, PSNR and SSIM are not valid reference metrics. Use no-reference or temporal-consistency metrics instead.

## Part 1: Baselines

Run a single clip:

```powershell
python -m part1_baseline.run_part1 `
  --input dataset/inputs_mp4/REDS-sample_002.mp4 `
  --out_dir part1_baseline/outputs/REDS-sample_002
```

Part 1 outputs belong in:

```text
part1_baseline/outputs/
```

## Part 2: SOTA Models

Real-ESRGAN:

```powershell
python -m part2_sota.infer_realesrgan `
  --input dataset/inputs_mp4/REDS-sample_002.mp4 `
  --out part2_sota/outputs/realesrgan/REDS-sample_002.mp4
```

BasicVSR:

```powershell
python -m part2_sota.infer_basicvsr_standalone `
  --input dataset/inputs_mp4/REDS-sample_002.mp4 `
  --out part2_sota/outputs/basicvsr/REDS-sample_002.mp4
```

Batch BasicVSR:

```powershell
python -m part2_sota.run_basicvsr_all
```

Part 2 outputs belong in:

```text
part2_sota/outputs/
```

## Part 3: Exploration

Part 3 contains four directions:

- **Direction A:** rectified / flow-matching style refinement.
- **Direction B:** ControlNet-Tile experiments, optical-flow temporal stabilization, temporal/generative blend.
- **Direction C:** uncertainty-aware adaptive fusion of Real-ESRGAN and BasicVSR.
- **Direction D:** distilled streaming diffusion VSR with optional block-sparse attention.

All Part 3 generated outputs now use one consistent root:

```text
part3_exploration/outputs/
```

Run all available Part 3 directions:

```powershell
python -m part3_exploration.run_all_part3 --eval_skip_fid
```

Run only Direction D:

```powershell
python -m part3_exploration.direction_d_distilled_streaming.run_batch
```

Direction D requires weights under `weights/part3_d/`. If weights are missing, the batch script prints a warning and skips generation.

## Evaluation

Overall comparison tables and figures are written under root `output/`:

```powershell
python scripts/evaluate_project.py
python scripts/evaluate_wild_metrics.py
python scripts/evaluate_wild_lpips.py
python scripts/merge_wild_tables.py
python -m part3_exploration.evaluate_part3_metrics --skip_fid
```

Expected output locations:

```text
output/tables/
output/figures/
```

## Packaging Videos

The Canvas video artifact should be named:

```text
videos.zip
```

It should contain processed videos only, not raw input videos or GT videos. A prepared package can be regenerated from the available processed outputs by collecting:

```text
part1_baseline/outputs/
part2_sota/outputs/
part3_exploration/outputs/
```

The generated `videos.zip` should be placed at the repository root.

## Notes

- PSNR and SSIM require GT/reference videos. Do not report them for real LR wild clips without GT.
- LPIPS can be used with GT; no-reference metrics and temporal consistency are better suited to real-world no-GT clips.
- Direction D is experimental and dependency-heavy. Keep its outputs under `part3_exploration/outputs/direction_d_distilled_streaming/` for consistency with the other Part 3 directions.
## GitHub Upload Notes

This repository is designed so that source code, reports, lightweight tables, and figures can be uploaded to GitHub, while large local artifacts stay out of git.

Recommended files to track:

```text
README.md
requirements.txt
.gitignore
part1_baseline/
part2_sota/
part3_exploration/
scripts/
docs/report/
docs/requirements/
output/tables/
output/figures/
weights/README.md
weights/part3_d/README.md
```

Do not commit these large or generated artifacts directly to GitHub:

```text
dataset/
weights/**/*.pth
weights/**/*.pt
weights/**/*.ckpt
weights/**/*.safetensors
part1_baseline/outputs/
part2_sota/outputs/
part3_exploration/outputs/
output/vimeo90k_small/
videos.zip
*.mp4
```

Submit `videos.zip` to Canvas separately. If processed videos or weights need to be shared publicly, use GitHub Releases, Google Drive, or another external download link and reference it in this README.