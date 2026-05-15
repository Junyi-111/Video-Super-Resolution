# Direction D 鈥?Distilled streaming diffusion VSR

**Report narrative:** a one-step distilled **streaming** super-resolution backbone with sparse attention 鈥?suitable when latency and long sequences matter alongside generative fidelity.

Implementation is split into:

- **Neutral entry points** at this folder root (`run_course_infer.py`, `run_batch.py`, `streaming_one_step_infer.py`).
- **Vendored runtime** used as a library checkpoint format in `streaming_distillation_upstream/` (diffusion package + example utilities). Follow that tree鈥檚 license / citation policy in your course materials if required.

## Environment (GPU)

- **Python 3.11** is recommended to match the vendored stack.
- Install the vendored package in editable mode from the repo root:

```bash
pip install -e part3_exploration/direction_d_distilled_streaming/streaming_distillation_upstream
```

- Install dependencies from the same tree (pin file inside `streaming_distillation_upstream/requirements.txt`). Your global `torch` must be **CUDA-capable**; CPU-only PyTorch will not run this path.
- The upstream stack expects a **block-sparse attention** extension for best performance; see the install notes inside `streaming_distillation_upstream/README.md` (compilation step). Without it, some setups may fail at import or runtime.

## Weights (training / distillation on data)

Place the **streaming distilled** checkpoint files in **`weights/part3_d/`** at the repo root (flat), **or** in a subfolder such as `weights/part3_d/streaming_ckpt_v11/`. These files are assumed to be **produced by the course distillation / fine-tuning pipeline on your datasets** (same filenames as the vendored inference layout). They are **not** described here as ready-made third-party downloads.

Required file names:

**Full pipeline (`--variant full`, default):**

- `diffusion_pytorch_model_streaming_dmd.safetensors`
- `Wan2.1_VAE.pth`
- `LQ_proj_in.ckpt`

**Tiny pipeline (`--variant tiny`, lower VRAM):**

- `diffusion_pytorch_model_streaming_dmd.safetensors`
- `LQ_proj_in.ckpt`
- `TCDecoder.ckpt`

Default `--weights_dir` is `weights/part3_d` (relative paths are resolved from the **repository root**, not the shell `cwd`).

## Code packaging (no video assets, no weight checkpoints)

From the CV repo root, create **zip** or **tar.gz** archives that exclude common video extensions **and** model files: **`.pth`**, **`.pt`**, **`.safetensors`**, **`.ckpt`**. Keep weights separately or reproduce via training.

```bash
bash scripts/package_cv_sources_no_video_zip.sh
# or tar.gz: bash scripts/package_cv_sources_no_video.sh
```

See `part3_exploration/README.md` for options (`EXCLUDE_GIT=1`, custom output path).

## Recommended: course batch (official-style v1.1 driver)

Same stack as upstream `examples/WanVSR` (`prepare_input_tensor` + `init_pipeline` + `pipe`). Writes `outputs/<stem>_streaming_distilled_x4.mp4` for Wild LR plus `dataset/inputs_mp4/*.mp4` (skips `wild*` in that folder).

From repo root:

```bash
python -m part3_exploration.direction_d_distilled_streaming.run_course_infer
python -m part3_exploration.direction_d_distilled_streaming.run_course_infer --limit 3
# Physical GPU 6 (CUDA_VISIBLE_DEVICES); full pipeline defaults to tiled VAE + allocator hints:
python -m part3_exploration.direction_d_distilled_streaming.run_course_infer --cuda-id 6
# Still OOM on very large / long clips: smaller VAE tiles, or switch to tiny:
python -m part3_exploration.direction_d_distilled_streaming.run_course_infer --cuda-id 6 --low-vram
python -m part3_exploration.direction_d_distilled_streaming.run_course_infer --variant tiny
# Max speed when you have spare VRAM:
python -m part3_exploration.direction_d_distilled_streaming.run_course_infer --no-tiled
```

VRAM notes: **`--variant full` enables tiled VAE by default** (`--no-tiled` matches upstream 鈥渇ast but hungry鈥?behaviour). The script also sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` if unset, to reduce fragmentation before the final decode.

Or `cd part3_exploration/direction_d_distilled_streaming/streaming_distillation_upstream/examples/WanVSR` and run `python infer_course_dataset.py 鈥.

`run_batch` is a short alias that maps `--weights_dir` / `--inputs_dir` / `--wild_lr` / `--out_dir` to the same script.

## Single clip (legacy wrapper)

```bash
python -m part3_exploration.direction_d_distilled_streaming.streaming_one_step_infer \
  --input dataset/wild_real_lr.mp4 \
  --out part3_exploration/outputs/direction_d_distilled_streaming/wild_real_lr_streaming_distilled_x4.mp4 \
  --weights_dir weights/part3_d \
  --variant full
```

Optional: `--tiled` (full only) for less VRAM.

## Full dataset + Wild

```bash
python -m part3_exploration.direction_d_distilled_streaming.run_batch
python -m part3_exploration.direction_d_distilled_streaming.run_batch --limit 3
```

Outputs (default): `part3_exploration/outputs/direction_d_distilled_streaming/<stem>_streaming_distilled_x4.mp4`

## Evaluation

After generation, run the Part 3 metrics aggregator (includes Direction D if outputs exist). **Default evaluation aligns all directions** to the same 128-multiple center crop on the 4脳 LR canvas and the same temporal length (`--no-unified-eval` for legacy):

```bash
python -m part3_exploration.evaluate_part3_metrics --skip_fid
```

## Troubleshooting: `Block-Sparse-Attention` / `CUDA_HOME` / CPU PyTorch

If `python setup.py install` fails with **`CUDA_HOME environment variable is not set`** or the log shows **`torch.__version__ = ...+cpu`**:

1. **Use a CUDA build of PyTorch** in the same conda env (CPU wheels cannot run or compile this stack):
   ```bash
   pip uninstall -y torch torchvision torchaudio
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
   python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
   ```
   Choose a wheel (`cu124`, `cu126`, 鈥? compatible with your **driver** (`nvidia-smi`).

2. **Expose the CUDA toolkit** (`nvcc` often lives under `/usr/local/cuda/bin`, not in default `PATH`):
   ```bash
   export CUDA_HOME=/usr/local/cuda
   export PATH="$CUDA_HOME/bin:$PATH"
   which nvcc && nvcc --version
   ```

3. **Re-run** from the `Block-Sparse-Attention` repo (use **`--no-build-isolation`** so `setup.py` sees your env鈥檚 `torch`):

```bash
cd Block-Sparse-Attention
pip install . --no-build-isolation
```

Without `nvcc` on the machine, ask an admin to install the CUDA toolkit; the GPU driver alone is not enough to compile extensions.

### `RuntimeError: The detected CUDA version (12.4) mismatches ... PyTorch (13.0)`

Your **`nvcc`** comes from **CUDA 12.4** (e.g. `CUDA_HOME=/usr/local/cuda` 鈫?12.4), but PyTorch is **`2.x.x+cu130`** (built against **CUDA 13.0**). The extension build **must** use the same major CUDA line as the PyTorch wheel.

**Practical fix (match the machine鈥檚 12.4 toolkit):** reinstall PyTorch with **`cu124`**, then rebuild:

```bash
conda activate vsr310
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
python -c "import torch; print(torch.__version__, torch.version.cuda)"
# expect: ...+cu124 and cuda 12.4

export CUDA_HOME=/usr/local/cuda   # or /usr/local/cuda-12.4
export PATH="$CUDA_HOME/bin:$PATH"
export BLOCK_SPARSE_ATTN_CUDA_ARCHS="80"
cd Block-Sparse-Attention && pip install . --no-build-isolation
```

**Alternative:** install a **CUDA 13.x** toolkit, set `CUDA_HOME` to it, and keep `+cu130` PyTorch 鈥?only if your **NVIDIA driver** is new enough (the 鈥渄river too old鈥?warning often means `+cu130` is a poor match for this host; **`cu124` is usually safer** on shared servers).

### `UserWarning: The NVIDIA driver on your system is too old`

The PyTorch **wheel鈥檚** bundled CUDA expects a **newer driver** than the one reported. Use a **lower** CUDA wheel (e.g. **`cu124`**) that matches both the driver and `/usr/local/cuda`.

### `nvcc fatal : Unsupported gpu architecture 'compute_120'`

Upstream **Block-Sparse-Attention** does **not** rely on `TORCH_CUDA_ARCH_LIST` alone. Its `setup.py` reads **`BLOCK_SPARSE_ATTN_CUDA_ARCHS`** (default `80;90;100;110;120`). It always appends a **forward-compat PTX** line for the **largest** arch in that list, so with the default you get **`compute_120`**, which **CUDA 12.4 `nvcc` does not support**.

**Fix:** override the arch list **before** `pip install .` (Ampere / RTX A6000: keep `80` so kernels match the existing `*_sm80.cu` sources):

```bash
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export BLOCK_SPARSE_ATTN_CUDA_ARCHS="80"
export MAX_JOBS=8
cd ~/responsitory/CV/Block-Sparse-Attention
rm -rf build/  # optional: clean failed objects
pip install . --no-build-isolation
```

Optional Hopper on the same build machine: `80;90` (still **no** `120` unless your `nvcc` is new enough).

`TORCH_CUDA_ARCH_LIST` can stay unset or `8.6`; the critical variable for this repo is **`BLOCK_SPARSE_ATTN_CUDA_ARCHS`**.

### `ModuleNotFoundError: No module named 'torch'` during `pip install .`

`pip install .` uses an **isolated build env** by default; `setup.py` imports **torch** before that env has it.

Install from the same conda env where PyTorch is already installed, **without** build isolation:

```bash
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export BLOCK_SPARSE_ATTN_CUDA_ARCHS="80"
export MAX_JOBS=8
cd ~/responsitory/CV/Block-Sparse-Attention
pip install . --no-build-isolation
```

Alternative: `python setup.py install` (uses current interpreter, no isolated env).
