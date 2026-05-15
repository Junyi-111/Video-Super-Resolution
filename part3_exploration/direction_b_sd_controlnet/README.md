# Direction B 鈥?Consistent enhancement (SD + ControlNet-Tile)

Course alignment: **Stable Diffusion** as generative prior + **ControlNet-Tile** for structure, plus **optical-flow temporal stabilization** on the Real-ESRGAN branch and an optional **temporal / generative blend** (texture mask) for cheap consistency without full video diffusion.

## Weights (what is trained vs pretrained)

This direction **does not** add a new small CNN checkpoint under `weights/part3_b/`. Batch metrics instead consume **Part 2 outputs** (Real-ESRGAN / BasicVSR / etc.) already produced on your dataset. **SD1.5 + ControlNet-Tile** use **public pretrained** diffusion weights when you run the keyframe script; flow stabilization and blending are **deterministic, data-conditioned** pipelines on those videos.

## Scripts

| Script | Role |
|--------|------|
| `run_controlnet_tile_keyframes.py` | SD1.5 + ControlNet-Tile on **sampled keyframes** (slow; needs `diffusers`, GPU). |
| `run_flow_stabilize.py` | Full-video **flow-guided** stabilization of Real-ESRGAN (fast, OpenCV). |
| `run_temporal_generative_blend.py` | Full-video **Temporal+Unsharp** vs **Real-ESRGAN** blend (fast). |
| `run_batch.py` | Runs **flow + blend** on every clip under `dataset/inputs_mp4` (+ Wild). Keyframe diffusion is **not** run in batch (use the keyframe script manually for figures). |

## Commands

```bash
# Full-dataset enhancement (recommended for metrics table)
python -m part3_exploration.direction_b_sd_controlnet.run_batch

# Keyframe ControlNet-Tile (optional, heavy)
python -m part3_exploration.direction_b_sd_controlnet.run_controlnet_tile_keyframes \
  --input part2_sota/outputs/wild_real_lr/realesrgan_x4.mp4 \
  --out_dir part3_exploration/outputs/direction_b_sd_controlnet/diffusion_keyframes
```

Outputs live in `part3_exploration/outputs/direction_b_sd_controlnet/`.
