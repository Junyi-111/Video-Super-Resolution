# Direction C 鈥?Uncertainty-aware refinement

**BasicVSR** (conservative video branch) vs **Real-ESRGAN** (generative branch), fused with a **pixel-wise uncertainty** map: texture / disagreement / temporal artifacts push toward Real-ESRGAN; **text-like** and **face** regions bias toward BasicVSR.

## Weights (training on data)

**`weights/part3_c/fusion_head.pth`** is produced by **`train_fusion`** on batches sampled from the same degradation logic as Direction A, using bicubic vs RefineNet(bicubic) branches and freezing RefineNet. Treat it as a **data-trained** fusion head, not a downloaded artifact.

## Train FusionHead (~2h GPU, after RefineNet)

Training uses **bicubic** as the conservative branch and **RefineNet(bicubic)** as the generative branch (same resolution as HR). At inference you can fuse **BasicVSR + Real-ESRGAN** (or RefineNet) with the same head.

```bash
python -m part3_exploration.direction_c_uncertainty.train_fusion \
  --epochs 15 --batch_size 8 --freeze_refinenet \
  --refinenet_ckpt weights/part3_a/refinenet_x4.pth \
  --out_ckpt weights/part3_c/fusion_head.pth
```

## Inference (learned fusion)

```bash
python -m part3_exploration.direction_c_uncertainty.infer_fusion \
  --basic part2_sota/outputs/basicvsr/wild_real_lr.mp4 \
  --gen part2_sota/outputs/wild_real_lr/realesrgan_x4.mp4 \
  --out part3_exploration/outputs/direction_c_uncertainty/wild_real_lr_fusion_x4.mp4 \
  --fusion_ckpt weights/part3_c/fusion_head.pth \
  --mask_out part3_exploration/outputs/direction_c_uncertainty/wild_real_lr_fusion_alpha.mp4
```

## One-shot 5h pipeline (A 鈫?C)

```bash
python -m part3_exploration.run_train_5h --epochs_a 8 --epochs_c 15
python -m part3_exploration.run_train_5h --skip_a --eval_wild   # only C + Wild eval
```

## Batch (rule-based mask, no training)

```bash
python -m part3_exploration.direction_c_uncertainty.run_batch
python -m part3_exploration.direction_c_uncertainty.run_batch --limit 10
```

Outputs: `part3_exploration/outputs/direction_c_uncertainty/<clip>_uncertainty_adaptive_x4.mp4` (+ `_alpha.mp4`).

## Single clip

```bash
python -m part3_exploration.direction_c_uncertainty.run_adaptive \
  --lr dataset/inputs_mp4/REDS-sample_002.mp4 \
  --realesrgan part2_sota/outputs/realesrgan/REDS-sample_002.mp4 \
  --basicvsr part2_sota/outputs/basicvsr/REDS-sample_002.mp4 \
  --out /tmp/out.mp4 --mask_out /tmp/alpha.mp4
```
