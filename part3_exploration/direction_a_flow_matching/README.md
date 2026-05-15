# Direction A 鈥?Flow Matching (exploration)

This folder hosts **Direction A** in the course report: *Generative VSR / rectified trajectories*.

## Weights (training on data)

The deliverable checkpoint **`weights/part3_a/refinenet_x4.pth`** is produced by **`train_refinenet`** on synthetic LR鈥揌R pairs built from course / REDS鈥揤imeo鈥搒tyle sharp frames and the degradation pipeline (`data/degrade_dataset.py`). It is not a third-party drop-in weight file.

## What is implemented here

1. **Trainable RefineNet (5h plan, Direction A)** 鈥?small CNN refines bicubic脳4 toward HR on synthetic degradations from official REDS/Vimeo sharp frames.
2. **Rectified-flow surrogate (no training)** 鈥?pixel-space Euler bridge between bicubic and Real-ESRGAN HR.

Full **Flow Matching** with a Flux-class backbone is out of scope for this repo (compute and full generative training).  
The **rectified refinement** below is a **lightweight, fully reproducible surrogate**:

**Iterative rectified refinement** 鈥?treat the generative branch (Real-ESRGAN HR) as a target \(x_1\) and a conservative HR anchor \(x_0\) (bicubic upsample). Starting from \(z_0 = x_0\), apply a few Euler-style updates

\[
z_{k+1} = z_k + \frac{1}{K}\bigl(x_1 - z_k\bigr)
\]

so that the trajectory is a **straight ODE in pixel space** with **K discrete steps** (report: fewer steps than diffusion, deterministic bridge). This is **not** the same as training rectified-flow generative models, but it illustrates the *straighter trajectory / few-step refinement* narrative next to heavy diffusion in Direction B.

## Train RefineNet (~2鈥?h GPU)

```bash
python -m part3_exploration.direction_a_flow_matching.train_refinenet \
  --epochs 8 --batch_size 16 --patch_hr 128 \
  --out_ckpt weights/part3_a/refinenet_x4.pth
```

## Inference (RefineNet)

```bash
python -m part3_exploration.direction_a_flow_matching.infer_refinenet \
  --input dataset/wild_real_lr.mp4 \
  --out part3_exploration/outputs/direction_a_flow_matching/wild_real_lr_refinenet_x4.mp4 \
  --ckpt weights/part3_a/refinenet_x4.pth
```

## Batch (rectified surrogate, no ckpt)

From repo root:

```bash
python -m part3_exploration.direction_a_flow_matching.run_batch
python -m part3_exploration.direction_a_flow_matching.run_batch --limit 5
```

Outputs: `part3_exploration/outputs/direction_a_flow_matching/<clip>_rectified_flow_x4.mp4`

## Dependencies

Training/inference: PyTorch + OpenCV. Rectified batch: OpenCV + NumPy only.
