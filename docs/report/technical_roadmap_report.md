# Technical Roadmap Report: Video Super-Resolution

## Abstract

This project studies video super-resolution (VSR) under a practical course setting: given low-resolution videos, recover sharper high-resolution outputs while preserving temporal consistency. We implemented a three-stage pipeline. Part 1 builds classical and lightweight baselines, including interpolation-based upsampling and a simple SRCNN-style reference. Part 2 evaluates stronger state-of-the-art methods, mainly Real-ESRGAN and BasicVSR. Part 3 explores several advanced directions beyond direct model inference, including rectified-flow/diffusion refinement, optical-flow temporal stabilization, uncertainty-aware adaptive fusion, and a teammate-provided distilled streaming diffusion direction.

Our experiments show that Real-ESRGAN is the strongest practical visual baseline on the mandatory videos, while optical-flow stabilization gives a useful temporal refinement without requiring new training. On the optional Vimeo-90K small benchmark, bicubic interpolation obtains the highest full-reference PSNR/SSIM because the synthetic degradation is mild and pixel-aligned, while Real-ESRGAN and the flow-stabilized variant produce better perceptual LPIPS. The full code and submission package are organized in the public repository: **https://github.com/Junyi-111/Video-Super-Resolution.git**.

## 1. Introduction

Video super-resolution aims to reconstruct high-resolution frames from low-resolution video input. Compared with single-image super-resolution, VSR has an additional temporal challenge: each frame should be sharp, but adjacent frames should also remain stable. A method that improves a single frame may still create flicker, hallucinated textures, or inconsistent details across time.

The project is organized into three mandatory parts. Part 1 establishes reproducible baselines. Part 2 applies stronger pretrained VSR or image restoration methods. Part 3 investigates whether additional temporal or generative refinement can improve the final result. We also include an optional Vimeo-90K small benchmark to satisfy the recommended standard-dataset requirement for a higher-scoring submission.

The repository is arranged so that each project part has its own code and local outputs, while global comparison tables, figures, and packaged submission videos are stored under the root `output/` folder and `videos.zip`.

## 2. Related Work

Classical interpolation methods such as bicubic and Lanczos are simple, fast, and deterministic baselines. They do not recover missing high-frequency detail, but they are important references because full-reference metrics such as PSNR and SSIM often reward conservative pixel-aligned reconstruction.

Early deep-learning super-resolution methods such as SRCNN introduced convolutional neural networks for single-image super-resolution. Later GAN-based methods, including ESRGAN and Real-ESRGAN, improved perceptual sharpness by learning realistic high-frequency textures. Real-ESRGAN is especially useful in this project because it is robust to real-world degradations and can be applied to arbitrary video frames.

For video-specific restoration, recurrent and propagation-based architectures such as BasicVSR and BasicVSR++ use neighboring frames to improve temporal consistency and recover detail. REDS and Vimeo-90K are common academic datasets for evaluating video restoration and video interpolation/super-resolution methods.

Generative models and diffusion models can produce visually plausible textures, but they may hallucinate incorrect details and introduce temporal instability. ControlNet-style conditioning and tiled diffusion can make diffusion more controllable, while distilled or streaming diffusion methods try to reduce inference cost. In Part 3 we therefore treat generative refinement as an exploratory direction rather than the default final pipeline.

## 3. Method

### 3.1 Part 1: Baseline Super-Resolution

Part 1 implements a set of reproducible baselines:

- Bicubic upsampling.
- Lanczos upsampling.
- A lightweight SRCNN-style baseline.
- Temporal/unsharp enhancement utilities for simple post-processing.

These methods are useful for sanity checking the data pipeline, output dimensions, frame ordering, and metric computation. Their outputs are stored under `part1_baseline/outputs/`.

### 3.2 Part 2: State-of-the-Art Models

Part 2 evaluates two stronger methods recommended by the project roadmap:

- **Real-ESRGAN:** frame-wise perceptual restoration with strong texture enhancement.
- **BasicVSR:** video-based propagation model designed to use temporal information across frames.

Real-ESRGAN was the most reliable visual method in our experiments. BasicVSR was also run on the available datasets, but on our current setting it did not consistently outperform Real-ESRGAN. This is likely because the available clips include real or mixed degradations rather than the exact degradation distribution expected by the BasicVSR checkpoint.

Part 2 outputs are stored under `part2_sota/outputs/`.

### 3.3 Part 3: Exploratory Refinement

Part 3 investigates whether additional processing can improve over direct Real-ESRGAN or BasicVSR outputs. The explored directions are:

| Direction | Idea | Main Purpose | Output Folder |
|---|---|---|---|
| A | Rectified-flow / diffusion-style refinement | Test generative texture recovery | `part3_exploration/outputs/direction_a_flow_matching/` |
| B | Optical-flow temporal stabilization | Reduce flicker and improve temporal consistency | `part3_exploration/outputs/direction_b_sd_controlnet/` |
| C | Uncertainty-aware adaptive fusion | Use conservative VSR in reliable regions and generative/detail enhancement in uncertain regions | `part3_exploration/outputs/direction_c_uncertainty/` |
| D | Distilled streaming diffusion | Teammate exploration of faster streaming generative VSR | `part3_exploration/outputs/direction_d_distilled_streaming/` |

Among these, Direction B is the most practical final Part 3 candidate because it is reproducible, does not require expensive training, and directly targets the main weakness of frame-wise enhancement: temporal flicker. Direction C is conceptually strong but did not produce a clear quantitative improvement in our tests. Direction D achieved promising perceptual and full-reference numbers in the current table, but it is dependency-heavy and needs more careful temporal validation before being treated as the main final method.

## 4. Experiments

### 4.1 Datasets

We evaluate on the mandatory course data and additional data:

- Mandatory REDS-sample clips.
- Mandatory Vimeo-LR clips.
- Wild/self-captured low-resolution videos.
- Optional Vimeo-90K small subset with 20 sampled clips and available ground truth.

For videos without ground-truth high-resolution frames, full-reference metrics such as PSNR and SSIM are not meaningful. In those cases, we rely on visual comparison, temporal stability analysis, and no-reference or proxy metrics where appropriate. For Vimeo-90K, ground truth is available, so PSNR, SSIM, LPIPS, and temporal MAE can be reported.

### 4.2 Metrics

We use the following metrics:

- **PSNR:** higher is better; measures pixel-level reconstruction fidelity.
- **SSIM:** higher is better; measures structural similarity.
- **LPIPS:** lower is better; perceptual similarity based on deep features.
- **Temporal MAE:** lower is better; measures temporal frame-difference stability.

PSNR and SSIM can favor smoother methods when the output is pixel-aligned with the ground truth. Perceptual methods such as Real-ESRGAN may have lower PSNR but better visual sharpness or lower LPIPS.

### 4.3 Vimeo-90K Small Benchmark

We ran a 20-clip Vimeo-90K subset as an optional standard benchmark. The summary table is:

| Method | Clips | PSNR higher | SSIM higher | LPIPS lower | Temporal MAE lower |
|---|---:|---:|---:|---:|---:|
| Bicubic | 20 | 27.1938 | 0.8315 | 0.3082 | 10.7551 |
| Real-ESRGAN | 20 | 25.0214 | 0.7895 | 0.1611 | 11.2486 |
| BasicVSR | 20 | 23.1032 | 0.7643 | 0.2197 | 16.4366 |
| Flow-Stabilized | 20 | 25.2300 | 0.7949 | 0.1593 | 10.8161 |

The table shows an important trade-off. Bicubic has the best PSNR and SSIM because it is conservative and closely matches the downsampled test distribution. Real-ESRGAN significantly improves LPIPS, indicating better perceptual similarity. The flow-stabilized method slightly improves Real-ESRGAN in PSNR, SSIM, LPIPS, and temporal MAE, making it the best practical enhancement among the learned/perceptual methods in this benchmark.

### 4.4 Part 3 Comparison

The current Part 3 comparison table is:

| Method | PSNR higher | SSIM higher | LPIPS lower | tLPIPS lower | Temporal MAE lower | Frames |
|---|---:|---:|---:|---:|---:|---:|
| Part 3-A Rectified-flow | 13.9390 | 0.8184 | 0.1453 | 0.0948 | 6.4678 | 229 |
| Part 3-B Flow-stabilized | 13.9781 | 0.8204 | 0.1439 | 0.0953 | 6.3549 | 229 |
| Part 3-B Temporal + Real-ESRGAN blend | 13.8252 | 0.8141 | 0.1638 | 0.1229 | 5.6631 | 229 |
| Part 3-C Uncertainty-adaptive | 13.9199 | 0.8169 | 0.1436 | 0.0921 | 6.4898 | 229 |
| Part 3-D Streaming distilled diffusion | 14.0615 | 0.8250 | 0.1344 | 0.1022 | 7.4867 | 229 |

Direction D has the best PSNR, SSIM, and LPIPS in this table, while the temporal blend has the lowest temporal MAE. However, Direction D also has worse temporal MAE than the other main directions, suggesting that its frame quality improvements may come with temporal instability. For the final submission narrative, we treat Direction B as the most stable and reproducible practical refinement, and Direction D as a promising additional exploration.

## 5. Qualitative Analysis

The qualitative comparison should focus on three visual aspects:

- **Sharpness:** Real-ESRGAN recovers stronger edges and textures than interpolation baselines.
- **Temporal consistency:** frame-wise enhancement can flicker, while optical-flow stabilization reduces unstable local changes.
- **Failure cases:** diffusion or generative refinement can create hallucinated textures, inconsistent detail, or over-sharpened regions.

Recommended figures for the final PDF:

- A pipeline flowchart showing Part 1, Part 2, and Part 3.
- Side-by-side frame comparisons for mandatory REDS/Vimeo/wild videos.
- Zoom-in patches for text, edges, and texture regions.
- A failure-case figure showing generative artifacts or temporal instability.

Existing figures and visual assets are organized under `output/figures/`, and processed video outputs are packaged in `videos.zip`.

## 6. Submission Package

The final submission repository contains:

- `README.md`: how to install dependencies, run each part, and locate results.
- `part1_baseline/`: baseline methods and outputs.
- `part2_sota/`: Real-ESRGAN and BasicVSR inference code and outputs.
- `part3_exploration/`: exploratory Part 3 methods and local outputs.
- `output/`: global comparison tables, figures, and optional benchmark outputs.
- `docs/requirements/`: original project requirement PDF.
- `docs/report/`: this technical report draft.
- `videos.zip`: packaged processed videos for Canvas submission.

The zip package excludes raw datasets and ground-truth videos where possible, and is intended to contain only processed/demo outputs needed for submission.

## 7. Conclusion

We completed all three required project parts and added an optional Vimeo-90K small benchmark. The experiments show that Real-ESRGAN is the strongest direct perceptual baseline, while optical-flow stabilization provides the most practical Part 3 improvement by improving temporal behavior without requiring expensive retraining. The uncertainty-aware and diffusion-based directions are valuable explorations, but they did not consistently outperform the simpler stabilized pipeline across all metrics and visual checks.

The main limitation is that not all mandatory or wild videos have ground-truth high-resolution references, so PSNR and SSIM cannot be computed for every result. Another limitation is that generative methods can improve local texture but may introduce temporal artifacts. Future work should include larger-scale evaluation on REDS and Vimeo-90K, better temporal perceptual metrics, and training or fine-tuning a video-specific generative model rather than applying mostly frame-wise or tiled refinement.

## References To Include In Final CVPR LaTeX

- Dong et al., SRCNN, Learning a Deep Convolutional Network for Image Super-Resolution.
- Wang et al., ESRGAN, Enhanced Super-Resolution Generative Adversarial Networks.
- Wang et al., Real-ESRGAN, Training Real-World Blind Super-Resolution with Pure Synthetic Data.
- Chan et al., BasicVSR, The Search for Essential Components in Video Super-Resolution and Beyond.
- Chan et al., BasicVSR++, Improving Video Super-Resolution with Enhanced Propagation and Alignment.
- Nah et al., REDS dataset and NTIRE video restoration challenge materials.
- Xue et al., Vimeo-90K dataset.
- Ho et al., Denoising Diffusion Probabilistic Models.
- Zhang et al., ControlNet, Adding Conditional Control to Text-to-Image Diffusion Models.
- Relevant Direction D / streaming distilled diffusion and block-sparse attention references from the teammate implementation.

