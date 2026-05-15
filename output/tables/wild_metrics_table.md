Wild 片段指标（与 `wild_metrics.csv` 一致；**235** 帧；**eval_policy** `unified_streaming_vsr_crop_minT`；**crop_twh** 1280×2176）。

| Method | PSNR (higher) | SSIM (higher) | Temporal MAE (lower) |
| --- | ---: | ---: | ---: |
| LR input | 14.4595 | 0.8028 | 7.4458 |
| Bicubic | 14.2616 | 0.8108 | 7.0879 |
| Lanczos | 14.2609 | 0.8072 | 7.1296 |
| Temporal+Unsharp | 13.9791 | 0.7969 | 4.2974 |
| Real-ESRGAN | 14.3791 | 0.8423 | 6.4277 |
| BasicVSR | 14.0972 | 0.7521 | 7.9398 |
| Part3 Hybrid | 13.8154 | 0.8144 | 5.5845 |
| Part3 Flow-Stabilized | 13.9665 | 0.8208 | 6.2615 |
| Part3 Uncertainty-Adaptive | 13.9082 | 0.8173 | 6.3957 |

LPIPS / tLPIPS 见 `wild_lpips_metrics.csv` 与对应表格。
