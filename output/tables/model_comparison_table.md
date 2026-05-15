# 模型指标对比（自动生成）

说明：**PSNR / SSIM 越高越好**；**LPIPS / tLPIPS / FID / Temporal MAE 越低越好**。
Part 3 数据集汇总：**有 GT** 片段上对各指标取均值（常为 Wild）；**无 GT** 片段单独报告 **tLPIPS（无参考）** clip 均值，勿与有参考 tLPIPS 混比。

## 1. Wild 视频（伪 GT：wild_real.mp4）：Part1 / Part2 / Part3 同窗对比

| Model | PSNR↑ | SSIM↑ | LPIPS↓ | tLPIPS↓ | FID↓ | Temp. MAE↓ | Frames |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Part 3-A (Rectified-flow) | 13.9390 | 0.8184 | 0.1453 | 0.0948 | — | 6.4678 | 229 |
| Part 3-B (Flow-stabilized) | 13.9781 | 0.8204 | 0.1439 | 0.0953 | — | 6.3549 | 229 |
| Part 3-B (Temporal + Real-ESRGAN blend) | 13.8252 | 0.8141 | 0.1638 | 0.1229 | — | 5.6631 | 229 |
| Part 3-C (Uncertainty-adaptive) | 13.9199 | 0.8169 | 0.1436 | 0.0921 | — | 6.4898 | 229 |
| Part 3-D (Streaming distilled diffusion) | 14.0615 | 0.8250 | 0.1344 | 0.1022 | — | 7.4867 | 229 |

## 2. Part 3 各方向 — 全数据集汇总（有 GT / 无 GT 分列）

| Model | # clips 有 GT | mean PSNR↑ | mean SSIM↑ | mean LPIPS↓ | mean tLPIPS↓(有 GT) | mean FID↓ | mean Temp.MAE↓ | # clips 无 GT | mean tLPIPS(nr)↓ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Part 3-A (Rectified-flow) | 1 | 13.9390 | 0.8184 | 0.1453 | 0.0948 | — | 6.4678 | 275 | 0.2282 |
| Part 3-B (Flow-stabilized) | 1 | 13.9781 | 0.8204 | 0.1439 | 0.0953 | — | 6.3549 | 275 | 0.2369 |
| Part 3-B (Temporal + Real-ESRGAN blend) | 1 | 13.8252 | 0.8141 | 0.1638 | 0.1229 | — | 5.6631 | 275 | 0.2284 |
| Part 3-C (Uncertainty-adaptive) | 1 | 13.9199 | 0.8169 | 0.1436 | 0.0921 | — | 6.4898 | 275 | 0.2333 |
| Part 3-D (Streaming distilled diffusion) | 1 | 14.0615 | 0.8250 | 0.1344 | 0.1022 | — | 7.4867 | 0 | — |
