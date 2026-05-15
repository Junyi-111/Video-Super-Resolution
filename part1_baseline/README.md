# Part 1：经典基线（插值、SRCNN、时域融合）

对应课程 **Section 5.1**：双三次 / Lanczos、SRCNN、双三次上采样后的**时域加权平均**、可选 **Unsharp Mask**。  
**工作目录**：请在仓库根目录 `VideoHR/` 下执行本文所有命令（保证 `python -m part1_baseline...` 可找到包）。

---

## 0. 准备数据与路径

1. 将课程数据放入 **`dataset/`**（结构说明见 [dataset/README.md](../dataset/README.md)）。
2. 在仓库根目录执行，**列出本机可用的输入路径**（勿手抄不存在的 `wild.mp4`、`000` 等）：

```bash
python scripts/print_dataset_layout.py
```

3. 从输出中复制 **Wild 视频**、**某一 REDS clip 目录**或 **某一 Vimeo 子目录**的完整路径，下文记为 **`<INPUT>`**。

---

## 1. 训练（SRCNN）

### 1.1 默认：`official` 模式（推荐）

使用 `dataset/` 下自动发现的 **REDS-sample** 与 **vimeo-RL** 帧：把每帧当作 HR，双三次下采样再双三次上采样得到 SRCNN 输入，训练 **bicubic(LR) → HR** 的映射。

在项目根目录执行：

```bash
# 完整默认：REDS + Vimeo、4×、50 epoch、权重写入 weights/srcnn/srcnn_x4.pth
python -m part1_baseline.train_srcnn --train_mode official --out_ckpt weights/srcnn/srcnn_x4.pth
```

常用可调参数：

| 参数 | 默认 | 说明 |
|------|------|------|
| `--dataset_root` | 仓库下 `dataset/` | 数据根目录；填绝对路径亦可 |
| `--official_source` | `both` | `reds` 只用 REDS；`vimeo` 只用 Vimeo；`both` 两者 |
| `--scale` | `4` | 与推理、作业倍率一致 |
| `--epochs` | `50` | 训练轮数 |
| `--batch_size` | `8` | 显存不足可改为 4 |
| `--lr` | `1e-4` | Adam 学习率 |
| `--patch_hr` | `64` | official 模式随机裁剪 HR 边长（需可被 `scale` 整除） |
| `--out_ckpt` | `weights/srcnn/srcnn_x4.pth` | 输出 checkpoint |

示例（仅 REDS、短训试跑）：

```bash
python -m part1_baseline.train_srcnn --train_mode official --official_source reds --epochs 10 --out_ckpt weights/srcnn/srcnn_reds_x4.pth
```

### 1.2 `folders` 模式（自备 LR/HR 成对图像）

目录约定（默认路径，可改）：

- `dataset/pairs/train_lr/`：LR 图像，与 HR **文件名一一对应**
- `dataset/pairs/train_hr/`：HR 图像（`.png` / `.jpg` / `.jpeg`）

```bash
python -m part1_baseline.train_srcnn --train_mode folders --lr_dir dataset/pairs/train_lr --hr_dir dataset/pairs/train_hr --out_ckpt weights/srcnn/srcnn_x4.pth
```

若目录为空会报错；需先放入成对样本。

---

## 2. 测试 / 推理（生成 HR 视频）

使用 **`<INPUT>`**：来自 `print_dataset_layout.py` 的 Wild **mp4** 路径，或某一 **REDS / Vimeo 帧目录**路径。

**默认输出目录**：`part1_baseline/outputs/`（可用 `--out_dir` 覆盖）。

### 2.0 批量：不写 `--input`，一次跑完全部低码率项

与 `--input` **二选一**：使用 **`--input_all`** 时，会在以下根目录中递归收集**全部**任务并逐项推理（与 `scripts/process_mandatory.py` 使用同一套发现逻辑）：

- 默认根目录：**`dataset/`** 与 **`data/raw/`**（若存在）；
- 每项包括：`*.mp4` / `*.avi` / `*.mov` / `*.mkv`、所有 **REDS** clip 帧目录、所有 **Vimeo** 子片段目录。

输出：在 **`--out_dir`** 下按「数据来源 + 相对路径」建子目录，与 `part1_baseline/outputs/mandatory/` 结构一致，避免不同 clip 同名覆盖。

```bash
# 全部数据跑双三次 4×（示例：输出根目录用 batch_run1）
python -m part1_baseline.run_part1 --input_all --mode bicubic --scale 4 --out_dir part1_baseline/outputs/batch_run1 --clip_fps 24

# 只扫描 dataset（不扫 data/raw）
python -m part1_baseline.run_part1 --input_all --scan_roots dataset --mode lanczos --scale 4 --out_dir part1_baseline/outputs/batch_lanczos

# 全部跑 SRCNN（需先有权重）
python -m part1_baseline.run_part1 --input_all --mode srcnn --scale 4 --srcnn_ckpt weights/srcnn/srcnn_x4.pth --out_dir part1_baseline/outputs/batch_srcnn
```

**注意**：数据量大时耗时会很长；可先 `--scan_roots dataset` 且用子集目录试跑。`--input` 与 `--input_all` **必须且只能**选一个。

### 2.1 空间基线：双三次 / Lanczos

```bash
python -m part1_baseline.run_part1 --input "<INPUT>" --scale 4 --mode bicubic
python -m part1_baseline.run_part1 --input "<INPUT>" --scale 4 --mode lanczos
```

帧目录建议显式指定帧率（写入 mp4 时）：

```bash
python -m part1_baseline.run_part1 --input "<INPUT>" --scale 4 --mode bicubic --clip_fps 24
```

### 2.2 时域基线：加权平均 + 可选 Unsharp

```bash
python -m part1_baseline.run_part1 --input "<INPUT>" --scale 4 --mode temporal --temporal_window 5
python -m part1_baseline.run_part1 --input "<INPUT>" --scale 4 --mode temporal --temporal_window 5 --unsharp
```

### 2.3 SRCNN 推理（需先完成第 1 节训练）

```bash
python -m part1_baseline.run_part1 --input "<INPUT>" --scale 4 --mode srcnn --srcnn_ckpt weights/srcnn/srcnn_x4.pth
```

`--srcnn_ckpt` 必须与训练时 `--out_ckpt` 指向同一文件（或你保存的其它 `.pth`）。

### 2.4 `run_part1` 参数速查

| 参数 | 说明 |
|------|------|
| `--input` | 必填；LR **视频文件** 或可识别 **帧目录** |
| `--out_dir` | 默认 `part1_baseline/outputs/` |
| `--scale` | 默认 `4` |
| `--mode` | `bicubic` \| `lanczos` \| `srcnn` \| `temporal` |
| `--srcnn_ckpt` | 仅 `mode=srcnn` 时必填 |
| `--temporal_window` | `temporal` 时窗口长度（建议奇数，默认 5） |
| `--unsharp` | `temporal` 后对结果做反锐化 |
| `--clip_fps` | 输入为帧目录时序列帧率，默认 `24` |

---

## 3. 批量跑必做数据（Part 1 + 可选 Part2）

在项目根目录：

```bash
python scripts/process_mandatory.py
```

- 会扫描 **`dataset/`** 与 **`data/raw/`** 下的视频与帧 clip。
- Part 1 输出目录：`part1_baseline/outputs/mandatory/...`（按数据来源分子目录）。

仅跑 Part 1、跳过 Real-ESRGAN：

```bash
python scripts/process_mandatory.py --skip_part2
```

其它参数见：`python scripts/process_mandatory.py --help`。

---

## 4. 定量测试（PSNR / SSIM / LPIPS 等）

`scripts/evaluate.py` 需要**成对的帧文件夹**（文件名排序后一一对齐）。

### 4.1 从视频导出帧（可选）

若只有 mp4，可先导出为 PNG 再评测：

```bash
python scripts/video_to_frames.py --input part1_baseline/outputs/某输出.mp4 --out_dir dataset/test/pred_run1
```

将对应 GT 帧放入 `dataset/test/gt`（或与 `pred` 同名对齐的另一目录）。

### 4.2 运行指标

在项目根目录：

```bash
python scripts/evaluate.py --pred_dir dataset/test/pred --gt_dir dataset/test/gt --metrics psnr,ssim,lpips,tlpips --csv part1_baseline/outputs/metrics_part1.csv
```

需要 **FID** 时加上 `fid`（依赖 `pytorch-fid`，可能较慢）。无 GT 的 Wild 片段可只做主观对比或仅用 `tlpips` 等无参考协议（需在报告中说明）。

---

## 5. 可选：合成小视频试跑

```bash
python scripts/make_synthetic_video.py
```

默认写入 `dataset/wild/demo_lr.mp4`（若目录不存在会自动创建）。再对该路径执行第 2 节命令。

---

## 6. 依赖与模块说明

- **依赖**：见仓库根目录 [requirements.txt](../requirements.txt) 与 [README.md](../README.md) 中的环境说明（`torch`、`opencv-python`、`imageio`、`imageio-ffmpeg` 等）。
- **`clip_io.py`**：`dataset/` 下 REDS / Vimeo / Wild 的发现与 `format_dataset_hints`（路径错误时的提示）。

---

## 7. 报告撰写提示

说明 Part 1 **无显式光流对齐**、**无循环时序网络**；时域融合仅为加权平均，易糊、易闪烁；与 Part 2（BasicVSR 等）在 **PSNR/SSIM** 与 **放大 patch 主观图** 上对比。

---

## 8. Windows 路径说明

路径含空格或中文时，请用**英文双引号**包住 `--input "<INPUT>"`。若终端中文乱码，可先执行 `chcp 65001` 或设置环境变量 `PYTHONIOENCODING=utf-8`。
