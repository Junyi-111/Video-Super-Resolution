# Part 2：BasicVSR(++) 与 Real-ESRGAN

## 只有老师给的 REDS-sample / vimeo-RL **帧目录**时怎么做

不必关心文档里的「Wild」——那是可选的自有视频。你只有 `./dataset` 里 REDS 与 Vimeo 的**帧序列**时，按下面顺序即可：

1. 在仓库根目录执行，**复制终端里打印出的某一帧目录完整路径**（不要手抄示例名）：
   ```bash
   python scripts/print_dataset_layout.py
   ```
2. 任选**一个** REDS clip 目录或 Vimeo 子目录，先打成 mp4（`--fps` 可与 Part1 一致，常用 `24`）：
   ```bash
   python scripts/frames_to_video.py --input "上一步复制的帧目录路径" --out part2_sota/outputs/my_clip.mp4 --fps 24
   ```
3. 用该 mp4 跑 Part2（Real-ESRGAN、BasicVSR++ 的 `--input` 都填这个文件）：
   ```bash
   python -m part2_sota.infer_realesrgan --input part2_sota/outputs/my_clip.mp4 --model_path weights/realesrgan/RealESRGAN_x4plus.pth
   python -m part2_sota.infer_basicvsr --input part2_sota/outputs/my_clip.mp4 --model_name basicvsr_pp
   ```

说明：帧目录里是**一整段 clip**（REDS 为纯数字文件名；Vimeo 为 `im1.png`…），与 Part1 能当 `--input` 的是同一类路径。

**一次性全部打成 mp4（与 `print_dataset_layout` 列表一致，留作 Part2 备用）**：在仓库根目录执行：

```bash
python scripts/batch_frames_to_video.py --fps 24
```

默认写入 `part2_sota/inputs_mp4/`，每个 clip 一个文件，名为相对 `dataset/` 的路径把 `/` 换成 `_` 再加 `.mp4`（例如 `REDS-sample_002.mp4`、`vimeo-RL_00018_0043.mp4`）。数据多时会较慢、占磁盘；中断后可加 `--skip_existing` 续跑。只要 REDS 可加 `--reds_only`，只要 Vimeo 可加 `--vimeo_only`。

**批量 Real-ESRGAN（对 `inputs_mp4` 下全部 `.mp4` 推理，模型只加载一次）**：先按下文安装依赖并下载权重，再在仓库根目录执行：

```bash
python scripts/batch_infer_realesrgan.py
```

默认输出目录 `part2_sota/outputs/realesrgan/`，与输入同名的 `*.mp4`。中断续跑可加 `--skip_existing`；显存不足可加 `--tile 400`（或 200）等。

**批量 BasicVSR++**（需先安装下文 **BasicVSR++（MMagic）** 中的依赖）：在仓库根目录执行：

```bash
python scripts/batch_infer_basicvsr.py
```

默认读取 `part2_sota/inputs_mp4/*.mp4`，**每个视频一个子目录** `part2_sota/outputs/basicvsr/<文件名不含扩展名>/`（MMagic 结果在该子目录内）。`--skip_existing` 可续跑；若 `MMagicInferencer` 不支持 `model_setting` 可加 `--no_model_setting`。

## Real-ESRGAN 依赖与权重

1. **先安装 PyTorch**（与 CUDA 匹配）：`pip install torch torchvision`（见根目录 [README.md](../README.md)）。

2. **再安装 Real-ESRGAN 栈**（不要一行 `pip install basicsr ...`，在 Windows 上容易失败）：

   **原因简述：**
   - `basicsr` 构建时会 `import torch`，pip 的**隔离构建环境**里没有 torch → 报 `No module named 'torch'`。  
     **解决：** 加 `--no-build-isolation`，让构建使用当前环境里的 torch。
   - PyPI 上 `basicsr` 声明依赖 `tb-nightly`，在不少环境下**无法解析**。  
     **解决：** 对 `basicsr` / `realesrgan` 使用 `--no-deps`，并手动安装常用依赖。
   - PyPI 的 `basicsr==1.4.2` 与**较新** `torchvision` 不兼容（`functional_tensor` 已移除）。  
     **解决：** 从 **GitHub 主线**安装 BasicSR（已改为 `torchvision.transforms.functional`）。

   **推荐命令：**

   bash
   pip install git+https://github.com/xinntao/BasicSR.git --no-build-isolation --no-deps
   pip install addict future lmdb scikit-image tensorboard pyyaml requests
   pip install realesrgan==0.3.0 --no-deps
   

   也可直接使用仓库内 [requirements-part2-realesrgan.txt](../requirements-part2-realesrgan.txt) 中的说明。

   **可选：** `pip install facexlib gfpgan`（人脸相关功能；若 pip 试图把 `numpy` 降到 1.20 等老版本，请中止并改用 `conda` 或固定 `numpy` 版本后再装）。

3. 下载权重到 `weights/realesrgan/`（文件名需与脚本默认一致，或通过参数覆盖）：

- [RealESRGAN_x4plus.pth](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth)

目录示例：

```
weights/realesrgan/RealESRGAN_x4plus.pth
```

4. 运行（**默认** `part2_sota/outputs/realesrgan_out.mp4`；可用 `--out` 覆盖）：

Real-ESRGAN **只接受视频文件**。请先运行 `python scripts/print_dataset_layout.py`，用其中**真实路径**替换下面占位符（勿手抄不存在的 `wild/wild.mp4`、`REDS-sample/000` 等）。

- **Wild**：若列表中有 **Wild 视频**，可直接将其 `.mp4` 路径作为 `--input`。
- **REDS-sample / Vimeo**（帧目录）：先把某一 clip **封装成 mp4**，再推理：

```bash
# 将 <REDS_OR_VIMEO_CLIP_DIR> 换成 print_dataset_layout 打印的某一帧目录
python scripts/frames_to_video.py --input "<REDS_OR_VIMEO_CLIP_DIR>" --out part2_sota/outputs/sample_from_clip.mp4 --fps 24

python -m part2_sota.infer_realesrgan --input part2_sota/outputs/sample_from_clip.mp4 --model_path weights/realesrgan/RealESRGAN_x4plus.pth
```

若已有 Wild 下或其它位置的 `.mp4`，可一步完成：

```bash
python -m part2_sota.infer_realesrgan --input "<本机实际 mp4 路径>" --model_path weights/realesrgan/RealESRGAN_x4plus.pth
```

## BasicVSR++（MMagic）

以下为 **OpenMMLab 官方文档**中的推荐顺序（与「先装一堆 pip 再 mim」相比，更容易一次下载到**匹配的预编译 wheel**，尤其是 **mmcv**）：

- [MMagic：Installation（Best practices）](https://mmagic.readthedocs.io/en/latest/get_started/install.html)
- [MMCV：Installation（Install with mim）](https://mmcv.readthedocs.io/en/latest/get_started/installation.html)
- [MMEngine 文档](https://mmengine.readthedocs.io/en/latest/)

### 推荐环境搭建顺序（mmcv → mmengine → mmagic）

1. **新建干净虚拟环境**（建议 **Python 3.10 或 3.11 x64**，避免过新/过旧导致无 wheel）：
   ```bash
   conda create -n mmagic310 python=3.10 -y
   conda activate mmagic310
   ```

2. **先安装 PyTorch**（按 [pytorch.org](https://pytorch.org/get-started/locally/) 选择与显卡、CUDA 一致的命令）。装好后确认：
   ```bash
   python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
   ```

3. **用 MIM 安装与当前 PyTorch/CUDA 匹配的 MMCV 2.x**（关键：日志里应出现下载 **`mmcv-*.whl`**，而不是长期停在编 `.tar.gz`）：
   ```bash
   python -m pip install -U pip
   pip install -U openmim
   mim install "mmcv>=2.0.0"
   ```
   国内 pip 可加镜像：`-i https://pypi.tuna.tsinghua.edu.cn/simple`；**mmcv 的 wheel 仍由 mim 从 OpenMMLab 索引解析**，若始终落到源码包，请核对 PyTorch 与 CUDA 是否为常见组合，或参阅 [MMCV 安装页](https://mmcv.readthedocs.io/en/latest/get_started/installation.html) 的 **pip + `-f` find-links** 方式手动指定 `cu1xx/torchx.x` 索引。

4. **再装 MMEngine**：
   ```bash
   mim install mmengine
   ```
   或 `pip install mmengine`。

5. **最后装 MMagic**（此时 **mmcv / mmengine 已就绪**，pip 较少再去编脆弱依赖）：
   ```bash
   mim install mmagic
   ```
   或 `pip install mmagic`。

6. **验证**：
   ```bash
   python -c "import mmcv, mmengine, mmagic; print(mmcv.__version__, mmengine.__version__, mmagic.__version__)"
   ```

**Windows + Anaconda 建议**：在步骤 5 之前先装齐易在 Windows 上编译失败的二进制依赖（与下文排错一致）：
```bash
conda install -c conda-forge av python-lmdb
```

**Docker（Linux，可复现的「一次配齐」）**：官方提供 [docker/Dockerfile](https://github.com/open-mmlab/mmagic/blob/main/docker/Dockerfile)，适合不想在宿主机折腾依赖时使用。

---

**若安装过程中在构建 `av`（PyAV）时报错 `av\logging.pyx` / `noexcept` / `Cython`：**  
这是 **从源码编译 PyAV** 时与 **Cython 3.x** 不兼容导致的。优先：`conda install -c conda-forge av`，或 `pip install "av>=12"` / `pip install av --only-binary av`，或 `pip install "cython<3.1"` 后再装 `av`（权宜之计）。

**若构建 `lmdb` 报错 `cannot be absolute`、路径含 `/home/runner/.../mdb.c`：**  
优先：`conda install -c conda-forge python-lmdb`（或 `lmdb`）；仅用 pip 时可试 `pip install lmdb --only-binary lmdb`。

2. 首次推理时 MMagic 可能自动下载配置与权重；也可从 [OpenMMLab 模型库](https://magic.openxlab.org.cn/mmagic) 查询 `basicvsr_pp` / `basicvsr` 的 checkpoint 路径。

3. 运行（**默认**输出目录 `part2_sota/outputs/basicvsr/`）：

```bash
python -m part2_sota.infer_basicvsr --input path/to/in.mp4 --model_name basicvsr_pp
```

4. **批量**（与 `inputs_mp4` 中全部样本一致）：`python scripts/batch_infer_basicvsr.py`（见上文「批量 BasicVSR++」）。

## 说明

- 输入支持常见视频格式（依赖 ffmpeg）。
- Part 2 与 Part 1 的倍率默认均为 **4×**；若数据集为其他倍率，请在报告中说明预处理（裁剪/填充）或改用对应权重。
