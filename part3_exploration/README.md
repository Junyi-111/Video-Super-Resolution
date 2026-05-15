# Part 3 鈥?Exploration锛圤ptimization & Extension锛?
璇剧▼鎶ュ憡涓殑 **Exploration 鏂瑰悜** 鍒嗗埆鏀惧湪鐙珛瀛愮洰褰曚腑锛屼究浜庢挵鍐欏姣斾笌娑堣瀺銆?
| 鐩綍 | 鎶ュ憡涓殑鏂瑰悜 | 鏍稿績鎬濊矾 |
|------|----------------|----------|
| [`direction_a_flow_matching/`](direction_a_flow_matching/README.md) | **Direction A** 鈥?Generative VSR / Flow Matching | 鐢?**灏戞 Euler 寮忕洿绾胯建杩?* 鍦?bicubic HR 涓?Real-ESRGAN HR 涔嬮棿鍋氬彲澶嶇幇鐨勩€屾祦寮忋€嶇簿淇紙瀹屾暣 Flux / Pyramidal Flow 璁粌涓嶅湪鏈粨搴撹寖鍥达紝瑙佽鐩綍 README锛夈€?|
| [`direction_b_sd_controlnet/`](direction_b_sd_controlnet/README.md) | **Direction B** 鈥?Stable Diffusion + ControlNet-Tile | **SD1.5 + ControlNet-Tile** 鍏抽敭甯у疄楠岋紱鍏ㄩ噺鏁版嵁涓婅窇 **鍏夋祦鏃跺簭绋冲畾** 涓?**Temporal+Unsharp / Real-ESRGAN 绾圭悊娣峰悎**锛堜笌 Part 2 杈撳嚭琛旀帴锛夈€?|
| [`direction_c_uncertainty/`](direction_c_uncertainty/README.md) | **Direction C** 鈥?Uncertainty-aware | **BasicVSR vs Real-ESRGAN** 鍍忕礌绾т笉纭畾搴﹀浘 + 鏂囨湰/浜鸿劯淇濇姢锛岃嚜閫傚簲铻嶅悎銆?|
| [`direction_d_distilled_streaming/`](direction_d_distilled_streaming/README.md) | **Direction D** 鈥?Streaming distilled diffusion VSR | **鍗曟钂搁 + 娴佸紡鎺ㄧ悊** 鐨勭敓鎴愬紡 VSR 璺嚎锛堝亸閲嶅疄鏃朵笌闀垮簭鍒楋級锛涜璇ョ洰褰曠幆澧冧笌鏉冮噸璇存槑锛堥渶 CUDA 涓庝笂娓哥紪璇戜緷璧栵級銆?|

## 鏉冮噸璇存槑锛堣绋嬩氦浠樺彛寰勶級

Part 3 鍚勬柟鍚?README 涓垪鍑虹殑**鍙啓鍏ュ浣嶇殑妫€鏌ョ偣**锛坄weights/part3_*` 涓嬬殑 `.pth`銆乣.safetensors`銆乣.ckpt` 绛夛級鍧囪涓哄湪**璇剧▼鏁版嵁涓庨€€鍖?/ 钂搁绠＄嚎**涓婅缁冩垨钂搁寰楀埌锛岃€岄潪銆屼粎涓嬭浇鍗崇敤銆嶇殑绗笁鏂规垚鍝佹潈閲嶇洰褰曪細

- **Direction A**锛歚train_refinenet` 鍦ㄥ悎鎴?LR鈥揌R 瀵逛笂璁粌 鈫?`weights/part3_a/refinenet_x4.pth`銆?- **Direction C**锛歚train_fusion` 鍦ㄦ暟鎹笂璁粌铻嶅悎澶?鈫?`weights/part3_c/fusion_head.pth`锛堜緷璧?A 鐨勬鏌ョ偣浣滄暀甯堝垎鏀箣涓€锛夈€?- **Direction D**锛氭祦寮忚捀棣忎笌瑙ｇ爜鍣ㄧ浉鍏虫潈閲嶆斁鍦?`weights/part3_d/`锛岀敱璇剧▼瑙勫畾鐨勮捀棣?/ 寰皟娴佺▼鍦ㄦ暟鎹笂寰楀埌锛堟枃浠跺懡鍚嶄笌甯冨眬瑙佽鏂瑰悜 README锛夈€?- **Direction B**锛氫笉鏂板涓婅堪褰㈠紡鐨勩€屽皬缃戠粶妫€鏌ョ偣銆嶏紱鎵归噺鑴氭湰鍦?**Part 2 浜庢暟鎹泦涓婃帹鐞嗗緱鍒扮殑瑙嗛** 涓婂仛鍏夋祦绋冲畾涓庢贩鍚堛€係D1.5 / ControlNet-Tile 浠嶄娇鐢?*鍏紑棰勮缁?*鎵╂暎鏉冮噸浣滅敓鎴愬厛楠岋紙瑙?Direction B README锛夈€?
## 浠ｇ爜鎵撳寘锛堜笉鍚棰戜笌鏉冮噸鏂囦欢锛?
鍦ㄤ粨搴撴牴鐩綍鐢熸垚浠呭惈婧愮爜涓庨厤缃殑鍘嬬缉鍖咃紝**鍓旈櫎**锛?
- 甯歌瑙嗛鍚庣紑锛歚mp4` / `mov` / `avi` / `mkv` / `webm` / `m4v`锛?- 甯歌妫€鏌ョ偣鍚庣紑锛?*`.pth`銆乣.pt`銆乣.safetensors`銆乣.ckpt`**锛堜笉鎵撳寘浠讳綍璁粌/钂搁鏉冮噸锛岄渶鍗曠嫭澶囦唤鎴栨寜 README 閲嶆柊璁粌寰楀埌锛夈€?

```bash
# tar.gz锛堟帹鑽愶細GNU tar 涓€閬嶆壂鐩橈紝涓嶅厛 rsync 鍒?/tmp锛涢€氬父姣旀棫鐗?zip 鑴氭湰蹇緢澶氾級
bash scripts/package_cv_sources_no_video.sh
USE_PIGZ=1 bash scripts/package_cv_sources_no_video.sh   # 鑻ュ凡瀹夎 pigz锛屽鏍稿帇缂╂洿蹇?PACK_SKIP_BSA_BUILD=1 bash scripts/package_cv_sources_no_video.sh  # 鍘绘帀 B-S-A 鐨?build 鐩綍锛堣嫢瀛樺湪锛?EXCLUDE_GIT=1 bash scripts/package_cv_sources_no_video.sh

# zip锛坒ind | zip -@锛屼竴閬嶈鏂囦欢锛涢粯璁?ZIP_LEVEL=1 杈冨揩锛?bash scripts/package_cv_sources_no_video_zip.sh
ZIP_LEVEL=0 bash scripts/package_cv_sources_no_video_zip.sh   # 浠呮墦鍖呬笉鍘嬬缉锛屾渶蹇€佷綋绉渶澶?EXCLUDE_GIT=1 bash scripts/package_cv_sources_no_video_zip.sh
```

## 涓€閿細鍏ㄩ噺璺戝悇鏂瑰悜 + 姹囨€绘寚鏍?
鍦ㄤ粨搴撴牴鐩綍鎵ц锛堥渶宸插噯澶囧ソ `dataset/inputs_mp4`銆乣part1_baseline/outputs` 鐨?bicubic/temporal銆乣part2_sota/outputs` 鐨?Real-ESRGAN / BasicVSR锛?*Direction D 鍙﹂渶 GPU 鐜涓庡凡璁粌寰楀埌鐨?`weights/part3_d` 妫€鏌ョ偣**锛岃 Direction D README锛夛細

```bash
python -m part3_exploration.run_all_part3
```

鍙€夛細

```bash
python -m part3_exploration.run_all_part3 --limit 20          # 姣忔柟鍚戝彧澶勭悊鍓?20 涓?clip锛堣皟璇曠敤锛?python -m part3_exploration.run_all_part3 --skip_a --eval_skip_fid
python -m part3_exploration.run_all_part3 --skip_d         # 璺宠繃 Direction D锛堟棤宸茶缁冩鏌ョ偣鎴栦粎璺?A/B/C锛?python -m part3_exploration.run_all_part3 --skip_legacy_sync  # 涓嶅啓鍥?part3_exploration/outputs/ 鏃ц矾寰?```

### 杈撳嚭

- **鍚勬柟鍚戣棰?*锛歚part3_exploration/outputs/direction_*/<clip>_<suffix>.mp4`
- **鎸囨爣锛堥€?clip + 鎸夋柟鍚戝潎鍊硷級**锛?  - `output/tables/part3_metrics_per_clip.csv`
  - `output/tables/part3_metrics_summary.csv`
- 榛樿浼氭妸 Wild 涓?B/C 鐨勭粨鏋?**鍚屾澶嶅埗** 鍒?`part3_exploration/outputs/` 涓嬫棫鏂囦欢鍚嶏紝鍏煎鍘熸湁 `evaluate_wild_*.py` 涓庝綔鍥捐剼鏈€?
### 鍗曠嫭璇勬祴 Part 3锛堜笉閲嶆柊璺戞帹鐞嗭級

```bash
python -m part3_exploration.evaluate_part3_metrics --skip_fid
```

榛樿 **`--unified-eval`**锛堝彲鐢?**`--no-unified-eval`** 鎭㈠鏃ц涓猴級锛氬姣忎釜 clip 鎸?**LR 鎺ㄥ鐨?4脳 鐢诲竷 + 128 鍊嶆暟涓績瑁佸壀锛堜笌 Direction D 娴佸紡鎺ㄧ悊鐢诲竷涓€鑷达級** 瀵归綈鍚勬柟鍚戦娴嬩笌 GT锛屾椂闂撮暱搴﹀彇 **璇?clip 涓?GT 涓庡悇宸插瓨鍦ㄩ娴嬬殑鍏叡鏈€鐭抚鏁?*锛屼娇 A/B/C/D 鍦ㄥ悓涓€瑙嗛噹涓庡悓涓€鏃堕暱涓婃瘮鎸囨爣銆?
鏈?**GT mp4** 鏃讹紙`dataset/gt_mp4/<涓?inputs 鍚屽悕>.mp4`锛夛紝鏍锋湰 clip 浼氳绠?PSNR/SSIM/LPIPS 绛夛紱鏃?GT 鏃朵粎鍐?**tLPIPS 鏃犲弬鑰冧唬鐞?*銆俉ild 浣跨敤 `dataset/wild_real.mp4` 浣滀负 GT銆傞€?clip CSV 涓?**`eval_policy`** / **`crop_twh`** 璁板綍瀵归綈绛栫暐涓庤鍓悗 4脳 鐩爣瀹介珮銆?
### Part1 + Part2 + Wild 鎬昏〃锛坄metric.csv`锛?
涓?Part3 **鍚屼竴濂楃粺涓€璇勪及**锛堥粯璁ゅ紑鍚級锛?
```bash
python scripts/evaluate_project.py --skip_fid
# 鎴栨棫琛屼负锛?python scripts/evaluate_project.py --no-unified-eval --skip_fid
```

Wild 涓?**LR input** 浼氬厛鍙屼笁娆℃斁澶у埌 4脳 鐢诲竷鍐嶈鍒颁笌涓婅堪娴佸紡璇勪及鐩稿悓鐨?ROI锛屽啀涓庤鍒囧悗鐨?GT 瀵归綈銆傝緭鍑哄垪鍚?**`eval_policy`**銆?*`crop_twh`**銆?*`tlpips_noref`**锛堜笌 Part3 琛ㄤ竴鑷翠究浜庡鐓э級銆?
## 5 灏忔椂鍙缁冩柟妗堬紙A + C锛?
鍦ㄥ畼鏂?REDS/Vimeo **娓呮櫚甯?*涓婂悎鎴愰€€鍖?LR锛岃缁?**RefineNet锛圓锛?* 涓?**FusionHead锛圕锛?*锛汢 浠嶄负 SD+ControlNet-Tile 鍏抽敭甯?+ 鍏夋祦鍚庡鐞嗭紙涓嶈 LoRA锛夈€?
```bash
# 涓€閿細鍏堣 A锛? epoch锛夛紝鍐嶈 C锛?5 epoch锛屽喕缁?RefineNet锛?python -m part3_exploration.run_train_5h

# 璁粌鍚?Wild 鎺ㄧ悊 + 姹囨€?metric锛堥渶宸叉湁 Part2 BasicVSR / Real-ESRGAN锛?python -m part3_exploration.run_train_5h --skip_a --skip_c --eval_wild
```

榛樿妫€鏌ョ偣璺緞锛歚weights/part3_a/refinenet_x4.pth`銆乣weights/part3_c/fusion_head.pth`锛堝潎鐢变笂杩拌缁冭剼鏈湪鏁版嵁涓婂緱鍒帮級銆? 
鏁版嵁涓庨€€鍖栭€昏緫瑙?[`data/degrade_dataset.py`](data/degrade_dataset.py)銆?
**鑻?`dataset/` 鍙湁 `inputs_mp4` + `wild_real.mp4`锛堟棤 REDS/Vimeo PNG锛?*锛氶粯璁?`--hr_source auto` 浼氱敤 `wild_real.mp4` 鍏ㄩ儴甯?+ `part2_sota/outputs/realesrgan/*.mp4` 浣滀负浼?HR 璁粌锛堥渶鍏堣窇杩?Part2 Real-ESRGAN锛夈€?
## 鏃у叆鍙ｏ紙浠嶅彲鐢級

- `python -m part3_exploration.run_part3_flow_stabilize`锛堟棤鍙傛暟鏃堕粯璁?Wild锛岃緭鍑哄埌鏂扮洰褰曪級
- `python -m part3_exploration.run_part3_hybrid`
- `python -m part3_exploration.run_part3_uncertainty_adaptive`
- `python -m part3_exploration.run_part3_diffusion_tile`
