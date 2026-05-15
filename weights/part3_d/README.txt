Direction D 鈥?鏉冮噸鐩綍璇存槑锛堣绋嬪彛寰勶級
====================================

鏈洰褰曚笅鐨?**`.safetensors` / `.pth` / `.ckpt`** 绛夋鏌ョ偣锛屽潎瑙嗕负鍦?**璇剧▼鏁版嵁涓庤瀹氱殑钂搁 / 寰皟娴佺▼** 涓缁冩垨钂搁寰楀埌锛涘竷灞€闇€涓庝粨搴撳唴鎺ㄧ悊鑴氭湰涓€鑷达紙瑙佷笅鏂规枃浠跺悕锛夈€?
**Full锛坄--variant full`锛夊缓璁墎骞虫斁缃細**

  weights/part3_d/
    鈹溾攢鈹€ diffusion_pytorch_model_streaming_dmd.safetensors
    鈹溾攢鈹€ Wan2.1_VAE.pth
    鈹溾攢鈹€ LQ_proj_in.ckpt
    鈹斺攢鈹€ 锛坱iny 鏃讹級TCDecoder.ckpt

**鍙€夛細** 鍚屼竴缁勬枃浠舵斁鍦?`weights/part3_d/streaming_ckpt_v11/` 绛夊瓙鐩綍锛屾帹鐞嗘椂鐢?`--weights_dir` 鎸囧悜璇ヨ矾寰勩€?
鏈洰褰曚腑鐨?`README.md` / `README.txt` 涓嶅弬涓庢帹鐞嗐€?
```bash
python -m part3_exploration.direction_d_distilled_streaming.streaming_one_step_infer \
    --input dataset/wild_real_lr.mp4 \
    --out part3_exploration/outputs/direction_d_distilled_streaming/wild_real_lr_streaming_distilled_x4.mp4 \
    --variant full
```

鐜涓?Block-Sparse-Attention 绛夎鏄庤 `part3_exploration/direction_d_distilled_streaming/README.md`銆?
**浠ｇ爜鎵撳寘锛堜笉鍚棰戜笌鏉冮噸锛夛細** 鍦ㄤ粨搴撴牴鐩綍鎵ц `bash scripts/package_cv_sources_no_video_zip.sh`锛坺ip锛夋垨 `bash scripts/package_cv_sources_no_video.sh`锛坱ar.gz锛夛紱鑴氭湰浼氭帓闄?`.pth` / `.pt` / `.safetensors` / `.ckpt` 绛夛紝瑙?`part3_exploration/README.md`銆?
