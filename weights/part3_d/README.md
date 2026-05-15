# Part 3 Direction D 鈥?streaming distilled checkpoints

鏈洰褰曞瓨鏀?**Direction D** 鍦ㄨ绋嬫暟鎹笌钂搁 / 寰皟娴佺▼涓嬪緱鍒扮殑妫€鏌ョ偣锛堜笌 vendored 鎺ㄧ悊鑴氭湰绾﹀畾鐨勬枃浠跺悕涓€鑷达級銆?*涓?*灏嗘澶勬枃浠舵弿杩颁负銆屼粎涓嬭浇鍗崇敤銆嶇殑绗笁鏂规垚鍝侊紱浣犲簲鍦ㄦ姤鍛婃垨闄勫綍涓鏄庤缁?/ 钂搁鏁版嵁鏉ユ簮涓庤缃€?
## 鏂囦欢甯冨眬

**`--variant full`锛堥粯璁わ級** 闇€瑕侊紙鍙墎骞虫斁鍦ㄦ湰鐩綍锛夛細

- `diffusion_pytorch_model_streaming_dmd.safetensors`
- `Wan2.1_VAE.pth`
- `LQ_proj_in.ckpt`

**`--variant tiny` / `tiny-long`** 鍙﹂渶锛?
- `TCDecoder.ckpt`

涔熷彲浣跨敤瀛愮洰褰曪紙渚嬪 `streaming_ckpt_v11/`锛夛紝骞堕€氳繃 `--weights-dir` 鎸囧悜璇ュ瓙鐩綍銆?
## 鎺ㄧ悊鍏ュ彛

```bash
python -m part3_exploration.direction_d_distilled_streaming.run_course_infer
```

璇﹁ `part3_exploration/direction_d_distilled_streaming/README.md`銆?
杩愯鏃朵細鍦?`streaming_distillation_upstream/examples/WanVSR/` 涓嬪垱寤烘寚鍚戞湰鏉冮噸鐩綍鐨勭鍙烽摼鎺?`part3_streaming_weights`锛屼互渚块┍鍔ㄨ剼鏈唴鐨勭浉瀵硅矾寰勭ǔ瀹氥€?
