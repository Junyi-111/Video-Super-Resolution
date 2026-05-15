# Model Weights

Large pretrained weights are intentionally not tracked by git. Place them here when reproducing the experiments:

```text
weights/realesrgan/RealESRGAN_x4plus.pth
weights/basicvsr/basicvsr_vimeo90k_bi.pth
weights/part3_d/...
```

Recommended practice for GitHub:

- Do not commit `.pth`, `.pt`, `.ckpt`, or `.safetensors` files.
- Provide download links in the README or GitHub Releases if needed.
- Keep this README and `weights/part3_d/README.md` tracked as documentation.
