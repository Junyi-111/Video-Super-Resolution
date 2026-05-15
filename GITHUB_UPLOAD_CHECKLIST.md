# GitHub Upload Checklist

Repository URL:
https://github.com/Junyi-111/Video-Super-Resolution.git

Upload / track these:

- README.md
- requirements.txt
- .gitignore
- part1_baseline/ source code and README
- part2_sota/ source code and README
- part3_exploration/ source code and README files
- scripts/ helper scripts
- docs/requirements/ project PDF
- docs/report/ report source, bibliography, and final PDF
- output/tables/ metric tables
- output/figures/ selected report/comparison figures
- weights/README.md and weights/part3_d/README.md only

Do not upload / do not track these directly:

- dataset/ raw datasets and Vimeo-90K files
- videos.zip
- *.mp4 videos
- weights/**/*.pth, *.pt, *.ckpt, *.safetensors
- part1_baseline/outputs/
- part2_sota/outputs/
- part3_exploration/outputs/
- output/vimeo90k_small/
- Block-Sparse-Attention/ external dependency checkout/build tree
- __pycache__/ and build artifacts

Canvas submission:

- Upload videos.zip separately to Canvas.
- Upload the final PDF report separately to Canvas if required.

Recommended git commands from this folder:

```powershell
cd D:\Codefield\pythonfield\CV_project\CV_submission\CV
git status --ignored
git add .
git status
git commit -m "Prepare video super-resolution project submission"
git branch -M main
git remote add origin https://github.com/Junyi-111/Video-Super-Resolution.git
git push -u origin main
```

If `git remote add origin` says the remote already exists, use:

```powershell
git remote set-url origin https://github.com/Junyi-111/Video-Super-Resolution.git
git push -u origin main
```
