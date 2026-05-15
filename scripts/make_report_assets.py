from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
path = OUT / "report_pipeline_flowchart.png"

W, H = 1800, 1050
img = Image.new("RGB", (W, H), "#f7f9fb")
d = ImageDraw.Draw(img)

try:
    title_font = ImageFont.truetype("arial.ttf", 54)
    header_font = ImageFont.truetype("arial.ttf", 34)
    body_font = ImageFont.truetype("arial.ttf", 27)
    small_font = ImageFont.truetype("arial.ttf", 23)
except Exception:
    title_font = header_font = body_font = small_font = ImageFont.load_default()

palette = {
    "blue": "#2f6fbd",
    "green": "#2e8b57",
    "orange": "#c47f22",
    "purple": "#7a5bc2",
    "gray": "#344054",
    "line": "#98a2b3",
    "fill": "#ffffff",
}

def box(xy, title, lines, color):
    x1, y1, x2, y2 = xy
    d.rounded_rectangle(xy, radius=22, fill=palette["fill"], outline=color, width=5)
    d.rectangle((x1, y1, x2, y1 + 72), fill=color)
    d.text((x1 + 26, y1 + 18), title, fill="white", font=header_font)
    y = y1 + 105
    for line in lines:
        d.text((x1 + 28, y), line, fill=palette["gray"], font=body_font)
        y += 43

def arrow(x1, y1, x2, y2):
    d.line((x1, y1, x2, y2), fill=palette["line"], width=7)
    # simple arrow head
    if x2 >= x1:
        pts = [(x2, y2), (x2 - 26, y2 - 15), (x2 - 26, y2 + 15)]
    else:
        pts = [(x2, y2), (x2 + 26, y2 - 15), (x2 + 26, y2 + 15)]
    d.polygon(pts, fill=palette["line"])

# title
d.text((W // 2, 55), "Video Super-Resolution Project Pipeline", fill="#1d2939", font=title_font, anchor="mm")
d.text((W // 2, 112), "Part 1 baselines, Part 2 SOTA reproduction, and Part 3 temporal/generative exploration", fill="#475467", font=body_font, anchor="mm")

# top row
box((70, 190, 390, 430), "Input Videos", ["REDS-sample", "Vimeo-LR", "Wild/self-captured", "Vimeo-90K small GT"], palette["gray"])
box((500, 190, 820, 430), "Part 1", ["Bicubic / Lanczos", "SRCNN-style CNN", "Temporal averaging", "Unsharp masking"], palette["blue"])
box((930, 190, 1250, 430), "Part 2", ["Real-ESRGAN", "BasicVSR", "Frame/video inference", "Processed videos"], palette["green"])
box((1360, 190, 1680, 430), "Part 3", ["Flow stabilization", "Uncertainty fusion", "Diffusion variants", "Distilled streaming"], palette["purple"])

arrow(390, 310, 500, 310)
arrow(820, 310, 930, 310)
arrow(1250, 310, 1360, 310)

# lower row
box((255, 590, 625, 850), "Evaluation", ["PSNR / SSIM if GT exists", "LPIPS perceptual score", "Temporal MAE / tLPIPS", "Qualitative frame panels"], palette["orange"])
box((735, 590, 1105, 850), "Comparison Outputs", ["output/tables/*.csv", "output/figures/*.png", "Vimeo-90K benchmark", "Part 3 method summary"], palette["blue"])
box((1215, 590, 1585, 850), "Submission", ["CVPR-style PDF report", "Public GitHub repository", "Clear README", "videos.zip"], palette["green"])

arrow(1040, 430, 440, 590)
arrow(625, 720, 735, 720)
arrow(1105, 720, 1215, 720)

# notes
d.rounded_rectangle((80, 920, 1720, 1000), radius=18, fill="#eef4ff", outline="#b2ccff", width=3)
d.text((110, 945), "Note: PSNR/SSIM are only valid for clips with ground-truth HR frames. For wild and LR-only clips, the report uses visual analysis and temporal/perceptual proxies.", fill="#1d2939", font=small_font)

img.save(path, quality=95)
print(path)
