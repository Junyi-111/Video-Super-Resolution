"""Create a report panel for diffusion-tile keyframe experiments."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw


def label_image(image: Image.Image, label: str) -> Image.Image:
    out = image.copy()
    draw = ImageDraw.Draw(out)
    draw.rectangle((0, 0, 260, 34), fill=(0, 0, 0))
    draw.text((10, 9), label, fill=(255, 255, 255))
    return out


def main() -> None:
    frames = [80, 120, 160]
    columns = [
        ("Real-ESRGAN", Path("part3_exploration/outputs/diffusion_tile_conservative")),
        ("Diffusion conservative", Path("part3_exploration/outputs/diffusion_tile_conservative")),
        ("Diffusion over-strong", Path("part3_exploration/outputs/diffusion_tile_report")),
    ]
    gap = 12
    rows = []
    for frame in frames:
        images = []
        for label, directory in columns:
            if label == "Real-ESRGAN":
                path = directory / f"frame_{frame:04d}_realesrgan_input.png"
            else:
                path = directory / f"frame_{frame:04d}_diffusion_tile.png"
            images.append(label_image(Image.open(path).convert("RGB"), label))

        row_w = sum(img.width for img in images) + gap * (len(images) - 1)
        row_h = max(img.height for img in images)
        row = Image.new("RGB", (row_w, row_h), (245, 245, 245))
        x = 0
        for img in images:
            row.paste(img, (x, 0))
            x += img.width + gap
        rows.append(row)

    out_w = max(row.width for row in rows)
    out_h = sum(row.height for row in rows) + gap * (len(rows) - 1)
    canvas = Image.new("RGB", (out_w, out_h), (245, 245, 245))
    y = 0
    for row in rows:
        canvas.paste(row, (0, y))
        y += row.height + gap

    out = Path("output/figures/wild_real_lr_part3/diffusion_tile_panel.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
