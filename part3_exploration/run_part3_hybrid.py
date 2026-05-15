"""Backward-compatible entry. Prefer ``python -m part3_exploration.direction_b_sd_controlnet.run_temporal_generative_blend``."""

import sys

from part3_exploration.direction_b_sd_controlnet.run_temporal_generative_blend import main


if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv += [
            "--temporal",
            "part1_baseline/outputs/wild_real_lr/wild_real_lr_part1_temporal_w5_unsharp_x4.mp4",
            "--realesrgan",
            "part2_sota/outputs/wild_real_lr/realesrgan_x4.mp4",
            "--out",
            "part3_exploration/outputs/direction_b_sd_controlnet/wild_real_lr_temporal_gen_blend_x4.mp4",
        ]
    main()
