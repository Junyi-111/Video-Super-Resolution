"""Backward-compatible entry. Prefer ``python -m part3_exploration.direction_c_uncertainty.run_adaptive``."""

import sys

from part3_exploration.direction_c_uncertainty.run_adaptive import main


if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv += [
            "--lr",
            "dataset/wild_real_lr.mp4",
            "--realesrgan",
            "part2_sota/outputs/wild_real_lr/realesrgan_x4.mp4",
            "--basicvsr",
            "part2_sota/outputs/basicvsr/wild_real_lr.mp4",
            "--out",
            "part3_exploration/outputs/direction_c_uncertainty/wild_real_lr_uncertainty_adaptive_x4.mp4",
            "--mask_out",
            "part3_exploration/outputs/direction_c_uncertainty/wild_real_lr_uncertainty_alpha.mp4",
        ]
    main()
