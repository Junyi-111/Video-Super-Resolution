"""Backward-compatible entry. Prefer ``python -m part3_exploration.direction_b_sd_controlnet.run_flow_stabilize``."""

import sys

from part3_exploration.direction_b_sd_controlnet.run_flow_stabilize import main


if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv += [
            "--lr",
            "dataset/wild_real_lr.mp4",
            "--realesrgan",
            "part2_sota/outputs/wild_real_lr/realesrgan_x4.mp4",
            "--out",
            "part3_exploration/outputs/direction_b_sd_controlnet/wild_real_lr_flow_stabilized_x4.mp4",
        ]
    main()
