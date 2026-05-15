"""Lightweight single-step HR refinement (Part 3-A student)."""

from __future__ import annotations

import torch
import torch.nn as nn


class RefineNet(nn.Module):
    """Residual CNN: output = bicubic_HR + delta."""

    def __init__(self, channels: int = 3, width: int = 64) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, width, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, channels, 3, 1, 1),
        )

    def forward(self, bicubic_hr: torch.Tensor) -> torch.Tensor:
        return bicubic_hr + self.body(bicubic_hr)
