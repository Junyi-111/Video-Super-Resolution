"""Learned pixel-wise fusion between conservative and generative HR branches."""

from __future__ import annotations

import torch
import torch.nn as nn


class FusionHead(nn.Module):
    """Predict alpha in [0,1]; output = (1-alpha)*basic + alpha*gen."""

    def __init__(self, in_channels: int = 9, width: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, width, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, 1, 3, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, basic: torch.Tensor, gen: torch.Tensor) -> torch.Tensor:
        diff = torch.abs(basic - gen)
        x = torch.cat([basic, gen, diff], dim=1)
        return self.net(x)

    def fuse(self, basic: torch.Tensor, gen: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        alpha = self.forward(basic, gen)
        out = (1.0 - alpha) * basic + alpha * gen
        return out, alpha
