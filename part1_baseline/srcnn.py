"""SRCNN [Dong et al. TPAMI 2015] — 3-layer CNN for spatial SR (RGB)."""

from __future__ import annotations

import torch
import torch.nn as nn


class SRCNN(nn.Module):
    """Patch-wise mapping LR -> HR in RGB. Default: 9-1-5 kernels, f1=64, f2=32."""

    def __init__(self, num_channels: int = 3, f1: int = 64, f2: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(num_channels, f1, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(f1, f2, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(f2, num_channels, kernel_size=5, padding=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def psnr_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mse = torch.mean((pred - target) ** 2)
    return -10.0 * torch.log10(mse + eps)
