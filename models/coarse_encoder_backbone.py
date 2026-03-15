from __future__ import annotations

from runtime_compat import ensure_runtime_compatibility

ensure_runtime_compatibility()

import torch
from torch import nn
from torch.nn import functional as F


def initialize_module_deterministically(module: nn.Module, seed: int) -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        for layer in module.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.ones_(layer.weight)
                nn.init.zeros_(layer.bias)


class CoarseBEVEncoderBackbone(nn.Module):
    def __init__(self, in_channels: int = 4, descriptor_dim: int = 256, init_seed: int = 0) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.projection = nn.Linear(128, descriptor_dim)
        initialize_module_deterministically(self, seed=init_seed)

    def forward(self, bev_tensor: torch.Tensor) -> torch.Tensor:
        features = self.features(bev_tensor)
        pooled = self.pool(features).flatten(1)
        descriptor = self.projection(pooled)
        return F.normalize(descriptor, dim=1)
