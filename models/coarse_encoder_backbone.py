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


class FixedBEVNormalizer(nn.Module):
    def __init__(self, count_clip: float = 32.0, z_min: float = 0.13, z_max: float = 4.73) -> None:
        super().__init__()
        self.count_clip = float(count_clip)
        self.z_min = float(z_min)
        self.z_max = float(z_max)

    def forward(self, bev_tensor: torch.Tensor) -> torch.Tensor:
        count_input = torch.clamp(bev_tensor[:, 0:1], min=0.0)
        count = torch.log1p(count_input)
        count = torch.clamp(count, max=torch.log1p(torch.tensor(self.count_clip)).item())
        count = count / torch.log1p(torch.tensor(self.count_clip, device=bev_tensor.device, dtype=bev_tensor.dtype))

        height_range = max(self.z_max - self.z_min, 1e-6)
        max_height = torch.clamp((bev_tensor[:, 1:2] - self.z_min) / height_range, min=0.0, max=1.0)
        mean_height = torch.clamp((bev_tensor[:, 2:3] - self.z_min) / height_range, min=0.0, max=1.0)
        occupancy = torch.clamp(bev_tensor[:, 3:4], min=0.0, max=1.0)
        return torch.cat((count, max_height, mean_height, occupancy), dim=1)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = F.relu(out + residual, inplace=True)
        return out


class GeMPooling(nn.Module):
    def __init__(self, p: float = 3.0, eps: float = 1.0e-6) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.tensor(float(p)))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), kernel_size=(x.shape[-2], x.shape[-1]))
        return pooled.pow(1.0 / self.p)


class CoarseBEVEncoderBackbone(nn.Module):
    def __init__(self, in_channels: int = 4, descriptor_dim: int = 256, init_seed: int = 0) -> None:
        super().__init__()
        self.normalizer = FixedBEVNormalizer()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.stage1 = nn.Sequential(
            ResidualBlock(32, 32, stride=1),
            ResidualBlock(32, 32, stride=1),
        )
        self.stage2 = nn.Sequential(
            ResidualBlock(32, 64, stride=2),
            ResidualBlock(64, 64, stride=1),
        )
        self.stage3 = nn.Sequential(
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128, stride=1),
        )
        self.stage4 = nn.Sequential(
            ResidualBlock(128, 192, stride=2),
            ResidualBlock(192, 192, stride=1),
        )
        self.pool = GeMPooling()
        self.projection = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(192, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, descriptor_dim),
        )
        initialize_module_deterministically(self, seed=init_seed)

    def forward(self, bev_tensor: torch.Tensor) -> torch.Tensor:
        x = self.normalizer(bev_tensor)
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        pooled = self.pool(x)
        descriptor = self.projection(pooled)
        return F.normalize(descriptor, dim=1)
