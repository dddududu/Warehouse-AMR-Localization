from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SeparableBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=stride, padding=1, groups=input_channels, bias=False),
            nn.BatchNorm2d(input_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(input_channels, output_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class PersonMaskNet(nn.Module):
    def __init__(self, base_channels: int = 24) -> None:
        super().__init__()
        channels = int(base_channels)
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )
        self.encoder1 = SeparableBlock(channels, channels * 2, stride=2)
        self.encoder2 = SeparableBlock(channels * 2, channels * 4, stride=2)
        self.encoder3 = SeparableBlock(channels * 4, channels * 6, stride=2)
        self.bottleneck = SeparableBlock(channels * 6, channels * 6)
        self.decoder2 = SeparableBlock(channels * 6 + channels * 4, channels * 4)
        self.decoder1 = SeparableBlock(channels * 4 + channels * 2, channels * 2)
        self.decoder0 = SeparableBlock(channels * 2 + channels, channels)
        self.head = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        stem = self.stem(inputs)
        encoded1 = self.encoder1(stem)
        encoded2 = self.encoder2(encoded1)
        encoded3 = self.bottleneck(self.encoder3(encoded2))
        decoded2 = self.decoder2(torch.cat((F.interpolate(encoded3, size=encoded2.shape[-2:], mode="bilinear", align_corners=False), encoded2), dim=1))
        decoded1 = self.decoder1(torch.cat((F.interpolate(decoded2, size=encoded1.shape[-2:], mode="bilinear", align_corners=False), encoded1), dim=1))
        decoded0 = self.decoder0(torch.cat((F.interpolate(decoded1, size=stem.shape[-2:], mode="bilinear", align_corners=False), stem), dim=1))
        return F.interpolate(self.head(decoded0), size=inputs.shape[-2:], mode="bilinear", align_corners=False)
