from __future__ import annotations

from runtime_compat import ensure_runtime_compatibility

ensure_runtime_compatibility()

import torch
from torch import nn
from torch.nn import functional as F

from models.coarse_encoder_backbone import (
    FixedBEVNormalizer,
    GeMPooling,
    ResidualBlock,
    initialize_module_deterministically,
)


class MultiScaleAttentionDescriptorHead(nn.Module):
    def __init__(self, descriptor_dim: int = 256) -> None:
        super().__init__()
        self.scale_attention = nn.Sequential(
            nn.Linear(64 + 128 + 192, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 3),
        )
        self.multiscale_projection = nn.Sequential(
            nn.Linear(64 + 128 + 192, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, descriptor_dim),
        )
        self.descriptor_fusion = nn.Sequential(
            nn.Linear(descriptor_dim * 2, descriptor_dim),
            nn.ReLU(inplace=True),
            nn.Linear(descriptor_dim, descriptor_dim),
        )

    def forward(
        self,
        pooled_stage2: torch.Tensor,
        pooled_stage3: torch.Tensor,
        pooled_stage4: torch.Tensor,
        legacy_descriptor: torch.Tensor,
    ) -> torch.Tensor:
        pooled_concat = torch.cat((pooled_stage2, pooled_stage3, pooled_stage4), dim=1)
        attention_logits = self.scale_attention(pooled_concat)
        attention = torch.softmax(attention_logits, dim=1)
        weighted_concat = torch.cat(
            (
                attention[:, 0:1] * pooled_stage2,
                attention[:, 1:2] * pooled_stage3,
                attention[:, 2:3] * pooled_stage4,
            ),
            dim=1,
        )
        multiscale_descriptor = self.multiscale_projection(weighted_concat)
        fused_descriptor = self.descriptor_fusion(torch.cat((legacy_descriptor, multiscale_descriptor), dim=1))
        return F.normalize(fused_descriptor, dim=1)


class MultiScaleCoarseBEVEncoderBackbone(nn.Module):
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
        self.multiscale_head = MultiScaleAttentionDescriptorHead(descriptor_dim=descriptor_dim)
        initialize_module_deterministically(self, seed=init_seed)

    def forward_features(self, bev_tensor: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.normalizer(bev_tensor)
        x = self.stem(x)
        x = self.stage1(x)
        stage2 = self.stage2(x)
        stage3 = self.stage3(stage2)
        stage4 = self.stage4(stage3)

        legacy_pooled = self.pool(stage4)
        legacy_descriptor = self.projection(legacy_pooled)
        pooled_stage2 = F.adaptive_avg_pool2d(stage2, output_size=1).flatten(1)
        pooled_stage3 = F.adaptive_avg_pool2d(stage3, output_size=1).flatten(1)
        pooled_stage4 = legacy_pooled.flatten(1)
        return {
            "stage2": stage2,
            "stage3": stage3,
            "stage4": stage4,
            "descriptor": self.multiscale_head(
                pooled_stage2=pooled_stage2,
                pooled_stage3=pooled_stage3,
                pooled_stage4=pooled_stage4,
                legacy_descriptor=legacy_descriptor,
            ),
        }

    def forward(self, bev_tensor: torch.Tensor) -> torch.Tensor:
        return self.forward_features(bev_tensor)["descriptor"]
