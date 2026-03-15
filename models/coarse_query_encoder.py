from __future__ import annotations

from torch import nn

from models.coarse_encoder_backbone import CoarseBEVEncoderBackbone


class CoarseQueryEncoder(nn.Module):
    def __init__(self, in_channels: int = 4, descriptor_dim: int = 256, init_seed: int = 0) -> None:
        super().__init__()
        self.backbone = CoarseBEVEncoderBackbone(
            in_channels=in_channels,
            descriptor_dim=descriptor_dim,
            init_seed=init_seed,
        )

    def forward(self, bev_tensor):
        return self.backbone(bev_tensor)

