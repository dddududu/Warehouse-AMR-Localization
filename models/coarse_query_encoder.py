from __future__ import annotations

from torch import nn

from models.coarse_encoder_backbone import CoarseBEVEncoderBackbone
from models.multiscale_coarse_encoder_backbone import MultiScaleCoarseBEVEncoderBackbone


def _build_backbone(
    backbone_variant: str,
    in_channels: int,
    descriptor_dim: int,
    init_seed: int,
) -> nn.Module:
    variant = str(backbone_variant).lower()
    if variant == "legacy":
        return CoarseBEVEncoderBackbone(
            in_channels=in_channels,
            descriptor_dim=descriptor_dim,
            init_seed=init_seed,
        )
    if variant in {"multiscale", "multiscale_attention"}:
        return MultiScaleCoarseBEVEncoderBackbone(
            in_channels=in_channels,
            descriptor_dim=descriptor_dim,
            init_seed=init_seed,
        )
    raise ValueError(f"Unsupported backbone_variant: {backbone_variant}")


class CoarseQueryEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        descriptor_dim: int = 256,
        init_seed: int = 0,
        backbone_variant: str = "legacy",
    ) -> None:
        super().__init__()
        self.backbone = _build_backbone(
            backbone_variant=backbone_variant,
            in_channels=in_channels,
            descriptor_dim=descriptor_dim,
            init_seed=init_seed,
        )

    def forward(self, bev_tensor):
        return self.backbone(bev_tensor)
