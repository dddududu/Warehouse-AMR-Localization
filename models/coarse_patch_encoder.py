from __future__ import annotations

from torch import nn

from models.coarse_query_encoder import _build_backbone


class CoarsePatchEncoder(nn.Module):
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

    def forward_features(self, bev_tensor):
        return self.backbone.forward_features(bev_tensor)
