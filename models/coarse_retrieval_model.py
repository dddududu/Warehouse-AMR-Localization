from __future__ import annotations

import torch
from torch import nn

from models.coarse_patch_encoder import CoarsePatchEncoder
from models.coarse_query_encoder import CoarseQueryEncoder
from models.local_spatial_matcher import LocalSpatialMatcher


class CoarseRetrievalModel(nn.Module):
    def __init__(
        self,
        descriptor_dim: int = 256,
        init_seed: int = 0,
        backbone_variant: str = "legacy",
        share_query_patch_encoder: bool = False,
        num_patch_classes: int | None = None,
        use_local_matcher: bool = False,
        local_matcher_feature_level: str = "stage4",
        local_matcher_hidden_channels: int = 48,
        local_matcher_max_shift_cells: int = 2,
    ) -> None:
        super().__init__()
        self.share_query_patch_encoder = bool(share_query_patch_encoder)
        self.backbone_variant = str(backbone_variant)
        self.local_matcher_feature_level = str(local_matcher_feature_level)
        self.query_encoder = CoarseQueryEncoder(
            descriptor_dim=descriptor_dim,
            init_seed=init_seed,
            backbone_variant=self.backbone_variant,
        )
        self.patch_encoder = (
            self.query_encoder
            if self.share_query_patch_encoder
            else CoarsePatchEncoder(
                descriptor_dim=descriptor_dim,
                init_seed=init_seed + 1,
                backbone_variant=self.backbone_variant,
            )
        )
        self.query_classifier = nn.Linear(descriptor_dim, int(num_patch_classes)) if num_patch_classes is not None else None
        local_matcher_channels = {
            "stage2": 64,
            "stage3": 128,
            "stage4": 192,
        }.get(self.local_matcher_feature_level, 192)
        self.local_matcher = (
            LocalSpatialMatcher(
                in_channels=local_matcher_channels,
                hidden_channels=local_matcher_hidden_channels,
                max_shift_cells=local_matcher_max_shift_cells,
            )
            if use_local_matcher
            else None
        )

    def encode_query(self, query_bev: torch.Tensor) -> torch.Tensor:
        return self.query_encoder(query_bev)

    def encode_patch(self, patch_bev: torch.Tensor) -> torch.Tensor:
        return self.patch_encoder(patch_bev)

    def classify_query_descriptor(self, query_descriptor: torch.Tensor) -> torch.Tensor | None:
        if self.query_classifier is None:
            return None
        return self.query_classifier(query_descriptor)

    def _select_local_feature(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.local_matcher_feature_level not in features:
            raise KeyError(f"Missing local feature level: {self.local_matcher_feature_level}")
        return features[self.local_matcher_feature_level]

    def forward(
        self,
        query_bev: torch.Tensor,
        positive_patch_bev: torch.Tensor,
        negative_patch_bevs: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        query_features = self.query_encoder.forward_features(query_bev)
        positive_features = self.patch_encoder.forward_features(positive_patch_bev)
        query_descriptor = query_features["descriptor"]
        positive_descriptor = positive_features["descriptor"]
        outputs = {
            "query_descriptor": query_descriptor,
            "positive_descriptor": positive_descriptor,
        }
        if self.query_classifier is not None:
            outputs["query_logits"] = self.classify_query_descriptor(query_descriptor)
        if self.local_matcher is not None:
            outputs["query_local_map"] = self.local_matcher.project(self._select_local_feature(query_features))
            outputs["positive_local_map"] = self.local_matcher.project(self._select_local_feature(positive_features))
        if negative_patch_bevs is not None:
            bsz, num_negatives, channels, height, width = negative_patch_bevs.shape
            if num_negatives == 0:
                negative_descriptor = torch.empty(
                    (bsz, 0, query_descriptor.shape[1]),
                    device=query_descriptor.device,
                    dtype=query_descriptor.dtype,
                )
            else:
                flat_negatives = negative_patch_bevs.reshape(bsz * num_negatives, channels, height, width)
                negative_features = self.patch_encoder.forward_features(flat_negatives)
                negative_descriptor = negative_features["descriptor"].reshape(bsz, num_negatives, -1)
                if self.local_matcher is not None:
                    negative_local_maps = self.local_matcher.project(
                        self._select_local_feature(negative_features)
                    )
                    outputs["negative_local_maps"] = negative_local_maps.reshape(
                        bsz,
                        num_negatives,
                        *negative_local_maps.shape[1:],
                    )
            outputs["negative_descriptor"] = negative_descriptor
        return outputs
