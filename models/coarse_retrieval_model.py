from __future__ import annotations

import torch
from torch import nn

from models.coarse_patch_encoder import CoarsePatchEncoder
from models.coarse_query_encoder import CoarseQueryEncoder


class CoarseRetrievalModel(nn.Module):
    def __init__(
        self,
        descriptor_dim: int = 256,
        init_seed: int = 0,
        backbone_variant: str = "legacy",
        share_query_patch_encoder: bool = False,
        num_patch_classes: int | None = None,
    ) -> None:
        super().__init__()
        self.share_query_patch_encoder = bool(share_query_patch_encoder)
        self.backbone_variant = str(backbone_variant)
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

    def encode_query(self, query_bev: torch.Tensor) -> torch.Tensor:
        return self.query_encoder(query_bev)

    def encode_patch(self, patch_bev: torch.Tensor) -> torch.Tensor:
        return self.patch_encoder(patch_bev)

    def classify_query_descriptor(self, query_descriptor: torch.Tensor) -> torch.Tensor | None:
        if self.query_classifier is None:
            return None
        return self.query_classifier(query_descriptor)

    def forward(
        self,
        query_bev: torch.Tensor,
        positive_patch_bev: torch.Tensor,
        negative_patch_bevs: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        query_descriptor = self.encode_query(query_bev)
        positive_descriptor = self.encode_patch(positive_patch_bev)
        outputs = {
            "query_descriptor": query_descriptor,
            "positive_descriptor": positive_descriptor,
        }
        if self.query_classifier is not None:
            outputs["query_logits"] = self.classify_query_descriptor(query_descriptor)
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
                negative_descriptor = self.encode_patch(flat_negatives).reshape(bsz, num_negatives, -1)
            outputs["negative_descriptor"] = negative_descriptor
        return outputs
