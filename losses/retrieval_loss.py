from __future__ import annotations

from runtime_compat import ensure_runtime_compatibility

ensure_runtime_compatibility()

import torch
from torch import nn
from torch.nn import functional as F


class RetrievalInfoNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = float(temperature)

    def forward(
        self,
        query_descriptor: torch.Tensor,
        positive_descriptor: torch.Tensor,
        negative_descriptor: torch.Tensor,
    ) -> torch.Tensor:
        positive_logits = torch.sum(query_descriptor * positive_descriptor, dim=1, keepdim=True)
        negative_logits = torch.einsum("bd,bnd->bn", query_descriptor, negative_descriptor)
        logits = torch.cat((positive_logits, negative_logits), dim=1) / self.temperature
        labels = torch.zeros(query_descriptor.shape[0], dtype=torch.long, device=query_descriptor.device)
        return F.cross_entropy(logits, labels)
