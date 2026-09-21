from __future__ import annotations

import numpy as np
import torch
from torch import nn


RELIABILITY_GATE_SOURCES = ("bev_init", "deep_init", "tracker_init")


class ReliabilityGate(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, len(RELIABILITY_GATE_SOURCES)),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def build_reliability_gate_features(
    *,
    deep_match_probability: float,
    deep_pose_confidence: float,
    bev_match_score: float,
    deep_bev_match_score: float | None,
    bev_inlier_ratio: float,
    deep_inlier_ratio: float | None,
    tracker_inlier_ratio: float | None,
    bev_rmse: float,
    deep_rmse: float | None,
    tracker_rmse: float | None,
    bev_temporal_score: float,
    deep_temporal_score: float | None,
    tracker_temporal_score: float | None,
    deep_available: bool,
    tracker_available: bool,
    person_ratio: float = 0.0,
    dynamic_ratio: float = 0.0,
    movable_ratio: float = 0.0,
) -> np.ndarray:
    def normalized_rmse(value: float | None) -> float:
        if value is None or not np.isfinite(value):
            return 1.0
        return float(np.clip(float(value), 0.0, 3.0) / 3.0)

    def finite_or_zero(value: float | None) -> float:
        if value is None or not np.isfinite(value):
            return 0.0
        return float(value)

    return np.asarray(
        [
            finite_or_zero(deep_match_probability),
            finite_or_zero(deep_pose_confidence),
            finite_or_zero(bev_match_score),
            finite_or_zero(deep_bev_match_score if deep_bev_match_score is not None else bev_match_score),
            finite_or_zero(bev_inlier_ratio),
            finite_or_zero(deep_inlier_ratio),
            finite_or_zero(tracker_inlier_ratio),
            normalized_rmse(bev_rmse),
            normalized_rmse(deep_rmse),
            normalized_rmse(tracker_rmse),
            finite_or_zero(bev_temporal_score),
            finite_or_zero(deep_temporal_score),
            finite_or_zero(tracker_temporal_score),
            1.0,
            float(bool(deep_available)),
            float(bool(tracker_available)),
            finite_or_zero(person_ratio),
            finite_or_zero(dynamic_ratio),
            finite_or_zero(movable_ratio),
        ],
        dtype=np.float32,
    )
