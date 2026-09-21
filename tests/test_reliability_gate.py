import torch

from models.reliability_gate import (
    RELIABILITY_GATE_SOURCES,
    ReliabilityGate,
    build_reliability_gate_features,
)


def test_reliability_gate_features_and_output_shape() -> None:
    features = build_reliability_gate_features(
        deep_match_probability=0.8,
        deep_pose_confidence=0.7,
        bev_match_score=0.2,
        deep_bev_match_score=0.3,
        bev_inlier_ratio=0.8,
        deep_inlier_ratio=0.9,
        tracker_inlier_ratio=None,
        bev_rmse=0.4,
        deep_rmse=0.3,
        tracker_rmse=None,
        bev_temporal_score=0.1,
        deep_temporal_score=0.2,
        tracker_temporal_score=None,
        deep_available=True,
        tracker_available=False,
    )

    assert features.shape == (19,)
    assert features[-6:-3].tolist() == [1.0, 1.0, 0.0]

    model = ReliabilityGate(feature_dim=features.size, hidden_dim=8)
    logits = model(torch.from_numpy(features[None]))
    assert logits.shape == (1, len(RELIABILITY_GATE_SOURCES))
