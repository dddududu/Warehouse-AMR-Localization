# -*- coding: utf-8 -*-
from experiments.target_motion_20260812.analyze_localization_risk_monitor import (
    SELECTED_FEATURES,
    extract_deployment_features,
)


def test_risk_features_do_not_read_ground_truth_error() -> None:
    frame = {
        "position_error_m": 999.0,
        "yaw_error_deg": 999.0,
        "semantic_occlusion_ratio": 0.2,
        "semantic_filter_semi_dynamic_point_ratio": 0.1,
        "candidate_results": [
            {"final_score": 1.0, "icp_rmse": 0.2, "deep_match_probability": 0.8},
            {"final_score": 0.5, "icp_rmse": 0.4, "deep_match_probability": 0.3},
        ],
        "selected_candidate_index": 0,
    }

    features = extract_deployment_features(frame)

    assert set(features) == set(SELECTED_FEATURES)
    assert 999.0 not in features.values()
    assert features["final_score_margin"] == 0.5
    assert features["icp_rmse_margin"] == -0.2
