from experiments.semantic_class_prior_20260808.analyze_semidynamic_filter_trigger import _hybrid_metrics


def test_hybrid_uses_structural_only_above_threshold() -> None:
    rows = [
        {"dynamic_error_m": 0.2, "structural_error_m": 0.1, "semi_dynamic_ratio": 0.02},
        {"dynamic_error_m": 0.3, "structural_error_m": 0.4, "semi_dynamic_ratio": 0.0},
    ]
    metrics = _hybrid_metrics(rows, threshold=0.01)
    assert metrics["mean_position_error_m"] == 0.2
    assert metrics["structural_selected_ratio"] == 0.5
