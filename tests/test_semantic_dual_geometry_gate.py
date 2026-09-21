from localization.deep_fine_localizer import DeepFineLocalizer


def _candidate(patch_id: int, inlier: float, rmse: float, valid: bool = True) -> dict[str, object]:
    return {
        "patch_id": patch_id,
        "icp_inlier_ratio": inlier,
        "icp_rmse": rmse,
        "icp_valid": valid,
    }


def test_dual_geometry_gate_accepts_only_nonworse_filtered_geometry() -> None:
    filtered = [_candidate(1, 0.8, 0.2), _candidate(2, 0.6, 0.4)]
    raw = [_candidate(1, 0.7, 0.3), _candidate(2, 0.7, 0.3)]

    merged, accepted = DeepFineLocalizer._merge_semantic_geometry_hypotheses(filtered, raw)

    assert accepted == 1
    assert merged[0] is filtered[0]
    assert merged[1] is raw[1]
