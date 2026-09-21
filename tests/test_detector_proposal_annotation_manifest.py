from experiments.target_motion_20260804.build_detector_proposal_annotation_manifest import _choose_diverse


def test_choose_diverse_prefers_confidence_and_time_separation() -> None:
    rows = [
        {"frame_idx": 0, "detector_score": 0.99},
        {"frame_idx": 10, "detector_score": 0.98},
        {"frame_idx": 50, "detector_score": 0.95},
        {"frame_idx": 100, "detector_score": 0.90},
    ]
    selected = _choose_diverse(rows, maximum=3, minimum_frame_gap=30)
    assert [row["frame_idx"] for row in selected] == [0, 50, 100]
