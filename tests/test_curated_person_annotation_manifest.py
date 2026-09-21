from experiments.target_motion_20260804.build_curated_person_annotation_manifest import _choose_diverse, _is_high_quality


def _row(frame_idx: int, *, height: int = 40, area: int = 800, depth_ratio: float = 0.9, depth: str = "2.0") -> dict[str, str]:
    return {"frame_idx": str(frame_idx), "height": str(height), "pixel_area": str(area), "valid_depth_ratio": str(depth_ratio), "median_depth_m": depth}


def test_high_quality_rule_requires_valid_near_depth() -> None:
    assert _is_high_quality(_row(0))
    assert not _is_high_quality(_row(0, height=31))
    assert not _is_high_quality(_row(0, area=399))
    assert not _is_high_quality(_row(0, depth_ratio=0.69))
    assert not _is_high_quality(_row(0, depth=""))


def test_choose_diverse_enforces_temporal_gap() -> None:
    rows = [_row(0, height=90), _row(10, height=80), _row(40, height=70), _row(80, height=60)]
    assert [int(row["frame_idx"]) for row in _choose_diverse(rows, maximum=3, minimum_frame_gap=30)] == [0, 40, 80]
