from __future__ import annotations

import json

from analysis.analyze_map import analyze_map
from analysis.analyze_sequence import analyze_sequence
from analysis.recommend_params import recommend_parameters


def test_analysis_scripts_emit_reports(synthetic_dataset, tmp_path) -> None:
    sequence_report_path = tmp_path / "sequence_report.json"
    map_report_path = tmp_path / "map_report.json"

    sequence_report = analyze_sequence(
        sequence_root=synthetic_dataset["sequence_root"],
        calibration_path=synthetic_dataset["calibration_path"],
        config={"imu_time_window_sec": 0.1},
        output_json=sequence_report_path,
    )
    map_report = analyze_map(
        synthetic_dataset["map_path"],
        cache_dir=tmp_path / "cache",
        output_json=map_report_path,
    )
    params = recommend_parameters(sequence_report, map_report)

    assert sequence_report["num_frames"] == 3
    assert map_report["point_count"] == 4
    assert "bev_range_candidates_xy_m" in params
    assert json.loads(sequence_report_path.read_text(encoding="utf-8"))["num_frames"] == 3
    assert json.loads(map_report_path.read_text(encoding="utf-8"))["has_normals"] is True
