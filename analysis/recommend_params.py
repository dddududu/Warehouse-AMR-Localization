from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from analysis.analyze_map import analyze_map
from analysis.analyze_sequence import analyze_sequence


def _round_up(value: float, base: float) -> float:
    if value <= 0.0:
        return base
    return math.ceil(value / base) * base


def recommend_parameters(sequence_report: dict, map_report: dict | None = None) -> dict:
    lidar_min = sequence_report.get("lidar_xyz_min") or [-10.0, -10.0, -2.0]
    lidar_max = sequence_report.get("lidar_xyz_max") or [10.0, 10.0, 2.0]
    xy_half_extent = max(abs(lidar_min[0]), abs(lidar_max[0]), abs(lidar_min[1]), abs(lidar_max[1]))
    base_half_range = _round_up(float(xy_half_extent), 5.0)

    median_translation = (sequence_report.get("relative_translation_m") or {}).get("median") or 0.1
    accumulation_candidates = sorted(
        {
            max(1, int(round(target_distance / max(median_translation, 1e-3))))
            for target_distance in (0.5, 1.0, 2.0)
        }
    )

    map_extent = None
    if map_report is not None:
        bbox = map_report.get("bbox_extent")
        if bbox is not None:
            map_extent = max(float(bbox[0]), float(bbox[1]))

    resolution_candidates = [0.05, 0.1, 0.2]
    if map_extent is not None and map_extent > 100.0:
        resolution_candidates = [0.1, 0.2, 0.3]

    report = {
        "bev_range_candidates_xy_m": [
            [-base_half_range, base_half_range, -base_half_range, base_half_range],
            [-1.5 * base_half_range, 1.5 * base_half_range, -1.5 * base_half_range, 1.5 * base_half_range],
        ],
        "patch_size_candidates_m": [base_half_range, _round_up(base_half_range * 1.5, 5.0)],
        "bev_resolution_candidates_m_per_cell": resolution_candidates,
        "accumulation_num_frames_candidates": accumulation_candidates,
        "rationale": {
            "bev_range": "Derived from sampled single-frame LiDAR XY span, rounded up to stable multiples of 5 m.",
            "patch_size": "Anchored to the same local coverage so patch size and BEV crop stay comparable.",
            "bev_resolution": "Coarser candidates are preferred when the global map spans a larger area.",
            "accumulation_num_frames": "Chosen to cover roughly 0.5 m, 1.0 m, and 2.0 m of median inter-frame motion.",
        },
    }
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Recommend coarse localization design parameters.")
    parser.add_argument("--sequence-root", default=None)
    parser.add_argument("--calibration-path", default=None)
    parser.add_argument("--config", default="configs/dataset_default.yaml")
    parser.add_argument("--map-path", default=None)
    parser.add_argument("--sequence-report", default=None)
    parser.add_argument("--map-report", default=None)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    if args.sequence_report is not None:
        sequence_report = json.loads(Path(args.sequence_report).read_text(encoding="utf-8"))
    else:
        sequence_report = analyze_sequence(
            sequence_root=args.sequence_root,
            calibration_path=args.calibration_path,
            config=args.config,
        )

    if args.map_report is not None:
        map_report = json.loads(Path(args.map_report).read_text(encoding="utf-8"))
    elif args.map_path is not None:
        map_report = analyze_map(args.map_path)
    else:
        map_report = None

    report = recommend_parameters(sequence_report, map_report=map_report)
    if args.output_json is not None:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
