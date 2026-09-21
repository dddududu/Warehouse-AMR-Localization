from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from experiments.semantic_stability_20260731.evaluate_multiframe_static_refinement import _load_source_context, _metrics, _source_configs
from preprocess.dynamic_point_filter import filter_dynamic_points_by_semantics
from preprocess.local_lidar_cropper import crop_local_lidar_points
from dataset_io.lidar_loader import load_pcd_xyz


SEMI_DYNAMIC_LABELS = (5, 7, 9, 10, 11)


def _load_yaml(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("Expected a mapping.")
    return payload


def _exposures(config_path: str | Path) -> dict[tuple[str, int], float]:
    config = _load_yaml(config_path)
    stride = max(1, int(config["frame_stride"]))
    output: dict[tuple[str, int], float] = {}
    for source in _source_configs(config["sources"]):
        report = json.loads(source.result_json.read_text(encoding="utf-8"))
        frames = list(report["frame_results"])[::stride]
        dataset, _, crop = _load_source_context(source)
        for frame in frames:
            frame_idx = int(frame["frame_idx"])
            record = dataset.frame_index[frame_idx]
            points = crop_local_lidar_points(load_pcd_xyz(record.lidar_path), crop)
            _, semi_dynamic = filter_dynamic_points_by_semantics(
                points,
                calibration=dataset.calibration,
                segmentation_left_path=record.segmentation_greyscale_left_path,
                segmentation_right_path=record.segmentation_greyscale_right_path,
                dynamic_labels=SEMI_DYNAMIC_LABELS,
                dilation_px=3,
            )
            output[(source.name, frame_idx)] = float(np.mean(semi_dynamic))
    return output


def _paired_rows(dynamic_summary_path: str | Path, structural_summary_path: str | Path) -> list[dict[str, Any]]:
    dynamic = json.loads(Path(dynamic_summary_path).read_text(encoding="utf-8"))["per_sequence"]
    structural = json.loads(Path(structural_summary_path).read_text(encoding="utf-8"))["per_sequence"]
    rows: list[dict[str, Any]] = []
    for name, dynamic_result in dynamic.items():
        structural_rows = {int(row["frame_idx"]): row for row in structural[name]["rows"]}
        for row in dynamic_result["rows"]:
            frame_idx = int(row["frame_idx"])
            if frame_idx not in structural_rows:
                raise ValueError(f"Missing structural result for {name}:{frame_idx}.")
            rows.append(
                {
                    "sequence": name,
                    "frame_idx": frame_idx,
                    "dynamic_error_m": float(row["refined_error_m"]),
                    "structural_error_m": float(structural_rows[frame_idx]["refined_error_m"]),
                }
            )
    return rows


def _hybrid_metrics(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    selected = np.asarray(
        [row["structural_error_m"] if float(row["semi_dynamic_ratio"]) >= threshold else row["dynamic_error_m"] for row in rows],
        dtype=np.float64,
    )
    structural_count = sum(float(row["semi_dynamic_ratio"]) >= threshold for row in rows)
    return {**_metrics(selected), "structural_selected_ratio": structural_count / max(len(rows), 1)}


def run(
    dynamic_summary_path: str | Path,
    structural_summary_path: str | Path,
    exposure_config_path: str | Path,
    output_path: str | Path,
    thresholds: list[float],
    frozen_threshold: float | None,
) -> dict[str, Any]:
    rows = _paired_rows(dynamic_summary_path, structural_summary_path)
    exposures = _exposures(exposure_config_path)
    for row in rows:
        key = (str(row["sequence"]), int(row["frame_idx"]))
        if key not in exposures:
            raise ValueError(f"Missing semantic exposure for {key}.")
        row["semi_dynamic_ratio"] = exposures[key]
    dynamic_errors = np.asarray([row["dynamic_error_m"] for row in rows], dtype=np.float64)
    dynamic_metrics = _metrics(dynamic_errors)
    candidates = {str(threshold): _hybrid_metrics(rows, threshold) for threshold in thresholds}
    if frozen_threshold is None:
        eligible = [
            threshold
            for threshold in thresholds
            if candidates[str(threshold)]["p95_position_error_m"] <= dynamic_metrics["p95_position_error_m"]
        ]
        chosen = min(eligible, key=lambda threshold: candidates[str(threshold)]["mean_position_error_m"], default=float("inf"))
    else:
        chosen = float(frozen_threshold)
        candidates[str(chosen)] = _hybrid_metrics(rows, chosen)
    selected = candidates[str(chosen)]
    summary = {
        "dynamic_only": dynamic_metrics,
        "threshold_results": candidates,
        "selected_threshold": chosen,
        "selected_hybrid": selected,
        "selection_rule": "验证阶段在 P95 不高于动态类过滤基线的阈值中选平均误差最低者；盲测阶段只执行冻结阈值。" if frozen_threshold is None else "盲测阶段使用六月验证冻结的阈值，不在十月重新选择。",
        "row_count": len(rows),
        "semi_dynamic_ratio_percentiles": np.quantile([row["semi_dynamic_ratio"] for row in rows], [0.1, 0.5, 0.9]).tolist(),
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Select or evaluate a semidynamic-exposure trigger for class-aware ICP.")
    parser.add_argument("--dynamic-summary", required=True)
    parser.add_argument("--structural-summary", required=True)
    parser.add_argument("--exposure-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.10])
    parser.add_argument("--frozen-threshold", type=float)
    args = parser.parse_args()
    print(json.dumps(run(args.dynamic_summary, args.structural_summary, args.exposure_config, args.output, args.thresholds, args.frozen_threshold), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
