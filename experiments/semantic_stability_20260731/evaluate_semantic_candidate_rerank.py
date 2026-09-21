from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

from dataset_io.lidar_loader import load_pcd_xyz
from dataset_io.sequence_dataset import WarehouseSequenceDataset
from geometry.se3 import transform_points
from preprocess.local_lidar_cropper import LocalCropConfig, crop_local_lidar_points

from build_semantic_stability_map import LABEL_TO_GROUP, _labels_from_stereo_points


LABEL_WEIGHTS = np.zeros(len(LABEL_TO_GROUP), dtype=np.float32)
LABEL_WEIGHTS[[1, 2, 4, 6, 8]] = 1.0
LABEL_WEIGHTS[[5, 7, 9, 10, 11]] = 0.35


@dataclass(frozen=True)
class SequenceConfig:
    name: str
    sequence_root: Path
    result_json: Path


def _load_config(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("Semantic candidate-rerank config must be a mapping.")
    return payload


def _encode_cells(cells: np.ndarray) -> np.ndarray:
    indices = np.asarray(cells, dtype=np.int64)
    return (indices[:, 0] + 100_000) * 1_000_000 + (indices[:, 1] + 100_000)


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def _build_semantic_lookup(stability_map_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    payload = np.load(stability_map_path)
    label_cells = payload["updated_label_cells"]
    label_counts = payload["updated_label_counts"].astype(np.float64)
    stability_cells = payload["updated_cells"]
    stability_values = payload["updated_stability"].astype(np.float64)
    stability_keys = _encode_cells(stability_cells)
    stability_order = np.argsort(stability_keys)
    stability_keys = stability_keys[stability_order]
    stability_values = stability_values[stability_order]
    label_keys = _encode_cells(label_cells)
    label_order = np.argsort(label_keys)
    label_keys = label_keys[label_order]
    label_probabilities = label_counts[label_order] / np.maximum(label_counts[label_order].sum(axis=1, keepdims=True), 1.0)
    positions = np.searchsorted(stability_keys, label_keys)
    stability = np.zeros(label_keys.shape[0], dtype=np.float64)
    valid = (positions < stability_keys.size) & (stability_keys[np.minimum(positions, stability_keys.size - 1)] == label_keys)
    stability[valid] = stability_values[positions[valid]]
    return label_keys, label_probabilities, stability


def _semantic_score_for_pose(
    points_sensor: np.ndarray,
    labels: np.ndarray,
    pose_4x4: np.ndarray,
    cell_size_m: float,
    map_keys: np.ndarray,
    map_label_probabilities: np.ndarray,
    map_stability: np.ndarray,
) -> tuple[float, float, int]:
    valid = (labels > 0) & (labels < len(LABEL_WEIGHTS)) & (LABEL_WEIGHTS[labels] > 0.0)
    if not np.any(valid):
        return 0.0, 0.0, 0
    valid_points = points_sensor[valid]
    valid_labels = labels[valid].astype(np.int64, copy=False)
    weights = LABEL_WEIGHTS[valid_labels].astype(np.float64, copy=False)
    world_points = transform_points(np.asarray(pose_4x4, dtype=np.float64), valid_points)
    cells = np.floor(world_points[:, :2] / float(cell_size_m)).astype(np.int32)
    keys = _encode_cells(cells)
    indices = np.searchsorted(map_keys, keys)
    matched = (indices < map_keys.size) & (map_keys[np.minimum(indices, map_keys.size - 1)] == keys)
    if not np.any(matched):
        return 0.0, 0.0, 0
    matched_indices = indices[matched]
    matched_labels = valid_labels[matched]
    matched_weights = weights[matched]
    label_agreement = map_label_probabilities[matched_indices, matched_labels]
    stability = map_stability[matched_indices]
    weighted_support = np.average(label_agreement * (0.5 + 0.5 * stability), weights=matched_weights)
    coverage = float(matched_weights.sum() / max(weights.sum(), 1.0e-6))
    score = float(weighted_support * np.sqrt(max(coverage, 0.0)))
    return score, coverage, int(np.count_nonzero(matched))


def _candidate_position_error(candidate: dict[str, Any], ground_truth_position: np.ndarray) -> float:
    pose = np.asarray(candidate["final_pose_4x4"], dtype=np.float64)
    return float(np.linalg.norm(pose[:2, 3] - ground_truth_position[:2]))


def _metrics(errors: np.ndarray, patch_ids: np.ndarray) -> dict[str, float | int | None]:
    if errors.size == 0:
        return {"frames": 0, "mean_position_error_m": None, "median_position_error_m": None, "p95_position_error_m": None, "below_0p5m": None, "patch_switches": 0}
    return {
        "frames": int(errors.size),
        "mean_position_error_m": float(errors.mean()),
        "median_position_error_m": float(np.median(errors)),
        "p95_position_error_m": float(np.percentile(errors, 95.0)),
        "below_0p5m": float(np.mean(errors < 0.5)),
        "patch_switches": int(np.count_nonzero(patch_ids[1:] != patch_ids[:-1])),
    }


def _mrr_and_recall(candidate_errors: list[np.ndarray]) -> dict[str, float]:
    reciprocal_ranks = []
    recalled = []
    for errors in candidate_errors:
        success = np.flatnonzero(errors < 0.5)
        reciprocal_ranks.append(0.0 if success.size == 0 else 1.0 / float(success[0] + 1))
        recalled.append(float(success.size > 0))
    return {"mrr_pose_success": float(np.mean(reciprocal_ranks)), "recall_at_3_pose_success": float(np.mean(recalled))}


def _draw_comparison(output_path: Path, reports: dict[str, dict[str, dict[str, float | int | None]]]) -> None:
    methods = ["candidate_final_score", "semantic_blend", "oracle_topk"]
    labels = {"candidate_final_score": "原候选分数", "semantic_blend": "语义融合", "semantic_only": "仅语义", "oracle_topk": "TopK 上限"}
    colors = {"candidate_final_score": (100, 130, 180), "semantic_blend": (71, 145, 95), "semantic_only": (219, 147, 66), "oracle_topk": (150, 150, 150)}
    canvas = Image.new("RGB", (1600, 1020), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.text((70, 35), "语义候选重排：冻结 TopK 上的离线比较", fill=(18, 30, 45), font=_font(38, True))
    draw.text((70, 95), "原始候选与 ICP 结果完全固定；语义融合仅改变 TopK 内排序，图中误差由候选自己的最终位姿计算。", fill=(80, 80, 80), font=_font(21))
    chart_left, chart_top, chart_right, chart_bottom = 150, 185, 930, 720
    draw.line((chart_left, chart_bottom, chart_right, chart_bottom), fill=(60, 60, 60), width=2)
    draw.line((chart_left, chart_top, chart_left, chart_bottom), fill=(60, 60, 60), width=2)
    max_value = max(
        float(reports[sequence][method]["mean_position_error_m"] or 0.0)
        for sequence in reports
        for method in methods
    )
    max_value = max(0.12, max_value * 1.25)
    for tick in np.linspace(0.0, max_value, 5):
        y = chart_bottom - int((chart_bottom - chart_top) * tick / max_value)
        draw.line((chart_left - 8, y, chart_left, y), fill=(60, 60, 60), width=1)
        draw.text((60, y - 12), f"{tick:.2f}", fill=(70, 70, 70), font=_font(19))
    sequence_names = list(reports)
    group_width = (chart_right - chart_left) // len(sequence_names)
    bar_width = 48
    for sequence_index, sequence in enumerate(sequence_names):
        center = chart_left + group_width * sequence_index + group_width // 2
        draw.text((center - 60, chart_bottom + 24), sequence, fill=(45, 45, 45), font=_font(24, True))
        for method_index, method in enumerate(methods):
            value = float(reports[sequence][method]["mean_position_error_m"] or 0.0)
            height = int((chart_bottom - chart_top) * value / max_value)
            x0 = center + (method_index - 1.0) * (bar_width + 22)
            y0 = chart_bottom - height
            draw.rectangle((x0, y0, x0 + bar_width, chart_bottom), fill=colors[method])
            draw.text((x0 - 7, y0 - 27), f"{value:.3f}", fill=(50, 50, 50), font=_font(17))
    failure_left, failure_top, failure_right, failure_bottom = 1080, 185, 1470, 720
    draw.line((failure_left, failure_bottom, failure_right, failure_bottom), fill=(60, 60, 60), width=2)
    draw.line((failure_left, failure_top, failure_left, failure_bottom), fill=(60, 60, 60), width=2)
    failure_values = [float(reports[sequence]["semantic_only"]["mean_position_error_m"] or 0.0) for sequence in sequence_names]
    failure_max = max(2.5, max(failure_values) * 1.2)
    for tick in np.linspace(0.0, failure_max, 5):
        y = failure_bottom - int((failure_bottom - failure_top) * tick / failure_max)
        draw.line((failure_left - 8, y, failure_left, y), fill=(60, 60, 60), width=1)
        draw.text((failure_left - 75, y - 12), f"{tick:.2f}", fill=(70, 70, 70), font=_font(18))
    draw.text((failure_left, 145), "仅语义排序（失败对照）", fill=(45, 45, 45), font=_font(24, True))
    for sequence_index, sequence in enumerate(sequence_names):
        center = failure_left + (failure_right - failure_left) * (sequence_index * 2 + 1) // 4
        value = float(reports[sequence]["semantic_only"]["mean_position_error_m"] or 0.0)
        height = int((failure_bottom - failure_top) * value / failure_max)
        draw.rectangle((center - bar_width // 2, failure_bottom - height, center + bar_width // 2, failure_bottom), fill=colors["semantic_only"])
        draw.text((center - 28, failure_bottom - height - 27), f"{value:.3f}", fill=(50, 50, 50), font=_font(17))
        draw.text((center - 62, failure_bottom + 24), sequence.replace("Aisle_", ""), fill=(45, 45, 45), font=_font(22, True))
    legend_y = 820
    for index, method in enumerate(methods):
        x = 85 + index * 260
        draw.rectangle((x, legend_y, x + 26, legend_y + 26), fill=colors[method])
        draw.text((x + 40, legend_y - 2), labels[method], fill=(45, 45, 45), font=_font(22))
    draw.text((150, 920), "注：这是候选级语义信号验证，不含时序滞回重新求解；完整系统集成将在后续实验中单独测试。", fill=(70, 70, 70), font=_font(20))
    canvas.save(output_path)


def evaluate_rerank(config_path: str | Path) -> dict[str, Any]:
    config = _load_config(config_path)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    map_keys, map_label_probabilities, map_stability = _build_semantic_lookup(Path(config["stability_map_path"]))
    crop_config = LocalCropConfig(**dict(config.get("crop", {})))
    semantic_weight = float(config["semantic_score_weight"])
    cell_size_m = float(config["cell_size_m"])
    all_rows: list[dict[str, Any]] = []
    reports: dict[str, dict[str, dict[str, float | int | None]]] = {}
    ranking_metrics: dict[str, dict[str, float]] = {}
    for raw_sequence in config["sequences"]:
        sequence = SequenceConfig(
            name=str(raw_sequence["name"]),
            sequence_root=Path(raw_sequence["sequence_root"]),
            result_json=Path(raw_sequence["result_json"]),
        )
        print(f"Scoring semantic candidate consistency for {sequence.name}", flush=True)
        dataset = WarehouseSequenceDataset(sequence.sequence_root, config["calibration_path"], config={"load_lidar": False})
        baseline = json.loads(sequence.result_json.read_text(encoding="utf-8"))
        errors_by_method: dict[str, list[float]] = {method: [] for method in ("candidate_final_score", "semantic_blend", "semantic_only", "oracle_topk")}
        patches_by_method: dict[str, list[int]] = {method: [] for method in errors_by_method}
        candidate_error_rows: list[np.ndarray] = []
        for output_index, frame_result in enumerate(baseline["frame_results"]):
            frame_idx = int(frame_result["frame_idx"])
            record = dataset.frame_index[frame_idx]
            points_sensor = crop_local_lidar_points(load_pcd_xyz(record.lidar_path), crop_config)
            labels = _labels_from_stereo_points(points_sensor, dataset, record)
            candidates = list(frame_result["candidate_results"])
            candidate_errors = np.asarray(
                [_candidate_position_error(candidate, dataset.ground_truth.positions[frame_idx]) for candidate in candidates],
                dtype=np.float64,
            )
            semantic_scores = np.zeros(len(candidates), dtype=np.float64)
            semantic_coverage = np.zeros(len(candidates), dtype=np.float64)
            semantic_points = np.zeros(len(candidates), dtype=np.int32)
            for candidate_index, candidate in enumerate(candidates):
                score, coverage, count = _semantic_score_for_pose(
                    points_sensor,
                    labels,
                    np.asarray(candidate["final_pose_4x4"], dtype=np.float64),
                    cell_size_m,
                    map_keys,
                    map_label_probabilities,
                    map_stability,
                )
                semantic_scores[candidate_index] = score
                semantic_coverage[candidate_index] = coverage
                semantic_points[candidate_index] = count
                all_rows.append(
                    {
                        "sequence_name": sequence.name,
                        "frame_idx": frame_idx,
                        "candidate_rank_by_final_score": candidate_index + 1,
                        "patch_id": int(candidate["patch_id"]),
                        "candidate_final_score": float(candidate["final_score"]),
                        "semantic_score": float(score),
                        "semantic_coverage": float(coverage),
                        "matched_semantic_points": int(count),
                        "candidate_position_error_m": float(candidate_errors[candidate_index]),
                    }
                )
            base_scores = np.asarray([float(candidate["final_score"]) for candidate in candidates], dtype=np.float64)
            selected_indices = {
                "candidate_final_score": 0,
                "semantic_blend": int(np.argmax(base_scores + semantic_weight * semantic_scores)),
                "semantic_only": int(np.argmax(semantic_scores)),
                "oracle_topk": int(np.argmin(candidate_errors)),
            }
            for method, selected_index in selected_indices.items():
                errors_by_method[method].append(float(candidate_errors[selected_index]))
                patches_by_method[method].append(int(candidates[selected_index]["patch_id"]))
            candidate_error_rows.append(candidate_errors)
            if (output_index + 1) % 100 == 0:
                print(f"[{sequence.name}] processed {output_index + 1}/{len(baseline['frame_results'])} frames", flush=True)
        reports[sequence.name] = {
            method: _metrics(np.asarray(errors_by_method[method]), np.asarray(patches_by_method[method]))
            for method in errors_by_method
        }
        ranking_metrics[sequence.name] = _mrr_and_recall(candidate_error_rows)
    combined_reports: dict[str, dict[str, float | int | None]] = {}
    for method in ("candidate_final_score", "semantic_blend", "semantic_only", "oracle_topk"):
        errors = []
        patch_ids = []
        for sequence_name in reports:
            sequence_rows = [row for row in all_rows if row["sequence_name"] == sequence_name]
            # Combined metrics are recomputed from selected rows in the sequence-level report output below.
            del sequence_rows
        for raw_sequence in config["sequences"]:
            sequence_name = str(raw_sequence["name"])
            sequence = SequenceConfig(sequence_name, Path(raw_sequence["sequence_root"]), Path(raw_sequence["result_json"]))
            baseline = json.loads(sequence.result_json.read_text(encoding="utf-8"))
            grouped = [row for row in all_rows if row["sequence_name"] == sequence_name]
            frame_groups: dict[int, list[dict[str, Any]]] = {}
            for row in grouped:
                frame_groups.setdefault(int(row["frame_idx"]), []).append(row)
            for frame_result in baseline["frame_results"]:
                frame_rows = sorted(frame_groups[int(frame_result["frame_idx"])], key=lambda item: int(item["candidate_rank_by_final_score"]))
                final_scores = np.asarray([float(item["candidate_final_score"]) for item in frame_rows])
                semantic_scores = np.asarray([float(item["semantic_score"]) for item in frame_rows])
                candidate_errors = np.asarray([float(item["candidate_position_error_m"]) for item in frame_rows])
                if method == "candidate_final_score":
                    index = 0
                elif method == "semantic_blend":
                    index = int(np.argmax(final_scores + semantic_weight * semantic_scores))
                elif method == "semantic_only":
                    index = int(np.argmax(semantic_scores))
                else:
                    index = int(np.argmin(candidate_errors))
                errors.append(float(candidate_errors[index]))
                patch_ids.append(int(frame_rows[index]["patch_id"]))
        combined_reports[method] = _metrics(np.asarray(errors), np.asarray(patch_ids))
    summary = {
        "semantic_score_weight": semantic_weight,
        "note": "The weight was fixed before Oct.12 scoring. This experiment freezes TopK candidates and candidate ICP poses, then measures only candidate-level reranking; it does not rerun online hysteresis.",
        "per_sequence": reports,
        "combined": combined_reports,
        "candidate_availability": ranking_metrics,
    }
    with (output_dir / "semantic_candidate_rerank_rows.csv").open("w", newline="", encoding="utf-8-sig") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    (output_dir / "semantic_candidate_rerank_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _draw_comparison(output_dir / "semantic_candidate_rerank_comparison.png", reports)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate candidate-level semantic consistency on frozen Aisle TopK results.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    evaluate_rerank(args.config)


if __name__ == "__main__":
    main()
