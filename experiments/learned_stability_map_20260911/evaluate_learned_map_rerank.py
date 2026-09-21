from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "semantic_stability_20260731"))

from build_semantic_stability_map import _labels_from_stereo_points
from evaluate_semantic_candidate_rerank import (
    SequenceConfig,
    _candidate_position_error,
    _metrics,
    _semantic_score_for_pose,
)
from dataset_io.lidar_loader import load_pcd_xyz
from dataset_io.sequence_dataset import WarehouseSequenceDataset
from preprocess.local_lidar_cropper import LocalCropConfig, crop_local_lidar_points


def _encode_cells(cells: np.ndarray) -> np.ndarray:
    cells = np.asarray(cells, dtype=np.int64)
    return (cells[:, 0] + 100_000) * 1_000_000 + (cells[:, 1] + 100_000)


def _load_map(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    payload = np.load(path)
    label_keys = _encode_cells(payload["label_cells"])
    order = np.argsort(label_keys)
    label_keys = label_keys[order]
    label_counts = payload["label_counts"][order].astype(np.float64)
    label_probabilities = label_counts / np.maximum(label_counts.sum(axis=1, keepdims=True), 1.0)
    stability_keys = _encode_cells(payload["cells"])
    stability_order = np.argsort(stability_keys)
    stability_keys = stability_keys[stability_order]
    learned = payload["learned_stability"][stability_order].astype(np.float64)
    heuristic = payload["heuristic_stability"][stability_order].astype(np.float64)
    positions = np.searchsorted(stability_keys, label_keys)
    matches = (positions < stability_keys.size) & (stability_keys[np.minimum(positions, stability_keys.size - 1)] == label_keys)
    learned_for_labels = np.zeros(label_keys.size, dtype=np.float64)
    heuristic_for_labels = np.zeros(label_keys.size, dtype=np.float64)
    learned_for_labels[matches] = learned[positions[matches]]
    heuristic_for_labels[matches] = heuristic[positions[matches]]
    return label_keys, label_probabilities, learned_for_labels, heuristic_for_labels


def _combined_metrics(rows: list[dict[str, Any]], methods: list[str]) -> dict[str, dict[str, float | int | None]]:
    reports: dict[str, dict[str, float | int | None]] = {}
    for method in methods:
        selected = [row for row in rows if row["method"] == method]
        errors = np.asarray([row["position_error_m"] for row in selected], dtype=np.float64)
        patches = np.asarray([row["patch_id"] for row in selected], dtype=np.int64)
        reports[method] = _metrics(errors, patches)
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description="Blind Oct12 candidate reranking with a Jun15-only learned stability map.")
    parser.add_argument("--config", required=True)
    arguments = parser.parse_args()
    config = yaml.safe_load(Path(arguments.config).read_text(encoding="utf-8"))
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    map_keys, label_probabilities, learned_stability, heuristic_stability = _load_map(Path(config["learned_map_path"]))
    crop_config = LocalCropConfig(**dict(config["crop"]))
    semantic_weight = float(config["semantic_score_weight"])
    methods = ["candidate_final_score", "heuristic_map_blend", "learned_map_blend", "oracle_topk"]
    all_rows: list[dict[str, Any]] = []
    per_sequence: dict[str, dict[str, dict[str, float | int | None]]] = {}
    for raw_sequence in config["sequences"]:
        sequence = SequenceConfig(str(raw_sequence["name"]), Path(raw_sequence["sequence_root"]), Path(raw_sequence["result_json"]))
        print(f"Scoring frozen candidates for {sequence.name}", flush=True)
        dataset = WarehouseSequenceDataset(sequence.sequence_root, config["calibration_path"], config={"load_lidar": False})
        baseline = json.loads(sequence.result_json.read_text(encoding="utf-8"))
        sequence_rows: list[dict[str, Any]] = []
        for output_index, frame_result in enumerate(baseline["frame_results"]):
            frame_idx = int(frame_result["frame_idx"])
            record = dataset.frame_index[frame_idx]
            points = crop_local_lidar_points(load_pcd_xyz(record.lidar_path), crop_config)
            labels = _labels_from_stereo_points(points, dataset, record)
            candidates = list(frame_result["candidate_results"])
            base_scores = np.asarray([float(candidate["final_score"]) for candidate in candidates], dtype=np.float64)
            errors = np.asarray([_candidate_position_error(candidate, dataset.ground_truth.positions[frame_idx]) for candidate in candidates], dtype=np.float64)
            learned_scores = np.zeros(len(candidates), dtype=np.float64)
            heuristic_scores = np.zeros(len(candidates), dtype=np.float64)
            for candidate_index, candidate in enumerate(candidates):
                pose = np.asarray(candidate["final_pose_4x4"], dtype=np.float64)
                learned_scores[candidate_index] = _semantic_score_for_pose(points, labels, pose, float(config["cell_size_m"]), map_keys, label_probabilities, learned_stability)[0]
                heuristic_scores[candidate_index] = _semantic_score_for_pose(points, labels, pose, float(config["cell_size_m"]), map_keys, label_probabilities, heuristic_stability)[0]
            chosen = {
                "candidate_final_score": 0,
                "heuristic_map_blend": int(np.argmax(base_scores + semantic_weight * heuristic_scores)),
                "learned_map_blend": int(np.argmax(base_scores + semantic_weight * learned_scores)),
                "oracle_topk": int(np.argmin(errors)),
            }
            for method, candidate_index in chosen.items():
                candidate = candidates[candidate_index]
                row = {
                    "sequence_name": sequence.name,
                    "frame_idx": frame_idx,
                    "method": method,
                    "patch_id": int(candidate["patch_id"]),
                    "position_error_m": float(errors[candidate_index]),
                    "selected_candidate_rank": candidate_index + 1,
                    "selected_base_score": float(base_scores[candidate_index]),
                    "selected_heuristic_score": float(heuristic_scores[candidate_index]),
                    "selected_learned_score": float(learned_scores[candidate_index]),
                }
                sequence_rows.append(row)
                all_rows.append(row)
            if (output_index + 1) % 100 == 0:
                print(f"[{sequence.name}] processed {output_index + 1}/{len(baseline['frame_results'])}", flush=True)
        per_sequence[sequence.name] = _combined_metrics(sequence_rows, methods)
    summary = {
        "protocol": "Oct12 is evaluated once as a blind test. All maps are built from Jun15 inputs only; learned supervision is Jun23 only. The fixed 0.10 blend weight is deliberately weak so semantic stability cannot override descriptor, classifier and ICP evidence.",
        "semantic_score_weight": semantic_weight,
        "per_sequence": per_sequence,
        "combined": _combined_metrics(all_rows, methods),
    }
    with (output_dir / "learned_map_rerank_rows.csv").open("w", newline="", encoding="utf-8-sig") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    (output_dir / "learned_map_rerank_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
