from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from analysis.analyze_map import load_map_vertices
from dataset_io.lidar_loader import load_pcd_xyz
from dataset_io.sequence_dataset import WarehouseSequenceDataset
from geometry.se3 import compose_transform, transform_points
from geometry.yaw_utils import wrap_to_pi, yaw_from_pose_matrix
from localization.config import load_fine_localization_config
from localization.icp_refiner import ICPResult, _estimate_planar_delta, voxel_downsample
from preprocess.dynamic_point_filter import filter_dynamic_points_by_semantics
from preprocess.local_lidar_cropper import LocalCropConfig, crop_local_lidar_points
from retrieval.config import load_coarse_retrieval_config


@dataclass(frozen=True)
class SourceConfig:
    name: str
    result_json: Path
    fine_config: Path


def _load_config(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("Multi-frame refinement config must be a mapping.")
    return payload


def _source_configs(raw_sources: list[dict[str, Any]]) -> list[SourceConfig]:
    return [
        SourceConfig(
            name=str(item["name"]),
            result_json=Path(item["result_json"]),
            fine_config=Path(item["fine_config"]),
        )
        for item in raw_sources
    ]


def _trimmed_refine_delta(
    source_world_xyz: np.ndarray,
    map_points_xyz_world: np.ndarray,
    *,
    voxel_size_m: float,
    max_iterations: int,
    max_correspondence_distance_m: float,
    min_correspondences: int,
    trim_ratio: float,
) -> ICPResult:
    source_down = voxel_downsample(source_world_xyz, voxel_size_m)
    map_down = voxel_downsample(map_points_xyz_world, voxel_size_m)
    identity = np.eye(4, dtype=np.float64)
    if source_down.shape[0] == 0 or map_down.shape[0] == 0:
        return ICPResult(identity, 0, 0.0, float("inf"), False, 0)
    flann_index = cv2.flann_Index(np.asarray(map_down, dtype=np.float32), {"algorithm": 1, "trees": 4})
    pose = identity.copy()
    max_corr_sq = float(max_correspondence_distance_m) ** 2
    best_num_inliers = 0
    best_rmse = float("inf")
    converged = False
    iteration_idx = 0
    for iteration_idx in range(max(1, int(max_iterations))):
        transformed = transform_points(pose, source_down).astype(np.float64)
        nearest_indices, nearest_sq = flann_index.knnSearch(
            np.asarray(transformed, dtype=np.float32),
            1,
            params={},
        )
        nearest_sq = nearest_sq.reshape(-1).astype(np.float64)
        nearest_indices = nearest_indices.reshape(-1).astype(np.int64)
        inlier_indices = np.flatnonzero(nearest_sq <= max_corr_sq)
        if inlier_indices.size < int(min_correspondences):
            break
        keep_count = max(int(min_correspondences), int(math.floor(inlier_indices.size * float(trim_ratio))))
        keep_count = min(keep_count, inlier_indices.size)
        retained = inlier_indices[np.argsort(nearest_sq[inlier_indices])[:keep_count]]
        matched_source = transformed[retained]
        matched_target = map_down[nearest_indices[retained]].astype(np.float64)
        rmse = float(np.sqrt(np.mean(np.sum((matched_source - matched_target) ** 2, axis=1))))
        if retained.size > best_num_inliers or (retained.size == best_num_inliers and rmse < best_rmse):
            best_num_inliers = int(retained.size)
            best_rmse = rmse
        delta, yaw_delta = _estimate_planar_delta(matched_source, matched_target)
        pose = compose_transform(delta, pose)
        if abs(yaw_delta) < 1.0e-3 and float(np.linalg.norm(delta[:3, 3])) < 1.0e-3:
            converged = True
            break
    return ICPResult(
        pose_4x4=pose,
        num_inliers=best_num_inliers,
        inlier_ratio=float(best_num_inliers / max(1, source_down.shape[0])),
        rmse=best_rmse,
        converged=converged,
        iterations=iteration_idx + 1,
    )


def _metrics(errors: np.ndarray) -> dict[str, float]:
    return {
        "mean_position_error_m": float(errors.mean()),
        "median_position_error_m": float(np.median(errors)),
        "p95_position_error_m": float(np.percentile(errors, 95)),
        "below_0p5m": float(np.mean(errors < 0.5)),
    }


def _load_source_context(source: SourceConfig) -> tuple[WarehouseSequenceDataset, np.ndarray, LocalCropConfig]:
    fine_config = load_fine_localization_config(source.fine_config)
    coarse_config = load_coarse_retrieval_config(fine_config.coarse_config_path)
    entry = coarse_config.resolve_single_sequence_entry(prefer_validation=True)
    dataset = WarehouseSequenceDataset(
        sequence_root=entry["sequence_root"],
        calibration_path=entry["calibration_path"],
        config={"load_lidar": False},
    )
    map_points, _ = load_map_vertices(entry["map_path"], cache_dir=Path(coarse_config.cache_dir) / "map_vertices")
    crop = LocalCropConfig(
        x_min=coarse_config.crop_x_min,
        x_max=coarse_config.crop_x_max,
        y_min=coarse_config.crop_y_min,
        y_max=coarse_config.crop_y_max,
        z_min=coarse_config.crop_z_min,
        z_max=coarse_config.crop_z_max,
    )
    return dataset, map_points, crop


def _query_static_points(
    dataset: WarehouseSequenceDataset,
    frame_idx: int,
    crop: LocalCropConfig,
    dynamic_labels: tuple[int, ...],
) -> np.ndarray:
    record = dataset.frame_index[int(frame_idx)]
    points = crop_local_lidar_points(load_pcd_xyz(record.lidar_path), crop)
    points, _ = filter_dynamic_points_by_semantics(
        points,
        calibration=dataset.calibration,
        segmentation_left_path=record.segmentation_greyscale_left_path,
        segmentation_right_path=record.segmentation_greyscale_right_path,
        dynamic_labels=dynamic_labels,
        dilation_px=3,
    )
    return points


def evaluate_source(source: SourceConfig, config: dict[str, Any]) -> dict[str, Any]:
    report = json.loads(source.result_json.read_text(encoding="utf-8"))
    frames = list(report.get("frame_results", []))
    frame_stride = max(1, int(config.get("frame_stride", 1)))
    frames = frames[::frame_stride]
    if not frames:
        raise ValueError(f"No frame results in {source.result_json}.")
    dataset, map_points, crop = _load_source_context(source)
    dynamic_labels = tuple(int(value) for value in config.get("dynamic_labels", [12, 13, 14, 15]))
    point_cache: dict[int, np.ndarray] = {}
    world_sources: list[tuple[int, np.ndarray, np.ndarray]] = []
    rows: list[dict[str, Any]] = []
    window_size = int(config.get("window_size", 3))
    max_pose_jump = float(config.get("max_history_pose_jump_m", 1.5))
    for frame in frames:
        frame_idx = int(frame["frame_idx"])
        base_pose = np.asarray(frame["pred_pose_4x4"], dtype=np.float64)
        points = point_cache.get(frame_idx)
        if points is None:
            points = _query_static_points(dataset, frame_idx, crop, dynamic_labels)
            point_cache[frame_idx] = points
        world_sources.append((frame_idx, base_pose, transform_points(base_pose, points)))
        usable_sources: list[np.ndarray] = []
        current_xy = base_pose[:2, 3]
        for _, history_pose, history_points_world in reversed(world_sources):
            if float(np.linalg.norm(history_pose[:2, 3] - current_xy)) > max_pose_jump:
                continue
            usable_sources.append(history_points_world)
            if len(usable_sources) >= window_size:
                break
        fused_points_world = np.concatenate(usable_sources, axis=0) if usable_sources else np.empty((0, 3), dtype=np.float32)
        refinement = _trimmed_refine_delta(
            fused_points_world,
            map_points,
            voxel_size_m=float(config.get("voxel_size_m", 0.3)),
            max_iterations=int(config.get("max_iterations", 20)),
            max_correspondence_distance_m=float(config.get("max_correspondence_distance_m", 0.8)),
            min_correspondences=int(config.get("min_correspondences", 36)),
            trim_ratio=float(config.get("trim_ratio", 0.7)),
        )
        delta_xy_m = float(np.linalg.norm(refinement.pose_4x4[:2, 3]))
        delta_yaw_deg = abs(math.degrees(float(yaw_from_pose_matrix(refinement.pose_4x4))))
        accepted = bool(
            refinement.num_inliers >= int(config.get("min_correspondences", 36))
            and refinement.inlier_ratio >= float(config.get("min_inlier_ratio", 0.55))
            and np.isfinite(refinement.rmse)
            and delta_xy_m <= float(config.get("max_delta_translation_m", 0.45))
            and delta_yaw_deg <= float(config.get("max_delta_yaw_deg", 3.0))
        )
        refined_pose = compose_transform(refinement.pose_4x4, base_pose) if accepted else base_pose
        gt_pose = dataset.ground_truth.poses_4x4[frame_idx]
        baseline_error = float(np.linalg.norm(base_pose[:2, 3] - gt_pose[:2, 3]))
        refined_error = float(np.linalg.norm(refined_pose[:2, 3] - gt_pose[:2, 3]))
        rows.append(
            {
                "frame_idx": frame_idx,
                "baseline_error_m": baseline_error,
                "refined_error_m": refined_error,
                "improvement_m": baseline_error - refined_error,
                "accepted": accepted,
                "history_frames": len(usable_sources),
                "num_inliers": int(refinement.num_inliers),
                "inlier_ratio": float(refinement.inlier_ratio),
                "rmse": float(refinement.rmse),
                "delta_translation_m": delta_xy_m,
                "delta_yaw_deg": delta_yaw_deg,
            }
        )
        if len(world_sources) > window_size * 3:
            world_sources = world_sources[-window_size * 3:]
    baseline_errors = np.asarray([row["baseline_error_m"] for row in rows], dtype=np.float64)
    refined_errors = np.asarray([row["refined_error_m"] for row in rows], dtype=np.float64)
    return {
        "name": source.name,
        "baseline": _metrics(baseline_errors),
        "multiframe_static": _metrics(refined_errors),
        "mean_improvement_m": float(baseline_errors.mean() - refined_errors.mean()),
        "accepted_rate": float(np.mean([row["accepted"] for row in rows])),
        "improved_frame_ratio": float(np.mean(refined_errors < baseline_errors)),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate semantic static multi-frame trimmed ICP refinement.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = _load_config(args.config)
    results = [evaluate_source(source, config) for source in _source_configs(config["sources"])]
    baseline_errors = np.concatenate(
        [np.asarray([row["baseline_error_m"] for row in result["rows"]]) for result in results]
    )
    refined_errors = np.concatenate(
        [np.asarray([row["refined_error_m"] for row in result["rows"]]) for result in results]
    )
    summary = {
        "config_path": str(Path(args.config)),
        "per_sequence": {result["name"]: result for result in results},
        "combined": {
            "baseline": _metrics(baseline_errors),
            "multiframe_static": _metrics(refined_errors),
            "mean_improvement_m": float(baseline_errors.mean() - refined_errors.mean()),
            "accepted_rate": float(np.mean([row["accepted"] for result in results for row in result["rows"]])),
            "improved_frame_ratio": float(np.mean(refined_errors < baseline_errors)),
        },
    }
    output_path = Path(config["output_json"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
