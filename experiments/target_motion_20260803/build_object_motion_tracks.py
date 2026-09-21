from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from dataset_io.depth_loader import load_depth_png
from dataset_io.sequence_dataset import WarehouseSequenceDataset
from geometry.se3 import transform_points


@dataclass(frozen=True)
class TargetClass:
    class_id: int
    name: str


@dataclass
class TrackState:
    track_id: int
    target_name: str
    last_frame_idx: int
    last_center_world: np.ndarray


def _load_config(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("Object motion config must be a mapping.")
    return payload


def _component_observations(
    mask: np.ndarray,
    depth_raw: np.ndarray,
    class_id: int,
    dataset: WarehouseSequenceDataset,
    pose_4x4: np.ndarray,
    analysis: dict[str, Any],
) -> list[dict[str, float | int | np.ndarray]]:
    binary = (mask == int(class_id)).astype(np.uint8)
    if not np.any(binary):
        return []
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    camera = dataset.camera_left
    observations: list[dict[str, float | int | np.ndarray]] = []
    for component_id in range(1, count):
        x, y, width, height, area = stats[component_id].tolist()
        if int(area) < int(analysis["min_component_pixels"]):
            continue
        pixels_y, pixels_x = np.where(labels == component_id)
        depth_m = depth_raw[pixels_y, pixels_x].astype(np.float64) * float(analysis["depth_scale_m"])
        valid = np.isfinite(depth_m) & (depth_m >= float(analysis["depth_min_m"])) & (depth_m <= float(analysis["depth_max_m"]))
        if int(valid.sum()) < int(analysis["min_valid_depth_pixels"]):
            continue
        pixels_x = pixels_x[valid].astype(np.float64)
        pixels_y = pixels_y[valid].astype(np.float64)
        depth_m = depth_m[valid]
        if depth_m.size > 1600:
            selection = np.linspace(0, depth_m.size - 1, 1600, dtype=np.int64)
            pixels_x, pixels_y, depth_m = pixels_x[selection], pixels_y[selection], depth_m[selection]
        points_camera = np.column_stack(
            (
                (pixels_x - float(camera.cx)) * depth_m / float(camera.fx),
                (pixels_y - float(camera.cy)) * depth_m / float(camera.fy),
                depth_m,
            )
        )
        points_sensor = transform_points(dataset.calibration.T_os_cam_left, points_camera)
        points_world = transform_points(pose_4x4, points_sensor)
        center_world = np.median(points_world, axis=0)
        radial_extent = float(np.linalg.norm(np.percentile(points_world, 90, axis=0) - np.percentile(points_world, 10, axis=0)))
        observations.append(
            {
                "center_world": center_world,
                "pixel_area": int(area),
                "bbox_x": int(x),
                "bbox_y": int(y),
                "bbox_width": int(width),
                "bbox_height": int(height),
                "median_depth_m": float(np.median(depth_m)),
                "radial_extent_m": radial_extent,
                "valid_depth_pixels": int(depth_m.size),
            }
        )
    return observations


def _associate_tracks(
    states: list[TrackState],
    observations: list[dict[str, float | int | np.ndarray]],
    frame_idx: int,
    max_distance_m: float,
    max_gap_frames: int,
    next_track_id: int,
) -> tuple[list[int], list[TrackState], int]:
    active = [state for state in states if int(frame_idx) - state.last_frame_idx <= int(max_gap_frames)]
    pairs: list[tuple[float, int, int]] = []
    for state_index, state in enumerate(active):
        for observation_index, observation in enumerate(observations):
            distance = float(np.linalg.norm(np.asarray(observation["center_world"]) - state.last_center_world))
            if distance <= float(max_distance_m):
                pairs.append((distance, state_index, observation_index))
    assignments = [-1] * len(observations)
    used_states: set[int] = set()
    used_observations: set[int] = set()
    for _, state_index, observation_index in sorted(pairs):
        if state_index in used_states or observation_index in used_observations:
            continue
        assignments[observation_index] = active[state_index].track_id
        active[state_index].last_frame_idx = int(frame_idx)
        active[state_index].last_center_world = np.asarray(observations[observation_index]["center_world"], dtype=np.float64)
        used_states.add(state_index)
        used_observations.add(observation_index)
    for observation_index, track_id in enumerate(assignments):
        if track_id >= 0:
            continue
        assignments[observation_index] = next_track_id
        active.append(
            TrackState(
                track_id=next_track_id,
                target_name="",
                last_frame_idx=int(frame_idx),
                last_center_world=np.asarray(observations[observation_index]["center_world"], dtype=np.float64),
            )
        )
        next_track_id += 1
    return assignments, active, next_track_id


def _add_temporal_labels(rows: list[dict[str, Any]], analysis: dict[str, Any]) -> list[dict[str, Any]]:
    rows_by_track: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_track[(str(row["sequence"]), str(row["target_name"]), int(row["track_id"]))].append(row)
    eligible_rows: list[dict[str, Any]] = []
    for track_rows in rows_by_track.values():
        track_rows.sort(key=lambda row: float(row["timestamp_sec"]))
        if len(track_rows) < int(analysis["min_track_observations"]):
            continue
        for index, row in enumerate(track_rows):
            previous = track_rows[max(0, index - 3):index]
            if previous:
                reference = previous[0]
                elapsed = max(float(row["timestamp_sec"]) - float(reference["timestamp_sec"]), 1.0e-6)
                past_speed = float(np.linalg.norm(np.asarray(row["center_world"]) - np.asarray(reference["center_world"])) / elapsed)
            else:
                past_speed = 0.0
            target_time = float(row["timestamp_sec"]) + float(analysis["future_horizon_sec"])
            future = min(track_rows[index + 1:], key=lambda item: abs(float(item["timestamp_sec"]) - target_time), default=None)
            if future is None or abs(float(future["timestamp_sec"]) - target_time) > float(analysis["future_horizon_tolerance_sec"]):
                continue
            future_elapsed = max(float(future["timestamp_sec"]) - float(row["timestamp_sec"]), 1.0e-6)
            future_speed = float(np.linalg.norm(np.asarray(future["center_world"]) - np.asarray(row["center_world"])) / future_elapsed)
            row["past_speed_mps"] = past_speed
            row["future_speed_mps"] = future_speed
            row["pseudo_dynamic"] = bool(future_speed >= float(analysis["dynamic_speed_threshold_mps"]))
            row["track_observations"] = len(track_rows)
            eligible_rows.append(row)
    return eligible_rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    serializable_rows: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        center = np.asarray(item.pop("center_world"), dtype=np.float64)
        item["world_x"] = float(center[0])
        item["world_y"] = float(center[1])
        item["world_z"] = float(center[2])
        serializable_rows.append(item)
    fieldnames = list(serializable_rows[0])
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(serializable_rows)


def run(config_path: str | Path) -> dict[str, Any]:
    config = _load_config(config_path)
    analysis = dict(config["analysis"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    target_classes = [TargetClass(**item) for item in config["target_classes"]]
    all_rows: list[dict[str, Any]] = []
    for sequence in config["sequences"]:
        dataset = WarehouseSequenceDataset(sequence["sequence_root"], sequence["calibration_path"], {"load_lidar": False})
        active_by_target: dict[str, list[TrackState]] = {target.name: [] for target in target_classes}
        next_track_id_by_target = {target.name: 0 for target in target_classes}
        for frame_idx, record in enumerate(dataset.frame_index):
            mask = cv2.imread(str(record.segmentation_greyscale_left_path), cv2.IMREAD_UNCHANGED)
            depth = load_depth_png(record.depth_left_path)
            if mask is None or mask.shape != depth.shape:
                raise RuntimeError(f"Invalid semantic/depth pair for {sequence['name']} frame {frame_idx}.")
            pose = dataset.ground_truth.poses_4x4[frame_idx]
            for target in target_classes:
                observations = _component_observations(mask, depth, target.class_id, dataset, pose, analysis)
                assignments, active_states, next_track_id = _associate_tracks(
                    active_by_target[target.name],
                    observations,
                    frame_idx,
                    float(analysis["association_max_distance_m"]),
                    int(analysis["association_max_gap_frames"]),
                    next_track_id_by_target[target.name],
                )
                active_by_target[target.name] = active_states
                next_track_id_by_target[target.name] = next_track_id
                for observation, track_id in zip(observations, assignments, strict=True):
                    all_rows.append(
                        {
                            "sequence": sequence["name"],
                            "split": sequence["split"],
                            "frame_idx": frame_idx,
                            "timestamp_sec": float(dataset.frame_times[frame_idx]),
                            "target_id": target.class_id,
                            "target_name": target.name,
                            "track_id": int(track_id),
                            **observation,
                        }
                    )
        print(f"tracked {sequence['name']}: {len(dataset)} frames")
    labeled_rows = _add_temporal_labels(all_rows, analysis)
    _write_csv(output_dir / "object_track_observations.csv", labeled_rows)
    summary: dict[str, Any] = {
        "total_component_observations": len(all_rows),
        "eligible_temporal_observations": len(labeled_rows),
        "by_split": {},
        "by_target": {},
        "analysis": analysis,
        "limitations": [
            "动静标签由真值位姿补偿后的未来三维位移自动生成，只能作为弱监督，不等同于人工真实运动标注。",
            "跟踪关联仅限同类目标和短时间窗口；遮挡、类别误分和多目标交叉会带来伪标签噪声。",
        ],
    }
    for key_name, selector in (("by_split", "split"), ("by_target", "target_name")):
        values = sorted({str(row[selector]) for row in labeled_rows})
        for value in values:
            selected = [row for row in labeled_rows if str(row[selector]) == value]
            dynamic = [row for row in selected if bool(row["pseudo_dynamic"])]
            summary[key_name][value] = {
                "observations": len(selected),
                "pseudo_dynamic_ratio": len(dynamic) / max(len(selected), 1),
                "tracks": len({(row["sequence"], row["target_name"], int(row["track_id"])) for row in selected}),
            }
    (output_dir / "object_motion_track_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build weakly supervised object motion tracks from semantic masks and depth.")
    parser.add_argument("--config", default="experiments/target_motion_20260803/object_motion_tracks.yaml")
    args = parser.parse_args()
    print(json.dumps(run(args.config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
