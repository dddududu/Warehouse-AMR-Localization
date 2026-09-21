from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

from dataset_io.lidar_loader import load_pcd_xyz
from dataset_io.sequence_dataset import WarehouseSequenceDataset
from geometry.se3 import transform_points
from preprocess.local_lidar_cropper import LocalCropConfig, crop_local_lidar_points


GROUP_NAMES = ("static", "semi_dynamic", "dynamic")
GROUP_COLORS = {
    "static": (61, 139, 74),
    "semi_dynamic": (230, 150, 51),
    "dynamic": (197, 67, 67),
}
LABEL_TO_GROUP = np.full(16, -1, dtype=np.int8)
LABEL_TO_GROUP[[1, 2, 4, 6, 8]] = 0
LABEL_TO_GROUP[[5, 7, 9, 10, 11]] = 1
LABEL_TO_GROUP[[12, 13, 14, 15]] = 2


@dataclass(frozen=True)
class SequenceEntry:
    name: str
    role: str
    sequence_root: Path
    calibration_path: Path


def _load_config(config_path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("Semantic stability config must be a mapping.")
    return payload


def _load_label_image(path: Path | None) -> np.ndarray | None:
    if path is None:
        return None
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim != 2:
        return None
    return image


def _project_semantic_labels(
    points_sensor: np.ndarray,
    label_image: np.ndarray | None,
    camera,
    transform_camera_sensor: np.ndarray,
) -> np.ndarray:
    labels = np.full(points_sensor.shape[0], -1, dtype=np.int16)
    if label_image is None or points_sensor.size == 0:
        return labels
    points_camera = transform_points(transform_camera_sensor, points_sensor).astype(np.float64)
    finite = np.isfinite(points_camera).all(axis=1)
    if not np.any(finite):
        return labels
    projected, _ = cv2.projectPoints(
        points_camera[finite],
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        camera.build_K(),
        camera.distortion,
    )
    image_points = projected.reshape(-1, 2)
    finite_image_points = np.isfinite(image_points).all(axis=1) & (np.abs(image_points).max(axis=1) < 1.0e9)
    rounded = np.zeros_like(image_points, dtype=np.int64)
    rounded[finite_image_points] = np.rint(image_points[finite_image_points]).astype(np.int64)
    source_indices = np.flatnonzero(finite)
    valid = (
        (points_camera[finite, 2] > 0.0)
        & finite_image_points
        & (rounded[:, 0] >= 0)
        & (rounded[:, 0] < label_image.shape[1])
        & (rounded[:, 1] >= 0)
        & (rounded[:, 1] < label_image.shape[0])
    )
    valid_indices = source_indices[valid]
    labels[valid_indices] = label_image[rounded[valid, 1], rounded[valid, 0]].astype(np.int16)
    return labels


def _labels_from_stereo_points(points_sensor: np.ndarray, dataset: WarehouseSequenceDataset, record) -> np.ndarray:
    left = _project_semantic_labels(
        points_sensor,
        _load_label_image(record.segmentation_greyscale_left_path),
        dataset.camera_left,
        dataset.calibration.T_cam1_os.matrix,
    )
    right = _project_semantic_labels(
        points_sensor,
        _load_label_image(record.segmentation_greyscale_right_path),
        dataset.camera_right,
        dataset.calibration.T_cam2_os.matrix,
    )
    return np.where(left >= 0, left, right).astype(np.int16, copy=False)


def _dominant_cell_groups(world_points: np.ndarray, groups: np.ndarray, cell_size_m: float) -> tuple[np.ndarray, np.ndarray]:
    valid = groups >= 0
    if not np.any(valid):
        return np.empty((0, 2), dtype=np.int32), np.empty((0,), dtype=np.int8)
    cell_indices = np.floor(world_points[valid, :2] / float(cell_size_m)).astype(np.int32)
    valid_groups = groups[valid].astype(np.int64, copy=False)
    cells, inverse = np.unique(cell_indices, axis=0, return_inverse=True)
    group_counts = np.bincount(inverse * len(GROUP_NAMES) + valid_groups, minlength=cells.shape[0] * len(GROUP_NAMES))
    dominant_groups = group_counts.reshape(cells.shape[0], len(GROUP_NAMES)).argmax(axis=1).astype(np.int8)
    return cells, dominant_groups


def _dominant_cell_labels(world_points: np.ndarray, labels: np.ndarray, cell_size_m: float) -> tuple[np.ndarray, np.ndarray]:
    valid = (labels > 0) & (labels < len(LABEL_TO_GROUP))
    if not np.any(valid):
        return np.empty((0, 2), dtype=np.int32), np.empty((0,), dtype=np.int8)
    cell_indices = np.floor(world_points[valid, :2] / float(cell_size_m)).astype(np.int32)
    valid_labels = labels[valid].astype(np.int64, copy=False)
    cells, inverse = np.unique(cell_indices, axis=0, return_inverse=True)
    label_counts = np.bincount(inverse * len(LABEL_TO_GROUP) + valid_labels, minlength=cells.shape[0] * len(LABEL_TO_GROUP))
    dominant_labels = label_counts.reshape(cells.shape[0], len(LABEL_TO_GROUP)).argmax(axis=1).astype(np.int8)
    return cells, dominant_labels


def _update_store(store: dict[tuple[int, int], np.ndarray], cells: np.ndarray, groups: np.ndarray) -> None:
    for cell, group in zip(cells.tolist(), groups.tolist()):
        value = store[tuple(cell)]
        value[0] += 1
        value[int(group) + 1] += 1


def _update_label_store(store: dict[tuple[int, int], np.ndarray], cells: np.ndarray, labels: np.ndarray) -> None:
    for cell, label in zip(cells.tolist(), labels.tolist()):
        store[tuple(cell)][int(label)] += 1


def _stability_score(counts: np.ndarray, min_observations: int) -> np.ndarray:
    counts_array = np.asarray(counts, dtype=np.float64)
    observed = counts_array[..., 0]
    semantic_total = np.maximum(counts_array[..., 1:].sum(axis=-1), 1.0)
    semantic_reliability = (counts_array[..., 1] + 0.5 * counts_array[..., 2]) / semantic_total
    observation_confidence = 1.0 - np.exp(-observed / max(float(min_observations), 1.0))
    return (semantic_reliability * observation_confidence).astype(np.float32)


def _store_arrays(store: dict[tuple[int, int], np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    if not store:
        return np.empty((0, 2), dtype=np.int32), np.empty((0, 4), dtype=np.int32)
    keys = np.asarray(sorted(store), dtype=np.int32)
    counts = np.stack([store[tuple(key)] for key in keys], axis=0).astype(np.int32)
    return keys, counts


def _combine_stores(*stores: dict[tuple[int, int], np.ndarray]) -> dict[tuple[int, int], np.ndarray]:
    combined: dict[tuple[int, int], np.ndarray] = {}
    for store in stores:
        for key, value in store.items():
            if key not in combined:
                combined[key] = np.zeros_like(value, dtype=np.int32)
            combined[key] += value
    return combined


def _shared_consistency(
    first: dict[tuple[int, int], np.ndarray],
    second: dict[tuple[int, int], np.ndarray],
    min_observations: int,
) -> dict[str, float | int | None]:
    shared = sorted(set(first).intersection(second))
    eligible = [
        key
        for key in shared
        if first[key][0] >= int(min_observations) and second[key][0] >= int(min_observations)
    ]
    if not eligible:
        return {
            "shared_cells": int(len(shared)),
            "eligible_cells": 0,
            "dominant_group_agreement": None,
            "mean_abs_stability_change": None,
        }
    first_counts = np.stack([first[key] for key in eligible], axis=0)
    second_counts = np.stack([second[key] for key in eligible], axis=0)
    agreement = float(
        np.mean(first_counts[:, 1:].argmax(axis=1) == second_counts[:, 1:].argmax(axis=1))
    )
    first_score = _stability_score(first_counts, min_observations)
    second_score = _stability_score(second_counts, min_observations)
    return {
        "shared_cells": int(len(shared)),
        "eligible_cells": int(len(eligible)),
        "dominant_group_agreement": agreement,
        "mean_abs_stability_change": float(np.abs(first_score - second_score).mean()),
    }


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def _draw_stability_panel(
    canvas: Image.Image,
    origin: tuple[int, int],
    title: str,
    cells: np.ndarray,
    counts: np.ndarray,
    score: np.ndarray,
    cell_size_m: float,
    cell_px: int,
) -> tuple[int, int]:
    draw = ImageDraw.Draw(canvas)
    title_font = _font(28, bold=True)
    label_font = _font(20)
    x0, y0 = origin
    draw.text((x0, y0), title, fill=(18, 30, 45), font=title_font)
    if cells.size == 0:
        draw.text((x0, y0 + 55), "没有可用语义栅格", fill=(110, 110, 110), font=label_font)
        return 300, 200
    min_cell = cells.min(axis=0)
    max_cell = cells.max(axis=0)
    width_cells, height_cells = (max_cell - min_cell + 1).tolist()
    map_image = np.full((height_cells * cell_px, width_cells * cell_px, 3), 245, dtype=np.uint8)
    color_values = cv2.applyColorMap(np.clip(score * 255.0, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    for index, cell in enumerate(cells):
        cell_x = int(cell[0] - min_cell[0])
        cell_y = int(max_cell[1] - cell[1])
        color = tuple(int(value) for value in color_values[index, 0])
        cv2.rectangle(
            map_image,
            (cell_x * cell_px, cell_y * cell_px),
            ((cell_x + 1) * cell_px - 1, (cell_y + 1) * cell_px - 1),
            color,
            thickness=-1,
        )
    map_rgb = cv2.cvtColor(map_image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(map_rgb)
    map_y = y0 + 52
    canvas.paste(image, (x0, map_y))
    draw.rectangle((x0, map_y, x0 + image.width, map_y + image.height), outline=(80, 80, 80), width=1)
    draw.text((x0, map_y + image.height + 8), f"X: {min_cell[0] * cell_size_m:.1f} 至 {max_cell[0] * cell_size_m:.1f} m", fill=(60, 60, 60), font=label_font)
    draw.text((x0, map_y + image.height + 34), f"Y: {min_cell[1] * cell_size_m:.1f} 至 {max_cell[1] * cell_size_m:.1f} m", fill=(60, 60, 60), font=label_font)
    return image.width, image.height + 90


def _write_stability_map_figure(
    output_path: Path,
    initial_cells: np.ndarray,
    initial_counts: np.ndarray,
    initial_score: np.ndarray,
    updated_cells: np.ndarray,
    updated_counts: np.ndarray,
    updated_score: np.ndarray,
    cell_size_m: float,
) -> None:
    canvas = Image.new("RGB", (1800, 800), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.text((70, 30), "Aisle 语义稳定性地图：初始地图与跨日期更新", fill=(18, 30, 45), font=_font(38, bold=True))
    _draw_stability_panel(canvas, (70, 110), "Jun.15 初始语义稳定性", initial_cells, initial_counts, initial_score, cell_size_m, 5)
    _draw_stability_panel(canvas, (920, 110), "Jun.15 + Jun.23 更新后的稳定性", updated_cells, updated_counts, updated_score, cell_size_m, 5)
    legend_x, legend_y = 70, 690
    draw.text((legend_x, legend_y), "颜色：蓝色为低稳定性，绿色/黄色为中等稳定性，红色为高稳定性。空白表示没有足够的语义投影观测。", fill=(55, 55, 55), font=_font(21))
    canvas.save(output_path)


def _write_evidence_figure(
    output_path: Path,
    role_reports: dict[str, dict[str, Any]],
    consistency: dict[str, dict[str, float | int | None]],
) -> None:
    canvas = Image.new("RGB", (1600, 1060), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.text((70, 35), "Aisle 跨日期语义证据统计", fill=(18, 30, 45), font=_font(38, bold=True))
    draw.text((70, 95), "每个条带表示被投影到地图栅格后的帧级主导语义证据；不是原始像素比例。", fill=(80, 80, 80), font=_font(21))
    roles = ["initial", "same_day_validation", "cross_day_update"]
    labels = {"initial": "Jun.15 初始 Run_1", "same_day_validation": "Jun.15 同日验证 Run_2", "cross_day_update": "Jun.23 跨日期 Run_1"}
    bar_x, bar_y, bar_width, bar_height = 360, 200, 1040, 58
    for role_index, role in enumerate(roles):
        report = role_reports.get(role, {})
        evidence = report.get("cell_group_evidence", {name: 0 for name in GROUP_NAMES})
        total = max(sum(int(evidence.get(name, 0)) for name in GROUP_NAMES), 1)
        y = bar_y + role_index * 155
        draw.text((70, y + 10), labels[role], fill=(35, 35, 35), font=_font(25, bold=True))
        cursor = bar_x
        for group in GROUP_NAMES:
            value = int(evidence.get(group, 0))
            width = int(bar_width * value / total)
            draw.rectangle((cursor, y, cursor + width, y + bar_height), fill=GROUP_COLORS[group])
            cursor += width
        draw.rectangle((bar_x, y, bar_x + bar_width, y + bar_height), outline=(80, 80, 80), width=1)
        draw.text((bar_x, y + 72), f"静态 {evidence.get('static', 0):,}   半动态 {evidence.get('semi_dynamic', 0):,}   动态 {evidence.get('dynamic', 0):,}", fill=(70, 70, 70), font=_font(20))
    legend_y = 680
    for index, group in enumerate(GROUP_NAMES):
        x = 70 + index * 250
        draw.rectangle((x, legend_y, x + 28, legend_y + 28), fill=GROUP_COLORS[group])
        name = {"static": "高可信静态", "semi_dynamic": "中可信半动态", "dynamic": "低可信动态"}[group]
        draw.text((x + 42, legend_y), name, fill=(45, 45, 45), font=_font(22))
    draw.text((70, 770), "重复观测一致性", fill=(18, 30, 45), font=_font(30, bold=True))
    y = 830
    for label, item in (("同日：Jun.15 Run_1 与 Run_2", consistency["same_day"]), ("跨日：Jun.15 Run_1 与 Jun.23 Run_1", consistency["cross_day"])):
        agreement = item["dominant_group_agreement"]
        delta = item["mean_abs_stability_change"]
        agreement_text = "无足够重叠栅格" if agreement is None else f"主导类别一致率 {agreement * 100:.2f}%"
        delta_text = "" if delta is None else f"，平均稳定性变化 {delta:.4f}"
        draw.text((70, y), f"{label}：共享栅格 {item['shared_cells']}，有效栅格 {item['eligible_cells']}，{agreement_text}{delta_text}", fill=(45, 45, 45), font=_font(23))
        y += 65
    canvas.save(output_path)


def _write_cell_csv(
    output_path: Path,
    initial: dict[tuple[int, int], np.ndarray],
    updated: dict[tuple[int, int], np.ndarray],
    cell_size_m: float,
    min_observations: int,
) -> None:
    keys = sorted(set(initial).union(updated))
    with output_path.open("w", newline="", encoding="utf-8-sig") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(
            [
                "cell_x_index", "cell_y_index", "world_x_m", "world_y_m",
                "initial_observed", "initial_static", "initial_semi_dynamic", "initial_dynamic", "initial_stability",
                "updated_observed", "updated_static", "updated_semi_dynamic", "updated_dynamic", "updated_stability",
            ]
        )
        for key in keys:
            initial_counts = initial.get(key, np.zeros(4, dtype=np.int32))
            updated_counts = updated.get(key, np.zeros(4, dtype=np.int32))
            writer.writerow(
                [
                    key[0], key[1], (key[0] + 0.5) * cell_size_m, (key[1] + 0.5) * cell_size_m,
                    *initial_counts.tolist(), float(_stability_score(initial_counts[None], min_observations)[0]),
                    *updated_counts.tolist(), float(_stability_score(updated_counts[None], min_observations)[0]),
                ]
            )


def _process_sequence(
    entry: SequenceEntry,
    crop_config: LocalCropConfig,
    frame_stride: int,
    cell_size_m: float,
    store: dict[tuple[int, int], np.ndarray],
    label_store: dict[tuple[int, int], np.ndarray],
) -> dict[str, Any]:
    dataset = WarehouseSequenceDataset(
        sequence_root=entry.sequence_root,
        calibration_path=entry.calibration_path,
        config={"load_lidar": False},
    )
    label_histogram = np.zeros(16, dtype=np.int64)
    group_point_counts = np.zeros(len(GROUP_NAMES), dtype=np.int64)
    frames_processed = 0
    cropped_points = 0
    labeled_points = 0
    for frame_idx in range(0, len(dataset), max(1, int(frame_stride))):
        record = dataset.frame_index[frame_idx]
        points_sensor = crop_local_lidar_points(load_pcd_xyz(record.lidar_path), crop_config)
        cropped_points += int(points_sensor.shape[0])
        labels = _labels_from_stereo_points(points_sensor, dataset, record)
        valid_labels = (labels >= 0) & (labels < len(LABEL_TO_GROUP))
        if np.any(valid_labels):
            label_histogram += np.bincount(labels[valid_labels], minlength=16).astype(np.int64)
        groups = np.full(labels.shape[0], -1, dtype=np.int8)
        groups[valid_labels] = LABEL_TO_GROUP[labels[valid_labels]]
        for group_index in range(len(GROUP_NAMES)):
            group_point_counts[group_index] += int(np.count_nonzero(groups == group_index))
        labeled_points += int(np.count_nonzero(groups >= 0))
        world_points = transform_points(dataset.ground_truth.poses_4x4[frame_idx], points_sensor)
        cells, dominant_groups = _dominant_cell_groups(world_points, groups, cell_size_m)
        _update_store(store, cells, dominant_groups)
        label_cells, dominant_labels = _dominant_cell_labels(world_points, labels, cell_size_m)
        _update_label_store(label_store, label_cells, dominant_labels)
        frames_processed += 1
        if frames_processed % 50 == 0:
            print(f"[{entry.name}] processed {frames_processed} sampled frames", flush=True)
    return {
        "name": entry.name,
        "role": entry.role,
        "source_frames": int(len(dataset)),
        "sampled_frames": int(frames_processed),
        "cropped_points": int(cropped_points),
        "semantic_labeled_points": int(labeled_points),
        "semantic_label_coverage": float(labeled_points / max(cropped_points, 1)),
        "point_group_counts": {name: int(group_point_counts[index]) for index, name in enumerate(GROUP_NAMES)},
        "label_histogram": {str(index): int(value) for index, value in enumerate(label_histogram)},
    }


def build_semantic_stability_map(config_path: str | Path) -> dict[str, Any]:
    config = _load_config(config_path)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    crop_config = LocalCropConfig(**dict(config.get("crop", {})))
    frame_stride = max(1, int(config.get("frame_stride", 5)))
    cell_size_m = float(config.get("cell_size_m", 0.5))
    min_observations = max(1, int(config.get("min_observations_for_full_confidence", 6)))
    entries = [
        SequenceEntry(
            name=str(item["name"]),
            role=str(item["role"]),
            sequence_root=Path(item["sequence_root"]),
            calibration_path=Path(item.get("calibration_path", config["shared_calibration_path"])),
        )
        for item in config["sequence_entries"]
    ]
    role_stores: dict[str, dict[tuple[int, int], np.ndarray]] = defaultdict(
        lambda: defaultdict(lambda: np.zeros(4, dtype=np.int32))
    )
    role_label_stores: dict[str, dict[tuple[int, int], np.ndarray]] = defaultdict(
        lambda: defaultdict(lambda: np.zeros(len(LABEL_TO_GROUP), dtype=np.int32))
    )
    sequence_reports = []
    for entry in entries:
        print(f"Building semantic evidence from {entry.name}", flush=True)
        sequence_reports.append(
            _process_sequence(
                entry,
                crop_config,
                frame_stride,
                cell_size_m,
                role_stores[entry.role],
                role_label_stores[entry.role],
            )
        )
    role_stores = {role: dict(store) for role, store in role_stores.items()}
    role_label_stores = {role: dict(store) for role, store in role_label_stores.items()}
    initial_store = role_stores.get("initial", {})
    validation_store = role_stores.get("same_day_validation", {})
    update_store = role_stores.get("cross_day_update", {})
    updated_store = _combine_stores(initial_store, update_store)
    initial_label_store = role_label_stores.get("initial", {})
    update_label_store = role_label_stores.get("cross_day_update", {})
    updated_label_store = _combine_stores(initial_label_store, update_label_store)
    initial_cells, initial_counts = _store_arrays(initial_store)
    updated_cells, updated_counts = _store_arrays(updated_store)
    initial_score = _stability_score(initial_counts, min_observations)
    updated_score = _stability_score(updated_counts, min_observations)
    initial_label_cells, initial_label_counts = _store_arrays(initial_label_store)
    updated_label_cells, updated_label_counts = _store_arrays(updated_label_store)
    role_reports: dict[str, dict[str, Any]] = {}
    for role, store in role_stores.items():
        _, counts = _store_arrays(store)
        group_evidence = counts[:, 1:].sum(axis=0) if counts.size else np.zeros(len(GROUP_NAMES), dtype=np.int64)
        role_reports[role] = {
            "populated_cells": int(counts.shape[0]),
            "cell_group_evidence": {name: int(group_evidence[index]) for index, name in enumerate(GROUP_NAMES)},
            "mean_stability": float(_stability_score(counts, min_observations).mean()) if counts.size else None,
        }
    consistency = {
        "same_day": _shared_consistency(initial_store, validation_store, min_observations),
        "cross_day": _shared_consistency(initial_store, update_store, min_observations),
    }
    summary = {
        "config_path": str(Path(config_path)),
        "cell_size_m": cell_size_m,
        "frame_stride": frame_stride,
        "min_observations_for_full_confidence": min_observations,
        "sequence_reports": sequence_reports,
        "role_reports": role_reports,
        "consistency": consistency,
        "initial_map": {
            "populated_cells": int(initial_counts.shape[0]),
            "mean_stability": float(initial_score.mean()) if initial_score.size else None,
        },
        "updated_map": {
            "populated_cells": int(updated_counts.shape[0]),
            "mean_stability": float(updated_score.mean()) if updated_score.size else None,
        },
    }
    np.savez_compressed(
        output_dir / "semantic_stability_map.npz",
        cell_size_m=np.asarray([cell_size_m], dtype=np.float32),
        initial_cells=initial_cells,
        initial_counts=initial_counts,
        initial_stability=initial_score,
        updated_cells=updated_cells,
        updated_counts=updated_counts,
        updated_stability=updated_score,
        initial_label_cells=initial_label_cells,
        initial_label_counts=initial_label_counts,
        updated_label_cells=updated_label_cells,
        updated_label_counts=updated_label_counts,
    )
    _write_cell_csv(output_dir / "semantic_stability_cells.csv", initial_store, updated_store, cell_size_m, min_observations)
    _write_stability_map_figure(
        output_dir / "semantic_stability_initial_vs_updated.png",
        initial_cells, initial_counts, initial_score,
        updated_cells, updated_counts, updated_score,
        cell_size_m,
    )
    _write_evidence_figure(output_dir / "semantic_stability_evidence.png", role_reports, consistency)
    (output_dir / "semantic_stability_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an Aisle semantic stability map from repeated TorWIC observations.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    build_semantic_stability_map(args.config)


if __name__ == "__main__":
    main()
