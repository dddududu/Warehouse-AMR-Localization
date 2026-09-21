from __future__ import annotations

import argparse
import json
from collections import defaultdict
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

from experiments.semantic_stability_20260731.build_semantic_stability_map import (
    LABEL_TO_GROUP,
    _dominant_cell_labels,
    _labels_from_stereo_points,
)


@dataclass(frozen=True)
class SequenceConfig:
    name: str
    sequence_root: Path
    calibration_path: Path


def _load_yaml(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("Semantic update-safety config must be a mapping.")
    return payload


def _sequence_configs(raw: list[dict[str, Any]]) -> list[SequenceConfig]:
    return [
        SequenceConfig(
            name=str(item["name"]),
            sequence_root=Path(item["sequence_root"]),
            calibration_path=Path(item["calibration_path"]),
        )
        for item in raw
    ]


def _cell_key(cell: np.ndarray) -> tuple[int, int]:
    return int(cell[0]), int(cell[1])


def _collect_cell_observations(
    sequences: list[SequenceConfig],
    crop: LocalCropConfig,
    cell_size_m: float,
    frame_stride: int,
) -> dict[tuple[int, int], np.ndarray]:
    store: dict[tuple[int, int], np.ndarray] = defaultdict(lambda: np.zeros(17, dtype=np.int32))
    for sequence in sequences:
        dataset = WarehouseSequenceDataset(
            sequence_root=sequence.sequence_root,
            calibration_path=sequence.calibration_path,
            config={"load_lidar": False},
        )
        for frame_idx in range(0, len(dataset), max(1, int(frame_stride))):
            record = dataset.frame_index[frame_idx]
            points = crop_local_lidar_points(load_pcd_xyz(record.lidar_path), crop)
            labels = _labels_from_stereo_points(points, dataset, record)
            world_points = transform_points(dataset.ground_truth.poses_4x4[frame_idx], points)
            cells, dominant_labels = _dominant_cell_labels(world_points, labels, cell_size_m)
            for cell, label in zip(cells, dominant_labels, strict=True):
                value = store[_cell_key(cell)]
                value[0] += 1
                value[int(label) + 1] += 1
    return dict(store)


def _map_lookup(stability_map_path: Path) -> dict[tuple[int, int], dict[str, float | int]]:
    payload = np.load(stability_map_path)
    evidence_cells = payload["updated_cells"]
    evidence_counts = payload["updated_counts"]
    stability = payload["updated_stability"]
    label_cells = payload["updated_label_cells"]
    label_counts = payload["updated_label_counts"]
    labels = {
        _cell_key(cell): int(np.asarray(count).argmax())
        for cell, count in zip(label_cells, label_counts, strict=True)
    }
    return {
        _cell_key(cell): {
            "evidence": int(count[0]),
            "dominant_group": int(np.asarray(count[1:]).argmax()),
            "stability": float(score),
            "dominant_label": labels.get(_cell_key(cell), 0),
        }
        for cell, count, score in zip(evidence_cells, evidence_counts, stability, strict=True)
    }


def _policy_accepts(
    policy: str,
    current: np.ndarray,
    map_info: dict[str, float | int] | None,
    min_support: int,
    min_stability: float,
    min_evidence: int,
) -> bool:
    current_label = int(np.asarray(current[1:]).argmax())
    current_group = int(LABEL_TO_GROUP[current_label])
    support = int(current[0])
    if policy == "naive":
        return True
    if current_group != 0:
        return False
    if policy == "semantic_static":
        return True
    if map_info is None:
        return False
    if (
        int(map_info["dominant_group"]) != 0
        or float(map_info["stability"]) < min_stability
        or int(map_info["evidence"]) < min_evidence
    ):
        return False
    if policy == "stable_static":
        return True
    return bool(support >= min_support and int(map_info["dominant_label"]) == current_label)


def _future_confirmed(current: np.ndarray, future: np.ndarray | None) -> bool | None:
    if future is None or int(future[0]) == 0:
        return None
    current_label = int(np.asarray(current[1:]).argmax())
    future_label = int(np.asarray(future[1:]).argmax())
    return bool(int(LABEL_TO_GROUP[current_label]) == 0 and future_label == current_label)


def _evaluate_policies(
    current_store: dict[tuple[int, int], np.ndarray],
    future_store: dict[tuple[int, int], np.ndarray],
    map_store: dict[tuple[int, int], dict[str, float | int]],
    config: dict[str, Any],
) -> dict[str, Any]:
    policies = ["naive", "semantic_static", "stable_static", "persistent_label"]
    outcome: dict[str, Any] = {}
    min_support = int(config.get("min_current_frame_support", 3))
    min_stability = float(config.get("min_stability", 0.70))
    min_evidence = int(config.get("min_map_evidence", 3))
    for policy in policies:
        accepted = [
            (cell, value)
            for cell, value in current_store.items()
            if _policy_accepts(policy, value, map_store.get(cell), min_support, min_stability, min_evidence)
        ]
        confirmations = [_future_confirmed(value, future_store.get(cell)) for cell, value in accepted]
        observed = [value for value in confirmations if value is not None]
        outcome[policy] = {
            "accepted_cells": int(len(accepted)),
            "coverage_over_current_cells": float(len(accepted) / max(1, len(current_store))),
            "future_observed_cells": int(len(observed)),
            "future_confirmed_safe_cells": int(sum(value is True for value in observed)),
            "future_confirmation_rate": float(sum(value is True for value in observed) / max(1, len(observed))),
            "future_contradiction_rate": float(sum(value is False for value in observed) / max(1, len(observed))),
        }
    return outcome


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def _draw_summary(output_path: Path, policies: dict[str, Any]) -> None:
    labels = {
        "naive": "无筛选",
        "semantic_static": "仅当前静态",
        "stable_static": "静态+历史稳定",
        "persistent_label": "稳定+跨帧同类",
    }
    colors = {"naive": "#dc2626", "semantic_static": "#f59e0b", "stable_static": "#2563eb", "persistent_label": "#16a34a"}
    canvas = Image.new("RGB", (1340, 790), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = _font(30, True)
    body_font = _font(21)
    small_font = _font(17)
    draw.text((48, 30), "实验五：语义辅助地图更新的安全性验证", font=title_font, fill="#1d2939")
    draw.text((48, 79), "决策只读取 Jun.15+Jun.23 历史层与 Jun.23 Run2 当前观测；Oct.12 仅作未来确认", font=body_font, fill="#475467")
    panels = [(65, 150, "可接受更新覆盖率", "coverage_over_current_cells"), (715, 150, "未来语义确认率", "future_confirmation_rate")]
    order = list(labels)
    for left, top, title, metric in panels:
        width, height = 560, 365
        draw.rounded_rectangle((left, top, left + width, top + height), radius=16, outline="#cbd5e1", width=2)
        draw.text((left + 24, top + 20), title, font=body_font, fill="#1d2939")
        chart_left, chart_top, chart_width, chart_height = left + 48, top + 76, width - 78, 205
        draw.line((chart_left, chart_top, chart_left, chart_top + chart_height), fill="#98a2b3", width=2)
        draw.line((chart_left, chart_top + chart_height, chart_left + chart_width, chart_top + chart_height), fill="#98a2b3", width=2)
        for level in (0.0, 0.5, 1.0):
            y = chart_top + chart_height - chart_height * level
            draw.line((chart_left, y, chart_left + chart_width, y), fill="#e2e8f0", width=1)
            draw.text((chart_left - 42, y - 9), f"{level:.1f}", font=small_font, fill="#667085")
        for index, policy in enumerate(order):
            value = float(policies[policy][metric])
            x = chart_left + 38 + index * 111
            bar_height = chart_height * value
            draw.rounded_rectangle((x, chart_top + chart_height - bar_height, x + 61, chart_top + chart_height), radius=5, fill=colors[policy])
            draw.text((x - 2, chart_top + chart_height - bar_height - 26), f"{value * 100:.1f}%", font=small_font, fill="#344054")
            draw.text((x - 20, chart_top + chart_height + 12), labels[policy], font=small_font, fill="#344054")
    draw.rounded_rectangle((65, 575, 1275, 720), radius=16, fill="#f8fafc", outline="#cbd5e1")
    strict = policies["persistent_label"]
    draw.text((92, 600), "最严格策略：仅在当前为静态、历史单元稳定、细类别一致且跨帧重复观察时写入。", font=body_font, fill="#344054")
    draw.text((92, 642), f"接受 {strict['accepted_cells']} 个单元，其中未来可观测 {strict['future_observed_cells']} 个，Oct.12 细类别确认率 {strict['future_confirmation_rate'] * 100:.2f}%。", font=body_font, fill="#344054")
    canvas.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate semantic safety gates for long-term map updates.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = _load_yaml(args.config)
    crop_values = config["crop"]
    crop = LocalCropConfig(
        x_min=float(crop_values["x_min"]),
        x_max=float(crop_values["x_max"]),
        y_min=float(crop_values["y_min"]),
        y_max=float(crop_values["y_max"]),
        z_min=float(crop_values["z_min"]),
        z_max=float(crop_values["z_max"]),
    )
    current_store = _collect_cell_observations(
        _sequence_configs(config["current_sequences"]),
        crop,
        float(config.get("cell_size_m", 0.5)),
        int(config.get("frame_stride", 8)),
    )
    future_store = _collect_cell_observations(
        _sequence_configs(config["future_sequences"]),
        crop,
        float(config.get("cell_size_m", 0.5)),
        int(config.get("frame_stride", 8)),
    )
    policies = _evaluate_policies(
        current_store,
        future_store,
        _map_lookup(Path(config["stability_map_path"])),
        config,
    )
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    _draw_summary(output_dir / "semantic_update_safety.png", policies)
    summary = {
        "num_current_cells": int(len(current_store)),
        "num_future_cells": int(len(future_store)),
        "policies": policies,
        "decision_data": "Jun.15+Jun.23 history and Jun.23 Run2 only",
        "future_confirmation_data": "Oct.12 Aisle only",
    }
    (output_dir / "semantic_update_safety_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
