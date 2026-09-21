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

from dataset_io.gt_loader import load_ground_truth


@dataclass(frozen=True)
class TestSequence:
    name: str
    sequence_root: Path
    result_json: Path


def _load_config(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("Localization-analysis config must be a mapping.")
    return payload


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def _percentile(values: np.ndarray, percentile: float) -> float | None:
    return float(np.percentile(values, percentile)) if values.size else None


def _summarize_errors(errors: np.ndarray) -> dict[str, float | int | None]:
    return {
        "frames": int(errors.size),
        "mean_position_error_m": float(errors.mean()) if errors.size else None,
        "median_position_error_m": float(np.median(errors)) if errors.size else None,
        "p95_position_error_m": _percentile(errors, 95.0),
        "position_error_below_0p5m": float(np.mean(errors < 0.5)) if errors.size else None,
        "position_error_above_0p5m": int(np.count_nonzero(errors >= 0.5)),
    }


def _local_stability(
    query_xy: np.ndarray,
    cell_xy: np.ndarray,
    cell_stability: np.ndarray,
    cell_observations: np.ndarray,
    radius_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.full(query_xy.shape[0], np.nan, dtype=np.float32)
    coverage = np.zeros(query_xy.shape[0], dtype=np.int32)
    radius_sq = float(radius_m) ** 2
    for index, position in enumerate(query_xy):
        distances_sq = np.square(cell_xy - position[None, :]).sum(axis=1)
        mask = distances_sq <= radius_sq
        coverage[index] = int(np.count_nonzero(mask))
        if np.any(mask):
            weights = np.maximum(cell_observations[mask], 1)
            values[index] = float(np.average(cell_stability[mask], weights=weights))
    return values, coverage


def _draw_scatter(output_path: Path, rows: list[dict[str, Any]], correlation: float | None, radius_m: float) -> None:
    canvas = Image.new("RGB", (1600, 1140), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.text((70, 35), "历史语义稳定性与 Oct.12 定位误差的诊断关系", fill=(18, 30, 45), font=_font(38, True))
    draw.text((70, 95), f"每个点为一个测试帧；稳定性由半径 {radius_m:.1f} m 内的 Jun.15 + Jun.23 地图栅格计算，仅用于事后分析。", fill=(80, 80, 80), font=_font(21))
    left, top, right, bottom = 150, 170, 1490, 830
    draw.line((left, bottom, right, bottom), fill=(60, 60, 60), width=2)
    draw.line((left, top, left, bottom), fill=(60, 60, 60), width=2)
    valid_errors = np.asarray([row["position_error_m"] for row in rows if np.isfinite(row["local_stability"])], dtype=np.float64)
    valid_stability = np.asarray([row["local_stability"] for row in rows if np.isfinite(row["local_stability"])], dtype=np.float64)
    max_error = max(1.0, float(np.percentile(valid_errors, 99.5)) * 1.12) if valid_errors.size else 1.0
    stability_min = max(0.0, float(valid_stability.min()) - 0.01) if valid_stability.size else 0.0
    stability_max = min(1.0, float(valid_stability.max()) + 0.01) if valid_stability.size else 1.0
    if stability_max - stability_min < 0.02:
        stability_min = max(0.0, stability_min - 0.01)
        stability_max = min(1.0, stability_max + 0.01)
    for tick in np.linspace(stability_min, stability_max, 6):
        x = left + int((right - left) * (tick - stability_min) / (stability_max - stability_min))
        draw.line((x, bottom, x, bottom + 8), fill=(60, 60, 60), width=1)
        draw.text((x - 20, bottom + 16), f"{tick:.3f}", fill=(70, 70, 70), font=_font(19))
    for tick in np.linspace(0.0, max_error, 6):
        y = bottom - int((bottom - top) * tick / max_error)
        draw.line((left - 8, y, left, y), fill=(60, 60, 60), width=1)
        draw.text((58, y - 12), f"{tick:.2f}", fill=(70, 70, 70), font=_font(19))
    threshold_y = bottom - int((bottom - top) * 0.5 / max_error)
    draw.line((left, threshold_y, right, threshold_y), fill=(180, 100, 100), width=2)
    draw.text((right - 170, threshold_y - 32), "0.5 m 误差线", fill=(145, 70, 70), font=_font(19))
    colors = {"Aisle_CCW": (56, 111, 190), "Aisle_CW": (209, 123, 48)}
    for row in rows:
        stability = float(row["local_stability"])
        error = float(row["position_error_m"])
        if not np.isfinite(stability):
            continue
        x = left + int((right - left) * np.clip((stability - stability_min) / (stability_max - stability_min), 0.0, 1.0))
        y = bottom - int((bottom - top) * np.clip(error, 0.0, max_error) / max_error)
        color = colors[str(row["sequence_name"])]
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)
    draw.text((left, bottom + 70), "局部历史语义稳定性", fill=(35, 35, 35), font=_font(24, True))
    draw.text((25, top - 20), "位置误差 (m)", fill=(35, 35, 35), font=_font(24, True))
    legend_y = 1000
    for index, (name, color) in enumerate(colors.items()):
        x = 80 + index * 220
        draw.ellipse((x, legend_y, x + 20, legend_y + 20), fill=color)
        draw.text((x + 32, legend_y - 3), name, fill=(45, 45, 45), font=_font(21))
    correlation_text = "无足够有效帧" if correlation is None else f"Pearson r = {correlation:.4f}"
    draw.text((980, 998), f"稳定性与误差：{correlation_text}", fill=(45, 45, 45), font=_font(22, True))
    canvas.save(output_path)


def analyze_stability_localization(config_path: str | Path) -> dict[str, Any]:
    config = _load_config(config_path)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    stability_payload = np.load(Path(config["stability_map_path"]))
    cell_size_m = float(stability_payload["cell_size_m"][0])
    cells = stability_payload["updated_cells"].astype(np.float64)
    cell_xy = (cells + 0.5) * cell_size_m
    cell_counts = stability_payload["updated_counts"]
    cell_stability = stability_payload["updated_stability"].astype(np.float64)
    radius_m = float(config.get("analysis_radius_m", 7.5))
    sequences = [
        TestSequence(str(item["name"]), Path(item["sequence_root"]), Path(item["result_json"]))
        for item in config["sequences"]
    ]
    rows: list[dict[str, Any]] = []
    per_sequence: dict[str, dict[str, Any]] = {}
    for sequence in sequences:
        report = json.loads(sequence.result_json.read_text(encoding="utf-8"))
        trajectory = load_ground_truth(sequence.sequence_root / "traj_gt.txt")
        frame_indices = np.asarray([int(item["frame_idx"]) for item in report["frame_results"]], dtype=np.int64)
        errors = np.asarray([float(item["position_error_m"]) for item in report["frame_results"]], dtype=np.float64)
        stability, coverage = _local_stability(
            trajectory.positions[frame_indices, :2], cell_xy, cell_stability, cell_counts[:, 0], radius_m
        )
        valid = np.isfinite(stability)
        correlation = float(np.corrcoef(stability[valid], errors[valid])[0, 1]) if np.count_nonzero(valid) >= 2 else None
        quartiles = np.quantile(stability[valid], [0.25, 0.75]) if np.any(valid) else np.asarray([np.nan, np.nan])
        low_errors = errors[valid & (stability <= quartiles[0])]
        high_errors = errors[valid & (stability >= quartiles[1])]
        per_sequence[sequence.name] = {
            "all_frames": _summarize_errors(errors),
            "valid_stability_frames": int(np.count_nonzero(valid)),
            "mean_local_stability": float(np.nanmean(stability)),
            "mean_local_coverage_cells": float(coverage[valid].mean()) if np.any(valid) else None,
            "pearson_stability_error": correlation,
            "low_stability_quartile": _summarize_errors(low_errors),
            "high_stability_quartile": _summarize_errors(high_errors),
        }
        for frame_idx, error, value, count in zip(frame_indices, errors, stability, coverage):
            rows.append(
                {
                    "sequence_name": sequence.name,
                    "frame_idx": int(frame_idx),
                    "position_error_m": float(error),
                    "local_stability": float(value),
                    "local_coverage_cells": int(count),
                }
            )
    values = np.asarray([row["local_stability"] for row in rows], dtype=np.float64)
    errors = np.asarray([row["position_error_m"] for row in rows], dtype=np.float64)
    valid = np.isfinite(values)
    overall_correlation = float(np.corrcoef(values[valid], errors[valid])[0, 1]) if np.count_nonzero(valid) >= 2 else None
    summary = {
        "analysis_radius_m": radius_m,
        "stability_map_cells": int(cells.shape[0]),
        "overall_pearson_stability_error": overall_correlation,
        "per_sequence": per_sequence,
        "note": "This is a post-hoc diagnostic using Oct.12 ground-truth positions. It is not an input to any deployed localizer or a source of Oct.12 tuning.",
    }
    with (output_dir / "stability_localization_correlation.csv").open("w", newline="", encoding="utf-8-sig") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(rows[0]) if rows else ["sequence_name", "frame_idx", "position_error_m", "local_stability", "local_coverage_cells"])
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "stability_localization_correlation.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _draw_scatter(output_dir / "stability_error_scatter.png", rows, overall_correlation, radius_m)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose the relationship between historical semantic stability and localization error.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    analyze_stability_localization(args.config)


if __name__ == "__main__":
    main()
