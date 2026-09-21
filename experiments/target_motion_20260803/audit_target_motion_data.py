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
from PIL import Image, ImageDraw, ImageFont

from dataset_io.frame_indexer import build_frame_index


@dataclass(frozen=True)
class TargetClass:
    class_id: int
    name: str
    group: str


def _load_config(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("Audit config must be a mapping.")
    return payload


def _load_errors(path: str | Path | None) -> dict[int, float]:
    if not path or not Path(path).is_file():
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload.get("frame_results", [])
    return {
        int(row["frame_idx"]): float(row["position_error_m"])
        for row in rows
        if row.get("frame_idx") is not None and row.get("position_error_m") is not None
    }


def _mask_metrics(mask: np.ndarray | None, class_id: int, min_component_pixels: int) -> tuple[int, int]:
    if mask is None or mask.ndim != 2:
        return 0, 0
    binary = (mask == int(class_id)).astype(np.uint8)
    pixel_count = int(binary.sum())
    if pixel_count == 0:
        return 0, 0
    _, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    component_count = int(np.sum(areas >= int(min_component_pixels)))
    return pixel_count, component_count


def _presence_segments(rows: list[dict[str, Any]], max_gap_frames: int, min_segment_frames: int) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["present"]:
            by_key[(str(row["sequence"]), str(row["target_name"]))].append(row)
    for (sequence, target_name), present_rows in by_key.items():
        active: list[dict[str, Any]] = []
        previous_frame: int | None = None
        for row in present_rows:
            frame_idx = int(row["frame_idx"])
            if previous_frame is not None and frame_idx - previous_frame > int(max_gap_frames) + 1:
                _append_segment(segments, active, sequence, target_name, min_segment_frames)
                active = []
            active.append(row)
            previous_frame = frame_idx
        _append_segment(segments, active, sequence, target_name, min_segment_frames)
    return segments


def _append_segment(
    segments: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    sequence: str,
    target_name: str,
    min_segment_frames: int,
) -> None:
    if len(rows) < int(min_segment_frames):
        return
    errors = [float(row["position_error_m"]) for row in rows if row["position_error_m"] is not None]
    segments.append(
        {
            "sequence": sequence,
            "target_name": target_name,
            "start_frame": int(rows[0]["frame_idx"]),
            "end_frame": int(rows[-1]["frame_idx"]),
            "observed_frames": int(len(rows)),
            "span_frames": int(rows[-1]["frame_idx"] - rows[0]["frame_idx"] + 1),
            "mean_pixel_coverage": float(np.mean([row["pixel_coverage"] for row in rows])),
            "max_pixel_coverage": float(np.max([row["pixel_coverage"] for row in rows])),
            "max_components": int(np.max([row["component_count"] for row in rows])),
            "mean_position_error_m": float(np.mean(errors)) if errors else None,
            "max_position_error_m": float(np.max(errors)) if errors else None,
        }
    )


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def _draw_summary(path: Path, target_summary: list[dict[str, Any]], total_frames: int) -> None:
    canvas = Image.new("RGB", (1600, 980), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = _font(38, True)
    body_font = _font(22)
    small_font = _font(19)
    draw.text((65, 45), "Aisle 目标语义时序审计：目标样本覆盖", fill="#172B4D", font=title_font)
    draw.text((65, 105), f"共审计 {total_frames:,} 帧；统计基于左目语义掩码的连通区域，非人工实例标注。", fill="#52606D", font=body_font)
    chart_left, chart_top, chart_width, chart_height = 390, 190, 1080, 390
    maximum = max((item["present_frames"] for item in target_summary), default=1)
    row_height = max(chart_height // max(len(target_summary), 1), 70)
    for index, item in enumerate(target_summary):
        y = chart_top + index * row_height
        draw.text((65, y + 10), item["target_name"], fill="#243B53", font=body_font)
        width = int(chart_width * item["present_frames"] / maximum)
        color = "#D64545" if item["group"] == "dynamic" else "#DE8F32"
        draw.rectangle((chart_left, y, chart_left + width, y + 38), fill=color)
        draw.text((chart_left + width + 18, y + 6), f"{item['present_frames']:,} 帧（{item['present_ratio'] * 100:.2f}%）", fill="#334E68", font=small_font)
    draw.text((65, 660), "跨日期可用性", fill="#172B4D", font=_font(29, True))
    draw.text((65, 710), "只有同时出现在训练、验证和十月盲测中的类别，才适合作为跨日期动静分类研究对象。", fill="#52606D", font=body_font)
    y = 765
    for item in target_summary:
        splits = " / ".join(item["positive_splits"]) if item["positive_splits"] else "无"
        status = "可进入跨日期研究" if item["cross_date_available"] else "仅能做探索性分析"
        draw.text((85, y), f"{item['target_name']}：{status}；出现数据划分：{splits}", fill="#334E68", font=body_font)
        y += 46
    canvas.save(path)


def _draw_error_effect(path: Path, error_summary: list[dict[str, Any]]) -> None:
    canvas = Image.new("RGB", (1600, 900), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = _font(38, True)
    body_font = _font(22)
    small_font = _font(19)
    draw.text((65, 45), "十月盲测：目标出现与定位误差的关联", fill="#172B4D", font=title_font)
    draw.text((65, 105), "仅描述同帧关联，不将相关性解释为因果关系。", fill="#52606D", font=body_font)
    labels = [item["target_name"] for item in error_summary]
    values = [item["mean_error_present_m"] for item in error_summary]
    baseline = [item["mean_error_absent_m"] for item in error_summary]
    maximum = max(values + baseline + [0.01])
    chart_left, chart_top, chart_width, chart_height = 390, 190, 760, 420
    row_height = max(chart_height // max(len(error_summary), 1), 70)
    for index, item in enumerate(error_summary):
        y = chart_top + index * row_height
        draw.text((65, y + 8), labels[index], fill="#243B53", font=body_font)
        present_width = int(chart_width * values[index] / maximum)
        absent_width = int(chart_width * baseline[index] / maximum)
        draw.rectangle((chart_left, y, chart_left + present_width, y + 23), fill="#D64545")
        draw.rectangle((chart_left, y + 30, chart_left + absent_width, y + 53), fill="#5B8FF9")
        draw.text((chart_left + max(present_width, absent_width) + 14, y + 6), f"出现 {values[index]:.4f} m；未出现 {baseline[index]:.4f} m", fill="#334E68", font=small_font)
    draw.rectangle((65, 700, 90, 725), fill="#D64545")
    draw.text((102, 699), "目标出现帧平均误差", fill="#52606D", font=body_font)
    draw.rectangle((385, 700, 410, 725), fill="#5B8FF9")
    draw.text((422, 699), "目标未出现帧平均误差", fill="#52606D", font=body_font)
    draw.text((65, 775), "后续训练仅使用能在十月盲测观察到、且存在连续片段的类别；单帧噪声不作为运动学习样本。", fill="#52606D", font=body_font)
    canvas.save(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(config_path: str | Path) -> dict[str, Any]:
    config = _load_config(config_path)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis = dict(config.get("analysis", {}))
    target_classes = [TargetClass(**item) for item in config["target_classes"]]
    frame_rows: list[dict[str, Any]] = []
    sequence_summary: list[dict[str, Any]] = []
    for sequence in config["sequences"]:
        root = Path(sequence["sequence_root"])
        records = build_frame_index(root)
        errors = _load_errors(sequence.get("localization_result_json"))
        present_counts = defaultdict(int)
        for frame_idx, record in enumerate(records):
            mask = cv2.imread(str(record.segmentation_greyscale_left_path), cv2.IMREAD_UNCHANGED)
            if mask is None or mask.ndim != 2:
                raise RuntimeError(f"Could not read greyscale semantic mask for {sequence['name']} frame {frame_idx}.")
            pixels_total = int(mask.size)
            for target in target_classes:
                pixel_count, component_count = _mask_metrics(mask, target.class_id, int(analysis["min_component_pixels"]))
                coverage = pixel_count / max(pixels_total, 1)
                present = bool(component_count > 0 and coverage >= float(analysis["min_pixel_coverage"]))
                present_counts[target.name] += int(present)
                frame_rows.append(
                    {
                        "sequence": sequence["name"],
                        "split": sequence["split"],
                        "frame_idx": frame_idx,
                        "target_id": target.class_id,
                        "target_name": target.name,
                        "target_group": target.group,
                        "pixel_count": pixel_count,
                        "pixel_coverage": coverage,
                        "component_count": component_count,
                        "present": present,
                        "position_error_m": errors.get(frame_idx),
                    }
                )
        sequence_summary.append(
            {
                "sequence": sequence["name"],
                "split": sequence["split"],
                "frames": len(records),
                "has_localization_errors": bool(errors),
                "present_frames": dict(present_counts),
            }
        )
        print(f"audited {sequence['name']}: {len(records)} frames")
    segments = _presence_segments(frame_rows, int(analysis["max_gap_frames"]), int(analysis["min_segment_frames"]))
    target_summary: list[dict[str, Any]] = []
    error_summary: list[dict[str, Any]] = []
    all_frames = len({(row["sequence"], row["frame_idx"]) for row in frame_rows})
    for target in target_classes:
        rows = [row for row in frame_rows if row["target_name"] == target.name]
        present_rows = [row for row in rows if row["present"]]
        positive_splits = sorted({str(row["split"]) for row in present_rows})
        target_summary.append(
            {
                "target_id": target.class_id,
                "target_name": target.name,
                "group": target.group,
                "present_frames": len(present_rows),
                "present_ratio": len(present_rows) / max(all_frames, 1),
                "continuous_segments": sum(1 for segment in segments if segment["target_name"] == target.name),
                "positive_splits": positive_splits,
                "cross_date_available": bool({"train", "validation", "blind_test"}.issubset(positive_splits)),
            }
        )
        error_rows = [row for row in rows if row["position_error_m"] is not None]
        present_errors = [float(row["position_error_m"]) for row in error_rows if row["present"]]
        absent_errors = [float(row["position_error_m"]) for row in error_rows if not row["present"]]
        if present_errors and absent_errors:
            error_summary.append(
                {
                    "target_name": target.name,
                    "frames_with_error": len(error_rows),
                    "present_error_frames": len(present_errors),
                    "mean_error_present_m": float(np.mean(present_errors)),
                    "mean_error_absent_m": float(np.mean(absent_errors)),
                    "p95_error_present_m": float(np.quantile(present_errors, 0.95)),
                    "p95_error_absent_m": float(np.quantile(absent_errors, 0.95)),
                }
            )
    error_thresholds = {
        sequence: float(np.quantile([float(row["position_error_m"]) for row in frame_rows if row["sequence"] == sequence and row["position_error_m"] is not None], float(analysis["hard_error_quantile"])))
        for sequence in {row["sequence"] for row in frame_rows}
        if any(row["sequence"] == sequence and row["position_error_m"] is not None for row in frame_rows)
    }
    hard_frames = [
        row for row in frame_rows
        if row["position_error_m"] is not None and row["present"] and float(row["position_error_m"]) >= error_thresholds[row["sequence"]]
    ]
    _write_csv(output_dir / "frame_target_observations.csv", frame_rows)
    _write_csv(output_dir / "target_segments.csv", segments)
    _write_csv(output_dir / "hard_target_frames.csv", hard_frames)
    _write_csv(output_dir / "target_error_association.csv", error_summary)
    _draw_summary(output_dir / "target_sample_coverage.png", target_summary, all_frames)
    if error_summary:
        _draw_error_effect(output_dir / "target_error_association.png", error_summary)
    summary = {
        "analysis": analysis,
        "total_audited_frames": all_frames,
        "sequences": sequence_summary,
        "targets": target_summary,
        "error_association": error_summary,
        "hard_error_thresholds_m": error_thresholds,
        "hard_target_frame_rows": len(hard_frames),
        "limitations": [
            "语义掩码提供类别像素，不提供人工实例、真实速度或真实动静标签。",
            "连续帧具有强相关性，样本数量不能视为独立对象样本数量。",
            "目标出现与定位误差仅做同帧关联，不构成因果结论。",
        ],
    }
    (output_dir / "target_motion_audit_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit target-class coverage and temporal segments in Aisle semantic masks.")
    parser.add_argument("--config", default="experiments/target_motion_20260803/target_motion_audit.yaml")
    args = parser.parse_args()
    summary = run(args.config)
    print(json.dumps({"total_audited_frames": summary["total_audited_frames"], "targets": summary["targets"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
