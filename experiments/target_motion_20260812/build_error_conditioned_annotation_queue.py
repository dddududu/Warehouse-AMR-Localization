# -*- coding: utf-8 -*-
"""Build a manual-review queue stratified by localization failure and control frames.

The offline error is only used to choose informative clips for annotation.  It
is deliberately retained as audit metadata and must never be used as a target
or feature for person detection or motion-classifier training.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


HIGH_ERROR_M = 0.20
CONTROL_MAX_ERROR_M = 0.10
WINDOW_RADIUS = 2
SOURCES = (
    {
        "sequence": "aisle_ccw_run_2",
        "split": "train",
        "report": "sidecar_gain_gate_train_jun15_ccw_full.json",
        "sequence_root": "D:/TorWIC/TorWIC SLAM Dataset/Jun. 15, 2022/Aisle_CCW_Run_2/Aisle_CCW_Run_2",
    },
    {
        "sequence": "aisle_cw_run_2",
        "split": "train",
        "report": "sidecar_gain_gate_train_jun15_cw_full.json",
        "sequence_root": "D:/TorWIC/TorWIC SLAM Dataset/Jun. 15, 2022/Aisle_CW_Run_2/Aisle_CW_Run_2",
    },
    {
        "sequence": "jun23_aisle_ccw_run_2",
        "split": "validation",
        "report": "sidecar_gain_gate_train_jun23_ccw_full.json",
        "sequence_root": "D:/TorWIC/TorWIC SLAM Dataset/Jun. 23, 2022/Aisle_CCW_Run_2/Aisle_CCW_Run_2",
    },
    {
        "sequence": "jun23_aisle_cw_run_2",
        "split": "validation",
        "report": "sidecar_gain_gate_train_jun23_cw_full.json",
        "sequence_root": "D:/TorWIC/TorWIC SLAM Dataset/Jun. 23, 2022/Aisle_CW_Run_2/Aisle_CW_Run_2",
    },
)


@dataclass(frozen=True)
class ErrorSegment:
    sequence: str
    split: str
    start_frame: int
    end_frame: int
    anchor_frame: int
    peak_error_m: float
    anchor_error_m: float
    patch_id: int
    source: dict[str, str]


def _segments(source: dict[str, str], frames: list[dict[str, Any]]) -> list[ErrorSegment]:
    high_frames = [frame for frame in frames if float(frame["position_error_m"]) >= HIGH_ERROR_M]
    grouped: list[list[dict[str, Any]]] = []
    for frame in high_frames:
        if not grouped or int(frame["frame_idx"]) > int(grouped[-1][-1]["frame_idx"]) + 1:
            grouped.append([frame])
        else:
            grouped[-1].append(frame)
    segments: list[ErrorSegment] = []
    for group in grouped:
        peak = max(group, key=lambda frame: float(frame["position_error_m"]))
        segments.append(
            ErrorSegment(
                sequence=source["sequence"],
                split=source["split"],
                start_frame=int(group[0]["frame_idx"]),
                end_frame=int(group[-1]["frame_idx"]),
                anchor_frame=int(peak["frame_idx"]),
                peak_error_m=float(peak["position_error_m"]),
                anchor_error_m=float(peak["position_error_m"]),
                patch_id=int(peak.get("best_patch_id", -1)),
                source=source,
            )
        )
    return segments


def choose_segments(segments: list[ErrorSegment], max_per_split: int) -> list[ErrorSegment]:
    selected: list[ErrorSegment] = []
    for split in ("train", "validation"):
        pool = [segment for segment in segments if segment.split == split]
        per_sequence: dict[str, list[ErrorSegment]] = {}
        for segment in pool:
            per_sequence.setdefault(segment.sequence, []).append(segment)
        for values in per_sequence.values():
            values.sort(key=lambda segment: (-segment.peak_error_m, segment.anchor_frame))
        while len([segment for segment in selected if segment.split == split]) < max_per_split and any(per_sequence.values()):
            progressed = False
            for sequence in sorted(per_sequence):
                if not per_sequence[sequence] or len([segment for segment in selected if segment.split == split]) >= max_per_split:
                    continue
                selected.append(per_sequence[sequence].pop(0))
                progressed = True
            if not progressed:
                break
    return selected


def choose_control(segment: ErrorSegment, frames: list[dict[str, Any]], all_high_frames: set[int]) -> dict[str, Any] | None:
    candidates = [
        frame
        for frame in frames
        if float(frame["position_error_m"]) <= CONTROL_MAX_ERROR_M
        and abs(int(frame["frame_idx"]) - segment.anchor_frame) > 30
        and int(frame["frame_idx"]) not in all_high_frames
    ]
    if not candidates:
        return None
    same_patch = [frame for frame in candidates if int(frame.get("best_patch_id", -1)) == segment.patch_id]
    candidates = same_patch or candidates
    return min(candidates, key=lambda frame: (abs(int(frame["frame_idx"]) - segment.anchor_frame), int(frame["frame_idx"])))


def _row(
    review_order: int,
    condition: str,
    source: dict[str, str],
    frame: dict[str, Any],
    segment: ErrorSegment,
) -> dict[str, Any]:
    anchor = int(frame["frame_idx"])
    return {
        "review_order": review_order,
        "review_condition": condition,
        "split": source["split"],
        "sequence": source["sequence"],
        "anchor_frame": anchor,
        "window_start_frame": max(0, anchor - WINDOW_RADIUS),
        "window_end_frame": anchor + WINDOW_RADIUS,
        "image_left_hint": str(Path(source["sequence_root"]) / "image_left"),
        "matched_error_segment_start": segment.start_frame,
        "matched_error_segment_end": segment.end_frame,
        "offline_position_error_m_for_sampling_only": f"{float(frame['position_error_m']):.6f}",
        "best_patch_id": int(frame.get("best_patch_id", -1)),
        "semantic_filter_applied": bool(frame.get("semantic_filter_applied", False)),
        "semi_dynamic_point_ratio": frame.get("semantic_filter_semi_dynamic_point_ratio"),
        "manual_person_label": "pending",
        "manual_motion_label": "pending",
        "manual_instance_id": "pending",
        "manual_box_json": "pending",
        "annotation_status": "pending_manual_verification",
    }


def run(report_dir: str | Path, output_dir: str | Path, max_segments_per_split: int) -> dict[str, Any]:
    report_dir = Path(report_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_by_sequence: dict[str, list[dict[str, Any]]] = {}
    source_by_sequence = {source["sequence"]: source for source in SOURCES}
    all_segments: list[ErrorSegment] = []
    for source in SOURCES:
        frames = json.loads((report_dir / source["report"]).read_text(encoding="utf-8"))["frame_results"]
        frames_by_sequence[source["sequence"]] = frames
        all_segments.extend(_segments(source, frames))
    selected_segments = choose_segments(all_segments, max_segments_per_split)
    rows: list[dict[str, Any]] = []
    review_order = 1
    for segment in selected_segments:
        source = source_by_sequence[segment.sequence]
        frames = frames_by_sequence[segment.sequence]
        anchor = next(frame for frame in frames if int(frame["frame_idx"]) == segment.anchor_frame)
        rows.append(_row(review_order, "high_error", source, anchor, segment))
        review_order += 1
        high_frames = {item.anchor_frame for item in _segments(source, frames)}
        control = choose_control(segment, frames, high_frames)
        if control is not None:
            rows.append(_row(review_order, "normal_control", source, control, segment))
            review_order += 1
    fields = list(rows[0]) if rows else []
    with (output_dir / "error_conditioned_person_motion_annotation_queue.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "high_error_threshold_m": HIGH_ERROR_M,
        "normal_control_max_error_m": CONTROL_MAX_ERROR_M,
        "high_error_segments_available": len(all_segments),
        "selected_error_segments": len(selected_segments),
        "review_rows": len(rows),
        "rows_by_split": {split: sum(row["split"] == split for row in rows) for split in ("train", "validation")},
        "rows_by_condition": {condition: sum(row["review_condition"] == condition for row in rows) for condition in ("high_error", "normal_control")},
        "guard": "The offline position error is used only for manual-review sampling. It is not a detection, motion, or localization training label.",
    }
    (output_dir / "error_conditioned_queue_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    instructions = """# 定位误差条件化人工核验说明

队列中的 `high_error` 与 `normal_control` 仅表示该片段被离线定位误差选中，用于形成有意义的病例-对照研究设计；它们不是人员标签、动静标签，也不能作为模型输入或监督目标。核验时首先判断窗口中是否真实存在人员；存在时再修正中心帧框、填写跨五帧一致的实例编号，并标为 `moving`、`static` 或 `uncertain`。若没有人员，填写 `non_person`。同时记录是否存在推车、叉车或其他会遮挡主要货架几何的物体。

训练只允许使用人工核验后的人员框与动静标签。定位误差字段必须在构建检测器或动静分类器训练集之前删除；它仅服务于最终的病例-对照统计，例如比较高误差组和正常对照组中真实人员遮挡的比例。
"""
    (output_dir / "annotation_instructions.md").write_text(instructions, encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an error-conditioned manual target review queue.")
    parser.add_argument("--report-dir", default="outputs/semantic_class_prior_20260808")
    parser.add_argument("--output-dir", default="outputs/target_motion_20260812/error_conditioned_annotation_queue")
    parser.add_argument("--max-segments-per-split", type=int, default=10)
    args = parser.parse_args()
    print(json.dumps(run(args.report_dir, args.output_dir, args.max_segments_per_split), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
