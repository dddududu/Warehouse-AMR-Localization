# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


MOTION_PRIORITY = {
    "potentially_moving": 4,
    "ambiguous_motion": 3,
    "likely_static_relative_to_map": 2,
    "insufficient_3d_evidence": 1,
}


@dataclass(frozen=True)
class QueueItem:
    row: dict[str, str]
    evidence_score: float
    uncertainty_score: float
    priority_score: float
    tier: str
    reason: str


def _json_floats(value: str) -> list[float]:
    try:
        values = json.loads(value)
    except json.JSONDecodeError:
        return []
    return [float(item) for item in values if item is not None]


def score_row(row: dict[str, str]) -> QueueItem:
    observation_count = int(row["track_observation_count"] or 0)
    scores = _json_floats(row["track_scores_json"])
    overlaps = _json_floats(row["track_link_iou_json"])
    non_anchor_overlaps = [value for value in overlaps if value > 0.0]
    mean_score = float(np.mean(scores)) if scores else 0.0
    mean_overlap = float(np.mean(non_anchor_overlaps)) if non_anchor_overlaps else 0.0
    evidence_score = 0.45 * min(observation_count / 5.0, 1.0) + 0.35 * mean_score + 0.20 * mean_overlap
    uncertainty_score = 0.65 * (1.0 - mean_score) + 0.35 * (1.0 - mean_overlap)
    hint = row["motion_hint"]
    motion_score = MOTION_PRIORITY.get(hint, 0) / max(MOTION_PRIORITY.values())

    if hint in {"potentially_moving", "ambiguous_motion"} and observation_count >= 3:
        tier = "A"
        reason = "具有连续三维证据的动静边界样本，优先确认真实人员与运动状态。"
    elif observation_count >= 3:
        tier = "B"
        reason = "具有连续三维证据，可用于补充静态/不确定对象的对照样本。"
    else:
        tier = "C"
        reason = "三维证据不足，但检测时序稳定性需要人工确认，主要服务目标识别。"
    priority_score = 0.50 * motion_score + 0.35 * evidence_score + 0.15 * uncertainty_score
    return QueueItem(
        row=row,
        evidence_score=float(evidence_score),
        uncertainty_score=float(uncertainty_score),
        priority_score=float(priority_score),
        tier=tier,
        reason=reason,
    )


def select_balanced(items: list[QueueItem], max_items: int) -> list[QueueItem]:
    ordered = sorted(
        items,
        key=lambda item: (
            item.tier,
            -item.priority_score,
            item.row["sequence"],
            int(item.row["anchor_frame"]),
        ),
    )
    selected: list[QueueItem] = []
    split_counts: Counter[str] = Counter()
    sequence_counts: Counter[str] = Counter()
    tier_counts: Counter[str] = Counter()
    deferred: list[QueueItem] = []
    target_validation = max(4, round(max_items / 3))
    target_per_sequence = max(2, int(np.ceil(max_items / 8)))
    target_per_tier = {"A": max(6, round(max_items * 0.45)), "B": max(4, round(max_items * 0.30))}
    for item in ordered:
        split = item.row["split"]
        sequence = item.row["sequence"]
        tier_limit = target_per_tier.get(item.tier, max_items)
        if (
            sequence_counts[sequence] >= target_per_sequence
            or tier_counts[item.tier] >= tier_limit
            or (split == "validation" and split_counts[split] >= target_validation)
        ):
            deferred.append(item)
            continue
        selected.append(item)
        split_counts[split] += 1
        sequence_counts[sequence] += 1
        tier_counts[item.tier] += 1
        if len(selected) >= max_items:
            return selected
    for item in deferred:
        if len(selected) >= max_items:
            break
        if item not in selected:
            selected.append(item)
    return selected


def run(manifest_path: str | Path, output_dir: str | Path, max_items: int) -> dict[str, Any]:
    rows = list(csv.DictReader(Path(manifest_path).open(encoding="utf-8-sig")))
    for source_index, row in enumerate(rows):
        row["source_review_packet"] = f"review_packet_{(source_index // 4) + 1:02d}.png"
    pending_rows = [row for row in rows if row.get("annotation_status") == "pending_manual_verification"]
    items = [score_row(row) for row in pending_rows]
    selected = select_balanced(items, max_items=max_items)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    fields = [
        "review_order", "annotation_tier", "selection_reason", "priority_score", "evidence_score",
        "uncertainty_score", "split", "sequence", "anchor_frame", "window_start_frame", "window_end_frame",
        "motion_hint", "track_observation_count", "world_path_length_m", "net_world_displacement_m",
        "manual_person_label", "manual_motion_label", "manual_instance_id", "manual_box_json", "annotation_status",
        "review_packet", "window_image_paths_json", "track_boxes_json", "track_scores_json", "track_link_iou_json",
    ]
    output_rows = []
    for order, item in enumerate(selected, start=1):
        row = item.row
        output_rows.append({
            "review_order": order,
            "annotation_tier": item.tier,
            "selection_reason": item.reason,
            "priority_score": f"{item.priority_score:.6f}",
            "evidence_score": f"{item.evidence_score:.6f}",
            "uncertainty_score": f"{item.uncertainty_score:.6f}",
            **{key: row.get(key, "") for key in fields if key not in {
                "review_order", "annotation_tier", "selection_reason", "priority_score", "evidence_score", "uncertainty_score", "review_packet"
            }},
            "review_packet": row["source_review_packet"],
        })
    queue_path = root / "active_person_annotation_queue.csv"
    with queue_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)
    summary = {
        "source_manifest": str(manifest_path),
        "pending_candidates": len(pending_rows),
        "selected_candidates": len(selected),
        "selection_policy": "先保证训练/验证、四条路线与三类证据层级覆盖，再按动静价值、三维证据和检测不确定性排序。",
        "selected_by_split": dict(Counter(item.row["split"] for item in selected)),
        "selected_by_sequence": dict(Counter(item.row["sequence"] for item in selected)),
        "selected_by_tier": dict(Counter(item.tier for item in selected)),
        "selected_by_motion_hint": dict(Counter(item.row["motion_hint"] for item in selected)),
        "policy": "队列只安排人工核验顺序；motion_hint、检测框和三维轨迹均不是真值，人工完成前禁止训练、调阈值或进入定位门控。",
    }
    (root / "active_person_annotation_queue_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    plan = """# 主动人工核验计划

本队列从 48 个待核验五帧片段中选取首批样本。排序不使用十月标签、定位真值误差或未来帧信息；它只利用六月训练/验证数据中已有的检测时序稳定性、三维证据完整性和自动运动提示。A 类优先回答“近距离真实人员是否在运动”，B 类提供带三维证据的静态/不确定对照，C 类主要排除高置信检测器的误检或三维缺失样本。

标注顺序应按 `review_order` 执行。每条记录先确认 `manual_person_label`；确认是人员后，修订中心帧框、填写五帧一致的 `manual_instance_id`，再给出 `manual_motion_label`。当三维证据不足或遮挡严重时，使用 `uncertain`，不得强行写成 moving/static。完成首批队列后，运行训练数据构建器检查有效实例数与训练/验证覆盖，再决定是否启动检测器微调。
"""
    (root / "annotation_plan.md").write_text(plan, encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a balanced, high-value manual person annotation queue.")
    parser.add_argument("--manifest", default="outputs/target_motion_20260804/temporal_person_review_packets/temporal_person_review_manifest.csv")
    parser.add_argument("--output-dir", default="outputs/target_motion_20260812/active_person_annotation_queue")
    parser.add_argument("--max-items", type=int, default=24)
    args = parser.parse_args()
    print(json.dumps(run(args.manifest, args.output_dir, args.max_items), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
