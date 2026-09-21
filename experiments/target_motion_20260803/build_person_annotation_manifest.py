from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

from dataset_io.frame_indexer import build_frame_index


def _load_yaml(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("Expected a mapping.")
    return payload


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = ["C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf"]
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def _component_boxes(path: Path, label: int = 13, minimum_pixels: int = 96) -> list[list[int]]:
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        return []
    count, _, stats, _ = cv2.connectedComponentsWithStats((mask == int(label)).astype(np.uint8), connectivity=8)
    return [[int(x), int(y), int(x + width), int(y + height)] for x, y, width, height, area in stats[1:].tolist() if int(area) >= int(minimum_pixels)]


def _choose_segments(rows: list[dict[str, str]], count: int) -> list[dict[str, str]]:
    per_sequence: dict[str, deque[dict[str, str]]] = defaultdict(deque)
    for row in sorted(rows, key=lambda item: (float(item["max_pixel_coverage"]), int(item["span_frames"])), reverse=True):
        per_sequence[row["sequence"]].append(row)
    chosen: list[dict[str, str]] = []
    while len(chosen) < int(count) and any(per_sequence.values()):
        for sequence in sorted(per_sequence):
            if per_sequence[sequence] and len(chosen) < int(count):
                chosen.append(per_sequence[sequence].popleft())
    return chosen


def _draw_contact_sheet(path: Path, rows: list[dict[str, Any]]) -> None:
    selected = rows[:12]
    columns, width, height = 3, 420, 236
    canvas = Image.new("RGB", (columns * width + 60, ((len(selected) + columns - 1) // columns) * (height + 85) + 105), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((25, 25), "人员实例与运动标注候选片段（绿框为语义建议框，需人工核验）", fill="#172B4D", font=_font(27, True))
    for index, row in enumerate(selected):
        image = cv2.imread(row["image_left_path"], cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        view = Image.fromarray(image).resize((width, height), resample=Image.Resampling.BILINEAR)
        view_draw = ImageDraw.Draw(view)
        scale_x = width / image.shape[1]
        scale_y = height / image.shape[0]
        for box in row["suggested_boxes"]:
            view_draw.rectangle(tuple(np.asarray(box) * np.asarray((scale_x, scale_y, scale_x, scale_y))), outline="#22A447", width=3)
        column = index % columns
        line = index // columns
        x = 20 + column * width
        y = 75 + line * (height + 85)
        canvas.paste(view, (x, y))
        draw.text((x, y + height + 10), f"{row['split']} | {row['sequence']} | 帧 {row['anchor_frame']}", fill="#334E68", font=_font(17, True))
    canvas.save(path)


def run(audit_config_path: str | Path, output_dir: str | Path, segments_per_split: int) -> dict[str, Any]:
    audit_config = _load_yaml(audit_config_path)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    source_root = Path(audit_config_path).parent
    audit_output = Path(audit_config["output_dir"])
    segment_rows = [row for row in csv.DictReader((audit_output / "target_segments.csv").open(encoding="utf-8-sig")) if row["target_name"] == "person"]
    sequence_info = {str(item["name"]): item for item in audit_config["sequences"]}
    selected: list[dict[str, Any]] = []
    for split in ("train", "validation", "blind_test"):
        candidates = [row for row in segment_rows if sequence_info[row["sequence"]]["split"] == split]
        for row in _choose_segments(candidates, segments_per_split):
            sequence = sequence_info[row["sequence"]]
            records = build_frame_index(Path(sequence["sequence_root"]))
            anchor = (int(row["start_frame"]) + int(row["end_frame"])) // 2
            record = records[anchor]
            boxes = _component_boxes(Path(record.segmentation_greyscale_left_path))
            selected.append(
                {
                    "split": split,
                    "sequence": row["sequence"],
                    "anchor_frame": anchor,
                    "window_start_frame": max(0, anchor - 2),
                    "window_end_frame": min(len(records) - 1, anchor + 2),
                    "image_left_path": str(record.image_left_path),
                    "semantic_mask_path": str(record.segmentation_greyscale_left_path),
                    "suggested_boxes": boxes,
                    "suggested_box_count": len(boxes),
                    "segment_span_frames": int(row["span_frames"]),
                    "mean_semantic_coverage": float(row["mean_pixel_coverage"]),
                    "max_semantic_coverage": float(row["max_pixel_coverage"]),
                    "max_position_error_m": row["max_position_error_m"],
                    "annotation_status": "pending_manual_verification",
                }
            )
    manifest = root / "person_instance_motion_annotation_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8-sig") as handle:
        fields = [key for key in selected[0] if key != "suggested_boxes"] + ["suggested_boxes_json"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in selected:
            serializable = {key: value for key, value in row.items() if key != "suggested_boxes"}
            serializable["suggested_boxes_json"] = json.dumps(row["suggested_boxes"], ensure_ascii=False)
            writer.writerow(serializable)
    _draw_contact_sheet(root / "person_annotation_candidates.png", selected)
    instructions = """# 人员实例与运动标注说明\n\n每条记录以中心帧及其前后两帧组成五帧短片段。绿色建议框来自语义标签连通域，仅用于加快人工工作，必须逐项确认或修改。\n\n需要填写：人员实例编号（同一短片段内保持一致）、中心帧二维框、是否为真实人员、可见性（完整/部分遮挡/极小目标）、五帧内的运动状态（静止/运动/不确定）以及是否遮挡了机器人前方主要几何区域。十月条目仅用于最终评测，不可回流至训练。\n\n建议先完成训练和验证各 20 个片段，再检查标注一致性；通过后再标注十月保留集。\n"""
    (root / "annotation_instructions.md").write_text(instructions, encoding="utf-8")
    summary = {"manifest_rows": len(selected), "splits": {split: sum(row["split"] == split for row in selected) for split in ("train", "validation", "blind_test")}, "source": str(source_root / "target_motion_audit.yaml"), "note": "This manifest contains suggestions only; it does not create manual ground truth."}
    (root / "annotation_manifest_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a balanced manual-verification manifest for person instances and motion clips.")
    parser.add_argument("--audit-config", default="experiments/target_motion_20260803/target_motion_audit.yaml")
    parser.add_argument("--output-dir", default="outputs/target_motion_20260803/annotation_manifest")
    parser.add_argument("--segments-per-split", type=int, default=20)
    args = parser.parse_args()
    print(json.dumps(run(args.audit_config, args.output_dir, args.segments_per_split), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
