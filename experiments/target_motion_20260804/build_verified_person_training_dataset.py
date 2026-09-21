from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


PERSON_LABELS = {"person", "non_person"}
MOTION_LABELS = {"static", "moving", "uncertain"}


def _parse_box(raw: str) -> list[float]:
    values = json.loads(raw)
    if not isinstance(values, list) or len(values) != 4:
        raise ValueError("manual_box_json must contain [x1, y1, x2, y2].")
    box = [float(value) for value in values]
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError("manual_box_json must have positive width and height.")
    return box


def _review_issue(row: dict[str, str]) -> str | None:
    person_label = row.get("manual_person_label", "pending")
    if person_label not in PERSON_LABELS:
        return "manual_person_label 未完成"
    if person_label == "person":
        if row.get("manual_motion_label", "pending") not in MOTION_LABELS:
            return "manual_motion_label 未完成"
        if row.get("manual_instance_id", "pending") in {"", "pending"}:
            return "manual_instance_id 未完成"
        try:
            _parse_box(row.get("manual_box_json", ""))
        except (TypeError, ValueError, json.JSONDecodeError):
            return "manual_box_json 无效"
    return None


def _coco_annotation(annotation_id: int, image_id: int, box: list[float]) -> dict[str, Any]:
    x1, y1, x2, y2 = box
    width, height = x2 - x1, y2 - y1
    return {"id": annotation_id, "image_id": image_id, "category_id": 1, "bbox": [x1, y1, width, height], "area": width * height, "iscrowd": 0}


def build(rows: list[dict[str, str]], require_verified: bool = True) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, str]]]:
    issues = [(index, _review_issue(row)) for index, row in enumerate(rows, start=2)]
    incomplete = [(index, issue) for index, issue in issues if issue is not None]
    if require_verified and incomplete:
        preview = "; ".join(f"CSV 第 {index} 行：{issue}" for index, issue in incomplete[:8])
        raise ValueError(f"拒绝构建训练集：仍有 {len(incomplete)} 条未核验记录。{preview}")
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    motions: list[dict[str, Any]] = []
    annotation_id = 1
    for image_id, row in enumerate(rows, start=1):
        if _review_issue(row) is not None:
            continue
        image_paths = json.loads(row["window_image_paths_json"])
        if not isinstance(image_paths, list) or not image_paths:
            raise ValueError("window_image_paths_json must contain at least one image path.")
        images.append({"id": image_id, "file_name": str(image_paths[len(image_paths) // 2]), "split": row["split"], "sequence": row["sequence"], "frame_idx": int(row["anchor_frame"])})
        if row["manual_person_label"] != "person":
            continue
        box = _parse_box(row["manual_box_json"])
        annotations.append(_coco_annotation(annotation_id, image_id, box))
        annotation_id += 1
        motions.append(
            {
                "image_id": image_id,
                "split": row["split"],
                "sequence": row["sequence"],
                "frame_idx": int(row["anchor_frame"]),
                "instance_id": row["manual_instance_id"],
                "motion_label": row["manual_motion_label"],
                "box_xyxy": box,
            }
        )
    coco = {"images": images, "annotations": annotations, "categories": [{"id": 1, "name": "person"}]}
    return coco, motions, [{"csv_row": index, "issue": issue} for index, issue in incomplete]


def run(manifest_path: str | Path, output_dir: str | Path, allow_unverified: bool = False) -> dict[str, Any]:
    rows = list(csv.DictReader(Path(manifest_path).open(encoding="utf-8-sig")))
    coco, motions, incomplete = build(rows, require_verified=not allow_unverified)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / "person_detection_coco.json").write_text(json.dumps(coco, ensure_ascii=False, indent=2), encoding="utf-8")
    with (root / "person_motion_labels.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        fields = list(motions[0]) if motions else ["image_id", "split", "sequence", "frame_idx", "instance_id", "motion_label", "box_xyxy"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in motions:
            serialized = dict(row)
            serialized["box_xyxy"] = json.dumps(serialized["box_xyxy"])
            writer.writerow(serialized)
    summary = {
        "input_records": len(rows),
        "verified_records": len(coco["images"]),
        "verified_person_boxes": len(coco["annotations"]),
        "motion_labels": dict(Counter(row["motion_label"] for row in motions)),
        "unverified_records": len(incomplete),
        "policy": "allow_unverified=true 只用于检查格式；训练必须使用 allow_unverified=false 的全核验输出。",
    }
    (root / "verified_dataset_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a training dataset only from manually verified person and motion labels.")
    parser.add_argument("--manifest", default="outputs/target_motion_20260804/temporal_person_review_packets/temporal_person_review_manifest.csv")
    parser.add_argument("--output-dir", default="outputs/target_motion_20260804/verified_person_training_dataset")
    parser.add_argument("--allow-unverified", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.manifest, args.output_dir, args.allow_unverified), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
