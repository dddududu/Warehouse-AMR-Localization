from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from experiments.target_motion_20260803.evaluate_pretrained_person_detector import _box_iou, _build_model
from experiments.target_motion_20260804.build_curated_person_annotation_manifest import _is_high_quality


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = ["C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf"]
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def _agreement(semantic_box: np.ndarray, prediction_boxes: np.ndarray, prediction_scores: np.ndarray) -> tuple[float, float, list[float] | None]:
    if prediction_boxes.size == 0:
        return 0.0, 0.0, None
    overlaps = _box_iou(semantic_box.reshape(1, 4), prediction_boxes).reshape(-1)
    combined = overlaps * prediction_scores
    best = int(np.argmax(combined))
    return float(overlaps[best]), float(prediction_scores[best]), prediction_boxes[best].tolist()


def _draw_examples(path: Path, rows: list[dict[str, Any]]) -> None:
    selected = rows[:12]
    columns, width, height = 3, 420, 236
    canvas = Image.new("RGB", (columns * width + 60, ((len(selected) + columns - 1) // columns) * (height + 100) + 105), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((25, 25), "语义候选与独立行人检测器一致：优先人工核验", fill="#172B4D", font=_font(27, True))
    for index, row in enumerate(selected):
        image = cv2.imread(row["image_path"], cv2.IMREAD_COLOR)
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        view = Image.fromarray(image).resize((width, height), resample=Image.Resampling.BILINEAR)
        image_draw = ImageDraw.Draw(view)
        scale_x, scale_y = width / image.shape[1], height / image.shape[0]
        for box, color in ((row["semantic_box"], "#22A447"), (row["detector_box"], "#D64545")):
            if box is not None:
                x1, y1, x2, y2 = np.asarray(box) * np.asarray((scale_x, scale_y, scale_x, scale_y))
                image_draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        column, line = index % columns, index // columns
        x, y = 20 + column * width, 75 + line * (height + 100)
        canvas.paste(view, (x, y))
        draw.text((x, y + height + 8), f"{row['split']} | {row['sequence']} | 帧 {row['frame_idx']}", fill="#334E68", font=_font(16, True))
        draw.text((x, y + height + 35), f"IoU {row['detector_iou']:.2f}，检测置信度 {row['detector_score']:.2f}", fill="#52606D", font=_font(15))
    canvas.save(path)


def run(quality_csv_path: str | Path, output_dir: str | Path, detector_threshold: float, agreement_iou: float) -> dict[str, Any]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    candidates = [row for row in csv.DictReader(Path(quality_csv_path).open(encoding="utf-8-sig")) if _is_high_quality(row)]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_name = _build_model({"name": "fasterrcnn_resnet50_fpn_v2"}, device)
    model.eval()
    ranked: list[dict[str, Any]] = []
    with torch.no_grad():
        for index, row in enumerate(candidates, start=1):
            image = cv2.imread(row["image_path"], cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"Could not read {row['image_path']}")
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float().div_(255.0).to(device)
            output = model([tensor])[0]
            is_person = output["labels"].detach().cpu().numpy() == 1
            boxes = output["boxes"].detach().cpu().numpy()[is_person].astype(np.float32, copy=False)
            scores = output["scores"].detach().cpu().numpy()[is_person].astype(np.float32, copy=False)
            semantic_box = np.asarray([[float(row["x"]), float(row["y"]), float(row["x"]) + float(row["width"]), float(row["y"]) + float(row["height"])]], dtype=np.float32)
            iou, score, detector_box = _agreement(semantic_box[0], boxes, scores)
            ranked.append(
                {
                    "split": row["split"],
                    "sequence": row["sequence"],
                    "frame_idx": int(row["frame_idx"]),
                    "image_path": row["image_path"],
                    "semantic_box": semantic_box[0].astype(int).tolist(),
                    "detector_box": detector_box,
                    "detector_iou": iou,
                    "detector_score": score,
                    "agreement_score": iou * score,
                    "prioritize_manual_review": bool(iou >= agreement_iou and score >= detector_threshold),
                }
            )
            if index % 20 == 0 or index == len(candidates):
                print(f"scored {index}/{len(candidates)} candidates")
    ranked.sort(key=lambda row: float(row["agreement_score"]), reverse=True)
    fields = [key for key in ranked[0] if key not in {"semantic_box", "detector_box"}] + ["semantic_box_json", "detector_box_json"] if ranked else []
    with (root / "detector_agreement_ranking.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in ranked:
            serialized = {key: value for key, value in row.items() if key not in {"semantic_box", "detector_box"}}
            serialized["semantic_box_json"] = json.dumps(row["semantic_box"])
            serialized["detector_box_json"] = json.dumps(row["detector_box"])
            writer.writerow(serialized)
    prioritized = [row for row in ranked if row["prioritize_manual_review"]]
    _draw_examples(root / "detector_agreement_examples.png", prioritized)
    counts = defaultdict(int)
    for row in prioritized:
        counts[row["split"]] += 1
    summary = {
        "model": model_name,
        "candidates": len(candidates),
        "prioritized_for_manual_review": len(prioritized),
        "prioritized_by_split": dict(counts),
        "detector_threshold": detector_threshold,
        "agreement_iou": agreement_iou,
        "interpretation": "检测框只用来排序人工核验优先级；不构成检测器训练真值，也不作为定位门控输入。",
    }
    (root / "detector_agreement_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank person semantic candidates with independent COCO detector agreement.")
    parser.add_argument("--quality-csv", default="outputs/target_motion_20260804/person_component_quality_audit/person_component_quality.csv")
    parser.add_argument("--output-dir", default="outputs/target_motion_20260804/detector_agreement_ranking")
    parser.add_argument("--detector-threshold", type=float, default=0.50)
    parser.add_argument("--agreement-iou", type=float, default=0.20)
    args = parser.parse_args()
    print(json.dumps(run(args.quality_csv, args.output_dir, args.detector_threshold, args.agreement_iou), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
