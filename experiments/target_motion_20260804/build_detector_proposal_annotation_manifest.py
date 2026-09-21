from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont

from dataset_io.frame_indexer import build_frame_index
from experiments.target_motion_20260803.evaluate_pretrained_person_detector import _build_model


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


def _choose_diverse(rows: list[dict[str, Any]], maximum: int, minimum_frame_gap: int) -> list[dict[str, Any]]:
    chosen: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: float(item["detector_score"]), reverse=True):
        if all(abs(int(row["frame_idx"]) - int(existing["frame_idx"])) >= minimum_frame_gap for existing in chosen):
            chosen.append(row)
        if len(chosen) == maximum:
            break
    return sorted(chosen, key=lambda item: int(item["frame_idx"]))


def _draw_contact_sheet(path: Path, rows: list[dict[str, Any]]) -> None:
    selected = rows[:12]
    columns, width, height = 3, 420, 236
    canvas = Image.new("RGB", (columns * width + 60, ((len(selected) + columns - 1) // columns) * (height + 100) + 105), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((25, 25), "COCO 行人检测提议：需人工确认后才可用于训练", fill="#172B4D", font=_font(27, True))
    for index, row in enumerate(selected):
        image = cv2.imread(row["image_left_path"], cv2.IMREAD_COLOR)
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        view = Image.fromarray(image).resize((width, height), resample=Image.Resampling.BILINEAR)
        x1, y1, x2, y2 = np.asarray(row["suggested_box"], dtype=np.float64) * np.asarray((width / image.shape[1], height / image.shape[0], width / image.shape[1], height / image.shape[0]))
        ImageDraw.Draw(view).rectangle((x1, y1, x2, y2), outline="#D64545", width=3)
        column, line = index % columns, index // columns
        x, y = 20 + column * width, 75 + line * (height + 100)
        canvas.paste(view, (x, y))
        draw.text((x, y + height + 8), f"{row['split']} | {row['sequence']} | 帧 {row['anchor_frame']}", fill="#334E68", font=_font(16, True))
        draw.text((x, y + height + 35), f"检测置信度 {row['detector_score']:.2f}", fill="#52606D", font=_font(15))
    canvas.save(path)


def run(config_path: str | Path, output_dir: str | Path, frame_stride: int, detector_threshold: float, samples_per_sequence: int, minimum_frame_gap: int, window_radius: int) -> dict[str, Any]:
    config = _load_yaml(config_path)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_name = _build_model({"name": "fasterrcnn_resnet50_fpn_v2"}, device)
    model.eval()
    proposals: list[dict[str, Any]] = []
    frame_counts: dict[str, int] = {}
    with torch.no_grad():
        for split, entries in config["splits"].items():
            for entry in entries:
                sequence_root = Path(entry["sequence_root"])
                sequence = sequence_root.name.lower()
                records = build_frame_index(sequence_root)
                selected_records = [(frame_idx, record) for frame_idx, record in enumerate(records) if frame_idx % frame_stride == 0]
                frame_counts[f"{split}:{sequence}"] = len(selected_records)
                for progress, (frame_idx, record) in enumerate(selected_records, start=1):
                    image = cv2.imread(str(record.image_left_path), cv2.IMREAD_COLOR)
                    if image is None:
                        raise RuntimeError(f"Could not read {record.image_left_path}")
                    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    tensor = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float().div_(255.0).to(device)
                    output = model([tensor])[0]
                    person = output["labels"].detach().cpu().numpy() == 1
                    boxes = output["boxes"].detach().cpu().numpy()[person].astype(np.float32, copy=False)
                    scores = output["scores"].detach().cpu().numpy()[person].astype(np.float32, copy=False)
                    for box, score in zip(boxes, scores, strict=True):
                        if float(score) >= detector_threshold:
                            proposals.append({"split": split, "sequence": sequence, "frame_idx": frame_idx, "image_left_path": str(record.image_left_path), "suggested_box": box.tolist(), "detector_score": float(score)})
                    if progress % 50 == 0 or progress == len(selected_records):
                        print(f"scored {split}:{sequence} {progress}/{len(selected_records)}")
    selected: list[dict[str, Any]] = []
    for split in ("train", "validation"):
        for sequence in sorted({row["sequence"] for row in proposals if row["split"] == split}):
            sequence_rows = _choose_diverse([row for row in proposals if row["split"] == split and row["sequence"] == sequence], samples_per_sequence, minimum_frame_gap)
            for row in sequence_rows:
                sequence_root = next(Path(entry["sequence_root"]) for entry in config["splits"][split] if Path(entry["sequence_root"]).name.lower() == sequence)
                frame_count = len(build_frame_index(sequence_root))
                selected.append({**row, "anchor_frame": row["frame_idx"], "window_start_frame": max(0, row["frame_idx"] - window_radius), "window_end_frame": min(frame_count - 1, row["frame_idx"] + window_radius), "annotation_status": "pending_manual_verification"})
    fields = [key for key in selected[0] if key != "suggested_box"] + ["suggested_box_json"] if selected else []
    with (root / "detector_proposal_annotation_manifest.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in selected:
            serialized = {key: value for key, value in row.items() if key != "suggested_box"}
            serialized["suggested_box_json"] = json.dumps(row["suggested_box"])
            writer.writerow(serialized)
    _draw_contact_sheet(root / "detector_proposal_annotation_candidates.png", selected)
    instructions = """# 行人检测提议人工核验说明

红框来自通用 COCO 行人检测器，只用于补充语义标签未覆盖的正样本，不能直接作为训练真值。每条记录对应中心帧和前后各两帧。请确认是否为真实行人；对真实行人修正中心帧二维框并保证五帧内实例编号一致；记录可见性、遮挡程度和运动状态；误检则明确标为非行人。完成核验后，训练集用于模型训练，验证集用于选阈值和早停；十月跨日期数据继续只用于最终测试，不得回流。
"""
    (root / "annotation_instructions.md").write_text(instructions, encoding="utf-8")
    summary = {
        "model": model_name,
        "scanned_frames": frame_counts,
        "raw_high_confidence_proposals": len(proposals),
        "selected_diverse_clips": len(selected),
        "selected_by_split": {split: sum(row["split"] == split for row in selected) for split in ("train", "validation")},
        "detector_threshold": detector_threshold,
        "blind_test_policy": "十月跨日期数据未扫描、未选阈值、未用于候选选择。",
        "annotation_policy": "所有红框均需人工确认与修订，才可用作训练标签。",
    }
    (root / "detector_proposal_annotation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a balanced manual-verification manifest from high-confidence COCO person proposals.")
    parser.add_argument("--config", default="experiments/target_motion_20260804/person_component_quality_audit.yaml")
    parser.add_argument("--output-dir", default="outputs/target_motion_20260804/detector_proposal_annotation_manifest")
    parser.add_argument("--frame-stride", type=int, default=10)
    parser.add_argument("--detector-threshold", type=float, default=0.80)
    parser.add_argument("--samples-per-sequence", type=int, default=8)
    parser.add_argument("--minimum-frame-gap", type=int, default=30)
    parser.add_argument("--window-radius", type=int, default=2)
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.output_dir, args.frame_stride, args.detector_threshold, args.samples_per_sequence, args.minimum_frame_gap, args.window_radius), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
