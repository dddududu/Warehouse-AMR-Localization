from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
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


def _is_high_quality(row: dict[str, str]) -> bool:
    depth = row["median_depth_m"]
    return (
        int(row["height"]) >= 32
        and int(row["pixel_area"]) >= 400
        and float(row["valid_depth_ratio"]) >= 0.70
        and bool(depth)
        and float(depth) <= 8.0
    )


def _choose_diverse(rows: list[dict[str, str]], maximum: int, minimum_frame_gap: int) -> list[dict[str, str]]:
    ranked = sorted(rows, key=lambda row: (int(row["height"]), int(row["pixel_area"]), float(row["valid_depth_ratio"])), reverse=True)
    chosen: list[dict[str, str]] = []
    for row in ranked:
        frame_idx = int(row["frame_idx"])
        if all(abs(frame_idx - int(existing["frame_idx"])) >= minimum_frame_gap for existing in chosen):
            chosen.append(row)
        if len(chosen) == maximum:
            break
    return sorted(chosen, key=lambda row: int(row["frame_idx"]))


def _draw_contact_sheet(path: Path, rows: list[dict[str, Any]]) -> None:
    columns, width, height = 3, 420, 236
    selected = rows[:12]
    canvas_height = ((len(selected) + columns - 1) // columns) * (height + 85) + 105
    canvas = Image.new("RGB", (columns * width + 60, canvas_height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((25, 25), "近距离人员候选：建议框需逐项人工确认", fill="#172B4D", font=_font(27, True))
    for index, row in enumerate(selected):
        image = cv2.imread(row["image_left_path"], cv2.IMREAD_COLOR)
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        view = Image.fromarray(image).resize((width, height), resample=Image.Resampling.BILINEAR)
        scale_x, scale_y = width / image.shape[1], height / image.shape[0]
        x1, y1, x2, y2 = row["suggested_box"]
        ImageDraw.Draw(view).rectangle((x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y), outline="#22A447", width=3)
        column, line = index % columns, index // columns
        x, y = 20 + column * width, 75 + line * (height + 85)
        canvas.paste(view, (x, y))
        draw.text((x, y + height + 10), f"{row['split']} | {row['sequence']} | 帧 {row['anchor_frame']}", fill="#334E68", font=_font(17, True))
    canvas.save(path)


def run(quality_config_path: str | Path, output_dir: str | Path, samples_per_sequence: int, minimum_frame_gap: int, window_radius: int) -> dict[str, Any]:
    config = _load_yaml(quality_config_path)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    quality_csv = Path(config["output_dir"]) / "person_component_quality.csv"
    source_rows = [row for row in csv.DictReader(quality_csv.open(encoding="utf-8-sig")) if _is_high_quality(row)]
    roots = {
        (split, Path(entry["sequence_root"]).name.lower()): Path(entry["sequence_root"])
        for split, entries in config["splits"].items()
        for entry in entries
    }
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in source_rows:
        grouped[(row["split"], row["sequence"])].append(row)

    selected: list[dict[str, Any]] = []
    for key in sorted(grouped):
        split, sequence = key
        records = build_frame_index(roots[key])
        for row in _choose_diverse(grouped[key], samples_per_sequence, minimum_frame_gap):
            frame_idx = int(row["frame_idx"])
            selected.append(
                {
                    "split": split,
                    "sequence": sequence,
                    "anchor_frame": frame_idx,
                    "window_start_frame": max(0, frame_idx - window_radius),
                    "window_end_frame": min(len(records) - 1, frame_idx + window_radius),
                    "image_left_path": row["image_path"],
                    "suggested_box": [int(row["x"]), int(row["y"]), int(row["x"]) + int(row["width"]), int(row["y"]) + int(row["height"])],
                    "semantic_component_pixels": int(row["pixel_area"]),
                    "median_depth_m": float(row["median_depth_m"]),
                    "valid_depth_ratio": float(row["valid_depth_ratio"]),
                    "annotation_status": "pending_manual_verification",
                }
            )

    fields = [key for key in selected[0] if key != "suggested_box"] + ["suggested_box_json"] if selected else []
    with (root / "curated_person_annotation_manifest.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in selected:
            serialized = {key: value for key, value in row.items() if key != "suggested_box"}
            serialized["suggested_box_json"] = json.dumps(row["suggested_box"])
            writer.writerow(serialized)
    _draw_contact_sheet(root / "curated_person_annotation_candidates.png", selected)
    instructions = """# 近距离人员实例人工核验说明

该清单从左目语义标签 13 的近距离、大尺寸组件中筛出，绿色建议框只用于加快标注，不能直接作为训练真值。每条记录对应中心帧和前后各两帧。

请逐条确认：建议框是否为真实行人；若是，修正中心帧二维框并给出五帧内一致的实例编号；记录可见性（完整、部分遮挡、极小）和相对机器人运动状态（静止、运动、不确定）；若并非行人，明确标为误标。训练与验证条目可用于构建训练样本和调阈值；十月数据不在此清单中，继续作为不可回流的跨日期盲测集。
"""
    (root / "annotation_instructions.md").write_text(instructions, encoding="utf-8")
    summary = {
        "source_high_quality_components": len(source_rows),
        "selected_clips": len(selected),
        "selected_by_split": {split: sum(row["split"] == split for row in selected) for split in ("train", "validation")},
        "rule": "height>=32px, pixel_area>=400, valid_depth_ratio>=0.70, median_depth<=8m",
        "blind_test_policy": "十月跨日期测试集未参与候选筛选或阈值设定。",
        "annotation_policy": "候选框必须人工核验后才可转为检测器训练真值。",
    }
    (root / "curated_person_annotation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a diverse left-camera manual-verification manifest for near-range people.")
    parser.add_argument("--quality-config", default="experiments/target_motion_20260804/person_component_quality_audit.yaml")
    parser.add_argument("--output-dir", default="outputs/target_motion_20260804/curated_person_annotation_manifest")
    parser.add_argument("--samples-per-sequence", type=int, default=5)
    parser.add_argument("--minimum-frame-gap", type=int, default=30)
    parser.add_argument("--window-radius", type=int, default=2)
    args = parser.parse_args()
    print(json.dumps(run(args.quality_config, args.output_dir, args.samples_per_sequence, args.minimum_frame_gap, args.window_radius), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
