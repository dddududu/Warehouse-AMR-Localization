from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

from dataset_io.depth_loader import load_depth_png
from dataset_io.sequence_dataset import WarehouseSequenceDataset
from geometry.se3 import transform_points


@dataclass(frozen=True)
class ComponentQuality:
    split: str
    sequence: str
    frame_idx: int
    image_path: str
    x: int
    y: int
    width: int
    height: int
    pixel_area: int
    valid_depth_ratio: float
    median_depth_m: float | None
    stereo_person_support: float | None


def _load_config(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("Quality audit config must be a mapping.")
    return payload


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = ["C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf"]
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def _right_support(
    pixels_x: np.ndarray,
    pixels_y: np.ndarray,
    depth_m: np.ndarray,
    right_mask: np.ndarray,
    dataset: WarehouseSequenceDataset,
) -> float | None:
    if depth_m.size == 0:
        return None
    left = dataset.camera_left
    normalized = cv2.undistortPoints(
        np.column_stack((pixels_x, pixels_y)).reshape(-1, 1, 2),
        left.build_K(),
        left.distortion,
    ).reshape(-1, 2)
    points_left = np.column_stack((normalized * depth_m[:, None], depth_m))
    points_sensor = transform_points(dataset.calibration.T_os_cam_left, points_left)
    points_right = transform_points(dataset.calibration.T_cam2_os.matrix, points_sensor)
    projected, _ = cv2.projectPoints(points_right, np.zeros(3), np.zeros(3), dataset.camera_right.build_K(), dataset.camera_right.distortion)
    coordinates = np.rint(projected.reshape(-1, 2)).astype(np.int64)
    valid = (
        (points_right[:, 2] > 0.0)
        & (coordinates[:, 0] >= 0)
        & (coordinates[:, 0] < right_mask.shape[1])
        & (coordinates[:, 1] >= 0)
        & (coordinates[:, 1] < right_mask.shape[0])
    )
    if not np.any(valid):
        return 0.0
    return float(np.mean(right_mask[coordinates[valid, 1], coordinates[valid, 0]] == 13))


def _frame_components(mask: np.ndarray, depth: np.ndarray, right_mask: np.ndarray, dataset: WarehouseSequenceDataset, split: str, sequence: str, frame_idx: int, image_path: Path, analysis: dict[str, Any], person_label: int) -> list[ComponentQuality]:
    binary = (mask == int(person_label)).astype(np.uint8)
    count, component_labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    components: list[ComponentQuality] = []
    for component_idx in range(1, count):
        x, y, width, height, area = stats[component_idx].tolist()
        if int(area) < int(analysis["min_component_pixels"]):
            continue
        pixel_y, pixel_x = np.where(component_labels == component_idx)
        depth_m = depth[pixel_y, pixel_x].astype(np.float64) * float(analysis["depth_scale_m"])
        valid = np.isfinite(depth_m) & (depth_m >= float(analysis["depth_min_m"])) & (depth_m <= float(analysis["depth_max_m"]))
        valid_ratio = float(np.mean(valid))
        selected_x = pixel_x[valid].astype(np.float64)
        selected_y = pixel_y[valid].astype(np.float64)
        selected_depth = depth_m[valid]
        if selected_depth.size > int(analysis["max_projected_points"]):
            indices = np.linspace(0, selected_depth.size - 1, int(analysis["max_projected_points"]), dtype=np.int64)
            selected_x, selected_y, selected_depth = selected_x[indices], selected_y[indices], selected_depth[indices]
        components.append(
            ComponentQuality(
                split=split,
                sequence=sequence,
                frame_idx=int(frame_idx),
                image_path=str(image_path),
                x=int(x),
                y=int(y),
                width=int(width),
                height=int(height),
                pixel_area=int(area),
                valid_depth_ratio=valid_ratio,
                median_depth_m=float(np.median(selected_depth)) if selected_depth.size else None,
                stereo_person_support=_right_support(selected_x, selected_y, selected_depth, right_mask, dataset) if selected_depth.size else None,
            )
        )
    return components


def _draw_examples(path: Path, items: list[ComponentQuality], title: str) -> None:
    canvas = Image.new("RGB", (1460, max(len(items), 1) * 220 + 100), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((35, 25), title, fill="#172B4D", font=_font(28, True))
    for index, item in enumerate(items):
        image = cv2.imread(item.image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        view = Image.fromarray(image).resize((430, 242), resample=Image.Resampling.BILINEAR)
        scale_x = 430.0 / image.shape[1]
        scale_y = 242.0 / image.shape[0]
        ImageDraw.Draw(view).rectangle((item.x * scale_x, item.y * scale_y, (item.x + item.width) * scale_x, (item.y + item.height) * scale_y), outline="#22A447", width=3)
        y = 75 + index * 220
        canvas.paste(view, (25, y))
        depth_text = "无有效深度" if item.median_depth_m is None else f"深度 {item.median_depth_m:.2f} m"
        support_text = "无双目支持" if item.stereo_person_support is None else f"右目支持 {item.stereo_person_support:.2f}"
        draw.text((480, y + 40), f"{item.split} | {item.sequence} | 帧 {item.frame_idx}", fill="#334E68", font=_font(21, True))
        draw.text((480, y + 83), f"框 {item.width}×{item.height}；面积 {item.pixel_area}；深度有效率 {item.valid_depth_ratio:.2f}", fill="#52606D", font=_font(19))
        draw.text((480, y + 122), f"{depth_text}；{support_text}", fill="#52606D", font=_font(19))
    canvas.save(path)


def _write_csv(path: Path, items: list[ComponentQuality]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ComponentQuality.__dataclass_fields__))
        writer.writeheader()
        writer.writerows(item.__dict__ for item in items)


def run(config_path: str | Path) -> dict[str, Any]:
    config = _load_config(config_path)
    analysis = dict(config["analysis"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    items: list[ComponentQuality] = []
    for split, entries in config["splits"].items():
        for entry in entries:
            dataset = WarehouseSequenceDataset(entry["sequence_root"], entry["calibration_path"], {"load_lidar": False})
            sequence = Path(entry["sequence_root"]).name.lower()
            for frame_idx, record in enumerate(dataset.frame_index):
                if frame_idx % int(analysis["frame_stride"]) != 0:
                    continue
                left_mask = cv2.imread(str(record.segmentation_greyscale_left_path), cv2.IMREAD_UNCHANGED)
                right_mask = cv2.imread(str(record.segmentation_greyscale_right_path), cv2.IMREAD_UNCHANGED)
                depth = load_depth_png(record.depth_left_path)
                if left_mask is None or right_mask is None or left_mask.shape != depth.shape:
                    raise RuntimeError(f"Invalid stereo semantic/depth data at {sequence}:{frame_idx}.")
                items.extend(_frame_components(left_mask, depth, right_mask, dataset, split, sequence, frame_idx, Path(record.image_left_path), analysis, int(config["person_label"])))
            print(f"audited {split}:{sequence}")
    _write_csv(output_dir / "person_component_quality.csv", items)
    supports = np.asarray([item.stereo_person_support for item in items if item.stereo_person_support is not None], dtype=np.float64)
    heights = np.asarray([item.height for item in items], dtype=np.float64)
    depths = np.asarray([item.median_depth_m for item in items if item.median_depth_m is not None], dtype=np.float64)
    high_confidence = [item for item in items if item.height >= 32 and item.pixel_area >= 400 and item.valid_depth_ratio >= 0.70 and item.median_depth_m is not None and item.median_depth_m <= 8.0]
    low_confidence = [item for item in items if item not in high_confidence]
    examples = int(analysis["visual_examples_per_group"])
    _draw_examples(output_dir / "high_confidence_component_examples.png", sorted(high_confidence, key=lambda item: (item.height, item.pixel_area), reverse=True)[:examples], "近距离大尺寸人员组件候选：左目尺寸和深度质量满足条件")
    _draw_examples(output_dir / "low_confidence_component_examples.png", sorted(low_confidence, key=lambda item: (item.height, item.pixel_area))[:examples], "低置信人员组件候选：过小、深度无效或过远，需剔除")
    summary = {
        "components": len(items),
        "high_confidence_components": len(high_confidence),
        "high_confidence_ratio": len(high_confidence) / max(len(items), 1),
        "height_percentiles_px": np.quantile(heights, [0.1, 0.5, 0.9]).tolist() if heights.size else [],
        "median_depth_percentiles_m": np.quantile(depths, [0.1, 0.5, 0.9]).tolist() if depths.size else [],
        "stereo_support_percentiles": np.quantile(supports, [0.1, 0.5, 0.9]).tolist() if supports.size else [],
        "proposed_rule": "height>=32px, pixel_area>=400, valid_depth_ratio>=0.70, median_depth<=8m",
        "right_camera_check": "左右语义类别和深度投影在该数据中不一致，不能用作实例级筛选条件；保留为数据质量限制。",
        "note": "筛选规则只在训练和验证日期的组件统计上形成，十月保留集不会用于设定阈值。",
    }
    (output_dir / "person_component_quality_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit geometric and stereo quality of person semantic components.")
    parser.add_argument("--config", default="experiments/target_motion_20260804/person_component_quality_audit.yaml")
    args = parser.parse_args()
    print(json.dumps(run(args.config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
