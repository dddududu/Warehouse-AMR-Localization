from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_mobilenet_v3_large_320_fpn,
    fasterrcnn_resnet50_fpn_v2,
)

from dataset_io.frame_indexer import build_frame_index


@dataclass(frozen=True)
class DetectorFrame:
    sequence: str
    frame_idx: int
    image_path: Path
    mask_path: Path


class DetectorDataset(Dataset[dict[str, Any]]):
    def __init__(self, frames: list[DetectorFrame], person_label: int, min_component_pixels: int) -> None:
        self.frames = frames
        self.person_label = int(person_label)
        self.min_component_pixels = int(min_component_pixels)

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, index: int) -> dict[str, Any]:
        frame = self.frames[index]
        image = cv2.imread(str(frame.image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(frame.mask_path), cv2.IMREAD_UNCHANGED)
        if image is None or mask is None:
            raise RuntimeError(f"Could not read detector frame {frame.sequence}:{frame.frame_idx}.")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return {
            "frame": frame,
            "image": torch.from_numpy(image.transpose(2, 0, 1).copy()).float().div_(255.0),
            "reference_boxes": _semantic_component_boxes(mask, self.person_label, self.min_component_pixels),
        }


def _collate(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return items


def _load_config(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("Pretrained detector config must be a mapping.")
    return payload


def _semantic_component_boxes(mask: np.ndarray, person_label: int, min_component_pixels: int) -> np.ndarray:
    binary = (mask == int(person_label)).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    boxes = []
    for component_id in range(1, count):
        x, y, width, height, area = stats[component_id].tolist()
        if int(area) >= int(min_component_pixels):
            boxes.append((float(x), float(y), float(x + width), float(y + height)))
    return np.asarray(boxes, dtype=np.float32).reshape(-1, 4)


def _box_iou(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    if first.size == 0 or second.size == 0:
        return np.zeros((first.shape[0], second.shape[0]), dtype=np.float32)
    top_left = np.maximum(first[:, None, :2], second[None, :, :2])
    bottom_right = np.minimum(first[:, None, 2:], second[None, :, 2:])
    intersection_hw = np.maximum(bottom_right - top_left, 0.0)
    intersection = intersection_hw[..., 0] * intersection_hw[..., 1]
    first_area = np.maximum(first[:, 2] - first[:, 0], 0.0) * np.maximum(first[:, 3] - first[:, 1], 0.0)
    second_area = np.maximum(second[:, 2] - second[:, 0], 0.0) * np.maximum(second[:, 3] - second[:, 1], 0.0)
    return intersection / np.maximum(first_area[:, None] + second_area[None, :] - intersection, 1.0e-6)


def _match_count(predictions: np.ndarray, references: np.ndarray, iou_threshold: float) -> int:
    overlaps = _box_iou(predictions, references)
    matches = 0
    used_predictions: set[int] = set()
    used_references: set[int] = set()
    for flat_index in np.argsort(overlaps.reshape(-1))[::-1]:
        prediction_index, reference_index = np.unravel_index(int(flat_index), overlaps.shape)
        if overlaps[prediction_index, reference_index] < float(iou_threshold):
            break
        if prediction_index in used_predictions or reference_index in used_references:
            continue
        used_predictions.add(prediction_index)
        used_references.add(reference_index)
        matches += 1
    return matches


def _build_frames(roots: list[str], frame_stride: int) -> list[DetectorFrame]:
    frames: list[DetectorFrame] = []
    for root_value in roots:
        root = Path(root_value)
        sequence = root.name.lower()
        for frame_idx, record in enumerate(build_frame_index(root)):
            if frame_idx % int(frame_stride) == 0:
                frames.append(
                    DetectorFrame(
                        sequence=sequence,
                        frame_idx=frame_idx,
                        image_path=Path(record.image_left_path),
                        mask_path=Path(record.segmentation_greyscale_left_path),
                    )
                )
    return frames


def _score(prediction_boxes: np.ndarray, prediction_scores: np.ndarray, reference_boxes: np.ndarray, thresholds: list[float], iou_threshold: float) -> list[dict[str, int]]:
    rows = []
    for threshold in thresholds:
        selected = prediction_boxes[prediction_scores >= float(threshold)]
        matches = _match_count(selected, reference_boxes, iou_threshold)
        rows.append({"threshold": threshold, "true_positive": matches, "false_positive": int(selected.shape[0] - matches), "false_negative": int(reference_boxes.shape[0] - matches)})
    return rows


def _metrics(counts: dict[str, int]) -> dict[str, float | int]:
    true_positive = int(counts["true_positive"])
    false_positive = int(counts["false_positive"])
    false_negative = int(counts["false_negative"])
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    return {
        **counts,
        "precision": precision,
        "recall": recall,
        "f1": 2.0 * precision * recall / max(precision + recall, 1.0e-8),
    }


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = ["C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf"]
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def _draw_examples(output_path: Path, examples: list[dict[str, Any]]) -> None:
    canvas = Image.new("RGB", (1550, max(len(examples), 1) * 265 + 100), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((45, 30), "十月盲测：预训练人员实例检测样例（绿：语义组件框，红：COCO 人员框）", fill="#172B4D", font=_font(30, True))
    for index, example in enumerate(examples):
        image = cv2.imread(example["image_path"], cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        view = Image.fromarray(image).resize((480, 270), resample=Image.Resampling.BILINEAR)
        view_draw = ImageDraw.Draw(view)
        scale_x = 480.0 / image.shape[1]
        scale_y = 270.0 / image.shape[0]
        for box in example["reference_boxes"]:
            view_draw.rectangle(tuple(np.asarray(box) * np.asarray((scale_x, scale_y, scale_x, scale_y))), outline="#22A447", width=3)
        for box, score in zip(example["prediction_boxes"], example["prediction_scores"], strict=True):
            view_draw.rectangle(tuple(np.asarray(box) * np.asarray((scale_x, scale_y, scale_x, scale_y))), outline="#D64545", width=3)
            view_draw.text((float(box[0]) * scale_x, max(0, float(box[1]) * scale_y - 22)), f"{score:.2f}", fill="#D64545", font=_font(18, True))
        y = 85 + index * 265
        canvas.paste(view, (35, y))
        draw.text((550, y + 35), f"{example['sequence']}，第 {example['frame_idx']} 帧", fill="#334E68", font=_font(23, True))
        draw.text((550, y + 80), f"语义组件数：{len(example['reference_boxes'])}；检测框数：{len(example['prediction_boxes'])}", fill="#52606D", font=_font(21))
    canvas.save(output_path)


def _build_model(model_config: dict[str, Any], device: torch.device) -> tuple[torch.nn.Module, str]:
    name = str(model_config["name"])
    if name == "fasterrcnn_mobilenet_v3_large_320_fpn":
        return fasterrcnn_mobilenet_v3_large_320_fpn(weights=FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT).to(device), "torchvision FasterRCNN_MobileNet_V3_Large_320_FPN COCO DEFAULT"
    if name == "fasterrcnn_resnet50_fpn_v2":
        return fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT).to(device), "torchvision FasterRCNN_ResNet50_FPN_V2 COCO DEFAULT"
    raise ValueError(f"Unsupported pretrained detector: {name}")


def _run_split(
    model: torch.nn.Module,
    frames: list[DetectorFrame],
    person_label: int,
    evaluation: dict[str, Any],
    device: torch.device,
    selected_threshold: float | None,
) -> tuple[list[dict[str, Any]], dict[float, dict[str, int]]]:
    dataset = DetectorDataset(frames, person_label, int(evaluation["min_component_pixels"]))
    loader = DataLoader(dataset, batch_size=int(evaluation.get("batch_size", 1)), shuffle=False, num_workers=int(evaluation.get("num_workers", 1)), pin_memory=device.type == "cuda", collate_fn=_collate)
    thresholds = [float(value) for value in evaluation["confidence_thresholds"]]
    counts = {threshold: {"true_positive": 0, "false_positive": 0, "false_negative": 0} for threshold in thresholds}
    examples: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            outputs = model([item["image"].to(device, non_blocking=True) for item in batch])
            for item, output in zip(batch, outputs, strict=True):
                person = output["labels"].detach().cpu().numpy() == 1
                boxes = output["boxes"].detach().cpu().numpy()[person].astype(np.float32, copy=False)
                scores = output["scores"].detach().cpu().numpy()[person].astype(np.float32, copy=False)
                references = np.asarray(item["reference_boxes"], dtype=np.float32)
                for row in _score(boxes, scores, references, thresholds, float(evaluation["match_iou_threshold"])):
                    threshold_counts = counts[float(row["threshold"])]
                    for key in threshold_counts:
                        threshold_counts[key] += int(row[key])
                if selected_threshold is not None and len(examples) < 6 and references.shape[0] > 0:
                    selected = scores >= float(selected_threshold)
                    examples.append(
                        {
                            "sequence": item["frame"].sequence,
                            "frame_idx": item["frame"].frame_idx,
                            "image_path": str(item["frame"].image_path),
                            "reference_boxes": references.tolist(),
                            "prediction_boxes": boxes[selected].tolist(),
                            "prediction_scores": scores[selected].tolist(),
                        }
                    )
    return examples, counts


def run(config_path: str | Path) -> dict[str, Any]:
    config = _load_config(config_path)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluation = dict(config["evaluation"])
    device = torch.device(str(evaluation["device"]) if torch.cuda.is_available() else "cpu")
    model, model_name = _build_model(dict(config["model"]), device)
    frames_by_split = {name: _build_frames(list(roots), int(evaluation["frame_stride"])) for name, roots in config["splits"].items()}
    _, validation_counts = _run_split(model, frames_by_split["validation"], int(config["person_label"]), evaluation, device, None)
    validation = {str(threshold): _metrics(counts) for threshold, counts in validation_counts.items()}
    selected_threshold = max(validation_counts, key=lambda threshold: _metrics(validation_counts[threshold])["f1"])
    examples, blind_counts = _run_split(model, frames_by_split["blind_test"], int(config["person_label"]), evaluation, device, float(selected_threshold))
    blind_test = _metrics(blind_counts[float(selected_threshold)])
    _draw_examples(output_dir / "pretrained_person_detector_oct12_examples.png", examples)
    summary = {
        "model": model_name,
        "reference": "Semantic label-13 connected-component boxes; they are automatic references rather than human instance boxes.",
        "frame_stride": int(evaluation["frame_stride"]),
        "validation_frames": len(frames_by_split["validation"]),
        "blind_test_frames": len(frames_by_split["blind_test"]),
        "validation": validation,
        "selected_confidence_threshold": float(selected_threshold),
        "blind_test": blind_test,
        "entry_criteria": {
            "minimum_validation_f1": 0.50,
            "minimum_blind_test_recall": 0.50,
            "passed": bool(_metrics(validation_counts[selected_threshold])["f1"] >= 0.50 and blind_test["recall"] >= 0.50),
        },
        "limitations": [
            "COCO 的 person 类未针对当前仓储采集条件微调。",
            "参考框由语义组件自动生成，实例分离错误会影响检测评测。",
            "阈值仅依据六月下旬验证集的 F1 选择，十月结果不参与选择。",
        ],
    }
    (output_dir / "pretrained_person_detector_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a pretrained COCO person detector against semantic component references.")
    parser.add_argument("--config", default="experiments/target_motion_20260803/pretrained_person_detector.yaml")
    args = parser.parse_args()
    print(json.dumps(run(args.config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
