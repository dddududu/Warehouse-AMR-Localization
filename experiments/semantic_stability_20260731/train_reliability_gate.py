from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dataset_io.sequence_dataset import WarehouseSequenceDataset
from localization.config import load_fine_localization_config
from models.reliability_gate import RELIABILITY_GATE_SOURCES, ReliabilityGate
from preprocess.dynamic_point_filter import semantic_label_ratio
from retrieval.config import load_coarse_retrieval_config


@dataclass(frozen=True)
class SourceConfig:
    name: str
    result_json: Path
    fine_config: Path


def _load_yaml(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("Reliability-gate config must be a mapping.")
    return payload


def _sources(raw_sources: list[dict[str, Any]]) -> list[SourceConfig]:
    return [
        SourceConfig(
            name=str(item["name"]),
            result_json=Path(item["result_json"]),
            fine_config=Path(item["fine_config"]),
        )
        for item in raw_sources
    ]


def _dataset_for_source(source: SourceConfig) -> WarehouseSequenceDataset:
    fine_config = load_fine_localization_config(source.fine_config)
    coarse_config = load_coarse_retrieval_config(fine_config.coarse_config_path)
    entry = coarse_config.resolve_single_sequence_entry(prefer_validation=True)
    return WarehouseSequenceDataset(
        sequence_root=entry["sequence_root"],
        calibration_path=entry["calibration_path"],
        config={"load_lidar": False},
    )


def _semantic_features(dataset: WarehouseSequenceDataset, frame_idx: int) -> np.ndarray:
    record = dataset.frame_index[int(frame_idx)]
    return np.asarray(
        [
            semantic_label_ratio(
                record.segmentation_greyscale_left_path,
                record.segmentation_greyscale_right_path,
                (13,),
            ),
            semantic_label_ratio(
                record.segmentation_greyscale_left_path,
                record.segmentation_greyscale_right_path,
                (12, 13, 14, 15),
            ),
            semantic_label_ratio(
                record.segmentation_greyscale_left_path,
                record.segmentation_greyscale_right_path,
                (5, 7, 9, 10, 11),
            ),
        ],
        dtype=np.float32,
    )


def _source_name(selected_init_source: str | None) -> str:
    if selected_init_source == "deep_score_tracker_pose":
        return "tracker_init"
    if selected_init_source in RELIABILITY_GATE_SOURCES:
        return str(selected_init_source)
    return "bev_init"


def _source_pose(candidate: dict[str, Any], source: str) -> np.ndarray | None:
    raw_pose = candidate.get(f"{source}_pose_4x4")
    if raw_pose is None:
        return None
    pose = np.asarray(raw_pose, dtype=np.float64)
    return pose if pose.shape == (4, 4) else None


def collect_samples(
    sources: list[SourceConfig],
    keep_baseline_margin_m: float,
    include_semantic_features: bool = True,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for source in sources:
        report = json.loads(source.result_json.read_text(encoding="utf-8"))
        dataset = _dataset_for_source(source)
        for frame in report.get("frame_results", []):
            candidate_results = frame.get("candidate_results", [])
            selected_index = int(frame.get("selected_candidate_index", 0))
            if not 0 <= selected_index < len(candidate_results):
                continue
            candidate = candidate_results[selected_index]
            base_features = np.asarray(candidate.get("reliability_gate_features"), dtype=np.float32)
            if base_features.shape not in {(16,), (19,)} or not np.all(np.isfinite(base_features)):
                continue
            frame_idx = int(frame["frame_idx"])
            features = (
                np.concatenate((base_features, _semantic_features(dataset, frame_idx)), axis=0)
                if include_semantic_features and base_features.shape == (16,)
                else base_features
            )
            gt_pose = dataset.ground_truth.poses_4x4[frame_idx]
            source_errors: dict[str, float] = {}
            available_sources: list[str] = []
            for init_source in RELIABILITY_GATE_SOURCES:
                pose = _source_pose(candidate, init_source)
                if pose is None:
                    continue
                source_errors[init_source] = float(np.linalg.norm(pose[:2, 3] - gt_pose[:2, 3]))
                available_sources.append(init_source)
            if not source_errors:
                continue
            baseline_source = _source_name(candidate.get("selected_init_source"))
            if baseline_source not in source_errors:
                baseline_source = "bev_init"
            oracle_source = min(source_errors, key=source_errors.get)
            label_source = oracle_source
            if (
                source_errors[baseline_source] - source_errors[oracle_source]
                < float(keep_baseline_margin_m)
            ):
                label_source = baseline_source
            samples.append(
                {
                    "split_source": source.name,
                    "frame_idx": frame_idx,
                    "features": features,
                    "available_sources": available_sources,
                    "baseline_source": baseline_source,
                    "oracle_source": oracle_source,
                    "label_source": label_source,
                    "baseline_error_m": source_errors[baseline_source],
                    "oracle_error_m": source_errors[oracle_source],
                    "bev_error_m": source_errors.get("bev_init"),
                    "deep_error_m": source_errors.get("deep_init"),
                    "tracker_error_m": source_errors.get("tracker_init"),
                }
            )
    return samples


def _array(samples: list[dict[str, Any]], field: str) -> np.ndarray:
    return np.asarray([sample[field] for sample in samples], dtype=np.float64)


def _available_mask(samples: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray(
        [
            [float(source in sample["available_sources"]) for source in RELIABILITY_GATE_SOURCES]
            for sample in samples
        ],
        dtype=np.float32,
    )


def _masked_logits(logits: torch.Tensor, availability: torch.Tensor) -> torch.Tensor:
    return logits.masked_fill(availability <= 0.0, -1.0e9)


def _metrics(samples: list[dict[str, Any]], decisions: list[str]) -> dict[str, Any]:
    errors = []
    baseline_errors = []
    oracle_errors = []
    correct = []
    for sample, decision in zip(samples, decisions, strict=True):
        error_key = f"{decision.split('_')[0]}_error_m" if decision != "tracker_init" else "tracker_error_m"
        error = sample.get(error_key)
        if error is None:
            error = sample["baseline_error_m"]
        errors.append(float(error))
        baseline_errors.append(float(sample["baseline_error_m"]))
        oracle_errors.append(float(sample["oracle_error_m"]))
        correct.append(decision == sample["label_source"])
    error_values = np.asarray(errors, dtype=np.float64)
    baseline_values = np.asarray(baseline_errors, dtype=np.float64)
    oracle_values = np.asarray(oracle_errors, dtype=np.float64)
    return {
        "num_samples": int(error_values.size),
        "action_accuracy": float(np.mean(correct)),
        "mean_position_error_m": float(error_values.mean()),
        "median_position_error_m": float(np.median(error_values)),
        "p95_position_error_m": float(np.percentile(error_values, 95)),
        "below_0p5m": float(np.mean(error_values < 0.5)),
        "baseline_mean_position_error_m": float(baseline_values.mean()),
        "oracle_mean_position_error_m": float(oracle_values.mean()),
        "mean_improvement_over_baseline_m": float(baseline_values.mean() - error_values.mean()),
        "oracle_headroom_m": float(baseline_values.mean() - oracle_values.mean()),
    }


def _choose_actions(
    probabilities: np.ndarray,
    samples: list[dict[str, Any]],
    confidence_min: float,
) -> tuple[list[str], np.ndarray]:
    actions: list[str] = []
    confidences: list[float] = []
    for probability, sample in zip(probabilities, samples, strict=True):
        available_indices = [
            index
            for index, source in enumerate(RELIABILITY_GATE_SOURCES)
            if source in sample["available_sources"]
        ]
        selected_index = max(available_indices, key=lambda index: float(probability[index]))
        confidence = float(probability[selected_index])
        actions.append(
            RELIABILITY_GATE_SOURCES[selected_index]
            if confidence >= confidence_min
            else str(sample["baseline_source"])
        )
        confidences.append(confidence)
    return actions, np.asarray(confidences, dtype=np.float64)


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def _draw_summary(
    output_path: Path,
    train_history: list[dict[str, float]],
    validation: dict[str, Any],
) -> None:
    canvas = Image.new("RGB", (1280, 620), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = _font(30, True)
    body_font = _font(22)
    small_font = _font(18)
    draw.text((54, 34), "实验三：学习式初始化可靠性门控", font=title_font, fill="#1d2939")
    left, top, width, height = 70, 120, 610, 380
    draw.rounded_rectangle((left, top, left + width, top + height), radius=14, outline="#cbd5e1", width=2)
    losses = np.asarray([item["train_loss"] for item in train_history], dtype=np.float64)
    val_losses = np.asarray([item["val_loss"] for item in train_history], dtype=np.float64)
    max_loss = max(float(losses.max(initial=1.0)), float(val_losses.max(initial=1.0)), 1.0e-6)
    for values, color in ((losses, "#2563eb"), (val_losses, "#dc2626")):
        points = []
        for index, value in enumerate(values):
            x = left + 45 + (width - 80) * index / max(1, len(values) - 1)
            y = top + height - 45 - (height - 85) * float(value) / max_loss
            points.append((x, y))
        if len(points) > 1:
            draw.line(points, fill=color, width=3)
    draw.text((left + 45, top + 18), "蓝：训练损失   红：验证损失", font=small_font, fill="#475467")
    draw.text((left + 15, top + height - 30), "训练轮次", font=small_font, fill="#475467")
    right = 755
    draw.rounded_rectangle((right, top, 1210, top + height), radius=14, outline="#cbd5e1", width=2)
    rows = [
        ("基线候选内误差", validation["baseline_mean_position_error_m"]),
        ("学习门控误差", validation["mean_position_error_m"]),
        ("候选内 oracle", validation["oracle_mean_position_error_m"]),
        ("门控动作准确率", validation["action_accuracy"]),
        ("门控触发比例", validation["gate_apply_rate"]),
    ]
    draw.text((right + 30, top + 22), "跨日期验证（Jun.23）", font=body_font, fill="#1d2939")
    for index, (label, value) in enumerate(rows):
        y = top + 85 + index * 55
        text = f"{label}：{value * 100:.2f}%" if "率" in label or "比例" in label else f"{label}：{value:.4f} m"
        draw.text((right + 30, y), text, font=body_font, fill="#344054")
    draw.text((70, 545), "仅用六月训练；十月数据未参与标签生成、归一化、早停或阈值选择。", font=body_font, fill="#475467")
    canvas.save(output_path)


def _write_rows(output_path: Path, samples: list[dict[str, Any]], probabilities: np.ndarray, actions: list[str]) -> None:
    fields = [
        "split_source", "frame_idx", "baseline_source", "oracle_source", "label_source",
        "gate_action", "gate_confidence", "baseline_error_m", "oracle_error_m",
        "bev_error_m", "deep_error_m", "tracker_error_m",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for sample, probability, action in zip(samples, probabilities, actions, strict=True):
            available_indices = [
                index for index, source in enumerate(RELIABILITY_GATE_SOURCES)
                if source in sample["available_sources"]
            ]
            writer.writerow(
                {
                    **{field: sample.get(field) for field in fields if field in sample},
                    "gate_action": action,
                    "gate_confidence": max(float(probability[index]) for index in available_indices),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a learned gate for BEV/deep/tracker ICP initializations.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = _load_yaml(args.config)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = int(config.get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(config.get("device", "cuda"))
    margin = float(config.get("keep_baseline_margin_m", 0.03))
    confidence_min = float(config.get("confidence_min", 0.55))
    train_samples = collect_samples(_sources(config["train_sources"]), margin)
    val_samples = collect_samples(_sources(config["val_sources"]), margin)
    if not train_samples or not val_samples:
        raise ValueError("Counterfactual reports must contain both train and validation samples.")
    train_features = np.stack([sample["features"] for sample in train_samples]).astype(np.float32)
    val_features = np.stack([sample["features"] for sample in val_samples]).astype(np.float32)
    feature_mean = train_features.mean(axis=0)
    feature_std = np.maximum(train_features.std(axis=0), 1.0e-6)
    train_labels = np.asarray([RELIABILITY_GATE_SOURCES.index(sample["label_source"]) for sample in train_samples], dtype=np.int64)
    val_labels = np.asarray([RELIABILITY_GATE_SOURCES.index(sample["label_source"]) for sample in val_samples], dtype=np.int64)
    train_masks = _available_mask(train_samples)
    val_masks = _available_mask(val_samples)
    class_counts = np.bincount(train_labels, minlength=len(RELIABILITY_GATE_SOURCES)).astype(np.float32)
    class_weights = class_counts.sum() / np.maximum(class_counts, 1.0)
    class_weights = class_weights / class_weights.mean()
    model = ReliabilityGate(feature_dim=train_features.shape[1], hidden_dim=int(config.get("hidden_dim", 32))).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("learning_rate", 0.002)),
        weight_decay=float(config.get("weight_decay", 1.0e-4)),
    )
    criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(class_weights).to(device))
    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy((train_features - feature_mean) / feature_std),
            torch.from_numpy(train_labels),
            torch.from_numpy(train_masks),
        ),
        batch_size=int(config.get("batch_size", 64)),
        shuffle=True,
    )
    val_tensor = torch.from_numpy((val_features - feature_mean) / feature_std).to(device)
    val_label_tensor = torch.from_numpy(val_labels).to(device)
    val_mask_tensor = torch.from_numpy(val_masks).to(device)
    best_state = None
    best_epoch = 0
    best_val_loss = float("inf")
    patience = int(config.get("early_stop_patience", 30))
    stale_epochs = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, int(config.get("epochs", 200)) + 1):
        model.train()
        train_losses = []
        for features, labels, availability in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = _masked_logits(model(features.to(device)), availability.to(device))
            loss = criterion(logits, labels.to(device))
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            val_logits = _masked_logits(model(val_tensor), val_mask_tensor)
            val_loss = float(criterion(val_logits, val_label_tensor).detach().cpu())
        history.append({"epoch": float(epoch), "train_loss": float(np.mean(train_losses)), "val_loss": val_loss})
        if val_loss < best_val_loss - 1.0e-5:
            best_val_loss = val_loss
            best_epoch = epoch
            stale_epochs = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break
    if best_state is None:
        raise RuntimeError("Reliability gate did not produce a checkpoint.")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_probabilities = torch.softmax(
            _masked_logits(model(val_tensor), val_mask_tensor), dim=1
        ).cpu().numpy()
        train_probabilities = torch.softmax(
            _masked_logits(
                model(torch.from_numpy((train_features - feature_mean) / feature_std).to(device)),
                torch.from_numpy(train_masks).to(device),
            ),
            dim=1,
        ).cpu().numpy()
    val_actions, val_confidences = _choose_actions(val_probabilities, val_samples, confidence_min)
    train_actions, _ = _choose_actions(train_probabilities, train_samples, confidence_min)
    validation = _metrics(val_samples, val_actions)
    validation["gate_apply_rate"] = float(np.mean(val_confidences >= confidence_min))
    training = _metrics(train_samples, train_actions)
    training["gate_apply_rate"] = float(np.mean(np.max(train_probabilities, axis=1) >= confidence_min))
    checkpoint_path = output_dir / "reliability_gate.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "config": {"feature_dim": int(train_features.shape[1]), "hidden_dim": int(config.get("hidden_dim", 32))},
            "feature_mean": feature_mean.tolist(),
            "feature_std": feature_std.tolist(),
            "sources": {
                "train": [source.name for source in _sources(config["train_sources"])],
                "val": [source.name for source in _sources(config["val_sources"])],
            },
            "best_epoch": int(best_epoch),
            "best_val_loss": float(best_val_loss),
        },
        checkpoint_path,
    )
    _write_rows(output_dir / "reliability_gate_validation_rows.csv", val_samples, val_probabilities, val_actions)
    _draw_summary(output_dir / "reliability_gate_training_summary.png", history, validation)
    summary = {
        "config_path": str(Path(args.config)),
        "checkpoint_path": str(checkpoint_path),
        "sources": {
            "train": [source.name for source in _sources(config["train_sources"])],
            "val": [source.name for source in _sources(config["val_sources"])],
        },
        "keep_baseline_margin_m": margin,
        "confidence_min": confidence_min,
        "class_counts": {source: int(class_counts[index]) for index, source in enumerate(RELIABILITY_GATE_SOURCES)},
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "training": training,
        "validation": validation,
        "history": history,
    }
    (output_dir / "reliability_gate_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
