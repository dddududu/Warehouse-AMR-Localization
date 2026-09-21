from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn_v2

from experiments.target_motion_20260803.evaluate_pretrained_person_detector import DetectorDataset, _build_frames, _collate, _metrics, _run_split


def _load_config(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("Fine-tune config must be a mapping.")
    return payload


def _configure_trainable_parameters(model: torch.nn.Module, freeze_backbone_except_layer4: bool) -> None:
    if not freeze_backbone_except_layer4:
        return
    for parameter in model.parameters():
        parameter.requires_grad = False
    for module in (model.backbone.fpn, model.backbone.body.layer4, model.roi_heads):
        for parameter in module.parameters():
            parameter.requires_grad = True


def _best_validation(model: torch.nn.Module, frames, person_label: int, evaluation: dict[str, Any], device: torch.device) -> tuple[float, float, dict[str, Any]]:
    _, counts = _run_split(model, frames, person_label, evaluation, device, None)
    reports = {float(threshold): _metrics(value) for threshold, value in counts.items()}
    threshold = max(reports, key=lambda value: reports[value]["f1"])
    return float(reports[threshold]["f1"]), float(threshold), reports


def run(config_path: str | Path) -> dict[str, Any]:
    config = _load_config(config_path)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    training = dict(config["training"])
    evaluation = dict(config["evaluation"])
    device = torch.device(str(training["device"]) if torch.cuda.is_available() else "cpu")
    person_label = int(config["person_label"])
    frames = {
        "train": _build_frames(list(config["splits"]["train"]), int(training["train_frame_stride"])),
        "validation": _build_frames(list(config["splits"]["validation"]), int(training["validation_frame_stride"])),
        "blind_test": _build_frames(list(config["splits"]["blind_test"]), int(training["blind_test_frame_stride"])),
    }
    train_dataset = DetectorDataset(frames["train"], person_label, int(evaluation["min_component_pixels"]))
    train_loader = DataLoader(train_dataset, batch_size=int(training["batch_size"]), shuffle=True, num_workers=int(training["num_workers"]), pin_memory=device.type == "cuda", collate_fn=_collate)
    model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT).to(device)
    _configure_trainable_parameters(model, bool(config["model"]["freeze_backbone_except_layer4"]))
    optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"]))
    best: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    for epoch in range(1, int(training["epochs"]) + 1):
        model.train()
        losses = []
        for batch in train_loader:
            images = [item["image"].to(device, non_blocking=True) for item in batch]
            targets = [
                {
                    "boxes": torch.as_tensor(item["reference_boxes"], dtype=torch.float32, device=device).reshape(-1, 4),
                    "labels": torch.ones((len(item["reference_boxes"]),), dtype=torch.int64, device=device),
                }
                for item in batch
            ]
            optimizer.zero_grad(set_to_none=True)
            losses_by_name = model(images, targets)
            loss = sum(losses_by_name.values())
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        validation_f1, threshold, validation = _best_validation(model, frames["validation"], person_label, evaluation, device)
        record = {"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_f1": validation_f1, "selected_threshold": threshold, "validation": validation}
        history.append(record)
        print(json.dumps(record, ensure_ascii=False))
        if best is None or validation_f1 > float(best["validation_f1"]):
            best = {**record, "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()}}
            torch.save({"state_dict": best["state_dict"], "selected_threshold": threshold, "model": config["model"]}, output_dir / "finetuned_person_detector.pt")
    if best is None:
        raise RuntimeError("Fine-tuning did not produce a checkpoint.")
    model.load_state_dict(best["state_dict"])
    _, blind_counts = _run_split(model, frames["blind_test"], person_label, evaluation, device, None)
    blind_test = _metrics(blind_counts[float(best["selected_threshold"])])
    summary = {
        "model": "FasterRCNN_ResNet50_FPN_V2 initialized from COCO, retaining COCO person label 1.",
        "train_frames": len(frames["train"]),
        "validation_frames": len(frames["validation"]),
        "blind_test_frames": len(frames["blind_test"]),
        "best_epoch": int(best["epoch"]),
        "best_validation_f1": float(best["validation_f1"]),
        "selected_confidence_threshold": float(best["selected_threshold"]),
        "blind_test": blind_test,
        "history": [{key: value for key, value in item.items() if key != "state_dict"} for item in history],
        "entry_criteria": {"minimum_validation_f1": 0.50, "minimum_blind_test_recall": 0.50, "passed": bool(float(best["validation_f1"]) >= 0.50 and blind_test["recall"] >= 0.50)},
        "limitations": [
            "训练框由语义连通域自动生成，尚未经过人工实例核验。",
            "十月盲测只在六月下旬选定最佳周期和置信度阈值后评估一次。",
        ],
    }
    (output_dir / "finetuned_person_detector_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a COCO person detector on semantic component boxes with cross-date evaluation.")
    parser.add_argument("--config", default="experiments/target_motion_20260803/finetune_person_detector.yaml")
    args = parser.parse_args()
    print(json.dumps(run(args.config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
