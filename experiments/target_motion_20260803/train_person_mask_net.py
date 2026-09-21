from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from dataset_io.frame_indexer import build_frame_index
from experiments.target_motion_20260803.person_mask_model import PersonMaskNet


@dataclass(frozen=True)
class ImageMaskPair:
    image_path: Path
    mask_path: Path
    positive: bool


def _load_config(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("Person mask config must be a mapping.")
    return payload


def _build_pairs(roots: list[str], person_label: int) -> list[ImageMaskPair]:
    pairs: list[ImageMaskPair] = []
    for root in roots:
        for record in build_frame_index(Path(root)):
            mask = cv2.imread(str(record.segmentation_greyscale_left_path), cv2.IMREAD_UNCHANGED)
            if mask is None:
                raise RuntimeError(f"Could not read semantic mask: {record.segmentation_greyscale_left_path}")
            pairs.append(
                ImageMaskPair(
                    image_path=Path(record.image_left_path),
                    mask_path=Path(record.segmentation_greyscale_left_path),
                    positive=bool(np.any(mask == int(person_label))),
                )
            )
    return pairs


class PersonMaskDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, pairs: list[ImageMaskPair], image_hw: tuple[int, int], person_label: int, augment: bool) -> None:
        self.pairs = pairs
        self.image_hw = image_hw
        self.person_label = int(person_label)
        self.augment = bool(augment)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        pair = self.pairs[index]
        image = cv2.imread(str(pair.image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(pair.mask_path), cv2.IMREAD_UNCHANGED)
        if image is None or mask is None:
            raise RuntimeError(f"Could not read pair at index {index}.")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        target = (mask == self.person_label).astype(np.uint8)
        if self.augment and random.random() < 0.5:
            image = cv2.flip(image, 1)
            target = cv2.flip(target, 1)
        height, width = self.image_hw
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
        target = cv2.resize(target, (width, height), interpolation=cv2.INTER_NEAREST)
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1).copy()).float().div_(255.0)
        image_tensor = (image_tensor - 0.5) / 0.5
        return {"image": image_tensor, "mask": torch.from_numpy(target[None].copy()).float()}


def _dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    intersection = (probabilities * target).sum(dim=(1, 2, 3))
    total = probabilities.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return 1.0 - ((2.0 * intersection + 1.0) / (total + 1.0)).mean()


def _metrics(logits: torch.Tensor, target: torch.Tensor, threshold: float) -> tuple[int, int, int]:
    prediction = torch.sigmoid(logits) >= float(threshold)
    truth = target >= 0.5
    intersection = int((prediction & truth).sum().item())
    union = int((prediction | truth).sum().item())
    target_pixels = int(truth.sum().item())
    return intersection, union, target_pixels


def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device, threshold: float) -> dict[str, float | int]:
    model.eval()
    intersection = union = target_pixels = 0
    positive_images = recalled_images = 0
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            target = batch["mask"].to(device, non_blocking=True)
            logits = model(image)
            current_intersection, current_union, current_target_pixels = _metrics(logits, target, threshold)
            intersection += current_intersection
            union += current_union
            target_pixels += current_target_pixels
            prediction = torch.sigmoid(logits) >= float(threshold)
            target_present = target.flatten(1).sum(dim=1) > 0
            prediction_present = prediction.flatten(1).sum(dim=1) > 0
            positive_images += int(target_present.sum().item())
            recalled_images += int((target_present & prediction_present).sum().item())
    return {
        "pixel_iou": intersection / max(union, 1),
        "pixel_recall": intersection / max(target_pixels, 1),
        "positive_images": positive_images,
        "positive_image_recall": recalled_images / max(positive_images, 1),
    }


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = ["C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf"]
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def _write_examples(model: nn.Module, dataset: PersonMaskDataset, device: torch.device, output_path: Path, threshold: float) -> None:
    positive_indices = [index for index, pair in enumerate(dataset.pairs) if pair.positive]
    selected = positive_indices[:6]
    canvas = Image.new("RGB", (960, max(len(selected), 1) * 220 + 90), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((25, 25), "十月盲测：人员识别预测样例（绿：标签，红：预测）", fill="#172B4D", font=_font(28, True))
    model.eval()
    with torch.no_grad():
        for row_index, index in enumerate(selected):
            sample = dataset[index]
            logits = model(sample["image"].unsqueeze(0).to(device))
            prediction = (torch.sigmoid(logits)[0, 0].cpu().numpy() >= float(threshold))
            target = sample["mask"][0].numpy() >= 0.5
            image = ((sample["image"].permute(1, 2, 0).numpy() * 0.5 + 0.5) * 255.0).clip(0, 255).astype(np.uint8)
            overlay = image.copy()
            overlay[target] = (40, 185, 80)
            overlay[prediction] = np.where(target[prediction, None], (245, 205, 40), (225, 65, 65))
            item = Image.fromarray(overlay).resize((320, 180), resample=Image.Resampling.BILINEAR)
            y = 80 + row_index * 220
            canvas.paste(item, (25, y))
            draw.text((370, y + 35), f"样例 {row_index + 1}", fill="#334E68", font=_font(22, True))
            draw.text((370, y + 78), "绿色：真实人员区域", fill="#27864B", font=_font(20))
            draw.text((370, y + 116), "红色：仅预测区域；黄色：预测正确区域", fill="#A63D3D", font=_font(20))
    canvas.save(output_path)


def run(config_path: str | Path) -> dict[str, Any]:
    config = _load_config(config_path)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    train_cfg = dict(config["training"])
    seed = int(train_cfg["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    image_hw = (int(config["model"]["input_height"]), int(config["model"]["input_width"]))
    person_label = int(config["person_label"])
    pairs_by_split = {name: _build_pairs(list(roots), person_label) for name, roots in config["splits"].items()}
    train_pairs = pairs_by_split["train"]
    repeat = int(train_cfg["positive_sample_repeat"])
    train_pairs = train_pairs + [pair for pair in train_pairs if pair.positive for _ in range(max(repeat - 1, 0))]
    datasets = {
        "train": PersonMaskDataset(train_pairs, image_hw, person_label, augment=True),
        "validation": PersonMaskDataset(pairs_by_split["validation"], image_hw, person_label, augment=False),
        "blind_test": PersonMaskDataset(pairs_by_split["blind_test"], image_hw, person_label, augment=False),
    }
    device = torch.device(str(train_cfg["device"]) if torch.cuda.is_available() else "cpu")
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=int(train_cfg["batch_size"]), shuffle=True, num_workers=int(train_cfg["num_workers"]), pin_memory=device.type == "cuda", persistent_workers=int(train_cfg["num_workers"]) > 0),
        "validation": DataLoader(datasets["validation"], batch_size=int(train_cfg["batch_size"]), shuffle=False, num_workers=int(train_cfg["num_workers"]), pin_memory=device.type == "cuda", persistent_workers=int(train_cfg["num_workers"]) > 0),
        "blind_test": DataLoader(datasets["blind_test"], batch_size=int(train_cfg["batch_size"]), shuffle=False, num_workers=int(train_cfg["num_workers"]), pin_memory=device.type == "cuda", persistent_workers=int(train_cfg["num_workers"]) > 0),
    }
    model = PersonMaskNet(base_channels=int(config["model"]["base_channels"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg["learning_rate"]), weight_decay=float(train_cfg["weight_decay"]))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    pos_weight = torch.tensor(float(train_cfg["positive_weight"]), device=device)
    best_state: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        model.train()
        losses: list[float] = []
        for batch in loaders["train"]:
            image = batch["image"].to(device, non_blocking=True)
            target = batch["mask"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(image)
                bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
                loss = bce + float(train_cfg["dice_weight"]) * _dice_loss(logits, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        validation = _evaluate(model, loaders["validation"], device, float(train_cfg["threshold"]))
        record = {"epoch": epoch, "train_loss": float(np.mean(losses)), "validation": validation}
        history.append(record)
        print(json.dumps(record, ensure_ascii=False))
        if best_state is None or float(validation["pixel_iou"]) > float(best_state["validation"]["pixel_iou"]):
            best_state = {"epoch": epoch, "validation": validation, "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()}}
    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint.")
    model.load_state_dict(best_state["model_state"])
    checkpoint_path = output_dir / "person_mask_net.pt"
    torch.save({"model": config["model"], "person_label": person_label, "threshold": float(train_cfg["threshold"]), "state_dict": best_state["model_state"]}, checkpoint_path)
    blind_test = _evaluate(model, loaders["blind_test"], device, float(train_cfg["threshold"]))
    _write_examples(model, datasets["blind_test"], device, output_dir / "person_mask_oct12_examples.png", float(train_cfg["threshold"]))
    summary = {
        "train_images_before_oversampling": len(pairs_by_split["train"]),
        "train_images_after_oversampling": len(train_pairs),
        "validation_images": len(pairs_by_split["validation"]),
        "blind_test_images": len(pairs_by_split["blind_test"]),
        "positive_images": {name: sum(pair.positive for pair in pairs) for name, pairs in pairs_by_split.items()},
        "best_epoch": int(best_state["epoch"]),
        "best_validation": best_state["validation"],
        "blind_test": blind_test,
        "history": history,
        "limitations": [
            "监督来自已有语义掩码，评估的是对该语义标签的跨日期复现能力，不等同于人工实例检测基准。",
            "该识别器尚未接入定位主链路；其定位收益必须在后续软门控实验中单独验证。",
        ],
    }
    (output_dir / "person_mask_training_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a lightweight RGB person-mask recognizer with cross-date evaluation.")
    parser.add_argument("--config", default="experiments/target_motion_20260803/person_mask_train.yaml")
    args = parser.parse_args()
    print(json.dumps(run(args.config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
