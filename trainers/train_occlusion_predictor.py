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
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from dataset_io.sequence_dataset import WarehouseSequenceDataset
from models.occlusion_predictor import StereoOcclusionPredictor


@dataclass(frozen=True)
class OcclusionPredictorTrainConfig:
    coarse_config_path: str
    output_checkpoint: str
    cache_dir: str = "./cache/occlusion_predictor"
    image_resize_hw: tuple[int, int] = (96, 160)
    occlusion_threshold: float = 0.05
    positive_sample_weight: float = 12.0
    train_batch_size: int = 32
    train_num_workers: int = 0
    train_epochs: int = 8
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    hidden_dim: int = 32
    device: str = "cpu"
    use_amp: bool = False
    model_seed: int = 0
    max_train_samples: int | None = None
    max_val_samples: int | None = None

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> "OcclusionPredictorTrainConfig":
        payload = dict(mapping)
        resize_hw = payload.get("image_resize_hw")
        if resize_hw is not None:
            payload["image_resize_hw"] = (int(resize_hw[0]), int(resize_hw[1]))
        return cls(**payload)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "OcclusionPredictorTrainConfig":
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(payload, dict):
            raise ValueError("Occlusion predictor config must parse to a mapping.")
        return cls.from_mapping(payload)


def _load_coarse_entries(coarse_config_path: str | Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    from retrieval.config import load_coarse_retrieval_config

    coarse_cfg = load_coarse_retrieval_config(coarse_config_path)
    return coarse_cfg.split_sequence_entries()


def _label_ratio(left_path: str | Path | None, right_path: str | Path | None) -> float:
    ratios = []
    for path in (left_path, right_path):
        if path is None:
            continue
        label = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if label is not None:
            ratios.append(float((label == 13).mean()))
    return max(ratios, default=0.0)


class OcclusionRatioDataset(Dataset):
    def __init__(
        self,
        entries: list[dict[str, str]],
        resize_hw: tuple[int, int],
        max_samples: int | None = None,
    ) -> None:
        self.resize_hw = (int(resize_hw[0]), int(resize_hw[1]))
        self.samples: list[dict[str, object]] = []
        for entry in entries:
            dataset = WarehouseSequenceDataset(
                sequence_root=entry["sequence_root"],
                calibration_path=entry["calibration_path"],
                config={"load_lidar": False},
            )
            for record in dataset.frame_index:
                self.samples.append(
                    {
                        "image_left_path": record.image_left_path,
                        "image_right_path": record.image_right_path,
                        "ratio": _label_ratio(
                            record.segmentation_greyscale_left_path,
                            record.segmentation_greyscale_right_path,
                        ),
                        "sequence_name": entry["sequence_name"],
                    }
                )
        if max_samples is not None and len(self.samples) > int(max_samples):
            self.samples = self.samples[: int(max_samples)]

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def ratios(self) -> np.ndarray:
        return np.asarray([float(item["ratio"]) for item in self.samples], dtype=np.float32)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = self.samples[index]
        stereo = []
        for key in ("image_left_path", "image_right_path"):
            image = cv2.imread(str(sample[key]), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(sample[key])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(
                image,
                (self.resize_hw[1], self.resize_hw[0]),
                interpolation=cv2.INTER_AREA,
            )
            image = np.transpose(image.astype(np.float32) / 255.0, (2, 0, 1))
            stereo.append(image)
        ratio = float(sample["ratio"])
        return {
            "stereo_image": torch.from_numpy(np.concatenate(stereo, axis=0)),
            "ratio": torch.tensor(ratio, dtype=torch.float32),
            "sequence_name": str(sample["sequence_name"]),
        }


def _evaluate(
    model: StereoOcclusionPredictor,
    dataset: OcclusionRatioDataset,
    device: torch.device,
    threshold: float,
) -> dict[str, float | int | None]:
    if len(dataset) == 0:
        return {"num_frames": 0}
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    model.eval()
    predictions = []
    targets = []
    with torch.no_grad():
        for batch in loader:
            images = batch["stereo_image"].to(device).float()
            target = batch["ratio"].to(device).float()
            output = model(images)
            predictions.append(output["ratio"].detach().cpu().numpy())
            targets.append(target.detach().cpu().numpy())
    pred = np.concatenate(predictions).astype(np.float64)
    target_np = np.concatenate(targets).astype(np.float64)
    pred_pos = pred >= float(threshold)
    target_pos = target_np >= float(threshold)
    true_pos = int(np.count_nonzero(pred_pos & target_pos))
    false_pos = int(np.count_nonzero(pred_pos & ~target_pos))
    false_neg = int(np.count_nonzero(~pred_pos & target_pos))
    return {
        "num_frames": int(target_np.shape[0]),
        "mae": float(np.mean(np.abs(pred - target_np))),
        "rmse": float(np.sqrt(np.mean((pred - target_np) ** 2))),
        "precision": float(true_pos / max(1, true_pos + false_pos)),
        "recall": float(true_pos / max(1, true_pos + false_neg)),
        "num_positive_target": int(np.count_nonzero(target_pos)),
        "num_positive_pred": int(np.count_nonzero(pred_pos)),
    }


def train_occlusion_predictor(config_path: str | Path) -> dict[str, object]:
    cfg = OcclusionPredictorTrainConfig.from_yaml(config_path)
    torch.manual_seed(int(cfg.model_seed))
    train_entries, val_entries = _load_coarse_entries(cfg.coarse_config_path)
    train_dataset = OcclusionRatioDataset(train_entries, cfg.image_resize_hw, cfg.max_train_samples)
    val_dataset = OcclusionRatioDataset(val_entries, cfg.image_resize_hw, cfg.max_val_samples)
    ratios = train_dataset.ratios
    sample_weights = np.ones(len(train_dataset), dtype=np.float64)
    sample_weights[ratios >= float(cfg.occlusion_threshold)] = float(cfg.positive_sample_weight)
    sampler = WeightedRandomSampler(sample_weights.tolist(), num_samples=len(sample_weights), replacement=True)
    device = torch.device(cfg.device)
    amp_enabled = bool(cfg.use_amp and device.type == "cuda")
    loader = DataLoader(
        train_dataset,
        batch_size=int(cfg.train_batch_size),
        sampler=sampler,
        num_workers=max(0, int(cfg.train_num_workers)),
        pin_memory=device.type == "cuda",
    )
    model = StereoOcclusionPredictor(hidden_dim=int(cfg.hidden_dim)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.learning_rate), weight_decay=float(cfg.weight_decay))
    scaler = torch.amp.GradScaler(device=device.type, enabled=amp_enabled)
    history = []
    best_state = None
    best_score = -1.0
    pos_weight = torch.tensor([float(cfg.positive_sample_weight)], device=device)
    for epoch_idx in range(int(cfg.train_epochs)):
        model.train()
        losses = []
        for batch in loader:
            images = batch["stereo_image"].to(device).float()
            target_ratio = batch["ratio"].to(device).float()
            target_cls = (target_ratio >= float(cfg.occlusion_threshold)).float()
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                output = model(images)
                ratio_loss = F.smooth_l1_loss(output["ratio"], target_ratio)
                cls_loss = F.binary_cross_entropy_with_logits(
                    output["occlusion_logit"],
                    target_cls,
                    pos_weight=pos_weight,
                )
                loss = ratio_loss + 0.25 * cls_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.item()))
        metrics = _evaluate(model, val_dataset, device, threshold=float(cfg.occlusion_threshold))
        report = {
            "epoch": epoch_idx + 1,
            "train_loss_mean": float(np.mean(losses)) if losses else None,
            "validation": metrics,
        }
        history.append(report)
        score = float(metrics.get("recall") or 0.0) - 0.1 * float(metrics.get("mae") or 0.0)
        if score >= best_score:
            best_score = score
            best_state = {
                "model": model.state_dict(),
                "config": cfg.__dict__,
                "history": history.copy(),
            }
        print(json.dumps(report, ensure_ascii=False))
    output_path = Path(cfg.output_checkpoint)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state if best_state is not None else {"model": model.state_dict(), "config": cfg.__dict__}, output_path)
    result = {
        "checkpoint_path": str(output_path),
        "num_train_frames": int(len(train_dataset)),
        "num_val_frames": int(len(val_dataset)),
        "history": history,
        "train_sequences": [item["sequence_name"] for item in train_entries],
        "val_sequences": [item["sequence_name"] for item in val_entries],
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train deployable stereo occlusion predictor.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    train_occlusion_predictor(args.config)


if __name__ == "__main__":
    main()
