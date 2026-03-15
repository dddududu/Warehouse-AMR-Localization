from __future__ import annotations

import argparse
import json
from pathlib import Path

from runtime_compat import ensure_runtime_compatibility

ensure_runtime_compatibility()

import torch
from torch.utils.data import DataLoader

from dataset_io.retrieval_dataset import CoarseRetrievalDataset
from losses.retrieval_loss import RetrievalInfoNCELoss
from models.coarse_retrieval_model import CoarseRetrievalModel
from retrieval.config import load_coarse_retrieval_config


def train_coarse_retrieval(config, output_checkpoint: str | Path | None = None) -> dict:
    cfg = load_coarse_retrieval_config(config)
    dataset = CoarseRetrievalDataset(cfg)
    dataloader = DataLoader(dataset, batch_size=cfg.train_batch_size, shuffle=True)
    device = torch.device(cfg.device)

    model = CoarseRetrievalModel(descriptor_dim=cfg.descriptor_dim, init_seed=cfg.model_seed).to(device)
    criterion = RetrievalInfoNCELoss(temperature=cfg.temperature)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    history: list[float] = []
    model.train()
    for _ in range(cfg.train_epochs):
        for batch in dataloader:
            query_bev = batch["query_bev"].to(device).float()
            positive_patch_bev = batch["positive_patch_bev"].to(device).float()
            negative_patch_bevs = batch["negative_patch_bevs"].to(device).float()

            outputs = model(query_bev, positive_patch_bev, negative_patch_bevs)
            loss = criterion(
                outputs["query_descriptor"],
                outputs["positive_descriptor"],
                outputs["negative_descriptor"],
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            history.append(float(loss.item()))

    if output_checkpoint is None:
        checkpoint_path = Path(cfg.cache_dir) / "coarse_retrieval_model.pt"
    else:
        checkpoint_path = Path(output_checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "query_encoder": model.query_encoder.state_dict(),
            "patch_encoder": model.patch_encoder.state_dict(),
            "history": history,
            "config": cfg.__dict__,
        },
        checkpoint_path,
    )
    result = {
        "checkpoint_path": str(checkpoint_path),
        "num_steps": len(history),
        "final_loss": history[-1] if history else None,
    }
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the coarse retrieval model.")
    parser.add_argument("--config", default="configs/coarse_retrieval_a.yaml")
    parser.add_argument("--output-checkpoint", default=None)
    args = parser.parse_args()
    train_coarse_retrieval(args.config, output_checkpoint=args.output_checkpoint)


if __name__ == "__main__":
    main()
