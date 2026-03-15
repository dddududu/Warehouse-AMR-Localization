from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from runtime_compat import ensure_runtime_compatibility

ensure_runtime_compatibility()

import torch

from models.coarse_patch_encoder import CoarsePatchEncoder
from preprocess.map_patch_builder import build_or_load_patch_cache
from retrieval.config import load_coarse_retrieval_config


def build_patch_database(config, checkpoint_path: str | Path | None = None, output_path: str | Path | None = None) -> dict:
    cfg = load_coarse_retrieval_config(config)
    patch_cache = build_or_load_patch_cache(cfg.map_path, cfg)
    patch_tensors = np.asarray(patch_cache["patch_tensors"], dtype=np.float32)
    metadata = patch_cache["metadata"]

    device = torch.device(cfg.device)
    encoder = CoarsePatchEncoder(descriptor_dim=cfg.descriptor_dim, init_seed=cfg.model_seed).to(device)
    if checkpoint_path is not None:
        state = torch.load(checkpoint_path, map_location=device)
        encoder.load_state_dict(state["patch_encoder"])
    encoder.eval()

    descriptors: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(patch_tensors), 16):
            batch = torch.from_numpy(patch_tensors[start : start + 16]).to(device)
            descriptors.append(encoder(batch).cpu().numpy())
    descriptor_bank = np.concatenate(descriptors, axis=0)

    result = {
        "patch_tensors_path": str(patch_cache["patch_cache_path"]),
        "metadata_path": str(patch_cache["metadata_path"]),
        "num_patches": int(len(metadata)),
    }
    if output_path is not None:
        output = Path(output_path)
    else:
        output = Path(cfg.cache_dir) / "descriptor_bank.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        descriptor_bank=descriptor_bank.astype(np.float32),
        patch_tensors=patch_tensors.astype(np.float32),
        patch_ids=np.arange(len(metadata), dtype=np.int32),
    )
    metadata_output = output.with_suffix(".json")
    metadata_output.write_text(
        json.dumps(
            {
                "patch_metadata": [
                    {
                        "patch_id": item.patch_id,
                        "center_xy": list(item.center_xy),
                        "bbox_xy": list(item.bbox_xy),
                        "grid_index_range": list(item.grid_index_range),
                    }
                    for item in metadata
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    result["descriptor_bank_path"] = str(output)
    result["descriptor_metadata_path"] = str(metadata_output)
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Build map patch tensors and descriptor bank.")
    parser.add_argument("--config", default="configs/coarse_retrieval_a.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-path", default=None)
    args = parser.parse_args()
    build_patch_database(args.config, checkpoint_path=args.checkpoint, output_path=args.output_path)


if __name__ == "__main__":
    main()
