from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analysis.visualize_map_patches import save_bev_channel_images
from preprocess.map_patch_builder import build_or_load_patch_cache
from retrieval.config import load_coarse_retrieval_config


def visualize_global_map_bev(
    config,
    output_dir: str | Path = "outputs/global_map_bev",
    prefix: str = "global_map",
) -> dict:
    cfg = load_coarse_retrieval_config(config)
    patch_cache = build_or_load_patch_cache(cfg.map_path, cfg)
    global_bev = np.asarray(patch_cache["global_bev"], dtype=np.float32)
    global_config = patch_cache["global_config"]

    saved_files = save_bev_channel_images(global_bev, output_dir=output_dir, prefix=prefix)
    report = {
        "map_path": str(cfg.map_path),
        "global_bev_shape": [int(dim) for dim in global_bev.shape],
        "bev_bounds_xy": {
            "x_min": float(global_config.x_min),
            "x_max": float(global_config.x_max),
            "y_min": float(global_config.y_min),
            "y_max": float(global_config.y_max),
            "resolution": float(global_config.resolution),
        },
        "saved_files": saved_files,
    }
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / f"{prefix}_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the complete global map BEV as four channel images.")
    parser.add_argument("--config", default="configs/coarse_retrieval_a.yaml")
    parser.add_argument("--output-dir", default="outputs/global_map_bev")
    parser.add_argument("--prefix", default="global_map")
    args = parser.parse_args()
    visualize_global_map_bev(args.config, output_dir=args.output_dir, prefix=args.prefix)


if __name__ == "__main__":
    main()
