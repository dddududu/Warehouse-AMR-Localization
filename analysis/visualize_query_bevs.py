from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analysis.visualize_map_patches import save_bev_channel_images
from dataset_io.lidar_loader import load_pcd_xyz
from dataset_io.sequence_dataset import WarehouseSequenceDataset
from preprocess.bev_builder import BEVConfig, points_to_bev
from preprocess.local_lidar_cropper import LocalCropConfig, crop_local_lidar_points
from retrieval.config import load_coarse_retrieval_config


def visualize_query_bevs(
    config,
    frame_indices: list[int],
    output_dir: str | Path = "outputs/query_bevs",
    prefix: str = "query",
) -> dict:
    cfg = load_coarse_retrieval_config(config)
    dataset = WarehouseSequenceDataset(
        sequence_root=cfg.sequence_root,
        calibration_path=cfg.calibration_path,
        config={"load_lidar": False},
    )
    crop_config = LocalCropConfig(
        x_min=cfg.crop_x_min,
        x_max=cfg.crop_x_max,
        y_min=cfg.crop_y_min,
        y_max=cfg.crop_y_max,
        z_min=cfg.crop_z_min,
        z_max=cfg.crop_z_max,
    )
    bev_config = BEVConfig(
        x_min=cfg.crop_x_min,
        x_max=cfg.crop_x_max,
        y_min=cfg.crop_y_min,
        y_max=cfg.crop_y_max,
        resolution=cfg.bev_resolution,
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    frames: list[dict] = []
    saved_files: list[str] = []
    for frame_idx in frame_indices:
        lidar_path = dataset.frame_index[frame_idx].lidar_path
        points = load_pcd_xyz(lidar_path)
        cropped = crop_local_lidar_points(points, crop_config)
        bev = points_to_bev(cropped, bev_config)

        frame_dir = output_path / f"{prefix}_{frame_idx:06d}"
        frame_dir.mkdir(parents=True, exist_ok=True)
        frame_files = save_bev_channel_images(bev, output_dir=frame_dir, prefix="")
        saved_files.extend(frame_files)
        frames.append(
            {
                "frame_idx": int(frame_idx),
                "timestamp": float(dataset.frame_times[frame_idx]),
                "lidar_path": str(lidar_path),
                "num_input_points": int(points.shape[0]),
                "num_cropped_points": int(cropped.shape[0]),
                "saved_files": frame_files,
            }
        )

    report = {
        "num_frames": len(frame_indices),
        "frame_indices": [int(index) for index in frame_indices],
        "saved_file_count": len(saved_files),
        "output_dir": str(output_path),
        "frames": frames,
    }
    (output_path / f"{prefix}_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Render query LiDAR BEV tensors for selected frames.")
    parser.add_argument("--config", default="configs/coarse_retrieval_a.yaml")
    parser.add_argument("--output-dir", default="outputs/query_bevs")
    parser.add_argument("--prefix", default="query")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=10)
    args = parser.parse_args()

    frame_indices = list(range(args.start_frame, args.start_frame + args.num_frames))
    visualize_query_bevs(
        args.config,
        frame_indices=frame_indices,
        output_dir=args.output_dir,
        prefix=args.prefix,
    )


if __name__ == "__main__":
    main()
