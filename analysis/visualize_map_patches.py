from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from preprocess.map_patch_builder import PatchMetadata, build_or_load_patch_cache
from retrieval.config import load_coarse_retrieval_config


CHANNEL_NAMES = (
    "point_count",
    "max_height",
    "mean_height",
    "occupancy",
)


def _round_key(value: float) -> float:
    return round(float(value), 6)


def _build_patch_layout(metadata: list[PatchMetadata]) -> tuple[dict[int, tuple[int, int]], int, int]:
    unique_x = sorted({_round_key(item.center_xy[0]) for item in metadata})
    unique_y = sorted({_round_key(item.center_xy[1]) for item in metadata}, reverse=True)
    x_to_col = {value: index for index, value in enumerate(unique_x)}
    y_to_row = {value: index for index, value in enumerate(unique_y)}

    positions: dict[int, tuple[int, int]] = {}
    for item in metadata:
        positions[item.patch_id] = (
            y_to_row[_round_key(item.center_xy[1])],
            x_to_col[_round_key(item.center_xy[0])],
        )
    return positions, len(unique_y), len(unique_x)


def _normalize_channel(channel_grid: np.ndarray) -> np.ndarray:
    values = np.asarray(channel_grid, dtype=np.float32)
    finite = np.isfinite(values)
    if not np.any(finite):
        return np.zeros(values.shape, dtype=np.uint8)
    values = values.copy()
    values[~finite] = 0.0
    vmin = float(values.min())
    vmax = float(values.max())
    if vmax <= vmin:
        return np.zeros(values.shape, dtype=np.uint8)
    scaled = (values - vmin) / (vmax - vmin)
    return np.clip(scaled * 255.0, 0.0, 255.0).astype(np.uint8)


def save_bev_channel_images(
    bev_tensor: np.ndarray,
    output_dir: str | Path,
    prefix: str,
    channel_names: tuple[str, ...] = CHANNEL_NAMES,
) -> list[str]:
    tensor = np.asarray(bev_tensor, dtype=np.float32)
    if tensor.ndim != 3:
        raise ValueError(f"Expected bev_tensor with shape (C, H, W), got {tensor.shape}.")
    if tensor.shape[0] != len(channel_names):
        raise ValueError(
            f"Channel count mismatch: tensor has {tensor.shape[0]} channels, "
            f"but {len(channel_names)} names were provided."
        )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    saved_files: list[str] = []
    for channel_index, channel_name in enumerate(channel_names):
        image = Image.fromarray(_normalize_channel(tensor[channel_index]), mode="L")
        file_name = f"{prefix}_{channel_name}.png" if prefix else f"{channel_name}.png"
        image_path = output_path / file_name
        image.save(image_path)
        saved_files.append(str(image_path))
    return saved_files


def render_patch_channel_grids(
    patch_tensors: np.ndarray,
    metadata: list[PatchMetadata],
    output_dir: str | Path,
    prefix: str = "map_patches",
) -> dict:
    tensors = np.asarray(patch_tensors, dtype=np.float32)
    if tensors.ndim != 4:
        raise ValueError(f"Expected patch_tensors with shape (N, C, H, W), got {tensors.shape}.")
    if tensors.shape[1] != 4:
        raise ValueError(f"Expected 4 channels, got {tensors.shape[1]}.")
    if len(metadata) != tensors.shape[0]:
        raise ValueError("Patch metadata length must match patch tensor count.")

    positions, num_rows, num_cols = _build_patch_layout(metadata)
    _, _, patch_height, patch_width = tensors.shape
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    saved_files: list[str] = []
    for channel_index, channel_name in enumerate(CHANNEL_NAMES):
        canvas = np.zeros((num_rows * patch_height, num_cols * patch_width), dtype=np.float32)
        for patch_id, (row, col) in positions.items():
            row_start = row * patch_height
            col_start = col * patch_width
            canvas[row_start : row_start + patch_height, col_start : col_start + patch_width] = tensors[
                patch_id,
                channel_index,
            ]
        image = Image.fromarray(_normalize_channel(canvas), mode="L")
        image_path = output_path / f"{prefix}_{channel_name}.png"
        image.save(image_path)
        saved_files.append(str(image_path))

    report = {
        "num_patches": int(tensors.shape[0]),
        "grid_rows": int(num_rows),
        "grid_cols": int(num_cols),
        "patch_shape": [int(patch_height), int(patch_width)],
        "saved_files": saved_files,
    }
    (output_path / f"{prefix}_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def render_individual_patch_channels(
    patch_tensors: np.ndarray,
    metadata: list[PatchMetadata],
    output_dir: str | Path,
    prefix: str = "map_patch",
) -> dict:
    tensors = np.asarray(patch_tensors, dtype=np.float32)
    if tensors.ndim != 4:
        raise ValueError(f"Expected patch_tensors with shape (N, C, H, W), got {tensors.shape}.")
    if tensors.shape[1] != 4:
        raise ValueError(f"Expected 4 channels, got {tensors.shape[1]}.")
    if len(metadata) != tensors.shape[0]:
        raise ValueError("Patch metadata length must match patch tensor count.")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    saved_files: list[str] = []
    for item in metadata:
        patch_dir = output_path / f"{prefix}_{item.patch_id:04d}"
        patch_dir.mkdir(parents=True, exist_ok=True)
        saved_files.extend(
            save_bev_channel_images(
                tensors[item.patch_id],
                output_dir=patch_dir,
                prefix="",
            )
        )

    report = {
        "num_patches": int(tensors.shape[0]),
        "channels_per_patch": len(CHANNEL_NAMES),
        "saved_file_count": len(saved_files),
        "output_dir": str(output_path),
        "saved_files": saved_files,
    }
    (output_path / f"{prefix}_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def visualize_map_patches(
    config,
    output_dir: str | Path = "outputs/map_patch_grids",
    prefix: str = "map_patches",
    export_individual: bool = False,
) -> dict:
    cfg = load_coarse_retrieval_config(config)
    patch_cache = build_or_load_patch_cache(cfg.map_path, cfg)
    patch_tensors = np.asarray(patch_cache["patch_tensors"], dtype=np.float32)
    metadata = patch_cache["metadata"]
    if export_individual:
        return render_individual_patch_channels(patch_tensors, metadata, output_dir=output_dir, prefix=prefix)
    return render_patch_channel_grids(patch_tensors, metadata, output_dir=output_dir, prefix=prefix)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render all map patches as one grid image per BEV channel.")
    parser.add_argument("--config", default="configs/coarse_retrieval_a.yaml")
    parser.add_argument("--output-dir", default="outputs/map_patch_grids")
    parser.add_argument("--prefix", default="map_patches")
    parser.add_argument("--export-individual", action="store_true")
    args = parser.parse_args()
    visualize_map_patches(
        args.config,
        output_dir=args.output_dir,
        prefix=args.prefix,
        export_individual=args.export_individual,
    )


if __name__ == "__main__":
    main()
