from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from analysis.analyze_map import load_map_vertices
from preprocess.bev_builder import BEVConfig, points_to_bev
from retrieval.config import CoarseRetrievalConfig


@dataclass(frozen=True)
class PatchMetadata:
    patch_id: int
    center_xy: tuple[float, float]
    bbox_xy: tuple[float, float, float, float]
    grid_index_range: tuple[int, int, int, int]


def _aligned_bounds(values: np.ndarray, resolution: float) -> tuple[float, float]:
    epsilon = resolution * 1.0e-3
    return (
        float(np.floor((values.min() - epsilon) / resolution) * resolution),
        float(np.ceil((values.max() + epsilon) / resolution) * resolution),
    )


def build_global_bev(points_xyz: np.ndarray, resolution: float) -> tuple[np.ndarray, BEVConfig]:
    points = np.asarray(points_xyz, dtype=np.float32)
    x_min, x_max = _aligned_bounds(points[:, 0], resolution)
    y_min, y_max = _aligned_bounds(points[:, 1], resolution)
    config = BEVConfig(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, resolution=resolution)
    return points_to_bev(points, config), config


def extract_patch_tensors(
    global_bev: np.ndarray,
    global_config: BEVConfig,
    patch_size_m: float,
    patch_stride_m: float,
) -> tuple[np.ndarray, list[PatchMetadata]]:
    patch_size_cells = int(round(patch_size_m / global_config.resolution))
    patch_stride_cells = int(round(patch_stride_m / global_config.resolution))
    _, height, width = global_bev.shape
    if patch_size_cells > height or patch_size_cells > width:
        raise ValueError("Patch size is larger than the global BEV dimensions.")

    patch_tensors: list[np.ndarray] = []
    metadata: list[PatchMetadata] = []
    patch_id = 0
    for row_start in range(0, height - patch_size_cells + 1, patch_stride_cells):
        row_end = row_start + patch_size_cells
        for col_start in range(0, width - patch_size_cells + 1, patch_stride_cells):
            col_end = col_start + patch_size_cells
            patch = global_bev[:, row_start:row_end, col_start:col_end]
            x_min = global_config.x_min + col_start * global_config.resolution
            x_max = x_min + patch_size_m
            y_max = global_config.y_max - row_start * global_config.resolution
            y_min = y_max - patch_size_m
            metadata.append(
                PatchMetadata(
                    patch_id=patch_id,
                    center_xy=((x_min + x_max) / 2.0, (y_min + y_max) / 2.0),
                    bbox_xy=(x_min, x_max, y_min, y_max),
                    grid_index_range=(row_start, row_end, col_start, col_end),
                )
            )
            patch_tensors.append(patch.astype(np.float32, copy=True))
            patch_id += 1
    return np.stack(patch_tensors, axis=0), metadata


def find_patch_ids_covering_xy(metadata: list[PatchMetadata], x: float, y: float) -> list[int]:
    matches: list[int] = []
    for item in metadata:
        x_min, x_max, y_min, y_max = item.bbox_xy
        if x_min <= x <= x_max and y_min <= y <= y_max:
            matches.append(item.patch_id)
    return matches


def choose_gt_patch_id(metadata: list[PatchMetadata], x: float, y: float) -> int:
    candidate_ids = find_patch_ids_covering_xy(metadata, x, y)
    if candidate_ids:
        best_patch = min(
            candidate_ids,
            key=lambda idx: (metadata[idx].center_xy[0] - x) ** 2 + (metadata[idx].center_xy[1] - y) ** 2,
        )
        return int(best_patch)
    return int(
        min(
            metadata,
            key=lambda item: (item.center_xy[0] - x) ** 2 + (item.center_xy[1] - y) ** 2,
        ).patch_id
    )


def _cache_key(map_path: Path, config: CoarseRetrievalConfig) -> str:
    payload = {
        "map_path": str(map_path.resolve()),
        "mtime_ns": map_path.stat().st_mtime_ns,
        "size": map_path.stat().st_size,
        "resolution": config.bev_resolution,
        "patch_size_m": config.patch_size_m,
        "patch_stride_m": config.patch_stride_m,
    }
    return hashlib.md5(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def build_or_load_patch_cache(
    map_path: str | Path,
    config: CoarseRetrievalConfig,
) -> dict[str, object]:
    map_file = Path(map_path)
    cache_dir = Path(config.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _cache_key(map_file, config)
    patch_path = cache_dir / f"patch_tensors_{key}.npz"
    metadata_path = cache_dir / f"patch_metadata_{key}.json"
    global_bev_path = cache_dir / f"global_bev_{key}.npz"

    if patch_path.is_file() and metadata_path.is_file() and global_bev_path.is_file():
        patch_cache = np.load(patch_path)
        global_cache = np.load(global_bev_path)
        metadata_raw = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata = [PatchMetadata(**item) for item in metadata_raw]
        global_config = BEVConfig(
            x_min=float(global_cache["x_min"]),
            x_max=float(global_cache["x_max"]),
            y_min=float(global_cache["y_min"]),
            y_max=float(global_cache["y_max"]),
            resolution=float(global_cache["resolution"]),
        )
        return {
            "patch_tensors": patch_cache["patch_tensors"],
            "metadata": metadata,
            "global_bev": global_cache["global_bev"],
            "global_config": global_config,
            "patch_cache_path": patch_path,
            "metadata_path": metadata_path,
        }

    xyz, _ = load_map_vertices(map_file, cache_dir=cache_dir / "map_vertices")
    global_bev, global_config = build_global_bev(xyz, resolution=config.bev_resolution)
    patch_tensors, metadata = extract_patch_tensors(
        global_bev,
        global_config,
        patch_size_m=config.patch_size_m,
        patch_stride_m=config.patch_stride_m,
    )

    np.savez_compressed(patch_path, patch_tensors=patch_tensors)
    np.savez_compressed(
        global_bev_path,
        global_bev=global_bev,
        x_min=np.array(global_config.x_min, dtype=np.float32),
        x_max=np.array(global_config.x_max, dtype=np.float32),
        y_min=np.array(global_config.y_min, dtype=np.float32),
        y_max=np.array(global_config.y_max, dtype=np.float32),
        resolution=np.array(global_config.resolution, dtype=np.float32),
    )
    metadata_path.write_text(json.dumps([asdict(item) for item in metadata], indent=2), encoding="utf-8")
    return {
        "patch_tensors": patch_tensors,
        "metadata": metadata,
        "global_bev": global_bev,
        "global_config": global_config,
        "patch_cache_path": patch_path,
        "metadata_path": metadata_path,
    }
