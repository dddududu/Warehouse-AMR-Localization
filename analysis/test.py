# 检查法向量分布
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


_PLY_TO_NUMPY = {
    "char": "i1",
    "uchar": "u1",
    "short": "i2",
    "ushort": "u2",
    "int": "i4",
    "uint": "u4",
    "float": "f4",
    "double": "f8",
}


def _cache_path(map_path: Path, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = hashlib.md5(
        f"{map_path.resolve()}::{map_path.stat().st_mtime_ns}::{map_path.stat().st_size}".encode("utf-8")
    ).hexdigest()
    return cache_dir / f"map_{fingerprint}.npz"


def load_map_vertices(map_path: str | Path, cache_dir: str | Path | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    path = Path(map_path)
    if cache_dir is not None:
        cache_file = _cache_path(path, Path(cache_dir))
        if cache_file.is_file():
            cached = np.load(cache_file)
            normals = cached["normals"] if "normals" in cached.files else None
            return cached["xyz"], normals
    else:
        cache_file = None

    properties: list[tuple[str, str]] = []
    vertex_count = None
    data_offset = 0
    in_vertex_block = False
    with path.open("rb") as file_obj:
        while True:
            line = file_obj.readline()
            if not line:
                raise ValueError(f"PLY header is incomplete: {path}")
            data_offset += len(line)
            decoded = line.decode("ascii").strip()
            if decoded.startswith("element vertex"):
                vertex_count = int(decoded.split()[2])
                in_vertex_block = True
                continue
            if decoded.startswith("element ") and not decoded.startswith("element vertex"):
                in_vertex_block = False
                continue
            if in_vertex_block and decoded.startswith("property "):
                _, dtype_name, prop_name = decoded.split()
                if dtype_name not in _PLY_TO_NUMPY:
                    raise ValueError(f"Unsupported PLY property type: {dtype_name}")
                properties.append((prop_name, _PLY_TO_NUMPY[dtype_name]))
            if decoded == "end_header":
                break

    if vertex_count is None:
        raise ValueError(f"PLY vertex element is missing: {path}")
    dtype = np.dtype(properties)
    with path.open("rb") as file_obj:
        file_obj.seek(data_offset)
        vertices = np.fromfile(file_obj, dtype=dtype, count=vertex_count)

    xyz = np.column_stack((vertices["x"], vertices["y"], vertices["z"])).astype(np.float32)
    normals = None
    if {"nx", "ny", "nz"}.issubset(vertices.dtype.names or ()):
        normals = np.column_stack((vertices["nx"], vertices["ny"], vertices["nz"])).astype(np.float32)
    if cache_file is not None:
        payload = {"xyz": xyz}
        if normals is not None:
            payload["normals"] = normals
        np.savez_compressed(cache_file, **payload)
    return xyz, normals


def analyze_map(map_path: str | Path, cache_dir: str | Path | None = None, output_json: str | Path | None = None) -> dict:
    xyz, normals = load_map_vertices(map_path, cache_dir=cache_dir)
    report = {
        "map_path": str(Path(map_path)),
        "point_count": int(xyz.shape[0]),
        "xyz_min": xyz.min(axis=0).tolist(),
        "xyz_max": xyz.max(axis=0).tolist(),
        "bbox_extent": (xyz.max(axis=0) - xyz.min(axis=0)).tolist(),
        "has_normals": normals is not None,
    }
    if normals is not None:
        report["normal_norm_mean"] = float(np.linalg.norm(normals, axis=1).mean())

    if output_json is not None:
        output_path = Path(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    return report


xyz, normals = load_map_vertices("D:\TorWIC\TorWIC SLAM Dataset\Jun. 15, 2022\Aisle_CCW_Run_1\Aisle_CCW_Run_1\groundtruth_map.ply")
if normals is not None:
    print("法向量均值:", np.mean(normals, axis=0))
    print("法向量模长均值:", np.linalg.norm(normals, axis=1).mean())