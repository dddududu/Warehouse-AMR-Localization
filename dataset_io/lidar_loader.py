from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class PCDHeader:
    fields: list[str]
    size: list[int]
    point_types: list[str]
    count: list[int]
    width: int
    height: int
    points: int
    data: str
    data_offset: int


def read_pcd_header(pcd_path: str | Path) -> PCDHeader:
    path = Path(pcd_path)
    fields: dict[str, str] = {}
    data_offset = 0
    with path.open("rb") as file_obj:
        while True:
            line = file_obj.readline()
            if not line:
                raise ValueError(f"PCD header is incomplete: {path}")
            data_offset += len(line)
            decoded = line.decode("ascii").strip()
            if not decoded or decoded.startswith("#"):
                continue
            parts = decoded.split()
            key = parts[0].upper()
            fields[key] = " ".join(parts[1:])
            if key == "DATA":
                break

    return PCDHeader(
        fields=fields.get("FIELDS", "").split(),
        size=[int(v) for v in fields.get("SIZE", "").split()],
        point_types=fields.get("TYPE", "").split(),
        count=[int(v) for v in fields.get("COUNT", "").split()],
        width=int(fields.get("WIDTH", "0")),
        height=int(fields.get("HEIGHT", "1")),
        points=int(fields.get("POINTS", fields.get("WIDTH", "0"))),
        data=fields.get("DATA", "").strip().lower(),
        data_offset=data_offset,
    )


def load_pcd_xyz(pcd_path: str | Path) -> np.ndarray:
    path = Path(pcd_path)
    header = read_pcd_header(path)
    if header.data != "binary":
        raise ValueError(f"Only binary PCD is supported, got {header.data!r}.")
    if header.fields != ["x", "y", "z"]:
        raise ValueError(f"Expected PCD fields ['x', 'y', 'z'], got {header.fields}.")
    if header.size != [4, 4, 4] or header.point_types != ["F", "F", "F"] or header.count != [1, 1, 1]:
        raise ValueError("PCD field specification must be float32 xyz.")

    expected_bytes = header.points * 3 * 4
    with path.open("rb") as file_obj:
        file_obj.seek(header.data_offset)
        buffer = file_obj.read(expected_bytes)
    if len(buffer) != expected_bytes:
        raise ValueError(
            f"PCD binary payload length mismatch: expected {expected_bytes} bytes, got {len(buffer)}."
        )
    points = np.frombuffer(buffer, dtype=np.float32)
    return points.reshape(header.points, 3).copy()

