from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class IMUSampleWindow:
    timestamps: np.ndarray
    gyro: np.ndarray
    accel: np.ndarray


class IMULoader:
    def __init__(self, timestamps: np.ndarray, gyro: np.ndarray, accel: np.ndarray) -> None:
        self.timestamps = np.asarray(timestamps, dtype=np.float64)
        self.gyro = np.asarray(gyro, dtype=np.float64)
        self.accel = np.asarray(accel, dtype=np.float64)
        if self.timestamps.ndim != 1:
            raise ValueError("IMU timestamps must be 1-D.")
        if self.gyro.shape != (len(self.timestamps), 3):
            raise ValueError("IMU gyro array must have shape (N, 3).")
        if self.accel.shape != (len(self.timestamps), 3):
            raise ValueError("IMU accel array must have shape (N, 3).")

    @classmethod
    def from_file(cls, imu_path: str | Path) -> "IMULoader":
        data = np.loadtxt(Path(imu_path), dtype=np.float64, ndmin=2)
        if data.ndim != 2 or data.shape[1] != 7:
            raise ValueError(f"Expected IMU file with shape (N, 7), got {data.shape}.")
        return cls(
            timestamps=data[:, 0],
            gyro=data[:, 1:4],
            accel=data[:, 4:7],
        )

    def __len__(self) -> int:
        return int(self.timestamps.shape[0])

    def estimate_frequency_hz(self) -> float | None:
        if len(self.timestamps) < 2:
            return None
        deltas = np.diff(self.timestamps)
        positive = deltas[deltas > 0.0]
        if positive.size == 0:
            return None
        return float(1.0 / np.median(positive))

    def query_range(self, start_time: float, end_time: float) -> IMUSampleWindow:
        left = int(np.searchsorted(self.timestamps, start_time, side="left"))
        right = int(np.searchsorted(self.timestamps, end_time, side="right"))
        return IMUSampleWindow(
            timestamps=self.timestamps[left:right].copy(),
            gyro=self.gyro[left:right].copy(),
            accel=self.accel[left:right].copy(),
        )

    def query_time_window(self, center_time: float, window_sec: float) -> IMUSampleWindow:
        half_window = float(window_sec) / 2.0
        return self.query_range(center_time - half_window, center_time + half_window)
