from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class LearningExamples:
    cells: np.ndarray
    features: np.ndarray
    targets: np.ndarray
    heuristic_stability: np.ndarray
    crossday_observations: np.ndarray


def encode_cells(cells: np.ndarray) -> np.ndarray:
    indices = np.asarray(cells, dtype=np.int64)
    return (indices[:, 0] + 100_000) * 1_000_000 + (indices[:, 1] + 100_000)


def heuristic_stability(counts: np.ndarray, minimum_observations: float = 6.0) -> np.ndarray:
    counts = np.asarray(counts, dtype=np.float64)
    observations = np.maximum(counts[:, 0], 1.0)
    semantic_reliability = (counts[:, 1] + 0.5 * counts[:, 2]) / observations
    observation_confidence = 1.0 - np.exp(-observations / float(minimum_observations))
    return (semantic_reliability * observation_confidence).astype(np.float32)


def future_reliability_target(initial_counts: np.ndarray, crossday_counts: np.ndarray, minimum_observations: float = 6.0) -> np.ndarray:
    initial_counts = np.asarray(initial_counts, dtype=np.float64)
    crossday_counts = np.asarray(crossday_counts, dtype=np.float64)
    initial_observations = np.maximum(initial_counts[:, 0], 1.0)
    future_observations = np.maximum(crossday_counts[:, 0], 1.0)
    initial_distribution = initial_counts[:, 1:] / initial_observations[:, None]
    future_distribution = crossday_counts[:, 1:] / future_observations[:, None]
    semantic_reliability = future_distribution[:, 0] + 0.5 * future_distribution[:, 1]
    support = 1.0 - np.exp(-future_observations / float(minimum_observations))
    agreement = np.sum(initial_distribution * future_distribution, axis=1)
    return (semantic_reliability * support * agreement).astype(np.float32)


def _cell_feature(counts: np.ndarray, max_log_observations: float) -> np.ndarray:
    observations = float(counts[0])
    if observations <= 0.0:
        return np.zeros(7, dtype=np.float32)
    probabilities = counts[1:] / observations
    entropy = -float(np.sum(probabilities * np.log(np.maximum(probabilities, 1.0e-8)))) / np.log(3.0)
    return np.asarray(
        [
            1.0,
            np.log1p(observations) / max_log_observations,
            probabilities[0],
            probabilities[1],
            probabilities[2],
            float(heuristic_stability(np.asarray([counts]))[0]),
            entropy,
        ],
        dtype=np.float32,
    )


def build_feature_patches(cells: np.ndarray, counts: np.ndarray, radius: int) -> np.ndarray:
    cells = np.asarray(cells, dtype=np.int32)
    counts = np.asarray(counts, dtype=np.float64)
    if radius < 0:
        raise ValueError("radius must be non-negative")
    lookup = {tuple(cell): row for cell, row in zip(cells, counts)}
    side = 2 * radius + 1
    output = np.zeros((len(cells), 7, side, side), dtype=np.float32)
    max_log_observations = max(float(np.log1p(np.max(counts[:, 0]))), 1.0)
    for sample_index, cell in enumerate(cells):
        for offset_x in range(-radius, radius + 1):
            for offset_y in range(-radius, radius + 1):
                neighbor = lookup.get((int(cell[0] + offset_x), int(cell[1] + offset_y)))
                if neighbor is None:
                    continue
                output[sample_index, :, offset_x + radius, offset_y + radius] = _cell_feature(neighbor, max_log_observations)
    return output


def build_learning_examples(
    map_payload: np.lib.npyio.NpzFile,
    radius: int = 4,
    minimum_initial_observations: int = 2,
    minimum_crossday_observations: int = 2,
) -> LearningExamples:
    initial_cells = np.asarray(map_payload["initial_cells"], dtype=np.int32)
    initial_counts = np.asarray(map_payload["initial_counts"], dtype=np.float64)
    updated_cells = np.asarray(map_payload["updated_cells"], dtype=np.int32)
    updated_counts = np.asarray(map_payload["updated_counts"], dtype=np.float64)
    updated_lookup = {tuple(cell): counts for cell, counts in zip(updated_cells, updated_counts)}
    selected_indices: list[int] = []
    crossday_rows: list[np.ndarray] = []
    for index, (cell, before) in enumerate(zip(initial_cells, initial_counts)):
        after = updated_lookup.get(tuple(cell))
        if after is None:
            continue
        crossday = np.maximum(after - before, 0.0)
        if before[0] < minimum_initial_observations or crossday[0] < minimum_crossday_observations:
            continue
        selected_indices.append(index)
        crossday_rows.append(crossday)
    selected = np.asarray(selected_indices, dtype=np.int64)
    selected_cells = initial_cells[selected]
    selected_counts = initial_counts[selected]
    crossday_counts = np.asarray(crossday_rows, dtype=np.float64)
    return LearningExamples(
        cells=selected_cells,
        features=build_feature_patches(selected_cells, selected_counts, radius),
        targets=future_reliability_target(selected_counts, crossday_counts),
        heuristic_stability=heuristic_stability(selected_counts),
        crossday_observations=crossday_counts[:, 0].astype(np.float32),
    )


def spatial_split(cells: np.ndarray, block_size_cells: int = 8, validation_group: int = 1) -> tuple[np.ndarray, np.ndarray]:
    cells = np.asarray(cells, dtype=np.int64)
    block_ids = np.floor_divide(cells[:, 0], int(block_size_cells)) + 3 * np.floor_divide(cells[:, 1], int(block_size_cells))
    validation = np.mod(block_ids, 5) == int(validation_group)
    return ~validation, validation


def regression_metrics(predictions: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    residuals = predictions - targets
    correlation = 0.0 if predictions.size < 2 or np.std(predictions) < 1.0e-8 or np.std(targets) < 1.0e-8 else float(np.corrcoef(predictions, targets)[0, 1])
    return {
        "mae": float(np.mean(np.abs(residuals))),
        "rmse": float(np.sqrt(np.mean(np.square(residuals)))),
        "pearson": correlation,
    }


class StabilityMapNet(nn.Module):
    def __init__(self, input_channels: int = 7) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 24, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 24, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 12, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(12, 1), nn.Sigmoid())

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(features)).squeeze(-1)
