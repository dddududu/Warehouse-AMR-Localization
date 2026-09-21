import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "learned_stability_map_20260911"))

from stability_map_learning import build_feature_patches, future_reliability_target, spatial_split


def test_future_reliability_rewards_consistent_static_observations() -> None:
    initial = np.asarray([[10, 10, 0, 0], [10, 0, 0, 10]], dtype=np.float64)
    future = np.asarray([[10, 10, 0, 0], [10, 0, 0, 10]], dtype=np.float64)
    targets = future_reliability_target(initial, future)
    assert targets[0] > 0.7
    assert targets[0] > targets[1]


def test_feature_patches_preserve_center_and_spatial_split_is_disjoint() -> None:
    cells = np.asarray([[0, 0], [1, 0], [10, 0]], dtype=np.int32)
    counts = np.asarray([[5, 5, 0, 0], [5, 0, 5, 0], [5, 0, 0, 5]], dtype=np.float64)
    patches = build_feature_patches(cells, counts, radius=1)
    assert patches.shape == (3, 7, 3, 3)
    assert patches[0, 0, 1, 1] == 1.0
    train_mask, validation_mask = spatial_split(cells, block_size_cells=2, validation_group=0)
    assert not np.any(train_mask & validation_mask)
    assert np.all(train_mask | validation_mask)
