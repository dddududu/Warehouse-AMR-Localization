from __future__ import annotations

from preprocess.map_patch_builder import build_or_load_patch_cache, choose_gt_patch_id
from retrieval.config import CoarseRetrievalConfig


def test_map_patch_builder_creates_fixed_size_patch_cache(synthetic_dataset, tmp_path) -> None:
    config = CoarseRetrievalConfig(
        sequence_root=str(synthetic_dataset["sequence_root"]),
        calibration_path=str(synthetic_dataset["calibration_path"]),
        map_path=str(synthetic_dataset["map_path"]),
        cache_dir=str(tmp_path / "cache"),
    )
    patch_cache = build_or_load_patch_cache(synthetic_dataset["map_path"], config)
    assert patch_cache["patch_tensors"].shape[1:] == (4, 200, 200)
    assert len(patch_cache["metadata"]) == 1
    assert choose_gt_patch_id(patch_cache["metadata"], 0.0, 0.0) == 0
    assert patch_cache["patch_cache_path"].is_file()

