from __future__ import annotations

import torch

from models.coarse_patch_encoder import CoarsePatchEncoder
from models.coarse_query_encoder import CoarseQueryEncoder
from trainers.train_coarse_retrieval import train_coarse_retrieval


def test_coarse_encoders_output_normalized_descriptors() -> None:
    query_encoder = CoarseQueryEncoder(descriptor_dim=256, init_seed=0)
    patch_encoder = CoarsePatchEncoder(descriptor_dim=256, init_seed=0)
    bev = torch.randn(2, 4, 200, 200)

    query_descriptor = query_encoder(bev)
    patch_descriptor = patch_encoder(bev)
    assert query_descriptor.shape == (2, 256)
    assert patch_descriptor.shape == (2, 256)
    assert torch.allclose(torch.linalg.norm(query_descriptor, dim=1), torch.ones(2), atol=1e-5)


def test_train_script_runs_on_minimal_sample(synthetic_dataset, tmp_path) -> None:
    result = train_coarse_retrieval(
        {
            "sequence_root": str(synthetic_dataset["sequence_root"]),
            "calibration_path": str(synthetic_dataset["calibration_path"]),
            "map_path": str(synthetic_dataset["map_path"]),
            "cache_dir": str(tmp_path / "cache"),
            "train_batch_size": 1,
            "train_epochs": 1,
            "num_hard_negative_patches": 0,
            "num_random_negative_patches": 0,
            "val_ratio": 0.0,
        },
        output_checkpoint=tmp_path / "model.pt",
    )
    assert result["checkpoint_path"].endswith("model.pt")


def test_train_script_supports_patch_classifier(synthetic_dataset, tmp_path) -> None:
    result = train_coarse_retrieval(
        {
            "sequence_root": str(synthetic_dataset["sequence_root"]),
            "calibration_path": str(synthetic_dataset["calibration_path"]),
            "map_path": str(synthetic_dataset["map_path"]),
            "cache_dir": str(tmp_path / "cache_classifier"),
            "train_batch_size": 1,
            "train_epochs": 1,
            "num_hard_negative_patches": 0,
            "num_random_negative_patches": 0,
            "use_patch_classification_loss": True,
            "classification_loss_weight": 0.2,
            "classifier_score_weight": 0.2,
            "val_ratio": 0.0,
        },
        output_checkpoint=tmp_path / "model_classifier.pt",
    )
    assert result["checkpoint_path"].endswith("model_classifier.pt")
