import torch

from experiments.target_motion_20260803.person_mask_model import PersonMaskNet


def test_person_mask_net_preserves_spatial_shape() -> None:
    model = PersonMaskNet(base_channels=8)
    outputs = model(torch.zeros((2, 3, 96, 160)))
    assert outputs.shape == (2, 1, 96, 160)
