from experiments.target_motion_20260803.finetune_person_detector import _configure_trainable_parameters
from experiments.target_motion_20260803.evaluate_pretrained_person_detector import _build_model


def test_finetune_keeps_roi_head_trainable() -> None:
    model, _ = _build_model({"name": "fasterrcnn_resnet50_fpn_v2"}, device="cpu")
    _configure_trainable_parameters(model, True)
    assert any(parameter.requires_grad for parameter in model.roi_heads.parameters())
    assert not any(parameter.requires_grad for parameter in model.backbone.body.layer1.parameters())
