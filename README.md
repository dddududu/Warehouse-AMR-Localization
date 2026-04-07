# TorWIC Coarse Retrieval

LiDAR coarse retrieval pipeline for the TorWIC warehouse dataset.

## Recommended training setup

- Multi-day training config: `configs/coarse_retrieval_multiday.yaml`
- Fast multi-day fine-tune config: `configs/coarse_retrieval_multiday_finetune_e2.yaml`
- Oct. 12 Aisle-CCW evaluation config: `configs/eval_oct12_aisle_ccw_multiday.yaml`
- Local-rerank best training config: `configs/coarse_retrieval_localmatcher_stage3_hardrerank_e1.yaml`
- Local-rerank best eval config: `configs/eval_oct12_aisle_ccw_localmatcher_hardrerank.yaml`
- Stride-10 classifier-head training config: `configs/coarse_retrieval_classifier_head_e2_stride10.yaml`
- Stride-10 best eval config: `configs/eval_oct12_aisle_ccw_classifier_head_stride10.yaml`
- Shared map/calibration follow the `Jun. 15, 2022` assets as requested.

## What changed

- Supports explicit `sequence_entries`, so different dates can be mixed in one training run.
- Shares query/patch encoder weights when enabled to reduce overfitting.
- Adds query-side BEV augmentation for better cross-date generalization.
- Adds optional patch-classification supervision and score fusion to improve top-1 retrieval.
- Adds a local spatial matcher and second-stage reranking path for harder near-neighbor disambiguation.

## Train

```bash
python -m trainers.train_coarse_retrieval --config configs/coarse_retrieval_multiday.yaml --output-checkpoint outputs/coarse_retrieval_multiday_best.pt
```

```bash
python -m trainers.train_coarse_retrieval --config configs/coarse_retrieval_multiday_finetune_e2.yaml --output-checkpoint outputs/coarse_retrieval_multiday_finetune_e2.pt
```

```bash
python -m trainers.train_coarse_retrieval --config configs/coarse_retrieval_localmatcher_stage3_hardrerank_e1.yaml --output-checkpoint outputs/coarse_retrieval_localmatcher_stage3_hardrerank_e1.pt
```

```bash
python -m trainers.train_coarse_retrieval --config configs/coarse_retrieval_classifier_head_e2_stride10.yaml --output-checkpoint outputs/coarse_retrieval_classifier_head_e2_stride10.pt
```

## Evaluate

```bash
python -m retrieval.evaluate_retrieval --config configs/eval_oct12_aisle_ccw_multiday.yaml --checkpoint outputs/coarse_retrieval_multiday_best.pt --output-json outputs/eval_oct12_aisle_ccw_multiday.json
```

```bash
python -m retrieval.evaluate_retrieval --config configs/eval_oct12_aisle_ccw_localmatcher_hardrerank.yaml --descriptor-bank outputs/descriptor_bank_oct12_aisle_ccw_top3_deg30.npz --checkpoint checkpoints/coarse_retrieval_localmatcher_stage3_hardrerank_e1_top1_0p4011_r3_0p6645.pt --output-json outputs/eval_oct12_aisle_ccw_localmatcher_stage3_hardrerank_best.json
```

```bash
python -m retrieval.evaluate_retrieval --config configs/eval_oct12_aisle_ccw_classifier_head_stride10.yaml --checkpoint checkpoints/coarse_retrieval_classifier_head_e2_stride10_top1_0p4426_r3_0p7672.pt --output-json outputs/eval_oct12_aisle_ccw_classifier_head_stride10.json
```

## Fine Localization

```bash
python -m localization.fine_localizer --config configs/fine_localization_oct12_aisle_ccw.yaml --frame-start 0 --num-frames 50 --output-json outputs/fine_localization_oct12_aisle_ccw_50f.json
```

```bash
python -m localization.fine_localizer --config configs/fine_localization_oct12_aisle_cw.yaml --frame-start 0 --num-frames 50 --output-json outputs/fine_localization_oct12_aisle_cw_50f.json
```

- Fine localization pipeline: `topK patch retrieval -> local submap BEV correlation -> ICP refinement`
- Trajectory mode adds temporal consistency reranking with constant-velocity prediction
- Full-sequence mode adds Viterbi sequence smoothing over top-k fine-localization candidates
- Validated on `Oct. 12, 2022 / Aisle_CCW` first 50 frames:
  - mean position error: `0.1784 m`
  - median position error: `0.1793 m`
  - mean yaw error: `0.6855 deg`
- Validated on `Oct. 12, 2022 / Aisle_CW` first 50 frames:
  - mean position error: `0.1449 m`
  - median position error: `0.1432 m`
  - mean yaw error: `0.4504 deg`
- Full `Oct. 12, 2022 / Aisle_CCW` with sequence smoothing:
  - mean position error: `2.6486 m`
  - median position error: `0.1645 m`
  - mean yaw error: `39.4893 deg`
  - frames below `1m`: `63.28%`
- Full `Oct. 12, 2022 / Aisle_CW` with sequence smoothing:
  - mean position error: `2.0649 m`
  - median position error: `0.1388 m`
  - mean yaw error: `29.4770 deg`
  - frames below `1m`: `65.20%`

## Deep Fine Localization (Experimental)

```bash
python -m trainers.train_fine_pose_matcher --config configs/fine_pose_matcher_train.yaml --output-checkpoint outputs/fine_pose_matcher_v1.pt
```

```bash
python -m localization.deep_fine_localizer --config configs/fine_localization_oct12_aisle_ccw_deep_v6_guidedbev_online.yaml --frame-start 0 --num-frames 50 --output-json outputs/fine_localization_oct12_aisle_ccw_deep_v6_guidedbev_online_50f.json
```

- Model path: `models/fine_pose_matcher.py`
- Training dataset: `dataset_io/fine_localization_dataset.py`
- First version uses `query BEV + candidate submap BEV -> match score + relative pose -> ICP`
- Current `Aisle_CCW` 50-frame result with `outputs/fine_pose_matcher_v1.pt` is not yet better than the geometric pipeline:
  - mean position error: `1.9704 m`
  - median position error: `1.9645 m`
  - mean yaw error: `50.4744 deg`
- V2 switches to `topK` candidate-set classification plus pose residual prediction:
  - training script still uses `trainers/train_fine_pose_matcher.py`
  - current best validation candidate classification accuracy: `66.73%`
  - current `Aisle_CCW` 50-frame deep-dominant result is still below the geometric baseline:
    - `outputs/fine_localization_oct12_aisle_ccw_deep_v2_50f.json`
    - mean position error: `1.9640 m`
    - mean yaw error: `60.7309 deg`
- V3 switches the pose head to `x/y/yaw` bin classification + residual decoding:
  - current validation candidate classification accuracy: `50.16%`
  - current `Aisle_CCW` 50-frame result:
    - `outputs/fine_localization_oct12_aisle_ccw_deep_v3_50f.json`
    - mean position error: `10.6707 m`
    - mean yaw error: `53.8335 deg`
- Stereo query-image branch is also implemented and trained:
  - checkpoint: `outputs/fine_pose_matcher_v4_stereo.pt`
  - current best validation candidate classification accuracy: `51.33%`
  - this is lower than the monocular V4 peak `69.75%`, so stereo is not promoted to the default branch yet
- Current retained deep configs are:
  - baseline online config: `configs/fine_localization_oct12_aisle_ccw_deep_v6_guidedbev_online.yaml`
  - `CCW` best deep config: `configs/fine_localization_oct12_aisle_ccw_deep_v6_guidedbev_pairresolver_guarded.yaml`
  - baseline online `CW` config: `configs/fine_localization_oct12_aisle_cw_deep_v6_guidedbev_online.yaml`
  - `CW` best deep config is the tracked `tracker_init` branch:
    - `configs/fine_localization_oct12_aisle_cw_deep_v6_guidedbev_pairresolver_corepatch_v2_trackerinit.yaml`
  - logic: `deep candidate scoring + deep-guided BEV initialization + ICP`, then apply online stabilization; `CW` best branch also adds `tracker_init`
  - full `Aisle_CCW` best deep result:
    - `outputs/fine_localization_oct12_aisle_ccw_deep_v6_guidedbev_pairresolver_guarded_full.json`
    - mean position error: `5.0596 m`
    - median position error: `3.4096 m`
    - frames below `1m`: `38.91%`
  - full `Aisle_CW` best deep result:
    - `outputs/fine_localization_oct12_aisle_cw_deep_v6_guidedbev_pairresolver_corepatch_v2_trackerinit_full.json`
    - mean position error: `0.3879 m`
    - median position error: `0.0752 m`
    - mean yaw error: `0.9824 deg`
    - frames below `1m`: `98.17%`

## Best checkpoint

- Repository path: `checkpoints/coarse_retrieval_classifier_head_e2_stride10_top1_0p4426_r3_0p7672.pt`
- GitHub Release: `best-checkpoint-top1-0.4426`
- Matching report: `analysis/multiday_finetune_results.md`

## Best system

- Eval config: `configs/eval_oct12_aisle_ccw_classifier_head_stride10.yaml`
- Overall best metrics on `Oct. 12, 2022 / Aisle_CCW`: `Top1=0.4426`, `Recall@3=0.7672`, `MRR=0.6463`

## Best ensemble under stride-10 patches

- Eval config: `configs/eval_oct12_aisle_ccw_multiday_ensemble_stride10.yaml`
- Metrics on `Oct. 12, 2022 / Aisle_CCW`: `Top1=0.4087`, `Recall@3=0.7945`, `MRR=0.6310`
