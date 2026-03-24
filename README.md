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
