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
