# TorWIC Coarse Retrieval

LiDAR coarse retrieval pipeline for the TorWIC warehouse dataset.

## Recommended training setup

- Multi-day training config: `configs/coarse_retrieval_multiday.yaml`
- Fast multi-day fine-tune config: `configs/coarse_retrieval_multiday_finetune_e2.yaml`
- Oct. 12 Aisle-CCW evaluation config: `configs/eval_oct12_aisle_ccw_multiday.yaml`
- Shared map/calibration follow the `Jun. 15, 2022` assets as requested.

## What changed

- Supports explicit `sequence_entries`, so different dates can be mixed in one training run.
- Shares query/patch encoder weights when enabled to reduce overfitting.
- Adds query-side BEV augmentation for better cross-date generalization.
- Adds optional patch-classification supervision and score fusion to improve top-1 retrieval.

## Train

```bash
python -m trainers.train_coarse_retrieval --config configs/coarse_retrieval_multiday.yaml --output-checkpoint outputs/coarse_retrieval_multiday_best.pt
```

```bash
python -m trainers.train_coarse_retrieval --config configs/coarse_retrieval_multiday_finetune_e2.yaml --output-checkpoint outputs/coarse_retrieval_multiday_finetune_e2.pt
```

## Evaluate

```bash
python -m retrieval.evaluate_retrieval --config configs/eval_oct12_aisle_ccw_multiday.yaml --checkpoint outputs/coarse_retrieval_multiday_best.pt --output-json outputs/eval_oct12_aisle_ccw_multiday.json
```

## Best checkpoint

- Repository path: `checkpoints/coarse_retrieval_multiday_finetune_e2_top1_0p3388_r3_0p7038.pt`
- Matching report: `analysis/multiday_finetune_results.md`
