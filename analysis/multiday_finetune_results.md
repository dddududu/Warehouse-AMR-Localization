# Multi-day Fine-tune Results

## Dataset split

- Train: `Jun. 15, 2022` aisle runs + `Jun. 23, 2022` `Aisle_CCW_Run_1` / `Aisle_CW_Run_1`
- Val: `Jun. 23, 2022` `Aisle_CCW_Run_2` / `Aisle_CW_Run_2`
- Test: `Oct. 12, 2022` `Aisle_CCW`
- Shared map / calibration: `Jun. 15, 2022` `Aisle_CCW_Run_1`

## Best current setting

- Train config: `configs/coarse_retrieval_classifier_head_e2_stride10.yaml`
- Eval config: `configs/eval_oct12_aisle_ccw_classifier_head_stride10.yaml`
- Checkpoint: `checkpoints/coarse_retrieval_classifier_head_e2_stride10_top1_0p4426_r3_0p7672.pt`
- Patch partition: `20m x 20m`, stride `10m`
- Best classifier fusion weight on `Oct. 12 / Aisle_CCW`: `0.75`

## Best overall system

- Single model remains the best overall after the stride-10 retest.
- Train config: `configs/coarse_retrieval_classifier_head_e2_stride10.yaml`
- Eval config: `configs/eval_oct12_aisle_ccw_classifier_head_stride10.yaml`
- Checkpoint: `checkpoints/coarse_retrieval_classifier_head_e2_stride10_top1_0p4426_r3_0p7672.pt`
- Metrics: `Top1=0.4426`, `Recall@3=0.7672`, `MRR=0.6463`

## Best ensemble under stride-10 patches

- Eval config: `configs/eval_oct12_aisle_ccw_multiday_ensemble_stride10.yaml`
- Checkpoints:
  - `checkpoints/coarse_retrieval_multiday_finetune_e2_stride10_top1_0p3934_r3_0p7410.pt`
  - `checkpoints/coarse_retrieval_classifier_head_e2_stride10_top1_0p4426_r3_0p7672.pt`
  - `checkpoints/coarse_retrieval_multiday_stage2_e1_stride10_top1_0p4230_r3_0p7311.pt`
- Ensemble weights: `0.1 / 0.75 / 0.15`
- Classifier weights: `0.25 / 0.5 / 0.0`
- Metrics: `Top1=0.4087`, `Recall@3=0.7945`, `MRR=0.6310`

## Best current local-rerank system

- Train config: `configs/coarse_retrieval_localmatcher_stage3_hardrerank_e1_stride10.yaml`
- Eval config: `configs/eval_oct12_aisle_ccw_localmatcher_hardrerank_stride10.yaml`
- Checkpoint: `checkpoints/coarse_retrieval_localmatcher_stage3_hardrerank_e1_stride10_top1_0p4098_r3_0p7563.pt`
- Inference recipe:
  - classifier fusion weight: `0.25`
  - local matcher rerank top-k: `6`
  - local matcher rerank weight: `0.2`

## Metrics on `Oct. 12, 2022 / Aisle_CCW`

| Model | Top1 | Recall@3 | MRR |
| --- | ---: | ---: | ---: |
| Baseline `coarse_retrieval_model_v2.pt` | 0.1738 | 0.4033 | 0.3190 |
| Multi-day fine-tune + classifier fusion (`weight=0.5`) | 0.3388 | 0.7038 | 0.5518 |
| 3-model ensemble | 0.3454 | 0.7421 | 0.5596 |
| Local-matcher hard rerank (`topk=22`, `weight=0.34`) | 0.4011 | 0.6645 | 0.5720 |
| Stride-10 baseline repartition retest | 0.1355 | 0.5333 | 0.3513 |
| Stride-10 fine-tune (`classifier=0.25`) | 0.3934 | 0.7410 | 0.6142 |
| Stride-10 stage2 (`classifier=0.0`) | 0.4230 | 0.7311 | 0.6017 |
| Stride-10 local-rerank (`classifier=0.25`) | 0.4098 | 0.7563 | 0.6229 |
| Stride-10 classifier-head (`classifier=0.75`) | 0.4426 | 0.7672 | 0.6463 |
| Stride-10 3-model ensemble | 0.4087 | 0.7945 | 0.6310 |

## Notes

- The main gain came from multi-day training, query augmentation, and classifier-score fusion.
- Classifier fusion improves cross-date retrieval strongly, but the score weight must be tuned with the checkpoint architecture.
- A light ensemble of complementary checkpoints adds a further top-1 gain without retraining the base descriptor from scratch.
- The next clear gain came from converting local matching into a second-stage reranker instead of fusing it against the full patch bank.
- Hard-negative-only local-matcher tuning improves top-1 further, but it trades away some `Recall@3`; this is a deliberate top-1 optimization.
- After changing the map partition to `20m x 20m` with stride `10m`, the old stride-5 classifier taxonomy becomes invalid, so retraining is mandatory before comparing accuracy.
- Under the new partition, the classifier-head branch becomes the strongest single model and beats both the local reranker and the tested ensembles on `Top1`.
