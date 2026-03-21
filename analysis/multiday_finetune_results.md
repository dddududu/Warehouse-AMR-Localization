# Multi-day Fine-tune Results

## Dataset split

- Train: `Jun. 15, 2022` aisle runs + `Jun. 23, 2022` `Aisle_CCW_Run_1` / `Aisle_CW_Run_1`
- Val: `Jun. 23, 2022` `Aisle_CCW_Run_2` / `Aisle_CW_Run_2`
- Test: `Oct. 12, 2022` `Aisle_CCW`
- Shared map / calibration: `Jun. 15, 2022` `Aisle_CCW_Run_1`

## Best current setting

- Train config: `configs/coarse_retrieval_multiday_finetune_e2.yaml`
- Eval config: `configs/eval_oct12_aisle_ccw_multiday.yaml`
- Checkpoint: `outputs/coarse_retrieval_multiday_finetune_e2.pt`
- Best classifier fusion weight on `Oct. 12 / Aisle_CCW`: `0.5`

## Best overall system

- Eval config: `configs/eval_oct12_aisle_ccw_multiday_ensemble.yaml`
- Checkpoints:
  - `checkpoints/coarse_retrieval_multiday_finetune_e2_top1_0p3388_r3_0p7038.pt`
  - `checkpoints/coarse_retrieval_classifier_head_e2_aux.pt`
  - `checkpoints/coarse_retrieval_multiday_stage2_e1_aux.pt`
- Ensemble weights: `0.7 / 0.1 / 0.2`
- Classifier weights: `0.5 / 0.6 / 0.65`

## Best current single system

- Train config: `configs/coarse_retrieval_localmatcher_stage3_hardrerank_e1.yaml`
- Eval config: `configs/eval_oct12_aisle_ccw_localmatcher_hardrerank.yaml`
- Checkpoint: `checkpoints/coarse_retrieval_localmatcher_stage3_hardrerank_e1_top1_0p4011_r3_0p6645.pt`
- Inference recipe:
  - classifier fusion weight: `0.5`
  - local matcher rerank top-k: `22`
  - local matcher rerank weight: `0.34`

## Metrics on `Oct. 12, 2022 / Aisle_CCW`

| Model | Top1 | Recall@3 | MRR |
| --- | ---: | ---: | ---: |
| Baseline `coarse_retrieval_model_v2.pt` | 0.1738 | 0.4033 | 0.3190 |
| Multi-day fine-tune + classifier fusion (`weight=0.5`) | 0.3388 | 0.7038 | 0.5518 |
| 3-model ensemble | 0.3454 | 0.7421 | 0.5596 |
| Local-matcher hard rerank (`topk=22`, `weight=0.34`) | 0.4011 | 0.6645 | 0.5720 |

## Notes

- The main gain came from multi-day training, query augmentation, and classifier-score fusion.
- Classifier fusion improves cross-date retrieval strongly, but the score weight must be tuned with the checkpoint architecture.
- A light ensemble of complementary checkpoints adds a further top-1 gain without retraining the base descriptor from scratch.
- The next clear gain came from converting local matching into a second-stage reranker instead of fusing it against the full patch bank.
- Hard-negative-only local-matcher tuning improves top-1 further, but it trades away some `Recall@3`; this is a deliberate top-1 optimization.
