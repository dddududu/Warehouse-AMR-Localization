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

## Metrics on `Oct. 12, 2022 / Aisle_CCW`

| Model | Top1 | Recall@3 | MRR |
| --- | ---: | ---: | ---: |
| Baseline `coarse_retrieval_model_v2.pt` | 0.1738 | 0.4033 | 0.3190 |
| Multi-day fine-tune + classifier fusion (`weight=0.5`) | 0.3388 | 0.7038 | 0.5518 |
| 3-model ensemble | 0.3454 | 0.7421 | 0.5596 |

## Notes

- The main gain came from multi-day training, query augmentation, and classifier-score fusion.
- Classifier fusion improves cross-date retrieval strongly, but the score weight must be tuned with the checkpoint architecture.
- A light ensemble of complementary checkpoints adds a further top-1 gain without retraining the base descriptor from scratch.
