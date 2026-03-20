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

## Metrics on `Oct. 12, 2022 / Aisle_CCW`

| Model | Top1 | Recall@3 | MRR |
| --- | ---: | ---: | ---: |
| Baseline `coarse_retrieval_model_v2.pt` | 0.1738 | 0.4033 | 0.3190 |
| Multi-day fine-tune + classifier fusion (`weight=0.4`) | 0.3268 | 0.6568 | 0.5400 |

## Notes

- The main gain came from multi-day training, query augmentation, and classifier-score fusion.
- Classifier fusion improves cross-date retrieval strongly, but the score weight must be tuned with the checkpoint architecture.
