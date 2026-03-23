# Checkpoints

- `coarse_retrieval_classifier_head_e2_stride10_top1_0p4426_r3_0p7672.pt`
  - Source config: `configs/coarse_retrieval_classifier_head_e2_stride10.yaml`
  - Recommended eval config: `configs/eval_oct12_aisle_ccw_classifier_head_stride10.yaml`
  - Patch partition: `20m x 20m`, stride `10m`
  - Test set: `Oct. 12, 2022 / Aisle_CCW`
  - Metrics: `Top1=0.4426`, `Recall@3=0.7672`, `MRR=0.6463`
- `coarse_retrieval_multiday_finetune_e2_stride10_top1_0p3934_r3_0p7410.pt`
  - Source config: `configs/coarse_retrieval_multiday_finetune_e2_stride10.yaml`
  - Recommended eval config: `configs/eval_oct12_aisle_ccw_multiday_stride10.yaml`
  - Patch partition: `20m x 20m`, stride `10m`
  - Test set: `Oct. 12, 2022 / Aisle_CCW`
  - Metrics: `Top1=0.3934`, `Recall@3=0.7410`, `MRR=0.6142`
- `coarse_retrieval_multiday_stage2_e1_stride10_top1_0p4230_r3_0p7311.pt`
  - Source config: `configs/coarse_retrieval_multiday_stage2_e1_stride10.yaml`
  - Recommended eval config: `configs/eval_oct12_aisle_ccw_multiday_stride10.yaml`
  - Patch partition: `20m x 20m`, stride `10m`
  - Test set: `Oct. 12, 2022 / Aisle_CCW`
  - Metrics: `Top1=0.4230`, `Recall@3=0.7311`, `MRR=0.6017`
- `coarse_retrieval_localmatcher_stage3_hardrerank_e1_stride10_top1_0p4098_r3_0p7563.pt`
  - Source config: `configs/coarse_retrieval_localmatcher_stage3_hardrerank_e1_stride10.yaml`
  - Recommended eval config: `configs/eval_oct12_aisle_ccw_localmatcher_hardrerank_stride10.yaml`
  - Patch partition: `20m x 20m`, stride `10m`
  - Test set: `Oct. 12, 2022 / Aisle_CCW`
  - Metrics: `Top1=0.4098`, `Recall@3=0.7563`, `MRR=0.6229`
- `coarse_retrieval_localmatcher_stage3_hardrerank_e1_top1_0p4011_r3_0p6645.pt`
  - Source config: `configs/coarse_retrieval_localmatcher_stage3_hardrerank_e1.yaml`
  - Recommended eval config: `configs/eval_oct12_aisle_ccw_localmatcher_hardrerank.yaml`
  - Test set: `Oct. 12, 2022 / Aisle_CCW`
  - Metrics: `Top1=0.4011`, `Recall@3=0.6645`, `MRR=0.5720`
- `coarse_retrieval_multiday_finetune_e2_top1_0p3388_r3_0p7038.pt`
  - Source config: `configs/coarse_retrieval_multiday_finetune_e2.yaml`
  - Recommended eval config: `configs/eval_oct12_aisle_ccw_multiday.yaml`
  - Test set: `Oct. 12, 2022 / Aisle_CCW`
  - Metrics: `Top1=0.3388`, `Recall@3=0.7038`, `MRR=0.5518`
- `coarse_retrieval_classifier_head_e2_aux.pt`
  - Auxiliary classifier-head checkpoint used by the best ensemble.
- `coarse_retrieval_multiday_stage2_e1_aux.pt`
  - Auxiliary stage2 checkpoint used by the best ensemble.

This directory is used to store the best reproducible checkpoint in the repo because:

- the file is small enough for normal Git storage;
- `outputs/` is intentionally ignored;
- the same best checkpoint is also published in GitHub Release `best-checkpoint-top1-0.4426`.
