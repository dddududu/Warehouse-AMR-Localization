# Checkpoints

- `coarse_retrieval_multiday_finetune_e2_top1_0p3388_r3_0p7038.pt`
  - Source config: `configs/coarse_retrieval_multiday_finetune_e2.yaml`
  - Recommended eval config: `configs/eval_oct12_aisle_ccw_multiday.yaml`
  - Test set: `Oct. 12, 2022 / Aisle_CCW`
  - Metrics: `Top1=0.3388`, `Recall@3=0.7038`, `MRR=0.5518`

This directory is used to store the best reproducible checkpoint in the repo because:

- the file is small enough for normal Git storage;
- `outputs/` is intentionally ignored;
- GitHub Release upload is currently blocked because `gh` is not logged in on this machine.
