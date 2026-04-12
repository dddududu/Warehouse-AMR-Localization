# Jun15-only Generic Deep Fine Localization on Full Oct12 Test Set

## Setup

- Fine matcher training config: `configs/fine_pose_matcher_train_jun15_fullroutes.yaml`
- Coarse split config: `configs/coarse_retrieval_jun15_fullroutes_stride10.yaml`
- Fine matcher checkpoint: `outputs/fine_pose_matcher_jun15_fullroutes_generic.pt`
- Policy: generic branch only, with no directional patch overrides
- Current generic best on `Hallway_Full`: keep `tracker_init`, but gate it by its own ICP quality instead of letting it compete unconditionally
- Hallway full-route configs:
  - `configs/fine_localization_oct12_hallway_full_cw_run1_deep_v6_guidedbev_generic_jun15coarse_trackerquality.yaml`
  - `configs/fine_localization_oct12_hallway_full_cw_run2_deep_v6_guidedbev_generic_jun15coarse_trackerquality.yaml`

## Full Oct12 Summary

- Test routes: `Aisle_CCW`, `Aisle_CW`, `Hallway_Full_CW_Run_1`, `Hallway_Full_CW_Run_2`, `Hallway_Straight_CCW`, `Hallway_Straight_CW`
- Total frames: `9630`
- Weighted mean position error: `6.7500 m`
- Weighted mean yaw error: `27.1564 deg`
- Frames below `1m`: `68.34%`
- Frames below `0.5m`: `67.47%`

## Per-route Results

- `Aisle_CCW`: `915` frames, mean position error `0.1073 m`, median position error `0.0828 m`, mean yaw error `0.4314 deg`, below `1m` `100.00%`
- `Aisle_CW`: `1092` frames, mean position error `0.2581 m`, median position error `0.0713 m`, mean yaw error `0.3965 deg`, below `1m` `94.23%`
- `Hallway_Full_CW_Run_1`: `2318` frames, mean position error `8.2763 m`, median position error `0.1487 m`, mean yaw error `45.9240 deg`, below `1m` `60.83%`
- `Hallway_Full_CW_Run_2`: `2320` frames, mean position error `4.0132 m`, median position error `0.0777 m`, mean yaw error `13.2431 deg`, below `1m` `85.78%`
- `Hallway_Straight_CCW`: `1175` frames, mean position error `8.0271 m`, median position error `0.2656 m`, mean yaw error `55.0601 deg`, below `1m` `58.81%`
- `Hallway_Straight_CW`: `1810` frames, mean position error `14.7486 m`, median position error `16.6116 m`, mean yaw error `32.4956 deg`, below `1m` `30.17%`

## Bottleneck

- `Aisle` routes are already stable under the generic branch.
- `Hallway_Full` no longer needs a blanket tracker shutdown; the effective generic fix is to keep `tracker_init` only when its ICP quality is strong enough.
- The dominant remaining error source is now `Hallway_Straight_CW`, followed by the long-tail failure frames in `Hallway_Straight_CCW`.
