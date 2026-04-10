# Jun15-only Generic Deep Fine Localization on Full Oct12 Test Set

## Setup

- Fine matcher training config: `configs/fine_pose_matcher_train_jun15_fullroutes.yaml`
- Coarse split config: `configs/coarse_retrieval_jun15_fullroutes_stride10.yaml`
- Fine matcher checkpoint: `outputs/fine_pose_matcher_jun15_fullroutes_generic.pt`
- Policy: generic `tracker_init` branch only, with no directional patch overrides

## Full Oct12 Summary

- Test routes: `Aisle_CCW`, `Aisle_CW`, `Hallway_Full_CW_Run_1`, `Hallway_Full_CW_Run_2`, `Hallway_Straight_CCW`, `Hallway_Straight_CW`
- Total frames: `9630`
- Weighted mean position error: `9.6776 m`
- Weighted mean yaw error: `30.4434 deg`
- Frames below `1m`: `59.50%`
- Frames below `0.5m`: `58.43%`

## Per-route Results

- `Aisle_CCW`: `915` frames, mean position error `0.1073 m`, median position error `0.0828 m`, mean yaw error `0.4314 deg`, below `1m` `100.00%`
- `Aisle_CW`: `1092` frames, mean position error `0.2581 m`, median position error `0.0713 m`, mean yaw error `0.3965 deg`, below `1m` `94.23%`
- `Hallway_Full_CW_Run_1`: `2318` frames, mean position error `12.3353 m`, median position error `0.1890 m`, mean yaw error `38.2420 deg`, below `1m` `55.65%`
- `Hallway_Full_CW_Run_2`: `2320` frames, mean position error `12.1099 m`, median position error `0.2822 m`, mean yaw error `34.5621 deg`, below `1m` `54.27%`
- `Hallway_Straight_CCW`: `1175` frames, mean position error `8.0271 m`, median position error `0.2656 m`, mean yaw error `55.0601 deg`, below `1m` `58.81%`
- `Hallway_Straight_CW`: `1810` frames, mean position error `14.7486 m`, median position error `16.6116 m`, mean yaw error `32.4956 deg`, below `1m` `30.17%`

## Bottleneck

- `Aisle` routes are already stable with the generic `tracker_init` path.
- The dominant remaining error source is long-tail failure on `Hallway`, especially `Hallway_Straight_CW`.
- This confirms the next optimization target is not generic `Aisle` tuning, but `Hallway`-specific patch disambiguation and pose stabilization under the generic branch.
