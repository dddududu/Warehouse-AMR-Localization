from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


def summarize(path: str) -> None:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    frames = data["frame_results"]
    print(f"\n=== {Path(path).name} ===")
    bad05 = [item for item in frames if item["position_error_m"] > 0.5]
    bad1 = [item for item in frames if item["position_error_m"] > 1.0]
    print("frames:", len(frames))
    print("bad > 0.5m:", len(bad05))
    print("bad > 1.0m:", len(bad1))

    init_counter = Counter()
    patch_counter = Counter()
    for item in bad05:
        selected = item["candidate_results"][item["selected_candidate_index"]]
        init_counter[str(selected.get("selected_init_source"))] += 1
        patch_counter[int(item["best_patch_id"])] += 1
    print("selected init on bad frames:", dict(init_counter))
    print("selected patch on bad frames:", patch_counter.most_common(10))

    start = None
    prev = None
    values: list[float] = []
    segments = []
    for item in frames:
        idx = int(item["frame_idx"])
        err = float(item["position_error_m"])
        if err > 0.5:
            if start is None or idx != prev + 1:
                if start is not None:
                    segments.append((start, prev, len(values), sum(values) / len(values), max(values)))
                start = idx
                prev = idx
                values = [err]
            else:
                prev = idx
                values.append(err)
        elif start is not None:
            segments.append((start, prev, len(values), sum(values) / len(values), max(values)))
            start = None
            prev = None
            values = []
    if start is not None:
        segments.append((start, prev, len(values), sum(values) / len(values), max(values)))
    print("bad segments (>0.5m):", segments)


if __name__ == "__main__":
    summarize("outputs/fine_localization_oct12_aisle_ccw_deep_v6_guidedbev_trackerinit_generic.json")
    summarize("outputs/fine_localization_oct12_aisle_cw_deep_v6_guidedbev_trackerinit_generic_jun15.json")
