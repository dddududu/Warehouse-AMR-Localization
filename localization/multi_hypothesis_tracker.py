from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np


@dataclass
class MultiHypothesisState:
    cumulative_score: float
    poses: list[np.ndarray] = field(default_factory=list)
    frame_results: list[dict[str, Any]] = field(default_factory=list)


def select_diverse_hypotheses(
    states: Iterable[MultiHypothesisState],
    *,
    beam_size: int,
    pose_merge_distance_m: float,
) -> list[MultiHypothesisState]:
    """Keep high-score hypotheses while preventing duplicate pose branches."""
    selected: list[MultiHypothesisState] = []
    for state in sorted(states, key=lambda item: item.cumulative_score, reverse=True):
        if not state.poses:
            continue
        pose_xy = np.asarray(state.poses[-1], dtype=np.float64)[:2, 3]
        duplicate = False
        for kept in selected:
            kept_xy = np.asarray(kept.poses[-1], dtype=np.float64)[:2, 3]
            if float(np.linalg.norm(pose_xy - kept_xy)) < float(pose_merge_distance_m):
                duplicate = True
                break
        if duplicate:
            continue
        selected.append(state)
        if len(selected) >= int(beam_size):
            break
    return selected
