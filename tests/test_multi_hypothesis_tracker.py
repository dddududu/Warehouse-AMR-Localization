import numpy as np

from localization.multi_hypothesis_tracker import MultiHypothesisState, select_diverse_hypotheses


def _state(score: float, x: float, y: float) -> MultiHypothesisState:
    pose = np.eye(4, dtype=np.float64)
    pose[:2, 3] = (x, y)
    return MultiHypothesisState(cumulative_score=score, poses=[pose])


def test_keeps_high_score_spatially_distinct_states() -> None:
    states = select_diverse_hypotheses(
        [_state(3.0, 0.0, 0.0), _state(2.9, 0.2, 0.0), _state(2.8, 2.0, 0.0)],
        beam_size=2,
        pose_merge_distance_m=0.5,
    )

    assert [state.cumulative_score for state in states] == [3.0, 2.8]


def test_empty_pose_state_is_not_selectable() -> None:
    states = select_diverse_hypotheses(
        [MultiHypothesisState(cumulative_score=10.0), _state(1.0, 0.0, 0.0)],
        beam_size=2,
        pose_merge_distance_m=0.5,
    )

    assert len(states) == 1
    assert states[0].cumulative_score == 1.0
