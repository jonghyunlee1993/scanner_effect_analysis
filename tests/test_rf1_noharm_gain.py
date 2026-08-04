import numpy as np

from analyze_rf1_noharm_gain import (
    IDENTITY_CAP,
    choose_candidate,
    choose_candidate_one_se,
    gamut_gate,
)
from build_rf1_noharm_nested_cell import selected_nested_pair, unordered_fold_pairs


def test_identity_is_selected_when_every_correction_worsens_one_fold():
    caps = np.asarray([IDENTITY_CAP, 1.05, 1.25])
    base = np.asarray([1.0, 1.0, 1.0, 1.0])
    candidate = np.asarray(
        [
            [1.0, 0.90, 0.80],
            [1.0, 0.91, 0.81],
            [1.0, 0.92, 1.01],
            [1.0, 1.02, 0.82],
        ]
    )
    gamut = np.ones_like(candidate, dtype=bool)
    selected, eligible = choose_candidate(
        caps, base, candidate, gamut, np.asarray([1.0, 0.94, 0.86])
    )
    assert selected == 0
    assert eligible.tolist() == [True, False, False]


def test_best_cross_fold_safe_candidate_is_selected():
    caps = np.asarray([IDENTITY_CAP, 1.05, 1.25])
    base = np.asarray([1.0, 1.2, 0.8])
    candidate = np.asarray(
        [
            [1.0, 0.91, 0.85],
            [1.2, 1.10, 1.05],
            [0.8, 0.76, 0.74],
        ]
    )
    gamut = np.ones_like(candidate, dtype=bool)
    gamut[:, 2] = False
    selected, eligible = choose_candidate(
        caps, base, candidate, gamut, np.asarray([0.99, 0.88, 0.80])
    )
    assert selected == 1
    assert eligible.tolist() == [True, True, False]


def test_exact_tie_prefers_weaker_candidate():
    caps = np.asarray([IDENTITY_CAP, 1.05, 1.25])
    base = np.asarray([1.0, 1.0])
    candidate = np.asarray([[1.0, 0.9, 0.9], [1.0, 0.9, 0.9]])
    gamut = np.ones_like(candidate, dtype=bool)
    selected, _ = choose_candidate(
        caps, base, candidate, gamut, np.asarray([1.0, 0.8, 0.8])
    )
    assert selected == 1


def test_gamut_gate_uses_frozen_mean_q99_projection_and_clamp_limits():
    material = np.zeros(1000)
    material[-5:] = 0.04
    projection = np.full(1000, 0.001)
    clamp = np.zeros(1000)
    assert gamut_gate(material, projection, clamp)
    projection[0] = 2.0
    assert not gamut_gate(material, projection, clamp)


def test_nested_pair_schedule_covers_each_unordered_pair_once_per_fov():
    pairs = unordered_fold_pairs()
    assert len(pairs) == 10
    assert len(set(pairs)) == 10
    assert all(first < second for first, second in pairs)
    scheduled = [selected_nested_pair(index) for index in range(30)]
    assert len(set(scheduled)) == 30
    assert {fov for fov, _ in scheduled} == {224, 256, 512}


def test_nested_pair_yields_two_strict_directions_with_three_training_folds():
    for _, (first, second) in [selected_nested_pair(index) for index in range(30)]:
        training = set(range(5)) - {first, second}
        assert len(training) == 3
        assert first not in training
        assert second not in training


def test_one_se_rule_prefers_identity_when_best_gain_is_not_precise():
    caps = np.asarray([IDENTITY_CAP, 1.05, 1.25])
    base = np.ones(4)
    candidate = np.asarray(
        [
            [1.0, 0.98, 0.60],
            [1.0, 0.97, 1.00],
            [1.0, 0.96, 1.00],
            [1.0, 0.95, 1.00],
        ]
    )
    gamut = np.ones_like(candidate, dtype=bool)
    selected, eligible, mean_delta, best, se, threshold, within = (
        choose_candidate_one_se(caps, base, candidate, gamut)
    )
    assert best == 2
    assert se > 0
    assert mean_delta[0] <= threshold
    assert within[0]
    assert selected == 0
    assert eligible.all()


def test_one_se_rule_chooses_weakest_candidate_within_best_uncertainty():
    caps = np.asarray([IDENTITY_CAP, 1.05, 1.25])
    base = np.ones(4)
    candidate = np.asarray(
        [
            [1.0, 0.87, 0.80],
            [1.0, 0.87, 0.90],
            [1.0, 0.87, 0.80],
            [1.0, 0.87, 0.90],
        ]
    )
    gamut = np.ones_like(candidate, dtype=bool)
    selected, _, _, best, _, _, within = choose_candidate_one_se(
        caps, base, candidate, gamut
    )
    assert best == 2
    assert within[1]
    assert selected == 1
