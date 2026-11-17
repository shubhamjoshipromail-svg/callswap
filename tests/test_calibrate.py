import json

import numpy as np
import pytest
from scipy.special import softmax

from callswap.calibrate import calibrate, choose_threshold, fit_temperature, logits, wilson


def test_wilson_matches_hand_computed_values():
    assert wilson(8, 10) == pytest.approx(.4901624715)
    assert wilson(0, 10) == pytest.approx(0, abs=1e-12)
    assert wilson(10, 10) == pytest.approx(10 / (10 + 1.959963984540054**2))
    assert wilson(0, 0) == 0.0


def test_wilson_sample_size_needed_for_99_percent():
    # With zero disagreements, at least 381 accepted calls are needed to prove 99%.
    assert wilson(380, 380) < .99 <= wilson(381, 381)


def test_wilson_vectorised_and_monotonic():
    bounds = wilson(np.array([9, 90, 900, 0]), np.array([10, 100, 1000, 0]))
    assert bounds[0] < bounds[1] < bounds[2] < .9 and bounds[3] == 0
    assert bounds[1] == pytest.approx(wilson(90, 100))


def test_threshold_is_lowest_that_meets_target():
    conf = np.array([.99, .95, .9, .8, .7, .6])
    correct = np.array([True, True, True, True, False, True])
    # Accepting everything gives 5 of 6 (bound 0.44); stopping above the error gives 4 of 4 (bound 0.51).
    assert wilson(5, 6) < .5 < wilson(4, 4)
    threshold, met, _ = choose_threshold(conf, correct, .5)
    assert met and threshold == .8
    assert choose_threshold(conf, np.ones(6, bool), .5)[0] == .6


def test_threshold_excludes_errors_when_needed():
    conf = np.concatenate([np.linspace(.9, .99, 400), np.linspace(.5, .6, 50)])
    correct = np.concatenate([np.ones(400, bool), np.zeros(50, bool)])
    threshold, met, _ = choose_threshold(conf, correct, .99)
    assert met and threshold == pytest.approx(.9)
    accepted = conf >= threshold
    assert accepted.sum() == 400 and correct[accepted].all()


def test_equal_confidences_are_accepted_together():
    conf = np.array([.9] * 5 + [.5] * 5)
    correct = np.array([True] * 5 + [True, False, False, False, False])
    threshold, met, curve = choose_threshold(conf, correct, .5)
    assert {p['accepted'] for p in curve} == {5, 10}
    assert met and threshold == .9


def test_unreachable_target_returns_best_bound():
    rng = np.random.RandomState(0)
    conf, correct = rng.rand(300), rng.rand(300) < .6
    threshold, met, curve = choose_threshold(conf, correct, .999)
    assert not met
    best = max(curve, key=lambda p: p['lower_bound'])
    assert threshold == best['threshold']


def test_curve_is_downsampled_and_contains_the_pick():
    rng = np.random.RandomState(1)
    conf = rng.rand(5000)
    correct = rng.rand(5000) < conf
    threshold, _, curve = choose_threshold(conf, correct, .9)
    assert len(curve) <= 201
    assert threshold in {p['threshold'] for p in curve}
    assert all(0 < p['coverage'] <= 1 and p['lower_bound'] <= p['agreement'] for p in curve)


def test_fit_temperature_recovers_known_scale():
    rng = np.random.RandomState(0)
    base = rng.randn(4000, 5) * 2
    labels = np.array([rng.choice(5, p=softmax(row)) for row in base])
    assert fit_temperature(base * 3, labels) == pytest.approx(3, rel=.15)
    assert fit_temperature(base, labels) == pytest.approx(1, rel=.15)


def test_fit_temperature_skips_unseen_labels():
    scores = np.array([[2., 0.], [0., 2.], [1., 0.]])
    assert fit_temperature(scores, np.array([0, 1, -1])) == fit_temperature(scores[:2], np.array([0, 1]))
    assert fit_temperature(scores, np.array([-1, -1, -1])) == 1.0


def test_binary_scores_become_two_columns():
    class Binary:
        def decision_function(self, texts):
            return np.array([2., -4.])
    assert logits(Binary(), ['a', 'b']).tolist() == [[-1., 1.], [2., -2.]]


def test_calibrate_writes_result_and_uses_requested_target(run_dir):
    result = calibrate(run_dir, .8)
    saved = json.loads((run_dir / 'calibration_result.json').read_text())
    assert saved == result and result['target_agreement'] == .8
    assert 0 < result['threshold'] <= 1 and result['temperature'] > 0
    assert calibrate(run_dir, .8) == result  # deterministic
