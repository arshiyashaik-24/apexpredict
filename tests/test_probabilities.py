import numpy as np
import pytest

from src.evaluation import brier, log_loss
from src.race_simulator import SimConfig, run_simulation, wilson_interval


@pytest.fixture
def res(inputs):
    return run_simulation(inputs, SimConfig(n_sims=2000, seed=9))


def test_win_probabilities_sum_to_one(res):
    assert res.summary.win_p.sum() == pytest.approx(1.0, abs=1e-9)


def test_podium_top5_top10_sums(res):
    assert res.summary.podium_p.sum() == pytest.approx(3.0, abs=1e-9)
    assert res.summary.top5_p.sum() == pytest.approx(5.0, abs=1e-9)
    assert res.summary.top10_p.sum() == pytest.approx(10.0, abs=1e-9)


def test_distribution_rows_and_columns(res):
    assert np.allclose(res.pos_dist.sum(axis=1), 1.0)
    assert np.allclose(res.pos_dist.sum(axis=0), 1.0)      # each position is taken by exactly one driver
    assert res.podium_combos.probability.sum() <= 1.0 + 1e-9


def test_ordering_and_ranges(res):
    s = res.summary
    assert s.win_p.is_monotonic_decreasing
    assert (s.win_p <= s.podium_p + 1e-12).all() and (s.podium_p <= s.top5_p + 1e-12).all()
    assert (s.win_lo <= s.win_p).all() and (s.win_p <= s.win_hi).all()
    assert s.exp_pos.between(1, len(s)).all() and (s.pos_sd >= 0).all()


def test_wilson_interval_known_value():
    lo, hi = wilson_interval(50, 100)
    assert lo == pytest.approx(0.404, abs=0.002) and hi == pytest.approx(0.596, abs=0.002)
    lo0, hi0 = wilson_interval(0, 100)
    assert lo0 == 0 and hi0 > 0


def test_scoring_rules():
    assert log_loss(0.5) == pytest.approx(np.log(2))
    assert brier(np.array([1.0, 0.0]), 0) == 0.0
    assert brier(np.array([0.5, 0.5]), 0) == pytest.approx(0.5)
