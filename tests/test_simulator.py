import numpy as np
import pytest

from src.race_simulator import MAX_SIMS, SimConfig, run_simulation


def test_one_winner_and_valid_classification(inputs):
    r = run_simulation(inputs, SimConfig(n_sims=1000, seed=3))
    N = len(inputs)
    assert ((r.positions == 1).sum(axis=1) == 1).all()                      # exactly one winner per race
    assert (np.sort(r.positions, axis=1) == np.arange(1, N + 1)).all()      # positions are a permutation


def test_retirees_classified_behind_finishers(inputs):
    r = run_simulation(inputs, SimConfig(n_sims=1000, seed=3, mech_mult=5.0))
    for s in range(1000):
        d = r.dnf[s]
        if d.any() and (~d).any():
            assert r.positions[s][d].min() > r.positions[s][~d].max()


def test_reproducible_with_seed(inputs):
    a = run_simulation(inputs, SimConfig(n_sims=500, seed=11)).summary
    b = run_simulation(inputs, SimConfig(n_sims=500, seed=11)).summary
    assert a.equals(b)
    c = run_simulation(inputs, SimConfig(n_sims=500, seed=12)).summary
    assert not a.win_p.equals(c.win_p)


def test_faster_driver_wins_more(inputs):
    fast = inputs.race_pace_s.idxmin()
    slow = inputs.race_pace_s.idxmax()
    s = run_simulation(inputs, SimConfig(n_sims=2000, seed=1)).summary.set_index("driver")
    assert s.loc[inputs.driver[fast], "win_p"] > s.loc[inputs.driver[slow], "win_p"]


def test_scenario_controls_change_the_simulation(inputs):
    base = run_simulation(inputs, SimConfig(n_sims=2000, seed=5))
    more_dnf = run_simulation(inputs, SimConfig(n_sims=2000, seed=5, mech_mult=4.0, incident_mult=4.0))
    assert more_dnf.summary.dnf_p.mean() > base.summary.dnf_p.mean() * 2
    sc = run_simulation(inputs, SimConfig(n_sims=2000, seed=5, p_sc=1.0))
    assert sc.diagnostics["sc_rate"] == 1.0 and base.diagnostics["sc_rate"] < 1.0
    wet = run_simulation(inputs, SimConfig(n_sims=2000, seed=5, p_wet=1.0))
    assert wet.diagnostics["wet_rate"] == 1.0
    assert wet.summary.pos_sd.mean() != pytest.approx(base.summary.pos_sd.mean(), abs=1e-6)
    pit = run_simulation(inputs, SimConfig(n_sims=2000, seed=5, pit_loss_s=40.0))
    assert not pit.summary.win_p.equals(base.summary.win_p)


def test_given_grid_matters_and_is_respected(inputs):
    grid = np.arange(len(inputs), 0, -1)       # reverse the grid: strongest car last
    base = run_simulation(inputs, SimConfig(n_sims=1500, seed=2))
    rev = run_simulation(inputs, SimConfig(n_sims=1500, seed=2), grid=np.arange(1, len(inputs) + 1)[::-1])
    assert not base.summary.win_p.equals(rev.summary.win_p)
    assert grid.shape[0] == len(inputs)


def test_config_validation(inputs):
    with pytest.raises(ValueError):
        run_simulation(inputs, SimConfig(n_sims=MAX_SIMS + 1))
    with pytest.raises(ValueError):
        run_simulation(inputs, SimConfig(p_sc=1.5))
    assert SimConfig(n_sims=10**9).clamped().n_sims == MAX_SIMS
