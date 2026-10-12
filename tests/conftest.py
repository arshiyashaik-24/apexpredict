"""Shared TEST FIXTURES. The synthetic data below exists only inside the test suite
to exercise code paths; the app never uses it."""
import numpy as np
import pandas as pd
import pytest

from src.data_loader import DataBundle, FALLBACK_ROSTER


def make_bundle(n_races=6, year=2026, seed=0):
    rng = np.random.default_rng(seed)
    roster = pd.DataFrame(FALLBACK_ROSTER, columns=["driver", "team"])
    strength = {t: s for t, s in zip(roster.team.unique(), np.linspace(-0.8, 0.8, roster.team.nunique()))}
    res, pace, races = [], [], []
    for rnd in range(1, n_races + 1):
        score = roster.team.map(strength).to_numpy() + rng.normal(0, 0.3, len(roster))
        order = np.argsort(score)
        finish = np.empty(len(roster)); finish[order] = np.arange(1, len(roster) + 1)
        for i, (d, t) in enumerate(zip(roster.driver, roster.team)):
            res.append(dict(year=year, round=rnd, event=f"GP{rnd}", driver=d, team=t, grid=finish[i],
                            finish=finish[i], status="Finished", dnf_kind=rng.choice(["none", "mech", "incident"], p=[.9, .05, .05]),
                            q1_s=90 + strength[t] + rng.normal(0, .1)))
            pace.append(dict(year=year, round=rnd, driver=d, team=t, pace_pct=strength[t] + rng.normal(0, .1), n_laps=40))
        races.append(dict(year=year, round=rnd, event=f"GP{rnd}", location="Elsewhere", date="", laps=56,
                          ref_lap_s=96.0, pit_loss_s=21.0, n_pit_samples=20, sc=False, vsc=False, wet=False))
    b = DataBundle(results=pd.DataFrame(res), pace=pd.DataFrame(pace), races=pd.DataFrame(races), mode="real",
                   roster=roster, roster_is_assumed=False)
    return b, strength


@pytest.fixture
def synth():
    return make_bundle()


@pytest.fixture
def inputs():
    b, _ = make_bundle()
    from src.prediction_model import build_forecast
    return build_forecast(b, n_sims=1000, seed=1).inputs
