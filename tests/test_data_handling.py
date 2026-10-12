import builtins
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src import data_loader as dl
from src.evaluation import backtest
from src.feature_engineering import pace_features
from src.prediction_model import build_forecast
from src.race_simulator import run_simulation
from tests.conftest import make_bundle


def test_fallback_when_fastf1_missing(monkeypatch):
    real_import = builtins.__import__

    def fake(name, *a, **k):
        if name == "fastf1":
            raise ImportError("no fastf1")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake)
    b = dl.load_data()
    assert b.mode == "demo" and b.results.empty and b.warnings
    assert len(b.roster) == 22


def test_demo_forecast_is_flagged_and_uses_assumptions():
    fc = build_forecast(dl.fallback_bundle("test"), n_sims=500)
    assert fc.status == "demo" and "DEMO" in fc.status_label
    assert (fc.inputs.pace_source == "assumed equal (no data)").all()
    assert fc.inputs.race_pace_s.abs().max() == 0
    assert any("EQUAL pace" in a for a in fc.assumptions)
    assert any("ASSUMPTION" in s for s in fc.context.source)
    res = run_simulation(fc.inputs, fc.config, fc.grid)
    assert res.summary.win_p.sum() == np.float64(1.0) or abs(res.summary.win_p.sum() - 1) < 1e-9


def test_real_forecast_status_and_ratings_recover_team_order(synth):
    b, strength = synth
    fc = build_forecast(b, n_sims=500)
    assert fc.status == "provisional"
    best = min(strength, key=strength.get)
    worst = max(strength, key=strength.get)
    mean_pace = fc.inputs.groupby("team").race_pace_s.mean()
    assert mean_pace[best] < mean_pace[worst]
    wk = dl.Weekend(grid=pd.DataFrame({"driver": b.roster.driver, "grid": np.arange(1, 23)}))
    assert build_forecast(b, wk, n_sims=500).status == "real"


def test_no_data_leakage_from_future_races(synth):
    b, _ = synth
    clean = build_forecast(b, n_sims=200, cutoff_round=4).inputs
    b2, _ = make_bundle()
    mask = b2.pace["round"] >= 4
    b2.pace.loc[mask, "pace_pct"] = 99.0                 # poison the future
    b2.results.loc[b2.results["round"] >= 4, "dnf_kind"] = "mech"
    poisoned = build_forecast(b2, n_sims=200, cutoff_round=4).inputs
    pd.testing.assert_frame_equal(clean, poisoned)


def test_missing_columns_and_empty_tables_do_not_crash(synth):
    b, _ = synth
    empty = pace_features(pd.DataFrame(columns=["year", "round", "driver", "team", "x"]), "x", b.roster, 2026)
    assert empty.team_mean.isna().all()
    b.races = b.races.iloc[0:0]
    b.stints = b.stints.iloc[0:0]
    fc = build_forecast(b, n_sims=200)
    assert not fc.inputs.isna().any().any()
    assert fc.config.deg_scale == 1.0


def test_status_classification():
    assert dl.classify_status("Finished") == "none" and dl.classify_status("+1 Lap") == "none"
    assert dl.classify_status("Collision damage") == "incident"
    assert dl.classify_status("Power Unit") == "mech"
    assert dl.classify_status("Disqualified") == "excluded"


def test_process_race_session_with_fake_session():
    laps = []
    for d, t, off in [("AAA", "T1", 0.0), ("BBB", "T1", 0.2), ("CCC", "T2", 0.5), ("DDD", "T2", 0.7),
                      ("EEE", "T3", 1.0), ("FFF", "T3", 1.2)]:
        for n in range(1, 31):
            pit_in = pd.Timedelta(seconds=1) if n == 15 else pd.NaT
            pit_out = pd.Timedelta(seconds=1) if n == 16 else pd.NaT
            lt = 90 + off + 0.05 * (n % 10) + (20 if n in (15, 16) else 0)
            laps.append(dict(Driver=d, Team=t, LapNumber=n, LapTime=pd.Timedelta(seconds=lt), Stint=1 + (n > 15),
                             Compound="MEDIUM", TyreLife=n if n <= 15 else n - 15, PitInTime=pit_in,
                             PitOutTime=pit_out, TrackStatus="1"))
    res = pd.DataFrame({"Abbreviation": ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"],
                        "TeamName": ["T1", "T1", "T2", "T2", "T3", "T3"], "GridPosition": [1, 2, 3, 4, 5, 6],
                        "Position": [1, 2, 3, 4, 5, 6.0], "Status": ["Finished"] * 5 + ["Engine"]})
    race = SimpleNamespace(results=res, laps=pd.DataFrame(laps), weather_data=pd.DataFrame({"Rainfall": [False]}))
    quali = SimpleNamespace(results=pd.DataFrame({"Abbreviation": res.Abbreviation,
                                                  "Q1": [pd.Timedelta(seconds=80 + i) for i in range(6)]}))
    out = dl.process_race_session(race, quali, dict(year=2026, round=1, event="Test GP", location="Nowhere", date="x"))
    p = out["pace"].set_index("driver").pace_pct
    assert p["AAA"] < p["FFF"]                                   # faster car has lower relative pace
    assert out["results"].set_index("driver").dnf_kind["FFF"] == "mech"
    assert out["results"].q1_s.notna().all()
    assert out["races"].laps.iloc[0] == 30 and not out["races"].sc.iloc[0]
    assert len(out["stints"]) > 0


def test_backtest_runs_chronologically(synth):
    b, _ = synth
    df, summ = backtest(b, n_sims=300, min_train_races=3)
    assert list(df["round"]) == [4, 5, 6]
    assert np.isfinite(df.logloss_model).all() and np.isfinite(summ["brier_model"])
    df2, s2 = backtest(make_bundle(n_races=2)[0], n_sims=300)
    assert df2.empty and "note" in s2
