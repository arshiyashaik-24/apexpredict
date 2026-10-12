"""Chronological backtest: forecast each race using only earlier races.

Metrics: winner log loss + Brier score (vs naive baselines), and separately
finishing-position error (MAE of expected position, Spearman correlation).
With a single season the sample is small, so treat results as indicative only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data_loader import DataBundle, TARGET_YEAR
from .prediction_model import build_forecast
from .race_simulator import run_simulation

EPS = 1e-12


def log_loss(p_winner: float) -> float:
    return float(-np.log(max(p_winner, EPS)))


def brier(probs: np.ndarray, winner_idx: int) -> float:
    y = np.zeros_like(probs)
    y[winner_idx] = 1.0
    return float(((probs - y) ** 2).sum())


def grid_baseline_probs(grid: np.ndarray) -> np.ndarray:
    """Naive reference: win probability decays geometrically with grid slot."""
    p = np.exp(-0.6 * (grid - 1))
    return p / p.sum()


def backtest(bundle: DataBundle, year: int = TARGET_YEAR, min_train_races: int = 3,
             n_sims: int = 2000, seed: int = 7) -> tuple[pd.DataFrame, dict]:
    res = bundle.results
    if res is None or res.empty:
        return pd.DataFrame(), {"note": "No results available to evaluate."}
    cur = res[res.year == year]
    rounds = sorted(cur["round"].unique())
    rows = []
    for rnd in rounds:
        if (np.array(rounds) < rnd).sum() < min_train_races:
            continue
        race = cur[cur["round"] == rnd].dropna(subset=["finish"])
        if race.grid.isna().all() or (race.finish == 1).sum() != 1:
            continue
        g = race.grid.fillna(len(race)).replace(0, len(race) + 1)   # pit-lane starts go last
        fc = build_forecast(bundle, n_sims=n_sims, seed=seed, year=year, cutoff_round=rnd,
                            roster=race[["driver", "team"]], grid_override=g.to_numpy())
        laps = bundle.races[(bundle.races.year == year) & (bundle.races["round"] == rnd)].laps
        if len(laps) and pd.notna(laps.iloc[0]):
            fc.config.laps = int(laps.iloc[0])
        sim = run_simulation(fc.inputs, fc.config, fc.grid)
        S = sim.summary.set_index("driver")
        drivers = fc.inputs.driver.tolist()
        counts = np.array([S.loc[d, "win_p"] for d in drivers]) * fc.config.n_sims
        p = (counts + 0.5) / (fc.config.n_sims + 0.5 * len(drivers))      # Jeffreys smoothing: no zero probabilities
        w = drivers.index(race.loc[race.finish == 1, "driver"].iloc[0])
        p_grid = grid_baseline_probs(g.to_numpy())
        exp_pos = np.array([S.loc[d, "exp_pos"] for d in drivers])
        act = race.set_index("driver").finish.reindex(drivers).to_numpy(float)
        rows.append({"round": rnd, "event": race.event.iloc[0], "winner": drivers[w],
                     "p_winner_model": p[w], "logloss_model": log_loss(p[w]), "brier_model": brier(p, w),
                     "logloss_uniform": log_loss(1 / len(drivers)), "brier_uniform": brier(np.full(len(drivers), 1 / len(drivers)), w),
                     "logloss_grid": log_loss(p_grid[w]), "brier_grid": brier(p_grid, w),
                     "pos_mae": float(np.nanmean(np.abs(exp_pos - act))),
                     "pos_spearman": float(pd.Series(exp_pos).corr(pd.Series(act), method="spearman"))})
    df = pd.DataFrame(rows)
    if df.empty:
        return df, {"note": f"Need at least {min_train_races} earlier races plus one test race; not enough data yet."}
    summ = {"n_races_tested": len(df), **{c: float(df[c].mean()) for c in df.columns
            if c.startswith(("logloss", "brier", "pos_"))},
            "note": "Small sample; each race forecast used only earlier races. Lower log loss/Brier is better."}
    return df, summ
