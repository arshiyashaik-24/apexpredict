"""Turn raw tables into per-driver features.

Every function takes an explicit (year, cutoff_round) and only uses rows
strictly before it, so a forecast for race k can never see race k or later.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HALF_LIFE_RACES = 4.0        # current-season recency weighting
PRIOR_SEASON_WEIGHT = 0.3    # older seasons count less (2026 regulation change)
DEV_SHRINK_K = 3.0           # pseudo-races shrinking driver-vs-teammate gaps to zero
REL_PRIOR_STRENGTH = 20.0    # pseudo driver-races shrinking reliability toward the field rate
REL_PRIOR_SEASON_WEIGHT = 0.3


def usable_rows(df: pd.DataFrame, year: int, cutoff_round=None) -> pd.DataFrame:
    if df.empty:
        return df
    cut = np.inf if cutoff_round is None else cutoff_round
    return df[(df.year < year) | ((df.year == year) & (df["round"] < cut))]


def _weights(d: pd.DataFrame, year: int, last_round: float):
    cur = (d.year == year).to_numpy()
    w_cur = 0.5 ** ((last_round - d["round"].to_numpy(float)) / HALF_LIFE_RACES)
    w_old = PRIOR_SEASON_WEIGHT * 0.5 ** np.maximum(year - d.year.to_numpy(float) - 1, 0)
    return np.where(cur, w_cur, w_old), cur


def pace_features(df: pd.DataFrame, col: str, roster: pd.DataFrame, year: int, cutoff_round=None) -> pd.DataFrame:
    """Team level (current season only) + driver-vs-teammate deviation (all seasons, shrunk).
    `col` is a lower-is-faster metric in per cent. Returns one row per roster driver."""
    out = roster[["driver", "team"]].copy()
    for c in ("team_mean", "team_se", "n_team_races", "dev", "dev_se", "n_dev_races"):
        out[c] = np.nan
    d = usable_rows(df.dropna(subset=[col]), year, cutoff_round).copy() if len(df) else df
    if d is None or d.empty:
        return out
    d[col] = d[col].astype(float)
    cur_rows = d[d.year == year]
    last_round = cur_rows["round"].max() if len(cur_rows) else 0
    d["w"], d["cur"] = _weights(d, year, last_round)
    grp = d.groupby(["year", "round", "team"])[col]
    d["team_mean"], d["n_team"] = grp.transform("mean"), grp.transform("count")

    # team rating: current season only (older cars are not comparable after the 2026 rules reset)
    tr = d[d.cur].drop_duplicates(["year", "round", "team"])
    stats = {}
    for team, g in tr.groupby("team"):
        w = g.w.to_numpy()
        m = float(np.average(g.team_mean, weights=w))
        stats[team] = (m, float(w.sum() ** 2 / (w**2).sum()), len(g), g.team_mean.to_numpy() - m)
    n_obs = sum(len(v[3]) for v in stats.values())
    pooled_sd = (np.sqrt(sum((v[3] ** 2).sum() for v in stats.values()) / (n_obs - len(stats)))
                 if n_obs > len(stats) else np.nan)

    dd = d[d.n_team >= 2].copy()
    dd["dev_row"] = dd[col] - dd.team_mean
    rms = float(np.sqrt((dd.dev_row**2).mean())) if len(dd) else np.nan
    for i, row in out.iterrows():
        if row.team in stats:
            m, n_eff, n, _ = stats[row.team]
            out.loc[i, ["team_mean", "n_team_races"]] = [m, n]
            out.loc[i, "team_se"] = pooled_sd / np.sqrt(n_eff) if np.isfinite(pooled_sd) else np.nan
        g = dd[dd.driver == row.driver]
        if len(g):
            ws = g.w.sum()
            out.loc[i, "dev"] = float((g.w * g.dev_row).sum() / (ws + DEV_SHRINK_K))
            out.loc[i, "dev_se"] = rms / np.sqrt(ws + DEV_SHRINK_K)
            out.loc[i, "n_dev_races"] = len(g)
    return out


def quali_table(results: pd.DataFrame) -> pd.DataFrame:
    """Q1 time as % slower than the field median of that session (Q1: everyone sets a time)."""
    r = results.dropna(subset=["q1_s"]).copy() if len(results) else results
    if r.empty:
        return pd.DataFrame(columns=["year", "round", "driver", "team", "quali_pct"])
    r["q1_s"] = r.q1_s.astype(float)
    med = r.groupby(["year", "round"])["q1_s"].transform("median")
    r["quali_pct"] = (r.q1_s / med - 1) * 100
    r = r[r.quali_pct.abs() < 10]
    return r[["year", "round", "driver", "team", "quali_pct"]]


def reliability_features(results: pd.DataFrame, roster: pd.DataFrame, year: int, cutoff_round=None) -> pd.DataFrame:
    """Per-race retirement probabilities, shrunk toward the pooled field rate.
    Mechanical risk is attached to the team (shared car); incident risk to the driver."""
    out = roster[["driver", "team"]].copy()
    out["p_mech"], out["p_inc"], out["n_starts"] = np.nan, np.nan, 0
    d = usable_rows(results, year, cutoff_round) if len(results) else results
    if d is None or d.empty:
        return out
    d = d[d.dnf_kind.isin(["none", "mech", "incident"])].copy()
    if d.empty:
        return out
    d["w"] = np.where(d.year == year, 1.0, REL_PRIOR_SEASON_WEIGHT)
    p0m = (d.w * (d.dnf_kind == "mech")).sum() / d.w.sum()
    p0i = (d.w * (d.dnf_kind == "incident")).sum() / d.w.sum()
    m = REL_PRIOR_STRENGTH
    for i, row in out.iterrows():
        t = d[(d.team == row.team) & (d.year == year)]
        out.loc[i, "p_mech"] = ((t.w * (t.dnf_kind == "mech")).sum() + m * p0m) / (t.w.sum() + m)
        g = d[d.driver == row.driver]
        out.loc[i, "p_inc"] = ((g.w * (g.dnf_kind == "incident")).sum() + m * p0i) / (g.w.sum() + m)
        out.loc[i, "n_starts"] = len(g)
    return out


def degradation_scale(stints: pd.DataFrame, year: int, cutoff_round=None, base=None, min_stints=10):
    """Ratio of measured to assumed tyre wear (median over compounds), or None if too little data."""
    from .race_simulator import BASE_DEG, COMPOUND_NAMES
    s = usable_rows(stints, year, cutoff_round) if len(stints) else stints
    if s is None or len(s) < min_stints:
        return None, {}
    s = s[s.year == year]
    if len(s) < min_stints:
        return None, {}
    med = s.groupby("compound")["deg_s_per_lap"].median()
    ratios = [med[c] / BASE_DEG[i] for i, c in enumerate(COMPOUND_NAMES) if c in med.index and med[c] > 0]
    if not ratios:
        return None, med.to_dict()
    return float(np.clip(np.median(ratios), 0.5, 2.0)), med.to_dict()


def circuit_features(races: pd.DataFrame, keyword: str, year: int | None = None, cutoff_round=None) -> dict:
    """Circuit history from earlier races at the same venue (small samples; Beta(1,1)-smoothed rates)."""
    ctx = {"n_races": 0}
    if races is None or races.empty:
        return ctx
    r = usable_rows(races, year, cutoff_round) if year is not None else races
    h = r[r.location.astype(str).str.contains(keyword, case=False, na=False)]
    n = len(h)
    ctx["n_races"] = n
    if n == 0:
        return ctx
    ctx["laps"] = int(h.laps.median()) if h.laps.notna().any() else None
    ctx["ref_lap_s"] = float(h.ref_lap_s.median()) if h.ref_lap_s.notna().any() else None
    ctx["pit_loss_s"] = float(h.pit_loss_s.median()) if h.pit_loss_s.notna().any() else None
    ctx["p_sc"] = float((h.sc.astype(bool).sum() + 1) / (n + 2))
    ctx["p_vsc"] = float((h.vsc.astype(bool).sum() + 1) / (n + 2))
    wet = h.wet.dropna()
    if len(wet):  # blended with an assumed 5% prior worth 5 races
        ctx["p_wet"] = float((wet.astype(bool).sum() + 0.25) / (len(wet) + 5))
    return ctx
