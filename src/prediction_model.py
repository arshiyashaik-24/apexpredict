"""Baseline statistical model: a shrunk team + driver pace rating.

pace(driver) = team pace (current season, recency weighted)
             + driver-vs-teammate gap (all seasons, shrunk towards 0)
Ratings are expressed in seconds per lap relative to the field mean, with a
standard error that the simulator turns into epistemic uncertainty.
No machine-learning model is fitted: a handful of races per season cannot
support one without overfitting (see README).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import feature_engineering as fe
from .data_loader import DataBundle, TARGET_KEYWORD, TARGET_YEAR, Weekend
from .race_simulator import SimConfig

DEFAULT_REF_LAP_S = 96.0
DEFAULT_LAPS = 56            # COTA race distance (track fact)
DEFAULT_PIT_LOSS_S = 20.0
DEFAULT_P_SC, DEFAULT_P_VSC, DEFAULT_P_WET = 0.50, 0.25, 0.05
DEFAULT_P_MECH, DEFAULT_P_INC = 0.035, 0.04
UNKNOWN_SD_S = 0.60          # rating uncertainty when a team has no current-season data
SINGLE_RACE_SD_S = 0.35      # ...when it has only one race (spread cannot be estimated)
SD_FLOOR_S = 0.08
PRACTICE_WEIGHT = 0.25       # weight of practice long-run pace in the race rating

LIMITATIONS = [
    "Few completed 2026 races: team ratings are noisy and will move as results arrive.",
    "Race pace comes from median clean-air lap time vs the field; it ignores fuel/engine modes and strategy.",
    "Practice long-run pace is hard to compare (fuel loads, engine modes are unknown) so it only gets a small weight.",
    "Overtaking, safety-car timing, strategy mix and tyre parameters are simplified, partly assumed models.",
    "Grid penalties are not read from data; use the custom grid editor to apply them.",
    "Weather forecasts are not available from FastF1; rain probability is a historical/assumed value.",
    "Austin circuit history is a small sample (a handful of races) and used only for context parameters.",
]


@dataclass
class Forecast:
    inputs: pd.DataFrame
    config: SimConfig
    grid: np.ndarray | None
    status: str                   # "demo" | "provisional" | "real"
    status_label: str
    context: pd.DataFrame         # parameter, value, source
    assumptions: list = field(default_factory=list)
    limitations: list = field(default_factory=lambda: list(LIMITATIONS))
    n_current_races: int = 0


def _center(x: pd.Series) -> pd.Series:
    return x - x.mean()


def build_forecast(bundle: DataBundle, weekend: Weekend | None = None, n_sims: int = 10_000, seed: int = 42,
                   year: int = TARGET_YEAR, cutoff_round=None, roster: pd.DataFrame | None = None,
                   use_practice: bool = True, grid_override: np.ndarray | None = None) -> Forecast:
    """Fit the baseline model on data strictly before `cutoff_round` and package simulator inputs."""
    assumptions = list(bundle.warnings)
    roster = (roster if roster is not None else bundle.roster)[["driver", "team"]].reset_index(drop=True)
    results = bundle.results
    pace_df = bundle.pace
    qual_df = fe.quali_table(results)

    circ = fe.circuit_features(bundle.races, TARGET_KEYWORD, year, cutoff_round)
    ctx = []

    def pick(name, key, default, note):
        v = circ.get(key)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            ctx.append((name, default, f"ASSUMPTION ({note})"))
            assumptions.append(f"{name} = {default} is an assumption: {note}.")
            return default
        ctx.append((name, round(v, 3) if isinstance(v, float) else v, f"Austin history, {circ['n_races']} race(s)"))
        return v

    ref_lap = pick("Reference lap (s)", "ref_lap_s", DEFAULT_REF_LAP_S, "no Austin lap-time history loaded")
    laps = int(pick("Race laps", "laps", DEFAULT_LAPS, "standard COTA distance"))
    pit_loss = pick("Pit-stop time loss (s)", "pit_loss_s", DEFAULT_PIT_LOSS_S, "no Austin pit-loss estimate")
    p_sc = pick("P(safety car)", "p_sc", DEFAULT_P_SC, "no Austin history")
    p_vsc = pick("P(virtual safety car)", "p_vsc", DEFAULT_P_VSC, "no Austin history")
    p_wet = pick("P(wet race)", "p_wet", DEFAULT_P_WET, "no forecast source; Austin rain history unavailable")
    deg_scale, deg_med = fe.degradation_scale(bundle.stints, year, cutoff_round)
    if deg_scale is None:
        deg_scale = 1.0
        ctx.append(("Tyre wear scale", 1.0, "ASSUMPTION (too few measured stints)"))
        assumptions.append("Tyre degradation uses assumed default compound wear rates.")
    else:
        ctx.append(("Tyre wear scale", round(deg_scale, 2), "measured on current-season stints (fuel-corrected)"))

    # ---- ratings -------------------------------------------------------------------
    rp = fe.pace_features(pace_df, "pace_pct", roster, year, cutoff_round)
    qp = fe.pace_features(qual_df, "quali_pct", roster, year, cutoff_round)
    rel = fe.reliability_features(results, roster, year, cutoff_round)
    k = ref_lap / 100.0
    n_races = int(rp.n_team_races.max()) if rp.n_team_races.notna().any() else 0

    def rating(f):
        pct = f.team_mean.fillna(0.0) + f.dev.fillna(0.0)
        return _center(pct) * k

    out = roster.copy()
    out["race_pace_s"] = rating(rp).to_numpy()
    out["quali_pace_s"] = rating(qp).to_numpy()
    sd_pct = np.sqrt(rp.team_se.fillna(0.0) ** 2 + rp.dev_se.fillna(0.0) ** 2)
    sd = np.maximum(sd_pct * k, SD_FLOOR_S)
    no_team = rp.team_mean.isna()
    one_race = (rp.n_team_races <= 1) | rp.team_se.isna()
    sd = np.where(no_team, UNKNOWN_SD_S, np.where(one_race, np.maximum(sd, SINGLE_RACE_SD_S), sd))
    out["pace_sd_s"] = sd
    out["n_team_races"] = rp.n_team_races.fillna(0).astype(int).to_numpy()
    out["pace_source"] = np.where(no_team, "assumed equal (no data)", "season data")
    out["p_mech"] = rel.p_mech.fillna(DEFAULT_P_MECH).to_numpy()
    out["p_inc"] = rel.p_inc.fillna(DEFAULT_P_INC).to_numpy()
    if rel.p_mech.isna().all():
        assumptions.append(f"Retirement risk uses assumed rates (mechanical {DEFAULT_P_MECH:.1%}, incident {DEFAULT_P_INC:.1%}).")
    if no_team.all():
        assumptions.append("No current-season pace data: all drivers are given EQUAL pace (assumption).")
    elif no_team.any():
        assumptions.append("Teams with no current-season data are given field-average pace with wide uncertainty.")

    # ---- practice update --------------------------------------------------------------
    out["practice_used"] = False
    if use_practice and weekend is not None and weekend.practice is not None and len(weekend.practice):
        fp = weekend.practice.set_index("driver").pace_pct
        have = out.driver.isin(fp.index)
        if have.sum() >= 8:
            fp_s = _center(out.driver.map(fp).astype(float)[have]) * k
            out.loc[have, "race_pace_s"] = ((1 - PRACTICE_WEIGHT) * out.loc[have, "race_pace_s"]
                                            + PRACTICE_WEIGHT * fp_s.to_numpy())
            out.loc[have, "practice_used"] = True
            assumptions.append(f"Practice long-run pace blended in at {PRACTICE_WEIGHT:.0%} weight (assumed weight).")
    out["race_pace_s"] = _center(out.race_pace_s)

    # ---- grid -----------------------------------------------------------------------------
    grid = None
    if grid_override is not None:
        grid = np.asarray(grid_override, float)
    elif weekend is not None and weekend.grid is not None and len(weekend.grid):
        gmap = weekend.grid.set_index("driver").grid
        g = out.driver.map(gmap).astype(float)
        g = g.fillna(pd.Series(len(out) + 1 + np.arange(len(out)), index=g.index))  # unknown drivers start at the back
        grid = g.to_numpy()

    if bundle.mode == "demo":
        status, label = "demo", "DEMO OUTPUT - fallback data, not a real-data forecast"
    elif grid is not None:
        status, label = "real", "REAL-DATA FORECAST - includes Austin qualifying grid"
    else:
        status, label = "provisional", "PROVISIONAL FORECAST - pre-qualifying, grid simulated"
    cfg = SimConfig(n_sims=n_sims, seed=seed, laps=laps, p_sc=float(p_sc), p_vsc=float(p_vsc), p_wet=float(p_wet),
                    pit_loss_s=float(pit_loss), deg_scale=deg_scale, ref_lap_s=float(ref_lap)).clamped()
    return Forecast(out, cfg, grid, status, label, pd.DataFrame(ctx, columns=["parameter", "value", "source"]),
                    assumptions, n_current_races=n_races)
