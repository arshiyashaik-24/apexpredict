"""Lap-by-lap Monte Carlo race simulator (vectorised over simulations).

Each simulated race is one coherent event: shared safety-car / weather draws,
per-driver pace, tyre strategy, pit stops, retirements and a simple traffic
model. Finishing order is derived from cumulative race time, so every race has
exactly one winner and a consistent classification.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

MAX_SIMS = 50_000
MIN_SIMS = 100
BIG = 1e7  # retired cars get BIG + (laps not completed) so they rank behind finishers

# Tyre model (ASSUMED defaults; deg is rescaled by measured degradation when available)
COMPOUND_NAMES = ("SOFT", "MEDIUM", "HARD")
COMPOUND_OFFSET = np.array([-0.45, 0.0, 0.35])   # s/lap vs medium
BASE_DEG = np.array([0.15, 0.09, 0.06])          # s/lap per lap of tyre age
CLIFF_AGE = np.array([14, 26, 38])               # laps before extra wear
CLIFF_RATE = 0.20                                # extra s/lap per lap beyond the cliff
PIT_FACTOR = np.array([1.0, 0.7, 0.5])           # pit loss multiplier: green, VSC, SC
# (name, compound ids per stint, pit fractions of race distance, weight). Always >= 2 compounds.
STRATEGIES = [
    ("M-H", (1, 2, 2), (0.40, None), 0.30),
    ("H-M", (2, 1, 1), (0.45, None), 0.25),
    ("S-H", (0, 2, 2), (0.28, None), 0.10),
    ("S-M-H", (0, 1, 2), (0.25, 0.60), 0.15),
    ("M-H-M", (1, 2, 1), (0.30, 0.65), 0.20),
]


@dataclass
class SimConfig:
    n_sims: int = 10_000
    seed: int = 42
    laps: int = 56
    p_sc: float = 0.50        # P(at least one safety car) in dry conditions
    p_vsc: float = 0.25       # P(at least one virtual safety car)
    p_wet: float = 0.05       # P(wet race)
    pit_loss_s: float = 20.0
    pit_loss_sd: float = 0.7
    mech_mult: float = 1.0    # scales mechanical retirement probability
    incident_mult: float = 1.0
    overtake_ease: float = 0.12   # per-lap pass probability when a car is "level" with the one ahead
    min_gap_s: float = 0.6        # following distance when stuck behind
    deg_scale: float = 1.0
    team_day_sd: float = 0.15     # s/lap, shared by both cars of a team in one race
    driver_day_sd: float = 0.12
    lap_noise_sd: float = 0.25
    quali_noise_sd: float = 0.12
    wet_pace_sd: float = 0.6      # extra driver-level pace spread in the wet
    grid_gap_s: float = 0.25      # time gap per grid slot at the start
    sc_laps: int = 4
    sc_bunch_gap_s: float = 0.8

    def validate(self) -> "SimConfig":
        if not (MIN_SIMS <= self.n_sims <= MAX_SIMS):
            raise ValueError(f"n_sims must be between {MIN_SIMS} and {MAX_SIMS}")
        if self.laps < 20:
            raise ValueError("laps must be >= 20")
        for name in ("p_sc", "p_vsc", "p_wet"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.pit_loss_s < 0 or self.mech_mult < 0 or self.incident_mult < 0:
            raise ValueError("pit loss and multipliers must be non-negative")
        return self

    def clamped(self) -> "SimConfig":
        return replace(self, n_sims=int(min(max(self.n_sims, MIN_SIMS), MAX_SIMS)))


@dataclass
class SimResult:
    summary: pd.DataFrame            # one row per driver, sorted by win probability
    positions: np.ndarray            # (n_sims, n_drivers) finishing position 1..N (DNFs last)
    dnf: np.ndarray                  # (n_sims, n_drivers) bool
    drivers: list
    pos_dist: pd.DataFrame           # index driver, columns 1..N
    podium_combos: pd.DataFrame
    diagnostics: dict
    config: SimConfig = field(default_factory=SimConfig)


def wilson_interval(k, n, z=1.96):
    """Wilson score interval for a binomial proportion (Monte Carlo sampling error only)."""
    k = np.asarray(k, dtype=float)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return np.clip(centre - half, 0, 1), np.clip(centre + half, 0, 1)


def summarise(positions, dnf, drivers, teams, config, diagnostics) -> SimResult:
    S, N = positions.shape
    win_k = (positions == 1).sum(0)
    lo, hi = wilson_interval(win_k, S)
    exp_pos = positions.mean(0)
    summary = pd.DataFrame({
        "driver": drivers, "team": teams,
        "win_p": win_k / S, "win_lo": lo, "win_hi": hi,
        "podium_p": (positions <= 3).mean(0), "top5_p": (positions <= 5).mean(0),
        "top10_p": (positions <= 10).mean(0), "dnf_p": dnf.mean(0),
        "exp_pos": exp_pos, "pos_sd": positions.std(0), "median_pos": np.median(positions, 0),
    }).sort_values("win_p", ascending=False).reset_index(drop=True)
    pos_dist = pd.DataFrame(np.stack([(positions == k).mean(0) for k in range(1, N + 1)], axis=1),
                            index=drivers, columns=range(1, N + 1))
    order = np.argsort(positions, axis=1)[:, :3]
    keys = order[:, 0] * N * N + order[:, 1] * N + order[:, 2]
    uniq, counts = np.unique(keys, return_counts=True)
    top = np.argsort(-counts)[:25]
    combos = pd.DataFrame({
        "P1": [drivers[u // (N * N)] for u in uniq[top]],
        "P2": [drivers[(u // N) % N] for u in uniq[top]],
        "P3": [drivers[u % N] for u in uniq[top]],
        "probability": counts[top] / S,
    })
    return SimResult(summary, positions, dnf, list(drivers), pos_dist, combos, diagnostics, config)


def _strategy_arrays(rng, S, N, L):
    w = np.array([s[3] for s in STRATEGIES], float)
    idx = rng.choice(len(STRATEGIES), size=(S, N), p=w / w.sum())
    comp_tab = np.array([s[1] for s in STRATEGIES])
    frac_tab = np.array([[np.nan if f is None else f for f in s[2]] for s in STRATEGIES])
    comps, frac = comp_tab[idx], frac_tab[idx]
    laps = np.rint(frac * L + rng.normal(0, 2.5, (S, N, 2)))
    first = np.clip(laps[..., 0], 5, L - 10)
    second = np.minimum(np.maximum(laps[..., 1], first + 8), L - 3)
    pit = np.stack([first, second], axis=2)
    pit = np.where(np.isnan(frac), L + 10, pit).astype(int)
    return comps, pit


def run_simulation(inputs: pd.DataFrame, config: SimConfig, grid=None) -> SimResult:
    """inputs: driver, team, race_pace_s, quali_pace_s, pace_sd_s, p_mech, p_inc.
    grid: optional array of starting positions aligned with `inputs` rows.
    If None, qualifying is simulated from quali pace for every race."""
    cfg = config.validate()
    rng = np.random.default_rng(cfg.seed)
    S, N, L = cfg.n_sims, len(inputs), cfg.laps
    drivers = inputs["driver"].tolist()
    team_codes, _ = pd.factorize(inputs["team"])
    race_pace = inputs["race_pace_s"].to_numpy(float)
    quali_pace = inputs["quali_pace_s"].to_numpy(float)
    pace_sd = inputs["pace_sd_s"].to_numpy(float)

    # --- race-level conditions (shared by all cars) --------------------------
    wet = rng.random(S) < cfg.p_wet
    p_sc = np.where(wet, cfg.p_sc + (1 - cfg.p_sc) * 0.4, cfg.p_sc)
    sc_on = rng.random(S) < p_sc
    sc_start = rng.integers(2, L - cfg.sc_laps - 1, S)
    vsc_on = rng.random(S) < cfg.p_vsc
    vsc_start = rng.integers(2, L - 3, S)
    lap_ix = np.arange(1, L + 1)[None, :]
    sc_mask = sc_on[:, None] & (lap_ix >= sc_start[:, None]) & (lap_ix < sc_start[:, None] + cfg.sc_laps)
    vsc_mask = vsc_on[:, None] & (lap_ix >= vsc_start[:, None]) & (lap_ix < vsc_start[:, None] + 2)
    status = np.where(sc_mask, 2, np.where(vsc_mask, 1, 0))  # (S, L)

    # --- retirements ------------------------------------------------------------
    p_mech = np.clip(inputs["p_mech"].to_numpy(float) * cfg.mech_mult, 0, 0.95)
    p_inc = np.clip(inputs["p_inc"].to_numpy(float) * cfg.incident_mult, 0, 0.95)
    mech = rng.random((S, N)) < p_mech[None, :]
    inc = rng.random((S, N)) < np.where(wet, 2.0, 1.0)[:, None] * p_inc[None, :]
    mech_lap = rng.integers(1, L + 1, (S, N))
    inc_lap = np.where(rng.random((S, N)) < 0.3, 1, rng.integers(1, L + 1, (S, N)))  # 30% first-lap
    retire_lap = np.minimum(np.where(mech, mech_lap, L + 99), np.where(inc, inc_lap, L + 99))
    dnf = retire_lap <= L

    # --- pace draws ----------------------------------------------------------------
    eps = rng.normal(0, 1, (S, N)) * pace_sd[None, :]          # rating uncertainty (shared by quali & race)
    team_day = rng.normal(0, cfg.team_day_sd, (S, team_codes.max() + 1))[:, team_codes]
    drv_day = rng.normal(0, cfg.driver_day_sd, (S, N))
    wet_extra = rng.normal(0, cfg.wet_pace_sd, (S, N)) * wet[:, None]
    base = race_pace[None, :] + eps + team_day + drv_day + wet_extra

    # --- grid ----------------------------------------------------------------------------
    if grid is not None:
        g = np.asarray(grid, float)
        grid_idx = np.broadcast_to(np.argsort(np.argsort(g, kind="stable"), kind="stable"), (S, N))
    else:
        qs = quali_pace[None, :] + eps + rng.normal(0, cfg.quali_noise_sd, (S, N))
        grid_idx = np.argsort(np.argsort(qs, axis=1), axis=1)

    # --- strategies (+ opportunistic stops under safety car) --------------------------
    comps, pit_laps = _strategy_arrays(rng, S, N, L)
    gt = pit_laps > sc_start[:, None, None]
    has, nxt = gt.any(2), gt.argmax(2)
    pp0 = np.concatenate([np.zeros((S, N, 1), int), pit_laps], 2)
    start_of_stint = np.take_along_axis(pp0, nxt[..., None], 2)[..., 0]
    cur = np.take_along_axis(pit_laps, nxt[..., None], 2)[..., 0]
    s1 = (sc_start + 1)[:, None]
    ok = (sc_on & ~wet)[:, None] & has & (cur <= L) & ((s1 - start_of_stint) >= 8) & (rng.random((S, N)) < 0.6)
    np.put_along_axis(pit_laps, nxt[..., None], np.where(ok, s1, cur)[..., None], 2)
    pp = np.concatenate([np.zeros((S, N, 1), int), pit_laps], 2)

    # --- lap loop ------------------------------------------------------------------------
    T = grid_idx * cfg.grid_gap_s + rng.normal(0, 0.4, (S, N))
    alive = np.ones((S, N), bool)
    for lap in range(1, L + 1):
        st = status[:, lap - 1]
        newly = alive & (retire_lap == lap)
        if newly.any():
            T = np.where(newly, BIG + (L - (lap - 1)) * 1000.0 + T / 1e4, T)
            alive = alive & ~newly
        nst = (pit_laps < lap).sum(2)
        age = lap - np.take_along_axis(pp, nst[..., None], 2)[..., 0]
        comp = np.take_along_axis(comps, nst[..., None], 2)[..., 0]
        wear = BASE_DEG[comp] * cfg.deg_scale * age + CLIFF_RATE * np.maximum(age - CLIFF_AGE[comp], 0)
        dry = base + COMPOUND_OFFSET[comp] + wear
        wetlap = base + 0.02 * age
        lap_t = np.where(wet[:, None], wetlap, dry) + rng.normal(0, cfg.lap_noise_sd, (S, N))
        lap_t = np.where((st > 0)[:, None], 0.0, lap_t)       # SC / VSC: common lap time
        pitting = alive & (pit_laps == lap).any(2)
        loss = (cfg.pit_loss_s + rng.normal(0, cfg.pit_loss_sd, (S, N))) * PIT_FACTOR[st][:, None]
        Tn = np.where(alive, T + lap_t + np.where(pitting, loss, 0.0), T)

        order = np.argsort(T, axis=1, kind="stable")           # running order at start of lap
        Ts = np.take_along_axis(Tn, order, 1)
        al = np.take_along_axis(alive, order, 1)
        green = st == 0
        if green.any():                                        # traffic: stuck behind slower car
            run = Ts[:, 0].copy()
            for r in range(1, N):
                c = Ts[:, r]
                need = run + cfg.min_gap_s
                adv = run - c
                p_pass = np.clip(cfg.overtake_ease + 0.3 * adv, 0.0, 0.95)
                passed = (adv > 0) & (rng.random(S) < p_pass)
                clamp = al[:, r] & green & (c < need) & ~passed
                c = np.where(clamp, need, c)
                Ts[:, r] = c
                run = np.maximum(run, c)
        sc_rows = np.flatnonzero(st == 2)
        if sc_rows.size:                                       # safety car: field bunches up
            sub = Ts[sc_rows]
            perm = np.argsort(sub, axis=1)
            sub = np.take_along_axis(sub, perm, 1)
            order[sc_rows] = np.take_along_axis(order[sc_rows], perm, 1)
            al_sub = np.take_along_axis(al[sc_rows], perm, 1)
            gaps = np.diff(sub, axis=1)
            gaps = np.where(al_sub[:, 1:], np.minimum(gaps, cfg.sc_bunch_gap_s), gaps)
            Ts[sc_rows] = np.concatenate([sub[:, :1], sub[:, :1] + np.cumsum(gaps, axis=1)], 1)
        T = Tn.copy()
        np.put_along_axis(T, order, Ts, 1)

    positions = (np.argsort(np.argsort(T, axis=1, kind="stable"), axis=1) + 1).astype(np.int16)
    diag = {"sc_rate": float(sc_on.mean()), "vsc_rate": float(vsc_on.mean()), "wet_rate": float(wet.mean()),
            "mean_dnfs_per_race": float(dnf.sum(1).mean()),
            "pit_stops_per_car": float(((pit_laps <= L).sum(2)).mean())}
    return summarise(positions, dnf, drivers, inputs["team"].tolist(), cfg, diag)


def compare_results(base: SimResult, alt: SimResult) -> pd.DataFrame:
    """Per-driver change (alt - base) in win/podium probability and expected position."""
    a = base.summary.set_index("driver")
    b = alt.summary.set_index("driver")
    out = pd.DataFrame({
        "win_base": a.win_p, "win_alt": b.win_p, "win_delta": b.win_p - a.win_p,
        "podium_base": a.podium_p, "podium_alt": b.podium_p,
        "exp_pos_base": a.exp_pos, "exp_pos_alt": b.exp_pos, "exp_pos_delta": b.exp_pos - a.exp_pos,
        "dnf_base": a.dnf_p, "dnf_alt": b.dnf_p,
    })
    return out.sort_values("win_alt", ascending=False)
