"""Data collection (FastF1) with local caching and an explicit demo fallback.

Nothing here invents race data. If FastF1 or the network is unavailable the
loader returns `fallback_bundle()`, which only contains a driver/team roster
and is flagged mode="demo".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "cache"          # FastF1's own HTTP/session cache
PROCESSED_DIR = ROOT / "data" / "processed"  # our per-race summaries

TARGET_YEAR = 2026
TARGET_KEYWORD = "Austin"
RACE_DATE = "2026-10-25"
PRIOR_SEASONS = (2025,)              # full seasons (driver-vs-teammate context)
CIRCUIT_YEARS = (2025, 2024, 2023, 2022)  # Austin race only (circuit history)
FUEL_EFFECT_S_PER_LAP = 0.03         # ASSUMPTION used for tyre-degradation fuel correction
FASTF1_SESSIONS = ("FP1", "FP2", "FP3", "Q")

RESULT_COLS = ["year", "round", "event", "driver", "team", "grid", "finish",
               "status", "dnf_kind", "q1_s"]
PACE_COLS = ["year", "round", "driver", "team", "pace_pct", "n_laps"]
STINT_COLS = ["year", "round", "driver", "team", "compound", "deg_s_per_lap", "n_laps"]
RACE_COLS = ["year", "round", "event", "location", "date", "laps", "ref_lap_s",
             "pit_loss_s", "n_pit_samples", "sc", "vsc", "wet"]

# Roster used ONLY by the demo fallback. It may be out of date; it is not race data.
FALLBACK_ROSTER = [
    ("VER", "Red Bull Racing"), ("HAD", "Red Bull Racing"),
    ("LAW", "Racing Bulls"), ("LIN", "Racing Bulls"),
    ("RUS", "Mercedes"), ("ANT", "Mercedes"),
    ("LEC", "Ferrari"), ("HAM", "Ferrari"),
    ("NOR", "McLaren"), ("PIA", "McLaren"),
    ("ALO", "Aston Martin"), ("STR", "Aston Martin"),
    ("GAS", "Alpine"), ("COL", "Alpine"),
    ("ALB", "Williams"), ("SAI", "Williams"),
    ("OCO", "Haas F1 Team"), ("BEA", "Haas F1 Team"),
    ("HUL", "Audi"), ("BOR", "Audi"),
    ("PER", "Cadillac"), ("BOT", "Cadillac"),
]


def _empty(cols):
    return pd.DataFrame({c: pd.Series(dtype="object") for c in cols})


@dataclass
class DataBundle:
    results: pd.DataFrame = field(default_factory=lambda: _empty(RESULT_COLS))
    pace: pd.DataFrame = field(default_factory=lambda: _empty(PACE_COLS))
    stints: pd.DataFrame = field(default_factory=lambda: _empty(STINT_COLS))
    races: pd.DataFrame = field(default_factory=lambda: _empty(RACE_COLS))
    roster: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(FALLBACK_ROSTER, columns=["driver", "team"]))
    mode: str = "demo"                  # "real" | "demo"
    roster_is_assumed: bool = True
    coverage: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    unavailable: list = field(default_factory=list)
    latest_session: str | None = None
    fetched_at: str = ""


@dataclass
class Weekend:
    status: dict = field(default_factory=dict)      # session -> human-readable state
    grid: pd.DataFrame | None = None                # driver, grid (from qualifying)
    practice: pd.DataFrame | None = None            # driver, pace_pct, n_laps
    latest_session: str | None = None
    fetched_at: str = ""


def _now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC").tz_localize(None)


def fallback_bundle(reason: str) -> DataBundle:
    b = DataBundle()
    b.warnings.append(f"DEMO MODE: {reason}. Showing a roster-only fallback with no real performance data.")
    b.unavailable = ["race results", "qualifying", "lap times", "tyres", "pit stops", "weather"]
    b.fetched_at = str(_now())[:19]
    return b


# ----------------------------------------------------------------------------
# Pure processing helpers (no network) - unit-testable with fake sessions
# ----------------------------------------------------------------------------
def _col(df, name, default=np.nan):
    return df[name] if name in df.columns else pd.Series(default, index=df.index)


def _secs(series) -> pd.Series:
    return pd.to_timedelta(series, errors="coerce").dt.total_seconds()


def _ensure(df: pd.DataFrame, cols) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df


INCIDENT_WORDS = ("collision", "accident", "spun", "crash")


def classify_status(status: str) -> str:
    """'none' (finished/lapped), 'incident', 'mech', or 'excluded' (DSQ/DNS)."""
    s = str(status).lower()
    if "disqualif" in s or "did not start" in s or "excluded" in s:
        return "excluded"
    if s == "finished" or s.startswith("+") or "lap" in s and "lapped" in s:
        return "none"
    if any(w in s for w in INCIDENT_WORDS):
        return "incident"
    return "mech"


def practice_pace(laps: pd.DataFrame, min_stint_laps: int = 6) -> pd.DataFrame:
    """Long-run pace from a practice session: median lap time on stints of >= N
    clean laps, compared with the field median *on the same compound*."""
    laps = _ensure(laps.copy(), ["Driver", "LapTime", "Stint", "Compound", "PitInTime", "PitOutTime", "TrackStatus"])
    laps["lap_s"] = _secs(laps["LapTime"])
    ok = laps[laps.lap_s.notna() & laps.PitInTime.isna() & laps.PitOutTime.isna()
              & (laps.TrackStatus.astype(str) == "1") & laps.Compound.isin(["SOFT", "MEDIUM", "HARD"])]
    stint_len = ok.groupby(["Driver", "Stint"])["lap_s"].transform("count")
    ok = ok[stint_len >= min_stint_laps]
    if ok.empty:
        return pd.DataFrame(columns=["driver", "pace_pct", "n_laps"])
    # drop slow outliers inside each stint (traffic / mistakes)
    ok = ok[ok.lap_s <= ok.groupby(["Driver", "Stint"])["lap_s"].transform("median") * 1.03]
    per = ok.groupby(["Driver", "Compound"])["lap_s"].agg(["median", "count"]).reset_index()
    field_med = per.groupby("Compound")["median"].transform("median")
    per["pct"] = (per["median"] / field_med - 1) * 100
    out = per.groupby("Driver").agg(pace_pct=("pct", "mean"), n_laps=("count", "sum")).reset_index()
    return out.rename(columns={"Driver": "driver"})


def process_race_session(race, quali, meta: dict) -> dict:
    """Reduce a loaded FastF1 race (and optional qualifying) session to compact tables."""
    y, rnd = meta["year"], meta["round"]
    res = race.results
    status = _col(res, "Status", "").astype(str)
    results = pd.DataFrame({
        "year": y, "round": rnd, "event": meta["event"],
        "driver": _col(res, "Abbreviation"), "team": _col(res, "TeamName"),
        "grid": pd.to_numeric(_col(res, "GridPosition"), errors="coerce"),
        "finish": pd.to_numeric(_col(res, "Position"), errors="coerce"),
        "status": status,
    })
    results["dnf_kind"] = [classify_status(s) for s in status]
    q1 = {}
    if quali is not None:
        qr = quali.results
        if "Q1" in qr.columns:
            q1 = dict(zip(qr["Abbreviation"], _secs(qr["Q1"])))
    results["q1_s"] = results["driver"].map(q1)

    laps = _ensure(race.laps.copy(), ["Driver", "Team", "LapNumber", "LapTime", "Stint", "Compound",
                                      "TyreLife", "PitInTime", "PitOutTime", "TrackStatus"])
    laps["lap_s"] = _secs(laps["LapTime"])
    laps["ts"] = laps["TrackStatus"].astype(str)
    clean = laps[laps.lap_s.notna() & laps.PitInTime.isna() & laps.PitOutTime.isna()
                 & (laps.ts == "1") & (laps.LapNumber > 1)].copy()

    # race pace: lap time relative to the field median *on the same lap number*
    lap_med = clean.groupby("LapNumber")["lap_s"].agg(["median", "count"])
    lap_med = lap_med[lap_med["count"] >= 5]["median"]
    c2 = clean[clean.LapNumber.isin(lap_med.index)].copy()
    c2["rel"] = (c2.lap_s / c2.LapNumber.map(lap_med) - 1) * 100
    c2 = c2[c2.rel.abs() < 7]
    pace = c2.groupby(["Driver", "Team"])["rel"].agg(pace_pct="median", n_laps="count").reset_index()
    pace = pace[pace.n_laps >= 10].rename(columns={"Driver": "driver", "Team": "team"})
    pace.insert(0, "round", rnd)
    pace.insert(0, "year", y)

    # tyre degradation: slope of fuel-corrected lap time vs tyre age, per stint
    rows = []
    for (drv, team, stint, comp), g in clean.groupby(["Driver", "Team", "Stint", "Compound"]):
        if comp not in ("SOFT", "MEDIUM", "HARD") or len(g) < 8 or g.TyreLife.nunique() < 5:
            continue
        corrected = g.lap_s + FUEL_EFFECT_S_PER_LAP * g.LapNumber
        slope = np.polyfit(g.TyreLife.astype(float), corrected, 1)[0]
        rows.append((y, rnd, drv, team, comp, float(slope), len(g)))
    stints = pd.DataFrame(rows, columns=STINT_COLS)

    # pit-stop time loss: in-lap + out-lap minus two normal laps (green flag only)
    drv_med = clean.groupby("Driver")["lap_s"].median()
    ls = laps.sort_values(["Driver", "LapNumber"]).copy()
    ls["next_out"] = ls.groupby("Driver")["PitOutTime"].shift(-1)
    ls["next_s"] = ls.groupby("Driver")["lap_s"].shift(-1)
    ls["next_ts"] = ls.groupby("Driver")["ts"].shift(-1)
    pit = ls[ls.PitInTime.notna() & ls.next_out.notna() & ls.lap_s.notna() & ls.next_s.notna()
             & (ls.ts == "1") & (ls.next_ts == "1")]
    loss = (pit.lap_s + pit.next_s - 2 * pit.Driver.map(drv_med)).dropna()
    loss = loss[(loss > 10) & (loss < 40)]

    rainfall = np.nan
    wd = getattr(race, "weather_data", None)
    if wd is not None and "Rainfall" in getattr(wd, "columns", []):
        rainfall = bool(wd["Rainfall"].any())
    races = pd.DataFrame([{
        "year": y, "round": rnd, "event": meta["event"], "location": meta.get("location", ""),
        "date": meta.get("date", ""), "laps": float(laps.LapNumber.max()) if laps.LapNumber.notna().any() else np.nan,
        "ref_lap_s": float(clean.lap_s.median()) if len(clean) else np.nan,
        "pit_loss_s": float(loss.median()) if len(loss) >= 3 else np.nan,
        "n_pit_samples": int(len(loss)),
        "sc": bool(laps.ts.str.contains("4").any()),
        "vsc": bool(laps.ts.str.contains("6|7").any()),
        "wet": rainfall,
    }])
    return {"results": results, "pace": pace, "stints": stints, "races": races}


# ----------------------------------------------------------------------------
# FastF1 access
# ----------------------------------------------------------------------------
def _completed_events(ff1, year: int) -> list[dict]:
    sched = ff1.get_event_schedule(year, include_testing=False)
    date_col = "Session5DateUtc" if "Session5DateUtc" in sched.columns else "EventDate"
    out = []
    for _, ev in sched.iterrows():
        d = pd.Timestamp(ev[date_col])
        if pd.notna(d) and d + pd.Timedelta(hours=3) < _now():
            out.append({"year": year, "round": int(ev["RoundNumber"]), "event": str(ev["EventName"]),
                        "location": str(ev.get("Location", "")), "date": str(d)[:10]})
    return out


def _fetch_race(ff1, meta: dict, refresh: bool) -> dict:
    path = PROCESSED_DIR / f"{meta['year']}_{meta['round']:02d}.pkl"
    if path.exists() and not refresh:
        return pd.read_pickle(path)
    race = ff1.get_session(meta["year"], meta["round"], "R")
    race.load(laps=True, telemetry=False, weather=True, messages=False)
    quali, quali_ok = None, True
    try:
        quali = ff1.get_session(meta["year"], meta["round"], "Q")
        quali.load(laps=False, telemetry=False, weather=False, messages=False)
    except Exception:
        quali, quali_ok = None, False
    out = process_race_session(race, quali, meta)
    if quali_ok:  # don't cache an incomplete record: retry next time
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        pd.to_pickle(out, path)
    return out


def load_data(refresh: bool = False, prior_seasons=PRIOR_SEASONS, circuit_years=CIRCUIT_YEARS) -> DataBundle:
    """Load completed races. Never raises: failures become warnings or a demo fallback."""
    try:
        import fastf1 as ff1
    except ImportError:
        return fallback_bundle("the fastf1 package is not installed")
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        ff1.Cache.enable_cache(str(CACHE_DIR))
        ff1.set_log_level("WARNING")
        target_events = _completed_events(ff1, TARGET_YEAR)
    except Exception as exc:  # network / API failure
        return fallback_bundle(f"could not load the {TARGET_YEAR} schedule ({type(exc).__name__}: {exc})")

    to_load = list(target_events)
    unavailable, warnings = [], []
    years_needed = list(dict.fromkeys(list(prior_seasons) + list(circuit_years)))
    for yr in years_needed:
        try:
            evs = _completed_events(ff1, yr)
        except Exception as exc:
            unavailable.append(f"{yr} schedule ({type(exc).__name__})")
            continue
        if yr not in prior_seasons:
            evs = [e for e in evs if TARGET_KEYWORD.lower() in e["location"].lower()]
        to_load += evs

    parts = {"results": [], "pace": [], "stints": [], "races": []}
    loaded = {}
    for meta in to_load:
        try:
            out = _fetch_race(ff1, meta, refresh)
            for k in parts:
                parts[k].append(out[k])
            loaded.setdefault(meta["year"], []).append(meta)
        except Exception as exc:
            unavailable.append(f"{meta['year']} R{meta['round']} {meta['event']} ({type(exc).__name__})")

    if not loaded:
        return fallback_bundle("no race sessions could be loaded (offline or API unavailable)")

    def cat(k, cols):
        frames = [f for f in parts[k] if len(f)]
        return pd.concat(frames, ignore_index=True) if frames else _empty(cols)

    b = DataBundle(results=cat("results", RESULT_COLS), pace=cat("pace", PACE_COLS),
                   stints=cat("stints", STINT_COLS), races=cat("races", RACE_COLS), mode="real")
    cur = b.results[b.results.year == TARGET_YEAR]
    if len(cur):
        last_round = cur["round"].max()
        b.roster = cur[cur["round"] == last_round][["driver", "team"]].drop_duplicates("driver").reset_index(drop=True)
        b.roster_is_assumed = False
        ev = cur[cur["round"] == last_round].event.iloc[0]
        b.latest_session = f"{TARGET_YEAR} {ev} - Race (round {last_round})"
    else:
        warnings.append(f"No completed {TARGET_YEAR} races could be loaded; roster is an ASSUMPTION and "
                        "current-season performance is unknown.")
    b.coverage = {
        "target_year_races_loaded": len(loaded.get(TARGET_YEAR, [])),
        "target_year_races_completed": len(target_events),
        "seasons_loaded": {int(y): len(v) for y, v in loaded.items()},
        "races_with_pace": int(b.pace.groupby(["year", "round"]).ngroups) if len(b.pace) else 0,
        "races_with_quali": int(b.results.dropna(subset=["q1_s"]).groupby(["year", "round"]).ngroups) if len(b.results) else 0,
        "races_with_pit_loss": int(b.races.pit_loss_s.notna().sum()) if len(b.races) else 0,
        "stints_for_degradation": int(len(b.stints)),
    }
    if len(b.stints) == 0:
        unavailable.append("tyre degradation (no usable stints)")
    if b.coverage["races_with_quali"] == 0:
        unavailable.append("qualifying pace")
    unavailable.append("weather forecast (FastF1 has no forecast; rain probability is an assumption)")
    b.unavailable, b.warnings = unavailable, warnings
    b.fetched_at = str(_now())[:19]
    return b


def load_weekend(year: int = TARGET_YEAR) -> Weekend:
    """Load whichever Austin FP1/FP2/FP3/Q sessions have happened. Never raises."""
    wk = Weekend(fetched_at=str(_now())[:19])
    try:
        import fastf1 as ff1
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        ff1.Cache.enable_cache(str(CACHE_DIR))
        ff1.set_log_level("WARNING")
        sched = ff1.get_event_schedule(year, include_testing=False)
        ev = sched[sched["Location"].astype(str).str.contains(TARGET_KEYWORD, case=False)]
        if ev.empty:
            raise LookupError("Austin not found in schedule")
        ev = ev.iloc[0]
    except Exception as exc:
        wk.status = {s: f"unavailable ({type(exc).__name__})" for s in FASTF1_SESSIONS}
        return wk

    practice_frames = []
    for i, name in enumerate(["FP1", "FP2", "FP3", "Q"], start=1):
        date = pd.Timestamp(ev.get(f"Session{i}DateUtc"))
        if pd.isna(date) or date + pd.Timedelta(hours=2) > _now():
            wk.status[name] = "not yet run"
            continue
        try:
            s = ff1.get_session(year, int(ev["RoundNumber"]), name)
            if name == "Q":
                s.load(laps=False, telemetry=False, weather=False, messages=False)
                r = s.results
                g = pd.DataFrame({"driver": r["Abbreviation"], "grid": pd.to_numeric(r["Position"], errors="coerce")})
                g = g.sort_values("grid", na_position="last").reset_index(drop=True)
                g["grid"] = np.arange(1, len(g) + 1)   # fill drivers without a time at the back
                wk.grid = g
            else:
                s.load(laps=True, telemetry=False, weather=False, messages=False)
                p = practice_pace(s.laps)
                if len(p):
                    p["session"] = name
                    practice_frames.append(p)
            wk.status[name] = "loaded"
            wk.latest_session = f"{year} United States GP - {name}"
        except Exception as exc:
            wk.status[name] = f"unavailable ({type(exc).__name__})"
    if practice_frames:
        allp = pd.concat(practice_frames)
        wk.practice = allp.groupby("driver").apply(
            lambda g: pd.Series({"pace_pct": np.average(g.pace_pct, weights=g.n_laps), "n_laps": g.n_laps.sum()}),
            include_groups=False).reset_index()
    return wk
