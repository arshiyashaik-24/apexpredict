# ApexPredict - 2026 US Grand Prix (Austin) race predictor

Data-driven Formula 1 forecast for the **2026 United States Grand Prix, Circuit of The Americas, 25 Oct 2026**:
FastF1 data -> baseline pace/reliability model -> lap-by-lap Monte Carlo simulation -> Streamlit dashboard.
No paid API key is needed.

## Architecture

```
app.py                      Streamlit dashboard (sections A-E)
src/data_loader.py          FastF1 loading, caching, demo fallback
src/feature_engineering.py  time-safe features (pace, reliability, circuit history)
src/prediction_model.py     baseline rating model -> simulator inputs + forecast status
src/race_simulator.py       vectorised Monte Carlo race simulator + summary statistics
src/evaluation.py           chronological backtest (log loss, Brier, position error)
tests/                      pytest suite
data/processed/             cached per-race summaries      cache/  FastF1's own cache
```

## Setup and launch

```bash
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The first launch downloads the 2026 season, 2025 season and Austin 2022-2025 races. This can take several
minutes; later launches read the local cache. If FastF1 or the network is unavailable the app still starts in
**demo mode** (red banner): a roster-only dataset where all drivers have equal assumed pace. Demo output is
not a prediction.

## Tests

```bash
python -m pytest
```

## Refreshing after Austin practice and qualifying

After each session finishes (FP1/FP2/FP3 Friday-Saturday, qualifying Saturday), click **Refresh data** in the
sidebar (or restart the app). The loader picks up whichever Austin sessions have been run:
- practice -> long-run pace is blended into race pace (25% weight, an assumption);
- qualifying -> the real starting grid replaces simulated qualifying and the status becomes **Real-data forecast**.

Grid penalties are not in the data: apply them in Section D with the *Custom grid* editor.
Before qualifying the status is **Provisional**; if data cannot load it is **Demo**.

## Data sources and limits

FastF1 supplies results, qualifying (Q1 times), laps, tyre compounds, pit in/out times, track status and weather.
Per-race summaries are cached in `data/processed/`. Limits:
- Not available from FastF1: weather forecasts (rain probability is a labelled assumption), grid penalties, engine modes, fuel loads.
- Lap data may be missing or incomplete for some sessions; failed sessions are listed under *Unavailable*, never filled with invented values.
- 2025 and older seasons are used only for driver-vs-teammate gaps, reliability priors and Austin context, at reduced weight, because the 2026 regulations changed car performance.

## How the model works

1. **Pace**: for each race, lap time relative to the field median on the same lap number (clean laps only: no pit laps, green flag).
   Team rating = recency-weighted mean (half-life 4 races) of the team's current-season pace. Driver rating = team rating +
   shrunk gap to teammate (all seasons, older seasons weighted 0.3). Converted to seconds/lap with the Austin reference lap time.
   Standard errors become per-driver rating uncertainty, sampled once per simulated race.
2. **Qualifying** pace uses the same method on Q1 times (used only if the grid is not known).
3. **Reliability**: mechanical retirement per team and incident retirement per driver, shrunk towards the field rate (20 pseudo-starts).
4. **Circuit context**: safety-car / VSC frequency, pit-stop loss and lap count from earlier Austin races (small samples, Beta(1,1)-smoothed); defaults are labelled assumptions.
5. **Tyres**: assumed compound offsets and wear rates, rescaled by measured current-season fuel-corrected degradation when >=10 stints exist.

No machine-learning model is fitted: a few races per season would overfit, so the baseline statistical model is used on purpose.

**Validation**: `src/evaluation.py` forecasts each completed 2026 race using only earlier races, with the actual grid, and reports winner
log loss and Brier score against uniform and grid-based baselines, plus finishing-position MAE and Spearman correlation.
Run it from the dashboard (*Model validation*). With one season the sample is tiny, so no accuracy claim is made.

## How the Monte Carlo simulation works

Each of N races (default 10,000; limits 100-50,000; seeded) is simulated lap by lap:
shared safety-car/VSC/wet draws -> grid (given, or simulated from qualifying pace) -> per-driver pace
(rating + rating uncertainty + team-day + driver-day + lap noise) -> one of five tyre strategies with wear and a cliff ->
pit stops (cheaper under SC/VSC; some drivers pit opportunistically under SC) -> retirements (random lap, 30% of incidents on lap 1) ->
traffic (a car stuck behind another passes with a probability depending on its pace advantage) -> safety car bunches the field.
Finishing order comes from cumulative race time; retirees rank behind finishers by laps completed. Hence one winner per race.
Probabilities are simulation frequencies; changing any scenario input re-runs the simulation.

## Interpreting results

Win probability of 18% means the driver won about 18 in 100 simulated races under the model's assumptions - not that the model is 18% "sure".
The error bars (Wilson 95%) only measure Monte Carlo sampling error; they do not include errors in the ratings or assumptions, which are larger.
Expected finishing position includes retirements (placed at the back).

## Known limitations / not implemented

- Lapping, red flags, team orders, start-line crashes beyond the lap-1 incident share, and car damage are not modelled.
- Overtaking, strategy mix, SC duration and noise parameters are hand-set, not fitted.
- Safety cars are independent of incidents.
- The FastF1 loading code could not be exercised against the live API in the build environment (no network); it is covered by a fake-session unit test only. Check the first real run.
- Demo fallback roster may be outdated.
