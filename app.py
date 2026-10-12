"""ApexPredict - Streamlit dashboard. Run with:  streamlit run app.py"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dataclasses import replace

from src import data_loader as dl
from src.evaluation import backtest
from src.prediction_model import Forecast, build_forecast
from src.race_simulator import MAX_SIMS, SimConfig, compare_results, run_simulation

RED, BG, PANEL, MUTED = "#E10600", "#121317", "#1B1D23", "#9AA0AA"
st.set_page_config(page_title="ApexPredict | 2026 US GP", page_icon="🏁", layout="wide")
st.markdown(f"""<style>
.stApp {{ background:{BG}; }} h1,h2,h3 {{ letter-spacing:.02em; font-weight:700; }}
h2 {{ border-left:4px solid {RED}; padding-left:.6rem; margin-top:2.2rem; }}
div[data-testid="stMetric"] {{ background:{PANEL}; padding:.8rem 1rem; border-radius:6px; border-top:2px solid {RED}; }}
.pill {{ display:inline-block; padding:.25rem .8rem; border:1px solid {RED}; color:#fff; border-radius:3px;
        font-size:.8rem; letter-spacing:.08em; background:rgba(225,6,0,.15); }}
.note {{ color:{MUTED}; font-size:.88rem; }}
</style>""", unsafe_allow_html=True)


def style(fig, h=520):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      height=h, margin=dict(l=10, r=10, t=40, b=10), font=dict(color="#E8E9EC"))
    return fig


@st.cache_data(ttl=3600, show_spinner="Loading FastF1 data (first run can take several minutes; cached afterwards)...")
def get_data(refresh: bool):
    return dl.load_data(refresh=refresh), dl.load_weekend()


@st.cache_data(show_spinner="Running Monte Carlo simulation...")
def simulate(inputs: pd.DataFrame, cfg: dict, grid):
    return run_simulation(inputs, SimConfig(**cfg), None if grid is None else np.array(grid))


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### ApexPredict")
    seed = st.number_input("Random seed", 0, 10**6, 42)
    use_fp = st.checkbox("Blend in Austin practice pace (if available)", True)
    refresh = st.button("Refresh data (after FP / qualifying)")
    if refresh:
        st.cache_data.clear()
bundle, weekend = get_data(refresh)

n_sims = int(st.session_state.get("n_sims", 10_000))
fc: Forecast = build_forecast(bundle, weekend, n_sims=n_sims, seed=int(seed), use_practice=use_fp)
inputs, base_cfg = fc.inputs, fc.config
base = simulate(inputs, base_cfg.__dict__, fc.grid)
S = base.summary

# ---------------------------------------------------------------- A. overview
st.title("2026 United States Grand Prix")
st.markdown(f"**Circuit of The Americas, Austin, Texas** &nbsp;·&nbsp; Race day {dl.RACE_DATE}")
st.markdown(f"<span class='pill'>{fc.status_label}</span>", unsafe_allow_html=True)
c = st.columns(4)
c[0].metric("Latest session used", (weekend.latest_session or bundle.latest_session or "none").replace("2026 ", ""))
c[1].metric("2026 races in model", f"{bundle.coverage.get('target_year_races_loaded', 0)}")
c[2].metric("Last data update (UTC)", bundle.fetched_at or "n/a")
c[3].metric("Simulations", f"{base_cfg.n_sims:,}")
if fc.status == "demo":
    st.error("Demo output: FastF1 data could not be loaded, so every driver has EQUAL assumed pace. "
             "These numbers only demonstrate the interface and are not a prediction.")
elif fc.status == "provisional":
    st.warning("Provisional: qualifying has not been loaded, so the starting grid is itself simulated. "
               "Refresh after qualifying for a grid-based forecast.")
with st.expander("Data coverage, session status and assumptions"):
    st.write("**Austin weekend sessions:**", weekend.status or "unavailable")
    st.write("**Coverage:**", bundle.coverage or "none")
    if bundle.unavailable:
        st.write("**Unavailable features / sessions:**", bundle.unavailable)
    st.write("**Assumptions (not measured from data):**")
    for a in dict.fromkeys(fc.assumptions):
        st.markdown(f"- {a}")
    st.dataframe(fc.context, hide_index=True, width="stretch")
    st.write("**Limitations:**")
    for l in fc.limitations:
        st.markdown(f"- {l}")

# ---------------------------------------------------------------- B. win probability
st.header("Win probability")
st.caption("Estimates from simulation frequencies, not certainties. Error bars are 95% Wilson intervals for "
           "Monte Carlo sampling error only; they do NOT include model or data uncertainty.")
top = S.head(15).iloc[::-1]
fig = go.Figure(go.Bar(
    x=top.win_p * 100, y=top.driver, orientation="h", marker_color=RED,
    error_x=dict(type="data", symmetric=False, array=(top.win_hi - top.win_p) * 100,
                 arrayminus=(top.win_p - top.win_lo) * 100, color="#aaa"),
    customdata=top[["team", "podium_p"]].assign(podium_p=lambda d: d.podium_p * 100),
    hovertemplate="%{y} (%{customdata[0]})<br>Win %{x:.1f}%<br>Podium %{customdata[1]:.1f}%<extra></extra>"))
fig.update_xaxes(title="Win probability (%)")
st.plotly_chart(style(fig, 560), width="stretch")
tbl = S[["driver", "team", "win_p", "podium_p", "top5_p", "top10_p", "dnf_p", "exp_pos", "pos_sd"]].copy()
tbl.columns = ["Driver", "Team", "Win", "Podium", "Top 5", "Top 10", "Retire", "Exp. pos", "Pos. SD"]
st.dataframe(tbl.style.format({"Win": "{:.1%}", "Podium": "{:.1%}", "Top 5": "{:.1%}", "Top 10": "{:.1%}",
                               "Retire": "{:.1%}", "Exp. pos": "{:.1f}", "Pos. SD": "{:.1f}"}),
             hide_index=True, width="stretch")

with st.expander("Model validation (chronological backtest on completed races)"):
    if st.button("Run backtest"):
        with st.spinner("Forecasting each completed race using only earlier races..."):
            df, summ = backtest(bundle, n_sims=1500)
        if df.empty:
            st.info(summ["note"])
        else:
            st.dataframe(df.round(3), hide_index=True, width="stretch")
            st.json({k: (round(v, 3) if isinstance(v, float) else v) for k, v in summ.items()})
            st.caption("Compare 'model' with the 'uniform' and 'grid' baselines (lower is better). "
                       "A handful of races cannot establish accuracy.")
    else:
        st.caption("No accuracy claim is made unless you run this on real data, and even then samples are small.")

# ---------------------------------------------------------------- C. pace
st.header("Race pace analysis")
st.caption("Seconds per lap relative to the field average (negative = faster). Teams without data are shown as "
           "assumed-equal and labelled.")
p = inputs.copy()
p["label"] = np.where(p.pace_source == "season data", "from data", "ASSUMED")
cc = st.columns(2)
f1 = px.bar(p.sort_values("race_pace_s"), x="race_pace_s", y="driver", color="team", orientation="h",
            error_x="pace_sd_s", pattern_shape="label", pattern_shape_map={"from data": "", "ASSUMED": "/"},
            labels={"race_pace_s": "Race pace vs field (s/lap, ±1 SD)"}, title="Race pace by driver")
f1.update_yaxes(autorange="reversed")
cc[0].plotly_chart(style(f1, 600), width="stretch")
f2 = px.scatter(p, x="quali_pace_s", y="race_pace_s", color="team", text="driver",
                labels={"quali_pace_s": "Qualifying pace (s/lap equiv.)", "race_pace_s": "Race pace (s/lap)"},
                title="Qualifying vs race pace")
f2.update_traces(textposition="top center", marker_size=10)
cc[1].plotly_chart(style(f2, 600), width="stretch")
team = p.groupby("team").agg(race=("race_pace_s", "mean"), quali=("quali_pace_s", "mean")).reset_index().melt(
    "team", var_name="session", value_name="s_per_lap")
f3 = px.bar(team.sort_values("s_per_lap"), x="team", y="s_per_lap", color="session", barmode="group",
            title="Constructor pace (mean of drivers)", labels={"s_per_lap": "s/lap vs field"})
st.plotly_chart(style(f3, 400), width="stretch")
st_cur = bundle.stints[bundle.stints.year == dl.TARGET_YEAR] if len(bundle.stints) else bundle.stints
if len(st_cur) >= 10:
    f4 = px.box(st_cur, x="compound", y="deg_s_per_lap", color="compound", points="all",
                title="Tyre degradation per stint, 2026 (fuel-corrected, ASSUMED 0.03 s/lap fuel effect)",
                labels={"deg_s_per_lap": "Apparent wear (s per lap of tyre age)"})
    st.plotly_chart(style(f4, 400), width="stretch")
else:
    st.info("Tyre degradation: not enough measured stints, so the simulator uses ASSUMED default wear rates.")

# ---------------------------------------------------------------- D. scenario
st.header("Race scenario simulator")
st.caption("Every control below feeds the lap-by-lap simulator; the baseline is the forecast above.")
with st.form("scenario"):
    a, b = st.columns(2)
    grid_mode = a.selectbox("Starting grid", ["Baseline (actual qualifying if loaded, else simulated)",
                                              "Custom grid (edit below)"])
    sims = b.number_input(f"Monte Carlo simulations (max {MAX_SIMS:,})", 1000, MAX_SIMS, n_sims, step=1000, key="n_sims_in")
    s1, s2, s3 = st.columns(3)
    p_sc = s1.slider("Safety-car probability", 0.0, 1.0, float(base_cfg.p_sc), 0.05)
    p_vsc = s1.slider("Virtual safety-car probability", 0.0, 1.0, float(base_cfg.p_vsc), 0.05)
    p_wet = s2.slider("Wet-race probability", 0.0, 1.0, float(base_cfg.p_wet), 0.05)
    pit = s2.slider("Pit-stop time loss (s)", 14.0, 32.0, float(round(base_cfg.pit_loss_s, 1)), 0.5)
    mech = s3.slider("Mechanical retirement multiplier", 0.0, 4.0, 1.0, 0.1)
    inc = s3.slider("Incident multiplier", 0.0, 4.0, 1.0, 0.1)
    default_grid = (fc.grid if fc.grid is not None else np.arange(1, len(inputs) + 1)).astype(int)
    custom = st.data_editor(pd.DataFrame({"driver": inputs.driver, "grid": default_grid}), hide_index=True,
                            disabled=["driver"], width="stretch", key="grid_editor") \
        if grid_mode.startswith("Custom") else None
    go_btn = st.form_submit_button("Run scenario")
st.session_state["n_sims"] = int(sims)
scen_cfg = replace(base_cfg, n_sims=int(sims), p_sc=p_sc, p_vsc=p_vsc, p_wet=p_wet, pit_loss_s=pit,
                   mech_mult=mech, incident_mult=inc)
scen_grid = fc.grid
if custom is not None:
    g = custom.grid.to_numpy()
    if sorted(g) == list(range(1, len(g) + 1)):
        scen_grid = g
    else:
        st.error("Custom grid must use each position 1..%d exactly once; using baseline grid instead." % len(g))
if int(sims) != base_cfg.n_sims:   # keep baseline on the same simulation count for a fair comparison
    base = simulate(inputs, replace(base_cfg, n_sims=int(sims)).__dict__, fc.grid)
scen = simulate(inputs, scen_cfg.__dict__, scen_grid)
cmp = compare_results(base, scen)
changed = (scen_cfg != replace(base_cfg, n_sims=int(sims))) or (scen_grid is not fc.grid)
st.caption("Baseline and scenario use the same random seed, so differences reflect the inputs you changed."
           if changed else "Scenario currently equals the baseline.")
d = cmp.head(12).reset_index().melt("driver", ["win_base", "win_alt"], "run", "p")
d["run"] = d.run.map({"win_base": "Baseline", "win_alt": "Scenario"})
fd = px.bar(d, x="driver", y=d.p * 100, color="run", barmode="group", color_discrete_map={"Baseline": "#6b7280", "Scenario": RED},
            labels={"y": "Win probability (%)"}, title="Win probability: baseline vs scenario")
st.plotly_chart(style(fd, 420), width="stretch")

# ---------------------------------------------------------------- E. distribution
st.header("Race outcome distribution")
default_sel = list(scen.summary.driver.head(3))
sel = st.multiselect("Drivers", list(inputs.driver), default_sel)
if sel:
    rows = []
    for name, r in (("Baseline", base), ("Scenario", scen)):
        for dr in sel:
            for pos, pr in r.pos_dist.loc[dr].items():
                rows.append((name, dr, pos, pr))
    dist = pd.DataFrame(rows, columns=["run", "driver", "position", "probability"])
    fe_ = px.line(dist, x="position", y=dist.probability * 100, color="driver", line_dash="run",
                  labels={"y": "Probability (%)", "position": "Finishing position (22 = last / retired)"},
                  title="Finishing-position distribution")
    st.plotly_chart(style(fe_, 460), width="stretch")
cp = st.columns(2)
for col, name, r in ((cp[0], "Baseline", base), (cp[1], "Scenario", scen)):
    t = r.podium_combos.head(10).copy()
    t["probability"] = (t.probability * 100).round(2).astype(str) + "%"
    col.markdown(f"**Most frequent podiums - {name}**")
    col.dataframe(t, hide_index=True, width="stretch")
st.markdown("**Scenario vs baseline (all drivers)**")
show = cmp.reset_index()[["driver", "win_base", "win_alt", "win_delta", "exp_pos_base", "exp_pos_alt", "dnf_base", "dnf_alt"]]
st.dataframe(show.style.format({c: "{:.1%}" for c in ["win_base", "win_alt", "win_delta", "dnf_base", "dnf_alt"]}
                               | {"exp_pos_base": "{:.1f}", "exp_pos_alt": "{:.1f}"}),
             hide_index=True, width="stretch")
st.caption(f"Simulator diagnostics (scenario): {scen.diagnostics}")
