import json
import re

import numpy as np

from src.race_simulator import SimConfig, run_simulation
from src.replay_view import build_html, replay_options, stint_table


def test_replay_is_consistent_with_simulation(inputs):
    r = run_simulation(inputs, SimConfig(n_sims=300, seed=4))
    rp = r.replay
    assert rp is not None and rp["t"].shape[0] == 12
    assert (np.diff(rp["t"], axis=1) > 0).all()                      # each car's clock only moves forward
    assert ((rp["positions"] == 1).sum(axis=1) == 1).all()
    assert len(replay_options(r)) == 12
    # recording the replay must not change the simulation itself
    assert r.summary.equals(run_simulation(inputs, SimConfig(n_sims=300, seed=4), replay_k=0).summary)


def test_html_payload_and_stints(inputs):
    r = run_simulation(inputs, SimConfig(n_sims=300, seed=4))
    html = build_html(r, inputs, 2)
    data = json.loads(re.search(r"const D=(\{.*?\});\n", html, re.S).group(1))
    assert len(data["drivers"]) == len(inputs) and len(data["race"]["t"]) == data["L"] + 1
    assert sorted(data["race"]["final"]) == list(range(len(inputs)))
    assert "NOT A REAL RACE" in html and "fallback" in html
    st = stint_table(r, 2)
    assert st.groupby("driver").laps.sum().le(data["L"]).all()
