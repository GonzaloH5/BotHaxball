"""Sondas held-out: censura, conservación, lado azul, pool y macros reproducibles."""
from types import SimpleNamespace
import json
import sys

import numpy as np
import pytest
import torch

from env.rs4z.core import RS4ZEnv
from learn import x4_coop_eval as CE
from tools import x4_metrics as XM


def _episode(T=100):
    pos = np.full((T, 8, 2), 3000.0)
    pos[:, 0] = [-20.0, 0.0]
    pos[:, 1] = [200.0, 0.0]
    pos[:, 4] = [70.0, 70.0]
    ball = np.zeros((T, 4))
    ball[10:, 0] = 180.0
    return XM.Episode(stride=3, ball=ball, ball_r=8.325, pos=pos, vel=np.zeros((T, 8, 2)),
                      move=np.zeros((T, 8), int), kick_key=np.zeros((T, 8), bool),
                      kicked=np.zeros((T, 8), bool), open_play=np.ones(T, bool),
                      restart_kind=np.zeros(T, int), restart_age=np.zeros(T, int), kickoff=np.zeros(T, bool))


def _data(T=120):
    pos = np.full((T, 8, 2), 3000.0)
    pos[:, 0] = [180.0, 0.0]
    pos[:, 1] = [400.0, 100.0]
    pos[:, 4] = [300.0, 40.0]
    ball = np.zeros((T, 4))
    ball[:, 0] = 200.0
    return SimpleNamespace(names=["test_rec.hbr2"], ticks=T, ball=ball, ball_r=np.full(T, 8.325),
                           pos=pos, vel=np.zeros((T, 8, 2)), inp=np.zeros((T, 8), np.uint8),
                           kicking=np.zeros((T, 8), bool), state=np.ones(T, int),
                           rkind=np.zeros(T, int), rage=np.zeros(T, int), kstr=np.full(T, 5.85),
                           mass=np.full(T, 0.3), rec_of_tick=np.zeros(T, int),
                           pid=np.tile(np.arange(8), (T, 1)), kick_ev=np.zeros((T, 8), bool),
                           valid=np.arange(45, T - 2))


def test_late_reception_is_censored_and_goal_before_window_is_success():
    ep = _episode(30)
    row = CE.probe_row(ep, 0, 0)
    assert row["passes"] == 1
    assert row["reception_censored"] == 1
    assert row["reception_observed"] == 0
    assert CE.probe_rates([row])["retention_after_reception"] is None
    goal = CE.probe_row(ep, 0, 0, "goal")
    assert goal["goal"] == 1 and goal["reception_retained"] == 1


def test_reception_lost_under_pressure_counts_failure():
    ep = _episode(30)
    ep.pos[:10, 4] = [50.0, 0.0]
    ep.pos[20:, 4] = [160.0, 0.0]
    ep.pos[20:, 1] = [500.0, 0.0]
    row = CE.probe_row(ep, 0, 0, "loss")
    assert row["reception_observed"] == 1
    assert row["reception_retained"] == 0
    assert row["loss"] == 1
    assert row["pressure_loss_no_pass"] == 0


def test_useful_pass_retains_and_exploits_progress():
    ep = _episode()
    ep.ball[20:, 0] = 260.0
    ep.pos[20:, 1] = [280.0, 0.0]
    row = CE.probe_row(ep, 0, 0)
    assert row["reception_retained"] == 1
    assert row["useful_passes"] == 1
    assert row["max_progress"] == 260.0


def test_goal_without_pass_counts_and_blue_progress_is_mirrored():
    ep = _episode()
    ep.pos[:, 1] = [3000.0, 3000.0]
    red = CE.probe_row(ep, 0, 0, "goal")
    assert red["goal"] == 1 and red["passes"] == 0
    ep.pos = ep.pos[:, [4, 5, 6, 7, 0, 1, 2, 3]].copy()
    ep.pos[:, :, 0] *= -1
    ep.ball[:, 0] *= -1
    blue = CE.probe_row(ep, 1, 4, "goal")
    assert blue["max_progress"] == red["max_progress"]


def test_bootstrap_clusters_by_recording_and_reports_support():
    rows = [CE.probe_row(_episode(), 0, 0), CE.probe_row(_episode(30), 0, 0)]
    result = CE.summarize_probes(rows, [7, 7], seed=123)
    assert result["recording_units"] == 1
    assert result["support"]["reception_censored"] == 1
    assert result["metrics"]["retention_after_reception"] == {"valor": 1.0, "ic90": [1.0, 1.0]}
    assert result == CE.summarize_probes(rows, [7, 7], seed=123)


def test_state_sampler_has_no_duplicate_or_contested_initial_possession():
    data = _data(1000)
    a = CE.select_states(data, 2, 1, 10, seed=9, min_spacing=300)
    b = CE.select_states(data, 2, 1, 10, seed=9, min_spacing=300)
    assert [s[:3] for s in a] == [s[:3] for s in b]
    assert 1 <= len(a) <= 4
    assert all(abs(x[0] - y[0]) >= 300 for i, x in enumerate(a) for y in a[i + 1:])
    assert all(s[3].sum() == 3 for s in a)
    data.pos[:, 4] = data.ball[:, :2]
    assert CE.select_states(data, 2, 1, 10, seed=9) == []


def test_restore_human_queue_world_frame_and_physics():
    data = _data()
    data.inp[:, 4] = 8       # derecha del mundo: no espejar para azul en act_hist
    data.ball_r[:] = 9.0
    data.kstr[:] = 5.7
    env = RS4ZEnv(1, map=CE.MAP, max_delay=15)
    active = np.zeros(8, bool)
    active[[0, 1, 4]] = True
    delays = CE.restore_states(env, data, [(60, 0, 0, active)], seed=23)
    assert np.isin(delays, CE.DELAYS).all()
    assert (env.act_hist[0, 4] == 3).all()
    assert env._s_act[0, 4] == 3
    assert env.radius[0, 0] == 9.0
    assert np.array_equal(env.active[0], active)
    assert np.allclose(env.ball_pos[0], [200.0, 0.0])


def test_model_sampling_uses_independent_reproducible_generator():
    class Uniform(torch.nn.Module):
        def forward(self, obs, pass_mask=None):
            return torch.zeros((len(obs), 18))
    p = SimpleNamespace(model=Uniform(), device="cpu", name="uniform", greedy=False)
    obs = np.zeros((40, 168), np.float32)
    a = CE.seeded_policy(p, 17)(obs, np.random.default_rng(0))
    torch.manual_seed(999)
    b = CE.seeded_policy(p, 17)(obs, np.random.default_rng(800))
    assert np.array_equal(a, b)


def test_macro_policy_receives_mask_and_produces_primitive_actions():
    class MacroPolicy:
        n_pass = 3
        model = SimpleNamespace(pass_ready=False)
        def __call__(self, obs, rng, pass_mask=None):
            assert pass_mask is not None
            return np.where(pass_mask.any(1), 18 + pass_mask.argmax(1), 0)
    data = _data()
    env = RS4ZEnv(1, map=CE.MAP, max_delay=15)
    active = np.ones(8, bool)
    CE.restore_states(env, data, [(60, 0, 0, active)], seed=2)
    env.player_pos[0, 4:] = np.array([[800.0, 400.0], [800, -400], [1000, 300], [1000, -300]])
    macro = CE.MacroState(1)
    actions = CE.choose_actions(env, MacroPolicy(), CE.Policy("still"), np.array([0]),
                                np.random.default_rng(0), macro)
    assert actions.shape == (1, 8) and actions.max() < 18
    assert macro.running().any()


def test_probe_smoke_preserves_active_roster_and_reports_no_training():
    data = _data()
    active = np.zeros(8, bool)
    active[[0, 1, 4]] = True
    rows, delays = CE.run_probes(CE.Policy("still"), CE.Policy("still"), data,
                                [(60, 0, 0, active)], horizon=3, seed=9)
    assert len(rows) == 1 and len(delays) == 1
    assert rows[0]["passes"] == 0 and rows[0]["reception_observed"] == 0
    human = CE.human_probes(data, [(60, 0, 0, active)], horizon=3)
    assert len(human) == 1


def test_incomplete_matches_are_not_adjudicated_as_draws():
    result = CE.match_summary([[0, 0], [1, 0]], [], finished=1, safety=0, seed=7)
    assert result["valid_for_strength"] is False
    assert result["wins"] is None and result["draws"] is None
    assert result["point_fraction"]["valor"] is None
    assert result["scores_at_cap"] == [[0, 0], [1, 0]]


def test_probe_split_accepts_only_dev_or_test():
    args = ["--policy", "still", "--bc", "random", "--out", "unused.json"]
    assert CE.parse_args(args).split == "test"
    assert CE.parse_args(args + ["--split", "dev"]).split == "dev"
    with pytest.raises(SystemExit):
        CE.parse_args(args + ["--split", "train"])


@pytest.mark.parametrize("split", ["dev", "test"])
def test_report_names_actual_held_out_split(monkeypatch, tmp_path, split):
    out = tmp_path / "report.json"
    monkeypatch.setattr(sys, "argv", ["coop-eval", "--policy", "still", "--bc", "random",
                                    "--split", split, "--probe-states", "0", "--out", str(out)])
    monkeypatch.setattr(CE, "evaluate_matches", lambda *args, **kwargs:
                        dict(wins=0, draws=0, losses=0, point_fraction=dict(valor=None)))
    CE.main()
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["probe_split"] == split
    assert report["training_access"] is False
