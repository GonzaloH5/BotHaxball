import numpy as np
import pytest
import torch
import yaml

from env.rewards import RewardConfig
from env.rs4_tactics import components, dynamic_targets
from env.tasks import load_catalog, make_env
from train import bc
from env.haxball_env import U_SELF_DIM, U_ENT_DIM
from tools.upgrade_rs4 import upgrade


def restart(team=0, optimized=True):
    env = make_env(load_catalog()["rs4_4v4"], 1, 7, seed=12, random_reset_prob=0,
                   optimize_rollout=optimized, frame_skip=3,
                   reward=RewardConfig(shaping_coef=0, kickoff_approach=0, team_spread_floor=0,
                                       rs4_restart_approach=.05, rs4_restart_stall=.25))
    env.reset()
    sim = env.sim
    sim.kickoff[:] = False
    sim.mask[:] = sim.base_mask
    sim.vel[:] = 0
    sim.player_pos[:] = [0, 0]
    sim.ball_pos[:] = [(1 if team == 0 else -1) * (env.field_w + 20), 200]
    env.last_touch[:] = 1 - team
    env._set_piece([0])
    env._phi = env._potentials()
    return env


@pytest.mark.parametrize("optimized", [True, False])
@pytest.mark.parametrize("team", [0, 1])
def test_corner_not_stolen_and_timeout_is_one_penalized_terminal(team, optimized):
    env = restart(team, optimized)
    sim = env.sim
    rival = np.flatnonzero(sim.player_team != team)[0]
    origin = sim.ball_pos.copy()
    actions = np.zeros((1, 8), int)
    actions[0, rival] = 9
    for _ in range(60):  # 3 seconds at 60Hz, opponent tries to steal every decision
        sim.player_pos[0, rival] = origin[0] - [20, 0]
        _, _, done, info = env.step(actions)
        assert not done[0] and not info["kicked"][0, rival]
        assert env.setpiece_team[0] == team
        np.testing.assert_array_equal(sim.ball_pos, origin)
    before = env._rs4_restart_potential()
    env.setpiece_ticks[:] = env.setpiece_limit - 1
    _, rewards, done, info = env.step(actions)
    assert done[0] and not info["truncated"][0] and info["rs4_restart_failed"][0]
    assert not info["kicked"][0, rival]
    assert info["events"]["restart_timeouts"][0, team] == 1
    np.testing.assert_allclose(rewards[0, sim.player_team == team], (-before - .25)[0, sim.player_team == team], atol=1e-7)
    assert (rewards[0, sim.player_team == team] < 0).all()


@pytest.mark.parametrize("team", [0, 1])
def test_owner_executes_without_auto_action_or_false_timeout(team):
    env = restart(team)
    own = np.flatnonzero(env.sim.player_team == team)[0]
    env.sim.player_pos[0, own] = env.sim.ball_pos[0] + [(1 if team == 0 else -1) * 14, 14]
    actions = np.zeros((1, 8), int)
    actions[0, own] = 9
    _, _, done, info = env.step(actions)
    assert info["kicked"][0, own] and not done[0]
    assert not info["rs4_restart_failed"][0]
    assert env.setpiece_team[0] == -1


def test_restart_potential_shared_and_stationary_does_not_pay():
    env = restart()
    before = env._rs4_restart_potential()
    env.sim.player_pos[0, 0] = env.sim.ball_pos[0] - [100, 0]
    closer = env._rs4_restart_potential()
    assert (closer[0, :4] > before[0, :4]).all()
    assert (env.rcfg.gamma * closer - closer <= 0).all()
    # Including terminal, no positive discounted farming cycle from starting state.
    series = [before, closer, before, np.zeros_like(before)]
    total = sum(env.rcfg.gamma**i * (env.rcfg.gamma * series[i+1] - series[i]) for i in range(3))
    np.testing.assert_allclose(total, -before, atol=1e-12)


def test_dynamic_block_advances_and_preserves_depth_after_loss():
    ball = np.array([550., 120.])
    attack = dynamic_targets(ball, .95, 1000., 600., 120.)
    defense = dynamic_targets(ball, .05, 1000., 600., 120.)
    assert attack[0, 0] > defense[0, 0] + 300
    assert attack[0, 0] < ball[0] - 400
    assert attack[3, 0] > ball[0] > attack[2, 0]
    assert defense[0, 0] < -850
    assert defense[2, 0] < defense[1, 0] < defense[3, 0]


def test_v2_finite_bounded_and_team_identity_vertical_symmetries():
    rng = np.random.default_rng(20)
    players = rng.uniform(-800, 800, (12, 8, 2))
    balls = rng.uniform(-650, 650, (12, 2))
    team = np.repeat([0, 1], 4)
    def calc(p, b):
        return components(p, team, b, 1000., 600., 120., 2)
    base = calc(players, balls)
    assert np.isfinite(base).all() and ((base >= 0) & (base <= 1)).all()
    np.testing.assert_allclose(calc(players * [1, -1], balls * [1, -1]), base, atol=1e-12)
    np.testing.assert_allclose(calc(players[:, [3, 2, 0, 1, 6, 7, 4, 5]], balls), base, atol=1e-12)
    np.testing.assert_allclose(calc(players[:, np.r_[4:8, 0:4]] * [-1, 1], balls * [-1, 1]), base[:, ::-1], atol=1e-12)


def test_v2_prefers_depth_and_balance_over_everyone_chasing():
    ball = np.array([[450., 50.]])
    own = dynamic_targets(ball[0], .99, 1000., 600., 120.)
    opponent = np.array([[-700., -300], [-700, 300], [-850, 0], [-600, 0]])
    team = np.repeat([0, 1], 4)
    organized = np.concatenate((own, opponent))[None]
    crowded = organized.copy()
    crowded[:, :4] = ball[:, None]
    def score(p):
        return components(p, team, ball, 1000., 600., 120., 2)[0, 0, 0]
    assert score(organized) > score(crowded) + .2


def test_filtered_bc_uses_only_rs_one_4v4_and_aligned_labels(tmp_path, monkeypatch):
    monkeypatch.setattr(bc, "DATA", tmp_path)
    for folder, stadium in (("rsx4", "rs_one"), ("futsalx3", "futsalx3")):
        directory = tmp_path / folder
        directory.mkdir()
        for name in ("a", "b"):
            obs = np.zeros((3, U_SELF_DIM + 7 * U_ENT_DIM))
            obs[:, 0] = [11, 22, 33]
            np.savez(directory / f"{name}.npz", stadium=stadium, weight=1.,
                     act=[1, 2, 3], act_lag6=[4, 5, 6], T=[4, 3, 4], obs_0=obs,
                     powershot=False)
    data, e = bc.load_dataset(.5, folders=["rsx4"], stadium="rs_one", team_size=4)
    assert e == 7
    rows = np.concatenate([part["obs"][:, 0] for part in data.values()])
    labels = np.concatenate([part["act"] for part in data.values()])
    np.testing.assert_array_equal(sorted(rows), [11, 11, 33, 33])
    np.testing.assert_array_equal(sorted(labels), [1, 1, 3, 3])
    assert all((p["stadium"] == "rs_one").all() for p in data.values())


def test_upgrade_keeps_ppo_bc_decay_and_checkpoint_bytes(tmp_path):
    cfg = dict(stages=[dict(tasks=["rs4_4v4"])], reward={}, log={}, bc_reference="runs/bc2/bc.pt",
               ppo=dict(schedule_steps=3000000000, lr_final=5e-5),
               rs4_tactics=dict(coef=.12, coef_final=0., start_steps=10, decay_steps=200000000))
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    torch.save(dict(steps=4000000000, model={}), tmp_path / "latest.pt")
    checkpoint = (tmp_path / "latest.pt").read_bytes()
    upgrade(path, tmp_path / "preview.yaml")
    preview = yaml.safe_load((tmp_path / "preview.yaml").read_text())
    assert preview["ppo"] == cfg["ppo"] and preview["bc_reference"] == cfg["bc_reference"]
    assert preview["rs4_tactics"]["start_steps"] == 10
    upgrade(path, renew_guide=True)
    new = yaml.safe_load(path.read_text())
    assert new["rs4_tactics"]["start_steps"] == 4000000000 and new["rs4_tactics"]["coef"] == .08
    assert new["ppo"] == cfg["ppo"] and (tmp_path / "latest.pt").read_bytes() == checkpoint
    assert len(list(tmp_path.glob("config_before_rs4_v2_*.yaml"))) == 1
    with pytest.raises(ValueError, match="no reiniciar"):
        upgrade(path, renew_guide=True)


def test_bc_fine_tuning_writes_separate_reference_not_ppo(tmp_path, monkeypatch):
    import sys
    from train.model import SetActorCritic
    data_dir = tmp_path / "data" / "rsx4"
    data_dir.mkdir(parents=True)
    import hashlib
    names = {}
    for i in range(100):
        name = f"game{i}.npz"
        split = int(hashlib.md5(name.encode()).hexdigest(), 16) % 1000 < 500
        names.setdefault(split, name)
    for name in names.values():
        np.savez(data_dir / name, stadium="rs_one", weight=1., act=np.arange(16) % 18,
                 act_lag6=np.arange(16) % 18, T=np.full(16, 4), powershot=False,
                 obs_0=np.zeros((16, U_SELF_DIM + 7 * U_ENT_DIM)))
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump(dict(model=dict(hidden=16, layers=1, ent_hidden=8, ent_layers=1),
                                     stages=[dict(tasks=["rs4_4v4"])])))
    model = SetActorCritic(U_SELF_DIM, U_ENT_DIM, hidden=16, layers=1, ent_hidden=8, ent_layers=1)
    source = tmp_path / "bc_parent.pt"
    torch.save(dict(model=model.state_dict(), model_config=model.config(), bc={}), source)
    original = source.read_bytes()
    monkeypatch.setattr(bc, "DATA", tmp_path / "data")
    monkeypatch.setattr(bc, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["bc", "--run", "specialist", "--config", str(cfg),
                                     "--init-from", str(source), "--epochs", "1", "--batch", "8",
                                     "--threads", "2", "--folders", "rsx4", "--stadium", "rs_one",
                                     "--team-size", "4", "--val-frac", ".5"])
    previous_threads = torch.get_num_threads()
    try:
        bc.main()
    finally:
        torch.set_num_threads(previous_threads)
    saved = torch.load(tmp_path / "runs/specialist/bc.pt", weights_only=False)
    assert saved["bc"]["team_size"] == 4 and saved["bc"]["stadium"] == "rs_one"
    assert saved["model_config"] == model.config() and source.read_bytes() == original
    with pytest.raises(SystemExit):
        bc.main()  # never overwrite an already trained specialist
    torch.set_num_threads(previous_threads)


def test_converter_filters_each_map_change_and_team_size(tmp_path, monkeypatch):
    import json
    from tools import build_bc_dataset as builder
    trace = tmp_path / "trace"
    trace.mkdir()
    meta = dict(stadium="rs_one", powershot=False, out_of_bounds=True, script=None)
    futsal = dict(stadium="futsalx3", powershot=False, out_of_bounds=False, script=None)
    lines = [dict(type="header", stadiumFile="initial.hbs")]
    for frame, stadium, size in ((0, "RS", 4), (1, "Futsal", 4), (2, "RS", 3), (3, "RS", 4)):
        if frame:
            lines.append(dict(type="event", event="stadium_change", frame=frame,
                              stadium=stadium, stadiumFile=f"map{frame}.hbs"))
        players = [dict(id=p, team=1 if p < size else 2, disc=p+1, input=0, kicking=False)
                   for p in range(2*size)]
        lines.append(dict(type="tick", frame=frame, state=1, ko=1, players=players,
                          discs=[dict(x=0, y=0, vx=0, vy=0)] * (1+2*size)))
    def fake_node(*args):
        (trace / "replay.jsonl").write_text("\n".join(json.dumps(row, separators=(",", ":")) for row in lines))
        return ""
    monkeypatch.setattr(builder, "node", fake_node)
    monkeypatch.setattr(builder, "replay_stadium", lambda path, m: (m["stadium"], 1))
    accepted = []
    class FakeLoader:
        B = 512
        def __init__(self, path, size, powershot, out):
            self.size = size
            assert path == "rs_one" and size == 4
        def obs(self, rows):
            accepted.extend(row[0]["frame"] for row in rows)
            n = len(rows) * 2 * self.size
            return np.zeros((n, U_SELF_DIM + 7*U_ENT_DIM)), np.zeros(n, dtype=np.uint8)
        def actions(self, rows, label):
            return np.zeros(len(rows)*2*self.size, dtype=np.uint8)
    monkeypatch.setattr(builder, "Loader", FakeLoader)
    result = builder.process_replay(tmp_path / "replay.hbr2", meta, 1, trace, {},
                                    np.random.default_rng(0), {"RS": meta, "Futsal": futsal}, "rs_one", 4)
    assert accepted == [0, 3]
    assert np.concatenate(result["T"]).tolist() == [4] * 16
