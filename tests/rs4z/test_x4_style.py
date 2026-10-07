"""Recompensa de estilo humano (`learn/x4_style.py`)."""
import numpy as np
import torch

from env.rs4z import kernel as K
from learn import x4_style as XS


def _random_windows(m, seed=0):
    rng = np.random.default_rng(seed)
    w = np.zeros((m, len(XS.OFFSETS), XS.BODIES, 4), np.float32)
    w[..., 0] = rng.uniform(-1100, 1100, w.shape[:3])
    w[..., 1] = rng.uniform(-600, 600, w.shape[:3])
    w[..., 2:] = rng.normal(0, 2, w.shape[:3] + (2,))
    return w


def _mirror(win):
    """El mismo momento visto con los equipos cambiados: x y vx espejados y los lugares rojos ↔ azules."""
    m = win.copy()
    m[..., 0] *= -1
    m[..., 2] *= -1
    return m[:, :, [0, 5, 6, 7, 8, 1, 2, 3, 4]]


def test_features_are_the_same_for_a_team_and_its_mirror():
    win = _random_windows(50)
    a = XS.features(win, np.zeros(50, np.int64))
    b = XS.features(_mirror(win), np.ones(50, np.int64))
    assert a.shape == (50, XS.FEAT)
    assert np.allclose(a, b)


def test_features_order_players_by_distance_to_the_ball_now():
    win = _random_windows(1, seed=3)
    win[0, :, 0, :2] = 0.0                                   # pelota en el centro
    for k, x in zip(range(1, 5), (400.0, 100.0, 300.0, 200.0)):
        win[0, -1, k, :2] = (x, 0.0)                         # compañeros a 400, 100, 300 y 200 px ahora
    f = XS.features(win, np.zeros(1, np.int64)).reshape(len(XS.OFFSETS), XS.BODIES, 4)
    assert np.allclose(f[-1, 1:5, 0] * XS.SCALE[0], [100.0, 200.0, 300.0, 400.0])
    # el mismo orden en los momentos anteriores: el lugar 2 (el más cercano ahora) va primero también hace 1 s
    assert np.isclose(f[0, 1, 0] * XS.SCALE[0], win[0, 0, 2, 0])


class _FakeEnv:
    def __init__(self, n):
        self.pos = np.zeros((n, 1, 2))
        self.vel = np.zeros((n, 1, 2))
        self.player_pos = np.zeros((n, 8, 2))
        self.player_vel = np.zeros((n, 8, 2))
        self.ri = np.zeros((n, 32), np.int64)
        self.ri[:, K.RI_TEAM] = -1


def test_history_needs_a_whole_second_of_open_play():
    env = _FakeEnv(2)
    h = XS.History(2)
    for t in range(XS.DEPTH):
        env.pos[:, 0, 0] = t                                  # la pelota marca el número de muestreo
        if t == 5:
            env.ri[1, K.RI_TEAM] = 0                          # el partido 1 tiene un saque en el muestreo 5
        elif t == 6:
            env.ri[1, K.RI_TEAM] = -1
        h.push(env)
    win, valid = h.windows()
    assert valid.tolist() == [True, False]
    assert win[0, :, 0, 0].tolist() == [XS.DEPTH - 1 - o for o in XS.OFFSETS]
    h.reset([0])
    assert not h.windows()[1][0]


def test_episode_windows_skip_restarts_and_cover_both_teams():
    from tools.x4_metrics import Episode
    T = 60
    open_play = np.ones(T, bool)
    open_play[30] = False
    ep = Episode(stride=3, ball=np.zeros((T, 4)), ball_r=8.0, pos=np.zeros((T, 8, 2)), vel=np.zeros((T, 8, 2)),
                 move=np.zeros((T, 8), np.int64), kick_key=np.zeros((T, 8), bool), kicked=np.zeros((T, 8), bool),
                 open_play=open_play, restart_kind=np.zeros(T, np.int64), restart_age=np.zeros(T, np.int64),
                 kickoff=np.zeros(T, bool))
    w = XS.episode_windows(ep)
    # momentos válidos: 20..29 (antes del saque) y 51..59 (un segundo entero después): 19, por dos equipos
    assert w.shape == (2 * 19, XS.FEAT)


def _bank(path, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(0.5, 0.2, (n, XS.FEAT)).astype(np.float16)
    np.savez(path, x=x, x_dev=x[:500], mean=x.astype(np.float32).mean(0), std=x.astype(np.float32).std(0) + 1e-3,
             meta="{}")
    return x


def test_discriminator_learns_to_pay_more_for_human_like_windows(tmp_path):
    torch.manual_seed(0)
    human = _bank(tmp_path / "bank.npz")
    st = XS.Style(tmp_path / "bank.npz", "cpu", lr=1e-3, steps=150, batch=256, seed=0)
    agent = np.random.default_rng(1).normal(-0.5, 0.2, (4000, XS.FEAT)).astype(np.float32)
    st.add(agent, st.reward(agent))
    stats = st.fit()
    assert stats["style_d_human"] > stats["style_d_agent"]
    assert st.reward(human[:200].astype(np.float32)).mean() > st.reward(agent[:200]).mean() + 0.3
    r = st.reward(agent[:50])
    assert r.min() >= 0.0 and r.max() <= 1.0


def test_trainer_pays_style_and_resumes_the_discriminator(tmp_path):
    from pathlib import Path
    import json
    from learn.x4_ppo import Trainer, parse_args
    bc = Path(__file__).resolve().parents[2] / "runs" / "x4_bc" / "final_sangu_rsone" / "best.pt"
    if not bc.exists():
        import pytest
        pytest.skip("falta el checkpoint de la imitación")
    _bank(tmp_path / "bank.npz")
    common = ["--bc", str(bc), "--out", str(tmp_path / "run"), "--device", "cpu", "--envs", "4", "--rollout", "30",
              "--critic-warmup", "0", "--bank-recordings", "0", "--eval-every", "0", "--threads", "1",
              "--epochs", "1", "--minibatches", "1", "--snapshot-every", "0", "--style-coef", "0.01",
              "--style-bank", str(tmp_path / "bank.npz"), "--style-warmup", "0", "--style-steps", "2",
              "--style-batch", "64", "--human-starts", "0"]
    t1 = Trainer(parse_args(common + ["--updates", "2"]))
    env = t1.arenas[0].env
    rng = np.random.default_rng(0)
    for row in range(env.N):                                  # juego abierto (sin saque inicial que esperar)
        env.place(row, ball_pos=(0.0, 0.0), ball_vel=(0.0, 0.0), player_pos=rng.uniform(-500, 500, (8, 2)),
                  player_vel=np.zeros((8, 2)), last_touch=-1)
    t1.train()
    rows = [json.loads(l) for l in open(tmp_path / "run" / "log.jsonl")]
    assert all("style_d_human" in r and r["style_valid_frac"] >= 0 for r in rows)
    t2 = Trainer(parse_args(common + ["--updates", "3", "--resume"]))
    for p1, p2 in zip(t1.style.D.parameters(), t2.style.D.parameters()):
        assert torch.equal(p1.detach(), p2.detach())
    assert t2.style.r_w == t1.style.r_w
