"""Controlador de pase X4: la predicción con las teclas en cola coincide con el simulador y el plan patea alineado."""
import json

import numpy as np

from env.rs4z import contract as C
from env.rs4z.core import RS4ZEnv
from learn.x4_pass_macro import PassMacro, queued_ticks, roll_queue


def test_queued_ticks_matches_delay():
    assert [queued_ticks(d) for d in (0, 1, 8, 9, 10, 11)] == [0, 1, 8, 9, 10, 11]


def test_roll_queue_matches_kernel():
    rng = np.random.default_rng(1)
    for delay in (6, 9, 12):
        env = RS4ZEnv(1, map="sanguchito_rs_x4", frame_skip=3, max_delay=15, deadline=0,
                      kickoff_deadline=C.KICKOFF_SAFETY, seed=0)
        env.start_match(np.arange(1), delay=np.full((1, 8), delay), match_ticks=3 * 3600)
        preds = {}
        for s in range(24):
            acts = np.zeros((1, 8), int)
            acts[0, 0] = rng.integers(1, 9)
            p, _, T = roll_queue(env.player_pos[0, 0], env.player_vel[0, 0], env.act_hist[0, 0], delay)
            preds[s + T // 3] = p
            env.step(acts)
            if s >= 8 and s + 1 in preds:
                assert np.allclose(preds[s + 1], env.player_pos[0, 0], atol=1e-6)


def test_plan_kicks_when_aligned_and_in_reach():
    m = PassMacro(1)
    mates = np.array([[0.0, 0.0], [300.0, 0.0]] + [[0.0, 500.0]] * 6)
    # pelota dentro del alcance, en la línea del receptor: patea ya (la dirección es jugador → pelota)
    ball = np.array([15.0 + 8.325 + 2.0, 0.0])
    a = m.act(0, ball, np.zeros(2), np.zeros(2), np.zeros(2), mates, np.zeros((8, 2)), 0,
              hist=np.zeros(6, int), delay=0)
    assert a >= 9 and m.aim is not None
    # receptor detrás: no patea, se mueve
    m = PassMacro(1)
    mates[1] = [-300.0, 0.0]
    a = m.act(0, ball, np.zeros(2), np.zeros(2), np.zeros(2), mates, np.zeros((8, 2)), 0,
              hist=np.zeros(6, int), delay=0)
    assert a < 9


def test_pass_table_matches_python_controller():
    from learn.x4_pass_macro import TEAM, pass_table, to_primitive
    rng = np.random.default_rng(3)
    N = 48
    env = RS4ZEnv(N, map="sanguchito_rs_x4", frame_skip=3, max_delay=15, deadline=0,
                  kickoff_deadline=C.KICKOFF_SAFETY, seed=0)
    env.start_match(np.arange(N), delay=rng.choice([8, 9, 10, 11], size=(N, 8)), match_ticks=3 * 3600)
    for n in range(N):
        b = rng.uniform([-500, -200], [500, 200])
        pp = rng.uniform([-600, -250], [600, 250], size=(8, 2))
        c = n % 8
        ang = rng.uniform(0, 2 * np.pi)
        pp[c] = b + (15.0 + 8.325 + rng.uniform(0, 25)) * np.array([np.cos(ang), np.sin(ang)])
        env.place(n, ball_pos=b, ball_vel=rng.normal(0, 1.5, 2), player_pos=pp, player_vel=rng.normal(0, 1, (8, 2)))
    for _ in range(4):
        env.step(rng.integers(0, 18, size=(N, 8)))
    table = pass_table(env)
    checked = 0
    from env.rs4z import kernel as K
    for n in range(N):
        if env.ri[n, K.RI_KIND] > 0 or env.ri[n, K.RI_KO] != 0:
            continue                                  # saques: los compara test_pass_table_lateral_matches_python
        for p in range(8):
            mates = [q for q in range(8) if q != p and TEAM[q] == TEAM[p] and env.active[n, q]]
            for k, q in enumerate(mates[:3]):
                if table[n, p, k] < 0:
                    continue
                m = PassMacro(q, ball_r=float(env.radius[n, 0]))
                a = m.act(p, env.pos[n, 0], env.vel[n, 0], env.player_pos[n, p], env.player_vel[n, p],
                          env.player_pos[n], env.player_vel[n], 0, hist=env.act_hist[n, p], delay=int(env.delay[n, p]))
                assert a == table[n, p, k], (n, p, k)
                checked += 1
    assert checked >= 20
    acts = np.full((N, 8), 18)
    prim = to_primitive(acts, table)
    assert prim.max() < 18 and prim.min() >= 0


def test_policy_with_pass_head_starts_like_primitive_policy():
    import torch
    from env.rs4z import obs_v3
    from learn.x4_policy import SetPolicy
    torch.manual_seed(0)
    base = SetPolicy(hidden=32, ent_hidden=16)
    mac = SetPolicy(hidden=32, ent_hidden=16, n_pass=3)
    missing, unexpected = mac.load_state_dict(base.state_dict(), strict=False)
    assert not unexpected and all(k.startswith("pass_") for k in missing)
    obs = torch.randn(5, obs_v3.OBS_DIM)
    lp0 = torch.log_softmax(base(obs), -1)
    # sin pases ofrecidos: idéntica a la de 18 teclas
    lp = mac(obs, torch.zeros(5, 3, dtype=torch.bool))
    assert torch.allclose(lp[:, :18], lp0, atol=1e-5) and (lp[:, 18:] < -1e8).all()
    # con pases: la compuerta σ(−3) reparte su masa entre los ofrecidos; el resto, las teclas escaladas
    pm = torch.tensor([[True, False, True]] * 5)
    lp = mac(obs, pm)
    p = lp.exp()
    assert torch.allclose(p.sum(-1), torch.ones(5), atol=1e-5)
    assert torch.allclose(p[:, 18:].sum(-1), torch.sigmoid(torch.tensor(-3.0)).expand(5), atol=1e-5)
    assert (p[:, 19] == 0).all()


def test_trainer_with_pass_macro_runs_and_resumes(tmp_path):
    from pathlib import Path

    import pytest
    import torch
    from learn.x4_eval import ModelPolicy, play
    from learn.x4_ppo import Trainer, parse_args
    bc = Path(__file__).resolve().parents[2] / "runs" / "x4_bc" / "final_sangu_rsone" / "best.pt"
    if not bc.exists():
        pytest.skip("falta el checkpoint de la imitación")
    common = ["--bc", str(bc), "--out", str(tmp_path), "--device", "cpu", "--envs", "6", "--rollout", "8",
              "--critic-warmup", "0", "--bank-recordings", "0", "--eval-every", "0", "--threads", "1",
              "--epochs", "1", "--minibatches", "1", "--pass-macro", "--pool-frac", "0.5", "--snapshot-every", "1"]
    t = Trainer(parse_args(common + ["--updates", "2"]))
    assert t.policy.n_pass == 3
    t.train()
    ck = torch.load(tmp_path / "last.pt", map_location="cpu")
    assert ck["n_pass"] == 3 and any(k.startswith("pass_") for k in ck["model"])
    rows = [json.loads(x) for x in (tmp_path / "log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all("macro_starts" in r and np.isfinite(r["kl_bc"]) for r in rows)
    # el pool ya tiene un snapshot con pases: la reanudación lo carga y juega contra él
    t2 = Trainer(parse_args(common + ["--updates", "3", "--resume"]))
    assert any(getattr(m, "n_pass", 0) for _, m in t2.pool)
    t2.train()
    me = ModelPolicy(t2.policy, "cpu", "aprendiz")
    r = play(me, me, matches=2, minutes=0.2, record=0)
    assert r["goals"].shape == (2, 2)


def test_dev_metrics_with_pass_head(tmp_path):
    from pathlib import Path

    import pytest
    import torch
    from env.rs4z import obs_v3
    from learn.x4_ppo import Trainer, parse_args
    bc = Path(__file__).resolve().parents[2] / "runs" / "x4_bc" / "final_sangu_rsone" / "best.pt"
    if not bc.exists():
        pytest.skip("falta el checkpoint de la imitación")
    t = Trainer(parse_args(["--bc", str(bc), "--out", str(tmp_path), "--device", "cpu", "--envs", "2", "--rollout", "2",
                            "--bank-recordings", "0", "--eval-every", "0", "--pass-macro"]))
    t.dev_set = (torch.randn(16, obs_v3.OBS_DIM), torch.randint(0, 18, (16,)))
    m = t.dev_metrics()
    assert np.isfinite(m["kl_bc"]) and m["kl_bc"] < 1e-4      # recién inicializada: la cabeza de pase no cambia las teclas


def test_ready_table_only_offers_kicks():
    from learn.x4_pass_macro import pass_table
    rng = np.random.default_rng(5)
    N = 64
    env = RS4ZEnv(N, map="sanguchito_rs_x4", frame_skip=3, max_delay=15, deadline=0,
                  kickoff_deadline=C.KICKOFF_SAFETY, seed=0)
    env.start_match(np.arange(N), delay=rng.choice([8, 9, 10, 11], size=(N, 8)), match_ticks=3 * 3600)
    for n in range(N):
        b = rng.uniform([-500, -200], [500, 200])
        pp = rng.uniform([-600, -250], [600, 250], size=(8, 2))
        ang = rng.uniform(0, 2 * np.pi)
        pp[n % 8] = b + (15.0 + 8.325 + rng.uniform(0, 6)) * np.array([np.cos(ang), np.sin(ang)])
        env.place(n, ball_pos=b, ball_vel=np.zeros(2), player_pos=pp, player_vel=np.zeros((8, 2)))
    full, ready = pass_table(env), pass_table(env, ready_only=True)
    off = ready >= 0
    assert off.any() and (ready[off] >= 9).all() and (ready[off] == full[off]).all()
    assert not ((full >= 9) & ~off).any()


def test_pass_table_lateral_matches_python():
    """En un lateral propio la tabla sólo se ofrece al equipo que saca, desde ≤ 150 px, y coincide con el controlador
    Python sin empujar (no_push)."""
    from env.rs4z import kernel as K
    from learn.x4_pass_macro import pass_table
    from learn.x4_ppo import restart_spot
    rng = np.random.default_rng(7)
    N = 32
    env = RS4ZEnv(N, map="sanguchito_rs_x4", frame_skip=3, max_delay=15, deadline=0,
                  kickoff_deadline=C.KICKOFF_SAFETY, seed=0)
    env.start_match(np.arange(N), delay=rng.choice([8, 9, 10, 11], size=(N, 8)), match_ticks=3 * 3600)
    for n in range(N):
        bx, sy = rng.uniform(-800, 800), (1 if n % 2 else -1)
        pp = np.column_stack([rng.uniform(-900, 900, 8), rng.uniform(-450, 450, 8)])
        pp[0] = [bx + rng.uniform(-60, 60), sy * rng.uniform(560, 660)]
        env.place(n, ball_pos=np.array([bx, sy * 690.0]), ball_vel=np.zeros(2), player_pos=pp,
                  player_vel=np.zeros((8, 2)))
        env.start_restart(n, K.LATERAL, 0, restart_spot(env, K.LATERAL, np.array([bx, sy * 690.0])))
    table = pass_table(env)
    assert (table[:, 4:] < 0).all()                    # el rival no tiene pase durante el lateral
    checked = 0
    for n in range(N):
        mates = [q for q in range(1, 4) if env.active[n, q]]
        for k, q in enumerate(mates):
            if table[n, 0, k] < 0:
                continue
            m = PassMacro(q, ball_r=float(env.radius[n, 0]), no_push=True)
            a = m.act(0, env.pos[n, 0], env.vel[n, 0], env.player_pos[n, 0], env.player_vel[n, 0], env.player_pos[n],
                      env.player_vel[n], 0, hist=env.act_hist[n, 0], delay=int(env.delay[n, 0]))
            assert a == table[n, 0, k], (n, k)
            checked += 1
    assert checked >= 30
