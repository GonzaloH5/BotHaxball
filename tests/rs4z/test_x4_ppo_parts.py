"""Piezas del trainer X4 (learn/x4_ppo.py): shaping CHECKPOINT y acciones próximas que ve el crítico."""
from types import SimpleNamespace

import numpy as np

from env.rs4z import kernel as K
from env.rs4z.core import MIRROR_ACTION
from learn.x4_ppo import ORDER, TEAM, Arena, upcoming_actions

FAR = np.array([[-300.0, 0], [-600, 200], [-600, -200], [-900, 0], [300, 0], [600, 200], [600, -200], [900, 0]])


def _arena(n=2, shaping=1.0):
    a = SimpleNamespace(delay_values=np.array([0]), delay_weights=np.array([1.0]), match_minutes=(10.0, 10.0),
                        human_starts=0.0, human_restart_frac=0.0, pool_frac=0.0, lambda_values=[0.06], shaping=shaping)
    return Arena("sanguchito_rs_x4", n, a, np.random.default_rng(0))


def test_checkpoint_shaping_is_zero_sum_once_per_point_and_live_only():
    ar = _arena()
    env = ar.env
    env.place(0, ball_pos=(600.0, 0.0), ball_vel=(0.0, 0.0), player_pos=FAR, player_vel=np.zeros((8, 2)), last_touch=0)
    env.place(1, ball_pos=(600.0, 0.0), ball_vel=(0.0, 0.0), player_pos=FAR, player_vel=np.zeros((8, 2)), last_touch=0)
    env.ri[1, K.RI_TEAM] = 1          # partido 1: saque del script activo (no es juego vivo)
    env.ri[1, K.RI_KIND] = 1
    r = ar.shaping(np.zeros(2, np.int64))
    k = int(600.0 / 1150.0 * 10)       # franja 5: se cobran las franjas 0..5
    assert np.isclose(r[0, 0], 0.1 * (k + 1)) and np.isclose(r[0, 1], -r[0, 0])
    assert np.allclose(r[1], 0.0)
    assert np.allclose(ar.shaping(np.zeros(2, np.int64))[0], 0.0), "una sola vez por punto"
    # gol del rojo: cobra el resto de las franjas y se reinicia
    r = ar.shaping(np.array([1, 0]))
    assert np.isclose(r[0, 0], 0.1 * (10 - (k + 1))) and not ar.regions[0].any()


def test_shaping_can_be_retired():
    ar = _arena(shaping=0.0)
    ar.env.place(0, ball_pos=(900.0, 0.0), ball_vel=(0.0, 0.0), player_pos=FAR, player_vel=np.zeros((8, 2)), last_touch=0)
    assert np.allclose(ar.shaping(np.zeros(2, np.int64)), 0.0)


def test_upcoming_actions_order_and_mirror():
    ar = _arena(n=1)
    env = ar.env
    env.delay[:] = 3                                   # h = 1: se aplica act_hist[0] de antes del paso
    env.act_hist[0, :, 0] = np.arange(8) + 2          # acción del mundo de cada lugar
    up = upcoming_actions(env).reshape(1, 8, 8, 18)
    for p in range(8):
        for k, q in enumerate(ORDER[p]):
            a = env.act_hist[0, q, 0]
            if TEAM[p] == 1:
                a = MIRROR_ACTION[a]
            assert up[0, p, k, a] == 1.0 and up[0, p, k].sum() == 1.0
    assert ORDER[5][0] == 5 and set(ORDER[5][1:4]) == {4, 6, 7}


def test_bc_sees_2k23_as_rs_one():
    from env.rs4z import obs_v3
    from learn.x4_ppo import Trainer
    a = SimpleNamespace(delay_values=np.array([0]), delay_weights=np.array([1.0]), match_minutes=(10.0, 10.0),
                        human_starts=0.0, human_restart_frac=0.0, pool_frac=0.0, lambda_values=[0.06], shaping=1.0)
    ar = Arena("haxarg_2k23", 1, a, np.random.default_rng(0))
    obs = obs_v3.observe(ar.env)
    view = Trainer.bc_view(ar, obs)
    j = obs_v3.SELF_FEATURES.index("map_rs_one")
    assert (obs[0, :, j + 2] == 1).all() and (obs[0, :, j] == 0).all()
    assert (view[0, :, j] == 1).all() and (view[0, :, j + 2] == 0).all()
    other = np.ones(obs.shape[-1], bool)
    other[[j, j + 2]] = False
    assert np.array_equal(view[..., other], obs[..., other])
    sangu = _arena(n=1)
    o2 = obs_v3.observe(sangu.env)
    assert Trainer.bc_view(sangu, o2) is o2


def test_forfeit_penalty_is_zero_sum():
    from learn.x4_ppo import forfeit_reward
    r = forfeit_reward(np.array([-1, 0, 1]), 0.1)
    assert np.allclose(r, [[0, 0], [-0.1, 0.1], [0.1, -0.1]])
    # sólo las causas pedidas: el lateral mal ejecutado (1) ya cuesta la pelota y no se penaliza aparte
    r = forfeit_reward(np.array([0, 1, 0]), 0.1, why=np.array([K.FW_BAD_THROW, K.FW_KICKOFF, K.FW_LAT_TIME]),
                       reasons=(2, 3, 4))
    assert np.allclose(r, [[0, 0], [0.1, -0.1], [-0.1, 0.1]])


def test_kickoff_deadline_reports_the_forfeiting_team():
    ar = _arena(n=1)
    env = ar.env
    env.kickoff_deadline = 30
    team = int(env.ri[0, K.RI_KO_TEAM])
    lost = -1
    for _ in range(15):
        ev = env.step(np.zeros((1, 8), np.int64))
        if ev["forfeit"][0] >= 0:
            lost = int(ev["forfeit"][0])
            break
    assert lost == team and env.ri[0, K.RI_KO_TEAM] == 1 - team
    assert ev["forfeit_why"][0] == K.FW_KICKOFF


def test_upcoming_actions_is_what_the_kernel_applies_next():
    """Con retardo D, el primer tick del próximo paso aplica act_hist[⌈D/3⌉ − 1] de antes de `step`."""
    ar = _arena(n=1)
    env = ar.env
    rng = np.random.default_rng(3)
    for D in (8, 9, 10, 11, 14):
        env.delay[:] = D
        env.act_hist[0] = rng.integers(0, 18, size=env.act_hist.shape[1:])
        up = upcoming_actions(env).reshape(1, 8, 8, 18)
        h = (D + 2) // 3
        expected = env.act_hist[0, :, h - 1].copy()
        env.step(rng.integers(0, 18, size=(1, 8)))
        assert np.array_equal(env.act_hist[0, :, h], expected)      # lo que leyó el kernel en el tick 0
        for p in range(8):
            q = ORDER[p][0]
            a = expected[q] if TEAM[p] == 0 else MIRROR_ACTION[expected[q]]
            assert up[0, p, 0, a] == 1.0
    env.delay[:] = 0
    assert upcoming_actions(env).sum() == 0          # D = 0: la acción del próximo tick todavía no se decidió


def _touch(n, *slots):
    t = np.zeros((n, 8), bool)
    for s in slots:
        t[0, s] = True
    return t


def test_pass_tracker_counts_passes_losses_and_goal_possession():
    from learn.x4_ppo import PassTracker, pass_bonus_reward
    tr = PassTracker(1)
    none = np.zeros((1, 8), bool)
    g0 = np.zeros(1, np.int64)
    ball = lambda x: np.array([[x, 0.0]])
    assert tr.step(_touch(1, 0), none, ball(0.0), g0)[0][0] == 0          # primer toque del rojo 0
    assert tr.step(_touch(1, 1), none, ball(100.0), g0)[0][0] == 1        # pase 0 → 1 (100 px)
    assert tr.step(_touch(1, 2), none, ball(110.0), g0)[0][0] == 0        # 1 → 2 con 10 px: no es pase
    assert tr.step(none, _touch(1, 3), ball(300.0), g0)[0][0] == 1        # pase 2 → 3 con patada
    assert tr.poss_passes[0].tolist() == [2, 0]
    done, scored = tr.step(none, none, ball(1200.0), np.array([1]))       # gol del rojo
    assert scored[0] == 2 and tr.poss_passes[0].tolist() == [0, 0]
    r = pass_bonus_reward(np.array([1]), scored, 0.05)
    assert np.allclose(r, [[0.1, -0.1]])
    tr.step(_touch(1, 0), none, ball(0.0), g0)
    tr.step(_touch(1, 4), none, ball(50.0), g0)                           # toca el azul: cambia la posesión
    assert tr.poss_passes[0].tolist() == [0, 0] and tr.last[0] == 4
    tr.step(_touch(1, 5, 1), none, ball(200.0), g0)                       # dividida
    assert tr.last[0] == -1
    assert np.allclose(pass_bonus_reward(np.array([1]), np.array([3]), 0.0), 0.0)


def test_drift_reasons_follow_the_preregistered_criterion():
    from learn.x4_ppo import drift_reasons, parse_args
    a = parse_args(["--bc", "x", "--out", "y"])
    ok = dict(vs_bc=dict(score=0.5), human_w1_mean=0.5, selfplay=dict(dist_ball_1=120.0))
    assert drift_reasons(ok, a) == []
    bad = dict(vs_bc=dict(score=0.3), human_w1_mean=1.4, selfplay=dict(dist_ball_1=260.0))
    assert len(drift_reasons(bad, a)) == 3
    assert drift_reasons(dict(vs_bc=dict(score=0.5), human_w1_mean=None, selfplay={}), a) == []


def test_resume_continues_from_last_checkpoint(tmp_path):
    from pathlib import Path
    import torch
    from learn.x4_ppo import Trainer, parse_args
    bc = Path(__file__).resolve().parents[2] / "runs" / "x4_bc" / "final_sangu_rsone" / "best.pt"
    if not bc.exists():
        import pytest
        pytest.skip("falta el checkpoint de la imitación")
    common = ["--bc", str(bc), "--out", str(tmp_path), "--device", "cpu", "--envs", "4", "--rollout", "4",
              "--critic-warmup", "1", "--bank-recordings", "0", "--eval-every", "0", "--threads", "1",
              "--snapshot-every", "2", "--epochs", "1", "--minibatches", "1"]
    t1 = Trainer(parse_args(common + ["--updates", "2"]))
    t1.train()
    ck = torch.load(tmp_path / "last.pt", map_location="cpu")
    assert ck["update"] == 2 and ck["pool_names"] == ["snap_00002"] and "opt_pi" in ck
    t2 = Trainer(parse_args(common + ["--updates", "3", "--resume"]))
    assert t2.update == 2 and t2.decisions == ck["decisions"] and len(t2.pool) == 2
    for p_old, p_new in zip(t1.policy.parameters(), t2.policy.parameters()):
        assert torch.equal(p_old.detach().cpu(), p_new.detach().cpu())
    t2.train()
    assert torch.load(tmp_path / "last.pt", map_location="cpu")["update"] == 3
    import pytest
    with pytest.raises(SystemExit):          # sin --resume no se reusa la carpeta (pisaría best.pt)
        Trainer(parse_args(common + ["--updates", "3"]))


def _tiny_trainer(tmp_path, extra=()):
    from pathlib import Path
    from learn.x4_ppo import Trainer, parse_args
    bc = Path(__file__).resolve().parents[2] / "runs" / "x4_bc" / "final_sangu_rsone" / "best.pt"
    if not bc.exists():
        import pytest
        pytest.skip("falta el checkpoint de la imitación")
    common = ["--bc", str(bc), "--out", str(tmp_path), "--device", "cpu", "--envs", "6", "--rollout", "4",
              "--critic-warmup", "1", "--bank-recordings", "0", "--eval-every", "0", "--threads", "1",
              "--epochs", "1", "--minibatches", "1"]
    return Trainer(parse_args(common + list(extra)))


def test_pool_cap_reindexes_running_matches(tmp_path):
    t = _tiny_trainer(tmp_path, ["--updates", "1", "--snapshot-every", "0", "--pool-max", "2"])
    for k in range(3):
        t.update = k + 1
        t.snapshot()
    # pool: bc + 2 snapshots; el que más le gana el aprendiz se descarta
    assert len(t.pool) == 3 and len(t.pool_wins) == 3
    ar = t.arenas[0]
    t.pool_wins[:] = [0.5, 0.9, 0.2]
    ar.opp[:] = [-1, 0, 1, 2, 2, 1]
    ar.learner[1:] = np.array([True] * 4 + [False] * 4)
    t.drop_from_pool(1)
    assert [n for n, _ in t.pool] == ["bc", "snap_00003"]
    assert list(ar.opp) == [-1, 0, -1, 1, 1, -1]
    assert ar.learner[2].all() and ar.learner[5].all() and not ar.learner[3].all()
    assert ar.pool_size == 2 and np.allclose(t.pool_wins, [0.5, 0.2])


def test_resume_refuses_a_run_stopped_by_drift(tmp_path):
    import pytest
    (tmp_path / "stopped.json").write_text("{}")
    with pytest.raises(SystemExit):
        _tiny_trainer(tmp_path, ["--updates", "1", "--resume"])
    t = _tiny_trainer(tmp_path, ["--updates", "1", "--resume", "--continue-after-stop"])
    assert t.update == 0


def test_evaluation_is_reproducible_and_confirms_best(tmp_path):
    t = _tiny_trainer(tmp_path, ["--updates", "1", "--eval-matches", "2", "--eval-minutes", "0.1",
                                 "--eval-selfplay", "1"])
    t.update = 5
    a = t.evaluate(strength_only=True)
    b = t.evaluate(strength_only=True)
    assert a == b
    assert t.policy.training


def test_lateral_started_with_ball_inside_does_not_ping_pong():
    """Un lateral iniciado con la pelota ya adentro (estado grabado a mitad de la colocación) no puede perderse por
    "entrada sin patada" en cada decisión."""
    from learn.x4_ppo import restart_spot
    ar = _arena(n=1)
    env = ar.env
    env.place(0, ball_pos=np.array([577.1, 678.1]), ball_vel=np.zeros(2), player_pos=FAR, player_vel=np.zeros((8, 2)))
    env.start_restart(0, K.LATERAL, 1, (577.1, 678.1))       # spot adentro: antes rebotaba en cada paso
    forfeits = 0
    for _ in range(20):
        ev = env.step(np.zeros((1, 8), np.int64))
        forfeits += int(ev["forfeit"][0] >= 0)
    assert forfeits <= 1
    x, y = restart_spot(env, K.LATERAL, (577.1, 678.1))
    assert x == 577.1 and y > 678.325                         # el StateBank coloca la pelota afuera, como el script
