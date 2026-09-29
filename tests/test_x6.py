import numpy as np
import pytest

from sim.physics import BatchSim
from sim.stadium import load_stadium

STAY, KICK = 0, 9


def _hold(sim, ticks, action=STAY, lateral=0.0):
    """Mantiene al jugador 0 pegado a la pelota (tocándola) sin moverla."""
    fp = sim.first_player
    for _ in range(ticks):
        sim.pos[0, 0] = (0.0, 0.0)
        sim.vel[0, 0] = 0.0
        sim.pos[0, fp] = (-(14 + 7), 0.0)  # contacto exacto: dist = r_p + r_b
        sim.vel[0, fp] = (0.0, lateral)
        sim.pos[0, fp + 1] = (200.0, 200.0)
        sim.vel[0, fp + 1] = 0.0
        sim.step(np.array([[action, STAY]]))


def _sim():
    sim = BatchSim(1, 1, 1, "x6_half", powershot=True)
    sim.reset_random([0])
    return sim


def test_half_scale_keeps_physics_and_halves_geometry():
    full, half = load_stadium("x6"), load_stadium("x6_half")
    assert half.player == full.player and half.ball == full.ball
    assert half.goal_x == pytest.approx(full.goal_x / 2)
    assert half.goal_half_height == pytest.approx(full.goal_half_height / 2)
    assert half.spawn_distance == pytest.approx(full.spawn_distance / 2)


def test_kick_impulse_scales_with_ball_invmass():
    sim = BatchSim(1, 1, 1, "x6_half")
    sim.reset_random([0])
    fp = sim.first_player
    sim.pos[0, 0] = (0, 0); sim.vel[0, 0] = 0
    sim.pos[0, fp:] = [(-22, 0), (200, 200)]; sim.vel[0, fp:] = 0
    sim.step(np.array([[KICK, STAY]]))
    assert sim.kicked[0, 0]
    assert sim.vel[0, 0, 0] == pytest.approx(3.8 * 1.5 * 0.99)


def test_powershot_charges_after_hold_and_kick_is_stronger_with_curve():
    sim = _sim()
    _hold(sim, 30 + 96 + 2)  # primer chequeo a los 30 ticks + 96 de carga
    assert sim.ps_comba[0] and sim.inv_env[0, 0] == pytest.approx(2.3)
    # patear cargado moviéndose lateralmente (hacia abajo, +y) -> curva
    fp = sim.first_player
    sim.pos[0, 0] = (0, 0); sim.vel[0, 0] = 0
    sim.pos[0, fp] = (-22, 0); sim.vel[0, fp] = (0, 3.0)
    sim.step(np.array([[KICK, STAY]]))
    assert sim.ps_kicked[0, 0] and not sim.ps_comba[0]
    assert sim.vel[0, 0, 0] == pytest.approx(3.8 * 2.3 * 0.99, rel=1e-6)
    # la gravedad arranca DESPUÉS del tick del tiro, perpendicular al tiro (+x): (0, -ci*0.1)
    lat = 1.0 * 3.0  # nx=1: lateral = nx*vy - ny*vx
    ci = min(1.0, lat / 3.0)
    np.testing.assert_allclose(sim.ball_grav[0], (0.0, -ci * 0.1), atol=1e-9)
    for _ in range(48):
        sim.step(np.array([[STAY, STAY]]))
    assert sim.inv_env[0, 0] == pytest.approx(1.5)          # invMass vuelve a los 48 ticks
    assert np.abs(sim.ball_grav[0]).sum() > 0               # la curva sigue
    for _ in range(84 - 48):
        sim.step(np.array([[STAY, STAY]]))
    np.testing.assert_allclose(sim.ball_grav[0], 0.0)       # y termina a los 84


def test_losing_the_ball_cancels_charge():
    sim = _sim()
    _hold(sim, 30 + 40)            # empezó a cargar
    assert sim.ps_held[0] == 0 and not sim.ps_comba[0]
    fp = sim.first_player
    sim.pos[0, fp] = (-200, -200)  # se aleja
    for _ in range(40):
        sim.step(np.array([[STAY, STAY]]))
    assert sim.ps_held[0] == -1 and not sim.ps_comba[0]
    assert sim.inv_env[0, 0] == pytest.approx(1.5)


def test_charged_then_lost_keeps_heavy_kick_like_the_room_script():
    """Réplica fiel del script: si se carga y se pierde sin patear, invMass queda en 2.3."""
    sim = _sim()
    _hold(sim, 30 + 96 + 2)
    assert sim.ps_comba[0]
    fp = sim.first_player
    sim.pos[0, fp] = (-200, -200)
    for _ in range(35):
        sim.step(np.array([[STAY, STAY]]))
    assert not sim.ps_comba[0]
    assert sim.inv_env[0, 0] == pytest.approx(2.3)


def test_classic_unaffected_by_invmass_change():
    sim = BatchSim(1, 1, 1, "classic")
    assert sim.inv_env[0, 0] == 1.0 and not sim.ps_on


def _env(**kw):
    from env.haxball_env import HaxballEnv
    return HaxballEnv(kw.pop("n", 4), 3, "x6_half", powershot=True, out_of_bounds=True, seed=0, **kw)


def test_env_3v3_obs_dim_and_mirror_symmetry():
    env = _env(n=1, random_reset_prob=1.0)
    env.reset()
    assert env.obs_dim == 22 + 4 * 2 + 6 * 3 + 6
    s = env.sim
    fp = s.first_player
    # estado espejado en x: rojo p <-> azul p
    s.pos[0, 0] = (0, -40); s.vel[0, 0] = (0, 1.5)
    for i, (x, y) in enumerate([(-100, 50), (-200, -80), (-50, 120)]):
        s.pos[0, fp + i] = (x, y); s.pos[0, fp + 3 + i] = (-x, y)
        s.vel[0, fp + i] = (0.5, -0.3); s.vel[0, fp + 3 + i] = (-0.5, -0.3)
    rng = np.random.default_rng(0)
    for _ in range(60):
        a = rng.integers(0, 18, 3)
        _, _, d, _ = env.step(np.concatenate([a, a])[None])
        if d[0]:
            break
        o = env.observe()
        np.testing.assert_allclose(o[0, :3], o[0, 3:], atol=1e-5)


def test_ball_out_gives_throw_in_to_other_team_and_penalizes_last_toucher():
    env = _env(n=1, random_reset_prob=1.0)
    env.reset()
    s = env.sim
    fp = s.first_player
    s.pos[0, fp:] = [(-100, -100), (-100, 0), (-100, 100), (100, -100), (100, 0), (100, 100)]
    s.vel[0, fp:] = 0
    s.pos[0, 0] = (0, 285); s.vel[0, 0] = (0, 6)  # sale por la banda (y > 300)
    env.last_touch[0] = 1                           # la tocó último el azul
    for _ in range(10):
        _, r, d, info = env.step(np.zeros((1, 6), dtype=np.int64))
        if info["out"][0]:
            break
    assert info["out"][0] and not d[0]              # no termina: lateral
    assert (r[0, 3:] < r[0, :3] - 0.05).all()       # penalidad al azul
    b = env.sim.ball_pos[0]
    assert abs(b[1]) < env.field_h and np.allclose(env.sim.ball_vel[0], 0)
    blue = env.sim.player_pos[0, 3:]
    assert (np.linalg.norm(blue - b, axis=1) >= env.setpiece_dist - 1e-6).all()


def test_ball_stuck_in_corner_is_called_out():
    env = _env(n=1, random_reset_prob=1.0)
    env.reset()
    s = env.sim
    fp = s.first_player
    W, H = env.field_w, env.field_h
    s.pos[0, fp:] = [(0, 0), (-100, 0), (-200, 0), (100, 0), (200, 0), (300, 0)]
    s.vel[0, fp:] = 0
    outs = 0
    for _ in range(80):
        s.pos[0, 0] = (W - 5, -(H - 5)); s.vel[0, 0] = 0   # la mantenemos trabada en la esquina
        _, _, _, info = env.step(np.zeros((1, 6), dtype=np.int64))
        outs += int(info["out"][0])
        if outs:
            break
    assert outs == 1


def test_scripted_roles_spread_players():
    from bots.scripted import scripted_actions
    env = _env(n=32, random_reset_prob=0.0)
    env.reset()
    for _ in range(200):
        env.step(scripted_actions(env))
    pp = env.sim.player_pos[:, :3]
    d = np.linalg.norm(pp[:, :, None] - pp[:, None], axis=-1)
    d[:, np.arange(3), np.arange(3)] = np.inf
    assert np.median(d.min(axis=2)) > 40  # no amontonados
