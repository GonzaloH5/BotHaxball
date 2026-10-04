"""RS-Pro: cinemática exacta, predicción de pelota, frenado y compensación de patada."""
import math

import numpy as np

from bots.rspro import brain as B
from bots.rspro.geom import HORIZON, TTR_TABLE, predict_ball, ttr
from env.rs4z.core import RS4ZEnv


def _simulate_player(vx0, target, max_ticks=3000):
    """Jugador acelerando a fondo hacia +x desde velocidad vx0: ticks hasta recorrer `target`."""
    x, v = 0.0, vx0
    for t in range(1, max_ticks):
        v += 0.12
        x += v
        v *= 0.96
        if x >= target:
            return t
    return max_ticks


def test_time_to_reach_matches_engine_kinematics():
    for d in (20.0, 100.0, 400.0, 1200.0):
        for u0 in (0.0, 1.5, -1.0):
            est = ttr(TTR_TABLE, 0.0, 0.0, u0, 0.0, d, 0.0, 0.0)
            assert abs(est - _simulate_player(u0, d)) <= 1.0, (d, u0, est)


def test_ball_prediction_matches_simulator_before_contact():
    env = RS4ZEnv(1, seed=0)
    far = np.tile([[0.0, 5000.0]], (8, 1))
    env.place(0, ball_pos=(-300.0, 50.0), ball_vel=(5.0, -1.2), player_pos=far, player_vel=np.zeros((8, 2)))
    env.active[:] = False
    traj = np.zeros((HORIZON + 1, 2))
    predict_ball(-300.0, 50.0, 5.0, -1.2, 0.0, 0.0, 0, 0.97, traj)
    env.frame_skip = 1
    worst = 0.0
    for t in range(1, 100):
        env.step(np.zeros((1, 8), dtype=np.int64))
        worst = max(worst, float(np.abs(env.ball_pos[0] - traj[t]).max()))
    assert worst < 1e-9


def test_move_action_arrives_and_brakes_without_overshoot():
    px, py, vx, vy = 0.0, 0.0, 0.0, 0.0
    target = (300.0, 120.0)
    from sim.physics import MOVE_UNIT
    closest = 1e9
    for step in range(400):
        m = B.move_action(px, py, vx, vy, target[0], target[1], 0.0, 1.0, False)
        for _ in range(3):
            vx += 0.12 * MOVE_UNIT[m, 0]
            vy += 0.12 * MOVE_UNIT[m, 1]
            px += vx
            py += vy
            vx *= 0.96
            vy *= 0.96
        closest = min(closest, math.hypot(px - target[0], py - target[1]))
    assert math.hypot(px - target[0], py - target[1]) < 12.0
    assert math.hypot(vx, vy) < 0.6


def test_kick_normal_compensates_ball_velocity():
    ux, uy = 1.0, 0.0
    bvx, bvy = 0.5, 1.6
    nx, ny = B.kick_normal(ux, uy, bvx, bvy)
    rx, ry = bvx + 6.1425 * nx, bvy + 6.1425 * ny
    assert abs(math.atan2(ry, rx)) < 1e-6


def test_orbit_approach_never_crosses_the_ball():
    # desde delante de la pelota, el punto intermedio queda a distancia segura
    for ang in np.linspace(0, 2 * np.pi, 24, endpoint=False):
        px, py = 60 * math.cos(ang), 60 * math.sin(ang)
        wx, wy, _ = B.approach_point(px, py, 0.0, 0.0, 1.0, 0.0)
        assert math.hypot(wx, wy) >= 15.0 + 8.325 - 1.0 - 1e-6


def _defend_scene(defenders, attacker=(-200.0, 0.0), ball=(-170.0, 0.0), decisions=150, level=5):
    """Rojo (aprendiz) quieto con la pelota; azul RS-Pro defiende su arco (x = +1162). Devuelve (env, bot)."""
    from bots.rspro.policy import RSPro, STYLE_BALANCED
    env = RS4ZEnv(1, seed=0, deadline=0)
    active = np.zeros(8, dtype=bool)
    active[0] = True
    pos = np.tile([[0.0, 0.0]], (8, 1))
    pos[0] = attacker
    for i, d in enumerate(defenders):
        active[4 + i] = True
        pos[4 + i] = d
    env.start_match([0], active=active[None], kickoff_team=0, match_ticks=10 ** 9)
    env.place(0, ball_pos=ball, ball_vel=(0.0, 0.0), player_pos=pos, player_vel=np.zeros((8, 2)), last_touch=0)
    bot = RSPro(env, seed=1)
    bot.configure([0], 1, level, STYLE_BALANCED)
    bot.sync(env, [0])
    out = np.zeros((1, 8), dtype=np.int64)
    ctrl = np.zeros((1, 8), dtype=bool)
    ctrl[0, 4:4 + len(defenders)] = True
    for _ in range(decisions):
        out[:] = 0
        bot.act(env, ctrl, out)
        env.step(out)
        bot.push(env)
    return env, bot


def test_last_man_contains_instead_of_rushing_the_carrier():
    """Último hombre lejos de su arco: se para entre la pelota y el arco a distancia, sin ir al contacto
    (regresión del 2026-10-04: salía a presionar y un regate en diagonal lo dejaba pasado)."""
    env, _ = _defend_scene([(700.0, 0.0)])
    d = env.player_pos[0, 4] - env.ball_pos[0]
    assert d[0] > 0, "debe quedar del lado del arco"
    assert 55.0 < np.hypot(*d) < 130.0, np.hypot(*d)


def test_presser_goes_to_contact_when_there_is_cover():
    env, _ = _defend_scene([(500.0, 0.0), (900.0, 0.0)])
    near = min(np.hypot(*(env.player_pos[0, 4 + i] - env.ball_pos[0])) for i in range(2))
    assert near < 45.0, near


def test_beaten_presser_is_relieved_by_the_goal_side_defender():
    """Presionante pasado (detrás de la pelota, más cerca de ella) y compañero del lado del arco: el que
    presiona es el compañero (relevo) y el pasado toma otro rol."""
    _, bot = _defend_scene([(240.0, 40.0), (800.0, 0.0)], attacker=(270.0, 0.0), ball=(300.0, 0.0), decisions=3)
    roles = bot.mem_i[0, 1, B.M_ROLE:B.M_ROLE + 8]
    assert roles[5] == B.R_PRESS, roles
    assert roles[4] != B.R_PRESS, roles

