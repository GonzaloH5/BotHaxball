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
    env.delay[:] = 0
    env.act_hist[0, :, 0] = np.arange(8) + 2          # acción del mundo de cada lugar
    up = upcoming_actions(env).reshape(1, 8, 8, 18)
    for p in range(8):
        for k, q in enumerate(ORDER[p]):
            a = env.act_hist[0, q, 0]
            if TEAM[p] == 1:
                a = MIRROR_ACTION[a]
            assert up[0, p, k, a] == 1.0 and up[0, p, k].sum() == 1.0
    assert ORDER[5][0] == 5 and set(ORDER[5][1:4]) == {4, 6, 7}
