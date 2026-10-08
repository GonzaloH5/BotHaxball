"""Reglas del árbitro HX11-1 en escenarios armados a mano (env/hx11)."""
import numpy as np
import pytest

from env.hx11 import contract as C
from env.hx11 import kernel as K
from env.hx11.core import HX11Env
from sim.physics import BatchSim
from sim.stadium import BLUE, C1, PLAYER_MASK, RED

ZERO = np.zeros((1, 22), np.int64)


@pytest.fixture(scope="module")
def env():
    return HX11Env(1, seed=0)


def open_play(env, ball=(0.0, 0.0), last=-1):
    """Juego abierto con todos los jugadores lejos de la pelota (rojos arriba, azules abajo)."""
    env.reset()
    fp = env.fp
    for p in range(22):
        y = -1100.0 if p < 11 else 1100.0
        env.pos[0, fp + p] = (-1000.0 + 90.0 * (p % 11), y)
    env.pos[0, 0] = ball
    env.vel[0] = 0.0
    env.mask[0, fp:] = PLAYER_MASK
    env.ri[0, K.RI_MODE] = K.OPEN
    env.ri[0, K.RI_KO] = 0
    env.ri[0, K.RI_LAST] = last


def run(env, n, actions=ZERO):
    ev = []
    for _ in range(n):
        ev.append(int(env.step(actions, 1)[0, 0]))
    return ev


def test_physics_matches_base_simulator():
    """Sin el árbitro (modos apagados y la pelota lejos de las líneas), la física es la de sim.physics."""
    env = HX11Env(1, seed=0, offside=False, fouls=False, powershot=False)
    sim = BatchSim(1, 11, 11, C.STADIUM)
    sim.pos[:] = env.pos
    sim.vel[:] = env.vel
    sim.mask[:] = env.mask
    sim.radius[1:3] = 0.0
    sim.kickoff[:] = True
    sim.kickoff_team[:] = 0
    rng = np.random.default_rng(3)
    for _ in range(240):
        a = rng.integers(0, 18, size=(1, 22))
        env.step(a, 1)
        sim.step(a)
    assert np.abs(env.pos - sim.pos).max() < 1e-9
    assert np.abs(env.vel - sim.vel).max() < 1e-9


def test_lateral(env):
    open_play(env, ball=(100.0, 1190.0), last=3)  # tocó un rojo
    env.vel[0, 0] = (0.5, 3.0)
    ev = run(env, 40)
    assert K.EV_OUT in ev and K.EV_PLACE in ev
    assert ev.index(K.EV_PLACE) - ev.index(K.EV_OUT) == C.LATERAL_DELAY
    assert env.ri[0, K.RI_KIND] == C.LATERAL and env.ri[0, K.RI_TEAM] == 1  # saca el azul
    assert env.pos[0, 0, 1] == C.LATERAL_Y and abs(env.pos[0, 0, 0] - env.rf[0, K.RF_CROSS_X]) < 1e-9
    assert env.inv[0, 0] == C.LATERAL_INV
    assert env.radius[0, 1] == 373.0 and env.mask[0, 1] == RED  # barrera contra el rojo


def test_corner_and_goal_kick(env):
    open_play(env, ball=(1795.0, 600.0), last=14)  # tocó un azul, que defiende x > 0: córner para el rojo
    env.vel[0, 0] = (3.0, 0.0)
    ev = run(env, 30)
    assert ev.index(K.EV_PLACE) - ev.index(K.EV_OUT) == C.END_LINE_DELAY
    assert env.ri[0, K.RI_KIND] == C.CORNER and env.ri[0, K.RI_TEAM] == 0
    assert tuple(env.pos[0, 0]) == C.CORNER_SPOT
    assert env.radius[0, 1] == 750.0 and env.mask[0, 1] == BLUE
    assert (env.inv[0, env.fp:] == C.GHOST_INV).all()
    open_play(env, ball=(1795.0, -600.0), last=4)  # tocó un rojo: saque de arco para el azul
    env.vel[0, 0] = (3.0, 0.0)
    run(env, 30)
    assert env.ri[0, K.RI_KIND] == C.GOAL_KICK and env.ri[0, K.RI_TEAM] == 1
    assert tuple(env.pos[0, 0]) == C.GOAL_KICK_SPOT
    assert (env.group[0, env.fp:env.fp + 11] == C1).all()  # rojos fantasma, fuera de la zona c1
    assert env.pos[0, 1, 0] == C.PENALTY_X and env.radius[0, 1] == 373.0  # barrera en el punto de penal


@pytest.mark.parametrize("where,kind", [((0.0, 0.0), C.FREE_KICK), ((1600.0, 0.0), C.PENALTY)])
def test_foul(env, where, kind):
    open_play(env, ball=(where[0] + 23.0, where[1]), last=5)
    fp = env.fp
    env.pos[0, fp + 5] = where                       # rojo con la pelota pegada
    env.pos[0, fp + 15] = (where[0], where[1] + 30.0)  # azul pegado al poseedor
    ev = run(env, C.FOUL_TICKS + C.FREEZE_TICKS + 5)
    assert ev.index(K.EV_FOUL) == C.FOUL_TICKS
    assert ev.index(K.EV_PLACE) - ev.index(K.EV_FOUL) == C.FREEZE_TICKS
    assert env.ri[0, K.RI_KIND] == kind and env.ri[0, K.RI_TEAM] == 0
    if kind == C.PENALTY:
        assert tuple(env.pos[0, 0]) == (C.PENALTY_X, 0.0) and env.inv[0, 0] == C.PENALTY_INV


def test_offside_called_when_marked_player_touches(env):
    open_play(env, ball=(123.0, 0.0), last=1)
    fp = env.fp
    env.pos[0, fp + 1] = (100.0, 0.0)          # rojo que pasa
    env.pos[0, fp + 9] = (1000.0, 300.0)       # rojo adelantado
    for i, p in enumerate(range(12, 22)):      # azules hasta x = 900; arquero en 1760
        env.pos[0, fp + p] = (500.0 + 40.0 * i, -600.0)
    env.pos[0, fp + 11] = (1760.0, 0.0)
    a = ZERO.copy()
    a[0, 1] = 9                                # patear
    run(env, 1, a)
    assert (env.ri[0, K.RI_OFF_MASK] >> 9) & 1
    env.pos[0, 0] = (1023.0, 300.0)            # la pelota llega al adelantado
    env.vel[0, 0] = 0.0
    ev = run(env, 1)
    assert ev == [K.EV_OFFSIDE] and env.ri[0, K.RI_TEAM] == 1
    ev = run(env, C.FREEZE_TICKS)
    assert env.ri[0, K.RI_MODE] == K.PIECE and env.ri[0, K.RI_KIND] == C.FREE_KICK


def test_powershot_ramp_and_kick(env):
    open_play(env, ball=(23.0, 0.0), last=5)
    env.pos[0, env.fp + 5] = (0.0, 0.0)
    run(env, C.PS_DELAY + 24)
    assert abs(env.inv[0, 0] - C.PS_LEVELS[0]) < 1e-6
    run(env, 209 - 24)
    assert abs(env.inv[0, 0] - C.PS_LEVELS[1]) < 1e-6
    a = ZERO.copy()
    a[0, 5] = 9
    env.vel[0, env.fp + 5] = (0.0, 2.0)        # se mueve en y al patear: comba
    run(env, 1, a)
    v = np.linalg.norm(env.vel[0, 0])
    assert v > 5.0 * C.PS_LEVELS[1] * 0.98     # patada con la masa cargada
    run(env, 2)
    assert env.inv[0, 0] == C.BASE_BALL_INV and env.grav[0, 1] < 0


def test_set_piece_kick_restores(env):
    test_corner_and_goal_kick(env)             # termina en un saque de arco del azul
    run(env, 125)                              # disco del punto (120) y barrera del arco (121) se van
    assert env.radius[0, 2] == 0.0 and env.radius[0, 1] == 0.0
    fp = env.fp
    env.pos[0, fp + 11] = (C.GOAL_KICK_SPOT[0] + 23.0, 0.0)   # el arquero azul detrás de la pelota
    a = ZERO.copy()
    a[0, 11] = 9
    ev = run(env, 1, a)
    assert ev == [K.EV_PIECE_KICK] and env.ri[0, K.RI_MODE] == K.OPEN
    run(env, 1)
    assert env.inv[0, 0] == C.BASE_BALL_INV
    assert (env.inv[0, fp:] == C.PLAYER_INV).all() and env.group[0, fp] == RED
