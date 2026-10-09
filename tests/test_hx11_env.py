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
    assert 100.0 < env.pos[0, 0, 0] < 106.0     # x de la pelota al pasar |y| > 1210 (avanza 0,5 por tick)
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
    if kind == C.FREE_KICK:
        assert tuple(env.pos[0, 0]) == where       # en la posición del que recibió la falta
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
    assert tuple(env.pos[0, 0]) == (1015.0, 300.0)  # el adelantado + 15 px hacia el arco rival
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


def test_goal_kick_clears_rivals_from_box(env):
    """La sala saca a los rivales del área grande a |x| = 1035 con su misma y, un tick después de la colocación."""
    open_play(env, ball=(1795.0, -600.0), last=4)  # tocó un rojo: saque de arco para el azul (arco en x > 0)
    fp = env.fp
    env.pos[0, fp + 9] = (1400.0, 200.0)           # rojo dentro del área
    env.pos[0, fp + 10] = (900.0, 100.0)           # rojo fuera del área
    env.vel[0, 0] = (3.0, 0.0)
    ev = run(env, 30)
    assert env.ri[0, K.RI_KIND] == C.GOAL_KICK and K.EV_PLACE in ev
    assert env.pos[0, fp + 9, 0] == pytest.approx(C.GK_CLEAR_X, abs=3.0)
    assert env.pos[0, fp + 9, 1] == pytest.approx(200.0, abs=3.0)
    assert env.pos[0, fp + 10, 0] < 1000.0


def test_lateral_carried_without_kick_stays(env):
    """Llevar la pelota con el cuerpo en un lateral no cambia el saque (los humanos lo hacen para buscar el ángulo; la
    regla estricta del 2026-10-09 trababa a los bots en un ida y vuelta de laterales)."""
    open_play(env, ball=(100.0, 1190.0), last=3)   # tocó un rojo: lateral para el azul
    env.vel[0, 0] = (0.5, 3.0)
    run(env, 40)
    assert env.ri[0, K.RI_KIND] == C.LATERAL and env.ri[0, K.RI_TEAM] == 1
    env.pos[0, 0, 0] += 40.0                       # la pelota se movió sin patada
    ev = run(env, 5)
    assert K.EV_PLACE not in ev and K.EV_OUT not in ev
    assert env.ri[0, K.RI_MODE] == K.PIECE and env.ri[0, K.RI_TEAM] == 1


def _lateral_kicked(env, thrower=14, vel=(9.0, 0.0)):
    """Lateral del azul en (100, 1230) recién pateado por `thrower`, con la pelota a `vel` (paralela: no entra)."""
    open_play(env, ball=(100.0, 1190.0), last=3)
    env.vel[0, 0] = (0.5, 3.0)
    run(env, 40)
    assert env.ri[0, K.RI_KIND] == C.LATERAL and env.ri[0, K.RI_TEAM] == 1
    spot = env.pos[0, 0].copy()
    env.ri[0, K.RI_MODE] = K.OPEN
    env.ri[0, K.RI_TICKS] = 0
    env.ri[0, K.RI_AFTER_KIND] = C.LATERAL
    env.ri[0, K.RI_AFTER_T] = 0
    env.ri[0, K.RI_LAST] = thrower
    env.ri[0, K.RI_LAT_THROWER] = thrower
    env.ri[0, K.RI_LAT_TOUCHED] = 0
    env.ri[0, K.RI_LAT_ARMED] = 0
    env.vel[0, 0] = vel
    return spot


def test_lateral_that_does_not_enter_goes_to_rival(env):
    spot = _lateral_kicked(env)
    ev = run(env, C.BAD_LATERAL_TICKS + C.LATERAL_DELAY + 3)
    assert K.EV_OUT in ev and K.EV_PLACE in ev
    assert ev.index(K.EV_PLACE) - ev.index(K.EV_OUT) == C.LATERAL_DELAY
    assert env.ri[0, K.RI_KIND] == C.LATERAL and env.ri[0, K.RI_TEAM] == 0     # ahora saca el rojo
    assert np.allclose(env.pos[0, 0], spot)                                      # en el punto original


def test_lateral_touched_by_rival_outside_goes_back(env):
    """Si un rival la toca afuera y no entra, vuelve a sacar el mismo equipo (contra el último que la tocó)."""
    spot = _lateral_kicked(env)
    env.ri[0, K.RI_LAST] = 3                        # la tocó un rojo antes de entrar
    ev = run(env, C.BAD_LATERAL_TICKS + C.LATERAL_DELAY + 3)
    assert K.EV_PLACE in ev
    assert env.ri[0, K.RI_KIND] == C.LATERAL and env.ri[0, K.RI_TEAM] == 1     # vuelve a sacar el azul
    assert np.allclose(env.pos[0, 0], spot)


def test_lateral_that_enters_is_play(env):
    _lateral_kicked(env, vel=(6.0, -4.0))           # con ángulo: entra
    ev = run(env, C.BAD_LATERAL_TICKS + C.LATERAL_DELAY + 3)
    assert env.ri[0, K.RI_LAT_ARMED] == 1 and K.EV_PLACE not in ev


def test_lateral_kicked_from_inside_goes_to_rival(env):
    """Meter la pelota empujando y patearla desde adentro es mal sacado: lateral para el rival en el punto original."""
    open_play(env, ball=(100.0, 1190.0), last=3)   # tocó un rojo: lateral para el azul en y = 1230
    env.vel[0, 0] = (0.5, 3.0)
    run(env, 40)
    assert env.ri[0, K.RI_KIND] == C.LATERAL and env.ri[0, K.RI_TEAM] == 1
    spot = env.pos[0, 0].copy()
    fp = env.fp
    env.pos[0, 0] = (spot[0], 1190.0)               # la metió empujando
    env.vel[0, 0] = 0.0
    env.pos[0, fp + 14] = (spot[0], 1190.0 - C.PLAYER_R - C.BALL_R - 1.0)   # azul pegado, del lado de la cancha
    env.vel[0, fp + 14] = 0.0
    a = np.zeros((1, 22), np.int64)
    a[0, 14] = 9                                    # patear sin moverse
    ev = run(env, C.LATERAL_DELAY + 4, a)
    assert K.EV_OUT in ev and K.EV_PLACE in ev
    assert env.ri[0, K.RI_KIND] == C.LATERAL and env.ri[0, K.RI_TEAM] == 0     # ahora saca el rojo
    assert np.allclose(env.pos[0, 0], spot)


def test_lateral_pushed_in_and_kept_goes_to_rival(env):
    """Meterla empujando y quedarse adentro sin patear es mal sacado pasados LATERAL_INSIDE_TICKS; menos tiempo, no."""
    open_play(env, ball=(100.0, 1190.0), last=3)
    env.vel[0, 0] = (0.5, 3.0)
    run(env, 40)
    spot = env.pos[0, 0].copy()
    env.pos[0, 0] = (spot[0], 1150.0)               # la metió y quedó adentro, quieta
    env.vel[0, 0] = 0.0
    ev = run(env, C.LATERAL_INSIDE_TICKS - 10)
    assert K.EV_OUT not in ev and env.ri[0, K.RI_MODE] == K.PIECE
    ev = run(env, 20 + C.LATERAL_DELAY)
    assert K.EV_OUT in ev and K.EV_PLACE in ev
    assert env.ri[0, K.RI_KIND] == C.LATERAL and env.ri[0, K.RI_TEAM] == 0
    assert np.allclose(env.pos[0, 0], spot)


def test_lateral_touched_by_teammate_outside_is_taken(env):
    """Si un compañero la toca antes de entrar, el lateral cuenta como sacado: no hay cobro a los 45 ticks."""
    _lateral_kicked(env, vel=(1.0, 0.0))
    env.ri[0, K.RI_LAST] = 15                       # un compañero del azul que sacó (14)
    ev = run(env, C.BAD_LATERAL_TICKS + C.LATERAL_DELAY + 10)
    assert K.EV_OUT not in ev and env.ri[0, K.RI_LAT_TOUCHED] == 1


def test_lateral_retouched_by_thrower_is_taken(env):
    """Si el que sacó la vuelve a tocar antes de que entre, el lateral cuenta como sacado (plazo largo)."""
    _lateral_kicked(env, vel=(1.0, 0.0))
    fp = env.fp
    run(env, 5)
    b = env.pos[0, 0].copy()
    env.pos[0, fp + 14] = (b[0] + C.PLAYER_R + C.BALL_R + 0.5, b[1])   # el mismo azul, delante de la pelota
    env.vel[0, fp + 14] = 0.0
    ev = run(env, C.BAD_LATERAL_TICKS + C.LATERAL_DELAY + 10)
    assert env.ri[0, K.RI_LAT_TOUCHED] == 1 and K.EV_OUT not in ev
