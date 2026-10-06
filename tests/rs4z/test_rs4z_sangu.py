"""Script de la sala SANGUCHITO RS X4 (medido en sus 7 grabaciones; ver `env/rs4z/contract.py`)."""
import numpy as np

from env.rs4z import contract as C
from env.rs4z import kernel as K
from env.rs4z.core import RS4ZEnv

MAP = "sanguchito_rs_x4"
PRM = C.MAPS[MAP]["overrides"]
CORNER_SPOT = (PRM["corner_x"], PRM["corner_y"])     # arco azul (+x), abajo (+y)
FAR = [(-400, 300), (-300, -300), (-200, 200), (-100, -100), (400, 300), (300, -300), (200, 200), (100, -100)]


def _env():
    return RS4ZEnv(1, contract="v2", map=MAP, seed=0, frame_skip=1, deadline=0, kickoff_deadline=0, max_delay=0)


def _place(env, players, ball=(0.0, 0.0)):
    env.place(0, ball_pos=ball, ball_vel=(0.0, 0.0), player_pos=np.asarray(players, dtype=float),
              player_vel=np.zeros((8, 2)), last_touch=-1)


def _act(**by_slot):
    a = np.zeros((1, 8), dtype=np.int64)
    for slot, v in by_slot.items():
        a[0, int(slot[1:])] = v
    return a


def test_map_adds_sanguchito_script_to_contract():
    env = _env()
    assert env.flags & C.FIX_SANGU and env.flags & C.FIX_LATERAL
    assert np.allclose(env.inv[0, env.fp:], 0.5) and np.isclose(env.inv[0, 0], 1.05)


def test_kickoff_formation_is_the_diamond_for_both_teams():
    env = _env()
    form = np.array([(PRM[f"form{i}_x"], PRM[f"form{i}_y"]) for i in range(4)])
    for kt in (0, 1):
        env.start_match([0], kickoff_team=kt)
        red = env.player_pos[0, :4]
        blue = env.player_pos[0, 4:]
        assert np.allclose(np.sort(red, axis=0), np.sort(form, axis=0))
        assert np.allclose(np.sort(blue, axis=0), np.sort(form * [-1.0, 1.0], axis=0))


def test_corner_fixes_ball_and_disc_expels_defenders_only():
    env = _env()
    players = list(FAR)
    players[1] = (1000.0, 500.0)      # rojo (atacante) dentro del radio: no lo afecta
    players[4] = (1010.0, 520.0)      # azul (defensor) dentro del radio: expulsado
    _place(env, players)
    env.start_restart(0, C.CORNER, 0, CORNER_SPOT)
    assert env.inv[0, 0] == 0.0
    assert np.isclose(env.radius[0, C.SD_BLUE], PRM["corner_disc_radius"])
    env.step(_act(p0=0))
    spot = np.array(CORNER_SPOT)
    assert np.hypot(*(env.player_pos[0, 4] - spot)) >= PRM["corner_disc_radius"] + 15.0 - 1e-6
    assert np.hypot(*(env.player_pos[0, 1] - spot)) < 300.0
    assert np.allclose(env.ball_pos[0], spot) and env.ri[0, K.RI_TEAM] == 0


def test_corner_kick_sets_script_velocity_and_curve():
    env = _env()
    players = list(FAR)
    players[0] = (CORNER_SPOT[0] + 24.0, CORNER_SPOT[1])   # detrás de la pelota: patea hacia la cancha
    _place(env, players)
    env.start_restart(0, C.CORNER, 0, CORNER_SPOT)
    env.step(_act(p0=9))                                   # patear sin moverse
    assert env.ri[0, K.RI_TEAM] == -1, "la patada hacia adentro libera el córner"
    assert np.allclose(env.ball_vel[0], (-PRM["corner_kick_speed"], 0.0), atol=1e-9)
    assert np.isclose(env.inv[0, 0], 1.05)
    g0 = np.array([0.0, -PRM["corner_grav_y"]])            # abajo (+y): la curva empuja hacia la cancha
    assert np.allclose(env.grav[0], g0)
    for _ in range(4):
        env.step(_act())
        assert np.allclose(env.grav[0], g0), "se mantiene 5 ticks"
    env.step(_act())
    assert np.allclose(env.grav[0], g0 * 0.97)


def test_outward_kick_does_not_release_the_corner():
    env = _env()
    players = list(FAR)
    players[0] = (CORNER_SPOT[0] - 24.0, CORNER_SPOT[1])   # del lado de la cancha: patearía hacia afuera
    _place(env, players)
    env.start_restart(0, C.CORNER, 0, CORNER_SPOT)
    env.step(_act(p0=9))
    assert env.ri[0, K.RI_TEAM] == 0 and env.inv[0, 0] == 0.0
    assert np.allclose(env.ball_pos[0], CORNER_SPOT) and np.allclose(env.ball_vel[0], 0.0)


def test_corner_kick_outward_in_y_does_not_release():
    """En el córner la sala ignora la patada que manda la pelota hacia afuera en y aunque vaya hacia adentro en x."""
    env = _env()
    players = list(FAR)
    sy = np.sign(CORNER_SPOT[1])
    players[0] = (CORNER_SPOT[0] + 16.0, CORNER_SPOT[1] - sy * 16.0)   # detrás en x pero del lado de la cancha en y
    _place(env, players)
    env.start_restart(0, C.CORNER, 0, CORNER_SPOT)
    env.step(_act(p0=9))
    assert env.ri[0, K.RI_TEAM] == 0, "la patada hacia afuera en y no libera el córner"
    assert np.allclose(env.ball_pos[0], CORNER_SPOT) and np.allclose(env.ball_vel[0], 0.0)
    env2 = _env()
    players[0] = (CORNER_SPOT[0] + 16.0, CORNER_SPOT[1] + sy * 16.0)   # detrás en x y en y: libera
    _place(env2, players)
    env2.start_restart(0, C.CORNER, 0, CORNER_SPOT)
    env2.step(_act(p0=9))
    assert env2.ri[0, K.RI_TEAM] == -1


def test_corner_direction_is_judged_before_the_tick():
    """La sala juzga la dirección con la posición del pateador al patear: si recién cruza la vertical de la
    pelota durante el tick, la patada sigue siendo hacia afuera y no libera."""
    env = _env()
    players = list(FAR)
    sy = np.sign(CORNER_SPOT[1])
    players[0] = (CORNER_SPOT[0] - 0.7, CORNER_SPOT[1] + sy * 23.5)   # apenas del lado de la cancha en x
    env.place(0, ball_pos=(0.0, 0.0), ball_vel=(0.0, 0.0), player_pos=np.asarray(players, dtype=float),
              player_vel=np.array([(3.0, 0.0)] + [(0.0, 0.0)] * 7))     # se mueve hacia afuera durante el tick
    env.start_restart(0, C.CORNER, 0, CORNER_SPOT)
    env.vel[0, env.fp] = (3.0, 0.0)
    env.step(_act(p0=9))
    assert env.player_pos[0, 0, 0] > CORNER_SPOT[0], "después del tick el pateador quedó detrás de la pelota"
    assert env.ri[0, K.RI_TEAM] == 0, "la patada se juzga antes del tick: hacia afuera, no libera"


def test_goal_kick_velocity_uses_goal_kick_speed():
    env = _env()
    spot = (PRM["goal_kick_x"], PRM["goal_kick_y"])
    players = list(FAR)
    players[4] = (spot[0] + 24.0, spot[1])                 # azul defiende +x y saca
    _place(env, players)
    env.start_restart(0, C.GOAL_KICK, 1, spot)
    env.step(_act(p4=9))
    assert env.ri[0, K.RI_TEAM] == -1
    assert np.allclose(env.ball_vel[0], (-PRM["goal_kick_speed"], 0.0), atol=1e-9)
    assert np.allclose(env.grav[0], 0.0)                   # sin velocidad vertical del pateador


def test_lateral_pushes_only_rivals_beyond_the_barrier():
    env = _env()
    players = list(FAR)
    players[4] = (900.0, 560.0)                            # cruzó la barrera (557,76), lejos en x
    players[5] = (100.0, 550.0)                            # no la cruzó
    _place(env, players)
    env.start_restart(0, C.LATERAL, 0, (0.0, PRM["lateral_ball_y"]))
    assert np.allclose(env.player_pos[0, 4], (900.0, PRM["push_y"]))
    assert np.allclose(env.player_pos[0, 5], (100.0, 550.0))


def test_goal_kick_pushes_only_rivals_inside_the_box():
    env = _env()
    players = list(FAR)
    players[0] = (900.0, 300.0)                            # dentro del área (840 / 353,11)
    players[1] = (900.0, 360.0)                            # fuera en y
    players[2] = (830.0, 0.0)                              # fuera en x: sin margen
    _place(env, players)
    env.start_restart(0, C.GOAL_KICK, 1, (PRM["goal_kick_x"], PRM["goal_kick_y"]))
    assert np.allclose(env.player_pos[0, 0], (825.0, 300.0))
    assert np.allclose(env.player_pos[0, 1], (900.0, 360.0))
    assert np.allclose(env.player_pos[0, 2], (830.0, 0.0))
