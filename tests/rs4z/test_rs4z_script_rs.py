"""Script de la familia RS ONE (RS ONE, 2K23): demora de colocación, máscara c0 de la pelota y curva exacta."""
import numpy as np

from env.rs4z import contract as C
from env.rs4z import kernel as K
from env.rs4z.core import RS4ZEnv

FAR = [(-400, 300), (-300, -300), (-200, 200), (-100, -100), (400, 300), (300, -300), (200, 200), (100, -100)]


def _env(map_name="rs_one"):
    return RS4ZEnv(1, contract="v2", map=map_name, seed=0, frame_skip=1, deadline=0, kickoff_deadline=0, max_delay=0)


def _place(env, players, ball, ball_vel, last_touch):
    env.place(0, ball_pos=ball, ball_vel=ball_vel, player_pos=np.asarray(players, dtype=float),
              player_vel=np.zeros((8, 2)), last_touch=last_touch)


def test_rs_maps_use_segment_boost_and_sanguchito_does_not():
    assert _env("rs_one").flags & C.FIX_SEGBOOST and _env("haxarg_2k23").flags & C.FIX_SEGBOOST
    assert not _env("sanguchito_rs_x4").flags & C.FIX_SEGBOOST


def test_corner_is_placed_after_the_room_delay_with_spot_disc():
    env = _env()
    prm = C.MAPS["rs_one"]["overrides"]
    # pelota saliendo por el fondo del arco rojo (-x), último toque del rojo → córner del azul
    _place(env, FAR, ball=(-1150.0, 400.0), ball_vel=(-12.0, 0.0), last_touch=0)
    started = None
    for t in range(80):
        ev = env.step(np.zeros((1, 8), dtype=np.int64))
        if env.ri[0, K.RI_PEND] > 0 and started is None:
            assert env.ri[0, K.RI_TEAM] < 0, "mientras demora, no hay saque activo"
            assert np.isclose(env.radius[0, C.SD_BOTH], C.V1["spot_disc_radius"] if "spot_disc_radius" in C.V1
                              else C.V2["spot_disc_radius"])
            started = t
        if ev["restart_start"][0] == C.CORNER:
            delay = t - started + 1
            assert prm["pend_corner_min"] <= delay <= prm["pend_corner_max"]
            assert env.ri[0, K.RI_TEAM] == 1 and np.allclose(env.ball_pos[0], (-C.V1["corner_x"], C.V1["corner_y"]))
            assert env.mask[0, 0] & (1 << 28) == 0 or env.ri[0, K.RI_C0] != 0
            break
    else:
        raise AssertionError("el córner nunca se colocó")


def test_corner_kick_sets_exact_curve_and_ball_c0_mask_until_restore():
    env = _env()
    _place(env, FAR, ball=(0.0, 0.0), ball_vel=(0.0, 0.0), last_touch=-1)
    spot = (C.V1["corner_x"], C.V1["corner_y"])             # arco azul (+x), abajo
    players = list(FAR)
    players[0] = (spot[0] + 20.0, spot[1] + 12.0)           # rojo ejecuta desde afuera
    _place(env, players, ball=(0.0, 0.0), ball_vel=(0.0, 0.0), last_touch=-1)
    env.start_restart(0, C.CORNER, 0, spot)
    assert env.ri[0, K.RI_C0] == -1
    env.step(np.zeros((1, 8), dtype=np.int64))
    assert env.mask[0, 0] & (1 << 28), "la pelota choca con los segmentos c0 durante el córner"
    act = np.zeros((1, 8), dtype=np.int64)
    act[0, 0] = 9
    # el script despeja el punto (radio 33) al colocar la pelota: el ejecutor vuelve a acercarse
    env.pos[0, env.fp] = (spot[0] + 20.0, spot[1] + 12.5)
    env.vel[0, env.fp] = (-0.7, 0.0)                         # velocidad del pateador antes del tick
    env.step(act)
    assert env.ri[0, K.RI_TEAM] == -1
    prm = C.MAPS["rs_one"]["overrides"]
    assert np.isclose(env.grav[0, 0], prm["corner_grav_kvx"] * -0.7)
    assert np.isclose(env.grav[0, 1], -prm["corner_grav_y"])
    assert env.ri[0, K.RI_C0] == prm["ball_c0_ticks"]
    for _ in range(int(prm["ball_c0_ticks"]) + 1):
        env.step(np.zeros((1, 8), dtype=np.int64))
    assert env.mask[0, 0] & (1 << 28) == 0, "el script le saca c0 a la pelota después de la patada"


def test_sanguchito_places_without_delay():
    env = _env("sanguchito_rs_x4")
    _place(env, FAR, ball=(0.0, 690.0), ball_vel=(0.0, 3.0), last_touch=0)
    for _ in range(3):
        ev = env.step(np.zeros((1, 8), dtype=np.int64))
        if ev["restart_start"][0]:
            break
    assert ev["restart_start"][0] == C.LATERAL and env.ri[0, K.RI_PEND] == 0
