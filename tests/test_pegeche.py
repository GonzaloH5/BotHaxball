"""Reglas del script Pegeche (env/pegeche.py): escenarios armados a mano sobre el x6."""
import numpy as np
import pytest

from env.haxball_env import U_ENT_DIM, U_SELF_DIM, HaxballEnv
from env.pegeche import (N_RULE_FEATS, ANIM, CORNER, FREEKICK, FROZEN_INV, GOALKICK,
                         NONE, PENALTY, SP_TIMEOUT, THROW, THROW_RETRY)

STAY, KICK = 0, 9
RIGHT = 3


def _env(T=1, stadium="x6"):
    env = HaxballEnv(1, T, stadium, powershot=True, obs_layout="universal", rules="pegeche",
                     random_reset_prob=0.0, seed=0)
    env.reset()
    sim = env.sim
    sim.kickoff[:] = False
    sim.mask[:] = sim.base_mask
    fp = sim.first_player
    for p in range(sim.P):  # todos lejos de la acción salvo que el test los ubique
        sim.pos[0, fp + p] = (-200.0 + 400 * sim.player_team[p], 400.0 - 60 * p)
    sim.vel[0] = 0.0
    sim.pos[0, 0] = (0.0, -300.0)
    return env


def _put(env, p, x, y, vx=0.0, vy=0.0):
    k = env.sim.first_player + p
    env.sim.pos[0, k] = (x, y)
    env.sim.vel[0, k] = (vx, vy)


def _steps(env, acts, n):
    out = []
    for _ in range(n):
        out.append(env.step(np.array([acts]))[3])
    return out


def test_obs_width_includes_rule_features():
    env = _env(2, "x6_half")
    assert env.observe().shape == (1, 4, U_SELF_DIM + 3 * U_ENT_DIM)


def test_slide_burst_then_frozen_then_cooldown():
    env = _env()
    r, sim, fp = env.rules, env.sim, env.sim.first_player
    _put(env, 0, -300, 0, 2.0, 0.0)
    speeds, slid_at = [], None
    for i in range(14):  # X mantenida moviéndose a la derecha
        info = env.step(np.array([[RIGHT + KICK, STAY]]))[3]
        speeds.append(np.hypot(*sim.vel[0, fp]))
        if info["rules"]["slide"][0] and slid_at is None:
            slid_at = i
    assert slid_at is not None and 9 <= slid_at <= 11          # ~500 ms
    assert max(speeds) > 4.5                                    # impulso x5.5 (tope 5.5)
    _steps(env, [RIGHT, STAY], 20)                             # pasa la fase de impulso (800 ms)
    assert r.sl_phase[0, 0] == 2
    sp = []
    for _ in range(30):
        env.step(np.array([[RIGHT, STAY]]))
        sp.append(np.hypot(*sim.vel[0, fp]))
    assert max(sp) < 0.5                                        # casi inmóvil 4 s
    _steps(env, [RIGHT, STAY], 60)
    assert r.sl_phase[0, 0] == 0 and r.sl_cd_until[0, 0] > r.clock[0]
    _put(env, 0, -300, 0, 2.0, 0.0)
    infos = _steps(env, [RIGHT + KICK, STAY], 15)               # cooldown: no hay otro slide
    assert not any(i["rules"]["slide"][0] for i in infos)


def test_throw_in_for_other_team_and_valid_throw():
    env = _env()
    r, sim, fp = env.rules, env.sim, env.sim.first_player
    r.last_touch[0] = 0
    _put(env, 0, 100, 450)                                     # rojo (rival del que saca) cerca de la banda
    sim.pos[0, 0] = (100.0, 595.0)
    sim.vel[0, 0] = (0.0, 3.0)
    info = _steps(env, [STAY, STAY], 3)
    assert sum(i["rules"]["out"][0] for i in info) == 1 and any(i["out"][0] for i in info)
    assert r.status[0] == THROW and r.sp_team[0] == 1
    _steps(env, [STAY, STAY], 15)                              # animación
    assert np.allclose(sim.pos[0, 0], (100.0, 618.0))
    assert sim.inv_env[0, fp] == FROZEN_INV and sim.inv_env[0, fp + 1] == 0.5
    _put(env, 0, 100, 495, 0.0, 3.0)                           # el rival no pasa la línea c1 (y = 500)
    _steps(env, [STAY, STAY], 2)
    assert sim.pos[0, fp, 1] <= 500 - 14 + 1e-6
    sim.vel[0, 0] = (0.0, -6.0)                                # el azul la saca hacia la cancha
    _steps(env, [STAY, STAY], 4)
    assert r.status[0] == NONE and r.in_play[0] and sim.inv_env[0, fp] == 0.5


def test_soft_throw_goes_to_the_other_team():
    env = _env()
    r, sim = env.rules, env.sim
    r.last_touch[0] = 0
    sim.pos[0, 0] = (100.0, 595.0)
    sim.vel[0, 0] = (0.0, 3.0)
    _steps(env, [STAY, STAY], 18)
    assert r.status[0] == THROW and r.sp_team[0] == 1
    sim.vel[0, 0] = (0.0, -2.0)                                # empujada sin patear
    _steps(env, [STAY, STAY], 12)
    assert r.status[0] == THROW and r.sp_team[0] == 0


@pytest.mark.parametrize("team", [0, 1])
@pytest.mark.parametrize("frame_skip", [1, 3])
def test_expired_throw_alternates_valid_teams_and_unfreezes_taker(team, frame_skip):
    env = _env(3, "x6_half")
    env.frame_skip = frame_skip
    r, sim, fp = env.rules, env.sim, env.sim.first_player
    for p in range(env.P):
        _put(env, p, -100 + 200 * sim.player_team[p], -100 + 20 * p)
    origin = np.array([100.0, env.field_h + 20.0])
    r._throw_in(0, team, origin)
    actions = [STAY] * env.P
    # Dos vencimientos sucesivos: rojo -> azul -> rojo (o al revés).
    for _ in range(2):
        remaining = SP_TIMEOUT - (r.clock[0] - r.sp_t0[0])
        _steps(env, actions, int(remaining // frame_skip))
        assert r.status[0] == NONE
        assert r.throw_retry_team[0] == 1 - team
        _steps(env, actions, THROW_RETRY // frame_skip)
        team = 1 - team
        assert r.status[0] == THROW and r.sp_team[0] == team
        assert np.array_equal(r.throw_origin[0], origin)
        assert r.throw_retry_at[0] == -1
        _steps(env, actions, (ANIM + frame_skip - 1) // frame_skip)
        own = sim.player_team == team
        assert np.all(sim.inv_env[0, fp:][own] == r.base_inv_p)
        assert np.all(sim.inv_env[0, fp:][~own] == FROZEN_INV)
        assert np.all(r.features()[0, own, 5] == 1.0)
        assert np.all(r.features()[0, ~own, 5] == -1.0)


def test_corner_and_goal_kick():
    env = _env()
    r, sim, fp = env.rules, env.sim, env.sim.first_player
    r.last_touch[0] = 0                                        # la tocó el rojo, sale por su fondo
    sim.pos[0, 0] = (-1150.0, 300.0)
    sim.vel[0, 0] = (-5.0, 0.0)
    _steps(env, [STAY, STAY], 3)
    assert r.status[0] == CORNER and r.sp_team[0] == 1
    _put(env, 0, -1050, 500)
    _steps(env, [STAY, STAY], 16)
    assert np.allclose(sim.pos[0, 0], (-1140.0, 590.0)) and sim.inv_env[0, 0] == 2.5
    assert np.hypot(*(sim.pos[0, fp] - sim.pos[0, 0])) >= 260
    assert sim.inv_env[0, fp] == FROZEN_INV

    env = _env()
    r, sim, fp = env.rules, env.sim, env.sim.first_player
    r.last_touch[0] = 1                                        # la tocó el azul: saque de arco rojo
    sim.pos[0, 0] = (-1150.0, 300.0)
    sim.vel[0, 0] = (-5.0, 0.0)
    _put(env, 1, -900, 100)                                    # azul dentro del área roja
    _steps(env, [STAY, STAY], 3)
    assert r.status[0] == GOALKICK and r.sp_team[0] == 0
    assert np.allclose(sim.pos[0, 0], (-1060.0, 0.0)) and sim.inv_env[0, 0] == 3.0
    _steps(env, [STAY, STAY], 2)
    assert sim.pos[0, fp + 1, 0] >= -840 + 14 - 1e-6 or abs(sim.pos[0, fp + 1, 1]) >= 320


def _foul(env, x):
    """Rojo rápido choca a un azul quieto, con la pelota cerca."""
    sim = env.sim
    _put(env, 0, x - 40, 0, 3.0, 0.0)
    _put(env, 1, x, 0)
    sim.pos[0, 0] = (x - 10, 40.0)
    for _ in range(6):
        env.step(np.array([[RIGHT, STAY]]))
        if env.rules.pf_active[0]:
            return
    raise AssertionError("no se cobró la falta")


def test_foul_claim_gives_free_kick():
    env = _env()
    r, sim, fp = env.rules, env.sim, env.sim.first_player
    _foul(env, 0)
    assert r.pf_fouler[0] == 0 and r.pf_victim[0] == 1 and not r.pf_pen[0]
    assert env.observe()[0, 1, 56 + 12] == 1.0                 # la víctima ve que puede pedirla
    _steps(env, [STAY, KICK], 11)                              # el azul mantiene X 500 ms
    assert r.status[0] == FREEKICK and r.sp_team[0] == 1
    assert sim.inv_env[0, 0] == 2.4
    assert np.hypot(*(sim.pos[0, fp] - sim.pos[0, 0])) >= 260


def test_foul_in_own_box_is_penalty():
    env = _env()
    r, sim, fp = env.rules, env.sim, env.sim.first_player
    _foul(env, -950)                                           # dentro del área roja
    assert r.pf_pen[0]
    _steps(env, [STAY, KICK], 11)
    assert r.status[0] == PENALTY and r.sp_team[0] == 1
    assert np.allclose(sim.pos[0, 0], (-935.0, 0.0))
    _steps(env, [RIGHT, STAY], 5)                              # el arquero (rojo) no sale de la línea
    assert sim.pos[0, fp, 0] <= -1150 + 1e-6 and abs(sim.pos[0, fp, 1]) <= 124


def test_unclaimed_foul_blocks_new_fouls_until_out():
    env = _env()
    r, sim = env.rules, env.sim
    _foul(env, 0)
    r.pf_card[0] = 1
    _steps(env, [STAY, STAY], 110)                             # vencen los 5 s
    assert r.pf_active[0] and r.pf_expired[0]
    r.last_touch[0] = 0
    sim.pos[0, 0] = (100.0, 595.0)
    sim.vel[0, 0] = (0.0, 3.0)
    _steps(env, [STAY, STAY], 3)
    assert not r.pf_active[0] and r.yellow[0, 0] == 1          # la amarilla se aplica en la salida


def test_second_yellow_expels():
    env = _env()
    r, sim, fp = env.rules, env.sim, env.sim.first_player
    r._card(0, 1, 1)
    r._card(0, 1, 1)
    _steps(env, [STAY, RIGHT], 41)
    assert r.expelled[0, 1] and sim.mask[0, fp + 1] == 0
    obs = env.observe()
    assert obs[0, 0, U_SELF_DIM] == 0.0                        # el rojo ya no lo ve
