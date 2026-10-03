"""Reglas del contrato RS4-Z-2 (mecanismos reales del script de Real Soccer ONE)."""
import numpy as np
import pytest

from env.rs4z import contract as C
from env.rs4z import kernel as K
from env.rs4z.core import RS4ZEnv

STAND = np.zeros((1, 8), dtype=np.int64)


def _env(**kw):
    kw.setdefault("seed", 0)
    return RS4ZEnv(1, contract="v2", **kw)


def _place(env, ball, players, last_touch=-1, ball_vel=(0.0, 0.0)):
    env.place(0, ball_pos=ball, ball_vel=ball_vel, player_pos=np.asarray(players, dtype=float),
              player_vel=np.zeros((8, 2)), last_touch=last_touch)


FAR = [(-400, 300), (-300, -300), (-200, 200), (-100, -100), (400, 300), (300, -300), (200, 200), (100, -100)]


def test_mass_is_map_value_after_reset_and_script_value_after_lateral_kick():
    env = _env()
    fp = env.fp
    assert np.allclose(env.inv[0, fp:], 0.5) and env.ri[0, K.RI_MASS] == 0
    players = list(FAR)
    players[0] = (100.0, 688.0 - 26.0)       # rojo junto a la pelota del lateral (por dentro)
    _place(env, (0.0, 0.0), players)
    env.ri[0, K.RI_MASS] = 0
    env.inv[0, fp:] = 0.5
    env.start_restart(0, C.LATERAL, 0, (100.0, 688.0))
    assert np.allclose(env.inv[0, fp:], 0.5), "el lateral no cambia la masa al empezar"
    kick_up = np.zeros((1, 8), dtype=np.int64)
    kick_up[0, 0] = 5 + 9                    # abajo (+y en el mundo) y patear
    for _ in range(4):
        env.step(kick_up)
        if env.ri[0, K.RI_MASS] == 1:
            break
    assert env.ri[0, K.RI_MASS] == 1 and np.allclose(env.inv[0, fp:], 0.3)


def test_piece_release_sets_script_mass_and_goal_restores_map_mass():
    env = _env()
    fp = env.fp
    players = list(FAR)
    players[4] = (-1140.0 + 20.0, -660.0 + 20.0)  # azul ejecuta el córner del arco rojo
    _place(env, (0.0, 0.0), players)
    env.inv[0, fp:] = 0.5
    env.ri[0, K.RI_MASS] = 0
    env.start_restart(0, C.CORNER, 1, (-1140.0, -660.0))
    assert np.allclose(env.inv[0, fp:], C.V1["piece_inv_mass"])
    act = np.zeros((1, 8), dtype=np.int64)
    for _ in range(40):
        bp = env.ball_pos[0]
        pp = env.player_pos[0, 4]
        d = bp - pp
        # azul: marco propio con x espejado
        own = np.array([-d[0], d[1]])
        from sim.physics import MOVE_UNIT
        act[0, 4] = int(np.argmax(MOVE_UNIT @ (own / max(np.hypot(*own), 1e-9)))) + 9
        env.step(act)
        if env.restart_team[0] < 0:
            break
    assert env.restart_team[0] < 0, "el córner debía ejecutarse"
    assert np.allclose(env.inv[0, fp:], 0.3) and env.ri[0, K.RI_MASS] == 1
    env.reset_kickoff([0], 0)
    assert np.allclose(env.inv[0, fp:], 0.5) and env.ri[0, K.RI_MASS] == 0


def test_clock_is_frozen_while_waiting_for_kickoff():
    env = _env()
    for _ in range(30):
        env.step(STAND)
    assert env.kickoff[0] and env.clock[0] == 0
    env.place(0, ball_pos=(0.0, 0.0), ball_vel=(1.0, 0.0), player_pos=np.asarray(FAR, float),
              player_vel=np.zeros((8, 2)))
    env.step(STAND)
    assert env.clock[0] == env.frame_skip


def test_corner_disc_excludes_only_defenders():
    env = _env()
    players = list(FAR)
    players[0] = (-1100.0, -600.0)            # rojo (defensor) junto al córner
    players[4] = (-1100.0, -620.0)            # azul (atacante) junto al córner
    _place(env, (0.0, 0.0), players)
    env.start_restart(0, C.CORNER, 1, (-1140.0, -660.0))
    for _ in range(3):
        env.step(STAND)
    centre = np.array([-1150.0, -740.0])
    red = np.hypot(*(env.player_pos[0, 0] - centre))
    blue = np.hypot(*(env.player_pos[0, 4] - centre))
    assert red >= 460.0 - 1e-6
    assert blue < 200.0


def test_goal_kick_box_blocks_rivals_at_real_segments_without_teleport():
    env = _env()
    players = list(FAR)
    players[5] = (-1000.0, -360.0)            # azul fuera del área, al costado (y < -335)
    _place(env, (0.0, 0.0), players)
    env.start_restart(0, C.GOAL_KICK, 0, (-1030.0, -180.0))
    down = np.zeros((1, 8), dtype=np.int64)
    down[0, 5] = 5                            # +y: intenta entrar al área por el costado
    for _ in range(40):
        env.step(down)
    x, y = env.player_pos[0, 5]
    assert y <= -320.0 - 15.0 + 1e-6, "lo frena el segmento lateral del área"
    assert abs(x + 1000.0) < 5.0, "no se teletransporta al frente del área"


def test_goal_kick_spot_disc_blocks_everyone_for_180_ticks_then_releases():
    env = _env()
    players = list(FAR)
    players[0] = (-1030.0 + 60.0, -180.0)     # rojo, ejecutor
    _place(env, (0.0, 0.0), players)
    env.start_restart(0, C.GOAL_KICK, 0, (-1030.0, -180.0))
    act = np.zeros((1, 8), dtype=np.int64)
    act[0, 0] = 7                             # rojo: -x (hacia la pelota)
    for _ in range(50):                       # 150 ticks
        env.step(act)
        assert np.hypot(*(env.player_pos[0, 0] - (-1030.0, -180.0))) >= 33.0 - 1e-6
    assert env.restart_team[0] == 0
    act[0, 0] = 7 + 9
    for _ in range(40):
        env.step(act)
        if env.restart_team[0] < 0:
            break
    assert env.restart_team[0] < 0, "tras 180 ticks el ejecutor puede patear"


def test_lateral_barrier_is_two_sided_segment_for_rivals_only():
    env = _env()
    players = list(FAR)
    players[4] = (0.0, 500.0)                 # azul (rival) adentro
    players[1] = (50.0, 500.0)                # rojo (ejecutor) adentro
    _place(env, (0.0, 0.0), players)
    env.start_restart(0, C.LATERAL, 0, (300.0, 688.0))
    down = np.zeros((1, 8), dtype=np.int64)
    down[0, 4] = 5
    down[0, 1] = 5
    for _ in range(40):
        env.step(down)
    assert env.player_pos[0, 4, 1] <= 555.0 - 15.0 + 1e-6
    assert env.player_pos[0, 1, 1] > 600.0


def test_corner_strip_has_no_dead_zone():
    env = _env()
    _place(env, (1155.0, 690.0), FAR, last_touch=0)
    env.ri[0, K.RI_ARMED] = 1
    env.step(STAND)
    assert env.restart_team[0] >= 0, "la pelota en la franja de la esquina se cobra"


@pytest.mark.parametrize("kind,new_kind", [(C.LATERAL, C.LATERAL), (C.CORNER, C.GOAL_KICK), (C.GOAL_KICK, C.CORNER)])
def test_training_deadline_hands_restart_to_rival(kind, new_kind):
    env = _env(deadline=600)
    _place(env, (0.0, 0.0), FAR)
    spot = {C.LATERAL: (200.0, 688.0), C.CORNER: (-1140.0, -660.0), C.GOAL_KICK: (-1030.0, -180.0)}[kind]
    taker = 1 if kind == C.CORNER else 0
    env.start_restart(0, kind, taker, spot)
    forfeits = []
    for _ in range(205):
        ev = env.step(STAND)
        if ev["forfeit"][0] >= 0:
            forfeits.append(int(ev["forfeit"][0]))
            break
    assert forfeits == [taker]
    assert env.restart_kind[0] == new_kind and env.restart_team[0] == 1 - taker


def test_kickoff_deadline_gives_kickoff_to_rival_and_eval_has_no_deadline():
    env = _env(kickoff_deadline=600)
    env.start_match([0], kickoff_team=0)
    for _ in range(205):
        ev = env.step(STAND)
        if ev["forfeit"][0] >= 0:
            break
    assert ev["forfeit"][0] == 0 and env.ri[0, K.RI_KO_TEAM] == 1 and env.kickoff[0]
    ev_env = _env(deadline=0, kickoff_deadline=C.KICKOFF_SAFETY)
    for _ in range(300):
        ev = ev_env.step(STAND)
        assert ev["forfeit"][0] < 0 or _ * 3 >= C.KICKOFF_SAFETY - 3


def test_goal_is_not_terminal_and_match_end_is():
    env = _env()
    env.start_match([0], kickoff_team=0, match_ticks=600)
    players = list(FAR)
    _place(env, (-1100.0, 0.0), players, ball_vel=(-6.0, 0.0), last_touch=1)
    ev = env.step(STAND)
    for _ in range(5):
        if ev["goal"][0]:
            break
        ev = env.step(STAND)
    assert ev["goal"][0] == -1 and not ev["match_end"][0]
    assert env.kickoff[0] and env.ri[0, K.RI_KO_TEAM] == 0 and tuple(env.score[0]) == (0, 1)
    _place(env, (0.0, 0.0), players, ball_vel=(0.5, 0.0))
    ended = False
    for _ in range(400):
        ev = env.step(STAND)
        if ev["match_end"][0]:
            ended = True
            break
    assert ended and env.clock[0] >= 600


def test_inactive_players_never_interact():
    env = _env()
    active = np.array([[True, False, False, False, True, False, False, False]])
    env.start_match([0], active=active, kickoff_team=0)
    assert (env.mask[0, env.fp:][~active[0]] == 0).all() and (env.group[0, env.fp:][~active[0]] == 0).all()
    rng = np.random.default_rng(0)
    for _ in range(300):
        env.step(rng.integers(0, 18, (1, 8)))
    assert (env.player_pos[0, ~active[0], 1] >= 5000).all()
    assert not env.ev_touch[0, ~active[0]].any()


def test_latency_delays_action_by_exact_ticks():
    env = _env(max_delay=12)
    _place(env, (0.0, 0.0), FAR)
    env.delay[0, 0] = 7
    right = np.zeros((1, 8), dtype=np.int64)
    right[0, 0] = 3                           # +x
    x0 = env.player_pos[0, 0, 0]
    env.step(right)                           # ticks 0..2: d=7 → todavía la acción vieja (quieto)
    env.step(right)                           # ticks 3..5
    assert env.player_pos[0, 0, 0] == x0
    env.step(right)                           # ticks 6..8: aplica desde el tick 7
    assert env.player_pos[0, 0, 0] > x0
