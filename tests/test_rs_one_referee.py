"""Árbitro rs_one_v1: reglas reconstruidas de grabaciones reales de Real Soccer ONE."""
import numpy as np
import pytest

from env.haxball_env import HaxballEnv
from env.rewards import RewardConfig
from env.rs_one_referee import CONTRACT, CORNER, GOAL_KICK, LATERAL


def env_real(n=1, **kwargs):
    reward = RewardConfig(shaping_coef=0, team_spread_floor=0, kickoff_approach=0, w_defense_support=0,
                          rs4_tactical_coef=0, corner_execute=0, out_penalty=0)
    env = HaxballEnv(n, 4, "rs_one", frame_skip=3, max_ticks=kwargs.pop("max_ticks", 7200), random_reset_prob=0,
                     seed=3, obs_layout="universal", max_entities=7, out_of_bounds=True, corner_reset_prob=0,
                     kickoff_timeout=180, reward=reward, referee="rs_one_v1", **kwargs)
    env.reset()
    env.sim.kickoff[:] = False
    return env


def park_players(env, row=0):
    """Jugadores lejos de la pelota y quietos (rojos 0-3 a la izquierda, azules 4-7 a la derecha)."""
    sim = env.sim
    for p in range(env.P):
        x = -300 - 60 * p if sim.player_team[p] == 0 else 300 + 60 * (p - 4)
        sim.player_pos[row, p] = (x, -100 + 50 * (p % 4))
    sim.player_vel[row] = 0


def idle(env):
    return np.zeros((env.N, env.P), dtype=np.int64)


def test_players_use_the_script_mass_and_legacy_default_is_unchanged():
    env = env_real()
    fp = env.sim.first_player
    assert np.allclose(env.sim.inv_env[:, fp:], CONTRACT["play_inv_mass"])
    legacy = HaxballEnv(1, 4, "rs_one", out_of_bounds=True, obs_layout="universal", max_entities=7)
    assert np.allclose(legacy.sim.inv_env[:, legacy.sim.first_player:], 0.5)
    assert legacy.referee == "simplified" and legacy.restart_timeout_terminal


def test_lateral_is_placed_outside_and_released_by_the_taker():
    env = env_real()
    sim = env.sim
    park_players(env)
    sim.ball_pos[0] = (100.0, 670.0)
    sim.ball_vel[0] = (0.0, 6.0)
    env.last_touch[0] = 0                       # tocó el rojo: saca el azul
    sim.player_pos[0, 1] = (150.0, 620.0)       # rival rojo cerca del punto
    sim.player_pos[0, 2] = (-400.0, 610.0)      # rival rojo lejos (fuera de ±270 px)
    env.step(idle(env))
    assert env.setpiece_kind[0] == LATERAL and env.setpiece_team[0] == 1
    assert sim.ball_pos[0, 1] == pytest.approx(CONTRACT["lateral_ball_y"])
    assert np.allclose(sim.ball_vel[0], 0)
    assert sim.player_pos[0, 1, 1] <= CONTRACT["barrier_y"] - 15 + 1e-6   # empujado detrás de la barrera
    spot = sim.ball_pos[0].copy()
    # Tocarla y llevarla por fuera de la línea no libera el saque...
    sim.player_pos[0, 4] = spot + (-23.0, 0.0)
    actions = idle(env)
    actions[0, 4] = 7                             # en coordenadas del azul: hacia +x del mundo
    env.step(actions)
    assert env.setpiece_team[0] == 1
    # ...empujarla hacia la cancha (|y| < 682) sí, como en las grabaciones.
    sim.player_pos[0, 4] = sim.ball_pos[0] + (0.0, 23.0)
    sim.player_vel[0, 4] = 0
    actions[0, 4] = 1                             # arriba: hacia el interior de la cancha
    for _ in range(10):
        env.step(actions)
        if env.setpiece_team[0] < 0:
            break
    assert env.setpiece_team[0] < 0 and abs(sim.ball_pos[0, 1]) < CONTRACT["lateral_release_y"]
    spot = sim.ball_pos[0].copy()
    # Con la pelota todavía fuera de la línea (|y| 681 > 678,3) no se cobra un lateral nuevo.
    sim.ball_pos[0] = (spot[0], 681.0)
    sim.ball_vel[0] = 0
    env.step(idle(env))
    assert env.setpiece_team[0] < 0


def test_corner_mass_boost_and_spin_follow_the_recordings():
    env = env_real()
    sim = env.sim
    park_players(env)
    sim.ball_pos[0] = (1155.0, 400.0)
    sim.ball_vel[0] = (6.0, 0.0)
    env.last_touch[0] = 1                         # tocó el azul (defiende +x): córner rojo
    env.step(idle(env))
    assert env.setpiece_kind[0] == CORNER and env.setpiece_team[0] == 0
    assert np.allclose(sim.ball_pos[0], CONTRACT["corner"])
    assert np.allclose(sim.inv_env[0, sim.first_player:], CONTRACT["piece_inv_mass"])
    sim.player_pos[0, 0] = sim.ball_pos[0] + (25.0, 0.0)  # pateador rojo pegado, del lado de afuera
    sim.player_vel[0, 0] = 0
    actions = idle(env)
    actions[0, 0] = 9                             # patear sin moverse
    speeds = []
    for _ in range(2):
        env.step(actions)
        speeds.append(float(np.hypot(*sim.ball_vel[0])))
        actions[0, 0] = 0
    assert env.setpiece_team[0] < 0
    assert np.allclose(sim.inv_env[0, sim.first_player:], CONTRACT["play_inv_mass"])
    assert 11.0 < speeds[0] < 12.5               # patada normal (~6,1) duplicada a los 2 ticks
    assert np.hypot(*sim.ball_grav[0]) > 0.02     # efecto activo


def test_goal_kick_clears_the_box_and_boosts_with_spin_from_kicker_velocity():
    env = env_real()
    sim = env.sim
    park_players(env)
    sim.ball_pos[0] = (1155.0, -400.0)
    sim.ball_vel[0] = (6.0, 0.0)
    env.last_touch[0] = 0                         # tocó el rojo (ataca +x): saque de arco azul
    sim.player_pos[0, 3] = (1000.0, -100.0)       # atacante rojo dentro del área
    env.step(idle(env))
    assert env.setpiece_kind[0] == GOAL_KICK and env.setpiece_team[0] == 1
    assert np.allclose(sim.ball_pos[0], (CONTRACT["goal_kick"][0], -CONTRACT["goal_kick"][1]))
    assert sim.player_pos[0, 3, 0] <= CONTRACT["box_front"] - 15 + 1e-6
    # Durante 180 ticks nadie puede acercarse a menos de 33 px del punto (disco del script).
    sim.player_pos[0, 4] = sim.ball_pos[0] + (25.0, 0.0)
    env.step(idle(env))
    assert np.hypot(*(sim.player_pos[0, 4] - sim.ball_pos[0])) >= CONTRACT["spot_clearance"] - 1e-6
    while env.setpiece_ticks[0] < CONTRACT["goal_kick_hold_ticks"]:
        env.step(idle(env))
    assert env.setpiece_team[0] == 1
    sim.player_pos[0, 4] = sim.ball_pos[0] + (25.0, 0.0)
    sim.player_vel[0, 4] = (0.0, 2.0)
    actions = idle(env)
    actions[0, 4] = 9
    env.step(actions)
    actions[0, 4] = 0
    env.step(actions)
    assert env.setpiece_team[0] < 0
    assert 14.0 < np.hypot(*sim.ball_vel[0]) < 17.5
    assert sim.ball_grav[0, 1] < 0                # gravedad opuesta al movimiento (+y) del pateador


def test_restarts_never_expire_as_terminals_and_have_a_safety_release():
    env = env_real(max_ticks=100000)
    sim = env.sim
    park_players(env)
    sim.ball_pos[0] = (0.0, 675.0)
    sim.ball_vel[0] = (0.0, 6.0)
    env.last_touch[0] = 1
    _, _, done, info = env.step(idle(env))
    assert env.setpiece_kind[0] == LATERAL
    for _ in range(CONTRACT["safety_ticks"] // 3 + 2):
        _, _, done, info = env.step(idle(env))
        assert not done[0]
        if env.setpiece_team[0] < 0:
            break
    assert env.setpiece_team[0] < 0
    assert not info.get("rs4_restart_failed", np.zeros(1, bool))[0]


def test_many_parallel_matches_with_simultaneous_restarts():
    env = env_real(n=6)
    sim = env.sim
    rng = np.random.default_rng(0)
    for row in range(6):
        park_players(env, row)
    # filas 0-1: córner; 2-3: saque de arco; 4-5: lateral
    sim.ball_pos[:2] = (1155.0, 400.0); env.last_touch[:2] = 1
    sim.ball_pos[2:4] = (1155.0, -400.0); env.last_touch[2:4] = 0
    sim.ball_pos[4:] = (0.0, 675.0); env.last_touch[4:] = 1
    sim.ball_vel[:4] = (6.0, 0.0); sim.ball_vel[4:] = (0.0, 6.0)
    env.step(idle(env))
    assert list(env.setpiece_kind) == [CORNER, CORNER, GOAL_KICK, GOAL_KICK, LATERAL, LATERAL]
    for _ in range(400):
        env.step(rng.integers(0, 18, (6, 8)))
    assert np.isfinite(sim.player_pos).all() and np.isfinite(sim.ball_pos).all()


def _untaken_corner(deadline, stall=0.25):
    env = env_real()
    env.rcfg.rs4_restart_stall = stall
    env._rs1.deadline = deadline
    park_players(env)
    env.sim.ball_pos[0] = (1155.0, 400.0)
    env.sim.ball_vel[0] = (6.0, 0.0)
    env.last_touch[0] = 1                        # tocó el azul en su línea: córner del rojo
    env.step(idle(env))
    assert env.setpiece_kind[0] == CORNER and env.setpiece_team[0] == 0
    return env


def test_training_deadline_fines_the_taker_once_and_play_continues():
    env = _untaken_corner(600)
    fines, fined_at = [], []
    for step in range(CONTRACT["safety_ticks"] // 3 + 2):
        _, reward, done, info = env.step(idle(env))
        assert not done[0]
        if reward[0].any():
            fines.append(reward[0].copy())
            fined_at.append(int(env.setpiece_ticks[0]))
            assert info["events"]["restart_timeouts"][0].tolist() == [1, 0]
        if env.setpiece_team[0] < 0:
            break
    assert len(fines) == 1 and 600 <= fined_at[0] < 603
    np.testing.assert_allclose(fines[0], [-.25] * 4 + [0.] * 4)
    assert env.setpiece_team[0] < 0  # la liberación de seguridad sigue en safety_ticks


def test_without_deadline_or_with_a_timely_kick_there_is_no_fine():
    env = _untaken_corner(0)
    for _ in range(CONTRACT["safety_ticks"] // 3 + 2):
        _, reward, _, _ = env.step(idle(env))
        assert not reward[0].any()
    env = _untaken_corner(600)
    spot = env.setpiece_pos[0].copy()
    env.sim.player_pos[0, 0] = spot + (-24.0, -1.0)   # el rojo patea el córner a tiempo
    kick = idle(env)
    kick[0, 0] = 9
    _, reward, _, _ = env.step(kick)
    assert env.setpiece_team[0] < 0 and not reward[0].any()
    for _ in range(150):  # 450 ticks: ni siquiera un saque nuevo podría llegar al plazo
        _, reward, _, info = env.step(idle(env))
        assert info["events"]["restart_timeouts"][0].sum() == 0
