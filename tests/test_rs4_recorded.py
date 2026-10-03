"""Situaciones técnicas desde estados reales (escenario `recorded`, PLAN_RS4 sección 3)."""
import numpy as np
import pytest

from env.haxball_env import HaxballEnv
from env.rs4_states import (ATTACK_RADIUS, GOAL_X, KIND_GOAL_KICK, KIND_LATERAL, KIND_OPEN, POOLS, StateBank,
                            apply_states, mix_weights)
from env.rs4_v3 import RECORDED, RS4ScenarioEnv
from env.rs_one_referee import CONTRACT, LATERAL

RED = [(-900.0, 0.0), (-400.0, -200.0), (-300.0, 250.0), (100.0, 0.0)]
BLUE = [(900.0, 0.0), (400.0, 200.0), (300.0, -250.0), (-100.0, 40.0)]


def state(kind=KIND_OPEN, ball=(0.0, 0.0), velocity=(0.0, 0.0), last=0, taker=-1, spot=(0.0, 0.0), players=None):
    positions = np.asarray(players if players is not None else RED + BLUE, dtype=np.float64)
    return dict(ball_pos=np.asarray(ball, dtype=np.float64), ball_vel=np.asarray(velocity, dtype=np.float64),
                player_pos=positions, player_vel=np.tile([[0.5, -0.25]], (8, 1)) * np.arange(1, 9)[:, None],
                kick_held=np.arange(8) == 2, last_touch=last, kind=kind, taker=taker,
                spot=np.asarray(spot, dtype=np.float64), goal_team=-1, goal_ticks=-1)


def bank_file(tmp_path, *states, name="banco.npz"):
    arrays = {key: np.stack([np.asarray(s[key]) for s in states]) for key in states[0]}
    path = tmp_path / name
    np.savez(path, **arrays)
    return path


def scenario_env(path, n=4, ticks=720, mix=None, referee="rs_one_v1", **settings):
    base = HaxballEnv(n, 4, "rs_one", frame_skip=3, out_of_bounds=True, obs_layout="universal", max_entities=7,
                      kickoff_timeout=180, seed=11, referee=referee)
    recorded = {"path": str(path), "ticks": ticks, **({"mix": mix} if mix else {})}
    settings = {"drill_fraction": .4, "scenario_weights": {"recorded": 1}, "recorded": recorded,
                "guide_coef": 0., **settings}
    return RS4ScenarioEnv(base, settings)


def idle(env):
    return np.zeros((env.N, env.P), dtype=np.int64)


def test_states_are_placed_exactly_and_mirror_swaps_colors(tmp_path):
    attack = state(ball=(700.0, 50.0), velocity=(1.0, -.5), last=0)
    env = scenario_env(bank_file(tmp_path, attack))
    env.reset()
    base, sim = env.base, env.sim
    apply_states(base, [0, 1], env._bank, [0, 0], mirror=[False, True])
    fp = sim.first_player
    np.testing.assert_allclose(sim.pos[0, 0], (700.0, 50.0))
    np.testing.assert_allclose(sim.vel[0, 0], (1.0, -.5))
    np.testing.assert_allclose(sim.pos[0, fp:], attack["player_pos"])
    np.testing.assert_array_equal(sim.kick_cancel[0], attack["kick_held"])
    assert base.last_touch[0] == 0 and base.setpiece_team[0] == -1 and not sim.kickoff[0]
    # Espejo: x reflejado y colores intercambiados (los rojos ocupan las posiciones azules).
    np.testing.assert_allclose(sim.pos[1, 0], (-700.0, 50.0))
    np.testing.assert_allclose(sim.vel[1, 0], (-1.0, -.5))
    expected = np.concatenate([attack["player_pos"][4:], attack["player_pos"][:4]]) * [-1, 1]
    np.testing.assert_allclose(sim.pos[1, fp:], expected)
    assert base.last_touch[1] == 1
    np.testing.assert_array_equal(sim.kick_cancel[1], np.concatenate([attack["kick_held"][4:], attack["kick_held"][:4]]))


def test_restart_state_starts_the_real_referee_set_piece(tmp_path):
    lateral = state(KIND_LATERAL, ball=(300.0, 688.0), last=0, taker=1, spot=(300.0, 688.0))
    env = scenario_env(bank_file(tmp_path, lateral))
    env.reset()
    apply_states(env.base, [0, 1], env._bank, [0, 0], mirror=[False, True])
    assert list(env.setpiece_kind[:2]) == [LATERAL, LATERAL]
    assert list(env.setpiece_team[:2]) == [1, 0]
    np.testing.assert_allclose(env.sim.ball_pos[0], (300.0, CONTRACT["lateral_ball_y"]))
    np.testing.assert_allclose(env.sim.ball_pos[1], (-300.0, CONTRACT["lateral_ball_y"]))


def test_assignment_uses_the_bank_and_records_pool_and_attacker(tmp_path):
    states = [state(ball=(700.0, 50.0), last=0), state(ball=(-200.0, 0.0), last=1),
              state(KIND_GOAL_KICK, ball=(1030.0, 180.0), last=0, taker=1, spot=(1030.0, 180.0))]
    env = scenario_env(bank_file(tmp_path, *states), n=10, ticks=600)
    env.learner_team = np.arange(10) % 2
    env.reset()
    rows = np.flatnonzero(env.is_drill)
    assert len(rows) == 4 and (env.scenario[rows] == RECORDED).all()
    assert (env.drill_limit[rows] == 600).all()
    assert set(env.recorded_pool[rows]) <= {0, 1, 2}
    for row in rows:
        pool = POOLS[env.recorded_pool[row]]
        if pool == "restart":
            assert env.recorded_from_restart[row] and env.recorded_attacker[row] == env.setpiece_team[row]
        else:
            assert not env.recorded_from_restart[row] and env.recorded_attacker[row] == env.last_touch[row]
            ball_x = env.sim.ball_pos[row, 0] * (1 if env.recorded_attacker[row] == 0 else -1)
            distance = np.hypot(GOAL_X - ball_x, env.sim.ball_pos[row, 1])
            assert (distance <= ATTACK_RADIUS) == (pool == "attack")
        assert env.recorded_state[row] in (0, 1, 2)


def test_observation_has_no_private_scenario_information(tmp_path):
    env = scenario_env(bank_file(tmp_path, state(ball=(700.0, 50.0), last=0)), n=2)
    env.reset()
    env.assign(np.arange(2), scenario="recorded", team=0)
    obs = env.observe()
    assert obs.shape == (2, 8, 127)
    env.scenario[:] = -1
    env.drill_team[:] = 1
    env.recorded_pool[:] = -1
    assert np.array_equal(obs, env.observe())


def test_open_play_situation_ends_when_a_new_restart_is_called(tmp_path):
    leaving = state(ball=(0.0, 655.0), velocity=(0.0, 6.0), last=0)
    env = scenario_env(bank_file(tmp_path, leaving), n=2)
    env.reset()
    env.assign(np.arange(2), scenario="recorded", team=0)
    finished = {}
    for _ in range(10):
        _, _, done, info = env.step(idle(env))
        for result in info["scenario_result"]:
            finished.setdefault(result["row"], (result, bool(done[result["row"]]), bool(info["truncated"][result["row"]])))
        if len(finished) == 2:
            break
    assert sorted(finished) == [0, 1]
    for result, done, truncated in finished.values():
        assert result["scenario"] == "recorded" and result["pool"] == "open"
        assert result["truncated"] and done and truncated and result["goal_team"] == -1
        assert result["ticks"] < 30


def test_restart_situation_survives_its_own_piece_and_ends_at_the_next(tmp_path):
    lateral = state(KIND_LATERAL, ball=(0.0, 688.0), last=0, taker=1, spot=(0.0, 688.0))
    env = scenario_env(bank_file(tmp_path, lateral), n=1)
    env.reset()
    env.assign(np.arange(1), scenario="recorded", team=0)
    assert env.recorded_from_restart[0] and env.setpiece_team[0] >= 0
    for _ in range(5):                      # nadie saca: la situación sigue con su propio saque
        _, _, done, info = env.step(idle(env))
        assert not done[0] and not info["scenario_result"]
    # El que saca la mete y queda en juego...
    env.sim.ball_pos[0] = (0.0, 300.0)
    env.sim.ball_vel[0] = (0.0, 0.0)
    _, _, done, info = env.step(idle(env))
    assert env.setpiece_team[0] == -1 and env.recorded_released[0] and not done[0]
    # ...y la próxima salida termina la situación.
    env.sim.ball_pos[0] = (0.0, -660.0)
    env.sim.ball_vel[0] = (0.0, -8.0)
    for _ in range(4):
        _, _, done, info = env.step(idle(env))
        if done[0]:
            break
    assert done[0] and info["truncated"][0]
    (result,) = info["scenario_result"]
    assert result["pool"] == "restart" and result["truncated"]


def test_situation_is_truncated_at_its_tick_limit(tmp_path):
    env = scenario_env(bank_file(tmp_path, state(ball=(-200.0, 0.0), last=1)), n=1, ticks=30)
    env.reset()
    env.assign(np.arange(1), scenario="recorded", team=0)
    for step in range(10):
        _, _, done, info = env.step(idle(env))
        if done[0]:
            break
    assert step == 9 and info["truncated"][0]
    (result,) = info["scenario_result"]
    assert result["ticks"] == 30 and result["truncated"] and not result["success"]


def test_goal_ends_the_situation_as_a_real_terminal_and_success_for_the_scorer(tmp_path):
    shot = state(ball=(1080.0, 0.0), velocity=(12.0, 0.0), last=0)
    env = scenario_env(bank_file(tmp_path, shot), n=2)
    env.reset()
    env.assign(np.arange(2), scenario="recorded", team=0)
    finished = {}
    for _ in range(5):
        _, reward, done, info = env.step(idle(env))
        for result in info["scenario_result"]:
            finished[result["row"]] = (result, bool(info["truncated"][result["row"]]), reward[result["row"]].copy())
        if len(finished) == 2:
            break
    assert sorted(finished) == [0, 1]
    for result, truncated, reward in finished.values():
        assert result["pool"] == "attack" and not truncated and not result["truncated"]
        assert result["goal_team"] == result["attacker"]
        assert result["success"] == (result["team"] == result["goal_team"])
        assert result["conceded"] == int(result["team"] != result["goal_team"])
        scorer = env.sim.player_team == result["goal_team"]
        assert (reward[scorer] > .9).all() and (reward[~scorer] < -.9).all()


def test_explicit_recorded_without_bank_or_real_referee_fails_clearly(tmp_path):
    base = HaxballEnv(1, 4, "rs_one", out_of_bounds=True, obs_layout="universal", referee="rs_one_v1")
    env = RS4ScenarioEnv(base, {})
    env.reset()
    with pytest.raises(ValueError, match="recorded"):
        env.assign(np.arange(1), scenario="recorded", team=0)
    with pytest.raises(ValueError, match="recorded"):
        env.configure({"scenario_weights": {"recorded": 1}})
    legacy = scenario_env(bank_file(tmp_path, state()), n=1, referee="simplified", scenario_weights={"attack": 1})
    legacy.reset()
    with pytest.raises(ValueError, match="rs_one_v1"):
        legacy.assign(np.arange(1), scenario="recorded", team=0)


def test_attack_means_near_the_attacked_goal_of_the_last_touch(tmp_path):
    bank = StateBank(bank_file(tmp_path, state(ball=(700.0, 0.0), last=0), state(ball=(-700.0, 0.0), last=1),
                               state(ball=(-700.0, 0.0), last=0), state(ball=(1000.0, 480.0), last=0),
                               state(ball=(1000.0, 0.0), last=-1)))
    np.testing.assert_array_equal(bank.attack_mask(), [True, True, False, False, False])
    np.testing.assert_array_equal(bank.pools()["open"], [2, 3, 4])


def test_mix_weights_skip_empty_pools_and_reject_unknown_kinds(tmp_path):
    bank = StateBank(bank_file(tmp_path, state(ball=(700.0, 0.0), last=0), state(ball=(-100.0, 0.0), last=0)))
    np.testing.assert_allclose(mix_weights(bank, {"attack": .4, "open": .3, "restart": .3}), [4 / 7, 3 / 7, 0])
    with pytest.raises(ValueError, match="desconocidos"):
        mix_weights(bank, {"corner": 1})
    with pytest.raises(ValueError, match="disponibles"):
        mix_weights(bank, {"restart": 1})


def test_support_potential_only_charges_distance_beyond_humans(tmp_path):
    from env.rs4_v3 import HUMAN_SUPPORT_P75
    env = scenario_env(bank_file(tmp_path, state()), n=1, support_coef=.15)
    env.reset()
    sim = env.sim
    ball = np.array([[0.0, 0.0]])
    red = np.array([[10.0, 0.0], [100.0, 0.0], [200.0, 0.0], [300.0, 0.0]])  # dentro del p75 humano
    blue = np.array([[-10.0, 0.0], [-100.0, 0.0], [-200.0, 0.0], [-900.0, 0.0]])  # el 4.º, 458 px de más
    positions = np.concatenate([red, blue])[None]
    phi = env.support_potential(positions, ball)
    np.testing.assert_allclose(phi[0, sim.player_team == 0], 0.0)
    np.testing.assert_allclose(phi[0, sim.player_team == 1], -(900.0 - HUMAN_SUPPORT_P75[2]) / env.field_w)
    clumped = np.concatenate([red[:1].repeat(4, 0), blue])[None]  # amontonarse no da premio
    assert (env.support_potential(clumped, ball)[0, sim.player_team == 0] == 0).all()


def test_support_guide_is_off_by_default_and_shapes_with_gamma(tmp_path):
    bank = bank_file(tmp_path, state(ball=(-200.0, 0.0), last=1))
    plain, shaped = scenario_env(bank, n=1), scenario_env(bank, n=1, support_coef=.15)
    rng = np.random.default_rng(0)
    for env in (plain, shaped):
        env.reset()
    for _ in range(40):
        actions = rng.integers(0, 18, (1, 8))
        before = shaped._support_phi.copy()
        _, r_plain, done_p, info_p = plain.step(actions)
        _, r_shaped, done_s, info = shaped.step(actions)
        positions = info["final_obs"][:, :, :2] * [shaped.field_w, shaped.field_h]
        positions[..., 0] *= shaped.sign[None]
        ball = info["final_obs"][:, 0, 4:6] * [shaped.field_w, shaped.field_h]
        after = shaped.support_potential(positions, ball)
        absorbing = done_s & ~info["truncated"]
        expected = r_plain + .15 * (shaped.rcfg.gamma * np.where(absorbing[:, None], 0., after) - before)
        np.testing.assert_allclose(r_shaped, expected, atol=1e-9)
        assert (done_p == done_s).all()
