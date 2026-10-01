"""Guía RS4 opt-in: simetrías, anti-farm, terminales, coeficiente y estilos."""
import copy

import numpy as np
import pytest

from env.rewards import RewardConfig
from env.rs4_tactics import components
from env.tasks import load_catalog, make_env
from train.rs4_specialization import coefficient, validate


TEAM = np.repeat([0, 1], 4)


def state():
    own = np.array([[-900, 0], [-300, 0], [-650, -220], [-650, 220]], dtype=float)
    return np.concatenate((own, own * [-1, 1]))[None], np.array([[-350., 0]])


def measure(players=None, ball=None):
    p, b = state()
    return components(p if players is None else players, TEAM, b if ball is None else ball, 1000., 600., 120.)


def test_components_are_bounded_finite_and_match_python_reference():
    rng = np.random.default_rng(19)
    players = rng.uniform(-1000, 1000, (25, 8, 2))
    ball = rng.uniform(-1000, 1000, (25, 2))
    actual = measure(players, ball)
    expected = components.py_func(players, TEAM, ball, 1000., 600., 120.)
    assert np.isfinite(actual).all() and ((actual >= 0) & (actual <= 1)).all()
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_player_identity_vertical_mirror_and_team_swap_invariance():
    players, ball = state()
    expected = measure(players, ball)
    np.testing.assert_allclose(measure(players[:, [3, 1, 0, 2, 6, 7, 4, 5]], ball), expected)
    np.testing.assert_allclose(measure(players * [1, -1], ball * [1, -1]), expected, atol=1e-12)
    swapped = players[:, np.r_[4:8, 0:4]] * [-1, 1]
    np.testing.assert_allclose(measure(swapped, ball * [-1, 1]), expected[:, ::-1], atol=1e-12)
    np.testing.assert_allclose(expected[:, 0, 1], expected[:, 1, 2])


def test_structure_prefers_coverage_to_all_four_chasing():
    players, ball = state()
    crowded = players.copy()
    crowded[:, :4] = ball[:, None]
    assert measure(players, ball)[0, 0, 0] > measure(crowded, ball)[0, 0, 0]


def test_covering_shot_lanes_reduces_threat_not_a_goal_probability():
    players, _ = state()
    ball = np.array([[600., 0]])
    players[:, :4] = [[580, 0], [200, -250], [200, 250], [-900, 0]]
    players[:, 4:] = [[0, 400], [0, -400], [-300, 400], [-300, -400]]
    open_shot = measure(players, ball)[0, 0, 1]
    covered = players.copy()
    covered[:, 4:] = [[900, 0], [800, -40], [800, 40], [700, 0]]
    assert measure(covered, ball)[0, 0, 1] < open_shot


def environment(coef=0.12, optimized=True, max_ticks=7200):
    env = make_env(load_catalog()["rs4_4v4"], 1, 7,
                   RewardConfig(shaping_coef=0, kickoff_approach=0, team_spread_floor=0,
                                rs4_tactical_coef=coef), seed=29, random_reset_prob=0,
                   optimize_rollout=optimized, max_ticks=max_ticks)
    env.reset()
    env.sim.kickoff[:] = False
    env.sim.mask[:] = env.sim.base_mask
    env.sim.vel[:] = 0
    players, ball = state()
    env.sim.player_pos[:] = players * [env.goal_x / 1000, env.field_h / 600]
    env.sim.ball_pos[:] = ball * [env.goal_x / 1000, env.field_h / 600]
    env._phi = env._potentials()
    env._rs4_phi = env._rs4_potential() if coef else None
    return env


def test_shared_team_rewards_and_no_guide_during_restarts():
    env = environment()
    phi = env._rs4_potential()
    assert phi.max() <= 0.12 and phi.min() >= 0
    np.testing.assert_allclose(phi[:, :4], np.repeat(phi[:, :1], 4, axis=1))
    np.testing.assert_allclose(phi[:, 4:], np.repeat(phi[:, 4:5], 4, axis=1))
    env.sim.kickoff[:] = True
    assert not env._rs4_potential().any()
    env.sim.kickoff[:] = False
    env.setpiece_team[:] = 0
    assert not env._rs4_potential().any()


def test_stationary_and_discounted_cycles_do_not_farm_positive_return():
    env = environment()
    phi = env._rs4_phi
    assert (env.rcfg.gamma * phi - phi <= 0).all()
    # Incluye annealing: el potencial lleva el coeficiente de su propio instante.
    series = [phi, phi * 0.4, phi * 0.7, np.zeros_like(phi)]
    g = env.rcfg.gamma
    total = sum(g**t * (g * series[t + 1] - series[t]) for t in range(3))
    np.testing.assert_allclose(total, -phi, atol=1e-12)


def test_goal_zeroes_terminal_potential_and_keeps_physics_identical():
    env = environment()
    env.sim.ball_pos[:] = [env.goal_x - 1, 0]
    env.sim.ball_vel[:] = [4, 0]
    env._rs4_phi = env._rs4_potential()
    previous = env._rs4_phi.copy()
    baseline = copy.deepcopy(env)
    baseline.rcfg.rs4_tactical_coef, baseline._rs4_phi = 0, None
    actual, expected = env.step(np.zeros((1, 8), int)), baseline.step(np.zeros((1, 8), int))
    assert actual[3]["goal"][0] == 1 and actual[2][0]
    np.testing.assert_allclose(actual[1] - expected[1], -previous, atol=1e-7)
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(env.sim.pos, baseline.sim.pos)
    assert env.rng.bit_generator.state == baseline.rng.bit_generator.state


def test_truncation_keeps_final_potential_for_bootstrap():
    env = environment(max_ticks=3)
    reference = copy.deepcopy(env)
    reference.max_ticks = 7200
    previous = env._rs4_phi.copy()
    actions = np.zeros((1, 8), int)
    actual, final = env.step(actions), reference.step(actions)
    assert actual[3]["truncated"][0]
    np.testing.assert_allclose(actual[3]["rs4_tactical_reward"],
                               env.rcfg.gamma * reference._rs4_phi - previous, atol=1e-12)
    assert np.any(reference._rs4_phi)


def test_decay_to_zero_accounts_for_previous_potential_once():
    env = environment()
    previous = env._rs4_phi.copy()
    env.rcfg.rs4_tactical_coef = 0
    first = env.step(np.zeros((1, 8), int))
    np.testing.assert_allclose(first[3]["rs4_tactical_reward"], -previous)
    assert env._rs4_phi is None
    assert "rs4_tactical_reward" not in env.step(np.zeros((1, 8), int))[3]


def test_style_is_held_across_goals_but_changes_on_full_match_reset():
    env = environment()
    env.hold_scripted_style_for_match = True
    env.reset()
    original = env.scripted_style.copy()
    env.match_ticks[:] = 100
    env._reset_envs(np.array([0]), kickoff_team=np.array([0]))
    np.testing.assert_array_equal(env.scripted_style, original)
    env.match_ticks[:] = env.max_ticks
    env._reset_envs(np.array([0]), kickoff_team=np.array([1]))
    np.testing.assert_array_equal(env.scripted_style, (original + 1) % 3)


def test_annealing_is_branch_local_and_validates_scope():
    cfg = {"stages": [{"tasks": ["rs4_4v4"]}], "rs4_tactics": {
        "coef": .12, "coef_final": 0., "start_steps": 3761000000, "decay_steps": 200000000}}
    validate(cfg)
    assert coefficient(cfg, 3761000000) == .12
    assert coefficient(cfg, 3861000000) == pytest.approx(.06)
    assert coefficient(cfg, 3961000000) == 0
    assert coefficient({}, 9999999999) == 0
    cfg["stages"][0]["tasks"].append("futsal_3v3")
    with pytest.raises(ValueError, match="exclusiva"):
        validate(cfg)


def test_optimized_and_reference_transitions_match_with_tactics():
    optimized, reference = environment(), environment(optimized=False)
    rng = np.random.default_rng(17)
    for _ in range(25):
        actions = rng.integers(0, 18, (1, 8))
        actual, wanted = optimized.step(actions), reference.step(actions)
        for a, b in zip(actual[:3], wanted[:3]):
            np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(actual[3]["rs4_tactical_reward"], wanted[3]["rs4_tactical_reward"])
