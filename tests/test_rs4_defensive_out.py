import copy

import numpy as np
import pytest

from env.rewards import RewardConfig
from env.tasks import load_catalog, make_env


def environment(scale=.2, optimized=True):
    env = make_env(load_catalog()["rs4_4v4"], 1, 7,
                   RewardConfig(shaping_coef=0, kickoff_approach=0, team_spread_floor=0,
                                rs4_defensive_out_scale=scale), seed=29,
                   random_reset_prob=0, optimize_rollout=optimized)
    env.reset()
    env.sim.kickoff[:] = False
    env.sim.mask[:] = env.sim.base_mask
    env.sim.vel[:] = 0
    env.sim.player_pos[:] = [0, 0]
    return env


def contact(env, team=0, own_x=-.7, pressured=True, ambiguous=False):
    sign = 1 if team == 0 else -1
    env.sim.ball_pos[:] = [sign * own_x * env.goal_x, 0]
    env.sim.player_pos[:] = [0, 0]
    rival = np.flatnonzero(env.sim.player_team != team)[0]
    if pressured:
        env.sim.player_pos[:, rival] = env.sim.ball_pos + [20, 0]
    env._out_pressure_players = env.sim.player_pos.copy()
    touches = np.zeros((1, 8), bool)
    touches[0, np.flatnonzero(env.sim.player_team == team)[0]] = True
    if ambiguous:
        touches[0, rival] = True
    env._record_defensive_out_touch(touches)
    env.last_touch[:] = team


def scale(env, side=True, stuck=False):
    return env._out_penalty_scale(np.array([side]), np.array([stuck]))[0]


@pytest.mark.parametrize("team", [0, 1])
def test_discount_requires_own_defensive_zone_and_pressure(team):
    env = environment()
    contact(env, team)
    assert scale(env) == .2
    contact(env, team, own_x=.7)
    assert scale(env) == 1
    contact(env, team, pressured=False)
    assert scale(env) == 1


def test_no_discount_for_endline_stuck_restart_ambiguous_or_old_contact():
    env = environment()
    contact(env)
    assert scale(env, side=False) == scale(env, stuck=True) == 1
    env.ticks[:] = 121
    assert scale(env) == 1
    contact(env)
    env.setpiece_team[:] = 0
    assert scale(env) == 1
    env.setpiece_team[:] = -1
    env.sim.kickoff[:] = True
    assert scale(env) == 1
    env.sim.kickoff[:] = False
    contact(env, ambiguous=True)
    assert scale(env) == 1
    env._out_pressure_open = np.array([False])
    contact(env)
    assert scale(env) == 1


def test_later_touch_replaces_context_and_reset_clears_it():
    env = environment()
    contact(env)
    env.last_touch[:] = 1
    assert scale(env) == 1
    contact(env)
    assert scale(env) == .2
    contact(env, own_x=0)
    assert scale(env) == 1
    contact(env)
    env.reset()
    assert env._defensive_out_team[0] == -1
    assert scale(env) == 1


@pytest.mark.parametrize("optimized", [True, False])
def test_real_out_remains_negative_and_physics_unchanged(optimized):
    env = environment(optimized=optimized)
    contact(env)
    env.sim.player_pos[:] = [0, 0]  # no new contacts while the ball exits
    env.sim.ball_pos[:] = [-.7 * env.goal_x, env.field_h + env.sim.st.ball["radius"] + 10]
    env._phi = env._potentials()
    baseline = copy.deepcopy(env)
    baseline.rcfg.rs4_defensive_out_scale = 1
    actions = np.zeros((1, 8), int)
    actual, expected = env.step(actions), baseline.step(actions)
    assert actual[3]["out"][0]
    np.testing.assert_allclose(actual[1][0, :4], -.02, atol=1e-7)
    np.testing.assert_allclose(expected[1][0, :4], -.1, atol=1e-7)
    np.testing.assert_array_equal(actual[1][0, 4:], expected[1][0, 4:])
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(env.sim.pos, baseline.sim.pos)
    assert env.rng.bit_generator.state == baseline.rng.bit_generator.state
    assert env._defensive_out_team[0] == -1


def test_opt_in_paths_match_with_real_contacts():
    optimized, reference = environment(), environment(optimized=False)
    for env in (optimized, reference):
        contact(env)
        env.sim.player_pos[:, 0] = env.sim.ball_pos + [10, 0]
        env._phi = env._potentials()
    rng = np.random.default_rng(18)
    for _ in range(20):
        actions = rng.integers(0, 18, (1, 8))
        actual, expected = optimized.step(actions), reference.step(actions)
        for a, b in zip(actual[:3], expected[:3]):
            np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(optimized._defensive_out_team, reference._defensive_out_team)


def test_default_keeps_original_penalty_and_rejects_wrong_scope():
    env = environment(scale=1)
    contact(env)
    assert scale(env) == 1
    with pytest.raises(ValueError, match="rs_one"):
        make_env(load_catalog()["futsal_3v3"], 1, 5, RewardConfig(rs4_defensive_out_scale=.2))
    for value in (0, -.1, 1.1, float("nan")):
        with pytest.raises(ValueError, match="scale"):
            environment(value)
