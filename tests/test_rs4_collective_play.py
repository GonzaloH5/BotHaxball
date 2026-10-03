"""Regressions for the user's replay: outlets, kickoffs and expired corners."""
import numpy as np
import pytest
from numba import get_num_threads, set_num_threads

from bots.scripted import scripted_actions
from env.rs4_tactics import passing_outlets, penetrating_outlet, components
from env.rs4_v3 import _restart_geometry
from env.rs4_v3 import RS4ScenarioEnv
from env.haxball_env import HaxballEnv
from env.tasks import load_catalog, make_env


@pytest.fixture(autouse=True)
def threads():
    previous = get_num_threads()
    set_num_threads(min(2, previous))
    yield
    set_num_threads(previous)


def test_receivers_must_have_open_lanes_and_be_distinct():
    ball = np.array([0., 0.])
    own = np.array([[-600., 0.], [-20., 0.], [250., -200.], [250., 200.]])
    opp = np.array([[900., 0.], [700., -400.], [700., 400.], [600., 0.]])
    open_score = passing_outlets(ball, own, opp, 1000.)
    blocked = opp.copy()
    blocked[:2] = own[2:] / 2
    assert open_score > passing_outlets(ball, own, blocked, 1000.)
    crowded = own.copy()
    crowded[2:] = [20., 20.]
    assert open_score > passing_outlets(ball, crowded, opp, 1000.)


@pytest.mark.parametrize('version', [3, 4])
def test_collective_geometry_is_bounded_and_has_no_color_or_identity_bias(version):
    rng = np.random.default_rng(91)
    players = rng.uniform([-900., -500.], [900., 500.], (20, 8, 2))
    ball = rng.uniform([-900., -500.], [900., 500.], (20, 2))
    teams = np.repeat([0, 1], 4)
    def score(p, b):
        return components(p, teams, b, 1000., 600., 120., version)
    actual = score(players, ball)
    assert np.isfinite(actual).all() and ((0 <= actual) & (actual <= 1)).all()
    np.testing.assert_allclose(score(players[:, [2, 0, 3, 1, 6, 4, 7, 5]], ball), actual)
    np.testing.assert_allclose(score(players * [1, -1], ball * [1, -1]), actual)
    np.testing.assert_allclose(score(players[:, [4, 5, 6, 7, 0, 1, 2, 3]] * [-1, 1], ball * [-1, 1]), actual[:, ::-1])


def test_area_receiver_needs_depth_open_lane_and_space():
    ball = np.array([600., 250.])
    own = np.array([[0., 0.], [580., 250.], [450., -200.], [820., 0.]])
    opp = np.array([[980., 0.], [750., -350.], [400., 350.], [900., 350.]])
    opened = penetrating_outlet(ball, own, opp, 1000., 120.)
    assert 0 < opened <= 1
    waiting = own.copy()
    waiting[3] = [500., 0.]
    assert opened > penetrating_outlet(ball, waiting, opp, 1000., 120.)
    blocked = opp.copy()
    blocked[1] = (ball + own[3]) / 2
    assert opened > penetrating_outlet(ball, own, blocked, 1000., 120.)
    marked = opp.copy()
    marked[1] = own[3]
    assert opened > penetrating_outlet(ball, own, marked, 1000., 120.)


def test_attack_guide_keeps_cover_and_prefers_receiver_to_rebound_waiter():
    ball = np.array([[600., 250.]])
    positions = np.array([[[0., 0.], [580., 250.], [450., -200.], [820., 0.],
                           [980., 0.], [750., -350.], [400., 350.], [900., 350.]]])
    teams = np.repeat([0, 1], 4)
    opened = components(positions, teams, ball, 1000., 600., 120., 4)[0, 0, 0]
    waiting = positions.copy()
    waiting[0, 3] = [500., 0.]
    assert opened > components(waiting, teams, ball, 1000., 600., 120., 4)[0, 0, 0]
    uncovered = positions.copy()
    uncovered[0, 0] = [850., 250.]
    assert opened > components(uncovered, teams, ball, 1000., 600., 120., 4)[0, 0, 0]
    # Opponent accesses the ball: do not introduce an attacking-run incentive.
    positions[0, :4] = [[-800., 0.], [-650., 200.], [-700., -200.], [-600., 0.]]
    positions[0, 4] = ball[0]
    np.testing.assert_allclose(components(positions, teams, ball, 1000., 600., 120., 4),
                               components(positions, teams, ball, 1000., 600., 120., 3))


@pytest.mark.parametrize('team', [0, 1])
def test_lateral_prefers_one_taker_and_distinct_supports_over_three_takers(team):
    teams = np.repeat([0, 1], 4)
    ball = np.array([[0., 590.]])
    positions = np.array([[[0., 620.], [-280., 398.], [180., 230.], [-480., 80.],
                           [850., 0.], [700., -300.], [700., 300.], [500., 0.]]])
    if team:
        positions = positions[:, [4, 5, 6, 7, 0, 1, 2, 3]] * [-1, 1]
    spread = _restart_geometry(positions, teams, np.array([team]), ball, 1000., 600., .025)
    crowded = positions.copy()
    ours = np.flatnonzero(teams == team)
    crowded[0, ours[1:3]] = [[30., 560.], [-30., 560.]]
    crowd = _restart_geometry(crowded, teams, np.array([team]), ball, 1000., 600., .025)
    assert spread[0, ours[0]] > crowd[0, ours[0]]


@pytest.mark.parametrize('team', [0, 1])
def test_practice_includes_crowded_laterals_and_compact_defenses(team):
    env = RS4ScenarioEnv(HaxballEnv(96, 4, 'rs_one', out_of_bounds=True,
                                   obs_layout='universal', seed=73), {'scenario_difficulty': 1.})
    env.reset()
    rows = np.arange(env.N)
    env.assign(rows, scenario='lateral', team=team)
    own = env.sim.player_team == team
    distance = np.linalg.norm(env.sim.player_pos[:, own] - env.sim.ball_pos[:, None], axis=-1)
    assert ((distance < .2 * env.field_w).sum(axis=1) >= 3).any()
    env.assign(rows, scenario='attack', team=team)
    sign = 1 if team == 0 else -1
    compact = sign * env.sim.ball_pos[:, 0] > .5 * env.field_w
    assert compact.any() and (~compact).any()
    assert np.isfinite(env.observe()).all()


def test_corner_executor_prefers_inward_alignment_at_equal_distance():
    teams = np.repeat([0, 1], 4)
    ball = np.array([[990., 590.]])
    positions = np.zeros((1, 8, 2))
    positions[0, 1:4] = [[700., 350.], [800., 100.], [400., 0.]]
    direction = ball[0] / np.linalg.norm(ball[0])
    positions[0, 0] = ball[0] + 30 * direction
    ready = _restart_geometry(positions, teams, np.array([0]), ball, 1000., 600., .025)
    positions[0, 0] = ball[0] - 30 * direction
    wrong_side = _restart_geometry(positions, teams, np.array([0]), ball, 1000., 600., .025)
    assert ready[0, 0] > wrong_side[0, 0]


def test_restart_guidance_accepts_open_backward_pass_but_not_blocked_lane():
    teams = np.repeat([0, 1], 4)
    positions = np.array([[[30., 0.], [-200., 0.], [-750., -200.], [-800., 250.],
                           [800., 0.], [700., 300.], [700., -300.], [500., 0.]]])
    ball = np.zeros((1, 2))
    opened = _restart_geometry(positions, teams, np.array([0]), ball, 1000., 600., .025)
    positions[0, 7] = [-100., 0.]
    blocked = _restart_geometry(positions, teams, np.array([0]), ball, 1000., 600., .025)
    assert opened[0, 0] > blocked[0, 0]


@pytest.mark.parametrize('team', [0, 1])
def test_difficult_practice_requires_support_runs_and_remote_restart_travel(team):
    env = RS4ScenarioEnv(HaxballEnv(64, 4, 'rs_one', out_of_bounds=True,
                                   obs_layout='universal', seed=51), {'scenario_difficulty': 1.})
    env.reset()
    rows = np.arange(env.N)
    env.assign(rows, scenario='attack', team=team)
    sign = 1 if team == 0 else -1
    own = env.sim.player_team == team
    ahead = (sign * env.sim.player_pos[:, own, 0] > sign * env.sim.ball_pos[:, :1]).sum(axis=1)
    assert (ahead == 0).any() and (ahead >= 2).any()
    env.assign(rows, scenario='corner', team=team)
    distance = np.linalg.norm(env.sim.player_pos[:, own] - env.sim.ball_pos[:, None], axis=-1).min(axis=1)
    assert (distance > .3 * env.field_w).any() and (distance < .25 * env.field_w).any()
    for row in rows:
        assert env.setpiece_limit[row] >= env._restart_travel_ticks(row, team) + 180


@pytest.mark.parametrize('team', [0, 1])
@pytest.mark.parametrize('optimized', [True, False])
def test_r3_kickoff_passes_toward_a_teammate(team, optimized):
    env = make_env(load_catalog()['rs4_4v4'], 1, 7, seed=1, random_reset_prob=0,
                   optimize_rollout=optimized)
    env.corner_reset_prob = 0
    env.reset()
    env._reset_envs(np.array([0]), kickoff_team=np.array([team]))
    sign = 1 if team == 0 else -1
    own = np.flatnonzero(env.sim.player_team == team)
    env.sim.player_pos[0, own] = np.array([[-30., 0.], [-180., 150.], [-650., -200.], [-900., 0.]]) * [sign, 1]
    env.sim.player_vel[:] = 0
    target = env.sim.player_pos[0, own[1]].copy()
    for _ in range(int(env.kickoff_limit[0]) // env.frame_skip):
        _, _, done, info = env.step(scripted_actions(env, policy='r3'))
        assert not done.any()
        if info['kicked'][0, own].any():
            velocity = env.sim.ball_vel[0]
            assert np.dot(velocity, target) / (np.linalg.norm(velocity) * np.linalg.norm(target)) > .9
            return
    pytest.fail('R3 did not execute its kickoff')


@pytest.mark.parametrize('policy', ['r2', 'r3'])
@pytest.mark.parametrize('decision_ticks', [1, 3])
def test_corners_complete_from_scattered_match_positions(policy, decision_ticks):
    env = make_env(load_catalog()['rs4_4v4'], 32, 7, seed=33, random_reset_prob=0)
    env.corner_reset_prob = 1
    env.reset()
    env.frame_skip = 1  # rendering repeats each decision over physics ticks
    rng = np.random.default_rng(48)
    env.sim.player_pos[:] = rng.uniform([-env.field_w*.8, -env.field_h*.8],
                                        [env.field_w*.8, env.field_h*.8], env.sim.player_pos.shape)
    env.sim.player_vel[:] = 0
    owners = env.setpiece_team.copy()
    for row in range(env.N):
        env.setpiece_limit[row] = max(env.setpiece_timeout, env._restart_travel_ticks(row, int(owners[row])) + 180)
    completed = np.zeros(env.N, bool)
    limit = int(env.setpiece_limit.max())
    for tick in range(limit):
        if tick % decision_ticks == 0:
            actions = scripted_actions(env, policy=policy)
        _, _, done, info = env.step(actions)
        kicked = (info['kicked'] & (env.sim.player_team[None, :] == owners[:, None])).any(axis=1)
        assert not (done & ~completed & ~kicked).any(), 'corner expired before execution'
        completed |= kicked
        if completed.all():
            return
    pytest.fail('some corners never completed')
