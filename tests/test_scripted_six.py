"""Six-player teams need stable spacing and a chaser even when losing possession."""
import numpy as np
import pytest
from numba import get_num_threads, set_num_threads

from bots.scripted import scripted_actions, _six_cover_target, _scripted_kernel
from env.tasks import load_catalog, make_env
from sim.physics import MOVE_UNIT


@pytest.fixture(autouse=True)
def threads():
    previous = get_num_threads()
    set_num_threads(min(previous, 2))
    yield
    set_num_threads(previous)


def setup():
    env = make_env(load_catalog()["jjrs_6v6"], 1, 11, seed=0, random_reset_prob=0)
    env.reset()
    env.sim.kickoff[:] = False
    env.sim.mask[:] = env.sim.base_mask
    env.sim.vel[:] = 0
    return env


@pytest.mark.parametrize("team", [0, 1])
def test_field_chaser_does_not_retreat_when_opponent_is_closer(team):
    env = setup()
    sign = 1 if team == 0 else -1
    own = np.flatnonzero(env.sim.player_team == team)
    rival = np.flatnonzero(env.sim.player_team != team)
    env.sim.ball_pos[0] = (-200 * sign, 0)
    for j, p in enumerate(own):
        env.sim.player_pos[0, p] = ((-300 - j * 100) * sign, 0)
    # Even a nearby goalkeeper must not replace the designated field chaser.
    env.sim.player_pos[0, own[-1]] = (-210 * sign, 0)
    env.sim.player_pos[0, rival] = (100 * sign, 200)
    env.sim.player_pos[0, rival[0]] = (-175 * sign, 0)
    action = scripted_actions(env)[0, own[0]] % 9
    assert MOVE_UNIT[action, 0] > 0  # own frame: towards ball, not own goal


@pytest.mark.parametrize("chaser", range(5))
def test_cover_targets_have_separate_lanes(chaser):
    targets = np.array([_six_cover_target(p, -200, 120, 1100, 600, 25)
                        for p in range(6) if p != chaser])
    distances = np.linalg.norm(targets[:, None] - targets[None, :], axis=-1)
    np.fill_diagonal(distances, np.inf)
    assert distances.min() > 100
    assert np.abs(targets[:, 0]).max() < 1100
    assert np.abs(targets[:, 1]).max() < 600


def test_amassed_defenders_spread_with_stationary_opponents():
    env = setup()
    sim = env.sim
    sim.ball_pos[0] = (-200, -100)
    sim.player_pos[0, :6] = (-env.goal_x * .55, -35)
    sim.player_pos[0, 6:] = (300, 250)
    sim.player_pos[0, 6] = (-180, -100)
    initial = sim.player_pos[0, :6].copy()
    for _ in range(200):
        actions = np.zeros((1, 12), dtype=np.int64)
        actions[:, :6] = scripted_actions(env, np.arange(6))
        env.step(actions)
    final = sim.player_pos[0, :6]
    assert np.ptp(final[:, 1]) > 200
    assert np.linalg.norm(final - initial, axis=-1).max() > 200
    assert np.isfinite(sim.pos).all()


def test_four_player_actions_are_unchanged():
    env = make_env(load_catalog()["rs4_4v4"], 4, 7, seed=23, random_reset_prob=1)
    env.reset()
    sim = env.sim
    sim.kickoff[:] = False
    for _ in range(10):
        sim.reset_random(np.arange(env.N))
        args = (np.arange(env.N), np.arange(env.P), sim.ball_pos, sim.ball_vel,
                sim.player_pos, sim.player_team, env.sign, env.goal_x, sim.st.field_half_h,
                float(sim.st.player["radius"] + sim.st.ball["radius"]))
        # T only gates the new six-player branch; legacy branch is identical.
        np.testing.assert_array_equal(_scripted_kernel(*args, 4), _scripted_kernel(*args, 3))
