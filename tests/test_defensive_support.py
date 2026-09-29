"""Guía JJRS: coberturas distintas, ambos colores, saques y continuidad PPO."""
import copy

import numpy as np
import pytest
from numba import get_num_threads, set_num_threads

from env.rewards import RewardConfig, defense_support_potential
from env.tasks import load_catalog, make_env
from tests.test_multitask import _trainer


@pytest.fixture(autouse=True)
def threads():
    previous = get_num_threads()
    set_num_threads(min(2, previous))
    yield
    set_num_threads(previous)


def positions():
    # Arquero, presión, tres coberturas y salida libre.
    return np.array([[[-920, 0], [-650, 0], [-780, -168],
                      [-780, 0], [-780, 168], [100, 300]]], dtype=float)


def potential(players, ball=None):
    return defense_support_potential(players, np.array([[-700., 0]]) if ball is None else ball,
                                     1000., 600.)


def test_cover_beats_five_waiting_at_midfield():
    waiting = positions()
    waiting[:, 1:] = [[-650, 0], [-100, -168], [-100, 0], [-100, 168], [100, 300]]
    assert potential(positions())[0] > potential(waiting)[0] + 0.3


def test_one_player_cannot_fill_all_cover_lanes():
    clump = positions()
    clump[:, 2:5] = [-780, 0]
    assert potential(positions())[0] > potential(clump)[0]


def test_keeper_and_pressure_do_not_replace_cover():
    only_two = positions()
    only_two[:, 2:] = [0, 0]
    assert potential(only_two)[0] < potential(positions())[0] - 0.3


def test_empty_goal_scores_less():
    empty = positions()
    empty[:, 0] = [0, 400]
    assert potential(empty)[0] < potential(positions())[0]


def test_identity_and_vertical_mirror_invariance():
    p = positions()
    np.testing.assert_allclose(potential(p[:, [4, 2, 0, 5, 1, 3]]), potential(p))
    ball = np.array([[-700., 250.]])
    mirrored = p.copy()
    mirrored[..., 1] *= -1
    np.testing.assert_allclose(potential(mirrored, ball * [1, -1]), potential(p, ball))


def test_no_defense_bonus_in_attack_or_small_formats():
    np.testing.assert_array_equal(potential(positions(), np.array([[600., 0.]])), [0])
    np.testing.assert_array_equal(potential(positions()[:, :4]), [0])


def env_with_reward(optimized=True):
    rc = RewardConfig(shaping_coef=0, w_defense_support=0.35, defense_shaping_floor=0.5,
                      kickoff_approach=0)
    env = make_env(load_catalog()["jjrs_6v6"], 1, 11, rc, seed=21,
                   random_reset_prob=0, optimize_rollout=optimized)
    env.reset()
    env.sim.kickoff[:] = False
    env.sim.vel[:] = 0
    env.sim.mask[:] = env.sim.base_mask
    env.sim.ball_pos[:] = [-0.70 * env.goal_x, 0]
    env.sim.player_pos[:, :6] = positions() * [env.goal_x / 1000, env.field_h / 600]
    env._phi = env._potentials()
    return env


def test_both_colors_and_restarts():
    env = env_with_reward()
    red_phi = env._defense_potential()[0, :6].copy()
    env.sim.player_pos[:] = env.sim.player_pos[:, np.r_[6:12, 0:6]] * [-1, 1]
    env.sim.ball_pos[:] *= [-1, 1]
    np.testing.assert_allclose(env._defense_potential()[0, 6:], red_phi)
    env.sim.kickoff[:] = True
    assert not env._defense_potential().any()
    env.sim.kickoff[:] = False
    env.setpiece_team[:] = 1
    assert not env._defense_potential().any()


def test_standing_still_is_not_a_positive_farming_reward():
    env = env_with_reward()
    phi = env._defense_potential()
    assert phi[0, 0] > 0
    assert (env.rcfg.gamma * phi - phi <= 0).all()
    # Un ciclo completo tampoco acumula retorno descontado positivo.
    g = env.rcfg.gamma
    cycle = [phi[0, 0], phi[0, 0] / 2, phi[0, 0]]
    discounted = (g * cycle[1] - cycle[0]) + g * (g * cycle[2] - cycle[1])
    assert discounted <= 0


def test_floor_remains_active_with_global_shaping_zero():
    env = env_with_reward()
    no_guide = copy.deepcopy(env)
    no_guide.rcfg.w_defense_support = 0
    no_guide._phi = no_guide._potentials()
    previous = env._phi[3].copy()
    actions = np.zeros((1, 12), dtype=int)
    _, actual, _, _ = env.step(actions)
    _, baseline, _, _ = no_guide.step(actions)
    expected = 0.35 * 0.5 * (env.rcfg.gamma * env._phi[3] - previous)
    # El potencial anterior positivo deja una diferencia incluso estando quietos.
    assert not np.array_equal(actual, baseline)
    np.testing.assert_allclose(actual - baseline, expected, atol=1e-7)
    assert np.isfinite(actual).all()
    np.testing.assert_array_equal(env.sim.pos, no_guide.sim.pos)


def test_goal_uses_zero_terminal_potential():
    env = env_with_reward()
    env.sim.ball_pos[:] = [-env.goal_x + 1, 0]
    env.sim.ball_vel[:] = [-4, 0]
    env._phi = env._potentials()
    previous = env._phi[3].copy()
    no_guide = copy.deepcopy(env)
    no_guide.rcfg.w_defense_support = 0
    no_guide._phi = no_guide._potentials()
    actions = np.zeros((1, 12), dtype=int)
    _, actual, done, info = env.step(actions)
    _, baseline, _, _ = no_guide.step(actions)
    assert done[0] and info["goal"][0] == -1
    np.testing.assert_allclose(actual - baseline, -0.35 * 0.5 * previous, atol=1e-7)


def test_optimized_and_reference_transitions_match():
    env, reference = env_with_reward(), env_with_reward(False)
    rng = np.random.default_rng(22)
    for _ in range(12):
        actions = rng.integers(0, 18, (1, 12))
        result, expected = env.step(actions), reference.step(actions)
        for actual, wanted in zip(result[:3], expected[:3]):
            np.testing.assert_array_equal(actual, wanted)
        for actual, wanted in zip(env._phi, reference._phi):
            np.testing.assert_array_equal(actual, wanted)


def test_only_jjrs_receives_override():
    import shutil
    trainer, directory = _trainer(1)
    try:
        for slot in trainer.slots:
            rc = slot.env.rcfg
            assert rc.w_defense_support == (0.35 if slot.task.name == "jjrs_6v6" else 0)
            assert rc.defense_shaping_floor == (0.5 if slot.task.name == "jjrs_6v6" else 0)
        assert trainer.rcfg.w_defense_support == 0
    finally:
        trainer.writer.close()
        shutil.rmtree(directory)
