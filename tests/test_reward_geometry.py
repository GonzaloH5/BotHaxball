"""Geometría acelerada: distancias, rewards y transiciones de la ruta de referencia."""
import numpy as np
import pytest

from env.tasks import load_catalog, make_env


@pytest.mark.parametrize("size", [1, 2, 3, 4, 5, 6, 7, 11])
@pytest.mark.parametrize("n", [1, 5, 48])
def test_geometry_matches_numpy(size, n):
    from env.haxball_env import HaxballEnv
    env = HaxballEnv(n, size, "big", seed=81)
    env.reset()
    rng = np.random.default_rng(44)
    env.sim.player_pos[:] = rng.normal(0, 150, env.sim.player_pos.shape)
    env.sim.ball_pos[:] = rng.normal(0, 100, env.sim.ball_pos.shape)
    env.sim.player_pos[0, 0] = env.sim.ball_pos[0]  # distancia cero
    if size > 1:
        env.sim.player_pos[0, 1] = env.sim.player_pos[0, 0]  # compañeros superpuestos
    fast_dist, fast_spread = env._team_ball_dist(), env._spread_potential()
    env.optimize_reward_geometry = False
    np.testing.assert_array_equal(fast_dist, env._team_ball_dist())
    # Reducciones pueden diferir en float64 en equipos >=8, sin fastmath.
    np.testing.assert_allclose(fast_spread, env._spread_potential(), atol=2e-16, rtol=2e-16)


@pytest.mark.parametrize("task", ["big_3v3", "futsal_af_3v3", "futsal_5v5",
                                  "futsal_7v7", "rs4_4v4", "jjrs_6v6"])
@pytest.mark.parametrize("shaping", [0.0, 0.7])
def test_geometry_preserves_full_transitions_and_rng(task, shaping):
    cfg = dict(n_envs=5, max_entities=13, seed=11, max_ticks=27, kickoff_timeout=12)
    fast = make_env(load_catalog()[task], **cfg)
    original = make_env(load_catalog()[task], **cfg)
    original.optimize_reward_geometry = False
    fast.rcfg.shaping_coef = original.rcfg.shaping_coef = shaping
    np.testing.assert_array_equal(fast.reset(), original.reset())
    rng = np.random.default_rng(65)
    for step in range(30):
        if step % 9 == 0:
            for env in (fast, original):
                env.sim.kickoff[0] = False
                env.sim.ball_pos[0] = (env.goal_x - 1, 0)
                env.sim.ball_vel[0] = (4, 0)
                env._phi = env._potentials()
        actions = rng.integers(0, 18, (fast.N, fast.P))
        actual, expected = fast.step(actions), original.step(actions)
        for a, b in zip(actual[:3], expected[:3]):
            np.testing.assert_array_equal(a, b)
        for key in actual[3]:
            if isinstance(actual[3][key], dict):
                for event in actual[3][key]:
                    np.testing.assert_array_equal(actual[3][key][event], expected[3][key][event])
            else:
                np.testing.assert_array_equal(actual[3][key], expected[3][key])
        assert fast.rng.bit_generator.state == original.rng.bit_generator.state
        assert fast.sim.rng.bit_generator.state == original.sim.rng.bit_generator.state
