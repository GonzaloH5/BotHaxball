"""No basta con mantener barreras: el rival debe ejecutar saques reales."""
import numpy as np
import pytest
from numba import get_num_threads, set_num_threads

from bots.scripted import scripted_actions
from env.tasks import load_catalog, make_env
from eval.agents import env_kwargs
from eval.render import record


@pytest.fixture(autouse=True)
def threads():
    previous = get_num_threads()
    set_num_threads(min(2, previous))
    yield
    set_num_threads(previous)


def scenario(task, team, kind, side=1, optimized=True, count=1):
    env = make_env(load_catalog()[task], count, 7, seed=0, random_reset_prob=0,
                   optimize_rollout=optimized)
    env.reset()
    sim = env.sim
    sim.kickoff[:] = False
    sim.mask[:] = sim.base_mask
    sim.vel[:] = 0
    for p in range(env.P):
        sim.player_pos[:, p] = (-150 + 300 * sim.player_team[p], 50 * p - 125)
    sx = -1 if team == 0 else 1
    origin = np.array([-sx * env.field_w * .8, side * (env.field_h + 20)])
    if kind == 6:
        env._reset_envs(np.array([0]), kickoff_team=np.array([team]))
    elif env.rules is not None:
        method = {1: env.rules._throw_in, 2: env.rules._corner, 3: env.rules._goal_kick}[kind]
        if kind != 1:
            origin = np.array([(-sx if kind == 2 else sx) * (env.field_w + 20), side * 200.])
        method(0, team, origin)
    else:
        sim.pos[0, 0] = origin if kind == 1 else ((-sx if kind == 2 else sx) * (env.field_w + 20), side * 200.)
        env.last_touch[0] = 1 - team
        env._set_piece([0])
    env._phi = env._potentials()
    return env


@pytest.mark.parametrize("task", ["rs4_3v3", "rs4_4v4", "rs_3v3"])
@pytest.mark.parametrize("team", [0, 1])
@pytest.mark.parametrize("kind", [1, 2, 3, 6])
@pytest.mark.parametrize("side", [-1, 1])
def test_scripted_executes_restart_before_expiry(task, team, kind, side):
    env = scenario(task, team, kind, side)
    own = env.sim.player_team == team
    if kind == 6:
        limit = int(env.kickoff_limit[0])
    elif env.rules is not None:
        limit = 420
    else:
        limit = int(env.setpiece_limit[0])
    for _ in range((limit + env.frame_skip - 1) // env.frame_skip):
        actions = scripted_actions(env)
        _, _, done, info = env.step(actions)
        assert not info["stall"].any() and not done.any()
        if info["kicked"][0, own].any():
            # Validar la reanudación, no sólo que presionó X.
            if kind == 1:
                assert side * env.sim.ball_vel[0, 1] < 0
                if env.rules is not None:
                    for _ in range(20):
                        env.step(scripted_actions(env))
                        if env.rules.in_play[0]:
                            break
                    assert env.rules.in_play[0]
            if kind == 6:
                assert not env.sim.kickoff[0]
            return
    pytest.fail(f"{task}: equipo {team} no ejecutó saque {kind} antes de {limit} ticks")


@pytest.mark.parametrize("task", ["rs4_3v3", "rs_3v3"])
@pytest.mark.parametrize("kind", [1, 3, 6])
def test_restarts_match_reference_subset_and_noise(task, kind):
    fast = scenario(task, 0, kind, count=3)
    reference = scenario(task, 0, kind, optimized=False, count=3)
    players = np.array([2, 0, 4])
    rows = np.array([2, 0])
    a, b = np.random.default_rng(71), np.random.default_rng(71)
    for _ in range(5):
        np.testing.assert_array_equal(scripted_actions(fast, players, .5, a, rows),
                                      scripted_actions(reference, players, .5, b, rows))
        assert a.bit_generator.state == b.bit_generator.state
        actions = scripted_actions(fast)
        x, y = fast.step(actions), reference.step(actions)
        for i in range(3):
            np.testing.assert_array_equal(x[i], y[i])


def test_kickoff_deadline_allows_rs_travel_without_changing_plain_timeout():
    rs = scenario("rs4_3v3", 0, 6)
    assert rs.kickoff_limit[0] > rs.kickoff_timeout == 180
    plain = make_env(load_catalog()["big_3v3"], 1, 5, kickoff_timeout=12, random_reset_prob=0)
    plain.reset()
    assert plain.kickoff_limit[0] == 12
    assert env_kwargs({})["kickoff_timeout"] == 180


def test_record_reports_waiting_after_goal_and_resets_without_fake_goals():
    class GoalThenStay:
        fired = False
        def __call__(self, env, obs, players):
            if not self.fired:
                self.fired = True
                env.sim.kickoff[:] = False
                env.sim.pos[0, 0] = (env.goal_x - 1, 0)
                env.sim.vel[0, 0] = (4, 0)
            return np.zeros((env.N, len(players)), dtype=np.int64)

    class Stay:
        def __call__(self, env, obs, players):
            return np.zeros((env.N, len(players)), dtype=np.int64)

    meta, frames = record(GoalThenStay(), Stay(), minutes=.04,
                          env_kw={"kickoff_timeout": 12})
    assert meta["kickoff_stalls"] > 0
    assert frames[-1][-2:] == [1, 0]
