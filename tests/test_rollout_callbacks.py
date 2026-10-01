"""Callbacks compilados frente a referencia: estados, eventos y transiciones."""
import numpy as np
import pytest

from env.tasks import load_catalog, make_env


def pair(task="rs4_4v4", n=12, delay=0):
    cfg = dict(n_envs=n, max_entities=13, seed=41, max_ticks=27,
               action_delay_max=delay, kickoff_timeout=12)
    fast = make_env(load_catalog()[task], **cfg)
    original = make_env(load_catalog()[task], **cfg)
    original.optimize_callbacks = False
    np.testing.assert_array_equal(fast.reset(), original.reset())
    return fast, original


def same_state(fast, original):
    for name, array in vars(fast).items():
        if isinstance(array, np.ndarray):
            np.testing.assert_allclose(array, getattr(original, name), atol=1e-13, rtol=1e-14,
                                       err_msg=name)
    for name in ("pos", "vel", "kickoff", "kicked", "touch"):
        np.testing.assert_allclose(getattr(fast.sim, name), getattr(original.sim, name),
                                   atol=1e-13, rtol=1e-14, err_msg=name)
    assert fast.rng.bit_generator.state == original.rng.bit_generator.state
    assert fast.sim.rng.bit_generator.state == original.sim.rng.bit_generator.state


def same_rewards_events(actual, expected):
    np.testing.assert_allclose(actual[0], expected[0], atol=1e-16, rtol=1e-14)
    for event in actual[1]:
        np.testing.assert_array_equal(actual[1][event], expected[1][event])


def test_pieces_mixed_release_kick_timeout_goal_and_coincidence():
    fast, original = pair()
    rng = np.random.default_rng(53)
    players = rng.normal(0, 100, fast.sim.player_pos.shape)
    velocity = rng.normal(0, 2, fast.sim.player_vel.shape)
    origin = rng.normal(0, 300, (fast.N, 2))
    owners = np.arange(fast.N) % 3 - 1
    kind = np.arange(fast.N) % 3 + 1
    goal = np.zeros(fast.N, dtype=np.int64)
    goal[4:6] = [1, -1]
    actions = rng.integers(0, 18, (fast.N, fast.P))
    saved_actions = actions.copy()
    for e in (fast, original):
        e.sim.player_pos[:] = players
        e.sim.player_vel[:] = velocity
        e.setpiece_pos[:] = origin
        e.setpiece_team[:] = owners
        e.setpiece_kind[:] = kind
        e.setpiece_ticks[:] = 3
        e.setpiece_limit[:] = 4
        e.sim.player_pos[1, e.T] = origin[1]  # rival exactamente sobre la pelota
        e.setpiece_kind[2] = 3
        e.setpiece_pos[2] = [e.field_w, 0]
        e.sim.player_pos[2, 0] = [e.field_w + 100, 0]  # invasión del área rival
        e.sim.kicked[:] = False
        e.sim.kicked[8, -1] = True
        e.goal_kick_speed = 12
        e.sim.ball_vel[:] = [1, 2]
    actual = fast._setpiece_pre_tick(actions)
    expected = original._setpiece_pre_tick(actions)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actions, saved_actions)
    same_state(fast, original)
    for a, b in zip(fast._setpiece_post_tick(goal), original._setpiece_post_tick(goal)):
        np.testing.assert_array_equal(a, b)
    same_state(fast, original)


@pytest.mark.parametrize("task", ["x1_1v1", "futsal_3v3", "futsal_7v7"])
def test_cooperation_random_contacts_and_pending_state(task):
    fast, original = pair(task)
    rng = np.random.default_rng(75)
    recorded_passes = 0
    for tick in range(50):
        positions = rng.normal(0, 250, fast.sim.player_pos.shape)
        ball = rng.normal(0, 250, (fast.N, 2))
        touch = rng.random((fast.N, fast.P)) < .12  # incluye 0/1/varios contactos
        for e in (fast, original):
            e.sim.player_pos[:] = positions
            e.sim.kickoff[:] = np.arange(e.N) % 4 == tick % 4
            e.setpiece_team[:] = np.where(np.arange(e.N) % 5 == tick % 5, 0, -1)
        same_rewards_events(fast._cooperation_touch_reward(touch, ball),
                            original._cooperation_touch_reward(touch, ball))
        same_state(fast, original)
        actual, expected = fast._advance_pending_passes(6), original._advance_pending_passes(6)
        same_rewards_events(actual, expected)
        recorded_passes += actual[1]["passes"].sum()
        same_state(fast, original)
    if fast.T > 1:
        assert recorded_passes > 0


def test_accumulation_preserves_rewards_events_and_public_output_ownership():
    fast, original = pair("futsal_3v3", n=1)
    for e in (fast, original):
        e.sim.kickoff[:] = False
        e.setpiece_team[:] = -1
        e.sim.player_pos[0] = [[0, 0], [100, 0], [200, 0], [0, 250], [100, 250], [200, 250]]
        e.pending_pass_team[0] = 0
        e.pending_pass_sender[0], e.pending_pass_receiver[0] = 0, 1
        e.pending_pass_progress[0] = .5
    bonus = np.full((1, fast.P), .125)
    events = fast._empty_cooperation_events()
    for event in events.values():
        event[:] = 3
    reward, reference_events = original._advance_pending_passes(12)
    result = fast._advance_pending_passes(12, (bonus, events))
    assert result[0] is bonus and result[1] is events
    np.testing.assert_array_equal(bonus, .125 + reward)
    for name in events:
        np.testing.assert_array_equal(events[name], 3 + reference_events[name])
    same_state(fast, original)
    # Otra llamada pública entrega buffers independientes, no un scratch mutable.
    saved = bonus.copy()
    empty, other = fast._advance_pending_passes(1)
    assert not np.shares_memory(empty, bonus)
    assert not np.shares_memory(other["passes"], events["passes"])
    np.testing.assert_array_equal(bonus, saved)


@pytest.mark.parametrize("task", ["rs4_4v4", "futsal_3v3", "futsal_7v7", "jjrs_6v6"])
@pytest.mark.parametrize("delay", [0, 4])
def test_complete_transitions_include_outs_resets_and_delays(task, delay):
    fast, original = pair(task, n=5, delay=delay)
    rng = np.random.default_rng(99)
    for tick in range(55):
        if tick % 11 == 0:
            for e in (fast, original):
                e.sim.kickoff[0] = False
                e.sim.ball_pos[0] = [e.goal_x - 1, 0]
                e.sim.ball_vel[0] = [4, 0]
                if e.out_of_bounds:
                    e.sim.ball_pos[1] = [0, e.field_h + 30]
                    e.sim.ball_vel[1] = 0
                e._phi = e._potentials()
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
        same_state(fast, original)
