"""Optimizaciones RS4 sin cambios de rewards, terminales, física ni RNG."""
import copy

import numpy as np
import pytest

from env.rewards import RewardConfig
from env.rs4_tactics import _formation_score, _formation_v2_score, dynamic_targets
from env.tasks import load_catalog, make_env


def pair(n=24):
    reward = RewardConfig(shaping_coef=0, rs4_tactical_coef=.08,
                          rs4_restart_approach=.05, rs4_restart_stall=.25,
                          rs4_defensive_out_scale=.2)
    fast = make_env(load_catalog()["rs4_4v4"], n, 7, reward, seed=73,
                    max_ticks=39, kickoff_timeout=12, action_delay_max=4)
    fast.rs4_formation_version = 2
    slow = copy.deepcopy(fast)
    slow.optimize_rs4 = False
    np.testing.assert_array_equal(fast.reset(), slow.reset())
    return fast, slow


def assert_same_state(fast, slow):
    for name, value in vars(fast).items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(value, getattr(slow, name), err_msg=name)
    for name, value in vars(fast.sim).items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(value, getattr(slow.sim, name), err_msg=name)
    assert fast.rng.bit_generator.state == slow.rng.bit_generator.state
    assert fast.sim.rng.bit_generator.state == slow.sim.rng.bit_generator.state


@pytest.mark.parametrize("snapshot", [False, True])
def test_contacts_match_numpy_for_mixed_flags_and_repeated_touches(snapshot):
    fast, slow = pair()
    rng = np.random.default_rng(401)
    for tick in range(40):
        players = rng.uniform(-1000, 1000, fast.sim.player_pos.shape)
        ball = rng.uniform(-1000, 1000, fast.sim.ball_pos.shape)
        touches = rng.random((fast.N, fast.P)) < .12
        kickoff = rng.random(fast.N) < .1
        owner = rng.integers(-1, 2, fast.N)
        open_before = rng.random(fast.N) < .8
        for env in (fast, slow):
            env.sim.player_pos[:] = players
            env.sim.ball_pos[:] = ball
            env.sim.kickoff[:] = kickoff
            env.setpiece_team[:] = owner
            env.ticks[:] = tick * 3
            env._out_pressure_players = players.copy() if snapshot else None
            env._out_pressure_open = open_before if snapshot else None
            env._record_defensive_out_touch(touches, ball)
        assert_same_state(fast, slow)


@pytest.mark.parametrize("team", [0, 1])
def test_defensive_pressure_threshold_and_zone_are_unchanged(team):
    fast, slow = pair(1)
    sign = 1 if team == 0 else -1
    own = np.flatnonzero(fast.sim.player_team == team)[0]
    rival = np.flatnonzero(fast.sim.player_team != team)[0]
    touches = np.zeros((1, fast.P), dtype=bool)
    touches[0, own] = True
    threshold = .12 * fast.goal_x
    for own_x in [-.7 * fast.goal_x, -.55 * fast.goal_x,
                  np.nextafter(-.55 * fast.goal_x, -np.inf)]:
        for pressure in [threshold, np.nextafter(threshold, -np.inf),
                         np.nextafter(threshold, np.inf)]:
            for env in (fast, slow):
                env.sim.kickoff[:] = False
                env.setpiece_team[:] = -1
                env.sim.ball_pos[:] = [sign * own_x, 0]
                env.sim.player_pos[:] = [0, 10000]
                env.sim.player_pos[:, rival] = env.sim.ball_pos + [0, pressure]
                env._record_defensive_out_touch(touches)
            assert_same_state(fast, slow)


def test_restart_potential_matches_numpy_and_outputs_are_independent():
    fast, slow = pair()
    rng = np.random.default_rng(36)
    for _ in range(20):
        players = rng.uniform(-3000, 3000, fast.sim.player_pos.shape)
        origin = rng.uniform(-2000, 2000, fast.setpiece_pos.shape)
        owners = rng.integers(-1, 2, fast.N)
        for env in (fast, slow):
            env.sim.player_pos[:] = players
            env.setpiece_pos[:] = origin
            env.setpiece_team[:] = owners
        actual = fast._rs4_restart_potential()
        np.testing.assert_array_equal(actual, slow._rs4_restart_potential())
        saved = actual.copy()
        other = fast._rs4_restart_potential()
        assert not np.shares_memory(other, actual)
        other[:] = 99
        np.testing.assert_array_equal(actual, saved)
    fast.rcfg.rs4_restart_approach = 0
    assert not fast._rs4_restart_potential().any()


@pytest.mark.parametrize("version", [1, 2])
def test_tactical_subsets_and_protected_rows_match_full_reference(version, monkeypatch):
    from env import rs4_tactics
    fast, slow = pair(12)
    rng = np.random.default_rng(83)
    for env in (fast, slow):
        env.rs4_formation_version = version
        env.sim.kickoff[:] = np.arange(env.N) % 3 == 0
        env.setpiece_team[:] = np.where(np.arange(env.N) % 3 == 1, 0, -1)
    players = rng.uniform(-1000, 1000, fast.sim.player_pos.shape)
    balls = rng.uniform(-1000, 1000, fast.sim.ball_pos.shape)
    for env in (fast, slow):
        env.sim.player_pos[:] = players
        env.sim.ball_pos[:] = balls
    reference = slow._rs4_potential()
    calls = []
    original = rs4_tactics.components
    def measured(players, *args):
        calls.append(len(players))
        return original(players, *args)
    monkeypatch.setattr(rs4_tactics, "components", measured)
    np.testing.assert_array_equal(fast._rs4_potential(), reference)
    assert calls == [4]  # no procesar los 8 partidos protegidos
    for indices in ([11, 0, 5, 1], [0, 1, 3, 4], []):
        calls.clear()
        actual = fast._rs4_potential(indices)
        np.testing.assert_array_equal(actual, reference[np.asarray(indices, dtype=int)])
        expected = sum(i % 3 == 2 for i in indices)
        assert calls == ([expected] if expected else [])


def test_v2_reused_role_costs_match_two_complete_assignments():
    rng = np.random.default_rng(15)
    for _ in range(100):
        ball = rng.uniform(-2000, 2000, 2)
        own = rng.uniform(-2000, 2000, (4, 2))
        control = rng.random()
        left = dynamic_targets(ball, control, 1000., 600., 120., -1.)
        right = dynamic_targets(ball, control, 1000., 600., 120., 1.)
        expected = max(_formation_score(own, left, 1000., 2),
                       _formation_score(own, right, 1000., 2))
        assert _formation_v2_score(own, left, right, 1000.) == expected


def test_reset_refresh_only_computes_changed_rows(monkeypatch):
    fast, slow = pair(12)
    for env in (fast, slow):
        env.sim.kickoff[:] = False
        env.setpiece_team[:] = -1
        env.sim.vel[:] = 0
        env.sim.mask[:] = env.sim.base_mask
        env.ticks[3] = env.max_ticks
        env.match_ticks[3] = env.max_ticks
        env._phi = env._potentials()
        env._rs4_phi = env._rs4_potential()
    original = fast._rs4_potential
    calls = []
    def measured(indices=None):
        calls.append(None if indices is None else indices.copy())
        return original(indices)
    monkeypatch.setattr(fast, "_rs4_potential", measured)
    actual, expected = fast.step(np.zeros((fast.N, fast.P), int)), slow.step(np.zeros((slow.N, slow.P), int))
    for a, b in zip(actual[:3], expected[:3]):
        np.testing.assert_array_equal(a, b)
    assert len(calls) == 2 and calls[0] is None
    np.testing.assert_array_equal(calls[1], [3])
    assert_same_state(fast, slow)


@pytest.mark.parametrize("callbacks", [True, False])
def test_full_transitions_preserve_guide_decay_resets_events_and_rng(callbacks):
    fast, slow = pair(8)
    fast.optimize_callbacks = slow.optimize_callbacks = callbacks
    rng = np.random.default_rng(94)
    for step in range(90):
        if step % 11 == 0:
            for env in (fast, slow):
                env.sim.kickoff[0] = False
                env.sim.ball_pos[0] = [env.goal_x - 1, 0]
                env.sim.ball_vel[0] = [4, 0]
                env.sim.kickoff[1] = False
                env.sim.ball_pos[1] = [10, env.field_h + 30]
                env.sim.ball_vel[1] = 0
                env._phi = env._potentials()
                env._rs4_phi = env._rs4_potential()
        if step % 13 == 0:
            for env in (fast, slow):
                env.setpiece_team[2] = step % 2
                env.setpiece_kind[2] = 2
                env.setpiece_pos[2] = [env.field_w, env.field_h]
                env.setpiece_ticks[2] = env.setpiece_limit[2] - 1
        if step in (20, 40, 60):
            coefficient = {20: .05, 40: .02, 60: 0}[step]
            fast.rcfg.rs4_tactical_coef = slow.rcfg.rs4_tactical_coef = coefficient
        actions = rng.integers(0, 18, (fast.N, fast.P))
        actual, expected = fast.step(actions), slow.step(actions)
        for a, b in zip(actual[:3], expected[:3]):
            np.testing.assert_array_equal(a, b)
        for name, value in actual[3].items():
            if isinstance(value, dict):
                for event, array in value.items():
                    np.testing.assert_array_equal(array, expected[3][name][event], err_msg=event)
            else:
                np.testing.assert_array_equal(value, expected[3][name], err_msg=name)
        assert_same_state(fast, slow)


def test_contact_snapshot_buffer_reused_without_aliasing_physics():
    fast, _ = pair(2)
    actions = np.zeros((2, 8), int)
    fast.step(actions)
    snapshot = fast._out_pressure_players
    assert not np.shares_memory(snapshot, fast.sim.player_pos)
    fast.step(actions)
    assert fast._out_pressure_players is snapshot
