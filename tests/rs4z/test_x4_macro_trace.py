"""Macro provenance survives latency/cancellation; coarse events stay honest."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from env.rs4z import contract as C
from env.rs4z.core import MIRROR_ACTION, RS4ZEnv
from learn.x4_macro_trace import MacroTrace
from learn.x4_pass_macro import MacroState


def _fake_env(delay=0, frame_skip=3, n=1):
    return SimpleNamespace(
        N=n, active=np.ones((n, 8), bool), team=np.repeat([0, 1], 4),
        act_hist=np.zeros((n, 8, 6), np.int64), episode=np.ones(n, np.int64),
        delay=np.full((n, 8), delay, np.int64), frame_skip=frame_skip,
        kick_cancel=np.zeros((n, 8), bool), _s_kick=np.zeros((n, 8), bool),
    )


def _synthetic_step(env, trace, macro, *, actions=None, kicked=(), touched=(), final=(), **overrides):
    if actions is None:
        actions = np.zeros((env.N, 8), np.int64)
    trace.before_step(env, macro, actions)
    world = actions.copy()
    world[:, 4:] = MIRROR_ACTION[world[:, 4:]]
    world[~env.active] = 0
    env.act_hist[:, :, 1:] = env.act_hist[:, :, :-1]
    env.act_hist[:, :, 0] = world
    env._s_kick.fill(False)
    env._s_kick[0, list(final)] = True
    events = dict(ticks=np.full(env.N, env.frame_skip), kicked=np.zeros((env.N, 8), bool),
                  touched=np.zeros((env.N, 8), bool), goal=np.zeros(env.N, int),
                  out=np.zeros(env.N, bool), restart_start=np.zeros(env.N, int),
                  match_end=np.zeros(env.N, bool), safety=np.zeros(env.N, int))
    events["kicked"][0, list(kicked)] = True
    events["touched"][0, list(touched)] = True
    events.update(overrides)
    result = trace.after_step(env, events)
    macro.after(events["kicked"])
    return result


def _launch(env, trace, macro, sender=0, receiver_index=0):
    macro.k[0, sender] = receiver_index
    actions = np.zeros((env.N, 8), np.int64)
    actions[0, sender] = 9
    return _synthetic_step(env, trace, macro, actions=actions, kicked=(sender,))


def test_delayed_macro_kick_still_counted_after_controller_cancelled():
    delays = np.array([4, 5, 8, 10, 11])
    env = RS4ZEnv(len(delays), map="sanguchito_rs_x4", frame_skip=3, max_delay=15,
                  deadline=0, kickoff_deadline=C.KICKOFF_SAFETY, seed=0)
    env.start_match(np.arange(env.N), delay=np.repeat(delays[:, None], 8, 1), match_ticks=3600)
    positions = np.array([[-25.0, 0], [250, 0], [-200, 200], [-200, -200],
                          [500, 250], [500, -250], [800, 200], [800, -200]])
    for row in range(env.N):
        env.place(row, ball_pos=[0, 0], ball_vel=[0, 0],
                  player_pos=positions, player_vel=np.zeros((8, 2)))
    trace = MacroTrace(env)
    macro = MacroState(env.N)
    macro.k[:, 0] = 0
    actual = np.zeros(env.N, int)
    counted = np.zeros(env.N, int)
    for step in range(7):
        actions = np.zeros((env.N, 8), np.int64)
        if step == 0:
            actions[:, 0] = 9
        trace.before_step(env, macro, actions)
        events = env.step(actions)
        delta = trace.after_step(env, events)
        actual += events["kicked"][:, 0]
        counted += delta["macro_kicks"][:, 0]
        macro.after(events["kicked"])
        macro.k[:, 0] = -1  # unavailable/cancelled before delayed commands apply
    assert np.array_equal(actual, np.ones(env.N))
    assert np.array_equal(counted, actual)
    stats = trace.stats()
    assert stats["macro_starts"] == env.N
    assert stats["macro_kick_emitted"] == env.N
    assert stats["macro_kicks"] == env.N
    assert stats["macro_kicks_ambiguous"] == 0
    assert stats["macro_applied_action_ticks"] == env.N * 3
    json.dumps(stats)


@pytest.mark.parametrize("frame_skip,delay", [(1, 4), (3, 10), (4, 7), (5, 8)])
def test_origin_tick_index_matches_kernel_with_other_frame_skips(frame_skip, delay):
    env = _fake_env(delay=delay, frame_skip=frame_skip)
    trace = MacroTrace(env)
    macro = MacroState(1)
    macro.k[0, 0] = 0
    for step in range((delay + frame_skip - 1) // frame_skip + 2):
        actions = np.zeros((1, 8), np.int64)
        if step == 0:
            actions[0, 0] = 9
        launch_now = step * frame_skip <= delay < (step + 1) * frame_skip
        _synthetic_step(env, trace, macro, actions=actions, kicked=(0,) if launch_now else ())
        macro.k[0, 0] = -1
    assert trace.stats()["macro_kicks"] == 1
    assert trace.stats()["macro_applied_action_ticks"] == frame_skip


def test_mixed_origin_coarse_kick_is_ambiguous_until_last_tick_is_known():
    for final, expected, ambiguous in [((), 0, 1), ((0,), 1, 0)]:
        env = _fake_env(delay=1)
        trace = MacroTrace(env)
        macro = MacroState(1)
        raw = np.zeros((1, 8), np.int64)
        raw[0, 0] = 9
        _synthetic_step(env, trace, macro, actions=raw)
        macro.k[0, 0] = 0
        delta = _synthetic_step(env, trace, macro, actions=raw, kicked=(0,), final=final)
        # Tick 0 uses raw old action; ticks 1, 2 use the new macro command.
        assert delta["macro_kicks"][0, 0] == expected
        assert delta["macro_kicks_ambiguous"][0, 0] == ambiguous
        assert trace.stats()["macro_starts"] == 1


def test_two_macro_targets_in_same_decision_are_not_conflated():
    env = _fake_env(delay=1)
    trace = MacroTrace(env)
    macro = MacroState(1)
    macro.k[0, 0] = 0
    actions = np.zeros((1, 8), np.int64)
    actions[0, 0] = 9
    _synthetic_step(env, trace, macro, actions=actions)
    macro.k[0, 0] = 1
    macro.steps[0, 0] = 0
    _synthetic_step(env, trace, macro, actions=actions, kicked=(0,))
    assert trace.stats()["macro_starts"] == 2
    assert trace.stats()["macro_kicks"] == 0
    assert trace.stats()["macro_kicks_ambiguous"] == 1


def test_each_delayed_kick_from_same_attempt_can_launch_an_outcome():
    env = _fake_env(delay=10)
    trace = MacroTrace(env)
    macro = MacroState(1)
    macro.k[0, 0] = 0
    kick = np.zeros((1, 8), np.int64)
    kick[0, 0] = 9
    # Kick, release, kick are emitted before the first kick traverses latency.
    _synthetic_step(env, trace, macro, actions=kick)
    _synthetic_step(env, trace, macro)
    _synthetic_step(env, trace, macro, actions=kick)
    _synthetic_step(env, trace, macro, kicked=(0,))
    _synthetic_step(env, trace, macro)
    _synthetic_step(env, trace, macro, kicked=(0,))
    _synthetic_step(env, trace, macro, touched=(1,))
    assert trace.stats()["macro_starts"] == 1
    assert trace.stats()["macro_kicks"] == 2
    assert trace.stats()["macro_receives"] == 1


def test_ambiguous_macro_kick_is_not_later_reported_as_never_kicked():
    env = _fake_env(delay=1)
    trace = MacroTrace(env)
    macro = MacroState(1)
    kick = np.zeros((1, 8), np.int64)
    kick[0, 0] = 9
    _synthetic_step(env, trace, macro, actions=kick)
    macro.k[0, 0] = 0
    _synthetic_step(env, trace, macro, actions=kick, kicked=(0,))
    for _ in range(env.act_hist.shape[2] + 1):
        _synthetic_step(env, trace, macro)
    assert trace.stats()["macro_kicks_ambiguous"] == 1
    assert trace.stats()["macro_cancelled_before_kick"] == 0


def test_receive_selected_mate_then_observed_retention():
    env = _fake_env()
    trace = MacroTrace(env, retention_ticks=6)
    macro = MacroState(1)
    _launch(env, trace, macro)
    delta = _synthetic_step(env, trace, macro, touched=(1,))
    assert delta["macro_receives"][0, 0] == 1
    _synthetic_step(env, trace, macro)
    delta = _synthetic_step(env, trace, macro)
    assert delta["macro_retained"][0, 0] == 1
    _synthetic_step(env, trace, macro)
    assert trace.stats()["macro_retained"] == 1


@pytest.mark.parametrize("contact,counter", [((4,), "macro_interceptions"),
                                             ((2,), "macro_other_teammate"),
                                             ((0,), "macro_sender_retouch"),
                                             ((1, 4), "macro_outcome_ambiguous"),
                                             ((1, 2), "macro_outcome_ambiguous")])
def test_first_contact_outcomes_are_distinguished(contact, counter):
    env = _fake_env()
    trace = MacroTrace(env)
    macro = MacroState(1)
    _launch(env, trace, macro)
    _synthetic_step(env, trace, macro, touched=contact)
    assert trace.stats()[counter] == 1
    assert trace.stats()["macro_receives"] == 0


def test_contact_in_launch_decision_does_not_invent_event_order():
    env = _fake_env()
    trace = MacroTrace(env)
    macro = MacroState(1)
    macro.k[0, 0] = 0
    actions = np.zeros((1, 8), np.int64)
    actions[0, 0] = 9
    _synthetic_step(env, trace, macro, actions=actions, kicked=(0,), touched=(1,))
    assert trace.stats()["macro_kicks"] == 1
    assert trace.stats()["macro_outcome_ambiguous"] == 1
    assert trace.stats()["macro_receives"] == 0


def test_out_and_rival_contact_after_receiving_interrupt_retention():
    for kwargs in [dict(out=np.array([True])), dict(touched=(4,))]:
        env = _fake_env()
        trace = MacroTrace(env, retention_ticks=6)
        macro = MacroState(1)
        _launch(env, trace, macro)
        _synthetic_step(env, trace, macro, touched=(1,))
        _synthetic_step(env, trace, macro, **kwargs)
        assert trace.stats()["macro_retention_lost"] == 1
        assert trace.stats()["macro_retained"] == 0


def test_out_and_timeout_before_receiving():
    for kind in ("out", "timeout"):
        env = _fake_env()
        trace = MacroTrace(env, max_flight_ticks=6)
        macro = MacroState(1)
        _launch(env, trace, macro)
        _synthetic_step(env, trace, macro)
        _synthetic_step(env, trace, macro, **({"out": np.array([True])} if kind == "out" else {}))
        assert trace.stats()["macro_" + kind] == 1


def test_external_episode_reset_forgets_delayed_macro_provenance():
    env = _fake_env(delay=4)
    trace = MacroTrace(env)
    macro = MacroState(1)
    macro.k[0, 0] = 0
    actions = np.zeros((1, 8), np.int64)
    actions[0, 0] = 9
    _synthetic_step(env, trace, macro, actions=actions)
    # Reset with retained queued input, as kickoff resets can do. The kick is
    # physically possible but may not count as success in the replaced episode.
    trace.reset([0])
    macro.k[0, 0] = -1
    _synthetic_step(env, trace, macro, kicked=(0,))
    assert trace.stats()["macro_kicks"] == 0
    assert trace.stats()["untracked_kicks"] == 1
    # Full new-match episode increments are automatically detected as well.
    env.episode += 1
    env.act_hist.fill(0)
    _synthetic_step(env, trace, macro)
    assert not (trace.origin_hist > 0).any()


def test_goal_reset_does_not_create_a_receive_from_next_kickoff():
    env = _fake_env()
    trace = MacroTrace(env)
    macro = MacroState(1)
    _launch(env, trace, macro)
    _synthetic_step(env, trace, macro, goal=np.array([1]))
    _synthetic_step(env, trace, macro, touched=(1,))
    assert trace.stats()["macro_reset_pending"] == 1
    assert trace.stats()["macro_receives"] == 0


def test_reset_stats_preserves_queued_origin_and_masked_stats():
    env = _fake_env(delay=4)
    trace = MacroTrace(env)
    macro = MacroState(1)
    macro.k[0, 0] = 0
    actions = np.zeros((1, 8), np.int64)
    actions[0, 0] = 9
    _synthetic_step(env, trace, macro, actions=actions)
    trace.reset_stats()
    macro.k[0, 0] = -1
    _synthetic_step(env, trace, macro, kicked=(0,))
    assert trace.stats()["macro_starts"] == 0
    assert trace.stats()["macro_kicks"] == 1
    own = np.zeros((1, 8), bool)
    own[0, 0] = True
    assert trace.stats(own)["macro_kicks"] == 1
    assert trace.stats(~own)["macro_kicks"] == 0


def test_rejected_sampled_macro_does_not_count_as_start_or_emission():
    env = _fake_env()
    trace = MacroTrace(env)
    macro = MacroState(1)
    sampled = np.full((1, 8), 18)
    table = np.full((1, 8, 3), -1)
    primitive, _ = macro.apply(sampled, table)
    trace.before_step(env, macro, primitive, sampled=sampled)
    assert trace.stats()["macro_starts"] == 0
    assert trace.stats()["macro_kick_emitted"] == 0


def test_trace_detects_unpaired_or_unobserved_steps():
    env = _fake_env()
    trace = MacroTrace(env)
    macro = MacroState(1)
    actions = np.zeros((1, 8), np.int64)
    with pytest.raises(RuntimeError, match="before_step"):
        trace.after_step(env, {})
    trace.before_step(env, macro, actions)
    with pytest.raises(RuntimeError, match="after_step"):
        trace.before_step(env, macro, actions)
    with pytest.raises(RuntimeError, match="outside"):
        trace.reset([0])
