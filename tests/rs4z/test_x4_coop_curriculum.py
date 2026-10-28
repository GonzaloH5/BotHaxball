"""Cooperative exercises: real human release, delayed commands and terminal team credit."""
from types import SimpleNamespace
import copy
import json

import numpy as np
import pytest

from env.rs4z import kernel as K, obs_v3
from env.rs4z.core import RS4ZEnv
from learn.x4_coop_curriculum import CoopBank, CoopCurriculum, TEAM, pressure_position, wilson_lower
from learn.x4_data import move_of_input


def _data():
    T, release, receive = 240, 60, 90
    pos = np.repeat(np.array([[-23., 0], [203., 0], [-500, 300], [-900, 0],
                              [100, 200], [500, -300], [800, 300], [1000, 0]])[None], T, axis=0)
    ball = np.zeros((T, 4))
    ball[release:receive + 1, 0] = np.arange(receive - release + 1) * 6.0
    ball[release:receive, 2] = 6.0
    ball[receive:, 0] = 180.0
    kick_ev = np.zeros((T, 8), bool)
    kick_ev[release, 0] = True
    inp = np.zeros((T, 8), np.int64)
    inp[release, 0] = 16
    return SimpleNamespace(state=np.ones(T, np.int64), rkind=np.zeros(T, np.int64),
                           valid=np.arange(T), map_idx=np.full(T, 1), ball=ball,
                           ball_r=np.full(T, 8.325), pos=pos, vel=np.zeros_like(pos),
                           inp=inp, kicking=np.ones((T, 8), bool), kick_ev=kick_ev,
                           pid=np.broadcast_to(np.arange(8), (T, 8)).copy(), rec_of_tick=np.zeros(T, np.int64),
                           mass=np.full(T, 0.5), kstr=np.full(T, 5.85))


def _setup(stage=1, n=1, delay=14, **kwargs):
    bank = CoopBank(_data(), limit=10)
    env = RS4ZEnv(n, map="sanguchito_rs_x4", max_delay=15, frame_skip=3, seed=1)
    arena = SimpleNamespace(env=env, learner=np.ones((n, 8), bool), opp=np.full(n, -1))
    cur = CoopCurriculum(bank, n, np.random.default_rng(4), frac=0.5, **kwargs)
    cur.stage = stage
    cur._reserve_rows()
    # Single-row unit tests isolate the scorer. Production batches reserve at
    # most floor(N/2) rows; keep row zero among them for multirow checks.
    if not cur.exercise_rows[0]:
        selected = np.flatnonzero(cur.exercise_rows)
        if len(selected):
            cur.exercise_rows[selected[0]] = False
        cur.exercise_rows[0] = True
    for _ in range(30):
        env.start_match(np.arange(n), delay=np.full((n, 8), delay), match_ticks=10000)
        cur.reset(arena, np.arange(n))
        if cur.active[0]:
            break
    assert cur.active[0]
    return cur, arena


def _event(n=1, *, kicks=(), touches=(), goal=0, out=False, end=False):
    kicked, touched = np.zeros((n, 8), bool), np.zeros((n, 8), bool)
    kicked[0, list(kicks)] = True
    touched[0, list(touches)] = True
    goals = np.zeros(n, np.int64)
    goals[0] = goal
    return dict(kicked=kicked, touched=touched, goal=goals, out=np.full(n, out),
                restart_start=np.zeros(n, np.int64), match_end=np.full(n, end), ticks=np.full(n, 3))


def _kick(cur, env):
    cur.age[0] = 30
    cur.before_step(env)
    env.ball_pos[0] = [10, 0]
    env.ball_vel[0] = [6, 0]
    return cur.step(env, _event(kicks=(0,)))


def _receive(cur, env):
    cur.before_step(env)
    env.ball_pos[0] = [180, 0]
    env.ball_vel[0] = [0, 0]
    return cur.step(env, _event(touches=(1,)))


def test_bank_mines_effective_kicked_human_pass():
    bank = CoopBank(_data(), limit=10)
    assert bank.records.tolist() == [[60, 90, 0, 1, 0]]


@pytest.mark.parametrize("corruption", ["no_kick", "interception", "record_boundary", "player_change", "restart", "bounce"])
def test_bank_rejects_ambiguous_or_discontinuous_passes(corruption):
    data = _data()
    if corruption == "no_kick":
        data.kick_ev[:] = False
    elif corruption == "interception":
        data.pos[75, 4] = data.ball[75, :2]
    elif corruption == "record_boundary":
        data.rec_of_tick[75:] = 1
    elif corruption == "player_change":
        data.pid[75:, 2] = 88
    elif corruption == "restart":
        data.rkind[75] = 1
    else:
        data.ball[75, 2] = -6.0
    with pytest.raises(ValueError, match="no clean human passes"):
        CoopBank(data)


def test_reset_masks_defenders_and_reconstructs_world_frame_pending_inputs():
    cur, arena = _setup(stage=2, n=16)
    env, data = arena.env, cur.bank.d
    # The ball is still at its human starting location. Recover the reset tick
    # through unique raw input patterns without ever using imitation labels.
    data.inp[:, :] = np.arange(len(data.state))[:, None] % 32
    data.label = np.full_like(data.inp, 17)
    env.delay[:] = np.arange(8) + 7
    # Fix the backplay choice so its t is known and inspect the exact queue.
    row, t = 0, 35
    cur._source_actions(env, row, t, env.delay[row])
    for h in range(env.H):
        idx = t + env.delay[row] + 1 - 3 * (h + 1)
        raw = data.inp[idx, np.arange(8)]
        expected = move_of_input(raw) + 9 * ((raw & 16) != 0)
        expected[~env.active[row]] = 0
        assert np.array_equal(env.act_hist[row, :, h], expected)
    rows = np.flatnonzero(cur.active)
    assert np.all(arena.opp[rows] == -2)
    for row in rows:
        assert np.all(arena.learner[row] == (TEAM == cur.attack_team[row]))
        assert not (arena.learner[row] & env.active[row] & (TEAM != cur.attack_team[row])).any()
        assert env.active[row].sum() in (3, 5)
    obs = obs_v3.observe(env)
    assert np.all(obs[~env.active] == 0)
    assert cur.descriptor().shape == (16, 4)


def test_stationary_policy_cannot_complete_backplay_from_inherited_human_kick():
    cur, arena = _setup(stage=1, timeout_s=2.0)
    env = arena.env
    assert not (env.act_hist[0] >= 9).any(), "the human release must not already be pending"
    reward = np.zeros((1, 2))
    for _ in range(45):
        cur.before_step(env)
        ev = env.step(np.zeros((1, 8), np.int64))
        reward, done, stats = cur.step(env, ev)
        assert stats["policy_passes"] == 0
        if done[0]:
            break
    assert done[0] and reward[0].tolist() == [-1.0, 1.0]
    assert not cur.received[0]


def test_body_rebound_without_policy_kick_is_not_a_pass():
    cur, arena = _setup(stage=1)
    reward, done, stats = _receive(cur, arena.env)
    assert not cur.received[0] and stats["policy_passes"] == 0
    assert not done[0] and np.all(reward == 0)


def test_early_seeded_kick_cannot_start_a_policy_pass():
    cur, arena = _setup(stage=1)
    env = arena.env
    cur.before_step(env)
    env.ball_pos[0] = [10, 0]
    env.ball_vel[0] = [6, 0]
    cur.step(env, _event(kicks=(0,)))
    assert not cur.pending[0]
    _receive(cur, env)
    assert not cur.received[0]


def test_clean_pass_requires_full_physical_retention_and_pays_once():
    cur, arena = _setup(stage=1)
    env = arena.env
    _kick(cur, env)
    reward, done, stats = _receive(cur, env)
    assert cur.received[0] and stats["policy_passes"] == 1
    assert not done[0] and cur.retained[0] == 0
    for _ in range(39):
        reward, done, stats = cur.step(env, _event(touches=(1,)))
        assert not done[0] and np.all(reward == 0)
    reward, done, stats = cur.step(env, _event(touches=(1,)))
    assert done[0] and reward[0].tolist() == [1.0, -1.0]
    assert stats["successes"] == 1
    again, done, stats = cur.step(env, _event(touches=(1,)))
    assert done[0] and np.all(again == 0) and stats["episodes"] == 0


def test_ball_left_far_away_does_not_count_as_retained_possession():
    cur, arena = _setup(stage=1)
    env = arena.env
    _kick(cur, env)
    _receive(cur, env)
    env.ball_pos[0] = [700, 600]
    for _ in range(45):
        reward, done, stats = cur.step(env, _event())
    assert cur.retained[0] == 0 and not done[0] and not np.any(reward)


def test_reversed_flight_is_not_a_qualifying_pass():
    cur, arena = _setup(stage=1)
    env = arena.env
    _kick(cur, env)
    env.ball_pos[0], env.ball_vel[0] = [180, 0], [-6, 0]
    _, _, stats = cur.step(env, _event(touches=(1,)))
    assert not cur.received[0] and stats["policy_passes"] == 0


def test_interception_dominates_simultaneous_receive_and_shared_reward_is_zero_sum():
    cur, arena = _setup(stage=2)
    env = arena.env
    _kick(cur, env)
    enemy = int(np.flatnonzero(env.active[0] & (TEAM != cur.attack_team[0]))[0])
    env.ball_pos[0] = [180, 0]
    reward, done, stats = cur.step(env, _event(touches=(1, enemy)))
    assert done[0] and reward[0].tolist() == [-1.0, 1.0]
    assert stats["losses"] == 1 and stats["safe_receives"] == 0


def test_human_reception_is_reported_separately_and_goal_is_not_a_receive_shortcut():
    cur, arena = _setup(stage=0)
    env = arena.env
    _, done, stats = _receive(cur, env)
    assert stats["human_receptions"] == 1 and stats["policy_passes"] == 0 and not done[0]
    reward, done, stats = cur.step(env, _event(goal=1))
    assert done[0] and reward[0].tolist() == [-1.0, 1.0]


def test_finishing_stage_accepts_legitimate_goal_without_mandatory_pass():
    cur, arena = _setup(stage=3)
    env = arena.env
    env.ri[0, K.RI_KO] = 1  # the engine already replaced a goal with a kickoff
    reward, done, stats = cur.step(env, _event(goal=1))
    assert done[0] and stats["successes"] == 1
    assert reward[0].tolist() == [1.0, -1.0] and not cur.received[0]


def test_mastery_requires_sample_and_wilson_bound_and_persists_without_partial_states():
    cur, _ = _setup(stage=0, min_episodes=20, window=32, threshold=0.6)
    cur.window.extend([1] * 19)
    assert cur.advance() is None
    cur.window.append(1)
    event = cur.advance()
    assert event["from_stage"] == 0 and cur.stage == 1
    assert event["wilson_lower"] >= 0.6
    cur.window.extend([1] * 12 + [0] * 8)
    assert cur.advance() is None and wilson_lower(12, 20) < 0.6
    state = json.loads(json.dumps(cur.state_dict()))
    copy_cur, _ = _setup(stage=0, min_episodes=20, window=32, threshold=0.6)
    copy_cur.load_state_dict(copy.deepcopy(state))
    assert copy_cur.stage == 1 and list(copy_cur.window) == list(cur.window)
    assert not copy_cur.active.any(), "resume resets episodes rather than attaching trackers to fresh physics"
    cur.stage = 4
    assert cur.exercise_fraction <= 0.15


def test_nonexercise_rows_have_no_curriculum_reward_or_terminal():
    cur, arena = _setup(n=8)
    cur.active[1:] = False
    reward, done, stats = cur.step(arena.env, _event(n=8))
    assert np.all(reward[1:] == 0) and not done[1:].any()
    assert np.all(cur.descriptor()[1:] == 0)


def test_short_episode_resets_preserve_batch_fraction_and_transfer_frees_rows():
    bank = CoopBank(_data())
    env = RS4ZEnv(20, map="sanguchito_rs_x4", max_delay=15, seed=1)
    arena = SimpleNamespace(env=env, learner=np.ones((20, 8), bool), opp=np.full(20, -1))
    cur = CoopCurriculum(bank, 20, np.random.default_rng(2), frac=0.5)
    rows = np.arange(20)
    for _ in range(5):
        env.start_match(rows, delay=np.full((20, 8), 8))
        cur.reset(arena, rows)
        assert cur.active.sum() == 10
    original = cur.exercise_rows.copy()
    cur.stage = 4
    cur._reserve_rows()
    env.start_match(rows, delay=np.full((20, 8), 8))
    cur.reset(arena, rows)
    assert cur.active.sum() <= 3 and np.all(cur.exercise_rows <= original)


def test_pressure_defender_avoids_other_attackers_defenders_ball_and_field_edge():
    # The previous direct placement at 65 px from the carrier lands exactly on
    # attacker 1. Nearby attacker 2 and defender 5 also obstruct alternatives.
    pp = np.array([[-23., 0], [42, 0], [42, 30], [-900, 0],
                   [200, 0], [42, -30], [900, 300], [1000, 0]])
    active = np.array([True, True, True, False, True, True, False, False])
    point = pressure_position(pp, active, 4, 0, np.array([0., 0]), 8.325)
    assert not np.allclose(point, [42, 0]), "direct 65 px placement overlaps an attacker"
    others = active.copy()
    others[4] = False
    assert np.all(np.linalg.norm(pp[others] - point, axis=1) >= 30.5)
    assert np.linalg.norm(point - [0, 0]) >= 23.825
    assert 59.9 <= np.linalg.norm(point - pp[0]) <= 70.1
    assert abs(point[0]) <= 1135 and abs(point[1]) <= 655
    # A carrier near the top-right corner must still get an in-field point.
    corner = pp.copy()
    corner[0] = [1120, 645]
    corner[4] = [1140, 800]
    corner[1:3] = [[900, 300], [900, -300]]
    corner[5] = [700, 0]
    point = pressure_position(corner, active, 4, 0, np.array([1120, 620]), 8.325)
    assert abs(point[0]) <= 1135 and abs(point[1]) <= 655
    assert np.all(np.linalg.norm(corner[others] - point, axis=1) >= 30.5)
    # The ball itself blocks the original preferred pressure point.
    simple = pp.copy()
    simple[1:4] = [[-500, 300], [-500, -300], [-900, 0]]
    simple[5] = [500, -300]
    point = pressure_position(simple, active, 4, 0, np.array([42., 0]), 8.325)
    assert np.linalg.norm(point - [42, 0]) >= 23.825


def test_pressure_placement_preserves_source_if_all_candidates_are_blocked():
    pp = np.array([[0., 0], [-500, 300], [-500, -300], [-900, 0],
                   [200, 0], [500, -300], [900, 300], [1000, 0]])
    active = np.ones(8, bool)
    # A synthetic huge ball covers every point in the local 60..70 px search;
    # this checks the fallback, independent of candidate enumeration order.
    point = pressure_position(pp, active, 4, 0, np.array([0., 0]), 100.0)
    assert np.array_equal(point, pp[4])


def test_mastery_needs_learning_time_and_disjoint_successful_windows():
    cur, _ = _setup(stage=1, min_episodes=32, window=32, threshold=0.4,
                    mastery_windows=2, min_stage_updates=100)
    cur.window.extend([1] * 32)
    assert cur.advance(learning_updates=99) is None and cur.stage == 1
    assert cur.advance(learning_updates=100) is None and cur.mastery_streak == 1
    assert not cur.window and cur.advance(learning_updates=101) is None
    state = json.loads(json.dumps(cur.state_dict()))
    restored, _ = _setup(stage=1, min_episodes=32, window=32, threshold=0.4,
                         mastery_windows=2, min_stage_updates=100)
    restored.load_state_dict(state)
    restored.window.extend([0] * 32)
    assert restored.advance(learning_updates=101) is None and restored.mastery_streak == 0
    restored.window.extend([1] * 32)
    assert restored.advance(learning_updates=102) is None
    restored.window.extend([1] * 32)
    assert restored.advance(learning_updates=103)['to_stage'] == 2
    assert restored.stage_started_learning_update == 103
    restored.window.extend([1] * 32)
    assert restored.advance(learning_updates=202) is None


def test_transfer_keeps_half_rows_and_fixed_replay_of_pass_skills():
    cur, arena = _setup(n=20, stage=4, transfer_frac=0.5, replay_frac=0.5)
    rows = np.arange(20)
    for _ in range(4):
        arena.env.start_match(rows, delay=np.full((20, 8), 8))
        cur.reset(arena, rows)
        assert cur.active.sum() == 10
        stages = cur.kind[cur.active]
        assert (stages == 4).sum() == 5
        assert (stages == 1).sum() == 3 and (stages == 2).sum() == 2


def test_team_retention_allows_flight_but_needs_control_to_finish():
    cur, arena = _setup(stage=1, possession_retention=True)
    env = arena.env
    _kick(cur, env)
    _receive(cur, env)
    env.ball_pos[0] = [0, 400]  # subsequent pass in flight, no own contact
    for _ in range(40):
        _, done, _ = cur.step(env, _event())
        assert not done[0]
    assert cur.retained[0] >= 120
    env.ball_pos[0] = env.player_pos[0, 1]
    reward, done, stats = cur.step(env, _event(touches=(1,)))
    assert done[0] and stats['successes'] == 1 and reward[0, 0] == 1


def test_team_retention_still_fails_on_opponent_touch():
    cur, arena = _setup(stage=1, possession_retention=True)
    _kick(cur, arena.env)
    _receive(cur, arena.env)
    enemy = int(np.flatnonzero(arena.env.active[0] & (TEAM != cur.attack_team[0]))[0]) if (arena.env.active[0] & (TEAM != cur.attack_team[0])).any() else 4
    arena.env.active[0, enemy] = True
    reward, done, _ = cur.step(arena.env, _event(touches=(enemy,)))
    assert done[0] and reward[0, 0] == -1
