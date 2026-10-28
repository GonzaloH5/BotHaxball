"""Goal credit and fixed finishing practice preserve cooperative mastery."""
from types import SimpleNamespace

import numpy as np
import pytest

from env.rs4z import contract as C, kernel as K
from env.rs4z.core import RS4ZEnv
from learn.x4_coop_curriculum import CoopBank, CoopCurriculum, TEAM
from tests.rs4z.test_x4_coop_curriculum import _data, _event, _kick, _receive, _setup


@pytest.mark.parametrize("stage", [1, 2])
def test_goal_after_policy_pass_rewards_team_without_waiting_for_retention(stage):
    cur, arena = _setup(stage=stage, reward_cooperative_goals=True)
    env = arena.env
    _kick(cur, env)
    _, done, stats = _receive(cur, env)
    assert cur.received[0] and stats["policy_passes"] == 1
    assert not done[0] and cur.retained[0] == 0

    # A real goal event can coincide with the engine having replaced play by a
    # kickoff. It must preserve the successful assist, without waiting two
    # seconds or requiring control of the now-reset ball.
    env.ri[0, K.RI_KO] = 1
    reward, done, stats = cur.step(env, _event(goal=1))
    assert done[0] and reward[0].tolist() == [1.0, -1.0]
    assert stats["cooperative_goals"] == 1 and stats["successes"] == 1
    assert list(cur.window) == [1] and cur.total_successes[stage] == 1
    again, _, stats = cur.step(env, _event(goal=1))
    assert not again.any() and stats["episodes"] == 0


@pytest.mark.parametrize("stage", [1, 2])
def test_goal_without_received_policy_pass_is_neutral_and_fails_mastery(stage):
    cur, arena = _setup(stage=stage, reward_cooperative_goals=True)
    cur.age[0] = 30
    arena.env.ri[0, K.RI_KO] = 1
    reward, done, stats = cur.step(arena.env, _event(goal=1))
    assert done[0] and not reward.any()
    assert stats["neutral_own_goals"] == 1 and stats["failures"] == 1
    assert stats["successes"] == stats["cooperative_goals"] == 0
    assert list(cur.window) == [0] and cur.total_successes[stage] == 0


@pytest.mark.parametrize("stage", [1, 2])
def test_cooperative_goal_cannot_credit_receipt_before_new_actions_apply(stage):
    cur, arena = _setup(stage=stage, reward_cooperative_goals=True)
    cur.received[0] = True
    # Set the pre-step age so the event lands exactly at the queue boundary:
    # the contract requires strict > delay_max + frame_skip.
    cur.age[0] = cur.delay_max[0]
    reward, done, stats = cur.step(arena.env, _event(goal=1))
    assert done[0] and not reward.any()
    assert stats["neutral_own_goals"] == 1 and list(cur.window) == [0]


@pytest.mark.parametrize("stage", [3, 4])
@pytest.mark.parametrize("early_kick", [False, True])
def test_finishing_goal_needs_own_kick_after_inherited_queue(stage, early_kick):
    # Reuse the scorer fixture without asking its central-field recording to
    # satisfy the attacking-third reset selector tested separately below.
    cur, arena = _setup(stage=1, reward_cooperative_goals=True)
    cur.stage = cur.kind[0] = stage
    cur.age[0] = cur.delay_max[0] if early_kick else 30
    arena.env.ri[0, K.RI_KO] = 1
    reward, done, stats = cur.step(arena.env, _event(goal=1, kicks=(0,) if early_kick else ()))
    assert done[0] and not reward.any() and not cur.finish_kicked[0]
    assert stats["neutral_own_goals"] == 1 and stats["failures"] == 1
    assert cur.total_successes[stage] == 0 and list(cur.window) == [0]


@pytest.mark.parametrize("stage", [3, 4])
@pytest.mark.parametrize("same_step", [False, True])
def test_finishing_rewards_legitimate_own_shot_without_requiring_another_pass(stage, same_step):
    cur, arena = _setup(stage=1, reward_cooperative_goals=True)
    cur.stage = cur.kind[0] = stage
    cur.age[0] = 30
    if same_step:
        arena.env.ri[0, K.RI_KO] = 1
        reward, done, stats = cur.step(arena.env, _event(kicks=(1,), goal=1))
    else:
        _, done, _ = cur.step(arena.env, _event(kicks=(1,)))
        assert not done[0] and cur.finish_kicked[0]
        arena.env.ri[0, K.RI_KO] = 1
        reward, done, stats = cur.step(arena.env, _event(goal=1))
    assert done[0] and cur.finish_kicked[0] and not cur.received[0]
    assert reward[0].tolist() == [1.0, -1.0]
    assert stats["successes"] == 1 and cur.total_successes[stage] == 1


def _attacking_third_bank():
    data = _data()
    data.ball[:, 0] += 500.0
    data.pos[:, :4, 0] += 500.0
    data.pos[:, 4:] = np.array([[850.0, 200.0], [1000.0, -200.0],
                               [700.0, 300.0], [1000.0, 400.0]])
    return CoopBank(data, limit=10)


def test_fixed_mixture_keeps_half_full_matches_and_resets_finishing_after_receipt():
    bank, n = _attacking_third_bank(), 32
    env = RS4ZEnv(n, map="sanguchito_rs_x4", max_delay=15, frame_skip=3, seed=1)
    arena = SimpleNamespace(env=env, learner=np.ones((n, 8), bool), opp=np.full(n, -1))
    cur = CoopCurriculum(bank, n, np.random.default_rng(4), frac=0.5, start_stage=2,
                         replay_frac=0.25, finish_frac=0.25, reward_cooperative_goals=True)
    rows = np.arange(n)
    for _ in range(3):
        env.start_match(rows, delay=np.full((n, 8), 14), match_ticks=10000)
        cur.reset(arena, rows)
        assert cur.active.sum() == 16 and (~cur.active).sum() == 16
        assert (cur.kind == 2).sum() == 8
        assert (cur.kind == 1).sum() == (cur.kind == 3).sum() == 4
        finishing = np.flatnonzero(cur.active & (cur.kind == 3))
        source = bank.d.ball[int(bank.records[0, 1]) + 1, :2]
        assert np.all(env.ball_pos[finishing] == source)
        assert np.all(env.ball_pos[finishing, 0] >= .35 * C.LINE_W)
        assert not cur.received[finishing].any() and not cur.finish_kicked[finishing].any()
        assert np.all(arena.opp[finishing] == -2)
        assert np.all(env.active[finishing].sum(axis=1) == 5)
        for row in finishing:
            own = TEAM == cur.attack_team[row]
            assert (env.active[row] & own).sum() == 3
            assert (env.active[row] & ~own).sum() == 2
            assert not (arena.learner[row] & env.active[row] & ~own).any()
            # Select both the source defender nearest the receiver and the
            # remaining defender covering the target goal, rather than two
            # opponents near the original passer.
            foes = np.flatnonzero(~own)
            pp = bank.d.pos[int(bank.records[0, 1]) + 1]
            nearest = foes[np.argmin(np.linalg.norm(pp[foes] - pp[cur.receiver[row]], axis=1))]
            remaining = foes[foes != nearest]
            goal = np.array([C.LINE_W, 0.0])
            cover = remaining[np.argmin(np.linalg.norm(pp[remaining] - goal, axis=1))]
            assert env.active[row, nearest] and env.active[row, cover]
        assert cur.stage == 2


def test_finishing_successes_cannot_promote_main_pressure_stage():
    cur, arena = _setup(stage=1, reward_cooperative_goals=True,
                        min_episodes=8, window=8, threshold=0.4)
    cur.stage = 2
    for _ in range(8):
        cur.kind[0] = 3
        cur.finished[0] = cur.finish_kicked[0] = False
        cur.age[0] = 30
        _, done, stats = cur.step(arena.env, _event(kicks=(1,), goal=1))
        assert done[0] and stats["successes"] == 1
    assert cur.total_episodes[3] == cur.total_successes[3] == 8
    assert cur.total_episodes[2] == 0 and not cur.window
    assert cur.advance(learning_updates=1000) is None and cur.stage == 2
