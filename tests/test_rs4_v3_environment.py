import json
from pathlib import Path

import numpy as np
import pytest
import torch

from env.haxball_env import HaxballEnv
from env.rs4_v3 import RS4ScenarioEnv, RestartCohorts, SCENARIOS, restart_potential
from eval.rs4_v3 import MixedTeamAgent, functional_trial
from eval.agents import ScriptedAgent
from tools.build_rs4_sequences import SequenceShards, iter_shards, replay_split
from tools.train_rs4_imitator import validation_gate


def practice(n=10, **settings):
    return RS4ScenarioEnv(HaxballEnv(n, 4, "rs_one", out_of_bounds=True,
                                  obs_layout="universal", seed=51), settings)


@pytest.mark.parametrize("scenario", SCENARIOS)
@pytest.mark.parametrize("color", (0, 1))
def test_practice_preserves_rs4_and_no_private_observation(scenario, color):
    env = practice(2)
    env.reset()
    env.assign(np.arange(2), scenario=scenario, team=color)
    assert env.P == 8 and env.T == 4 and env.sim.st.name
    obs = env.observe()
    assert obs.shape == (2, 8, 127)
    env.scenario[:] = -1
    env.drill_team[:] = 1 - color
    assert np.array_equal(obs, env.observe())


def test_quota_keeps_at_least_sixty_percent_full_matches():
    env = practice(11, drill_fraction=.4)
    env.reset()
    assert env.is_drill.sum() == 4
    rows = np.flatnonzero(env.is_drill)
    for _ in range(10):
        env.assign(rows)
        assert env.is_drill.sum() == 4
    env.configure({"drill_fraction": .2})
    env.assign(rows)
    assert env.is_drill.sum() == 2


def test_new_drill_retargets_to_learner_color_without_changing_kind_or_quota():
    env = practice(10)
    env.learner_team = np.arange(10) % 2
    env.reset()
    rows = np.flatnonzero(env.is_drill)
    np.testing.assert_array_equal(env.drill_team[rows], env.learner_team[rows])
    kinds = env.scenario.copy()
    normal = ~env.is_drill
    before = env.observe()[normal].copy()
    env.retarget(rows, 1 - env.drill_team[rows])
    np.testing.assert_array_equal(kinds, env.scenario)
    np.testing.assert_array_equal(before, env.observe()[normal])
    assert env.is_drill.sum() == 4


def test_program_aliases_apply_and_unchanged_phase_does_not_rebase():
    env = practice(2)
    env.reset()
    phase = {"exercise_fraction": .3, "guide_coef": .06, "restart_execute_bonus": .01,
             "exercise_weights": {"throw_in": 3, "build_up": 2, "defensive_transition": 1}}
    env.configure(phase)
    assert env.rcfg.rs4_tactical_coef == .06
    assert env.restart_bonus == .01
    assert env.weights[1] == .5
    assert env.rcfg.kickoff_approach == 0 and env.rcfg.rs4_restart_stall == .25
    potential = env.base._rs4_phi
    env.configure(phase)
    assert env.base._rs4_phi is potential


def test_practice_cut_is_truncated_has_pre_cut_bootstrap_and_no_match_result():
    env = practice(2, guide_coef=0)
    env.reset()
    env.assign(np.arange(2), scenario="attack", team=0)
    env.drill_limit[:] = 3
    obs, _, done, info = env.step(np.zeros((2, 8), dtype=np.int64))
    assert done.all() and info["truncated"].all()
    assert not info["match_done"].any()
    assert (info["final_score"] == -1).all()
    assert not np.array_equal(obs, info["final_obs"])
    assert len(info["scenario_result"]) == 2


def test_shared_goal_reward_and_timeout_penalty_are_not_redefined_by_player():
    env = practice(2, guide_coef=0, restart_potential_coef=0, restart_bonus=0)
    env.reset()
    env.assign(np.arange(2), scenario="corner", team=0)
    env.base.setpiece_limit[:] = 1
    _, reward, done, info = env.step(np.zeros((2, 8), dtype=np.int64))
    assert done.all()
    assert np.allclose(reward[:, :4], -.25)
    assert np.allclose(reward[:, 4:], 0)
    assert env.rcfg.goal == 1


def test_executor_support_potential_is_bounded_shared_and_permutation_invariant():
    env = practice(2)
    env.reset()
    env.assign(np.arange(2), scenario="corner", team=0)
    phi = restart_potential(env.base)
    assert np.all(phi >= 0) and np.all(phi <= env.rcfg.rs4_restart_approach)
    assert np.array_equal(phi[:, :1], phi[:, 1:2])
    assert np.all(phi[:, 4:] == 0)
    env.sim.player_pos[:, :4] = env.sim.player_pos[:, :4][:, [3, 1, 0, 2]]
    assert np.allclose(phi, restart_potential(env.base))


def test_executor_ties_do_not_depend_on_player_order_and_final_guide_is_zero():
    env = practice(2)
    env.reset()
    env.assign(np.arange(2), scenario="lateral", team=0)
    ball = env.sim.ball_pos.copy()
    env.sim.player_pos[:, 0] = ball + [-30, 40]
    env.sim.player_pos[:, 1] = ball + [-30, -40]
    phi = restart_potential(env.base)
    env.sim.player_pos[:, [0, 1]] = env.sim.player_pos[:, [1, 0]]
    np.testing.assert_allclose(phi, restart_potential(env.base), rtol=0, atol=0)
    env.configure({"guide_coef": 0, "restart_bonus": 0})
    assert env.rcfg.rs4_tactical_coef == 0 and env.rcfg.rs4_restart_approach == 0


def test_restart_potential_discounted_cycle_has_no_positive_reward():
    env = practice(2)
    env.reset()
    env.assign(np.arange(2), scenario="corner", team=0)
    first = restart_potential(env.base)
    env.sim.player_pos[:, :4, 0] -= 10
    second = restart_potential(env.base)
    gamma = env.rcfg.gamma
    outward = gamma * second - first
    return_trip = gamma * first - second
    total = outward + gamma * return_trip
    np.testing.assert_allclose(total, (gamma**2 - 1) * first, atol=1e-16)
    assert (total <= 1e-16).all()


def test_temporal_cut_censors_restart_instead_of_counting_failure():
    env = practice(2)
    env.reset()
    env.assign(np.arange(2), scenario="corner", team=0)
    tracker = RestartCohorts(env.base)
    tracker.begin()
    info = {"kicked": np.zeros((2, 8), dtype=bool), "goal": np.zeros(2, dtype=int),
            "final_obs": env.observe(), "truncated": np.ones(2, dtype=bool),
            "events": {"restart_timeouts": np.zeros((2, 2), dtype=int)}}
    assert tracker.update(info, np.ones(2, dtype=bool)) == []
    report = tracker.report()["corner_red"]
    assert report["resolved"] == 0 and report["censored"] == 2 and report["success_rate"] is None


def test_match_end_auto_reset_does_not_turn_censored_restart_into_a_failure():
    env = practice(2)
    env.reset()
    env.assign(np.arange(2), scenario="corner", team=0)
    tracker = RestartCohorts(env.base)
    tracker.begin()
    final = env.observe().copy()
    env.base._reset_envs(np.arange(2), kickoff_team=1)
    info = {"kicked": np.zeros((2, 8), dtype=bool), "goal": np.zeros(2, dtype=int),
            "final_obs": final, "truncated": np.ones(2, dtype=bool),
            "events": {"restart_timeouts": np.zeros((2, 2), dtype=int)}}
    assert tracker.update(info, np.ones(2, dtype=bool)) == []
    assert tracker.report()["corner_red"]["censored"] == 2


@pytest.mark.parametrize("new_owner,new_kind", [(1, 1), (1, 2), (0, 1), (0, 2)])
def test_bad_restart_immediately_resolves_before_next_opportunity(new_owner, new_kind):
    env = practice(2)
    env.reset()
    env.assign(np.arange(2), scenario="corner", team=0)
    tracker = RestartCohorts(env.base)
    tracker.begin()
    # The corner was kicked, went out, and was replaced before twelve ticks.
    # last_touch stays red, so possession-transfer detection alone cannot work.
    env.base.setpiece_team[:] = new_owner
    env.base.setpiece_kind[:] = new_kind
    env.base.setpiece_pos[:, 1] -= 15
    env.base.last_touch[:] = 0
    info = {"kicked": np.zeros((2, 8), dtype=bool), "goal": np.zeros(2, dtype=int),
            "final_obs": env.observe(), "truncated": np.zeros(2, dtype=bool),
            "events": {"restart_timeouts": np.zeros((2, 2), dtype=int)}}
    info["kicked"][:, 0] = True
    outcomes = tracker.update(info, np.zeros(2, dtype=bool))
    assert len(outcomes) == 2 and not any(row["success"] for row in outcomes)
    assert tracker.report()["corner_red"]["resolved"] == 2
    tracker.begin()
    assert np.all(tracker.owner == new_owner) and np.all(tracker.kind == new_kind)
    key = f"{'lateral' if new_kind == 1 else 'corner'}_{'red' if new_owner == 0 else 'blue'}"
    assert tracker.report()[key]["open"] == 2


def test_restart_cohort_resolves_once_and_separates_open_from_success_rate():
    env = practice(2)
    env.reset()
    env.assign(np.arange(2), scenario="corner", team=0)
    tracker = RestartCohorts(env.base)
    tracker.begin()
    tracker.begin()
    assert tracker.report()["corner_red"]["opportunities"] == 2
    info = {"kicked": np.zeros((2, 8), dtype=bool), "goal": np.zeros(2, dtype=int),
            "final_obs": env.observe(), "events": {"restart_timeouts": np.array([[1, 0], [0, 0]])}}
    outcomes = tracker.update(info, np.array([True, False]))
    assert len(outcomes) == 1 and not outcomes[0]["success"]
    assert tracker.report()["corner_red"]["resolved"] == 1
    assert tracker.report()["corner_red"]["open"] == 1
    tracker.cut([1])
    assert tracker.report()["corner_red"]["censored"] == 1
    assert tracker.report()["corner_red"]["resolved"] == 1


@pytest.mark.parametrize("optimized", (False, True))
def test_participant_credit_preserves_shared_reward_and_possession_cap(optimized):
    env = HaxballEnv(1, 4, "rs_one", out_of_bounds=True, seed=0)
    env.reset()
    env.optimize_callbacks = optimized
    env.rcfg.rs4_pass_participant = .0005
    env.pending_pass_sender[0], env.pending_pass_receiver[0] = 0, 1
    env.pending_pass_team[0], env.pending_pass_progress[0] = 0, 1
    bonus, _ = env._advance_pending_passes(12)
    assert bonus[0, 0] == bonus[0, 1] and bonus[0, 0] > bonus[0, 2] > 0
    assert env.coop_reward_spent[0, 0] <= env.rcfg.team_pass_possession_cap
    env.pending_pass_sender[0], env.pending_pass_receiver[0] = 1, 0
    env.pending_pass_team[0], env.pending_pass_progress[0] = 0, 0
    env.last_pass_sender[0], env.last_pass_receiver[0] = 0, 1
    env.sim.player_pos[0, 0] = env.sim.player_pos[0, 1]
    bonus, _ = env._advance_pending_passes(12)
    assert np.all(bonus == 0)


def test_applied_previous_action_strips_protected_rivals_kick():
    env = practice(2)
    env.reset()
    env.assign(np.arange(2), scenario="corner", team=0)
    _, _, _, info = env.step(np.full((2, 8), 9, dtype=np.int64))
    assert np.all(info["executed_actions"][:, 4:] == 0)


def test_streamed_shards_mask_padding_and_preserve_player_identity(tmp_path):
    writer = SequenceShards(tmp_path, "replay", "validation", length=4, shard_sequences=1)
    writer.add(17, [(np.ones(127), 2, 18, 0, -1, True), (np.ones(127) * 2, 3, 2, 3, 2, False)])
    writer.flush()
    manifest = {"replays": [{"split": "validation", "shards": writer.manifest}]}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    shard = next(iter_shards(path, "validation"))
    assert shard["valid"].tolist() == [[True, True, False, False]]
    assert shard["player_id"].tolist() == [17]
    assert shard["previous_action"].tolist() == [[18, 2, 18, 18]]
    assert replay_split("same-replay") == replay_split("same-replay")


def test_imitation_activation_gate_rejects_regression_and_is_never_automatic():
    baseline = {"general": {"samples": 1000, "accuracy": .5, "cross_entropy": 2.0},
                "recognized_restarts": {"samples": 200, "accuracy": .4}}
    candidate = {"general": {"samples": 1000, "accuracy": .51, "cross_entropy": 1.9},
                 "recognized_restarts": {"samples": 200, "accuracy": .5}}
    assert validation_gate(candidate, baseline)["approved_as_weak_reference"]
    assert not validation_gate(candidate, baseline)["activated"]
    candidate["general"]["accuracy"] = .49
    assert not validation_gate(candidate, baseline)["approved_as_weak_reference"]
