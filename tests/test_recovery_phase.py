"""Contratos de la fase de recuperación colectiva."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from bots.scripted import (_r3_base_roles, _r3_direct_press, _r3_pass_score,
                           _r3_possession, scripted_actions)
from env.haxball_env import HaxballEnv, U_ENT_DIM, U_SELF_DIM
from train.model import SetActorCritic
from train.recurrent_model import RecurrentSetActorCritic


def observations(batch=4, entities=5):
    x = torch.randn(batch, U_SELF_DIM + entities * U_ENT_DIM)
    x[:, U_SELF_DIM::U_ENT_DIM] = 1
    x[:, U_SELF_DIM + U_ENT_DIM::2 * U_ENT_DIM] = 1
    return x


def test_attention_migration_is_exactly_closed_then_trainable():
    torch.manual_seed(14)
    baseline = SetActorCritic(U_SELF_DIM, hidden=32, layers=2, ent_hidden=8, ent_layers=1,
                              pooling="meanmax", rule_observation="masked")
    checkpoint = {"model_config": baseline.config(), "model": baseline.state_dict()}
    candidate = SetActorCritic(U_SELF_DIM, hidden=32, layers=2, ent_hidden=8, ent_layers=1,
                               pooling="attentive_meanmax", rule_observation="masked")
    candidate.initialize_from(checkpoint)
    x = observations()
    for actual, expected in zip(candidate(x), baseline(x)):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert candidate.attn_gate.item() == 0
    candidate(x)[0].sum().backward()
    assert candidate.attn_gate.grad is not None and candidate.attn_gate.grad.abs() > 0


def test_attention_gru_migration_preserves_policy_and_value():
    baseline = SetActorCritic(U_SELF_DIM, hidden=32, layers=2, ent_hidden=8, ent_layers=1,
                              pooling="meanmax", rule_observation="masked")
    checkpoint = {"model_config": baseline.config(), "model": baseline.state_dict()}
    candidate = RecurrentSetActorCritic(U_SELF_DIM, hidden=32, layers=2, ent_hidden=8, ent_layers=1,
                                        pooling="attentive_meanmax", rule_observation="masked", memory_size=64)
    candidate.initialize_from(checkpoint)
    x = observations()
    logits, value, _ = candidate.step(x, torch.randn(len(x), 64), torch.randint(0, 19, (len(x),)))
    expected_logits, expected_value = baseline(x)
    torch.testing.assert_close(logits, expected_logits, rtol=0, atol=0)
    torch.testing.assert_close(value, expected_value, rtol=0, atol=0)
    assert candidate.attn_gate.item() == 0 and candidate.memory_size == 64


def test_r2_golden_actions_for_both_colours_and_team_sizes():
    fixture = json.loads((Path(__file__).parent / "fixtures/scripted_r2_golden.json").read_text())
    for case in fixture["cases"]:
        env = HaxballEnv(1, case["T"], fixture["stadium"], seed=100, random_reset_prob=0)
        env.reset()
        env.sim.kickoff[:] = False
        env.sim.vel[:] = 0
        env.sim.ball_pos[0] = case["ball"]
        env.sim.ball_vel[0] = case["ball_vel"]
        env.sim.player_pos[0] = case["players"]
        np.testing.assert_array_equal(scripted_actions(env, policy="r2")[0], case["expected"])


def test_r3_components_are_independently_verifiable():
    teams = np.array([0, 0, 0, 1, 1, 1])
    keeper, count = _r3_base_roles(teams, 0, 3)
    assert (keeper, count) == (2, 3)
    possession = _r3_possession(np.array([10., 80., 90., 50., 60., 70.]), teams, 0, 30.)
    assert possession[0] == 1 and possession[1] == 0
    assert not _r3_direct_press(-1, 3, 20., 2, -500., 1000.)
    assert _r3_direct_press(-1, 3, 5., 2, -500., 1000.)
    assert _r3_pass_score(.5, .5, .5, .2, 2) > _r3_pass_score(.5, .5, .5, .2, 1)


def test_r3_styles_rotate_uniformly_by_episode_without_rng():
    env = HaxballEnv(9, 3, "big", seed=9, random_reset_prob=0)
    env.reset()
    before = env.rng.bit_generator.state
    assert np.bincount(env.scripted_style, minlength=3).tolist() == [3, 3, 3]
    old = env.scripted_style.copy()
    env._reset_envs(np.arange(env.N), kickoff_team=np.zeros(env.N, dtype=np.int64))
    np.testing.assert_array_equal(env.scripted_style, (old + 1) % 3)
    assert env.rng.bit_generator.state != before  # kickoff side/reset still uses normal env RNG


def _touch(env, player, position):
    touches = np.zeros((env.N, env.P), dtype=bool)
    touches[0, player] = True
    pos = np.zeros((env.N, 2), dtype=float)
    pos[0] = position
    return env._cooperation_touch_reward(touches, pos)


def test_pass_hold_chain_return_farming_and_possession_cap():
    env = HaxballEnv(1, 3, "big", seed=1, random_reset_prob=0)
    env.reset()
    env.sim.kickoff[:] = False
    env.setpiece_team[:] = -1
    env.sim.player_pos[0] = [[0, 0], [100, 0], [200, 0], [0, 250], [100, 250], [200, 250]]
    env.last_touch_player[0], env.last_touch_pos[0] = 0, (0, 0)
    _touch(env, 1, (100, 0))
    reward, events = env._advance_pending_passes(env.rcfg.team_pass_hold_ticks - 1)
    assert reward.sum() == 0 and events["passes"].sum() == 0
    reward, events = env._advance_pending_passes(1)
    assert reward.sum() > 0 and events["passes"][0, 0] == 1
    # B -> A sin progreso ni espacio: cuenta telemetría, no paga ni forma cadena.
    _touch(env, 0, (0, 0))
    reward, events = env._advance_pending_passes(env.rcfg.team_pass_hold_ticks)
    assert reward.sum() == 0 and events["pass_chains"].sum() == 0
    # A -> B -> C progresivo: una sola cadena.
    _touch(env, 1, (100, 0))
    env._advance_pending_passes(env.rcfg.team_pass_hold_ticks)
    _touch(env, 2, (200, 0))
    _, events = env._advance_pending_passes(env.rcfg.team_pass_hold_ticks)
    assert events["pass_chains"][0, 0] == 1
    assert env.coop_reward_spent[0, 0] <= env.rcfg.team_pass_possession_cap


def test_match_final_score_reports_scoreless_draw():
    env = HaxballEnv(2, 1, "classic", frame_skip=3, max_ticks=6, seed=4, random_reset_prob=0)
    env.reset()
    actions = np.zeros((2, 2), dtype=np.int64)
    env.step(actions)
    _, _, _, info = env.step(actions)
    assert info["match_done"].all()
    np.testing.assert_array_equal(info["final_score"], np.zeros((2, 2), dtype=np.int64))

