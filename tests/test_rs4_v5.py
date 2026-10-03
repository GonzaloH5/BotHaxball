"""RS4 v5: recompensa anti-bloque, programa simple y saque inicial de R3."""
import numpy as np
import pytest

from bots.scripted import scripted_actions
from env.haxball_env import HaxballEnv
from env.rewards import RewardConfig
from env.rs4_v3 import RS4ScenarioEnv
from env.tasks import load_catalog, make_env
from train.rs4_program_v5 import ProgramStateV5, default_program_v5, program_from_config


def rs4_env(n=4, max_ticks=7200, **reward):
    base = dict(shaping_coef=0, team_spread_floor=0, kickoff_approach=0, w_defense_support=0,
                rs4_tactical_coef=0, corner_execute=0)
    base.update(reward)
    return HaxballEnv(n, 4, "rs_one", frame_skip=3, max_ticks=max_ticks, random_reset_prob=0, seed=7,
                      obs_layout="universal", max_entities=7, out_of_bounds=True, corner_reset_prob=0,
                      kickoff_timeout=180, reward=RewardConfig(**base))


def run_until_done(env, actions=None):
    env.reset()
    for _ in range(10_000):
        act = np.zeros((env.N, env.P), dtype=np.int64) if actions is None else actions(env)
        _, rew, done, info = env.step(act)
        if done.any():
            return rew, done, info
    pytest.fail("el episodio no terminó")


def test_no_goal_penalty_is_a_real_terminal_for_both_teams():
    env = rs4_env(max_ticks=60, no_goal_penalty=.3)
    rew, done, info = run_until_done(env)
    assert done.all() and not info["truncated"].any()
    assert np.allclose(rew, -.3)
    assert np.allclose(info["reward_terms"]["no_goal"], -.3)


def test_without_penalty_the_timeout_stays_truncated():
    env = rs4_env(max_ticks=60)
    rew, done, info = run_until_done(env)
    assert done.all() and info["truncated"].all()
    assert np.allclose(rew, 0)


def test_match_with_a_goal_is_not_penalized():
    env = rs4_env(n=1, max_ticks=60, no_goal_penalty=.3)
    env.reset()
    env.match_score[0] = (1, 0)  # un gol anterior en el mismo partido
    for _ in range(100):
        _, rew, done, info = env.step(np.zeros((1, 8), dtype=np.int64))
        if done.any():
            break
    assert info["truncated"][0] and np.allclose(rew, 0)


def test_possession_change_is_zero_sum_between_teams():
    env = rs4_env(n=16, possession_change=.01)
    env.reset()
    rng = np.random.default_rng(0)
    seen = 0.
    for _ in range(1500):
        act = scripted_actions(env, policy="r3", rng=rng)
        _, _, _, info = env.step(act)
        terms = info["reward_terms"]["possession"]
        red, blue = terms[:, env.sim.player_team == 0], terms[:, env.sim.player_team == 1]
        # Cada jugador del equipo recibe lo mismo; un equipo gana lo que el otro pierde.
        assert np.allclose(red, red[:, :1]) and np.allclose(blue, blue[:, :1])
        assert np.allclose(red[:, 0], -blue[:, 0])
        seen += np.abs(red).sum()
    assert seen > 0, "R3 contra R3 debería producir alguna pérdida de balón"


def test_v5_settings_configure_the_rs4_environment():
    program = ProgramStateV5(default_program_v5())
    env = RS4ScenarioEnv(rs4_env(n=10), program.settings())
    rc = env.rcfg
    assert rc.rs4_tactical_coef == 0 and rc.shaping_coef == 1
    assert rc.no_goal_penalty == pytest.approx(.3) and rc.possession_change == pytest.approx(.01)
    assert rc.team_pass_possession_cap == pytest.approx(.05) and rc.gamma == pytest.approx(.995)
    assert rc.w_ball_progress == pytest.approx(.1) and rc.kick_to_goal == 0
    env.reset()
    assert env.is_drill.sum() == 1  # 10% de ejercicios


def test_v5_lr_rises_when_policy_barely_moves_and_respects_ceiling():
    program = ProgramStateV5(default_program_v5())
    start = program.lr
    for iteration in range(1, 41):
        program.update_lr(.0005, iteration)
    assert program.lr > start
    assert program.lr <= program.lr_ceiling + 1e-12
    for iteration in range(41, 81):
        program.update_lr(.05, iteration)
    assert program.lr < start


def test_v5_schedules_and_state_roundtrip():
    cfg = {"rs4_program": default_program_v5()}
    program = program_from_config(cfg)
    assert isinstance(program, ProgramStateV5)
    assert program.bc_coef == pytest.approx(.05) and program.entropy_coef == pytest.approx(.005)
    program.advance_steps(1_500_000_000)
    assert program.bc_coef == pytest.approx(.03)
    assert program.snapshot_due and program.evaluation_due
    restored = program_from_config(cfg, program.state_dict())
    assert restored.relative_steps == program.relative_steps and restored.lr == program.lr
    other = dict(default_program_v5(), frozen_teammates=.5)
    with pytest.raises(ValueError):
        program_from_config({"rs4_program": other}, program.state_dict())


def test_v3_programs_still_use_the_v3_state():
    from train.rs4_program import ProgramState, default_program
    assert isinstance(program_from_config({"rs4_program": default_program()}), ProgramState)


def test_r3_kickoff_never_expires_against_moving_rivals():
    env = make_env(load_catalog()["rs4_4v4"], 32, 7, seed=3, random_reset_prob=0)
    env.corner_reset_prob = 0
    env.reset()
    rng = np.random.default_rng(3)
    stalls = 0
    for _ in range(1200):
        act = scripted_actions(env, policy="r3", rng=rng)
        rival = env.sim.kickoff[:, None] & (env.sim.player_team[None, :] != env.sim.kickoff_team[:, None])
        act = np.where(rival, rng.integers(0, 9, act.shape), act)
        _, _, _, info = env.step(act)
        stalls += int(info["stall"].sum())
    assert stalls == 0


def test_critic_warmup_freezes_lr_and_ends_on_schedule():
    program = ProgramStateV5(dict(default_program_v5(), critic_warmup_steps=1000))
    assert program.critic_warmup
    start = program.lr
    for iteration in range(1, 41):
        program.update_lr(0., iteration)
    assert program.lr == start
    program.advance_steps(1000)
    assert not program.critic_warmup


def test_older_v5_checkpoints_resume_without_new_fields():
    program = ProgramStateV5(default_program_v5())
    saved = program.state_dict()
    del saved["config"]["critic_warmup_steps"]
    assert ProgramStateV5(default_program_v5(), saved).relative_steps == 0


def test_rounding_negative_kl_after_warmup_is_accepted():
    program = ProgramStateV5(dict(default_program_v5(), critic_warmup_steps=10))
    program.advance_steps(10)
    assert program.update_lr(-1e-9, 10) == program.lr
    with pytest.raises(ValueError):
        program.update_lr(-.01, 20)


def test_step_pressure_charges_both_teams_only_in_open_play():
    env = rs4_env(n=2, no_goal_step_penalty=.0005)
    env.reset()
    _, rew, _, info = env.step(np.zeros((2, 8), dtype=np.int64))
    assert env.sim.kickoff.all() and np.allclose(info["reward_terms"]["pressure"], 0)
    env.sim.kickoff[:] = False
    _, rew, done, info = env.step(np.zeros((2, 8), dtype=np.int64))
    pressure = info["reward_terms"]["pressure"]
    assert np.allclose(pressure[~done], -.0005)


def test_step_pressure_must_stay_below_the_restart_stall_penalty():
    with pytest.raises(ValueError):
        ProgramStateV5(dict(default_program_v5(), reward=dict(default_program_v5()["reward"], no_goal_step_penalty=.002)))
    program = ProgramStateV5(dict(default_program_v5(), reward=dict(default_program_v5()["reward"], no_goal_step_penalty=.0005)))
    assert program.settings()["no_goal_step_penalty"] == pytest.approx(.0005)


def test_v5_checkpoints_without_new_reward_fields_resume():
    program = ProgramStateV5(default_program_v5())
    saved = program.state_dict()
    del saved["config"]["reward"]["no_goal_step_penalty"]
    ProgramStateV5(default_program_v5(), saved)


def test_critic_features_are_in_team_frame_and_track_score():
    env = rs4_env(n=2)
    env.reset()
    env.match_score[0] = (2, 1)
    env.last_touch[:] = 1
    features = env.critic_features()
    assert features.shape == (2, 8, env.CRITIC_FEATURES)
    red, blue = features[0, 0], features[0, 4]
    assert red[1] == pytest.approx(.4) and red[2] == pytest.approx(.2) and red[3] == pytest.approx(1 / 3)
    assert blue[1] == pytest.approx(.2) and blue[3] == pytest.approx(-1 / 3)
    assert red[4] == 0 and features[1, 0, 4] == 1  # sin goles sólo en el segundo partido
    assert red[11] == 0 and blue[11] == 1          # último toque azul
    kickoff_red = env.sim.kickoff_team[0] == 0
    assert red[5] == float(kickoff_red) and blue[5] == float(not kickoff_red)


def test_privileged_critic_starts_identical_and_never_touches_the_policy():
    import torch
    from train.model import SetActorCritic
    from env.haxball_env import U_SELF_DIM
    base = SetActorCritic(U_SELF_DIM, ent_hidden=16, pooling="attentive_meanmax", rule_observation="masked")
    ck = {"model": base.state_dict(), "model_config": base.config()}
    critic = SetActorCritic(U_SELF_DIM, ent_hidden=16, pooling="attentive_meanmax", rule_observation="masked",
                            critic_features=12)
    critic.initialize_from(ck)
    obs, extra = torch.randn(5, 127), torch.randn(5, 12)
    assert torch.allclose(base.value(obs) if hasattr(base, "value") else base(obs)[1], critic.value(obs, extra))
    with torch.no_grad():
        critic.critic_proj.weight.normal_()
    logits, value = critic(obs, extra)
    assert torch.allclose(logits, base.logits(obs)) and not torch.allclose(value, base(obs)[1])
    assert critic.config()["critic_features"] == 12
