"""Córners durante entrenamiento, saques tras gol intactos y métricas sin doble conteo."""
from dataclasses import replace

import numpy as np
import pytest

from env.haxball_env import RESTART_EVENT_NAMES
from env.tasks import load_catalog, make_env


def environment(n=8, prob=1., **kwargs):
    task = replace(load_catalog()["rs4_4v4"], corner_reset_prob=prob)
    return make_env(task, n, 7, seed=51, random_reset_prob=0., **kwargs)


@pytest.mark.parametrize("optimized", [False, True])
def test_automatic_non_goal_resets_keep_corner_curriculum(optimized):
    env = environment(max_ticks=3, optimize_rollout=optimized)
    env.reset()
    for _ in range(6):
        assert (env.setpiece_kind == 2).all()
        assert not env.sim.kickoff.any()
        _, _, done, info = env.step(np.zeros((env.N, env.P), int))
        assert done.all() and info["truncated"].all()
        assert not info["goal"].any()
        assert info["events"]["corner_opportunities"].sum() == env.N
        np.testing.assert_array_equal(info["events"]["corner_curriculum_starts"],
                                      info["events"]["corner_opportunities"])
        assert info["final_obs"].shape == (env.N, env.P, env.obs_dim)


def test_real_five_percent_curriculum_reappears_after_initial_reset(monkeypatch):
    env = environment(n=144, prob=.05, max_ticks=3)
    batches = []
    original = env._reset_corner_curriculum
    def record(idx):
        batches.append(len(idx))
        return original(idx)
    monkeypatch.setattr(env, "_reset_corner_curriculum", record)
    env.reset()
    batches.clear()
    for _ in range(20):
        _, _, _, info = env.step(np.zeros((env.N, env.P), int))
        assert not info["goal"].any()
    # El bug anterior producía exactamente cero en estos 2.880 reinicios.
    assert 40 < sum(batches) < 250


@pytest.mark.parametrize("team", [0, 1])
def test_explicit_kickoff_is_not_replaced_and_subset_is_preserved(team):
    env = environment(n=3)
    env.reset()
    untouched_pos = env.sim.pos[1:].copy()
    untouched_owners = env.setpiece_team[1:].copy()
    env._reset_envs(np.array([0]), kickoff_team=team)
    assert env.sim.kickoff[0] and env.sim.kickoff_team[0] == team
    assert env.setpiece_team[0] == -1 and env.setpiece_kind[0] == 0
    np.testing.assert_array_equal(env.sim.pos[1:], untouched_pos)
    np.testing.assert_array_equal(env.setpiece_team[1:], untouched_owners)


def test_mixed_goals_and_truncation_only_randomize_non_goal_row(monkeypatch):
    env = environment(n=3, max_ticks=3, optimize_rollout=False)
    env.reset()
    env.setpiece_team[:] = -1
    env.setpiece_kind[:] = 0
    env.sim.kickoff[:] = False
    env.sim.mask[:] = env.sim.base_mask
    env.sim.ball_pos[:] = 0
    env.sim.kicked[:] = False
    env.sim.touch[:] = False
    # Aislar el contrato de reset: gol rojo, gol azul y fin de partido simultáneos.
    monkeypatch.setattr(env.sim, "step", lambda actions: np.array([1, -1, 0]))
    _, _, done, info = env.step(np.zeros((3, 8), int))
    np.testing.assert_array_equal(info["goal"], [1, -1, 0])
    assert done.all()
    np.testing.assert_array_equal(env.sim.kickoff[:2], [True, True])
    np.testing.assert_array_equal(env.sim.kickoff_team[:2], [1, 0])
    np.testing.assert_array_equal(env.setpiece_kind, [0, 0, 2])
    np.testing.assert_array_equal(info["truncated"], [False, False, True])


def test_curriculum_opportunity_is_counted_once_not_each_decision():
    env = environment()
    env.reset()
    expected = np.column_stack([env.setpiece_team == t for t in (0, 1)]).astype(int)
    _, _, _, first = env.step(np.zeros((env.N, env.P), int))
    np.testing.assert_array_equal(first["events"]["corner_opportunities"], expected)
    np.testing.assert_array_equal(first["events"]["corner_curriculum_starts"], expected)
    for _ in range(5):
        _, _, done, info = env.step(np.zeros((env.N, env.P), int))
        assert not done.any()
        assert not info["events"]["corner_opportunities"].any()
        assert not info["events"]["corner_curriculum_starts"].any()


@pytest.mark.parametrize("optimized", [False, True])
@pytest.mark.parametrize("team", [0, 1])
@pytest.mark.parametrize("kind", [1, 2, 3])
def test_corner_timeout_is_separate_from_other_restarts(kind, team, optimized):
    env = environment(n=1, prob=0., optimize_rollout=optimized)
    env.reset()
    sim = env.sim
    sim.kickoff[:] = False
    sim.mask[:] = sim.base_mask
    sim.vel[:] = 0
    sim.player_pos[:] = [0, 0]
    sx = (1 if team == 0 else -1) * (1 if kind == 2 else -1)
    sim.ball_pos[:] = ([100, env.field_h + 20] if kind == 1
                      else [sx * (env.field_w + 20), 200])
    env.last_touch[:] = 1 - team
    env._set_piece([0])
    assert env.setpiece_team[0] == team and env.setpiece_kind[0] == kind
    env.setpiece_ticks[:] = env.setpiece_limit - 1
    _, _, done, info = env.step(np.zeros((1, env.P), int))
    assert done[0] and not info["truncated"][0]
    events = info["events"]
    assert events["restart_timeouts"][0, team] == 1
    assert events["corner_timeouts"][0, team] == int(kind == 2)
    assert events["corner_opportunities"][0, team] == int(kind == 2)
    assert not events["corner_curriculum_starts"].any()
    assert not events["corner_attempts"].any()
    assert not events["corner_successes"].any()
    _, _, _, next_info = env.step(np.zeros((1, env.P), int))
    assert not next_info["events"]["corner_timeouts"].any()


def test_natural_corner_counts_once_and_can_be_executed():
    env = environment(n=1, prob=0.)
    env.reset()
    sim = env.sim
    sim.kickoff[:] = False
    sim.mask[:] = sim.base_mask
    sim.vel[:] = 0
    sim.player_pos[:] = [0, 0]
    sim.ball_pos[:] = [env.field_w + 20, 200]
    env.last_touch[:] = 1
    # Generar el córner mediante la transición de salida normal.
    _, _, _, initial = env.step(np.zeros((1, env.P), int))
    assert initial["out"][0] and env.setpiece_kind[0] == 2
    assert not initial["events"]["corner_opportunities"].any()
    sim.player_pos[0, 0] = sim.ball_pos[0] + [14, 14]
    actions = np.zeros((1, env.P), int)
    actions[0, 0] = 9
    _, _, done, kicked = env.step(actions)
    assert not done[0]
    assert kicked["events"]["corner_opportunities"][0, 0] == 1
    assert kicked["events"]["corner_attempts"][0, 0] == 1
    assert not kicked["events"]["corner_curriculum_starts"].any()
    useful = int(kicked["events"]["corner_successes"].sum())
    for _ in range(16):
        _, _, _, info = env.step(np.zeros((1, env.P), int))
        useful += int(info["events"]["corner_successes"].sum())
        assert not info["events"]["corner_opportunities"].any()
        assert not info["events"]["corner_attempts"].any()
    assert useful == 1


def test_evaluation_factory_disables_curriculum_on_automatic_resets():
    env = environment(corner_curriculum=False, max_ticks=3)
    assert env.corner_reset_prob == 0
    env.reset()
    for _ in range(6):
        assert not (env.setpiece_kind == 2).any()
        _, _, done, info = env.step(np.zeros((env.N, env.P), int))
        assert done.all() and not info["goal"].any()
        assert not info["events"]["corner_curriculum_starts"].any()


def test_arena_never_enables_training_curriculum(monkeypatch):
    from eval import arena
    original = arena.HaxballEnv
    environments = []
    def capture(*args, **kwargs):
        env = original(*args, **kwargs)
        environments.append(env)
        return env
    monkeypatch.setattr(arena, "HaxballEnv", capture)
    def idle(env, obs, players):
        return np.zeros((env.N, len(players)), int)
    details = arena.play(idle, idle, n_games=2, minutes=.005, n_per_team=4,
                         stadium="rs_one", env_kw=dict(out_of_bounds=True, obs_layout="universal",
                                                       corner_reset_prob=1.), return_details=True)
    assert environments[0].corner_reset_prob == 0
    assert details["events_red"]["corner_curriculum_starts"] == 0
    assert details["events_blue"]["corner_curriculum_starts"] == 0


def test_training_log_accumulates_events_between_prints(tmp_path, monkeypatch, capsys):
    from train import multitask
    from train.runtime import load_config
    root = multitask.ROOT
    cfg = load_config(root / "train/config_rs4_tactical.yaml")
    cfg.pop("bc_reference", None)
    cfg["scripted_readiness"] = {"required": False}
    cfg["tasks_file"] = str(root / "train/tasks.yaml")
    cfg["env"].update(agents=16, max_ticks=3)
    cfg["model"].update(hidden=16, layers=1, ent_hidden=8, ent_layers=1)
    cfg["ppo"].update(device="cpu", rollout_len=2, epochs=1, minibatch=16,
                      torch_threads=2, numba_threads=1)
    cfg["log"].update(every=3, checkpoint_every=1000000, replay_every=0)
    # Los contadores no dependen de que quede guía táctica activa.
    cfg.pop("rs4_tactics", None)
    monkeypatch.setattr(multitask, "ROOT", tmp_path)
    trainer = multitask.MultiTrainer(cfg, "corner_test", False)
    stats = []
    original_log = trainer.log
    def record(data, *args):
        stats.append(dict(data))
        return original_log(data, *args)
    monkeypatch.setattr(trainer, "log", record)
    slot = trainer.slots[0]
    slot.env.corner_reset_prob = 1.
    try:
        for _ in range(3):
            trainer.iterate()
        output = capsys.readouterr().out
        assert output.count("RS4 córners (desde último log") == 1
        assert "oportunidades/currículo/intentos/útiles/expirados" in output
        assert not slot.rs4_restart_log_events.any()
        for event in RESTART_EVENT_NAMES:
            for color in ("red", "blue"):
                assert stats[2][f"rs4/window_{event}_{color}"] == sum(
                    row[f"rs4/{event}_{color}"] for row in stats[:3])
        assert sum(stats[2][f"rs4/window_corner_curriculum_starts_{color}"]
                   for color in ("red", "blue")) > 0
        trainer.iterate()
        for event in RESTART_EVENT_NAMES:
            for color in ("red", "blue"):
                assert stats[3][f"rs4/window_{event}_{color}"] == stats[3][f"rs4/{event}_{color}"]
    finally:
        trainer.writer.close()
