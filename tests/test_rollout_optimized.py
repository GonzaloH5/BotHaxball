"""Equivalencia de observaciones, física, bots, resets y timeouts del rollout."""
import copy

import numpy as np
import pytest
import torch
from numba import get_num_threads, set_num_threads

from bots.scripted import _scripted_reference, scripted_actions
from env.haxball_env import HaxballEnv
from env.tasks import load_catalog, make_env


@pytest.fixture(autouse=True)
def threads():
    previous = get_num_threads()
    set_num_threads(min(2, previous))
    yield
    set_num_threads(previous)


@pytest.mark.parametrize("task", list(load_catalog()))
def test_observations_match_reference_all_tasks(task):
    env = make_env(load_catalog()[task], 5, max_entities=23, seed=41)
    env.reset()
    rng = np.random.default_rng(23)
    env.sim.touch[:] = rng.random(env.sim.touch.shape) < 0.5
    env.sim.kick_cancel[:] = rng.random(env.sim.kick_cancel.shape) < 0.5
    env.ticks[:] = rng.integers(0, env.max_ticks, env.N)
    if env.sim.ps_on:
        env.sim.ps_comba[:] = True
        env.sim.ps_charge[:] = [0, 1, 31, 77, -1]
        env.sim.ps_held[:] = [0, 1, -1, 0, 1]
        env.sim.ball_grav[:] = rng.normal(0, 0.1, (env.N, 2))
        env.sim.inv_env[:, 0] += 0.5
    if env.rules is not None:
        env.rules.expelled[0, ::2] = True
        env.rules.expelled[1, 1::2] = True
    reference = env._observe_universal()
    np.testing.assert_array_equal(env.observe(), reference)
    np.testing.assert_array_equal(env.observe(np.array([3, 1, 0])), reference[[3, 1, 0]])


@pytest.mark.parametrize("task", ["big_3v3", "aha_3v3", "x1_1v1", "rf_7v7", "rs_3v3"])
@pytest.mark.parametrize("delay", [0, 5])
def test_complete_transitions_match_reference(task, delay):
    params = dict(n_envs=5, max_entities=15, seed=73, max_ticks=27,
                  kickoff_timeout=12, random_reset_prob=0.5, action_delay_max=delay)
    env = make_env(load_catalog()[task], **params)
    legacy = make_env(load_catalog()[task], **params, optimize_rollout=False)
    np.testing.assert_array_equal(env.reset(), legacy.reset())
    rng = np.random.default_rng(32)
    for step in range(50):
        # Forzar goles y salidas además de episodios truncados/reinicios normales.
        if step % 9 == 0:
            for e in (env, legacy):
                e.sim.kickoff[0] = False
                e.sim.pos[0, 0] = (e.goal_x - 1, 0)
                e.sim.vel[0, 0] = (4, 0)
                if e.out_of_bounds:
                    e.sim.pos[1, 0] = (10, e.field_h + 30)
                    e.sim.vel[1, 0] = 0
                e._phi = e._potentials()
        actions = rng.integers(0, 18, (env.N, env.P))
        actual, expected = env.step(actions), legacy.step(actions)
        for a, b in zip(actual[:3], expected[:3]):
            np.testing.assert_array_equal(a, b)
        for key in actual[3]:
            a, b = actual[3][key], expected[3][key]
            if isinstance(a, dict):
                for k in a:
                    np.testing.assert_array_equal(a[k], b[k])
            else:
                np.testing.assert_array_equal(a, b)
        for key, a in vars(env.sim).items():
            # Buffers privados de reutilización pueden existir sólo en la ruta
            # fusionada; no forman parte del estado físico observable.
            if isinstance(a, np.ndarray) and not key.startswith("_coop_"):
                np.testing.assert_array_equal(a, getattr(legacy.sim, key), err_msg=key)
        for a, b in zip(env._phi, legacy._phi):
            np.testing.assert_array_equal(a, b)
        assert env.rng.bit_generator.state == legacy.rng.bit_generator.state
        assert env.sim.rng.bit_generator.state == legacy.sim.rng.bit_generator.state


@pytest.mark.parametrize("T", [1, 2, 3, 6, 7, 11])
@pytest.mark.parametrize("eps", [0.0, 0.1, 0.5])
@pytest.mark.parametrize("policy", ["r2", "r3"])
def test_selected_bots_keep_actions_and_random_state(T, eps, policy):
    env = HaxballEnv(17, T, "big", seed=13, random_reset_prob=1.0)
    env.reset()
    players, rows = np.arange(T, 2 * T), np.array([0, 4, 11, 16])
    a, b = np.random.default_rng(61), np.random.default_rng(61)
    for _ in range(12):
        env.sim.reset_random(np.arange(env.N))
        reference = _scripted_reference(env, players, eps, a, policy=policy)
        result = scripted_actions(env, players, eps, b, env_indices=rows, policy=policy)
        np.testing.assert_array_equal(result, reference[rows])
        assert a.bit_generator.state == b.bit_generator.state


def test_fused_physics_events_across_ticks():
    env = HaxballEnv(3, 1, "classic", seed=2)
    sim = env.sim
    sim.reset_random(np.arange(3))
    sim.pos[0, 0] = (0, 0)
    sim.pos[0, sim.first_player] = (-24, 0)
    sim.vel[0] = 0
    sim.pos[1, 0], sim.vel[1, 0] = (env.goal_x - 1, 0), (4, 0)
    old = copy.deepcopy(sim)
    actions = np.array([[13, 0], [0, 0], [9, 4]])
    expected_goal = np.zeros(3, dtype=np.int64)
    expected_kicked = np.zeros((3, 2), dtype=bool)
    touches = np.full(3, -1)
    for _ in range(3):
        g = old.step(actions)
        expected_goal = np.where(expected_goal == 0, g, expected_goal)
        expected_kicked |= old.kicked
        for t in (0, 1):
            touches[old.touch[:, old.player_team == t].any(axis=1)] = t
    last_touch = np.full(3, -1)
    goal, kicked = sim.step_frames(actions, 3, last_touch)
    np.testing.assert_array_equal(goal, expected_goal)
    np.testing.assert_array_equal(kicked, expected_kicked)
    np.testing.assert_array_equal(last_touch, touches)
    assert kicked[0, 0]  # patada en primer tick, no en el último: no debe perderse
    assert goal[1] == 1


def test_cpu_ppo_keeps_training_trajectory(tmp_path, monkeypatch):
    from train import multitask
    from train.runtime import load_config
    cfg = load_config(multitask.ROOT / "train/config_multi.yaml")
    cfg["tasks_file"] = str(multitask.ROOT / "train/tasks.yaml")
    cfg["stages"] = [{"tasks": ["big_3v3", "x1_1v1"], "min_steps": 0, "max_steps": 100000}]
    cfg["env"].update(agents=24, max_ticks=6)
    cfg["model"].update(hidden=16, ent_hidden=8)
    cfg["ppo"].update(device="cpu", rollout_len=4, epochs=1, minibatch=16, torch_threads=2, numba_threads=2)
    cfg["log"].update(every=100, checkpoint_every=100, replay_every=0)
    cfg["curriculum"][0].update(scripted=0.25, selfplay=0.25, pool=0.5)
    monkeypatch.setattr(multitask, "ROOT", tmp_path)
    old_cfg = copy.deepcopy(cfg)
    old_cfg["runtime"] = {"optimize_rollout": False}
    fast = multitask.MultiTrainer(cfg, "fast", False)
    old = multitask.MultiTrainer(old_cfg, "reference", False)
    try:
        for trainer in (fast, old):
            trainer.league.add_snapshot(trainer.model, "fixed")
        for iteration in range(3):
            for trainer in (fast, old):
                torch.manual_seed(1700 + iteration)
                trainer.iterate()
            assert fast.steps == old.steps
            assert fast.rng.bit_generator.state == old.rng.bit_generator.state
            for key, tensor in fast.model.state_dict().items():
                torch.testing.assert_close(tensor, old.model.state_dict()[key], rtol=0, atol=0)
            for a, b in zip(fast._obs, old._obs):
                np.testing.assert_array_equal(a, b)
    finally:
        fast.writer.close()
        old.writer.close()
