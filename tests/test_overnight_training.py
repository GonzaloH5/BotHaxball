"""Perfil nocturno: continuación, rivales con evidencia y métricas actuales."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from train import multitask
from train.multitask import MultiTrainer, TaskSlot, SCRIPTED, POOL, SELF
from train.runtime import annealing_fraction, load_config


def profile():
    return load_config(multitask.ROOT / "train/config_gpu_overnight.yaml")


def trainer_and_slot(stage=2, n=100):
    t = MultiTrainer.__new__(MultiTrainer)
    t.cfg = profile()
    t.rng = np.random.default_rng(31)
    t.league = SimpleNamespace(members=[object()], sample=lambda *args: [0])
    s = TaskSlot(SimpleNamespace(name="futsal_3v3"), SimpleNamespace(N=n, P=6, T=3))
    s.opp_stage = stage
    cst = t.cfg["curriculum"][stage]
    s.set_regression_context(cst["scripted_eps"], cst["scripted_policy"], cst["scripted_style"])
    t.slots = [s]
    return t, s


def test_profile_preserves_ppo_and_rewards_but_increases_futsal_practice():
    cfg = profile()
    original = load_config(multitask.ROOT / "train/config_runpod.yaml")
    assert cfg["env"] == original["env"]
    assert cfg["model"] == original["model"]
    assert cfg["reward"] == original["reward"]
    assert cfg["task_reward_overrides"] == original["task_reward_overrides"]
    for key, value in original["ppo"].items():
        if key not in ("numba_threads", "total_steps"):
            assert cfg["ppo"][key] == value
    assert cfg["ppo"]["total_steps"] == 5_000_000_000
    assert cfg["ppo"]["schedule_steps"] == original["ppo"]["total_steps"]
    assert cfg["ppo"]["numba_threads"] <= 10
    assert cfg["runtime"]["cuda_decisions"] == "auto"
    assert cfg["runtime"]["reuse_ppo_batch"]
    assert cfg["log"]["replay_every"] == 0
    assert cfg["schedule"]["min_games"] == 128
    for old, new in zip(original["curriculum"], cfg["curriculum"]):
        for key in ("advance_points", "scripted_policy", "scripted_eps"):
            assert old[key] == new[key]
    weights = cfg["stages"][2]["weights"]
    assert sum(weights.values()) == pytest.approx(1)
    assert sum(v for k, v in weights.items() if k.startswith("futsal")) == pytest.approx(.75)


@pytest.mark.parametrize("steps", [2_916_200_000, 3_000_000_000, 4_000_000_000])
def test_overnight_extension_keeps_annealing_and_clamps_final_values(steps):
    cfg = profile()["ppo"]
    fraction = annealing_fraction(steps, cfg)
    assert fraction == min(steps / 3_000_000_000, 1)
    for first, last in (("lr", "lr_final"), ("ent_coef", "ent_coef_final"), ("bc_kl_coef", "bc_kl_final")):
        value = cfg[first] + (cfg[last] - cfg[first]) * fraction
        assert value >= cfg[last] - 1e-12


def test_goal_metrics_refresh_without_bypassing_points_promotion():
    t, s = trainer_and_slot()
    s.wr_window = [(20, 0)]
    s.match_window = [(9, 0, 0, 20, 0, 0)]
    s.goals[SCRIPTED] = [0, 10]
    t.opponent_curriculum()
    assert s.winrate() == (2 / 3, 30)
    assert s.match_performance()[0:2] == (1, 9)
    assert s.opp_stage == 2 and s.promotion_streak == 0
    assert s.goals[SCRIPTED] == [0, 0]
    t.opponent_curriculum()
    assert s.winrate() == (2 / 3, 30)


def test_mastery_requires_128_games_and_has_regression_hysteresis():
    t, s = trainer_and_slot(3)
    s.match_window = [(127, 0, 0, 1000, 0, 0)]
    t.update_scripted_mastery(s)
    assert not s.scripted_mastered
    s.match_window = [(128, 0, 0, 1000, 0, 0)]
    t.assign_modes(s)
    assert s.scripted_mastered
    np.testing.assert_array_equal(np.bincount(s.modes, minlength=3), [25, 70, 5])
    modes, rivals = s.modes.copy(), s.opp_id.copy()
    s.match_finished[:] = False
    s.match_window = [(119, 0, 9, 100, 10, 0)]
    t.assign_modes(s)
    assert s.scripted_mastered  # .93 se mantiene; entrada .95, salida <.90
    s.match_window = [(112, 0, 16, 100, 20, 0)]
    t.assign_modes(s)
    assert not s.scripted_mastered
    # Los rivales no se cambian a mitad de partido ni siquiera ante regresión.
    np.testing.assert_array_equal(s.modes, modes)
    np.testing.assert_array_equal(s.opp_id, rivals)
    s.match_finished[:] = True
    t.assign_modes(s)
    np.testing.assert_array_equal(np.bincount(s.modes, minlength=3), [40, 45, 15])


@pytest.mark.parametrize("has_bc", [True, False])
def test_overnight_resume_keeps_checkpoint_optimizer_and_bc_reference(tmp_path, monkeypatch, has_bc):
    source = multitask.ROOT
    new = profile()
    new["tasks_file"] = str(source / "train/tasks.yaml")
    new["env"].update(agents=96, max_ticks=6)
    new["model"].update(hidden=16, layers=1, ent_hidden=8, ent_layers=1)
    new["ppo"].update(device="cpu", torch_threads=2, numba_threads=1,
                      rollout_len=2, epochs=1, minibatch=16)
    new["log"].update(every=1000, checkpoint_every=1000)
    old = copy.deepcopy(new)
    old["runtime"]["preserve_bc_reference"] = False
    old.pop("bc_reference", None)
    old["ppo"]["total_steps"] = 3_000_000_000
    old["stages"][2] = load_config(source / "train/config_multi.yaml")["stages"][2]
    monkeypatch.setattr(multitask, "ROOT", tmp_path)
    trainers = []
    try:
        t = MultiTrainer(old, "resume", False)
        trainers.append(t)
        t.stage = 2
        t.build_envs()
        t.iterate()
        ref = tmp_path / "custom_bc.pt"
        torch.save({"model_config": t.model.config(), "model": t.model.state_dict()}, ref)
        # Emular config.yaml efectivo del run con una referencia personalizada.
        import yaml
        saved = copy.deepcopy(old)
        if has_bc:
            saved["bc_reference"] = str(ref)
        (t.run_dir / "config.yaml").write_text(yaml.safe_dump(saved), encoding="utf-8")
        s = next(s for s in t.slots if s.task.name == "futsal_3v3")
        s.opp_stage = 3
        s.set_regression_context(0, "r3", -1)
        s.match_window = [(128, 0, 0, 400, 0, 0)]
        s.scripted_mastered = True
        t.steps = 3_100_000_000  # ya superó el límite anterior, debe seguir
        t.league.add_snapshot(t.model, "previous")
        t.save(t.run_dir / "latest.pt")
        state = copy.deepcopy(t.model.state_dict())
        optimizer = copy.deepcopy(t.opt.state_dict())
        resumed = MultiTrainer(new, "resume", True)
        trainers.append(resumed)
        assert resumed.steps == t.steps < resumed.cfg["ppo"]["total_steps"]
        assert resumed.cfg["bc_reference"] == (str(ref) if has_bc else None)
        assert (resumed.bc_model is not None) == has_bc
        assert resumed.league.members[0].name == "previous"
        assert resumed.league.recent_weight == .5
        assert next(s for s in resumed.slots if s.task.name == "futsal_3v3").scripted_mastered
        for key, tensor in state.items():
            torch.testing.assert_close(tensor, resumed.model.state_dict()[key], atol=0, rtol=0)
        for parameter, values in optimizer["state"].items():
            for key, tensor in values.items():
                torch.testing.assert_close(tensor, resumed.opt.state_dict()["state"][parameter][key], atol=0, rtol=0)
        resumed.iterate()
        assert resumed.steps > t.steps
        assert resumed.opt.param_groups[0]["lr"] == pytest.approx(new["ppo"]["lr_final"])
    finally:
        for trainer in trainers:
            trainer.writer.close()
