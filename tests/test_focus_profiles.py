"""Fases manuales 50%/20% de 1v1, sin reiniciar checkpoint ni modificar PPO."""
import copy
from types import SimpleNamespace

import pytest
import torch

from train import multitask
from train.multitask import MultiTrainer
from train.runtime import load_config


PROFILES = [
    ("config_1v1_focus.yaml", 0.5, 0.5 / 6),
    ("config_cooperation.yaml", 0.2, 0.8 / 6),
]


def test_balanced_profile_gives_every_stage_a_task_the_same_share():
    cfg = load_config(multitask.ROOT / "train/config_balanced.yaml")
    trainer = MultiTrainer.__new__(MultiTrainer)
    trainer.cfg, trainer.stages, trainer.stage = cfg, cfg["stages"], 0
    trainer.task_state = {n: {"boost": 9.0} for n in trainer.active_tasks()}
    shares = trainer.task_shares()
    assert set(shares) == set(cfg["stages"][0]["tasks"])
    assert all(value == pytest.approx(1 / len(shares)) for value in shares.values())


@pytest.mark.parametrize("profile,one,other", PROFILES)
def test_fixed_profiles_inherit_training_and_ignore_saved_boosts(profile, one, other):
    base = load_config(multitask.ROOT / "train/config_runpod.yaml")
    cfg = load_config(multitask.ROOT / "train" / profile)
    for key in ("ppo", "runtime", "model", "env", "reward", "curriculum", "league", "schedule", "bc_reference"):
        assert cfg[key] == base[key]
    assert cfg["stages"][0]["tasks"] == base["stages"][0]["tasks"]
    assert len(cfg["stages"]) == 1
    trainer = MultiTrainer.__new__(MultiTrainer)
    trainer.cfg, trainer.stages, trainer.stage = cfg, cfg["stages"], 0
    trainer.task_state = {n: {"boost": 4.0 if n != "x1_1v1" else 2.0} for n in trainer.active_tasks()}
    shares = trainer.task_shares()
    assert sum(shares.values()) == pytest.approx(1)
    for name, value in shares.items():
        assert value == pytest.approx(one if name == "x1_1v1" else other)
    trainer.stages[0]["fixed_weights"] = False
    assert trainer.task_shares()["x1_1v1"] != pytest.approx(one)  # ruta adaptativa original intacta


@pytest.mark.parametrize("profile,one,other", PROFILES)
def test_fixed_profile_does_not_rebalance_or_advance_automatically(profile, one, other):
    cfg = load_config(multitask.ROOT / "train" / profile)
    trainer = MultiTrainer.__new__(MultiTrainer)
    trainer.cfg, trainer.stages, trainer.stage = cfg, cfg["stages"], 0
    trainer.stage_steps, trainer.iteration = 10**12, 25
    trainer.slots = [SimpleNamespace(opp_stage=2, boost=3.0, best_wr=0.95,
                                     winrate=lambda: (0.1, 200))]
    # Sin save/build_envs: llamar a cualquiera sería un error en este objeto mínimo.
    trainer.maybe_rebalance_or_advance()
    assert trainer.stage == 0 and trainer.slots[0].boost == 3.0


@pytest.mark.parametrize("stage", [-1, 1])
def test_profile_rejects_incompatible_checkpoint_stage_before_loading_weights(tmp_path, stage):
    checkpoint = tmp_path / "incompatible.pt"
    torch.save({"model_config": {"rule_observation": "masked"}, "stage": stage}, checkpoint)
    trainer = MultiTrainer.__new__(MultiTrainer)
    trainer.model = SimpleNamespace(rule_observation="masked")
    trainer.stages = [{"tasks": ["x1_1v1"]}]
    with pytest.raises(ValueError, match="configuración compatible"):
        trainer.load(checkpoint)


def test_resume_focus_then_cooperation_keeps_model_league_steps_and_metrics(tmp_path, monkeypatch):
    source = multitask.ROOT
    base = load_config(source / "train/config_multi.yaml")
    profiles = [load_config(source / "train" / name) for name, _, _ in PROFILES]
    def small_config(cfg):
        cfg = copy.deepcopy(cfg)
        cfg["tasks_file"] = str(source / "train/tasks.yaml")
        cfg.pop("bc_reference", None)
        cfg["model"].update(hidden=16, ent_hidden=8, layers=1, ent_layers=1)
        cfg["ppo"].update(device="cpu", torch_threads=2, numba_threads=1,
                          rollout_len=2, epochs=1, minibatch=512)
        cfg["log"].update(every=100, checkpoint_every=100, replay_every=0)
        return cfg
    monkeypatch.setattr(multitask, "ROOT", tmp_path)
    trainers = []
    try:
        initial = MultiTrainer(small_config(base), "saved", False)
        trainers.append(initial)
        initial.iterate()  # checkpoint con momentos Adam y normalizadores realmente actualizados
        initial.steps, initial.stage_steps, initial.iteration = 176_900_000, 176_900_000, 1720
        initial.league.add_snapshot(initial.model, "existing")
        for slot in initial.slots:
            slot.opp_stage, slot.boost = 2, 3.0
            slot.set_regression_context(0.0)
            slot.boost, slot.best_wr, slot.wr_window = 3.0, 0.9, [(140, 60)]
        initial.save(initial.run_dir / "latest.pt")
        weights = {name: tensor.clone() for name, tensor in initial.model.state_dict().items()}
        optimizer = copy.deepcopy(initial.opt.state_dict())
        assert optimizer["state"]
        for cfg, (_, one, other) in zip(profiles, PROFILES):
            resumed = MultiTrainer(small_config(cfg), "saved", True)
            trainers.append(resumed)
            assert (resumed.steps, resumed.stage_steps, resumed.iteration) == (176_900_000, 176_900_000, 1720)
            assert resumed.league.members[0].name == "existing"
            for name, tensor in resumed.model.state_dict().items():
                torch.testing.assert_close(tensor, weights[name], rtol=0, atol=0)
            restored_opt = resumed.opt.state_dict()
            assert restored_opt["param_groups"] == optimizer["param_groups"]
            for param, state in optimizer["state"].items():
                for key, value in state.items():
                    torch.testing.assert_close(restored_opt["state"][param][key], value, rtol=0, atol=0)
            total = sum(s.N * s.P for s in resumed.slots)
            for slot in resumed.slots:
                expected = one if slot.task.name == "x1_1v1" else other
                assert slot.N * slot.P / total == pytest.approx(expected, abs=0.005)
                assert slot.opp_stage == 2 and slot.boost == 3.0
                assert slot.best_wr == 0.9 and slot.wr_window == [(140, 60)]
                assert slot.intro_step == 0  # no reiniciar shaping por cambiar de nombre de fase
            if one == 0.5:
                assert next(s for s in resumed.slots if s.task.name == "x1_1v1").N == 288
            resumed.save(resumed.run_dir / "latest.pt")
        # Los perfiles sólo afectan este run temporal; el reparto original sigue adaptativo.
    finally:
        for trainer in trainers:
            trainer.writer.close()
