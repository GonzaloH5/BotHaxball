"""Small CPU integration, exact useful budgets and dynamic controller masks."""
import copy

import numpy as np
import pytest
import torch
import yaml

from train import multitask
from train.multitask import MultiTrainer
from train.rs4_migration import migrate_checkpoint
from train.rs4_program import ProgramState, default_program
from train.rs4_trainer import RS4V3Trainer, generalized_advantage, sequence_arrays, useful_mask
from train.runtime import load_config


def test_exact_budget_never_counts_frozen_or_padding():
    original = np.array([[[1, 0, 1]], [[0, 1, 1]]], bool)
    got = useful_mask(original, 3)
    assert got.sum() == 3
    assert not (got & ~original).any()
    np.testing.assert_array_equal(original, [[[1, 0, 1]], [[0, 1, 1]]])


def test_credit_stops_when_controller_changes():
    rewards = np.array([[[1., 0.]], [[100., 0.]]], np.float32)
    values = np.zeros_like(rewards)
    learner = np.array([[[1, 1]], [[0, 1]]], bool)
    got = generalized_advantage(rewards, values, np.zeros((2, 1), bool), np.zeros((1, 2)), .998, .97, learner)
    assert got[0, 0, 0] == 1


def test_sequences_preserve_players_and_keep_only_valid():
    acts = np.arange(16).reshape(4, 2, 2)
    b = {key: acts.copy() for key in ("act", "logp", "adv", "ret", "previous_action", "episode_start")}
    b["obs"] = acts[..., None].copy()
    b["valid"] = np.zeros_like(acts, bool)
    b["valid"][0, 0, 0] = True
    b["valid"][3, 1, 1] = True
    packed, selected = sequence_arrays(b, 2)
    assert packed["valid"].sum() == 2
    np.testing.assert_array_equal(packed["act"], [[0, 11], [4, 15]])
    assert selected.sum() == 2


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    repository = multitask.ROOT
    cfg = load_config(repository / "train/config_rs4_v3.yaml")
    cfg.pop("rs4_program")
    cfg.pop("bc_reference", None)
    cfg["scripted_readiness"] = {"required": False}
    cfg["tasks_file"] = str(repository / "train/tasks.yaml")
    cfg["env"].update(agents=16, max_ticks=12)
    cfg["model"].update(hidden=16, layers=1, ent_hidden=8, ent_layers=1)
    cfg["ppo"].update(device="cpu", rollout_len=4, sequence_length=2, minibatch=16, epochs=1,
                      torch_threads=2, numba_threads=1, total_steps=100000)
    cfg["log"].update(every=1000000, replay_every=0)
    monkeypatch.setattr(multitask, "ROOT", tmp_path)
    source = MultiTrainer(copy.deepcopy(cfg), "source", False)
    source.iterate()
    source.save(source.run_dir / "latest.pt")
    source.writer.close()
    ck = torch.load(source.run_dir / "latest.pt", map_location="cpu", weights_only=False)
    return tmp_path, cfg, ck


@pytest.mark.parametrize("memory", [False, True])
def test_mixed_training_and_resume_keep_program_norms_and_budget(prepared, memory):
    root, cfg, ck = prepared
    cfg["rs4_program"] = default_program(ck["steps"])
    cfg["rs4_program"]["phases"][0]["teammates"] = 1.
    if memory:
        cfg["model"].update(type="recurrent_set", memory_size=32, pooling="attentive_meanmax")
    else:
        cfg["model"]["type"] = "set"
    cfg["ppo"]["total_steps"] = ck["steps"] + 13
    migrated = migrate_checkpoint(ck, cfg)
    migrated["rs4_program_state"] = ProgramState.from_config(cfg).state_dict()
    destination = root / "runs" / "v3"
    destination.mkdir()
    torch.save(migrated, destination / "latest.pt")
    (destination / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    trainer = RS4V3Trainer(cfg, "v3", True)
    before = {key: value.clone() for key, value in trainer.model.state_dict().items() if "norm" in key}
    recorded = {}
    original_log = trainer.log
    def capture(stats, *args):
        recorded.update(stats)
        original_log(stats, *args)
    trainer.log = capture
    trainer.iterate()
    assert trainer.program.relative_steps == 13
    assert trainer.steps == ck["steps"] + 13
    assert trainer.remaining_steps == 0
    assert recorded["program/segment_remaining"] == 0
    assert recorded["program/budget_remaining"] == trainer.program.remaining_steps > 0
    assert recorded["rollout/learner_samples"] == 13
    assert recorded["rollout/simulated_samples"] == cfg["ppo"]["rollout_len"] * trainer.total_rows
    assert recorded["rollout/learner_fraction"] == 13 / recorded["rollout/simulated_samples"]
    for key, value in before.items():
        torch.testing.assert_close(trainer.model.state_dict()[key], value, atol=0, rtol=0)
    assert len(trainer.inference.controllers) <= 4
    trainer.save(destination / "latest.pt")
    trainer.writer.close()
    resumed = RS4V3Trainer(copy.deepcopy(cfg), "v3", True)
    assert resumed.program.state_dict() == trainer.program.state_dict()
    assert resumed.steps == trainer.steps
    resumed.writer.close()


def test_frozen_rows_cannot_change_ppo_loss(prepared):
    root, cfg, ck = prepared
    cfg["rs4_program"] = default_program(ck["steps"])
    cfg["model"].update(type="recurrent_set", memory_size=32, pooling="attentive_meanmax")
    cfg["ppo"]["lr"] = 0.
    trainer = object.__new__(RS4V3Trainer)
    trainer.model = __import__("train.model", fromlist=["build_model"]).build_model(migrate_checkpoint(ck, cfg)["model_config"])
    trainer.device = torch.device("cpu")
    trainer.opt = torch.optim.Adam(trainer.model.parameters(), lr=0.)
    trainer.cfg, trainer.program, trainer.bc_coef, trainer.obs_dim = cfg, ProgramState.from_config(cfg), 0., 127
    b = dict(obs=torch.randn(2, 2, 127), initial_memory=torch.zeros(2, 32), previous_action=torch.full((2, 2), 18),
             episode_start=torch.ones(2, 2, dtype=torch.bool), act=torch.zeros(2, 2, dtype=torch.long),
             adv=torch.ones(2, 2), ret=torch.zeros(2, 2), logp=torch.zeros(2, 2),
             valid=torch.tensor([[1, 0], [1, 0]], dtype=torch.bool))
    # A completely invalid sequence is not normally packed, but even a partially
    # invalid sequence must not contribute its actions/rewards to any PPO term.
    b["valid"][0, 1] = True
    first = trainer.update(b, .003)
    b["ret"][1, 1] = 1e20
    b["adv"][1, 1] = -1e20
    second = trainer.update(b, .003)
    assert first["v_loss"] == pytest.approx(second["v_loss"])
