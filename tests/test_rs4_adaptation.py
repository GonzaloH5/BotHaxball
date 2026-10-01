"""Fase LR aislada: fuente intacta, Adam heredado y calendarios independientes."""
import copy
import json

import pytest
import torch
import yaml

from tools import prepare_rs4_adaptation as adaptation
from train import multitask
from train.multitask import MultiTrainer
from train.ppo_selfplay import lerp
from train.runtime import annealing_fraction, learning_rate, load_config
from train.rs4_specialization import coefficient


def assert_same(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_same(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for left, right in zip(a, b):
            assert_same(left, right)
    else:
        assert a == b


@pytest.fixture
def source(tmp_path, monkeypatch):
    repository = multitask.ROOT
    cfg = load_config(repository / "train/config_rs4_tactical.yaml")
    cfg.pop("bc_reference", None)
    cfg["scripted_readiness"] = {"required": False}
    cfg["tasks_file"] = str(repository / "train/tasks.yaml")
    cfg["env"].update(agents=16, max_ticks=12)
    cfg["model"].update(hidden=16, layers=1, ent_hidden=8, ent_layers=1)
    cfg["ppo"].update(device="cpu", rollout_len=2, minibatch=16, epochs=1,
                      torch_threads=2, numba_threads=1, schedule_steps=3_000_000_000)
    cfg["log"].update(every=1000000, replay_every=0)
    cfg["rs4_tactics"].update(coef=.08, start_steps=4_250_000_000)
    monkeypatch.setattr(multitask, "ROOT", tmp_path)
    monkeypatch.setattr(adaptation, "ROOT", tmp_path)
    trainer = MultiTrainer(cfg, "rs4", False)
    try:
        trainer.steps = 4_300_000_000
        trainer.slots[0].opp_stage = 3
        trainer.slots[0].set_regression_context(0., "r3", -1)
        trainer.iterate()  # Adam tiene momentos, no sólo pesos iniciales
        teacher = tmp_path / "bc_rs4.pt"
        torch.save(dict(model=trainer.model.state_dict(), model_config=trainer.model.config()), teacher)
        cfg["bc_reference"] = str(teacher)
        (trainer.run_dir / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
        trainer.save(trainer.run_dir / "latest.pt")
    finally:
        trainer.writer.close()
    return trainer.run_dir / "latest.pt", cfg, teacher


@pytest.mark.parametrize("steps", [0, 2_900_000_000, 3_000_000_000, 4_400_000_000])
def test_legacy_lr_is_unchanged(steps):
    p = dict(lr=.0005, lr_final=.00005, total_steps=5_000_000_000, schedule_steps=3_000_000_000)
    expected = p["lr"] + (p["lr_final"] - p["lr"]) * annealing_fraction(steps, p)
    assert learning_rate(steps, p) == expected


def test_fixed_anchor_clamps_and_budget_extension_does_not_restart_phase():
    p = dict(lr=.0005, lr_final=.00005, total_steps=4_500_000_000, schedule_steps=3_000_000_000,
             lr_schedule=dict(start_steps=4_300_000_000, duration_steps=200_000_000,
                              initial=.0001, final=.00005))
    for elapsed, expected in ((-1, .0001), (0, .0001), (100_000_000, .000075),
                              (200_000_000, .00005), (500_000_000, .00005)):
        assert learning_rate(4_300_000_000 + elapsed, p) == pytest.approx(expected)
        assert annealing_fraction(4_300_000_000 + elapsed, p) == 1
    before = learning_rate(4_400_000_000, p)
    p["total_steps"] += 500_000_000
    assert learning_rate(4_400_000_000, p) == before


@pytest.mark.parametrize("key,value", [("start_steps", -1), ("start_steps", float("nan")),
    ("duration_steps", 0), ("duration_steps", float("inf")), ("initial", 0),
    ("initial", True), ("final", -.1), ("final", .01), ("final", "0.00005")])
def test_invalid_lr_phase_is_rejected(key, value):
    phase = dict(start_steps=0, duration_steps=100, initial=.0001, final=.00005)
    phase[key] = value
    with pytest.raises(ValueError):
        learning_rate(0, dict(lr_schedule=phase))


@pytest.mark.parametrize("setting", [{}, 3, "phase"])
def test_incomplete_or_wrong_type_phase_is_rejected(setting):
    with pytest.raises(ValueError):
        learning_rate(0, dict(lr_schedule=setting))


def test_prepare_is_byte_exact_and_changes_only_lr_and_budget(source):
    path, cfg, teacher = source
    before = {p: p.read_bytes() for p in (path, path.with_name("config.yaml"), teacher)}
    destination = adaptation.prepare(path)
    for p, data in before.items():
        assert p.read_bytes() == data
    assert (destination / "parent.pt").read_bytes() == (destination / "latest.pt").read_bytes() == before[path]
    assert (destination / "parent_config.yaml").read_bytes() == before[path.with_name("config.yaml")]
    checkpoint = torch.load(path, weights_only=False)
    new = load_config(destination / "config.yaml")
    expected = copy.deepcopy(cfg)
    expected["run_name"] = "rs4_adapt"
    expected["ppo"]["total_steps"] = checkpoint["steps"] + 200_000_000
    expected["ppo"]["lr_schedule"] = dict(start_steps=checkpoint["steps"], duration_steps=200_000_000,
                                          initial=.0001, final=learning_rate(checkpoint["steps"], cfg["ppo"]))
    assert new == expected
    assert coefficient(new, checkpoint["steps"]) == coefficient(cfg, checkpoint["steps"])
    manifest = json.loads((destination / "specialization.json").read_text())
    assert manifest["kind"] == "rs4_lr_adaptation" and manifest["reference_label"] == "RS4 anterior"
    assert manifest["source_sha256"] == adaptation.file_hash(path)
    assert manifest["source_steps"] == checkpoint["steps"]


def test_resume_inherits_optimizer_and_phase_does_not_reset(source):
    path, cfg, _ = source
    destination = adaptation.prepare(path)
    new = load_config(destination / "config.yaml")
    before = torch.load(path, weights_only=False)
    trainer = MultiTrainer(new, "rs4_adapt", True)
    try:
        assert trainer.steps == before["steps"]
        assert_same(trainer.model.state_dict(), before["model"])
        assert_same(trainer.opt.state_dict(), before["opt"])
        trainer.iterate()
        assert trainer.opt.param_groups[0]["lr"] == .0001
        assert trainer.bc_coef == lerp(cfg["ppo"]["bc_kl_coef"], cfg["ppo"]["bc_kl_final"], 1)
        halfway = new["ppo"]["lr_schedule"]["start_steps"] + 100_000_000
        trainer.steps = halfway
        trainer.save(destination / "latest.pt")
    finally:
        trainer.writer.close()
    resumed = MultiTrainer(load_config(destination / "config.yaml"), "rs4_adapt", True)
    try:
        assert resumed.steps == halfway
        resumed.iterate()
        assert resumed.opt.param_groups[0]["lr"] == pytest.approx(.000075)
        assert resumed.bc_coef == lerp(cfg["ppo"]["bc_kl_coef"], cfg["ppo"]["bc_kl_final"], 1)
        assert resumed.cfg["rs4_tactics"] == cfg["rs4_tactics"]
    finally:
        resumed.writer.close()
    assert_same(torch.load(path, weights_only=False), before)


def test_repeated_preparation_and_reanchoring_are_rejected(source):
    path, _, _ = source
    destination = adaptation.prepare(path)
    before = {p: p.read_bytes() for p in destination.iterdir()}
    with pytest.raises(FileExistsError):
        adaptation.prepare(path)
    assert all(p.read_bytes() == data for p, data in before.items())
    with pytest.raises(ValueError, match="ya tiene fase"):
        adaptation.prepare(destination / "latest.pt", "another_phase")
    assert not (adaptation.ROOT / "runs/another_phase").exists()


def test_phase_cannot_start_fresh_or_resume_missing_or_older_checkpoint(source):
    path, _, _ = source
    destination = adaptation.prepare(path)
    cfg = load_config(destination / "config.yaml")
    before = (destination / "config.yaml").read_bytes()
    with pytest.raises(ValueError, match="requiere --resume"):
        MultiTrainer(cfg, "rs4_adapt", False)
    with pytest.raises(ValueError, match="latest.pt"):
        MultiTrainer(cfg, "missing_adaptation", True)
    assert not (adaptation.ROOT / "runs/missing_adaptation").exists()
    older = torch.load(destination / "latest.pt", weights_only=False)
    older["steps"] = cfg["ppo"]["lr_schedule"]["start_steps"] - 1
    torch.save(older, destination / "latest.pt")
    with pytest.raises(ValueError, match="anterior al ancla"):
        MultiTrainer(cfg, "rs4_adapt", True)
    assert (destination / "config.yaml").read_bytes() == before


@pytest.mark.parametrize("run", ["../oops", "rs4", "", "C:/oops", ".."])
def test_invalid_or_existing_run_is_not_overwritten(source, run):
    path, _, _ = source
    original = path.read_bytes()
    with pytest.raises((ValueError, FileExistsError)):
        adaptation.prepare(path, run)
    assert path.read_bytes() == original


def test_generalist_and_missing_bc_are_rejected(source):
    path, cfg, _ = source
    config = path.with_name("config.yaml")
    cfg["stages"][0]["tasks"].append("futsal_3v3")
    config.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="exclusivamente"):
        adaptation.prepare(path)
    cfg["stages"][0]["tasks"].pop()
    cfg["bc_reference"] = "missing_bc.pt"
    config.write_text(yaml.safe_dump(cfg))
    with pytest.raises(FileNotFoundError, match="referencia BC"):
        adaptation.prepare(path)
    assert not (adaptation.ROOT / "runs/rs4_adapt").exists()
