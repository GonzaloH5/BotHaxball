"""Preparación aislada de RS4 y evaluación pareada sobre referencias congeladas."""
import json
from pathlib import Path

import pytest
import torch
import yaml

from tools import configure_rs4_rewards, evaluate_rs4, prepare_rs4
from train import multitask
from train.multitask import MultiTrainer
from train.runtime import annealing_fraction, load_config


def assert_same(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_same(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_same(a, b)
    else:
        assert left == right


@pytest.fixture
def source_run(tmp_path, monkeypatch):
    original = multitask.ROOT
    cfg = load_config(original / "train/config_gpu_overnight.yaml")
    cfg.pop("bc_reference", None)
    cfg["scripted_readiness"] = {"required": False}
    cfg["tasks_file"] = str(original / "train/tasks.yaml")
    cfg["env"].update(agents=24, max_ticks=12)
    cfg["model"].update(hidden=16, layers=1, ent_hidden=8, ent_layers=1)
    cfg["ppo"].update(device="cpu", rollout_len=2, minibatch=16, epochs=1, torch_threads=2, numba_threads=1)
    cfg["log"].update(every=1000000, checkpoint_every=1000000, replay_every=0)
    monkeypatch.setattr(multitask, "ROOT", tmp_path)
    monkeypatch.setattr(prepare_rs4, "ROOT", tmp_path)
    monkeypatch.setattr(evaluate_rs4, "ROOT", tmp_path)
    monkeypatch.setattr(configure_rs4_rewards, "ROOT", tmp_path)
    trainer = MultiTrainer(cfg, "multi", False)
    try:
        trainer.stage = 2
        trainer.build_envs()
        trainer.iterate()
        trainer.steps, trainer.iteration, trainer.stage_steps = 3_761_018_624, 36562, 900000
        trainer.league.add_snapshot(trainer.model, "frozen_opponent")
        rs4 = next(slot for slot in trainer.slots if slot.task.name == prepare_rs4.TASK)
        rs4.opp_stage = 3
        rs4.set_regression_context(0.0, "r3", -1)
        rs4.match_window = [(69, 54, 7, 90, 10, 0)]
        trainer.save(trainer.run_dir / "latest.pt")
    finally:
        trainer.writer.close()
    return trainer.run_dir / "latest.pt", cfg


def test_prepare_preserves_learning_state_and_resume_is_rs4_only(source_run):
    source, cfg = source_run
    before, config_before = source.read_bytes(), source.with_name("config.yaml").read_bytes()
    old = torch.load(source, weights_only=False)
    branch = prepare_rs4.prepare(source, additional_steps=123456)
    assert source.read_bytes() == before and source.with_name("config.yaml").read_bytes() == config_before
    assert (branch / "parent.pt").read_bytes() == before
    assert (branch / "parent_config.yaml").read_bytes() == config_before
    manifest = json.loads((branch / "specialization.json").read_text(encoding="utf-8"))
    assert manifest["source_sha256"] == prepare_rs4.file_hash(source)
    assert manifest["source_stage"] == 2 and manifest["source_steps"] == old["steps"]
    new = torch.load(branch / "latest.pt", weights_only=False)
    for key in old:
        if key not in ("stage", "stage_steps", "env"):
            assert_same(old[key], new[key])
    assert new["stage"] == new["stage_steps"] == 0
    assert new["env"]["tasks"] == [prepare_rs4.TASK]
    branch_cfg = load_config(branch / "config.yaml")
    for key in ("model", "reward", "curriculum", "league", "env", "schedule"):
        assert_same(cfg[key], branch_cfg[key])
    assert branch_cfg["ppo"]["total_steps"] == old["steps"] + 123456
    for key in cfg["ppo"]:
        if key != "total_steps":
            assert branch_cfg["ppo"][key] == cfg["ppo"][key]
    assert annealing_fraction(old["steps"], cfg["ppo"]) == annealing_fraction(old["steps"], branch_cfg["ppo"])
    resumed = MultiTrainer(branch_cfg, "rs4", True)
    try:
        assert len(resumed.slots) == 1 and resumed.slots[0].task.name == prepare_rs4.TASK
        assert resumed.task_shares() == {prepare_rs4.TASK: 1.0}
        assert resumed.slots[0].opp_stage == 3
        assert resumed.slots[0].match_performance()[0] == pytest.approx((69 + 27) / 130)
        assert_same(resumed.model.state_dict(), old["model"])
        assert_same(resumed.opt.state_dict(), old["opt"])
        assert resumed.league.members[0].name == "frozen_opponent"
        resumed.iterate()
        assert resumed.steps > old["steps"]
        resumed.save(branch / "latest.pt")
    finally:
        resumed.writer.close()
    assert source.read_bytes() == before
    assert (branch / "parent.pt").read_bytes() == before


def test_existing_branch_is_never_overwritten(source_run):
    source, _ = source_run
    branch = prepare_rs4.prepare(source)
    before = {path.name: path.read_bytes() for path in branch.iterdir()}
    with pytest.raises(FileExistsError):
        prepare_rs4.prepare(source)
    assert before == {path.name: path.read_bytes() for path in branch.iterdir()}


def test_extending_budget_does_not_restart_legacy_annealing(source_run):
    source, _ = source_run
    saved = load_config(source.with_name("config.yaml"))
    saved["ppo"].pop("schedule_steps", None)
    source.with_name("config.yaml").write_text(yaml.safe_dump(saved), encoding="utf-8")
    branch = prepare_rs4.prepare(source)
    cfg = load_config(branch / "config.yaml")
    steps = torch.load(source, weights_only=False)["steps"]
    assert cfg["ppo"]["schedule_steps"] == saved["ppo"]["total_steps"]
    assert annealing_fraction(steps, cfg["ppo"]) == annealing_fraction(steps, saved["ppo"])


@pytest.mark.parametrize("missing", ["opt", "league", "rs4_state"])
def test_incomplete_checkpoint_is_rejected(source_run, missing):
    source, _ = source_run
    checkpoint = torch.load(source, weights_only=False)
    if missing == "rs4_state":
        checkpoint["task_state"].pop(prepare_rs4.TASK)
    else:
        checkpoint.pop(missing)
    torch.save(checkpoint, source)
    with pytest.raises(ValueError):
        prepare_rs4.prepare(source)
    assert not (source.parent.parent / "rs4").exists()


@pytest.mark.parametrize("run,steps", [("../multi", 10), ("C:\\multi", 10), ("rs4", 0), ("rs4", -1)])
def test_invalid_target_or_budget_creates_nothing(tmp_path, monkeypatch, run, steps):
    monkeypatch.setattr(prepare_rs4, "ROOT", tmp_path)
    with pytest.raises(ValueError):
        prepare_rs4.prepare(tmp_path / "missing.pt", run, steps)
    assert not (tmp_path / "runs").exists()


def test_missing_source_or_config_never_initializes_random_model(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare_rs4, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="config.yaml"):
        prepare_rs4.prepare(tmp_path / "missing.pt")
    assert not (tmp_path / "runs").exists()


def test_paired_evaluation_freezes_candidates_and_same_opponents(source_run, monkeypatch):
    source, _ = source_run
    branch = prepare_rs4.prepare(source)
    hashes = {name: prepare_rs4.file_hash(branch / name) for name in ("parent.pt", "latest.pt")}
    calls = []
    def fake_eval(command, cwd, env, check):
        assert cwd == evaluate_rs4.ROOT and check and env["NUMBA_NUM_THREADS"] == "2"
        assert command[command.index("--tasks") + 1] == prepare_rs4.TASK
        assert command[command.index("--seed") + 1] == "51"
        assert command[command.index("--games") + 1] == "8"
        candidate, rival = Path(command[3]), command[command.index("--vs") + 1]
        assert candidate.is_file() and candidate not in (source, branch / "latest.pt", branch / "parent.pt")
        calls.append((candidate, rival))
        out = Path(command[command.index("--out") + 1])
        out.write_text(json.dumps({"runs": [{"tasks": {prepare_rs4.TASK: {"points": 0.5}}}]}))
    monkeypatch.setattr(evaluate_rs4.subprocess, "run", fake_eval)
    output, summary = evaluate_rs4.evaluate(games=8)
    assert len(calls) == 4
    assert calls[0][1] == calls[1][1] == "scripted:r3"
    assert calls[2][1] == calls[3][1]
    assert len(summary["results"]) == 2
    assert summary["generalist_sha256"] == hashes["parent.pt"]
    assert summary["specialist_sha256"] == hashes["latest.pt"]
    assert (output / "summary.json").exists()
    assert all(not path.exists() for path, _ in calls)
    for name, digest in hashes.items():
        assert prepare_rs4.file_hash(branch / name) == digest


def test_changed_reference_is_rejected(source_run):
    source, _ = source_run
    branch = prepare_rs4.prepare(source)
    (branch / "parent.pt").write_bytes(b"different reference")
    with pytest.raises(ValueError, match="referencia"):
        evaluate_rs4.evaluate(games=8)
    assert not (branch / "evaluations").exists()


def test_evaluation_subprocess_smoke_with_real_models(source_run, monkeypatch):
    source, _ = source_run
    prepare_rs4.prepare(source)
    repository = prepare_rs4.PROFILE.parent.parent
    actual_run = evaluate_rs4.subprocess.run
    def run_in_repository(command, **kwargs):
        # Sólo redirigir cwd: fixture almacena runs en tmp, módulos siguen en repo.
        kwargs["cwd"] = repository
        return actual_run(command, **kwargs)
    monkeypatch.setattr(evaluate_rs4.subprocess, "run", run_in_repository)
    output, summary = evaluate_rs4.evaluate(games=2, minutes=0.001)
    assert (output / "summary.json").exists()
    assert len(summary["results"]) == 2
    for row in summary["results"]:
        assert row["generalist"]["games"] == row["specialist"]["games"] == 2
        assert row["generalist"] == row["specialist"]
        assert row["points_delta"] == 0


def test_configure_tactics_is_opt_in_preserves_ppo_and_checkpoints(source_run):
    source, _ = source_run
    branch = prepare_rs4.prepare(source)
    before = load_config(branch / "config.yaml")
    original_bytes = (branch / "config.yaml").read_bytes()
    hashes = {name: prepare_rs4.file_hash(branch / name) for name in ("latest.pt", "parent.pt")}
    backup = configure_rs4_rewards.configure()
    assert backup.read_bytes() == original_bytes
    cfg = load_config(branch / "config.yaml")
    assert cfg["rs4_tactics"]["start_steps"] == 3_761_018_624
    for key in ("ppo", "model", "league", "env", "curriculum", "stages", "schedule"):
        assert_same(cfg[key], before[key])
    for name, digest in hashes.items():
        assert prepare_rs4.file_hash(branch / name) == digest
    resumed = MultiTrainer(cfg, "rs4", True)
    try:
        assert resumed.slots[0].env.rcfg.rs4_tactical_coef == .12
        assert resumed.slots[0].env.rcfg.rs4_defensive_out_scale == .2
        assert resumed.slots[0].env.hold_scripted_style_for_match
        resumed.iterate()
    finally:
        resumed.writer.close()
    with pytest.raises(ValueError, match="ya está"):
        configure_rs4_rewards.configure()
    assert prepare_rs4.file_hash(source) == hashes["parent.pt"]


def test_cannot_configure_generalist(source_run):
    source, _ = source_run
    before = source.with_name("config.yaml").read_bytes()
    with pytest.raises(ValueError, match="generalista"):
        configure_rs4_rewards.configure("multi")
    assert source.with_name("config.yaml").read_bytes() == before
