import pytest
import torch

from train import checkpoints


def test_atomic_save_replaces_with_loadable_checkpoint(tmp_path):
    path = tmp_path / "latest.pt"
    checkpoints.atomic_torch_save({"iteration": 1, "model": torch.ones(3)}, path)
    checkpoints.atomic_torch_save({"iteration": 2, "model": torch.zeros(3)}, path)
    state = torch.load(path, weights_only=False)
    assert state["iteration"] == 2
    torch.testing.assert_close(state["model"], torch.zeros(3))
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("failure", [OSError("disk failure"), KeyboardInterrupt()])
def test_failed_or_interrupted_write_preserves_previous_file(tmp_path, monkeypatch, failure):
    path = tmp_path / "latest.pt"
    checkpoints.atomic_torch_save({"iteration": 123}, path)
    before = path.read_bytes()
    def fail(checkpoint, stream):
        stream.write(b"partial checkpoint")
        raise failure
    monkeypatch.setattr(checkpoints.torch, "save", fail)
    with pytest.raises(type(failure)):
        checkpoints.atomic_torch_save({"iteration": 124}, path)
    assert path.read_bytes() == before
    assert torch.load(path, weights_only=False)["iteration"] == 123
    assert list(tmp_path.iterdir()) == [path]


def test_failed_publish_preserves_previous_file(tmp_path, monkeypatch):
    path = tmp_path / "latest.pt"
    checkpoints.atomic_torch_save({"iteration": 123}, path)
    def fail(*args):
        raise PermissionError("destination busy")
    monkeypatch.setattr(checkpoints.os, "replace", fail)
    with pytest.raises(PermissionError):
        checkpoints.atomic_torch_save({"iteration": 124}, path)
    assert torch.load(path, weights_only=False)["iteration"] == 123
    assert list(tmp_path.iterdir()) == [path]


def test_recovery_validates_then_preserves_corrupt_latest(tmp_path):
    from tools.restore_checkpoint import restore_checkpoint
    source, latest = tmp_path / "ckpt_002975.pt", tmp_path / "latest.pt"
    state = dict(model={}, model_config={}, opt={}, steps=300000000, iteration=2975,
                 stage=1, stage_steps=50000000, task_state={}, league=[], learner_elo=1200, scripted_elo=1200)
    torch.save(state, source)
    latest.write_bytes(b"corrupt original")
    backup = restore_checkpoint(source, latest)
    assert backup.read_bytes() == b"corrupt original"
    assert torch.load(latest, weights_only=False) == state
    assert torch.load(source, weights_only=False) == state


@pytest.mark.parametrize("readable", [True, False])
def test_invalid_recovery_source_never_touches_latest(tmp_path, readable):
    from tools.restore_checkpoint import restore_checkpoint
    source, latest = tmp_path / "source.pt", tmp_path / "latest.pt"
    if readable:
        torch.save({"model": {}}, source)
    else:
        source.write_bytes(b"invalid checkpoint")
    latest.write_bytes(b"keep original")
    with pytest.raises(Exception):
        restore_checkpoint(source, latest)
    assert latest.read_bytes() == b"keep original"
    assert sorted(tmp_path.iterdir()) == sorted([source, latest])
