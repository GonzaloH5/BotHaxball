from types import SimpleNamespace

import pytest

from train import checkpoints


def trainer(tmp_path, log):
    calls = []
    def save(path):
        calls.append(path.name)
        path.write_bytes(b"complete")
    return SimpleNamespace(cfg={"log": log}, run_dir=tmp_path, iteration=1, save=save), calls


def test_clock_latest_and_history_are_independent(tmp_path, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(checkpoints.time, "monotonic", lambda: clock[0])
    tr, calls = trainer(tmp_path, dict(checkpoint_interval_seconds=300,
                                     checkpoint_history_interval_seconds=1800, checkpoint_keep=2))
    assert checkpoints.maybe_save_checkpoints(tr) is None
    clock[0] = 399
    assert checkpoints.maybe_save_checkpoints(tr) is None
    clock[0] = 400
    assert checkpoints.maybe_save_checkpoints(tr) is None
    assert calls == ["latest.pt"]
    clock[0] = 1900
    tr.iteration = 1000
    assert checkpoints.maybe_save_checkpoints(tr).name == "ckpt_001000.pt"
    assert calls == ["latest.pt", "latest.pt", "ckpt_001000.pt"]


def test_retention_only_removes_old_automatic_regular_files(tmp_path):
    automatic = ["ckpt_000001.pt", "ckpt_000025.pt", "ckpt_1000000.pt"]
    protected = ["latest.pt", "parent.pt", "fin_etapa0.pt", "saved_best.pt", "ckpt_000000.pt.bak", "ckpt_manual.pt"]
    for name in automatic + protected:
        (tmp_path / name).write_bytes(b"data")
    (tmp_path / "ckpt_000005.pt").mkdir()
    assert [path.name for path in checkpoints.prune_checkpoint_history(tmp_path, 2)] == [automatic[0]]
    assert all((tmp_path / name).exists() for name in automatic[1:] + protected)
    assert (tmp_path / "ckpt_000005.pt").is_dir()


def test_symlink_is_never_deleted_or_counted(tmp_path):
    target = tmp_path / "outside.pt"
    target.write_bytes(b"protected")
    link = tmp_path / "ckpt_000001.pt"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlinks no habilitados")
    (tmp_path / "ckpt_000002.pt").write_bytes(b"data")
    assert checkpoints.prune_checkpoint_history(tmp_path, 1) == []
    assert link.is_symlink() and target.read_bytes() == b"protected"


def test_failed_history_save_does_not_prune_anything(tmp_path):
    tr, calls = trainer(tmp_path, dict(checkpoint_every=1, checkpoint_keep=1))
    old = tmp_path / "ckpt_000000.pt"
    old.write_bytes(b"old complete")
    def fail(path):
        if path.name.startswith("ckpt_"):
            raise OSError("disk full")
        path.write_bytes(b"new complete")
    tr.save = fail
    with pytest.raises(OSError):
        checkpoints.maybe_save_checkpoints(tr)
    assert old.read_bytes() == b"old complete"


def test_legacy_schedule_and_no_limit_still_work(tmp_path):
    tr, calls = trainer(tmp_path, dict(checkpoint_every=2))
    assert checkpoints.maybe_save_checkpoints(tr) is None
    tr.iteration = 2
    assert checkpoints.maybe_save_checkpoints(tr).name == "ckpt_000002.pt"
    assert calls == ["latest.pt", "ckpt_000002.pt"]
    assert checkpoints.prune_checkpoint_history(tmp_path, None) == []


@pytest.mark.parametrize("keep", [0, -1, True, 1.5])
def test_invalid_retention_never_saves_or_deletes(tmp_path, keep):
    tr, calls = trainer(tmp_path, dict(checkpoint_every=1, checkpoint_keep=keep))
    with pytest.raises(ValueError):
        checkpoints.maybe_save_checkpoints(tr)
    assert calls == []


@pytest.mark.parametrize("latest,history", [(0, 1800), (300, 200), (float("nan"), 1800)])
def test_invalid_intervals_do_not_save(tmp_path, latest, history):
    tr, calls = trainer(tmp_path, dict(checkpoint_interval_seconds=latest,
                                     checkpoint_history_interval_seconds=history))
    with pytest.raises(ValueError):
        checkpoints.maybe_save_checkpoints(tr)
    assert calls == []
