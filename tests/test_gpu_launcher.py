import json
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from tools import launch_gpu_overnight as launcher
from tools.launch_gpu_overnight import choose_result


def test_short_probe_keeps_baseline_for_noise_but_adopts_clear_gain():
    baseline = dict(numba_threads=4, steps_per_second=40_000)
    assert choose_result([baseline, dict(numba_threads=8, steps_per_second=40_800)], 4) == baseline
    assert choose_result([baseline, dict(numba_threads=8, steps_per_second=44_000)], 4)["numba_threads"] == 8


def prepare_launcher(tmp_path, monkeypatch, *extra):
    run = tmp_path / "runs" / "multi"
    run.mkdir(parents=True)
    checkpoint = run / "latest.pt"
    checkpoint.write_bytes(b"original training checkpoint")
    (run / "config.yaml").write_text("bc_reference: saved_teacher.pt\n", encoding="utf-8")
    monkeypatch.setattr(launcher, "ROOT", tmp_path)
    monkeypatch.setattr(launcher, "cpu_budget", lambda: 8)
    from numba import config
    monkeypatch.setattr(config, "NUMBA_NUM_THREADS", 8)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(launcher, "load_config", lambda _: {"ppo": {"device": "cuda", "numba_threads": 4}})
    monkeypatch.setattr(sys, "argv", ["launch_gpu_overnight", "--run", "multi", *extra])
    return checkpoint


def test_probes_fixed_copies_confirms_gain_and_resumes_original(tmp_path, monkeypatch):
    checkpoint = prepare_launcher(tmp_path, monkeypatch)
    calls, launched, frozen_paths = [], [], []

    def probe(command, cwd, check):
        assert cwd == tmp_path and check
        assert "tools.benchmark_multitask" in command
        from pathlib import Path
        frozen = Path(command[command.index("--checkpoint") + 1])
        frozen_paths.append(frozen)
        assert frozen != checkpoint and frozen.read_bytes() == checkpoint.read_bytes()
        assert (frozen.parent / "config.yaml").read_text(encoding="utf-8") == "bc_reference: saved_teacher.pt\n"
        threads = int(command[command.index("--numba-threads") + 1])
        calls.append(threads)
        output = Path(command[command.index("--json-output") + 1])
        # Inicialmente 8 gana, pero la confirmación descubre que era ruido.
        speed = {4: 40_000, 6: 42_000, 8: 44_000}[threads] if len(calls) <= 3 else 40_100
        output.write_text(json.dumps(dict(numba_threads=threads, steps_per_second=speed)), encoding="utf-8")

    def launch(command, cwd):
        launched.append(command)
        assert cwd == tmp_path and "train.multitask" in command and "--resume" in command
        return SimpleNamespace(wait=lambda: 0)

    monkeypatch.setattr(launcher.subprocess, "run", probe)
    monkeypatch.setattr(launcher.subprocess, "Popen", launch)
    launcher.main()
    assert calls == [4, 6, 8, 8, 4]  # 10 supera el cupo, no se prueba
    assert len(launched) == 1 and launched[0][-1] == "ppo.numba_threads=4"
    assert checkpoint.read_bytes() == b"original training checkpoint"
    assert not any(p.exists() for p in frozen_paths)
    reports = list((tmp_path / "reports/gpu_startup").glob("probe_*/selected.json"))
    assert len(reports) == 1 and json.loads(reports[0].read_text())["numba_threads"] == 4


def test_probe_failure_never_launches_training_or_changes_latest(tmp_path, monkeypatch):
    checkpoint = prepare_launcher(tmp_path, monkeypatch)

    def fail(command, **_):
        raise subprocess.CalledProcessError(1, command)

    def forbidden(*_, **__):
        pytest.fail("un sondeo fallido no debe lanzar entrenamiento")

    monkeypatch.setattr(launcher.subprocess, "run", fail)
    monkeypatch.setattr(launcher.subprocess, "Popen", forbidden)
    with pytest.raises(subprocess.CalledProcessError):
        launcher.main()
    assert checkpoint.read_bytes() == b"original training checkpoint"


def test_missing_checkpoint_does_not_start_from_scratch(tmp_path, monkeypatch):
    checkpoint = prepare_launcher(tmp_path, monkeypatch)
    checkpoint.unlink()
    with pytest.raises(SystemExit) as error:
        launcher.main()
    assert error.value.code == 2
