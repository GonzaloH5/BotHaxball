"""CLI: warmup excluido, overrides CPU y checkpoint fuente intacto."""
import json
import sys
from types import SimpleNamespace

import torch

from tools import benchmark_multitask as benchmark


def test_summary_excludes_warmup_and_preserves_source(tmp_path, monkeypatch):
    source = tmp_path / "original.pt"
    source.write_bytes(b"original checkpoint")
    output = tmp_path / "result.json"
    cfg = dict(ppo={}, runtime={"optimize_cpu": True, "cache_bc_logits_cpu": True,
                              "optimize_rs4": True},
               log={}, league={}, schedule={})
    monkeypatch.setattr(benchmark, "ROOT", tmp_path)
    monkeypatch.setattr(benchmark, "load_config", lambda path: cfg)

    class Trainer:
        def __init__(self, config, run, resume):
            assert resume
            self.config = config
            assert config["ppo"] == dict(device="cpu", torch_threads=4, numba_threads=6)
            assert config["runtime"]["optimize_cpu"] is False
            assert config["runtime"]["cache_bc_logits_cpu"] is True
            assert config["runtime"]["optimize_rs4"] is False
            checkpoint = tmp_path / "runs" / run / "latest.pt"
            assert checkpoint.read_bytes() == source.read_bytes()
            checkpoint.write_bytes(b"temporary learner")
            self.device = torch.device("cpu")
            self.writer = SimpleNamespace(close=lambda: None)
            self.iteration = 0

        def log(self, *args):
            pass

        def iterate(self):
            self.iteration += 1
            self.log(dict(entropy=1.9, approx_kl=0.01, clipfrac=0.02, bc_kl=0.1,
                          **{"timing/prepare_seconds": 0.0}), 0, self.iteration * 100, 0.0, 0.0)
            self._last_iteration_timings = dict(setup=0.0, maintenance=0.0, logging=0.0,
                                               schedule=0.0, total=0.0)

    monkeypatch.setattr(benchmark, "MultiTrainer", Trainer)
    monkeypatch.setattr(sys, "argv", ["benchmark", "--device", "cpu", "--torch-threads", "4",
                                      "--numba-threads", "6", "--no-optimize-cpu", "--no-optimize-rs4", "--warmup", "1",
                                      "--iters", "2", "--checkpoint", str(source),
                                      "--json-output", str(output)])
    benchmark.main()
    result = json.loads(output.read_text())
    assert result["samples"] == 500
    assert result["warmup_samples"] == 100
    assert result["diagnostic_ppo_samples"] == 600
    assert result["samples_per_iteration"] == [200, 300]
    assert result["learning_metrics"]["entropy"] == 1.9
    assert result["cpu_seconds"] >= 0
    assert result["cpu_budget"] >= 1
    assert result["cpu_core_equivalents"] == result["cpu_seconds"] / result["elapsed_seconds"]
    assert source.read_bytes() == b"original checkpoint"
    assert not list((tmp_path / "runs").iterdir())
    assert cfg["runtime"]["optimize_cpu"] is True  # CLI no muta el objeto original
    assert cfg["runtime"]["optimize_rs4"] is True


def test_v3_profile_measures_own_helpers_and_restores_them(monkeypatch):
    from train import rs4_trainer

    scripted = lambda *a, **kw: "actions"
    transfer = lambda *a, **kw: "batch"
    monkeypatch.setattr(rs4_trainer, "scripted_actions", scripted)
    monkeypatch.setattr(rs4_trainer, "batch_to_device", transfer)
    cohorts = SimpleNamespace(begin=lambda: None, update=lambda *a: None)
    inference = SimpleNamespace(infer=lambda *a: "prediction")
    env = SimpleNamespace(cohorts=cohorts, sim=SimpleNamespace())
    trainer = SimpleNamespace(program=object(), device=torch.device("cpu"), model=SimpleNamespace(),
                              inference=inference, slots=[SimpleNamespace(env=env, task=SimpleNamespace(name="rs4_4v4"))])
    profile = benchmark.RolloutProfile(trainer)
    try:
        assert rs4_trainer.scripted_actions() == "actions"
        assert rs4_trainer.batch_to_device() == "batch"
        assert inference.infer() == "prediction"
        cohorts.begin()
        cohorts.update()
        for label in ("bots CPU (dentro de decisiones)",
                      "lote PPO: empaquetado/transferencia (dentro de preparación)",
                      "inferencia agrupada host (dentro de decisiones/bootstrap)",
                      "cohortes: apertura (dentro de entorno)", "cohortes: resolución (dentro de entorno)"):
            assert profile.calls[label] == 1
    finally:
        profile.close()
    assert rs4_trainer.scripted_actions is scripted
    assert rs4_trainer.batch_to_device is transfer
