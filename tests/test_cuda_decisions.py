"""Probabilidades PPO, RNG fresco y graphs sin pesos/normalizadores/rivales obsoletos.

Las pruebas CUDA son estrictas (no aceptan fallback) y deben ejecutarse en el Pod.
"""
import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from env.haxball_env import U_SELF_DIM
from train.cuda_decisions import CudaDecisionGraph, CudaDecisionProfile, sample_decisions
from train.model import SetActorCritic
from train.multitask import MultiTrainer
from train.runtime import CudaRolloutTransfer, load_config


CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="Necesita CUDA: ejecutar en el Pod")


def mixed_trainer(device):
    trainer = MultiTrainer.__new__(MultiTrainer)
    trainer.device = torch.device(device)
    trainer.model = SetActorCritic(U_SELF_DIM, hidden=16, layers=1, ent_hidden=8,
                                  ent_layers=1, rule_observation="masked").to(device).eval()
    opponents = []
    for i in range(2):
        model = copy.deepcopy(trainer.model).eval()
        with torch.no_grad():
            model.pi.bias.fill_(-2)
            model.pi.bias[6 + i] = 2
            model.v.bias.fill_(10 + i)  # nunca usar sus valores para el aprendiz
        model.requires_grad_(False)
        opponents.append(SimpleNamespace(model=model))
    trainer.league = SimpleNamespace(members=opponents)
    trainer.slots = [SimpleNamespace(N=2, P=4, T=2, opp_id=np.array([0, -1])),
                     SimpleNamespace(N=2, P=2, T=1, opp_id=np.array([-1, 1]))]
    trainer._prepare_policy_groups()
    generator = torch.Generator().manual_seed(23)
    obs = torch.randn(12, U_SELF_DIM + 5 * 8, generator=generator)
    entities = obs[:, U_SELF_DIM:].reshape(12, 5, 8)
    entities[..., :2] = 0
    entities[:8, :3, 0] = 1
    entities[:8, 1:3, 1] = 1
    entities[8:, 0, :2] = 1
    return trainer, obs.to(device)


@torch.no_grad()
def expected_mixed(trainer, flat):
    logits, values = trainer.model(flat)
    for oid, indices in trainer._policy_groups:
        logits[indices] = trainer.league.members[oid].model.logits(flat[indices])
    return logits, values


def test_mixed_logits_keep_exact_learner_logp_values_and_norm():
    trainer, flat = mixed_trainer("cpu")
    main_logits, main_values = trainer.model(flat)
    before = {key: value.clone() for key, value in trainer.model.state_dict().items()}
    logits, values = trainer._mixed_policy_logits(flat)
    torch.testing.assert_close((logits, values), expected_mixed(trainer, flat), rtol=0, atol=0)
    learner = torch.tensor([0, 1, 4, 5, 6, 7, 8, 9, 10])
    actions, logp, _ = sample_decisions(logits, values)
    original_logp = torch.distributions.Categorical(logits=main_logits).log_prob(actions)
    torch.testing.assert_close(logp[learner], original_logp[learner], rtol=0, atol=0)
    torch.testing.assert_close(values, main_values, rtol=0, atol=0)
    for key, value in trainer.model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


def test_sampler_is_one_fresh_categorical_for_all_rows():
    logits = torch.tensor([[0.2, -0.1, 0.4], [1.2, 0.5, -0.7]]).repeat(128, 1)
    values = torch.arange(len(logits), dtype=torch.float32)
    state = torch.get_rng_state()
    dist = torch.distributions.Categorical(logits=logits, validate_args=False)
    expected = dist.sample()
    expected_state = torch.get_rng_state()
    torch.set_rng_state(state)
    actions, logp, val = sample_decisions(logits, values)
    torch.testing.assert_close(actions, expected)
    torch.testing.assert_close(logp, dist.log_prob(actions))
    assert val is values
    assert torch.equal(torch.get_rng_state(), expected_state)
    again, _, _ = sample_decisions(logits, values)
    assert not torch.equal(actions, again)


def test_runpod_keeps_training_settings_and_31_threads():
    from train.multitask import ROOT
    base = load_config(ROOT / "train/config_multi.yaml")
    cfg = load_config(ROOT / "train/config_runpod.yaml")
    for section in ("model", "reward", "curriculum", "league", "env", "stages"):
        assert cfg[section] == base[section]
    assert cfg["runtime"]["cuda_decisions"] == "auto"
    assert cfg["ppo"]["numba_threads"] == 31
    for key, value in base["ppo"].items():
        if key not in ("device", "torch_threads"):
            assert cfg["ppo"][key] == value


def test_auto_capture_incompatibility_warns_once_and_stays_disabled(monkeypatch):
    engine = CudaDecisionGraph("cuda", "auto")
    calls = []
    def unsupported(*args):
        calls.append(1)
        raise RuntimeError("operation not permitted when stream is capturing")
    monkeypatch.setattr(engine, "_capture", unsupported)
    flat = torch.ones(3, 4)
    with pytest.warns(RuntimeWarning, match="usando decisiones eager"):
        result = engine.run(lambda x: (x + 1, x[:, 0]), flat)
    torch.testing.assert_close(result[0], flat + 1)
    assert engine.graph is None and engine.captures == engine.replays == 0
    engine.reset()
    engine.run(lambda x: (x, x[:, 0]), flat)
    assert calls == [1] and engine.disabled_reason is not None


@pytest.mark.parametrize("mode,message", [
    ("graph", "operation not permitted when stream is capturing"),
    ("auto", "out of memory during graph capture"),
    ("auto", "illegal memory access during graph capture"),
    ("auto", "device-side assert during graph capture"),
    ("auto", "unrelated model bug"),
    ("auto", "model bug in captured logits"),
])
def test_capture_errors_not_hidden(monkeypatch, mode, message):
    engine = CudaDecisionGraph("cuda", mode)
    def fail(*args):
        raise RuntimeError(message)
    monkeypatch.setattr(engine, "_capture", fail)
    with pytest.raises(RuntimeError, match=message):
        engine.run(lambda x: x, torch.ones(1))
    assert engine.disabled_reason is None


def test_invalid_graph_mode_and_cpu_input_fail():
    with pytest.raises(ValueError, match="desconocido"):
        CudaDecisionGraph("cuda", "typo")
    with pytest.raises(ValueError, match="tensor CUDA"):
        CudaDecisionGraph("cuda", "graph").run(lambda x: x, torch.ones(1))


@pytest.mark.parametrize("baseline", [False, True])
def test_benchmark_profile_and_baseline_use_disposable_checkpoint(tmp_path, monkeypatch, capsys, baseline):
    import sys
    from train import multitask
    from tools import benchmark_multitask as benchmark
    cfg = load_config(multitask.ROOT / "train/config_multi.yaml")
    cfg["tasks_file"] = str(multitask.ROOT / "train/tasks.yaml")
    cfg["stages"] = [{"tasks": ["big_3v3", "x1_1v1"], "min_steps": 0, "max_steps": 100000}]
    cfg["env"].update(agents=24, max_ticks=6)
    cfg["model"].update(hidden=16, ent_hidden=8)
    cfg["ppo"].update(device="cpu", rollout_len=4, epochs=1, minibatch=16,
                      torch_threads=2, numba_threads=1)
    cfg["runtime"] = {"cuda_decisions": "auto"}
    cfg["log"].update(every=1, checkpoint_every=100, replay_every=0)
    monkeypatch.setattr(multitask, "ROOT", tmp_path)
    monkeypatch.setattr(benchmark, "ROOT", tmp_path)
    initial = MultiTrainer(cfg, "saved", False)
    checkpoint = initial.run_dir / "latest.pt"
    try:
        initial.save(checkpoint)
    finally:
        initial.writer.close()
    before = checkpoint.read_bytes()
    monkeypatch.setattr(benchmark, "load_config", lambda _: copy.deepcopy(cfg))
    trainers = []
    def factory(*args, **kwargs):
        trainer = MultiTrainer(*args, **kwargs)
        trainers.append(trainer)
        return trainer
    monkeypatch.setattr(benchmark, "MultiTrainer", factory)
    argv = ["benchmark", "--checkpoint", str(checkpoint), "--warmup", "1", "--iters", "1", "--profile-rollout"]
    if baseline:
        argv.append("--baseline")
    monkeypatch.setattr(sys, "argv", argv)
    benchmark.main()
    output = capsys.readouterr().out
    assert "pasos/s reales" in output and "Desglose inclusivo" in output
    assert trainers[0].cuda_decisions == ("legacy" if baseline else "auto")
    assert trainers[0]._decision_profile is None
    assert checkpoint.read_bytes() == before
    assert not list((tmp_path / "runs").glob("_benchmark_*"))


@CUDA
def test_graph_matches_mixed_policies_changes_inputs_and_keeps_rng_outside():
    trainer, flat = mixed_trainer("cuda")
    graph = CudaDecisionGraph(trainer.device, "graph")
    transfer = CudaRolloutTransfer(trainer.device)
    first = flat.cpu().numpy()
    inputs = [first, first * 0.5 + 0.1, first * 0.5 + 0.1]
    action_rows = []
    for obs in inputs:
        current = transfer.upload_many([obs[:8], obs[8:]])
        cpu_state, gpu_state = torch.get_rng_state(), torch.cuda.get_rng_state(trainer.device)
        logits, values = graph.run(trainer._mixed_policy_logits, current)
        assert torch.equal(cpu_state, torch.get_rng_state())
        assert torch.equal(gpu_state, torch.cuda.get_rng_state(trainer.device))
        torch.testing.assert_close((logits, values), expected_mixed(trainer, current))
        actions, logp, value = sample_decisions(logits, values)
        learner = torch.tensor([0, 1, 4, 5, 6, 7, 8, 9, 10], device=trainer.device)
        main_logp = torch.distributions.Categorical(logits=trainer.model.logits(current)).log_prob(actions)
        torch.testing.assert_close(logp[learner], main_logp[learner])
        transfer.enqueue_output(actions, logp, value)
        result = transfer.wait_output()
        action_rows.append(result[0].copy())
        np.testing.assert_allclose(result[1], logp.cpu().numpy())
        np.testing.assert_allclose(result[2], values.cpu().numpy())
    assert graph.captures == 1 and graph.replays == len(inputs)
    assert not np.array_equal(action_rows[-1], action_rows[-2])
    # También aceptar otro tensor del mismo layout copiándolo al input fijo del graph.
    different = flat * 0.7
    logits, values = graph.run(trainer._mixed_policy_logits, different)
    torch.testing.assert_close((logits, values), expected_mixed(trainer, different))
    with pytest.raises(ValueError, match="layout"):
        graph.run(trainer._mixed_policy_logits, different[:3])
    torch.cuda.synchronize(trainer.device)
    graph.reset()


@CUDA
def test_reset_refreshes_weights_norm_and_opponents():
    trainer, flat = mixed_trainer("cuda")
    trainer._decision_graph = CudaDecisionGraph(trainer.device, "graph")
    trainer._decision_profile = None
    old_logits, _ = trainer._cuda_inference(flat)
    old_logits = old_logits.clone()
    torch.cuda.synchronize(trainer.device)
    trainer._reset_decision_graph()
    assert trainer._decision_graph.graph is None
    old_var = trainer.model.self_norm.var  # mantener viva para verificar cambio real de dirección
    with torch.no_grad():
        trainer.model.pi.bias[3].add_(5)
        trainer.model.update_norm(flat * 2)
    assert trainer.model.self_norm.var.data_ptr() != old_var.data_ptr()
    trainer.slots[0].opp_id[:] = 1
    trainer.slots[1].opp_id[:] = -1
    trainer._prepare_policy_groups()
    current = trainer._cuda_inference(flat)
    torch.testing.assert_close(current, expected_mixed(trainer, flat))
    assert not torch.allclose(old_logits, current[0])
    assert trainer._decision_graph.captures == trainer._decision_graph.replays == 1
    torch.cuda.synchronize(trainer.device)
    trainer._reset_decision_graph()


@CUDA
def test_cuda_profile_reads_events_after_existing_transfer_wait():
    device = torch.device("cuda")
    profile = CudaDecisionProfile(device)
    transfer = CudaRolloutTransfer(device)
    profile.mark(0)
    flat = transfer.upload(np.ones((16, 18), dtype=np.float32))
    profile.mark(1)
    logits, values = flat + 0.3, flat[:, 0]
    profile.mark(2)
    actions, logp, values = sample_decisions(logits, values)
    profile.mark(3)
    transfer.enqueue_output(actions, logp, values)
    profile.mark(4)
    transfer.record_ready()
    transfer.wait_output()
    profile.finish(0.1, 0.01)
    assert profile.calls == 1 and len(profile.totals) == 6
    assert all(np.isfinite(value) and value >= 0 for value in profile.totals.values())
    profile.reset()
    assert profile.calls == 0 and not profile.totals
