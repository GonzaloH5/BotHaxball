"""Compatibilidad CPU/CUDA y agrupamiento de políticas sin alterar PPO."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from train.cuda_decisions import CudaDecisionProfile
from train.multitask import MultiTrainer
from train.runtime import CudaRolloutTransfer, batch_to_device, load_config


def test_config_inherits_training_settings(tmp_path):
    (tmp_path / "base.yaml").write_text("ppo: {gamma: 0.995, device: cpu}\nreward: {goal: 1}\n")
    (tmp_path / "gpu.yaml").write_text("extends: base.yaml\nppo: {device: cuda}\n")
    cfg = load_config(tmp_path / "gpu.yaml")
    assert cfg == {"ppo": {"gamma": 0.995, "device": "cuda"}, "reward": {"goal": 1}}
    (tmp_path / "base.yaml").write_text("extends: gpu.yaml\n")
    with pytest.raises(ValueError, match="circular"):
        load_config(tmp_path / "gpu.yaml")


class FixedPolicy:
    def __init__(self, action):
        self.action = action
        self.calls = 0

    def logits(self, obs):
        self.calls += 1
        logits = obs.new_full((len(obs), 18), -100.0)
        logits[:, self.action] = 0
        return logits

    def __call__(self, obs):
        return self.logits(obs), obs[:, 0]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("backend", ["legacy", "eager"])
def test_opponents_grouped_across_tasks_and_learner_values_preserved(device, backend):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("Necesita CUDA: ejecutar en el Pod")
    trainer = MultiTrainer.__new__(MultiTrainer)
    trainer.device = torch.device(device)
    trainer.model = FixedPolicy(3)
    opponent = FixedPolicy(7)
    trainer.league = SimpleNamespace(members=[SimpleNamespace(model=opponent)])
    # Dos formatos: rivales en filas diferentes, self-play en las restantes.
    trainer.slots = [SimpleNamespace(N=2, P=4, T=2, opp_id=np.array([0, -1])),
                     SimpleNamespace(N=2, P=2, T=1, opp_id=np.array([-1, 0]))]
    flat = torch.arange(12, dtype=torch.float32, device=device).reshape(-1, 1)
    def decide():
        if backend == "legacy":
            return trainer._policy_decisions(flat)
        return trainer._sample_cuda(*trainer._mixed_policy_logits(flat))
    actions, logp, values = decide()
    expected = torch.full((12,), 3, device=device)
    expected[[2, 3, 11]] = 7
    torch.testing.assert_close(actions, expected)
    torch.testing.assert_close(values, flat[:, 0])
    torch.testing.assert_close(logp, torch.zeros_like(logp))
    assert opponent.calls == 1 and trainer.model.calls == 1
    decide()
    assert opponent.calls == 2
    trainer.slots[0].opp_id[:] = -1
    trainer.slots[1].opp_id[:] = -1
    trainer._prepare_policy_groups()  # cambia la asignación en la siguiente iteración
    actions, _, _ = decide()
    torch.testing.assert_close(actions, torch.full((12,), 3, device=device))
    assert opponent.calls == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Necesita CUDA: ejecutar en el Pod")
def test_cuda_rollout_transfer_reuse_and_batch_dtypes():
    device = torch.device("cuda")
    transfer = CudaRolloutTransfer(device)
    for rows in (8, 8, 12):
        obs = np.arange(rows * 4, dtype=np.float32).reshape(rows, 4)
        tensor = transfer.upload(obs)
        acts = torch.arange(rows, device=device) % 18
        transfer.enqueue_output(acts, tensor[:, 1], tensor[:, 2])
        actions, logp, values = transfer.wait_output()
        np.testing.assert_array_equal(actions, np.arange(rows) % 18)
        np.testing.assert_array_equal(logp, obs[:, 1])
        np.testing.assert_array_equal(values, obs[:, 2])
        assert transfer.obs.is_pinned() and transfer.output.is_pinned()
    host_pointer, device_pointer = transfer.obs.data_ptr(), transfer.device_obs.data_ptr()
    tensor = transfer.upload_many([obs[:3], obs[3:7], obs[7:]])
    np.testing.assert_array_equal(tensor.cpu().numpy(), obs)
    assert transfer.obs.data_ptr() == host_pointer
    assert transfer.device_obs.data_ptr() == device_pointer
    batch = batch_to_device({"obs": [obs], "act": [actions]}, device)
    assert batch["act"].dtype == torch.int64
    assert batch["obs"].device.type == "cuda"


@pytest.mark.parametrize("device,backend", [("cpu", "legacy"), ("cuda", "legacy"),
                                           ("cuda", "eager"), ("cuda", "graph")])
def test_resume_cpu_checkpoint_and_update_with_bc_pool_and_timeouts(tmp_path, monkeypatch, device, backend):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("Necesita CUDA: ejecutar en el Pod")
    from train import multitask
    source = multitask.ROOT
    cfg = load_config(source / "train/config_multi.yaml")
    cfg["tasks_file"] = str(source / "train/tasks.yaml")
    cfg["stages"] = [{"tasks": ["big_3v3", "x1_1v1"], "min_steps": 0, "max_steps": 100000}]
    cfg["env"].update(agents=24, max_ticks=6)
    cfg["model"].update(hidden=16, ent_hidden=8)
    cfg["ppo"].update(device="cpu", rollout_len=4, epochs=1, minibatch=16, torch_threads=2, numba_threads=1)
    cfg["log"].update(every=1, checkpoint_every=100, replay_every=0)
    cfg["curriculum"][0].update(scripted=0.25, selfplay=0.25, pool=0.5)
    monkeypatch.setattr(multitask, "ROOT", tmp_path)
    initial = MultiTrainer(cfg, "test", False)
    resumed = None
    try:
        initial.league.add_snapshot(initial.model, "cpu_snapshot")
        initial.iterate()
        reference = tmp_path / "bc.pt"
        torch.save({"model_config": initial.model.config(), "model": initial.model.state_dict()}, reference)
        initial.save(initial.run_dir / "latest.pt")
        cfg["bc_reference"] = str(reference)
        cfg["ppo"]["device"] = device
        cfg["runtime"] = {"cuda_decisions": backend, "optimize_cpu": True}
        resumed = MultiTrainer(cfg, "test", True)
        if backend == "graph":
            resumed._decision_profile = CudaDecisionProfile(resumed.device)
        assert resumed.steps == initial.steps
        before = [p.detach().clone() for p in resumed.model.parameters()]
        stats = []
        resumed.log = lambda values, *args: stats.append(values)
        resumed.iterate()
        assert resumed.steps > initial.steps
        assert np.isfinite(list(stats[0].values())).all()
        assert stats[0]["bc_kl"] >= 0
        assert any(not torch.equal(old, new) for old, new in zip(before, resumed.model.parameters()))
        assert all(torch.isfinite(p).all() for p in resumed.model.parameters())
        assert resumed.league.members[0].model.pi.weight.device.type == device
        cached = [s._buffer_cache["obs"] for s in resumed.slots]
        resumed.iterate()
        assert all(s._buffer_cache["obs"] is before for s, before in zip(resumed.slots, cached))
        if device == "cuda":
            assert resumed._batch_transfer is not None
            # El reparto compensado de modos puede hacer que el segundo lote pase
            # el siguiente umbral de capacidad (p. ej. 60 -> 68 filas, 64 -> 128).
            # Esa ampliación única es válida; después debe reutilizarse.
            allocations_after_growth = resumed._batch_transfer.allocations
            assert allocations_after_growth in (5, 10)
            resumed.iterate()
            assert resumed._batch_transfer.allocations == allocations_after_growth
            assert resumed._batch_transfer.last_allocations == 0
            assert stats[-1]["runtime/ppo_batch_allocations"] == 0
        if backend == "graph":
            assert resumed._decision_graph.graph is None  # liberado antes de RunningNorm/Adam
            for values in stats:
                assert values["runtime/cuda_graph_captures"] == 1
                assert values["runtime/cuda_graph_replays"] == cfg["ppo"]["rollout_len"]
            profile = resumed._decision_profile
            expected_iterations = 3 if device == "cuda" else 2
            assert profile.captures == expected_iterations
            assert profile.replays == profile.calls == expected_iterations * cfg["ppo"]["rollout_len"]
            assert all(np.isfinite(value) and value >= 0 for value in profile.totals.values())
    finally:
        initial.writer.close()
        if resumed is not None:
            resumed.writer.close()
