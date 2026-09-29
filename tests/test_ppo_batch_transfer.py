"""Lotes persistentes: datos/dtypes intactos, sin colas viejas ni carreras H2D/PPO."""
import copy

import numpy as np
import pytest
import torch

from train.runtime import PpoBatchTransfer, batch_to_device, load_config


CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="Necesita CUDA: ejecutar en el Pod")


def parts(rows, width=7, seed=13):
    rng = np.random.default_rng(seed)
    arrays = {
        "obs": rng.normal(size=(rows, width)).astype(np.float32),
        "act": rng.integers(0, 18, size=rows, dtype=np.int64),
        "logp": rng.normal(size=rows).astype(np.float32),
        "adv": rng.normal(size=rows).astype(np.float32),
        "ret": rng.normal(size=rows).astype(np.float32),
    }
    return {key: [value[:rows // 3], value[rows // 3:rows // 2], value[rows // 2:]]
            for key, value in arrays.items()}


def pointers(transfer):
    return {key: (buf.host.data_ptr(), buf.device.data_ptr()) for key, buf in transfer.buffers.items()}


def assert_equal(batch, source, device):
    expected = batch_to_device(source, device)
    for key in source:
        assert batch[key].dtype == expected[key].dtype
        assert batch[key].shape == expected[key].shape
        torch.testing.assert_close(batch[key], expected[key], rtol=0, atol=0)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_values_dtypes_capacity_and_reuse_across_varying_batches(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("Necesita CUDA: ejecutar en el Pod")
    device = torch.device(device)
    transfer = PpoBatchTransfer(device)
    old = None
    for rows in (12, 12, 9, 16, 0):
        source = parts(rows, seed=rows + 7)
        batch = batch_to_device(source, device, transfer)
        assert_equal(batch, source, device)
        if old is not None:
            assert pointers(transfer) == old
            assert transfer.last_allocations == 0
        old = pointers(transfer)
        transfer.mark_consumed()
    assert transfer.allocations == 5
    source = parts(23)
    assert_equal(transfer.upload(source), source, device)
    assert transfer.last_allocations == 5 and transfer.allocations == 10
    assert all(buf.capacity == 32 for buf in transfer.buffers.values())
    old = pointers(transfer)
    source = parts(19)
    assert_equal(transfer.upload(source), source, device)
    assert pointers(transfer) == old and transfer.last_allocations == 0
    source = parts(19, width=11)
    assert_equal(transfer.upload(source), source, device)
    assert transfer.last_allocations == 1
    assert pointers(transfer)["obs"] != old["obs"]
    assert pointers(transfer)["act"] == old["act"]
    if device.type == "cuda":
        assert all(buf.host.is_pinned() for buf in transfer.buffers.values())
        torch.cuda.synchronize(device)


def test_pack_handles_noncontiguous_tasks_and_dtype_promotion():
    transfer = PpoBatchTransfer("cpu")
    a = np.arange(20, dtype=np.int16).reshape(5, 4)[:, ::2]
    b = np.arange(12, dtype=np.float32).reshape(3, 4)[::-1, ::2]
    source = {"obs": [a, b]}
    assert_equal(transfer.upload(source), source, torch.device("cpu"))
    assert transfer.buffers["obs"].array.dtype == np.float32
    before = pointers(transfer)
    source = {"obs": [a.astype(np.float64), b]}
    assert_equal(transfer.upload(source), source, torch.device("cpu"))
    assert transfer.last_allocations == 1 and pointers(transfer) != before


@pytest.mark.parametrize("source", [
    {"obs": []}, {"obs": [np.array(1)]},
    {"obs": [np.zeros((3, 2)), np.zeros((3, 4))]},
    {"obs": [np.zeros((3, 2))], "act": [np.zeros(4, dtype=np.int64)]},
])
def test_invalid_layout_rejected_before_copy(source):
    with pytest.raises(ValueError):
        PpoBatchTransfer("cpu").upload(source)


def test_transfer_device_must_match():
    with pytest.raises(ValueError, match="mismo dispositivo"):
        batch_to_device(parts(3), torch.device("cuda"), PpoBatchTransfer("cpu"))


@CUDA
def test_back_to_back_uploads_protect_pinned_source_without_external_sync():
    device = torch.device("cuda")
    transfer = PpoBatchTransfer(device)
    sums = []
    sources = [parts(32768, width=71, seed=seed) for seed in range(6)]
    expected = torch.stack([torch.from_numpy(np.concatenate(source["obs"])).to(device).sum()
                            for source in sources])
    torch.cuda.synchronize(device)  # preparar referencias antes del tramo que prueba reutilización
    # Reusar inmediatamente tras encolar lectores: mark_consumed debe proteger ambos buffers.
    for source in sources:
        batch = transfer.upload(source)
        sums.append(batch["obs"].sum())
        transfer.mark_consumed()
    torch.cuda.synchronize(device)
    torch.testing.assert_close(torch.stack(sums), expected, rtol=0, atol=0)
    assert transfer.allocations == 5


@CUDA
def test_consumers_then_reuse_on_another_cuda_stream():
    device = torch.device("cuda")
    transfer = PpoBatchTransfer(device)
    stream = torch.cuda.Stream(device=device)
    default = torch.cuda.current_stream(device)
    source = parts(4096, width=71)
    first = transfer.upload(source)
    stream.wait_stream(default)
    with torch.cuda.stream(stream):
        saved = first["obs"].clone()
        transfer.mark_consumed()
    # upload espera el evento del consumidor, no sólo su antigua copia H2D.
    next_source = parts(4096, width=71, seed=19)
    second = transfer.upload(next_source)
    torch.cuda.synchronize(device)
    torch.testing.assert_close(saved.cpu(), torch.from_numpy(np.concatenate(source["obs"])), rtol=0, atol=0)
    assert_equal(second, next_source, device)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_ppo_trajectory_inputs_optimizer_and_weights_match_reference(tmp_path, monkeypatch, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("Necesita CUDA: ejecutar en el Pod")
    from train import multitask
    cfg = load_config(multitask.ROOT / "train/config_multi.yaml")
    cfg["tasks_file"] = str(multitask.ROOT / "train/tasks.yaml")
    cfg["stages"] = [{"tasks": ["big_3v3", "x1_1v1"], "min_steps": 0, "max_steps": 100000}]
    cfg["env"].update(agents=24, max_ticks=6)
    cfg["model"].update(hidden=16, ent_hidden=8)
    cfg["ppo"].update(device=device, rollout_len=4, epochs=1, minibatch=16, torch_threads=2, numba_threads=1)
    cfg["runtime"] = {"reuse_ppo_batch": False, "cuda_decisions": "graph" if device == "cuda" else "legacy"}
    cfg["log"].update(every=100, checkpoint_every=100, replay_every=0)
    cfg["curriculum"][0].update(scripted=0.25, selfplay=0.25, pool=0.5)
    monkeypatch.setattr(multitask, "ROOT", tmp_path)
    reference = multitask.MultiTrainer(copy.deepcopy(cfg), "reference", False)
    cached = multitask.MultiTrainer(copy.deepcopy(cfg), "cached", False)
    cached._batch_transfer = PpoBatchTransfer(device)  # ejercitar el empaquetado también en CPU
    inputs = [[], []]
    try:
        for index, trainer in enumerate((reference, cached)):
            trainer.league.add_snapshot(trainer.model, "fixed")
            update = trainer.update
            def checked(batch, coef, index=index, update=update):
                inputs[index].append({key: value.detach().clone() for key, value in batch.items()})
                return update(batch, coef)
            trainer.update = checked
        old = None
        for iteration in range(3):
            for trainer in (reference, cached):
                torch.manual_seed(71 + iteration)
                trainer.iterate()
            assert reference.steps == cached.steps
            assert reference.rng.bit_generator.state == cached.rng.bit_generator.state
            for key, tensor in inputs[0][-1].items():
                torch.testing.assert_close(tensor, inputs[1][-1][key], rtol=0, atol=0)
            for key, tensor in reference.model.state_dict().items():
                torch.testing.assert_close(tensor, cached.model.state_dict()[key], rtol=0, atol=0)
            for param, state in reference.opt.state_dict()["state"].items():
                for key, tensor in state.items():
                    torch.testing.assert_close(tensor, cached.opt.state_dict()["state"][param][key], rtol=0, atol=0)
            if old is not None:
                assert pointers(cached._batch_transfer) == old
            old = pointers(cached._batch_transfer)
    finally:
        reference.writer.close()
        cached.writer.close()
