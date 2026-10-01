"""RS4 v3 exact migration, named optimizer state, routed memory and parity."""
from __future__ import annotations

import copy
import json
import sys

import numpy as np
import pytest
import torch

from env.haxball_env import U_SELF_DIM, U_ENT_DIM
from train.model import SetActorCritic, build_model
from train.recurrent_model import RecurrentPolicyOnly
from train.rs4_inference import RoutedPolicyInference, _MultiInputGraph
from train.rs4_migration import migrate_checkpoint, optimizer_parameter_names
from tools.benchmark_multitask import acceptance
from tools.benchmark_rs4_v3 import compare_reports


def observations(batch=3, entities=7, device="cpu"):
    obs = torch.randn(batch, U_SELF_DIM + U_ENT_DIM * entities, device=device)
    obs[:, U_SELF_DIM::U_ENT_DIM] = 1
    obs[:, U_SELF_DIM + 1::U_ENT_DIM] = torch.arange(entities, device=device) >= 3
    return obs


def source_checkpoint():
    model = SetActorCritic(U_SELF_DIM, ent_hidden=8, hidden=16, ent_layers=1,
                          rule_observation="masked")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, eps=1e-5)
    obs = observations()
    logits, values = model(obs)
    (logits.square().sum() + values.square().sum()).backward()
    optimizer.step()
    model.self_norm.mean.fill_(0.07)
    return dict(model=model.state_dict(), model_config=model.config(), opt=optimizer.state_dict(),
                steps=123456, iteration=45), model


def memory_checkpoint():
    source, _ = source_checkpoint()
    return migrate_checkpoint(source, {"type": "recurrent_set", "pooling": "attentive_meanmax", "memory_size": 32})


@pytest.mark.parametrize("kind", ["set", "recurrent_set"])
def test_migration_exact_initial_policy_values_adam_and_source_intact(kind):
    source, source_model = source_checkpoint()
    prior = copy.deepcopy(source)
    target = dict(type=kind)
    if kind == "recurrent_set":
        target.update(memory_size=32, pooling="attentive_meanmax")
    result = migrate_checkpoint(source, {"model": target})
    model = build_model(result["model_config"])
    model.load_state_dict(result["model"])
    obs = observations()
    expected = source_model(obs)
    actual = model.step(obs, torch.randn(3, 32), torch.tensor([0, 8, 18]))[:2] if kind == "recurrent_set" else model(obs)
    for old, new in zip(expected, actual):
        torch.testing.assert_close(old, new, rtol=0, atol=0)
    source_names = list(dict(source_model.named_parameters()))
    target_names = result["optimizer_param_names"][0]
    for number, name in enumerate(source_names):
        target_number = target_names.index(name)
        for field, value in source["opt"]["state"][number].items():
            torch.testing.assert_close(result["opt"]["state"][target_number][field], value, atol=0, rtol=0)
    for name in result["rs4_migration"]["new_parameters"]:
        assert target_names.index(name) not in result["opt"]["state"]
    for name, value in prior["model"].items():
        torch.testing.assert_close(source["model"][name], value, atol=0, rtol=0)
    result["model"]["self_norm.mean"].add_(9)
    torch.testing.assert_close(source["model"]["self_norm.mean"], prior["model"]["self_norm.mean"], atol=0, rtol=0)
    assert result["frozen_normalizers"]
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, eps=1e-5)
    optimizer.load_state_dict(result["opt"])
    optimizer.zero_grad()
    sum(part.sum() for part in actual).backward()
    optimizer.step()  # New states lazily initialize, unchanged Adam states remain valid.


def test_optimizer_manifest_resolves_reordered_parameter_numbers():
    source, model = source_checkpoint()
    names = list(dict(model.named_parameters()))
    numbers = source["opt"]["param_groups"][0]["params"]
    source["opt"]["param_groups"][0]["params"] = numbers[::-1]
    source["optimizer_param_names"] = [names[::-1]]
    result = migrate_checkpoint(source, model.config())
    for number in numbers:
        torch.testing.assert_close(result["opt"]["state"][number]["exp_avg"],
                                   source["opt"]["state"][number]["exp_avg"], atol=0, rtol=0)


@pytest.mark.parametrize("change", [dict(hidden=32), dict(memory_size=64, type="recurrent_set", pooling="attentive_meanmax"),
                                  dict(rule_observation="full"), dict(pooling="attention")])
def test_incompatible_migration_is_rejected(change):
    source, _ = source_checkpoint()
    with pytest.raises(ValueError):
        migrate_checkpoint(source, change)


def test_routed_memory_actual_actions_bootstrap_and_selected_resets():
    checkpoint = memory_checkpoint()
    model = build_model(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model"])
    model.memory_pi.weight.data.normal_(std=0.1)
    runtime = RoutedPolicyInference("cpu")
    runtime.register("learn", model, 8)
    indices, obs = torch.tensor([2, 5, 7]), observations()
    hidden = model.initial_state(3)
    previous, start = torch.full((3,), 18), torch.ones(3, dtype=torch.bool)
    expected = model.step(obs, hidden, previous, start)
    actual = runtime.infer("learn", obs, indices)
    for left, right in zip(actual, expected):
        torch.testing.assert_close(left, right)
    executed = torch.tensor([9, 4, 0])  # Not necessarily the model's sampled action.
    runtime.record_actions("learn", indices, executed)
    before = runtime.get_state("learn", clone=True)
    runtime.infer("learn", obs, indices, commit=False)
    for left, right in zip(runtime.get_state("learn"), before):
        torch.testing.assert_close(left, right)
    expected = model.step(obs, expected[2], executed, torch.zeros(3, dtype=torch.bool))
    actual = runtime.infer("learn", obs, indices)
    for left, right in zip(actual, expected):
        torch.testing.assert_close(left, right)
    runtime.reset_rows([5])
    hidden, prev, starts = runtime.get_state("learn")
    assert hidden[5].abs().sum() == 0 and prev[5] == 18 and starts[5]
    assert hidden[2].abs().sum() > 0 and prev[2] == 9 and not starts[2]
    runtime.invalidate()
    assert runtime.generation == 1


def test_independent_controller_tables_and_four_policy_limit():
    checkpoint = memory_checkpoint()
    model = build_model(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model"])
    runtime = RoutedPolicyInference("cpu")
    for key in ("learner", "bc", "historical", "recent"):
        runtime.register(key, model, 8)
    runtime.infer("learner", observations(1), [2])
    assert runtime.get_state("learner")[0][2].abs().sum() > 0
    assert runtime.get_state("bc")[0].abs().sum() == 0
    with pytest.raises(ValueError, match="four"):
        runtime.register("fifth", model, 8)
    runtime.unregister("recent")
    runtime.register("replacement", model, 8)
    assert runtime.get_state("replacement")[2].all()
    with pytest.raises(ValueError, match="unique"):
        runtime.infer("learner", observations(2), [2, 2])


def test_named_optimizer_manifest_matches_live_optimizer_groups():
    _, model = source_checkpoint()
    optimizer = torch.optim.Adam([{"params": model.pi.parameters()}, {"params": model.v.parameters()}])
    assert optimizer_parameter_names(model, optimizer) == [["pi.weight", "pi.bias"], ["v.weight", "v.bias"]]


def test_static_graph_padding_cannot_keep_previous_player_state():
    graph = _MultiInputGraph(torch.device("cpu"), "graph")
    graph.inputs = (torch.full((4, 6), 9.), torch.full((4, 32), 9.),
                    torch.full((4,), 7), torch.zeros(4, dtype=torch.bool))
    arguments = (torch.ones(2, 6), torch.ones(2, 32), torch.tensor([0, 1]), torch.zeros(2, dtype=torch.bool))
    graph._copy_inputs(arguments, 2, 18)
    assert graph.inputs[0][2:].sum() == 0 and graph.inputs[1][2:].sum() == 0
    assert (graph.inputs[2][2:] == 18).all() and graph.inputs[3][2:].all()
    for static, actual in zip(graph.inputs, arguments):
        torch.testing.assert_close(static[:2], actual)


def test_benchmark_requires_matching_companions_and_rejects_throughput_memory():
    contract = {"frozen_teammates": 0.35, "ppo": {"epochs": 3}}
    control = dict(steps_per_second=100000, cuda_peak_allocated_gib=3.0, comparison_contract=contract)
    memory = dict(steps_per_second=71000, cuda_peak_allocated_gib=4.0, comparison_contract=contract)
    report = compare_reports([control, control], [memory, memory], max_memory_gib=7)
    assert report["acceptance"]["accepted"]
    assert acceptance({**memory, "steps_per_second": 69000}, control)["accepted"] is False
    assert acceptance(memory, control, max_memory_gib=3.5)["accepted"] is False
    with pytest.raises(ValueError, match="mixture"):
        compare_reports([control], [{**memory, "comparison_contract": {"frozen_teammates": 0}}])


@pytest.mark.parametrize("memory", [False, True])
def test_v3_benchmark_at_exhausted_pilot_is_disposable(tmp_path, monkeypatch, memory):
    import yaml
    from tools import benchmark_multitask as benchmark
    from train import multitask
    from train.multitask import MultiTrainer
    from train.rs4_program import ProgramState, default_program
    from train.runtime import load_config

    repository = multitask.ROOT
    cfg = load_config(repository / "train/config_rs4_v3.yaml")
    cfg.pop("rs4_program")
    cfg.pop("bc_reference", None)
    cfg["scripted_readiness"] = {"required": False}  # Tiny synthetic fixture, not a Pod profile.
    cfg["tasks_file"] = str(repository / "train/tasks.yaml")
    cfg["env"].update(agents=16, max_ticks=12)
    cfg["model"].update(hidden=16, layers=1, ent_hidden=8, ent_layers=1)
    cfg["ppo"].update(device="cpu", rollout_len=4, sequence_length=2, minibatch=16, epochs=1,
                      torch_threads=2, numba_threads=1, total_steps=100000)
    cfg["log"].update(every=1000000, replay_every=0)
    monkeypatch.setattr(multitask, "ROOT", tmp_path)
    monkeypatch.setattr(benchmark, "ROOT", tmp_path)
    source = MultiTrainer(copy.deepcopy(cfg), "source", False)
    try:
        source.save(source.run_dir / "latest.pt")
    finally:
        source.writer.close()
    checkpoint = torch.load(source.run_dir / "latest.pt", map_location="cpu", weights_only=False)
    cfg["rs4_program"] = default_program(checkpoint["steps"])
    if memory:
        cfg["model"].update(type="recurrent_set", memory_size=32, pooling="attentive_meanmax")
    migrated = migrate_checkpoint(checkpoint, cfg)
    state = ProgramState.from_config(cfg)
    state.advance_steps(200_000_000)
    migrated["rs4_program_state"] = state.state_dict()
    migrated["steps"] += 200_000_000
    cfg["ppo"]["total_steps"] = migrated["steps"]  # Original pilot segment is exhausted.
    candidate = tmp_path / "runs" / "candidate"
    candidate.mkdir()
    torch.save(migrated, candidate / "latest.pt")
    config = candidate / "config.yaml"
    config.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    before, config_before = (candidate / "latest.pt").read_bytes(), config.read_bytes()
    report = tmp_path / "benchmark.json"
    monkeypatch.setattr(sys, "argv", ["benchmark", "--config", str(config), "--checkpoint", str(candidate / "latest.pt"),
                                      "--device", "cpu", "--warmup", "1", "--iters", "1", "--max-memory-gib", "7",
                                      "--companion-fraction", ".1", "--json-output", str(report)])
    benchmark.main()
    result = json.loads(report.read_text(encoding="utf-8"))
    assert result["samples"] > 0 and result["memory_gate_passed"]
    assert result["model_config"]["type"] == ("recurrent_set" if memory else "set")
    assert result["rs4_profile"]["frozen_teammates_fraction"] == .1
    assert before == (candidate / "latest.pt").read_bytes() and config_before == config.read_bytes()
    assert not list((tmp_path / "runs").glob("_benchmark_*"))


def test_v3_gru32_onnx_state_and_reset_parity(tmp_path):
    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    checkpoint = memory_checkpoint()
    model = build_model(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model"])
    model.memory_pi.weight.data.normal_(std=0.1)
    model.attn_gate.data.fill_(0.3)
    path = tmp_path / "rs4-v3.onnx"
    torch.onnx.export(RecurrentPolicyOnly(model), (observations(2), model.initial_state(2), torch.full((2,), 18)),
                      str(path), input_names=["obs", "memory", "previous_action"], output_names=["logits", "memory_out"],
                      dynamic_axes={"obs": {0: "batch", 1: "width"}, "memory": {0: "batch"},
                                    "previous_action": {0: "batch"}, "logits": {0: "batch"}, "memory_out": {0: "batch"}},
                      opset_version=17, dynamo=False)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(str(path), sess_options=options)
    hidden, onnx_hidden = model.initial_state(2), np.zeros((2, 32), dtype=np.float32)
    for tick in range(5):
        obs, previous = observations(2), torch.tensor([tick, 18])
        if tick == 3:  # Client resets only the replaced player's state and action.
            hidden[0] = 0
            onnx_hidden[0] = 0
            previous[0] = 18
        with torch.no_grad():
            logits, _, hidden = model.step(obs, hidden, previous)
        actual, onnx_hidden = session.run(None, {"obs": obs.numpy(), "memory": onnx_hidden, "previous_action": previous.numpy()})
        np.testing.assert_allclose(actual, logits.numpy(), atol=1e-6, rtol=1e-5)
        np.testing.assert_allclose(onnx_hidden, hidden.numpy(), atol=1e-6, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA Pod validation required")
def test_multiinput_cuda_graph_matches_eager_and_does_not_sample_inside_capture():
    checkpoint = memory_checkpoint()
    model = build_model(checkpoint["model_config"]).cuda()
    model.load_state_dict(checkpoint["model"])
    model.memory_pi.weight.data.normal_(std=0.1)
    graph, eager = RoutedPolicyInference("cuda", "graph"), RoutedPolicyInference("cuda", "eager")
    for runtime in (graph, eager):
        runtime.register("learn", model, 8)
    for tick in range(3):
        indices = [0, 2, 5, 6] if tick == 1 else [0, 2, 6]
        obs = observations(len(indices), device="cuda")
        before = torch.cuda.get_rng_state()
        actual = graph.infer("learn", obs, indices)
        assert torch.equal(before, torch.cuda.get_rng_state())
        expected = eager.infer("learn", obs, indices)
        for left, right in zip(actual, expected):
            torch.testing.assert_close(left, right)
        for runtime in (graph, eager):
            runtime.record_actions("learn", indices, torch.arange(len(indices), device="cuda") + tick)
    assert graph.captures == 1 and graph.replays == 3
    assert graph.graph_input_rows == 10 and graph.graph_padding_rows == 2
    graph.invalidate()
    graph.infer("learn", observations(3, device="cuda"), [0, 2, 6])
    assert graph.captures == 1  # A fresh graph after weight updates, not stale reuse.
