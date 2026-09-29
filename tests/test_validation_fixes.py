"""Regresiones de observabilidad, currículo y evaluación por partidos completos."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from env.haxball_env import U_SELF_DIM, U_ENT_DIM
from env.pegeche import N_RULE_FEATS
from env.tasks import load_catalog
from eval.benchmark import aggregate, assess
from eval.matrix import eval_task
from eval.protocol import opponent_identity, protocol_id
from train.model import SetActorCritic, build_model
from train.multitask import MultiTrainer, TaskSlot, SCRIPTED


def make_slot():
    slot = TaskSlot(SimpleNamespace(name="test"), SimpleNamespace(N=1, P=2, T=1))
    slot.set_regression_context(0.5)
    return slot


def test_masked_policy_cannot_depend_on_private_script_state():
    torch.manual_seed(7)
    model = SetActorCritic(U_SELF_DIM, hidden=16, ent_hidden=8, rule_observation="masked")
    obs = torch.randn(4, U_SELF_DIM + 3 * U_ENT_DIM)
    other = obs.clone()
    other[:, U_SELF_DIM - N_RULE_FEATS:U_SELF_DIM] = torch.randn(4, N_RULE_FEATS) * 100
    model.update_norm(obs)
    a, av = model(obs)
    b, bv = model(other)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    torch.testing.assert_close(av, bv, rtol=0, atol=0)
    restored = build_model(model.config())
    restored.load_state_dict(model.state_dict())
    torch.testing.assert_close(a, restored(other)[0])


def test_legacy_bc_config_and_weights_keep_their_behavior():
    model = SetActorCritic(U_SELF_DIM, hidden=16, ent_hidden=8)
    config = model.config()
    del config["rule_observation"]
    restored = build_model(config)
    restored.load_state_dict(model.state_dict())
    obs = torch.randn(2, U_SELF_DIM + U_ENT_DIM)
    assert restored.rule_observation == "full"
    torch.testing.assert_close(model(obs)[0], restored(obs)[0], rtol=0, atol=0)


def test_opponent_promotion_discards_easier_record_and_boost():
    slot = make_slot()
    slot.best_wr, slot.boost = 0.99, 3.0
    slot.goals[SCRIPTED] = [190, 10]
    trainer = MultiTrainer.__new__(MultiTrainer)
    trainer.slots = [slot]
    trainer.cfg = {"curriculum": [{"scripted_eps": 0.5, "advance_winrate": 0.8, "name": "easy"},
                                  {"scripted_eps": 0.1, "advance_winrate": 2.0, "name": "hard"}]}
    trainer.league = SimpleNamespace(members=[object()])
    trainer.opponent_curriculum()
    assert slot.opp_stage == 1
    assert slot.best_wr == 0 and slot.boost == 1 and slot.wr_window == []
    slot.goals[SCRIPTED] = [110, 90]
    trainer.opponent_curriculum()
    assert slot.best_wr == pytest.approx(0.55)
    assert slot.regression_context == (1, 0.1)


def test_same_difficulty_preserves_window_but_changed_noise_resets_it():
    slot = make_slot()
    slot.best_wr, slot.boost, slot.wr_window = 0.9, 2, [(90, 10)]
    slot.set_regression_context(0.5)
    assert slot.wr_window == [(90, 10)] and slot.best_wr == 0.9
    slot.set_regression_context(0.1)
    assert slot.wr_window == [] and slot.best_wr == 0 and slot.boost == 1


def test_task_metrics_survive_rebuild_and_checkpoint_state(monkeypatch):
    from train import multitask
    trainer = MultiTrainer.__new__(MultiTrainer)
    trainer.cfg = {"env": {"agents": 4, "frame_skip": 3, "max_ticks": 100, "random_reset_prob": 0},
                   "seed": 0, "schedule": {"min_share": 0.04},
                   "curriculum": [{"scripted_eps": 0.5}]}
    trainer.stages = [{"tasks": ["classic_1v1"]}]
    trainer.stage, trainer.rcfg = 0, None
    trainer.catalog = load_catalog()
    slot = make_slot()
    slot.task = trainer.catalog["classic_1v1"]
    slot.best_wr, slot.boost, slot.wr_window = 0.9, 2, [(90, 10)]
    trainer.slots, trainer.task_state = [slot], {}
    trainer.sync_task_state()
    # Simular también la serialización de tuplas a listas.
    trainer.task_state = json.loads(json.dumps(trainer.task_state))
    monkeypatch.setattr(multitask, "make_env", lambda *a, **k: SimpleNamespace(N=2, P=2, T=1, obs_dim=79))
    trainer.build_envs()
    assert trainer.slots[0].wr_window == [(90, 10)]
    assert trainer.slots[0].best_wr == 0.9 and trainer.slots[0].boost == 2
    del trainer.task_state["classic_1v1"]["regression_context"]
    trainer.build_envs()
    assert trainer.slots[0].best_wr == 0 and trainer.slots[0].boost == 1


def test_evaluation_counts_games_draws_and_both_colours(monkeypatch):
    import eval.matrix as matrix
    calls = []
    def play(red, blue, n, *a, **kw):
        calls.append((red, blue, n, kw["seed"]))
        return {"wins": 2, "draws": 1, "losses": 1, "goals_for": 10, "goals_against": 3, "scoreless_games": 1}
    monkeypatch.setattr(matrix, "play", play)
    a, b = object(), object()
    result = eval_task(a, b, load_catalog()["futsal_3v3"], 8, 3, seed=20)
    assert calls == [(a, b, 4, 20), (b, a, 4, 21)]
    assert result["games"] == 8 and result["points"] == 0.5
    assert result["wins"] == 3 and result["draws"] == 2 and result["scoreless_games"] == 2
    assert result["goals_for"] == 13
    with pytest.raises(ValueError):
        eval_task(a, b, load_catalog()["futsal_3v3"], 7, 3)


def test_real_short_evaluation_is_reproducible_and_preserves_torch_rng():
    from eval.agents import RandomAgent
    torch.manual_seed(91)
    state = torch.get_rng_state().clone()
    task = load_catalog()["classic_1v1"]
    args = RandomAgent(), RandomAgent(), task, 2, 0.02
    first = eval_task(*args, seed=123)
    second = eval_task(*args, seed=123)
    assert first == second and first["games"] == 2
    assert torch.equal(state, torch.get_rng_state())


def test_reference_identity_depends_on_content_not_filename(tmp_path):
    path = tmp_path / "latest.pt"
    path.write_bytes(b"first")
    a = protocol_id({"opponent": opponent_identity(str(path)), "minutes": 3})
    path.write_bytes(b"second")
    b = protocol_id({"opponent": opponent_identity(str(path)), "minutes": 3})
    assert a != b


def test_gate_requires_reference_and_rejects_scoreless_draws():
    suite = {"gates": {"min_games": 64, "min_points_vs_scripted": 0.6,
                       "min_points_vs_reference": 0.5, "max_scoreless_fraction": 0.25}}
    row = {"games": 64, "wins": 0, "draws": 64, "losses": 0,
           "goals_for": 0, "goals_against": 0, "scoreless_games": 64}
    result = aggregate([row])
    gate = assess({"futsal_3v3/reference": result}, suite, False)
    assert not gate["passed"] and len(gate["reasons"]) == 2
    assert gate["room_validation"] == "pending"


def test_masked_onnx_matches_torch_with_different_team_sizes(tmp_path):
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    from train.model import PolicyOnly
    model = SetActorCritic(U_SELF_DIM, hidden=16, ent_hidden=8, rule_observation="masked").eval()
    path = tmp_path / "masked.onnx"
    dummy = torch.zeros(1, U_SELF_DIM + 3 * U_ENT_DIM)
    torch.onnx.export(PolicyOnly(model), dummy, str(path), input_names=["obs"], output_names=["logits"],
                      dynamic_axes={"obs": {0: "batch", 1: "width"}, "logits": {0: "batch"}},
                      opset_version=17, dynamo=False)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(str(path), sess_options=options)
    for entities in (1, 5, 11):
        x = torch.randn(2, U_SELF_DIM + entities * U_ENT_DIM)
        with torch.no_grad():
            expected = model.logits(x).numpy()
        x[:, U_SELF_DIM - N_RULE_FEATS:U_SELF_DIM] = 1000
        actual = session.run(None, {"obs": x.numpy()})[0]
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_bc_to_rl_iteration_checkpoint_and_resume(tmp_path, monkeypatch):
    import yaml
    from train import multitask
    source_root = multitask.ROOT
    cfg = yaml.safe_load((source_root / "train/config_validation.yaml").read_text(encoding="utf-8"))
    cfg["tasks_file"] = str(source_root / "train/tasks.yaml")
    cfg["env"]["agents"] = 12
    cfg["ppo"].update(rollout_len=4, epochs=1, minibatch=8, torch_threads=2)
    cfg["model"].update(hidden=16, ent_hidden=8)
    cfg["log"].update(every=1, checkpoint_every=1)
    bc = SetActorCritic(U_SELF_DIM, hidden=16, ent_hidden=8, ent_layers=1)
    path = tmp_path / "bc.pt"
    torch.save({"model": bc.state_dict(), "model_config": bc.config()}, path)
    before = path.read_bytes()
    monkeypatch.setattr(multitask, "ROOT", tmp_path)
    trainer = MultiTrainer(cfg, "pilot", False, str(path))
    resumed = None
    try:
        trainer.iterate()
        assert trainer.steps > 0
        assert trainer.model.rule_observation == trainer.bc_model.rule_observation == "masked"
        resumed = MultiTrainer(cfg, "pilot", True)
        assert resumed.steps == trainer.steps
        assert resumed.slots[0].wr_window == trainer.slots[0].wr_window
        assert path.read_bytes() == before
    finally:
        trainer.writer.close()
        if resumed:
            resumed.writer.close()
