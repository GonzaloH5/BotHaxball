"""Public-only RS4 cues, legacy parity, optimizer/program continuation and client parity."""
import copy
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from env.public_signals import PublicSignalTracker, FEATURES
from env.tasks import load_catalog, make_env
from train.model import build_model, PolicyOnly
from train.recurrent_model import RecurrentPolicyOnly
from train.rs4_program import ProgramState
from train.runtime import load_config
from tools import upgrade_rs4_public_signals as upgrade
from tools import prepare_rs4_v3 as preparer
from test_rs4_runtime import source_checkpoint, memory_checkpoint, observations
from test_rs4_program import source, good_report  # synthetic source fixture, no user runs

ROOT = Path(__file__).resolve().parents[1]


def test_visible_colors_conflict_geometry_symmetry_and_unknown():
    tracker = PublicSignalTracker(6)
    bp = np.array([[950, 400], [-950, -400], [0, 400], [800, 0], [0, 0], [0, 0]], dtype=float)
    bv = np.zeros_like(bp)
    teams = np.array([0, 1])
    ball = np.array([0xFF0000, 0x0000FF, 0xFF0000, 0xFFFFFF, 0xFF0000, 0x123456])
    barrier = np.array([-1, -1, 0x0000FF, 0x0000FF, -2, -1])
    f = tracker.features(bp, bv, teams, ball, barrier, 1000, 450)
    np.testing.assert_array_equal(f[0, 0], f[1, 1])
    assert f[0, 0, 1] == f[1, 1, 1] == f[0, 0, 7] == 1
    assert f[2, 0, 3] == 1 and f[2, :, 1:3].sum() == 0
    assert f[3, 1, 1] == f[3, 1, 8] == 1  # barrier alone
    assert f[4, 0, 3] == 1 and f[4, :, 1:3].sum() == 0
    assert f[5, :, :12].sum() == 0
    bv[0] = [1, 0]
    assert tracker.features(bp, bv, teams, ball, barrier, 1000, 450)[0, :, :9].sum() == 0
    bv[0] = [.05, 0]
    assert tracker.features(bp, bv, teams, ball, barrier, 1000, 450)[0, 0, 1] == 1
    assert len(FEATURES) == 15


def test_geometric_contact_is_uncertain_decays_and_resets():
    tracker = PublicSignalTracker(1)
    bp, bv = np.zeros((1, 2)), np.zeros((1, 2))
    pp, teams = np.array([[[20., 0], [100., 0]]]), np.array([0, 1])
    args = (bp, bv, pp, teams, 15, 10, np.array([0xFFFFFF]), np.array([-1]))
    tracker.sample(*args, 3)
    assert tracker.memory[0, 0] == 0 and tracker.memory[0, 2] == .6
    pp[:] = 100
    tracker.sample(*args, 300)
    f = tracker.features(bp, bv, teams, args[6], args[7], 1000, 450)
    assert f[0, 0, 9] == 1 and f[0, 0, 11] == pytest.approx(.3)
    pp[:] = 0
    tracker.sample(*args, 3)
    assert tracker.memory[0, 0] == -1 and tracker.memory[0, 2] == 0
    bp[:] = 500
    pp[:] = 500
    tracker.sample(*args, 3)
    assert tracker.memory[0, 0] == -1  # reset jump cannot fabricate contact
    tracker.reset([0])
    assert tracker.memory[0, 1] == 600 and tracker.memory[0, 7] == 0


@pytest.mark.parametrize("memory", [False, True])
def test_migration_exact_logits_values_named_adam_and_new_channel_gradient(memory):
    checkpoint = memory_checkpoint() if memory else source_checkpoint()[0]
    checkpoint["rs4_program_state"] = {"relative_steps": 551409181, "skill_debts": ["defense"]}
    prior = copy.deepcopy(checkpoint)
    result = upgrade.migrate_public_checkpoint(checkpoint)
    old, new = build_model(checkpoint["model_config"]), build_model(result["model_config"])
    old.load_state_dict(checkpoint["model"])
    new.load_state_dict(result["model"])
    obs = observations(4)
    obs[:, 56:71] = torch.rand(4, 15)
    if memory:
        hidden, previous = torch.randn(4, 32), torch.tensor([0, 3, 7, 18])
        expected, actual = old.step(obs, hidden, previous), new.step(obs, hidden, previous)
    else:
        expected, actual = old(obs), new(obs)
    for a, b in zip(expected, actual):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert result["rs4_program_state"] == prior["rs4_program_state"]
    old_names = checkpoint.get("optimizer_param_names", [list(dict(old.named_parameters()))])[0]
    new_names = result["optimizer_param_names"][0]
    for i, name in enumerate(old_names):
        if i not in checkpoint["opt"]["state"]:
            continue
        for field, value in checkpoint["opt"]["state"][i].items():
            torch.testing.assert_close(result["opt"]["state"][new_names.index(name)][field], value, atol=0, rtol=0)
    assert new_names.index("public_proj.weight") not in result["opt"]["state"]
    opt = torch.optim.Adam(new.parameters())
    opt.load_state_dict(result["opt"])
    sum(p.square().sum() for p in actual[:2]).backward()
    assert new.public_proj.weight.grad.abs().sum() > 0
    opt.step()
    assert new.public_proj.weight.abs().sum() > 0
    for name, value in prior["model"].items():
        torch.testing.assert_close(checkpoint["model"][name], value, atol=0, rtol=0)
    with pytest.raises(ValueError, match="already"):
        upgrade.migrate_public_checkpoint(result)


def rs4(n=2, optimized=True):
    task = load_catalog()["rs4_4v4"]
    return make_env(task, n, task.n_entities, seed=51, corner_curriculum=False, optimize_rollout=optimized)


def test_env_public_packet_replaces_private_slots_and_keeps_physics_rng():
    legacy, public = rs4(), rs4()
    legacy.reset()
    public.reset()
    public.enable_public_signals()
    actions = np.zeros((2, 8), dtype=np.int64)
    for _ in range(6):
        a, ar, ad, _ = legacy.step(actions)
        b, br, bd, _ = public.step(actions)
        np.testing.assert_array_equal(a[..., :56], b[..., :56])
        np.testing.assert_array_equal(a[..., 71:], b[..., 71:])
        np.testing.assert_array_equal(ar, br)
        np.testing.assert_array_equal(ad, bd)
        np.testing.assert_array_equal(legacy.sim.pos, public.sim.pos)
        assert np.all(b[..., 70] == 1)
    public.public_ball_visible = False
    public.public_barrier_visible = False
    public.setpiece_team[:] = 0
    before = public.public_features().copy()
    public.setpiece_team[:] = 1
    public.setpiece_kind[:] = 3
    public.last_touch[:] = 7
    np.testing.assert_array_equal(before, public.public_features())  # no hidden owner/contact leakage
    public._public_signals.memory[:, 0] = 0
    public._reset_envs(np.array([0]))
    assert public._public_signals.memory[0, 0] == -1
    assert public._public_signals.memory[1, 0] == 0
    public.optimize_rollout = False
    slow = public.observe()
    public.optimize_rollout = True
    np.testing.assert_allclose(slow, public.observe(), atol=1e-6)


def test_python_js_sampled_history_and_features_parity():
    if not shutil.which("node"):
        pytest.skip("Node required")
    tracker = PublicSignalTracker(1)
    teams = np.array([0, 1])
    frames, expected = [], []
    for i in range(12):
        bp = np.array([[950., 400.]]) if i < 6 else np.array([[0., 0.]])
        bv = np.array([[0., 0.]]) if i != 4 else np.array([[1., 0.]])
        pp = np.array([[bp[0] + [20, 0], bp[0] + [200, 0]]])
        if i in (3, 10):
            pp[0, 1] = bp[0]
        if i in (7, 8):
            pp[:] = 300
        ball, barrier = (0xFF0000 if i % 2 else 0x0000FF), (-2 if i == 5 else -1)
        tracker.sample(bp, bv, pp, teams, 15, 10, np.array([ball]), np.array([barrier]), 3)
        expected.append(tracker.features(bp, bv, teams, np.array([ball]), np.array([barrier]), 1000, 450)[0].tolist())
        frames.append(dict(ball=dict(pos=bp[0].tolist(), vel=bv[0].tolist(), color=ball),
                           players=[dict(team=int(t), pos=p.tolist()) for t, p in zip(teams, pp[0])], barrier=barrier))
    script = """const fs=require('fs'), {PublicSignalTracker,publicFeatures}=require('./deploy/public_signals');
const t=new PublicSignalTracker(); console.log(JSON.stringify(JSON.parse(fs.readFileSync(0,'utf8')).map(f=>{
const packet=t.sample(f.ball,f.players,3,15,10,f.barrier);
const state={ball:f.ball,players:f.players,publicSignals:packet};
return f.players.map((p,i)=>publicFeatures(state,i,{goal_x:1000,field_half_h:450}));})));"""
    result = subprocess.run(["node", "-e", script], cwd=ROOT, input=json.dumps(frames), text=True, capture_output=True, check=True)
    np.testing.assert_allclose(json.loads(result.stdout), expected, atol=1e-7)


def test_preparer_preserves_phase_lr_debts_ledger_and_originals(source, monkeypatch):
    path, _, _ = source
    ck = torch.load(path, weights_only=False)
    ck["model_config"]["rule_observation"] = "masked"
    ck["task_state"]["rs4_4v4"] = dict(opp_stage=0, best_wr=0., boost=1., steps=0, intro_step=0)
    torch.save(ck, path)
    original = preparer.prepare(path)
    for name, steps in (("control", 551409181), ("memory", 200000000)):
        cp = original / name / "latest.pt"
        ck = torch.load(cp, weights_only=False)
        cfg = load_config(cp.with_name("config.yaml"))
        state = ProgramState(cfg["rs4_program"], ck["rs4_program_state"])
        if name == "control":
            state.advance_steps(100000000)
            state.record_evaluation(good_report())
            state.advance_steps(100000000)
            state.record_evaluation(good_report())
            state.advance_steps(200000000)
            state.record_evaluation(good_report())
            state.advance_steps(steps - 400000000)
        else:
            state.advance_steps(steps)
        state.update_lr(.0001, 44125)
        state.skill_debts = ["defense"]
        ck["rs4_program_state"] = state.state_dict()
        ck["steps"] += steps
        cfg["ppo"]["total_steps"] = ck["steps"] + 100000000
        cp.with_name("config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf8")
        torch.save(ck, cp)
    ledger = json.loads((original / "ledger.json").read_text())
    ledger["selected"] = "control"
    ledger["diagnostic_ppo_steps"] = 5605350
    (original / "ledger.json").write_text(json.dumps(ledger))
    before = {p.relative_to(original): p.read_bytes() for p in original.rglob("*") if p.is_file()}
    monkeypatch.setattr(upgrade, "ROOT", path.parents[2])
    summary = upgrade.prepare(dry_run=True)
    destination = path.parents[1] / "rs4_v3_public"
    assert not destination.exists() and summary["budget_expansion"] == 0
    upgrade.prepare()
    for rel, contents in before.items():
        assert (original / rel).read_bytes() == contents
    old = torch.load(original / "control/latest.pt", weights_only=False)
    new = torch.load(destination / "control/latest.pt", weights_only=False)
    assert old["rs4_program_state"] == new["rs4_program_state"]
    assert new["rs4_program_state"]["phase_index"] == 1
    assert new["steps"] == old["steps"]
    assert (destination / "public_source.pt").read_bytes() == before[Path("control/latest.pt")]
    cfg = load_config(destination / "control/config.yaml")
    assert cfg["rs4_program"] == load_config(original / "control/config.yaml")["rs4_program"]
    assert cfg["model"]["public_signals_version"] == 1
    assert not load_config(destination / "memory/config.yaml")["model"].get("public_signals_version")
    ledger = json.loads((destination / "ledger.json").read_text())
    assert ledger["consumed_steps"] == 757014531
    assert ledger["total_budget_steps"] == 6000000000
    assert json.loads((destination / "specialization.json").read_text())["evaluation_directory"] == "evaluations_public_v1"
    with pytest.raises(ValueError, match="destination exists"):
        upgrade.prepare()
    with pytest.raises(ValueError, match="exclude ball"):
        upgrade.prepare(run="invalid", barrier_discs=[0])
    # Resume actual PPO in a tiny disposable copy, not the source or the Pod.
    from train import multitask
    from train.rs4_trainer import RS4V3Trainer
    repository = multitask.ROOT
    monkeypatch.setattr(multitask, "ROOT", path.parents[2])
    cfg["scripted_readiness"] = {"required": False}
    cfg["tasks_file"] = str(repository / "train/tasks.yaml")
    cfg["env"].update(agents=16, max_ticks=12)
    cfg["ppo"].update(device="cpu", rollout_len=4, sequence_length=2, minibatch=16, epochs=1,
                      torch_threads=2, numba_threads=1, total_steps=new["steps"] + 13)
    cfg["log"].update(every=1000000, replay_every=0)
    trainer = RS4V3Trainer(cfg, "rs4_v3_public/control", True)
    try:
        assert trainer.program.state_dict() == new["rs4_program_state"]
        assert trainer.slots[0].env._public_signals is not None
        trainer.iterate()
        assert trainer.steps == new["steps"] + 13
        assert trainer.program.relative_steps == new["rs4_program_state"]["relative_steps"] + 13
        trainer.save(destination / "control/latest.pt")
        saved = torch.load(destination / "control/latest.pt", weights_only=False)
        assert saved["model_config"]["public_signals_version"] == 1
        assert saved["public_signal_migration"] == new["public_signal_migration"]
    finally:
        trainer.writer.close()
    for rel, contents in before.items():
        assert (original / rel).read_bytes() == contents


@pytest.mark.parametrize("memory", [False, True])
def test_public_projection_onnx_and_full_client_observation_parity(tmp_path, memory):
    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    ck = upgrade.migrate_public_checkpoint(memory_checkpoint() if memory else source_checkpoint()[0])
    model = build_model(ck["model_config"])
    model.load_state_dict(ck["model"])
    model.public_proj.weight.data.normal_(std=.05)  # nonzero: catches a missing public branch
    obs = observations(2)
    model.eval()
    onnx = tmp_path / "public.onnx"
    inputs = (obs, model.initial_state(2), torch.tensor([3, 18])) if memory else obs
    wrapper = RecurrentPolicyOnly(model) if memory else PolicyOnly(model)
    torch.onnx.export(wrapper, inputs, str(onnx), input_names=["obs", "memory", "previous_action"] if memory else ["obs"],
                      output_names=["logits", "memory_out"] if memory else ["logits"], opset_version=17, dynamo=False)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(str(onnx), sess_options=options)
    feeds = {"obs": obs.numpy()}
    if memory:
        feeds.update(memory=inputs[1].numpy(), previous_action=inputs[2].numpy())
    with torch.no_grad():
        expected = wrapper(*inputs) if memory else wrapper(inputs)
    outputs = session.run(None, feeds)
    for a, b in zip(outputs, expected if memory else [expected]):
        np.testing.assert_allclose(a, b.numpy(), atol=1e-6, rtol=1e-5)
    if shutil.which("node"):
        from export.to_onnx import make_fixture_universal
        fixture = make_fixture_universal(model, tasks=["rs4_4v4"], per_task=3)
        assert any(s["publicSignals"]["ballColor"] != 0xFFFFFF for s in fixture["states"])
        script = """const fs=require('fs'),{buildObsUniversal}=require('./deploy/obs_universal');
const f=JSON.parse(fs.readFileSync(0,'utf8'));let error=0;
for(const s of f.states)for(let p=0;p<s.players.length;p++){
const obs=buildObsUniversal(s,p,f.geoms[s.stadium],s.opts);
obs.forEach((v,i)=>error=Math.max(error,Math.abs(v-s.obs[p][i])));}
console.log(error);"""
        result = subprocess.run(["node", "-e", script], cwd=ROOT, input=json.dumps(fixture),
                                text=True, capture_output=True, check=True)
        assert float(result.stdout) < 1e-4
