"""Memoria temporal real, PPO por trayectorias y compatibilidad con BC."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from env.haxball_env import U_SELF_DIM, U_ENT_DIM, HaxballEnv
from train.model import SetActorCritic, build_model
from train.recurrent_model import RecurrentSetActorCritic, RecurrentPolicyOnly
from train.recurrent_ppo import RecurrentTrainer, pack_sequences


def model(trained=True):
    torch.manual_seed(8)
    m = RecurrentSetActorCritic(U_SELF_DIM, hidden=16, ent_hidden=8, ent_layers=1,
                               memory_size=8, rule_observation="masked")
    if trained:
        torch.nn.init.normal_(m.memory_pi.weight, std=0.1)
        torch.nn.init.normal_(m.memory_v.weight, std=0.1)
    return m


def observations(length=5, batch=3, entities=3):
    obs = torch.randn(length, batch, U_SELF_DIM + U_ENT_DIM * entities)
    obs[..., U_SELF_DIM::U_ENT_DIM] = 1
    return obs


def test_bc_initialization_preserves_policy_and_old_checkpoint_loading():
    bc = SetActorCritic(U_SELF_DIM, hidden=16, ent_hidden=8, ent_layers=1, rule_observation="masked")
    ck = {"model_config": bc.config(), "model": bc.state_dict()}
    recurrent = model(False)
    recurrent.initialize_from(ck)
    x = observations()[0]
    logits, value, _ = recurrent.step(x, torch.randn(3, 8), torch.tensor([0, 5, 18]))
    torch.testing.assert_close(logits, bc(x)[0], atol=0, rtol=0)
    torch.testing.assert_close(value, bc(x)[1], atol=0, rtol=0)
    restored = build_model(recurrent.config())
    restored.load_state_dict(recurrent.state_dict())
    assert restored.memory_size == 8
    old = dict(ck["model_config"])
    del old["rule_observation"]
    assert not getattr(build_model(old), "is_recurrent", False)


def test_identical_present_observation_can_have_different_history_and_action():
    m = model()
    obs = observations()
    h = m.initial_state(3)
    prev = torch.tensor([18, 18, 18])
    _, _, h1 = m.step(obs[0], h, prev)
    _, _, h2 = m.step(obs[1], h, prev)
    a = m.step(obs[2], h1, prev)[0]
    b = m.step(obs[2], h2, prev)[0]
    assert not torch.allclose(a, b)
    c = m.step(obs[2], h1, torch.tensor([0, 0, 0]))[0]
    assert not torch.allclose(a, c)
    starts = torch.ones(3, dtype=torch.bool)
    torch.testing.assert_close(m.step(obs[2], h1, prev, starts)[0], m.step(obs[2], h2, prev, starts)[0])


def test_sequence_equals_steps_and_resets_only_selected_players():
    m = model()
    obs = observations()
    previous = torch.randint(0, 19, (5, 3))
    starts = torch.zeros(5, 3, dtype=torch.bool)
    starts[2, 1] = True
    h = torch.randn(3, 8)
    logits, values, memory = m.sequence(obs, h, previous, starts)
    for t in range(5):
        lg, v, h = m.step(obs[t], h, previous[t], starts[t])
        torch.testing.assert_close(lg, logits[t], atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(v, values[t], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(h, memory)


def test_temporal_gradient_flows_back_and_stops_at_episode_boundary():
    m = model()
    obs = observations(length=4, batch=1).requires_grad_()
    prev = torch.full((4, 1), 18, dtype=torch.long)
    starts = torch.zeros(4, 1, dtype=torch.bool)
    m.sequence(obs, m.initial_state(1), prev, starts)[0][-1].sum().backward()
    assert obs.grad[0].abs().sum() > 0
    obs.grad = None
    starts[2] = True
    m.sequence(obs, m.initial_state(1), prev, starts)[0][-1].sum().backward()
    assert obs.grad[:2].abs().sum() == 0
    assert obs.grad[2].abs().sum() > 0


def test_private_rule_features_cannot_enter_memory():
    from env.pegeche import N_RULE_FEATS
    m = model()
    a = observations()
    b = a.clone()
    b[..., U_SELF_DIM-N_RULE_FEATS:U_SELF_DIM] = 99
    args = m.initial_state(3), torch.zeros(5, 3, dtype=torch.long), torch.zeros(5, 3, dtype=torch.bool)
    for x, y in zip(m.sequence(a, *args), m.sequence(b, *args)):
        torch.testing.assert_close(x, y, atol=0, rtol=0)


def test_learns_a_hidden_bit_from_earlier_observations():
    torch.manual_seed(8)
    # Tarea binaria: separar la retención temporal del aprendizaje de 16 acciones
    # que nunca son correctas en este experimento sintético.
    m = RecurrentSetActorCritic(U_SELF_DIM, hidden=16, ent_hidden=8, ent_layers=1,
                               memory_size=8, n_actions=2, rule_observation="masked")
    target = torch.arange(16) % 2
    obs = torch.zeros(4, 16, U_SELF_DIM + U_ENT_DIM)
    obs[0, :, 0] = target.float() * 2 - 1
    previous = torch.full((4,16), 2, dtype=torch.long)
    starts = torch.zeros(4,16, dtype=torch.bool)
    opt = torch.optim.Adam(m.parameters(), lr=0.003)
    for _ in range(150):
        logits = m.sequence(obs, m.initial_state(16), previous, starts)[0][-1]
        loss = torch.nn.functional.cross_entropy(logits, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = m.sequence(obs, m.initial_state(16), previous, starts)[0][-1].argmax(-1)
    # La observación final es idéntica para todos; sólo el recuerdo permite acertar.
    assert (pred == target).float().mean() >= 0.95


def test_sequence_packing_never_mixes_players_or_time():
    steps, n, players = 6, 2, 2
    tag = np.arange(steps * n * players).reshape(steps, n, players)
    buffer = {k: tag.copy() for k in ("act", "previous_action", "episode_start", "logp", "adv", "ret")}
    buffer["obs"], buffer["memory"] = tag[..., None], (tag * 100)[..., None]
    packed = pack_sequences(buffer, np.array([[True, False], [True, True]]), 3)
    for col, expected in enumerate(([0,4,8], [2,6,10], [3,7,11], [12,16,20], [14,18,22], [15,19,23])):
        np.testing.assert_array_equal(packed["act"][:, col], expected)
        assert packed["initial_memory"][col, 0] == expected[0] * 100


def make_trainer(tmp_path, monkeypatch):
    from train import multitask
    source = multitask.ROOT
    cfg = yaml.safe_load((source / "train/config_memory.yaml").read_text(encoding="utf-8"))
    cfg["tasks_file"] = str(source / "train/tasks.yaml")
    cfg["env"].update(agents=24, max_ticks=12)
    cfg["model"].update(hidden=16, ent_hidden=8, memory_size=8)
    cfg["ppo"].update(rollout_len=8, sequence_length=4, epochs=2, minibatch=16, torch_threads=2)
    cfg["log"].update(checkpoint_every=1)
    cfg["curriculum"][0].update(scripted=0.25, selfplay=0.25, pool=0.5)
    bc = SetActorCritic(U_SELF_DIM, hidden=16, ent_hidden=8, ent_layers=1)
    path = tmp_path / "bc.pt"
    torch.save({"model_config": bc.config(), "model": bc.state_dict()}, path)
    monkeypatch.setattr(multitask, "ROOT", tmp_path)
    trainer = RecurrentTrainer(cfg, "memory", False, str(path))
    return trainer, cfg, path


def test_recurrent_ppo_bc_pool_timeouts_and_resume(tmp_path, monkeypatch):
    trainer, cfg, bc_path = make_trainer(tmp_path, monkeypatch)
    before = bc_path.read_bytes()
    trainer.league.add_snapshot(trainer.model, "fixed")
    original_norm = trainer.model.self_norm.mean.clone()
    calls = []
    original_values = trainer.state_values
    def values(obs, memory, previous):
        calls.append((memory.copy(), previous.copy()))
        prior = memory.copy()
        result = original_values(obs, memory, previous)
        np.testing.assert_array_equal(memory, prior)
        return result
    trainer.state_values = values
    resumed = None
    try:
        trainer.iterate()
        assert trainer.steps > 0 and len(calls) >= 3
        assert any(np.any(h) and (p < 18).all() for h, p in calls)
        assert trainer.model.memory_pi.weight.abs().sum() > 0
        assert torch.equal(original_norm, trainer.model.self_norm.mean)
        for slot in trainer.slots:
            assert np.all(slot.memory == 0)  # cada 4 decisiones hay timeout
            assert np.all(slot.opponent_memory == 0)
            assert np.all(slot.previous_action == 18)
        trainer.iterate()
        assert torch.isfinite(trainer.model.gru.weight_hh).all()
        resume_cfg = {k: v for k, v in cfg.items() if k != "bc_reference"}
        resumed = RecurrentTrainer(resume_cfg, "memory", True)
        assert resumed.steps == trainer.steps
        assert resumed.bc_model is not None
        assert len(resumed.league.members) == len(trainer.league.members)
        assert all(np.all(s.memory == 0) for s in resumed.slots)
        assert bc_path.read_bytes() == before
    finally:
        trainer.writer.close()
        if resumed:
            resumed.writer.close()


def test_ppo_ratio_is_one_before_any_parameter_change():
    m = model()
    trainer = RecurrentTrainer.__new__(RecurrentTrainer)
    trainer.model, trainer.device, trainer.bc_coef = m, torch.device("cpu"), 0
    trainer.cfg = {"ppo": {"minibatch": 8, "epochs": 1, "clip": 0.2, "vf_coef": 0.5, "max_grad_norm": 0.5}}
    trainer.opt = torch.optim.Adam(m.parameters(), lr=0)
    obs = observations(length=4)
    memory = torch.randn(3, 8)
    previous = torch.randint(0, 19, (4, 3))
    starts = torch.zeros(4, 3, dtype=torch.bool)
    starts[2, 1] = True
    with torch.no_grad():
        logits, values, _ = m.sequence(obs, memory, previous, starts)
        dist = torch.distributions.Categorical(logits=logits)
        actions = dist.sample()
        old_logp = dist.log_prob(actions)
    stats = trainer.update({"obs": obs, "initial_memory": memory, "previous_action": previous,
                            "episode_start": starts, "act": actions, "logp": old_logp,
                            "adv": torch.randn(4, 3), "ret": values}, 0.01)
    assert abs(stats["approx_kl"]) < 1e-7
    assert stats["clipfrac"] == 0


def test_snapshot_switch_resets_blue_memory_without_erasing_red(monkeypatch):
    from train.multitask import MultiTrainer, POOL
    trainer = RecurrentTrainer.__new__(RecurrentTrainer)
    trainer.model = model()
    trainer.league = SimpleNamespace(members=[SimpleNamespace(name="new")])
    slot = SimpleNamespace(N=2, P=2, T=1, modes=np.array([POOL, POOL]), opp_id=np.array([0,0]),
                           memory=np.ones((2,2,8),np.float32), opponent_memory=np.ones((2,2,8),np.float32),
                           previous_action=np.zeros((2,2),np.int64), episode_start=np.zeros((2,2),bool),
                           opponent_keys=["old","new"])
    monkeypatch.setattr(MultiTrainer, "assign_modes", lambda self,s: None)
    trainer.assign_modes(slot)
    assert np.all(slot.memory[:,0] == 1)
    assert np.all(slot.memory[0,1] == 0) and np.all(slot.memory[1,1] == 1)
    assert np.all(slot.opponent_memory[0,1] == 0)
    assert slot.previous_action[0,1] == 18 and slot.episode_start[0,1]


def test_evaluation_keeps_independent_players_and_resets_done_rows(tmp_path):
    from eval.agents import ModelAgent, reset_agents
    m = model()
    path = tmp_path / "rec.pt"
    torch.save({"model_config": m.config(), "model": m.state_dict()}, path)
    agent = ModelAgent(str(path))
    env = HaxballEnv(2, 1, "classic", obs_layout="universal", seed=2)
    obs = env.reset()
    agent(env, obs, np.array([0]))
    h, previous = agent._states[env]
    assert h[:, 0].abs().sum() > 0 and h[:, 1].abs().sum() == 0
    first_red = h[:, 0].clone()
    agent(env, obs, np.array([1]))
    torch.testing.assert_close(h[:, 0], first_red)
    reset_agents((agent, agent), env, np.array([True, False]))
    assert h[0].abs().sum() == 0 and h[1].abs().sum() > 0
    assert (previous[0] == 18).all()
    reset_agents((agent,), env)
    assert agent._states[env][0].abs().sum() == 0


def test_onnx_temporal_outputs_match_torch_across_team_sizes(tmp_path):
    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    m = model().eval()
    path = tmp_path / "memory.onnx"
    torch.onnx.export(RecurrentPolicyOnly(m), (observations()[0], m.initial_state(3), torch.full((3,),18,dtype=torch.long)),
        str(path), input_names=["obs","memory","previous_action"], output_names=["logits","memory_out"],
        dynamic_axes={"obs":{0:"batch",1:"width"},"memory":{0:"batch"},"previous_action":{0:"batch"},
                      "logits":{0:"batch"},"memory_out":{0:"batch"}}, opset_version=17, dynamo=False)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(str(path), sess_options=options)
    for entities in (1,5,11):
        obs = observations(length=4,batch=2,entities=entities)
        h = m.initial_state(2)
        actual_h = h.numpy()
        for t in range(4):
            previous = torch.tensor([t,18])
            with torch.no_grad():
                logits, _, h = m.step(obs[t],h,previous)
            actual_logits, actual_h = session.run(None,{"obs":obs[t].numpy(),"memory":actual_h,
                                                       "previous_action":previous.numpy()})
            np.testing.assert_allclose(actual_logits,logits.numpy(),atol=1e-6,rtol=1e-5)
            np.testing.assert_allclose(actual_h,h.numpy(),atol=1e-6,rtol=1e-5)
