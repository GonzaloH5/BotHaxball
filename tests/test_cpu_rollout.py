"""Agrupamiento CPU: rutas de rivales, RNG, valores y reutilización de buffers."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from train import multitask
from train.model import ActorCritic, EntityActorCritic, SetActorCritic
from train.multitask import MultiTrainer, POOL, SELF, SCRIPTED


class Policy:
    def __init__(self, bias=0):
        self.bias, self.calls = bias, 0

    def logits(self, obs):
        self.calls += 1
        return obs[:, :1] * torch.arange(18) / 30 + self.bias * torch.arange(18) / 10

    def __call__(self, obs):
        return self.logits(obs), obs[:, 0] * 2


def test_grouped_cpu_keeps_sampling_order_and_learner_outputs(monkeypatch):
    scripted_calls = []

    def scripted(env, players, eps, rng, env_indices, **kwargs):
        scripted_calls.append((env, tuple(env_indices)))
        return rng.integers(0, 18, (len(env_indices), len(players)))

    monkeypatch.setattr(multitask, "scripted_actions", scripted)
    t = MultiTrainer.__new__(MultiTrainer)
    t.device, t.obs_dim, t.model = torch.device("cpu"), 3, Policy()
    t.optimize_rollout = True
    opponents = [Policy(1), Policy(-1)]
    t.league = SimpleNamespace(members=[SimpleNamespace(model=p) for p in opponents])
    t.slots = [SimpleNamespace(N=4, P=p, T=p // 2, opp_id=np.array([0, 1, -1, -1]),
                               modes=np.array([POOL, POOL, SCRIPTED, SELF]), env=idx,
                               scripted_eps=0.1, scripted_policy="r3", scripted_style=-1)
               for idx, p in enumerate((2, 6, 4))]
    obs = [np.random.default_rng(p).normal(size=(4, p, 3)).astype(np.float32) for p in (2, 6, 4)]
    for iteration in range(2):
        if iteration:
            t.slots[0].opp_id[:] = -1
            t.slots[0].modes[:] = SELF
            t._prepare_policy_groups()  # reconstruir tras reasignar rivales
        results, states, calls, bot_calls = [], [], [], []
        for fast in (False, True):
            t.optimize_cpu = fast
            t.rng = np.random.default_rng(21)
            torch.manual_seed(13)
            for policy in opponents:
                policy.calls = 0
            scripted_calls.clear()
            results.append(t.act(obs))
            states.append((torch.get_rng_state(), t.rng.bit_generator.state))
            calls.append([policy.calls for policy in opponents])
            bot_calls.append(list(scripted_calls))
        for fast, old in zip(results[1], results[0]):
            for a, b in zip(fast, old):
                np.testing.assert_array_equal(a, b)
        torch.testing.assert_close(states[0][0], states[1][0], atol=0, rtol=0)
        assert states[0][1] == states[1][1]
        assert calls == [[3 - iteration, 3 - iteration], [1, 1]]
        assert bot_calls[0] == bot_calls[1]
        before = t._cpu_obs
        saved = [tuple(x.copy() for x in row) for row in results[1]]
        t.act([o + 1 for o in obs])
        assert t._cpu_obs is before
        for a, b in zip(saved, results[1]):
            for x, y in zip(a, b):
                np.testing.assert_array_equal(x, y)


@pytest.mark.parametrize("kind", ["mlp", "entity", "meanmax", "attention", "attentive_meanmax"])
def test_grouped_bootstrap_uses_only_critic_and_preserves_order(kind):
    torch.manual_seed(42)
    width = 5 if kind == "mlp" else 5 + 3 * 8
    model = (ActorCritic(5, hidden=16, layers=1) if kind == "mlp" else
             EntityActorCritic(5, ent_dim=8, hidden=16, ent_hidden=8) if kind == "entity" else
             SetActorCritic(5, hidden=16, ent_hidden=8, pooling=kind)).eval()
    t = MultiTrainer.__new__(MultiTrainer)
    t.model, t.device, t.obs_dim = model, torch.device("cpu"), width
    t.optimize_rollout = True
    observations = [np.random.default_rng(i).normal(size=(n, p, width)).astype(np.float32)
                    for i, (n, p) in enumerate(((3, 2), (2, 6), (1, 4)))]
    t.optimize_cpu = False
    reference = t.values_many(observations)
    policy_calls = []
    hook = model.pi.register_forward_hook(lambda *args: policy_calls.append(1))
    try:
        t.optimize_cpu = True
        actual = t.values_many(observations)
    finally:
        hook.remove()
    for expected, result in zip(reference, actual):
        np.testing.assert_allclose(result, expected, atol=1e-6, rtol=1e-5)
    assert policy_calls == []
    assert t.values_many([]) == []
