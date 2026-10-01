"""Cache BC: misma pérdida/update y menos inferencias del profesor congelado."""
import copy
import pytest
import torch

from train.model import ActorCritic, SetActorCritic
from train.multitask import MultiTrainer


@pytest.mark.parametrize("n,epochs,coef", [(24, 3, 0.03), (27, 3, 0.03),
                                         (24, 1, 0.03), (24, 3, 0.0)])
@pytest.mark.parametrize("kind", ["mlp", "set"])
@pytest.mark.parametrize("cache_key", ["cache_bc_logits_cpu", "cache_bc_logits"])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requiere CUDA"))])
def test_bc_cache_preserves_update_and_refreshes(n, epochs, coef, kind, cache_key, device):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        torch.manual_seed(23)
        factory = (lambda: ActorCritic(5, hidden=16, layers=1)) if kind == "mlp" else (
            lambda: SetActorCritic(5, hidden=16, layers=1, ent_hidden=8, ent_layers=1))
        width = 5 if kind == "mlp" else 5 + 3 * 8
        model = factory().to(device)
        teacher = factory().to(device).eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        teacher_before = copy.deepcopy(teacher.state_dict())
        calls = []
        original = teacher.logits

        def counted(obs):
            calls.append(len(obs))
            return original(obs)

        teacher.logits = counted
        trainers = []
        for cache in (False, True):
            trainer = MultiTrainer.__new__(MultiTrainer)
            trainer.cfg = {"ppo": {"epochs": epochs, "minibatch": 8, "clip": 0.2,
                                   "vf_coef": 0.5, "max_grad_norm": 0.5},
                           "runtime": {cache_key: cache}}
            trainer.model = copy.deepcopy(model)
            trainer.bc_model = teacher
            trainer.bc_coef = coef
            trainer.optimize_cpu = cache
            trainer.opt = torch.optim.Adam(trainer.model.parameters(), lr=5e-4, eps=1e-5)
            trainers.append(trainer)
        for iteration in range(2):
            obs = torch.randn(n, width, device=device) + iteration
            with torch.no_grad():
                dist = torch.distributions.Categorical(logits=model.logits(obs))
                act = dist.sample()
                logp = dist.log_prob(act)
            batch = dict(obs=obs, act=act, logp=logp, adv=torch.randn(n, device=device),
                         ret=torch.randn(n, device=device))
            results = []
            for trainer in trainers:
                calls.clear()
                torch.manual_seed(100 + iteration)
                results.append(trainer.update(batch, 0.01))
                if coef == 0:
                    assert calls == []
                elif (trainer.cfg["runtime"][cache_key] and epochs > 1
                      and (device == "cpu" or cache_key == "cache_bc_logits")):
                    assert sum(calls) == n  # incluye cola, aunque PPO mantenga su descarte
                    assert max(calls) <= 8
                else:
                    assert sum(calls) == (n // 8) * 8 * epochs
            for key in results[0]:
                assert results[1][key] == pytest.approx(results[0][key], abs=1e-6, rel=1e-5)
            for key, tensor in trainers[0].model.state_dict().items():
                torch.testing.assert_close(tensor, trainers[1].model.state_dict()[key], atol=1e-6, rtol=1e-5)
            for key, tensor in teacher.state_dict().items():
                torch.testing.assert_close(tensor, teacher_before[key], atol=0, rtol=0)
    finally:
        torch.set_num_threads(previous_threads)
