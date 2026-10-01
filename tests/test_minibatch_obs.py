"""Buffer de observaciones: mismos minibatches, RNG, pérdidas, pesos y Adam."""
import copy

import pytest
import torch

from train.model import ActorCritic, SetActorCritic
from train.multitask import MultiTrainer


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requiere CUDA"))])
@pytest.mark.parametrize("kind", ["mlp", "set"])
@pytest.mark.parametrize("n,bc_coef", [(27, 0.0), (27, 0.03), (6, 0.03)])
def test_minibatch_obs_preserves_updates(device, kind, n, bc_coef):
    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        torch.manual_seed(51)
        factory = (lambda: ActorCritic(5, hidden=16, layers=1)) if kind == "mlp" else (
            lambda: SetActorCritic(5, hidden=16, layers=1, ent_hidden=8, ent_layers=1))
        width = 5 if kind == "mlp" else 29
        initial = factory().to(device)
        teacher = factory().to(device).eval().requires_grad_(False)
        trainers = []
        for reuse in (False, True):
            trainer = MultiTrainer.__new__(MultiTrainer)
            trainer.cfg = {"ppo": {"epochs": 3, "minibatch": 8, "clip": 0.2,
                                   "vf_coef": 0.5, "max_grad_norm": 0.5},
                           "runtime": {"reuse_minibatch_obs": reuse, "cache_bc_logits": True,
                                       "profile_update": True}}
            trainer.optimize_cpu = reuse
            trainer.model = copy.deepcopy(initial)
            trainer.bc_model, trainer.bc_coef = teacher, bc_coef
            trainer.opt = torch.optim.Adam(trainer.model.parameters(), lr=5e-4, eps=1e-5)
            trainers.append(trainer)
        for iteration in range(2):
            obs = torch.randn(n, width, device=device) + iteration
            with torch.no_grad():
                dist = torch.distributions.Categorical(logits=initial.logits(obs))
                act = dist.sample()
                logp = dist.log_prob(act)
            batch = dict(obs=obs, act=act, logp=logp, adv=torch.randn(n, device=device),
                         ret=torch.randn(n, device=device))
            before = {key: tensor.clone() for key, tensor in batch.items()}
            results, states = [], []
            for trainer in trainers:
                pointers = []
                hook = trainer.model.register_forward_pre_hook(lambda _, args: pointers.append(args[0].data_ptr()))
                try:
                    torch.manual_seed(100 + iteration)
                    results.append(trainer.update(batch, 0.01))
                    states.append((torch.get_rng_state(), torch.cuda.get_rng_state() if device == "cuda" else None))
                finally:
                    hook.remove()
                assert len(pointers) == 3 * (n // min(n, 8))
                if trainer is trainers[1]:
                    assert len(set(pointers)) == 1
            for key, value in results[0].items():
                if not key.startswith("timing/"):
                    assert results[1][key] == pytest.approx(value, abs=1e-6, rel=1e-5)
            for key, tensor in trainers[0].model.state_dict().items():
                torch.testing.assert_close(tensor, trainers[1].model.state_dict()[key], atol=1e-6, rtol=1e-5)
            for left, right in zip(trainers[0].opt.state.values(), trainers[1].opt.state.values()):
                for key in left:
                    torch.testing.assert_close(left[key], right[key], atol=1e-6, rtol=1e-5)
            assert torch.equal(states[0][0], states[1][0])
            if device == "cuda":
                assert torch.equal(states[0][1], states[1][1])
                for result in results:
                    gpu = [value for key, value in result.items() if key.startswith("timing/update_gpu_")]
                    assert len(gpu) == 5 and all(value >= 0 for value in gpu)
            for key in batch:
                torch.testing.assert_close(batch[key], before[key], atol=0, rtol=0)
    finally:
        torch.set_num_threads(threads)
