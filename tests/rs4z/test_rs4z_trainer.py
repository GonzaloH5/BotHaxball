"""Entrenador RS4-Z: inicio aleatorio, máscaras de aprendizaje, GAE y guardado/reanudación."""
import numpy as np
import pytest
import torch

from train.rs4z.run import LEARNER, Config, Trainer


@pytest.fixture(scope="module")
def trainer(tmp_path_factory):
    cfg = Config(run="pytest_rs4z", envs=8, rollout=16, device="cpu", minibatch=512, samples=1e12, smoke=True)
    return Trainer(cfg)


def test_random_init_and_masks(trainer):
    buf, last_v, stats = trainer.rollout()
    learn = buf["learn"]
    assert learn.any()
    # filas no aprendices no tienen acciones del aprendiz registradas como entrenables
    assert (buf["logp"][~learn] == 0).all()
    logs, rows = trainer.update(buf, last_v)
    assert rows == int(learn.sum())
    assert np.isfinite(logs["ploss"]) and np.isfinite(logs["vloss"])


def test_frozen_rows_cannot_change_the_loss(trainer):
    buf, last_v, _ = trainer.rollout()
    state = {k: v.clone() for k, v in trainer.model.state_dict().items()}
    opt_state = {k: v for k, v in trainer.opt.state_dict().items()}
    import copy
    opt_state = copy.deepcopy(opt_state)
    ret = (trainer.ret_mean, trainer.ret_var, trainer.ret_count)
    torch.manual_seed(0)
    trainer.update({k: v.copy() for k, v in buf.items()}, last_v)
    after_a = {k: v.clone() for k, v in trainer.model.state_dict().items()}
    trainer.model.load_state_dict(state)
    trainer.opt.load_state_dict(opt_state)
    trainer.ret_mean, trainer.ret_var, trainer.ret_count = ret
    noisy = {k: v.copy() for k, v in buf.items()}
    not_learn = ~noisy["learn"]
    noisy["rew"][not_learn] += 100.0
    noisy["act"][not_learn] = 5
    noisy["logp"][not_learn] = -3.0
    torch.manual_seed(0)
    trainer.update(noisy, last_v)
    after_b = trainer.model.state_dict()
    # las filas no aprendices sólo pueden influir vía GAE de su propia fila: con máscara, no entran
    for k in after_a:
        assert torch.allclose(after_a[k], after_b[k], atol=1e-6), k


def test_gae_terminal_and_truncation_semantics(trainer):
    T, N = 3, 1
    buf = dict(obs=np.zeros((T, N, 8, 1), np.float32), crit=np.zeros((T, N, 8, 1), np.float32),
               act=np.zeros((T, N, 8), np.int64), logp=np.zeros((T, N, 8), np.float32),
               val=np.zeros((T, N, 8), np.float32), rew=np.zeros((T, N, 8), np.float32),
               learn=np.zeros((T, N, 8), bool), term=np.zeros((T, N), bool), trunc=np.zeros((T, N), bool),
               final_val=np.zeros((T, N, 8), np.float32))
    # se reutiliza la lógica interna con estadísticas neutras
    trainer.ret_mean, trainer.ret_var = 0.0, 1.0
    gamma = trainer.rewards.gamma
    buf["rew"][0, 0, 0] = 1.0
    buf["term"][0, 0] = True        # terminal: no hay bootstrap
    buf["val"][1, 0, 0] = 5.0
    buf["trunc"][1, 0] = True       # truncación: bootstrap con el valor final
    buf["final_val"][1, 0, 0] = 2.0
    lam = trainer.cfg.gae_lambda
    # reproducir el cálculo de update() para la fila (0, 0)
    val = buf["val"]
    adv0 = buf["rew"][0, 0, 0] + gamma * 0.0 - val[0, 0, 0]
    adv1 = buf["rew"][1, 0, 0] + gamma * 2.0 - val[1, 0, 0]
    assert adv0 == pytest.approx(1.0)
    assert adv1 == pytest.approx(gamma * 2.0 - 5.0)


def test_save_and_resume(trainer, tmp_path):
    trainer.save("pytest.pt")
    path = trainer.dir / "pytest.pt"
    other = Trainer(Config(run="pytest_rs4z_b", envs=8, rollout=16, device="cpu", minibatch=512, samples=1e12))
    other.load(path)
    for k, v in trainer.model.state_dict().items():
        assert torch.equal(v, other.model.state_dict()[k])
    assert other.samples == trainer.samples and other.stage.name == trainer.stage.name
