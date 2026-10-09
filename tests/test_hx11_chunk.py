"""Acciones en bloque (ACT) de L0: etiquetas futuras, cabezas y ensamble temporal."""
import numpy as np
import torch

from learn.hx11_data import Match, chunk_labels
from learn.hx11_policy import Ensemble, SetPolicy, policy_logits
from learn.hx11_bc import flip_y


def _match(T=40):
    z = lambda *sh: np.zeros(sh, np.int16)
    inp = np.zeros((T, 22), np.int8)
    inp[:, 0] = 8                      # derecha
    inp[20:, 0] = 8 | 16               # derecha + patada desde el tick 20
    inp[:, 11] = 8                     # azul: derecha en la cancha = izquierda en coordenadas del equipo
    pid = np.tile(np.arange(22, dtype=np.int16), (T, 1))
    pid[30:, 0] = 99                   # otro jugador ocupa el lugar 0 desde el tick 30
    return Match(name="x", T=T, pos=z(T, 22, 2), vel=z(T, 22, 2), ball=z(T, 2), bvel=z(T, 2),
                 binv=np.zeros(T, np.float16), bg=np.zeros((T, 2), np.float16), inp=inp, kick=np.zeros((T, 22), np.int8),
                 score=z(T, 2), pid=pid, gk=np.full((T, 2), -1, np.int32), phase=np.zeros(T, np.int8),
                 pkind=np.zeros(T, np.int8), pteam=np.full(T, -1, np.int8), pticks=z(T), spot=z(T, 2),
                 offside=np.zeros(T, np.int8), fouls=0, anchor={}, line={})


def test_chunk_labels_future_mask_and_team_frame():
    m = _match()
    y = chunk_labels(m, np.array([15, 36]), np.array([0, 11]), 4)
    assert y[0].tolist() == [3, 3, 12, 12]          # derecha; desde t = 21, derecha + patada
    assert y[1].tolist() == [7, 7, -100, -100]      # azul rotado (izquierda = 7); t = 42 y 45 fuera del partido
    y0 = chunk_labels(m, np.array([24]), np.array([0]), 4)
    assert y0[0].tolist() == [12, 12, -100, -100]   # desde t = 30 el lugar es de otro jugador


def test_flip_keeps_ignored():
    assert flip_y(np.array([3, -100, 12])).tolist() == [3, -100, 12]
    assert flip_y(np.array([1, 10])).tolist() == [5, 14]


def test_chunk_policy_head0_matches_forward():
    torch.manual_seed(0)
    net = SetPolicy(hidden=32, ent_hidden=16, sin_historia=True, chunk=5)
    own, ents = torch.randn(3, 69), torch.randn(3, 21, 9)
    full = net.forward_chunk(own, ents)
    assert full.shape == (3, 5, 18)
    assert torch.allclose(full[:, 0], net(own, ents))


def test_ensemble_averages_predictions_for_the_same_tick():
    ens = Ensemble(3, m=0.0, every=3)
    a = np.zeros((1, 3, 2)); a[0, :, 0] = 1.0          # decisión en k = 0: todo a la acción 0
    b = np.zeros((1, 3, 2)); b[0, :, 1] = 1.0          # decisión en k = 3: todo a la acción 1
    assert ens.step([0], [0], a)[0].tolist() == [1.0, 0.0]
    assert ens.step([0], [3], b)[0].tolist() == [0.5, 0.5]   # promedio de lo predicho para k = 3
    # una decisión salteada reinicia la clave
    assert ens.step([0], [12], a)[0].tolist() == [1.0, 0.0]


def test_policy_logits_without_ensemble_is_forward():
    net = SetPolicy(hidden=32, ent_hidden=16, chunk=4)
    own, ents = torch.randn(2, 69), torch.randn(2, 21, 9)
    assert torch.allclose(policy_logits(net, own, ents), net(own, ents))
    lg = policy_logits(net, own, ents, Ensemble(4), keys=[0, 1], ticks=[0, 0])
    assert torch.allclose(torch.softmax(lg, -1), torch.softmax(net(own, ents), -1), atol=1e-5)


def test_sticky_noise_keeps_choice_and_marginal():
    from learn.hx11_policy import StickyNoise, sample
    p = torch.tensor([[0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0] + [0.0] * 9]) + 1e-12
    lg = torch.log(p)
    sn = StickyNoise(0.0, seed=1)
    picks = [int(sample(lg, 1.0, gumbel=sn.get(["a"]))[0] % 9) for _ in range(20)]
    assert len(set(picks)) == 1                      # sin renovar: siempre la misma dirección
    sn = StickyNoise(1.0, seed=2)
    picks = np.array([int(sample(lg, 1.0, gumbel=sn.get(["a"]))[0] % 9) for _ in range(4000)])
    assert set(picks.tolist()) <= {1, 7} and abs((picks == 1).mean() - 0.5) < 0.04   # marginal intacta
