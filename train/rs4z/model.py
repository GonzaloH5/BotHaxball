"""Red RS4-Z: actor sobre la observación pública v2 y crítico centralizado con información privilegiada.

* Entidades (3 compañeros, 4 rivales): MLP compartida por tipo; pooling con máscara (media y máximo),
  invariante al orden y válido para 1v0…4v4 (entidades ausentes no aportan).
* Actor: [propio, compañeros(media,máx), rivales(media,máx)] → 2×256 → 18 logits. Lo único que se exporta.
* Crítico (MAPPO): la misma codificación de la escena + features privilegiadas (reloj, marcador, saques,
  último toque, latencia rival). Nunca se exporta ni lo ve el actor.
Inicialización aleatoria (ortogonal); última capa del actor con ganancia 0,01 (política inicial casi uniforme).
"""
from __future__ import annotations

import torch
from torch import nn

from env.rs4z.obs_v2 import CRITIC_DIM, ENT_DIM, N_ENT, N_MATES, OBS_DIM, SELF_DIM

N_ACTIONS = 18


def _mlp(sizes, act=nn.SiLU, last_act=True):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2 or last_act:
            layers.append(act())
    return nn.Sequential(*layers)


def _init(module, gain=2 ** 0.5):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain)
            nn.init.zeros_(m.bias)


class SceneEncoder(nn.Module):
    def __init__(self, hidden=128, ent_hidden=64):
        super().__init__()
        self.self_net = _mlp([SELF_DIM, hidden, hidden])
        self.mate_net = _mlp([ENT_DIM, ent_hidden, ent_hidden])
        self.rival_net = _mlp([ENT_DIM, ent_hidden, ent_hidden])
        self.out_dim = hidden + 4 * ent_hidden

    @staticmethod
    def _pool(x, present):
        m = present.unsqueeze(-1)
        cnt = m.sum(dim=-2).clamp(min=1.0)
        mean = (x * m).sum(dim=-2) / cnt
        mx = torch.where(m > 0, x, torch.full_like(x, -1e4)).max(dim=-2).values
        mx = torch.where(m.sum(dim=-2) > 0, mx, torch.zeros_like(mx))
        return mean, mx

    def forward(self, obs):
        s = obs[..., :SELF_DIM]
        ents = obs[..., SELF_DIM:].reshape(*obs.shape[:-1], N_ENT, ENT_DIM)
        mates, rivals = ents[..., :N_MATES, :], ents[..., N_MATES:, :]
        pm, pr = mates[..., 0], rivals[..., 0]
        hm, hr = self.mate_net(mates), self.rival_net(rivals)
        mm, mx = self._pool(hm, pm)
        rm, rx = self._pool(hr, pr)
        return torch.cat([self.self_net(s), mm, mx, rm, rx], dim=-1)


class ActorCritic(nn.Module):
    def __init__(self, hidden=256, enc_hidden=128, ent_hidden=64):
        super().__init__()
        self.actor_enc = SceneEncoder(enc_hidden, ent_hidden)
        self.critic_enc = SceneEncoder(enc_hidden, ent_hidden)
        d = self.actor_enc.out_dim
        self.pi = _mlp([d, hidden, hidden, N_ACTIONS], last_act=False)
        self.v = _mlp([d + CRITIC_DIM, hidden, hidden, 1], last_act=False)
        _init(self)
        nn.init.orthogonal_(self.pi[-1].weight, 0.01)
        nn.init.orthogonal_(self.v[-1].weight, 1.0)
        self.config = dict(hidden=hidden, enc_hidden=enc_hidden, ent_hidden=ent_hidden, obs_dim=OBS_DIM,
                           critic_dim=CRITIC_DIM, obs_version="rs4z-obs-v2")

    def logits(self, obs):
        return self.pi(self.actor_enc(obs))

    def value(self, obs, critic):
        return self.v(torch.cat([self.critic_enc(obs), critic], dim=-1)).squeeze(-1)

    def forward(self, obs, critic):
        return self.logits(obs), self.value(obs, critic)


class PolicyOnly(nn.Module):
    """Sólo el actor (export ONNX / despliegue): obs → logits. Sin entradas privilegiadas."""

    def __init__(self, model):
        super().__init__()
        self.enc = model.actor_enc
        self.pi = model.pi

    def forward(self, obs):
        return self.pi(self.enc(obs))
