"""Política X4: codificador de conjuntos para compañeros y rivales sobre la observación v3.

Arquitectura (plan E1; revisión §4.4: Deep Sets arXiv 1703.06114, MARLadona arXiv 2409.20326):
* bloque propio (SELF_DIM) → MLP;
* cada entidad (9 features) concatenada con el contexto propio-pelota (las primeras 15 features propias)
  → codificador compartido por tipo (compañero / rival) → pooling enmascarado (media y máximo);
* [propio, compañeros, rivales] → MLP → 18 logits (9 direcciones × patada).
Invariante a permutaciones dentro de compañeros y dentro de rivales; sirve para 4v3, 3v4, etc.

Un único modelo para los 8 jugadores (parámetros compartidos; MAPPO arXiv 2103.01955). El crítico del
RL usa el mismo tronco más features privilegiadas (estado sin retardo, marcador, reloj).
"""
from __future__ import annotations

import torch
from torch import nn

from env.rs4z import obs_v3

CTX = 15                       # features propias que acompañan a cada entidad (posición, pelota)


class SetPolicy(nn.Module):
    def __init__(self, hidden=256, ent_hidden=128, n_actions=18):
        super().__init__()
        S, E = obs_v3.SELF_DIM, obs_v3.ENT_DIM
        self.self_net = nn.Sequential(nn.Linear(S, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.mate_net = nn.Sequential(nn.Linear(E + CTX, ent_hidden), nn.ReLU(), nn.Linear(ent_hidden, ent_hidden), nn.ReLU())
        self.riv_net = nn.Sequential(nn.Linear(E + CTX, ent_hidden), nn.ReLU(), nn.Linear(ent_hidden, ent_hidden), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(hidden + 4 * ent_hidden, hidden), nn.ReLU(),
                                  nn.Linear(hidden, hidden), nn.ReLU())
        self.logits = nn.Linear(hidden, n_actions)
        self.hidden = hidden

    @staticmethod
    def _pool(x, present):
        m = present.unsqueeze(-1)
        cnt = m.sum(1).clamp(min=1.0)
        mean = (x * m).sum(1) / cnt
        mx = torch.where(m > 0, x, torch.full_like(x, -1e4)).max(1).values
        mx = torch.where(cnt > 0, mx, torch.zeros_like(mx)) * (m.sum(1) > 0).float()
        return torch.cat([mean, mx], -1)

    def features(self, obs):
        S, E = obs_v3.SELF_DIM, obs_v3.ENT_DIM
        own = obs[:, :S]
        ents = obs[:, S:].reshape(-1, obs_v3.N_ENT, E)
        ctx = own[:, :CTX].unsqueeze(1)
        mates = torch.cat([ents[:, :obs_v3.N_MATES], ctx.expand(-1, obs_v3.N_MATES, -1)], -1)
        rivs = torch.cat([ents[:, obs_v3.N_MATES:], ctx.expand(-1, obs_v3.N_ENT - obs_v3.N_MATES, -1)], -1)
        pm = ents[:, :obs_v3.N_MATES, 0]
        pr = ents[:, obs_v3.N_MATES:, 0]
        h = torch.cat([self.self_net(own), self._pool(self.mate_net(mates), pm), self._pool(self.riv_net(rivs), pr)], -1)
        return self.head(h)

    def forward(self, obs):
        return self.logits(self.features(obs))


def param_count(m):
    return sum(p.numel() for p in m.parameters())
