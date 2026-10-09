"""Política de HaxEleven (L0): codificador de conjuntos sobre la observación de `learn/hx11_data.py`.

Igual que la de X4 (`learn/x4_policy.py`; Deep Sets, 1703.06114):
* bloque propio → MLP;
* cada entidad, junto con el contexto propio y de la pelota, → codificador compartido por tipo (compañero o rival)
  → pooling enmascarado (media y máximo);
* [propio, compañeros, rivales] → MLP → 18 logits (9 direcciones × patada).

Una sola red para los 22 lugares y todos los puestos, condicionada por el ancla y la línea (P2). Es invariante a
permutaciones dentro de compañeros y dentro de rivales, así que sirve con 10v11 o 9v11.
"""
from __future__ import annotations

import torch
from torch import nn

from learn.hx11_data import CTX, ENT_DIM, EXTRA_DIM, HOLD_REP, N_ENT, N_MATES, SELF_DIM

HIST_LO, HIST_HI = 14, 30        # tecla de patear sostenida y las 5 entradas propias anteriores (`hx11_data.build`)


class SetPolicy(nn.Module):
    def __init__(self, hidden=256, ent_hidden=128, n_actions=18, sin_historia=False, extra=False, hold=False):
        super().__init__()
        self.hold = bool(hold)                   # cabeza de duración (`hx11_data.HOLD_REP`)
        self.extra = bool(extra)                 # observación v2 (rasgos de "voy yo", `hx11_data.engage`)
        self_dim = SELF_DIM + (EXTRA_DIM if self.extra else 0)
        self.sin_historia = bool(sin_historia)   # brazo contra el copycat (de Haan 2019): sin las entradas propias
        self.self_net = nn.Sequential(nn.Linear(self_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.mate_net = nn.Sequential(nn.Linear(ENT_DIM + CTX, ent_hidden), nn.ReLU(),
                                      nn.Linear(ent_hidden, ent_hidden), nn.ReLU())
        self.riv_net = nn.Sequential(nn.Linear(ENT_DIM + CTX, ent_hidden), nn.ReLU(),
                                     nn.Linear(ent_hidden, ent_hidden), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(hidden + 4 * ent_hidden, hidden), nn.ReLU(),
                                  nn.Linear(hidden, hidden), nn.ReLU())
        self.logits = nn.Linear(hidden, n_actions)
        if self.hold:
            self.hold_head = nn.Linear(hidden, len(HOLD_REP))
        self.config = dict(hidden=hidden, ent_hidden=ent_hidden, n_actions=n_actions, sin_historia=self.sin_historia,
                           extra=self.extra, hold=self.hold)

    @staticmethod
    def _pool(x, present):
        m = present.unsqueeze(-1)
        cnt = m.sum(1).clamp(min=1.0)
        mean = (x * m).sum(1) / cnt
        mx = torch.where(m > 0, x, torch.full_like(x, -1e4)).max(1).values
        mx = mx * (m.sum(1) > 0).float()
        return torch.cat([mean, mx], -1)

    def forward(self, own, ents):
        return self.logits(self.trunk(own, ents))

    def forward_hold(self, own, ents):
        """(logits de la acción, logits de la duración)."""
        h = self.trunk(own, ents)
        return self.logits(h), self.hold_head(h)

    def trunk(self, own, ents):
        if self.sin_historia:
            own = torch.cat([own[:, :HIST_LO], torch.zeros_like(own[:, HIST_LO:HIST_HI]), own[:, HIST_HI:]], 1)
        ctx = own[:, :CTX].unsqueeze(1)
        mates = torch.cat([ents[:, :N_MATES], ctx.expand(-1, N_MATES, -1)], -1)
        rivs = torch.cat([ents[:, N_MATES:], ctx.expand(-1, N_ENT - N_MATES, -1)], -1)
        h = torch.cat([self.self_net(own), self._pool(self.mate_net(mates), ents[:, :N_MATES, 0]),
                       self._pool(self.riv_net(rivs), ents[:, N_MATES:, 0])], -1)
        return self.head(h)


def sample(logits, temp_move=1.0, prev=None, hyst=0.0):
    """Acción 0..17 de los logits. La dirección se sortea de su marginal con temperatura `temp_move` (< 1 la hace más
    decidida); la patada, de P(patada | dirección) sin temperatura, para no apagar una acción rara.

    Histéresis (`hyst` > 0, con `prev` la acción anterior de cada fila): se mantiene la dirección anterior si su
    probabilidad es al menos `hyst` × la de la dirección más probable. Contra la vibración de la política sin historia
    propia (14,7 cambios de tecla por segundo contra 3,8 de los humanos, 2026-10-08)."""
    p = torch.softmax(logits.float(), -1).reshape(-1, 2, 9)          # [sin patada, con patada] × 9 direcciones
    pm = p.sum(1).clamp_min(1e-12)
    if temp_move != 1.0:
        pm = pm ** (1.0 / temp_move)
    mv = torch.multinomial(pm / pm.sum(1, keepdim=True), 1).squeeze(1)
    ar = torch.arange(len(mv))
    if hyst > 0 and prev is not None:
        pr = torch.as_tensor(prev, dtype=torch.long)
        pv = pr.clamp_min(0) % 9                                     # prev < 0: sin acción anterior (no se mantiene)
        # "quieto" (dirección 0) nunca se sostiene: en un saque todos están quietos y la histéresis los dejaba trabados
        # (sala del 2026-10-09: 20 de 22 bots sin dirección el 100% del tiempo)
        keep = (pr >= 0) & (pv != 0) & (pm[ar, pv] >= hyst * pm.max(1).values)
        mv = torch.where(keep, pv, mv)
    pk = p[ar, 1, mv] / p[ar, :, mv].sum(1).clamp_min(1e-12)
    kick = (torch.rand(len(mv)) < pk).long()
    return (mv + 9 * kick).numpy()
