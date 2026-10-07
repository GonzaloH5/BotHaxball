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
    """`n_pass` > 0 agrega las macro-acciones "pasarle al compañero k" (learn/x4_pass_macro): la salida pasa a ser
    log-probabilidades de 18 + n_pass acciones. Dos niveles para que arranque igual que la política de 18: compuerta
    p(pasar) = σ(g) (sesgo inicial −3, peso 0) y, si pasa, a qué compañero con un puntaje por entidad (puntero sobre el
    conjunto, invariante a permutaciones). `pass_mask` (B, n_pass) dice qué pases se ofrecen; sin ninguno, la compuerta
    se ignora y la política es la de 18."""

    def __init__(self, hidden=256, ent_hidden=128, n_actions=18, n_pass=0):
        super().__init__()
        S, E = obs_v3.SELF_DIM, obs_v3.ENT_DIM
        self.self_net = nn.Sequential(nn.Linear(S, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.mate_net = nn.Sequential(nn.Linear(E + CTX, ent_hidden), nn.ReLU(), nn.Linear(ent_hidden, ent_hidden), nn.ReLU())
        self.riv_net = nn.Sequential(nn.Linear(E + CTX, ent_hidden), nn.ReLU(), nn.Linear(ent_hidden, ent_hidden), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(hidden + 4 * ent_hidden, hidden), nn.ReLU(),
                                  nn.Linear(hidden, hidden), nn.ReLU())
        self.logits = nn.Linear(hidden, n_actions)
        self.hidden = hidden
        self.n_pass = int(n_pass)
        self.pass_ready = False         # "pase listo" (pass_table ready_only); lo fija quien carga el checkpoint
        if self.n_pass:
            assert self.n_pass == obs_v3.N_MATES
            self.pass_gate = nn.Linear(hidden, 1)
            nn.init.zeros_(self.pass_gate.weight)
            nn.init.constant_(self.pass_gate.bias, -3.0)
            self.pass_score = nn.Sequential(nn.Linear(hidden + ent_hidden, ent_hidden), nn.ReLU(),
                                            nn.Linear(ent_hidden, 1))

    @staticmethod
    def _pool(x, present):
        m = present.unsqueeze(-1)
        cnt = m.sum(1).clamp(min=1.0)
        mean = (x * m).sum(1) / cnt
        mx = torch.where(m > 0, x, torch.full_like(x, -1e4)).max(1).values
        mx = torch.where(cnt > 0, mx, torch.zeros_like(mx)) * (m.sum(1) > 0).float()
        return torch.cat([mean, mx], -1)

    def features(self, obs, with_mates=False):
        S, E = obs_v3.SELF_DIM, obs_v3.ENT_DIM
        own = obs[:, :S]
        ents = obs[:, S:].reshape(-1, obs_v3.N_ENT, E)
        ctx = own[:, :CTX].unsqueeze(1)
        mates = torch.cat([ents[:, :obs_v3.N_MATES], ctx.expand(-1, obs_v3.N_MATES, -1)], -1)
        rivs = torch.cat([ents[:, obs_v3.N_MATES:], ctx.expand(-1, obs_v3.N_ENT - obs_v3.N_MATES, -1)], -1)
        pm = ents[:, :obs_v3.N_MATES, 0]
        pr = ents[:, obs_v3.N_MATES:, 0]
        me = self.mate_net(mates)
        h = torch.cat([self.self_net(own), self._pool(me, pm), self._pool(self.riv_net(rivs), pr)], -1)
        h = self.head(h)
        return (h, me) if with_mates else h

    def forward(self, obs, pass_mask=None):
        if not self.n_pass:
            return self.logits(self.features(obs))
        h, me = self.features(obs, with_mates=True)
        prim = torch.log_softmax(self.logits(h).float(), -1)
        if pass_mask is None:
            pass_mask = torch.zeros(obs.shape[0], self.n_pass, dtype=torch.bool, device=obs.device)
        pass_mask = pass_mask.bool()
        any_pass = pass_mask.any(-1, keepdim=True)
        g = self.pass_gate(h).float()
        score = self.pass_score(torch.cat([h.unsqueeze(1).expand(-1, self.n_pass, -1), me], -1))[..., 0].float()
        score = torch.where(pass_mask, score, torch.full_like(score, -1e9))
        lp_pass = torch.nn.functional.logsigmoid(g) + torch.log_softmax(score, -1)
        lp_prim = prim + torch.where(any_pass, torch.nn.functional.logsigmoid(-g), torch.zeros_like(g))
        lp_pass = torch.where(pass_mask, lp_pass, torch.full_like(lp_pass, -1e9))
        return torch.cat([lp_prim, lp_pass], -1)


def load_policy(ck, **kw):
    """SetPolicy desde un checkpoint (dict de torch.load) con su tamaño y sus macro-acciones."""
    m = SetPolicy(hidden=ck.get("hidden", 256), n_pass=ck.get("n_pass", 0), **kw)
    m.load_state_dict(ck["model"])
    m.pass_ready = bool(ck.get("pass_ready", False))
    return m


def param_count(m):
    return sum(p.numel() for p in m.parameters())
