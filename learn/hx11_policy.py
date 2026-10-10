"""Política de HaxEleven (L0): codificador de conjuntos sobre la observación de `learn/hx11_data.py`.

Igual que la de X4 (`learn/x4_policy.py`; Deep Sets, 1703.06114):
* bloque propio → MLP;
* cada entidad, junto con el contexto propio y de la pelota, → codificador compartido por tipo (compañero o rival)
  → pooling enmascarado (media y máximo);
* [propio, compañeros, rivales] → MLP → 18 logits (9 direcciones × patada).

Acciones en bloque (`chunk` = K > 1; ACT, Zhao et al. 2023): la última capa da K × 18 logits, la acción de la decisión
actual y las de las K − 1 siguientes (cada 3 ticks). `forward` devuelve solo la actual (el resto del código no cambia);
`Ensemble` promedia en la inferencia lo predicho para este momento en las últimas decisiones (ensamble temporal), contra
la vibración sin perder la reacción de decidir cada 3 ticks.

Una sola red para los 22 lugares y todos los puestos, condicionada por el ancla y la línea (P2). Es invariante a
permutaciones dentro de compañeros y dentro de rivales, así que sirve con 10v11 o 9v11.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from learn.hx11_data import CTX, ENT_DIM, EXTRA_DIM, HOLD_REP, N_ENT, N_MATES, SELF_DIM

HIST_LO, HIST_HI = 14, 30        # tecla de patear sostenida y las 5 entradas propias anteriores (`hx11_data.build`)


class SetPolicy(nn.Module):
    def __init__(self, hidden=256, ent_hidden=128, n_actions=18, sin_historia=False, extra=False, hold=False, chunk=0):
        super().__init__()
        self.n_actions = n_actions
        self.chunk = max(int(chunk), 1)          # acciones en bloque: cuántas decisiones predice (1 = solo la actual)
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
        self.logits = nn.Linear(hidden, n_actions * self.chunk)
        if self.hold:
            self.hold_head = nn.Linear(hidden, len(HOLD_REP))
        self.config = dict(hidden=hidden, ent_hidden=ent_hidden, n_actions=n_actions, sin_historia=self.sin_historia,
                           extra=self.extra, hold=self.hold, chunk=self.chunk)

    @staticmethod
    def _pool(x, present):
        m = present.unsqueeze(-1)
        cnt = m.sum(1).clamp(min=1.0)
        mean = (x * m).sum(1) / cnt
        mx = torch.where(m > 0, x, torch.full_like(x, -1e4)).max(1).values
        mx = mx * (m.sum(1) > 0).float()
        return torch.cat([mean, mx], -1)

    def forward(self, own, ents):
        return self.logits(self.trunk(own, ents))[:, :self.n_actions]

    def forward_chunk(self, own, ents):
        """(B, K, 18): la acción de esta decisión y las de las K − 1 siguientes."""
        return self.logits(self.trunk(own, ents)).reshape(len(own), self.chunk, self.n_actions)

    def forward_hold(self, own, ents):
        """(logits de la acción, logits de la duración)."""
        h = self.trunk(own, ents)
        return self.logits(h)[:, :self.n_actions], self.hold_head(h)

    def trunk(self, own, ents):
        if self.sin_historia:
            own = torch.cat([own[:, :HIST_LO], torch.zeros_like(own[:, HIST_LO:HIST_HI]), own[:, HIST_HI:]], 1)
        ctx = own[:, :CTX].unsqueeze(1)
        mates = torch.cat([ents[:, :N_MATES], ctx.expand(-1, N_MATES, -1)], -1)
        rivs = torch.cat([ents[:, N_MATES:], ctx.expand(-1, N_ENT - N_MATES, -1)], -1)
        h = torch.cat([self.self_net(own), self._pool(self.mate_net(mates), ents[:, :N_MATES, 0]),
                       self._pool(self.riv_net(rivs), ents[:, N_MATES:, 0])], -1)
        return self.head(h)


class Ensemble:
    """Ensamble temporal de acciones en bloque (ACT, Zhao et al. 2023). En cada decisión la política predice las
    probabilidades de las próximas K decisiones; la de ahora es el promedio de lo que se predijo para este momento en
    las últimas K decisiones, con peso exp(−m · antigüedad): m > 0 favorece lo nuevo (más reactivo), m = 0 es parejo.
    Una clave por jugador; si una clave se saltea una decisión (`tick` no es el anterior + `every`), se reinicia."""

    def __init__(self, K, m=0.0, every=3):
        self.K, self.m, self.every = int(K), float(m), int(every)
        self.w = np.exp(-self.m * np.arange(self.K))
        self.buf = {}

    def step(self, keys, ticks, probs):
        """probs (B, K' >= K, A) de `forward_chunk` (softmax) → (B, A) para la decisión actual."""
        probs = np.asarray(probs, np.float64)
        A = probs.shape[-1]
        out = np.empty((len(keys), A))
        for b, (key, tk) in enumerate(zip(keys, ticks)):
            st = self.buf.get(key)
            if st is None or tk - st[3] != self.every:
                st = [np.zeros((self.K, A)), np.zeros(self.K), 0, tk]
            acc, den, ptr = st[0], st[1], st[2]
            for j in range(self.K):
                acc[(ptr + j) % self.K] += self.w[j] * probs[b, j]
                den[(ptr + j) % self.K] += self.w[j]
            out[b] = acc[ptr] / den[ptr]
            acc[ptr] = 0.0
            den[ptr] = 0.0
            st[2], st[3] = (ptr + 1) % self.K, tk
            self.buf[key] = st
        return out


def policy_logits(policy, own, ents, ens=None, keys=None, ticks=None):
    """Logits de la decisión actual; con `ens` (y una política en bloque), los del ensamble temporal."""
    if ens is None or getattr(policy, "chunk", 1) < 2:
        return policy(own, ents)
    p = torch.softmax(policy.forward_chunk(own, ents).detach().float(), -1).numpy()
    return torch.from_numpy(np.log(np.clip(ens.step(keys, ticks, p), 1e-12, None)).astype(np.float32))


class StickyNoise:
    """Ruido persistente para sortear la dirección (Gumbel-max con la misma tirada durante varias decisiones).

    Cada decisión sigue eligiendo argmax(log p + g) con g ~ Gumbel, así que su marginal es la de la política. Pero
    dos decisiones seguidas con distribuciones parecidas eligen lo mismo: el sorteo independiente hacía vibrar al bot
    (11 cambios de tecla por segundo contra 3,8 humanos), y la histéresis lo arreglaba sesgando hacia la acción anterior
    (quieto en el saque, sala del 2026-10-09). `renew`: probabilidad de renovar la tirada en cada decisión (1 = sorteo
    independiente, como antes)."""

    def __init__(self, renew, seed=0, renew_kick=1.0):
        self.renew = float(renew)
        self.renew_kick = float(renew_kick)
        self.rng = np.random.default_rng(seed)
        self.g = {}
        self.u = {}

    def get_kick(self, keys, fresh=None):
        """Tirada uniforme persistente para la tecla de patear: patea si u < P(patada | dirección). Con la misma u,
        mientras P no cambie mucho la tecla queda apretada (una sola patada), en vez de partirse en varias apretadas
        (la patada sale al apretar; el sorteo independiente cada 3 ticks pateaba un 33% más por segundo de control).

        `fresh` (máscara por fila): tirada nueva en cada decisión, como el sorteo independiente. Va en las pelotas
        paradas y el saque inicial (`piece_rows`): con la tirada persistente el que saca se guardaba la patada y metía
        la pelota empujando (laterales bien sacados 75% contra 91% con sorteo independiente, 400 laterales de
        desarrollo, 2026-10-09)."""
        out = np.empty(len(keys))
        for b, k in enumerate(keys):
            u = self.u.get(k)
            if u is None or self.rng.random() < self.renew_kick or (fresh is not None and fresh[b]):
                u = self.rng.random()
                self.u[k] = u
            out[b] = u
        return out

    def get(self, keys, fresh=None):
        """`fresh` (máscara por fila): tirada nueva (sorteo independiente). Va cerca de la pelota (`near_rows`): la
        dirección persistente bajaba el temblor a nivel humano (4,5 cambios/s) pero cerca de la pelota le quitaba
        reacción (patadas 0,66, toques 0,78 en desarrollo, 2026-10-09)."""
        out = np.empty((len(keys), 9))
        for b, k in enumerate(keys):
            g = self.g.get(k)
            if g is None or self.rng.random() < self.renew or (fresh is not None and fresh[b]):
                g = self.rng.gumbel(size=9)
                self.g[k] = g
            out[b] = g
        return out


NEAR_DIR_PX = 100.0    # distancia de borde a borde jugador–pelota bajo la cual la dirección va con sorteo independiente


def near_rows(me, ball, near=NEAR_DIR_PX):
    """Filas con el jugador a <= `near` px (borde a borde) de la pelota. me, ball: (B, 2) en px."""
    d = np.hypot(*(np.asarray(me, np.float64) - np.asarray(ball, np.float64)).T) - 15.0 - 8.0
    return d <= near


def piece_rows(phases):
    """Filas en pelota parada (2) o saque inicial (5): ahí la tecla de patear va con sorteo independiente."""
    ph = np.asarray(phases)
    return (ph == 2) | (ph == 5)


def sample(logits, temp_move=1.0, prev=None, hyst=0.0, gumbel=None, kick_u=None):
    """Acción 0..17 de los logits. La dirección se sortea de su marginal con temperatura `temp_move` (< 1 la hace más
    decidida); la patada, de P(patada | dirección) sin temperatura, para no apagar una acción rara.

    Histéresis (`hyst` > 0, con `prev` la acción anterior de cada fila): se mantiene la dirección anterior si su
    probabilidad es al menos `hyst` × la de la dirección más probable. Contra la vibración de la política sin historia
    propia (14,7 cambios de tecla por segundo contra 3,8 de los humanos, 2026-10-08)."""
    p = torch.softmax(logits.float(), -1).reshape(-1, 2, 9)          # [sin patada, con patada] × 9 direcciones
    pm = p.sum(1).clamp_min(1e-12)
    if temp_move != 1.0:
        pm = pm ** (1.0 / temp_move)
    if gumbel is not None:
        # Gumbel-max con la tirada dada (`StickyNoise`): misma marginal que el sorteo
        mv = (torch.log(pm / pm.sum(1, keepdim=True)) + torch.as_tensor(gumbel, dtype=pm.dtype)).argmax(1)
    else:
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
    u = torch.as_tensor(kick_u, dtype=pk.dtype) if kick_u is not None else torch.rand(len(mv))
    kick = (u < pk).long()
    return (mv + 9 * kick).numpy()
