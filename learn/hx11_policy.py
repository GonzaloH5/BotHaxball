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

from learn.hx11_data import (CTX, ENT_DIM, ESTADO_DIM, EXTRA_DIM, HOLD_REP, MEM_DIM, MOVE_DX, MOVE_DY, N_ENT, N_MATES,
                             SELF_DIM)

HIST_LO, HIST_HI = 14, 30        # tecla de patear sostenida y las 5 entradas propias anteriores (`hx11_data.build`)


class SetPolicy(nn.Module):
    def __init__(self, hidden=256, ent_hidden=128, n_actions=18, sin_historia=False, extra=False, hold=False, chunk=0):
        super().__init__()
        self.n_actions = n_actions
        self.chunk = max(int(chunk), 1)          # acciones en bloque: cuántas decisiones predice (1 = solo la actual)
        self.hold = bool(hold)                   # cabeza de duración (`hx11_data.HOLD_REP`)
        # bits de observación: 1 = v2 (rasgos de "voy yo", `hx11_data.engage`), 2 = v3 (estado del juego,
        # `hx11_data.estado`), 4 = v4 (memoria de 2 s: cuánto se movió cada uno); True de los modelos viejos vale 1
        self.extra = int(extra)
        self.mem = bool(self.extra & 4)
        self_dim = (SELF_DIM + (EXTRA_DIM if self.extra & 1 else 0) + (ESTADO_DIM if self.extra & 2 else 0)
                    + (MEM_DIM if self.mem else 0))
        ent_in = ENT_DIM + CTX + (MEM_DIM if self.mem else 0)     # memoria al final: [entidad, contexto, memoria]
        self.sin_historia = bool(sin_historia)   # brazo contra el copycat (de Haan 2019): sin las entradas propias
        self.self_net = nn.Sequential(nn.Linear(self_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.mate_net = nn.Sequential(nn.Linear(ent_in, ent_hidden), nn.ReLU(),
                                      nn.Linear(ent_hidden, ent_hidden), nn.ReLU())
        self.riv_net = nn.Sequential(nn.Linear(ent_in, ent_hidden), nn.ReLU(),
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
        mem = ents[..., ENT_DIM:]                     # memoria (vacía sin el bit 4), va después del contexto
        ents = ents[..., :ENT_DIM]
        mates = torch.cat([ents[:, :N_MATES], ctx.expand(-1, N_MATES, -1), mem[:, :N_MATES]], -1)
        rivs = torch.cat([ents[:, N_MATES:], ctx.expand(-1, N_ENT - N_MATES, -1), mem[:, N_MATES:]], -1)
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


# Regla de patada en las pelotas paradas propias (opcional: cerebro de la sala `--regla-saques`, simulador)
BAR_MIN = 2.8      # masa inversa mínima de la pelota para patear con barra (p10 humano: córner 3,3, tiro libre 2,8)
BAR_WAIT = 600     # pasados estos ticks de espera se patea igual (no trabarse)
LAT_MIN_DEG = 6.0  # lateral: patear desde afuera de la pelota, con al menos este ángulo hacia adentro (p10 humano 6°,
                   # p50 27°, 3.124 laterales de entrenamiento; en el simulador entra desde 5°: desde adentro la pelota
                   # corre por la línea y solo vale si la toca un compañero). Con 22 bots, el 49% de los laterales se
                   # repetía (patada paralela, nadie cerca); con la regla, 21%. Con compañeros humanos (colector desde
                   # laterales reales) baja de 90% a 84% bien: el bot bloqueado mete la pelota empujando, y la patada por
                   # la línea a un compañero cercano sí servía. Por eso va aparte (`lateral`), 2026-10-10


def kick_rule_ok(me, ball, kind, team, binv, wait, bar=True, lateral=True):
    """True si el que saca una pelota parada de su equipo puede patear. Lateral (1): desde afuera de la pelota con ángulo
    >= LAT_MIN_DEG hacia la cancha. Córner (2): del lado de la línea de fondo. Saque de arco (3) y tiro libre (4): del
    lado de su arco (patea hacia adelante); en esos tres, además, con la barra alta (`bar`). Pasados BAR_WAIT ticks de
    espera, siempre. Coordenadas del mapa en px; los demás tipos (y el lateral con `lateral` falso), siempre."""
    if kind not in (1, 2, 3, 4) or wait >= BAR_WAIT or (kind == 1 and not lateral):
        return True
    if kind == 1:
        side = 1.0 if ball[1] > 0 else -1.0
        return bool(np.degrees(np.arctan2((me[1] - ball[1]) * side, abs(me[0] - ball[0]))) >= LAT_MIN_DEG)
    if kind == 2:
        ok = (me[0] - ball[0]) * (1.0 if ball[0] > 0 else -1.0) > 0.0
    else:
        ok = (me[0] - ball[0]) * (1.0 if team == 0 else -1.0) < 0.0
    return bool(ok and (not bar or binv >= BAR_MIN))


def seen_now(w, L, s, d):
    """Posiciones (px) del bot `s` y de la pelota al aplicarse la decisión, estimadas con lo que ve el bot: la última
    fila vista L (ya con el retardo d) más la velocidad × (d + 1). Sin esto, el bot que llega a la pelota todavía se ve
    lejos y la regla bloquea patadas buenas (laterales en el simulador, 2026-10-10)."""
    k = d + 1
    me = w.pos[L, s].astype(np.float64) / 10 + w.vel[L, s].astype(np.float64) / 1000 * k
    ball = w.ball[L].astype(np.float64) / 10 + w.bvel[L].astype(np.float64) / 1000 * k
    return me, ball


SIDE_Y = 1000.0           # juego abierto: pelota pegada a la banda
# Constantes de la variante dura (v3, 2026-10-10), elegidas en el lazo cerrado entre humanos con unos 310 min de juego
# cerca de la pelota por variante (h2). Patadas atrás que terminan en un rival o afuera, por minuto: sin regla 0,228,
# primera regla (−0,5, 35°, 700 px, sin mirar rivales) 0,146, v3 0,087 (humano 0,049). Afuera: 0,79 → 0,55. Patadas
# efectivas: 5,31 → 4,81 (humano 6,50).
BACK_DX = -0.2            # "hacia atrás": componente x de la dirección (coordenadas del equipo) menor que esto (v2:
                          # con 0.0 / v3 la puerta P3 fallaba en patadas de DEF y MED: despejan de costado)
BACK_CONE = 25.0          # pase atrás válido: un compañero a menos de BACK_CONE° de la dirección de la patada…
BACK_DIST = 500.0         # …a menos de BACK_DIST px de la pelota…
LANE_PX = 45.0            # …sin un rival a menos de LANE_PX px de la línea del pase (0: no se mira)…
MARK_PX = 70.0            # …ni a menos de MARK_PX px del compañero (0: no se mira)


def open_kick_ok(me, ball, team, mates, rivals=None):
    """Regla de patada en juego abierto (la dirección de la patada es la de jugador → pelota). Bloquea:
    * pegado a la banda (|y| > 1000), patear hacia afuera (|dy| > 0,7 hacia la línea): en el banco el 56% sale;
    * en campo propio, patear hacia atrás (dx < BACK_DX en coordenadas del equipo) sin un compañero libre en esa
      dirección (cono, distancia, línea de pase y marca: constantes de arriba). Los humanos patean hacia atrás en el 6%
      de sus patadas, casi siempre como pase (0,24 por minuto llegan a un compañero, 0,04 a un rival o afuera); el bot,
      sin la regla, 0,08 y 0,43. En el partido real del 2026-10-09 la pateó para atrás por la banda (lateral del rival
      en campo propio) y lo echaron.
    me, ball: (2,) px; mates: (k, 2) px de los compañeros (sin él); rivals: (r, 2) px."""
    v = np.asarray(ball, np.float64) - np.asarray(me, np.float64)
    n = np.hypot(*v)
    if n < 1e-6:
        return True
    v = v / n
    if abs(ball[1]) > SIDE_Y and v[1] * np.sign(ball[1]) > 0.7:
        return False
    att = 1.0 if team == 0 else -1.0
    if ball[0] * att < 0 and v[0] * att < BACK_DX:
        if not len(mates):
            return False
        b = np.asarray(ball, np.float64)
        w = np.asarray(mates, np.float64) - b
        dist = np.hypot(*w.T)
        cos = (w @ v) / np.maximum(dist, 1e-6)
        cand = np.nonzero((dist < BACK_DIST) & (cos > np.cos(np.radians(BACK_CONE))))[0]
        rv = np.asarray(rivals, np.float64).reshape(-1, 2) - b if rivals is not None else np.zeros((0, 2))
        for j in cand:
            if len(rv) and LANE_PX > 0:
                u = w[j] / max(dist[j], 1e-6)
                along = rv @ u
                perp = np.abs(rv[:, 0] * u[1] - rv[:, 1] * u[0])
                if ((along > 0) & (along < dist[j]) & (perp < LANE_PX)).any():
                    continue
            if len(rv) and MARK_PX > 0 and (np.hypot(*(rv - w[j]).T) < MARK_PX).any():
                continue
            return True
        return False
    return True


def open_kick_blocked(w, L, s, d):
    """`open_kick_ok` con lo que ve el bot (fila L, retardo d; posiciones llevadas al momento de la decisión)."""
    if int(w.phase[L]) != 0:
        return False
    team = 1 if s >= 11 else 0
    me, ball = seen_now(w, L, s, d)
    k = d + 1
    pos = w.pos[L].astype(np.float64) / 10 + w.vel[L].astype(np.float64) / 1000 * k
    on = w.pid[L] >= 0
    mates = [pos[q] for q in range(11 * team, 11 * team + 11) if q != s and on[q]]
    rivals = [pos[q] for q in range(11 * (1 - team), 11 * (1 - team) + 11) if on[q]]
    return not open_kick_ok(me, ball, team, np.array(mates).reshape(-1, 2), np.array(rivals).reshape(-1, 2))


KEEP_PX = 90.0            # pelota parada o saque inicial que saca otro: no acercarse a la pelota a menos de esto…
KEEP_SLOW_PX = 140.0      # …y entre KEEP_PX y esto, no ir hacia ella
KICKOFF_KIND = 9          # código de "saque_inicial" en `sin_kinds` (los tipos de pelota parada van de 1 a 5)
_DIRS = np.stack([MOVE_DX, MOVE_DY], 1).astype(np.float64)


def _dir_toward(v):
    """Dirección 1..8 (coordenadas del equipo) más parecida al vector v (0 si es nulo)."""
    n = np.hypot(*v)
    if n < 1e-6:
        return 0
    c = (_DIRS[1:] @ v) / (np.hypot(*_DIRS[1:].T) * n)
    return 1 + int(np.argmax(c))


def keep_away(own, act, sin_kinds=()):
    """Red de seguridad de pelotas paradas (partidos reales del 2026-10-09/10: el bot se paraba al lado del compañero
    que sacaba un córner y tomaba el saque inicial después del gol). En una pelota parada o saque inicial que saca
    otro (del rival, o del equipo y de un tipo en `sin_kinds`; KICKOFF_KIND para el saque inicial), el bot no patea,
    no se acerca a la pelota a menos de KEEP_PX (si está más cerca, se aleja) ni va hacia ella hasta KEEP_SLOW_PX.
    Usa solo la observación (coordenadas del equipo), así que vale igual para cualquier puesto, en la sala y en el
    simulador. own: (B, ≥ SELF_DIM); act: (B,) acciones 0..17 en coordenadas del equipo. Devuelve las acciones."""
    out = np.asarray(act).copy()
    for i in range(len(out)):
        o = own[i]
        ph = int(np.argmax(o[31:37]))
        if ph not in (1, 2, 3, 5):
            continue
        pk = int(np.argmax(o[37:45]))
        pt = int(np.argmax(o[45:48]))                 # 0 propio, 1 rival, 2 nadie
        if ph == 5:
            ajena = KICKOFF_KIND in sin_kinds
        else:
            ajena = pt == 1 or (pt == 0 and pk in sin_kinds)
        if not ajena:
            continue
        to_ball = np.asarray(o[4:6], np.float64) * 1000
        dist = float(o[10]) * 1000
        mv = int(out[i]) % 9
        if dist < KEEP_PX:
            mv = _dir_toward(-to_ball)
        elif dist < KEEP_SLOW_PX and mv and _DIRS[mv] @ to_ball > 0:
            mv = 0
        out[i] = mv
    return out


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
