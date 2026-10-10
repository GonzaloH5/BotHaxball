"""Refuerzo A3 (piloto corto, aprobado por el usuario el 2026-10-10): un protagonista por partido sobre la población de
imitadores (L2 de docs/PLAN_HAXELEVEN.md).

Lo que falta después de L0 es llevar la pelota: con 22 bots el tiempo en contacto es 0,4 del humano, las rachas largas
0,3 y las patadas cargadas 0,43 (un bot entre humanos). Copiar la conducción no lo arregló (ajuste c1: retiene más pero
patea menos y falla la puerta P3). Acá se mira el resultado.

* **Partidos:** ventanas reales del banco de entrenamiento (saques iniciales y juego abierto), con las reglas de la
  sala (offside, faltas, powershot). 21 jugadores con la política de la población (`--poblacion`, e1: la mejor
  coordinación de 22 bots), temperatura 0,7, ruido persistente y las reglas de patada de pelota parada; el protagonista
  (un jugador de campo al azar) con la red a ajustar (`--init`, h2: el bot de sala), muestreado igual que en la sala
  (temperatura 0,7 en la dirección, ruido persistente, sin patear córners, tiros libres ni saques de arco propios).
* **Premios del protagonista** (`PREMIOS`, se cambian con `--premios`): goles del equipo (±0,5); en juego abierto,
  al terminar cada posesión suya, el terreno ganado (avance de la pelota hacia el arco rival desde que la tomó hasta el
  próximo toque, con signo) más un poco si la recibe un compañero y menos un poco si la recibe un rival o sale;
  recuperación; patada cargada; offside propio. Calibrado con los humanos del banco: el 38% de sus posesiones termina en
  un compañero y el 47% en un rival, pero la pelota avanza en promedio igual (101 y 182 px). Con la versión anterior
  (premio por contacto y pérdida cara, rl_a3_p1) el bot aprendió a no soltarla: entre humanos, patadas 0,77 → 0,48.
* **Atado a la imitación:** PPO con penalización KL(π || π_ref) exacta sobre las 18 acciones con la misma temperatura,
  con π_ref = la red de partida congelada y coeficiente adaptativo hacia `--kl-objetivo` (regularización hacia la
  política humana, como piKL: Jacob et al. 2022). Lo que se busca es jugar mejor sin dejar de parecer humano: la puerta
  P3 (`learn.hx11_puerta`) y la retención (`learn.hx11_closed_loop`) son el control.
* **Crítico:** misma arquitectura que la política (cabeza de 1 salida), arranca con el tronco de `--init`; las primeras
  `--calentar` iteraciones solo se entrena el crítico.

* **`--entorno humanos`** (2026-10-10, después de p1–p3 entre bots sin mejora entre humanos): el protagonista juega
  ventanas de partidos grabados (`--ventana` ticks, desde un momento de juego abierto, como el jugador de campo más
  cercano a la pelota) y los otros 21 repiten sus teclas grabadas, como en la evaluación de retención
  (`learn.hx11_closed_loop --cerca`) y como en la sala, entre humanos. La ventana termina al salir del juego abierto.
  Los grabados no reaccionan al bot: por eso ventanas cortas, sin offside ni faltas (igual que la evaluación), y la
  puerta P3 como control.

Guarda `model.pt` (formato de L0: sirve para la puerta, la retención y el cerebro de la sala), `ultimo.pt` (todo, para
`--reanudar`) e `historia.json`.

  python -m learn.hx11_rl --init runs/hx11_l0/l0_ajuste_corner_h2/model_final.pt \\
      --poblacion runs/hx11_l0/l0_quienva_e1/model.pt --out runs/hx11_l0/rl_a3_p1
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from env.hx11 import contract as C
from env.hx11 import kernel as K
from env.hx11.core import HX11Env
from learn.hx11_closed_loop import FRAME_SKIP, action_of_bits, bits_of_action, load_state
from learn.hx11_data import D_MAX, D_MIN, MAX_BACK, ROT_MOVE, SELF_DIM, Match, build, load_match
from learn.hx11_policy import SetPolicy, StickyNoise, near_rows, piece_rows, sample
from learn.hx11_selfplay import kick_blocked, store_all, window_sp
from tools.hx11_d0 import KICKOFF, OPEN

ROOT = Path(__file__).resolve().parent.parent
CONTACT_PX = C.PLAYER_R + C.BALL_R + 4.0      # control: la pelota a <= 4 px del borde del jugador
OFFSIDE_PX = C.PLAYER_R + C.BALL_R + 6.0
PREMIOS = dict(
    gol=0.5,            # gol del equipo (en contra: −0,5); poco por protagonista y con mucho ruido de los otros 21
    avance=0.0005,      # por px (con signo) que avanzó la pelota hacia el arco rival entre que la tomó y el próximo toque
    pase=0.05,          # su posesión termina en un compañero (juego abierto)
    perdida=-0.05,      # su posesión termina en un rival, o la tira afuera y saca el rival
    recupera=0.05,      # se la saca a un rival
    contacto=0.0,       # por tick con la pelota pegada en juego abierto, hasta `contacto_max` ticks por racha
    contacto_max=150,
    cargada=0.1,        # patada cargada (powershot)
    offside=-0.3,       # offside propio
)


def tempered_logp(logits, temp):
    """log π(a) de las 18 acciones tal como se muestrea en la sala (`hx11_policy.sample`): la dirección con su marginal
    a temperatura `temp`, la patada con P(patada | dirección) sin temperatura."""
    lp = torch.log_softmax(logits.float(), -1).reshape(-1, 2, 9)        # [sin patada, con patada] × 9 direcciones
    lpm = torch.logsumexp(lp, 1)                                        # marginal de la dirección
    lpk = lp - lpm.unsqueeze(1)                                         # P(patada | dirección)
    return (lpk + torch.log_softmax(lpm / temp, -1).unsqueeze(1)).reshape(-1, 18)


def load_net(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = SetPolicy(**ck["config"])
    net.load_state_dict(ck["state"])
    return net


def make_critic(policy):
    crit = SetPolicy(**dict(policy.config, n_actions=1))
    sd = {k: v for k, v in policy.state_dict().items() if not k.startswith("logits.")}
    crit.load_state_dict(sd, strict=False)
    torch.nn.init.zeros_(crit.logits.weight)
    torch.nn.init.zeros_(crit.logits.bias)
    return crit


def game_starts(matches, H, rng):
    """(partido, t0): saques iniciales y 6 momentos de juego abierto por partido."""
    out = []
    for mi, m in enumerate(matches):
        ar = np.arange(m.T)
        ko = np.nonzero((m.phase == KICKOFF) & (ar >= MAX_BACK) & (ar < m.T - H - 1))[0]
        op = np.nonzero((m.phase == OPEN) & (ar >= MAX_BACK) & (ar < m.T - H - 1))[0]
        out += [(mi, int(t)) for t in ko[np.r_[True, np.diff(ko) > 1]]] if len(ko) else []
        out += [(mi, int(t)) for t in rng.choice(op, size=min(6, len(op)), replace=False)]
    return out


class Games:
    """n partidos en paralelo, todos con la misma duración (se reinician juntos)."""

    def __init__(self, n, matches, starts, H, seed, renew=0.33, renew_kick=0.33):
        self.n, self.matches, self.starts, self.H = n, matches, starts, H
        self.rng = np.random.default_rng(seed)
        self.env = HX11Env(n, seed=seed)
        self.noise = StickyNoise(renew, seed, renew_kick)
        self.renew = renew
        self.episode = 0
        self.reset()

    def reset(self):
        env, rng = self.env, self.rng
        self.W, self.D, self.prot, self.team = [], [], [], []
        self.k = 0
        self.episode += 1
        for i in range(self.n):
            mi, t0 = self.starts[int(rng.integers(len(self.starts)))]
            m = self.matches[mi]
            w = window_sp(m, t0, self.H)
            w.score[:] = 0
            load_state(env, i, w, MAX_BACK)
            env.ri[i, K.RI_OFFSIDE_ON], env.ri[i, K.RI_FOULS_ON], env.ri[i, K.RI_PS_ON] = 1, 1, 1
            if m.phase[t0] == KICKOFF:
                env.ri[i, K.RI_MODE] = K.KICKOFF
                env.ri[i, K.RI_KO] = 1
                env.ri[i, K.RI_KO_TEAM] = max(int(m.pteam[t0]), 0)
            kk = MAX_BACK
            on = (w.pid[kk:kk + self.H + 1] == w.pid[kk]).all(0) & (w.pid[kk] >= 0)    # el mismo toda la ventana
            gk = np.where(np.arange(22) < 11, w.gk[kk, 0], w.gk[kk, 1])
            field = np.nonzero(on & (w.pid[kk] != gk))[0]
            p = int(rng.choice(field))
            self.W.append(w)
            self.D.append(rng.integers(D_MIN, D_MAX + 1, 22))
            self.prot.append(p)
            self.team.append(int(p >= 11))
        self.acts = np.zeros((self.n, 22), np.int64)
        self.holder = [False] * self.n           # el protagonista es el último que tocó (juego abierto)
        self.gain_x = np.zeros(self.n)
        self.streak = np.zeros(self.n, np.int64)


def protagonist_no_kick(w: Match, L, s):
    """Como el bot de sala (`--sin-saques corner,tiro_libre,saque_arco`): esas pelotas paradas propias las saca otro."""
    team = 1 if s >= 11 else 0
    return int(w.phase[L]) == 2 and int(w.pteam[L]) == team and int(w.pkind[L]) in (2, 3, 4)


def tick_reward(g: Games, i, pre, ev, st):
    """Premio del protagonista en el partido i por el tick recién simulado. `pre`: estado antes del tick."""
    env, p, tp = g.env, g.prot[i], g.team[i]
    ri = env.ri[i]
    att = 1.0 if tp == 0 else -1.0
    r = 0.0
    score = ri[[K.RI_SCORE0, K.RI_SCORE1]]
    d_for = int(score[tp] - pre["score"][tp])
    d_ag = int(score[1 - tp] - pre["score"][1 - tp])
    if d_for > 0 or d_ag > 0:
        r += PREMIOS["gol"] * (d_for - d_ag)
        st["goles_favor"] += d_for
        st["goles_contra"] += d_ag
    ball = env.pos[i, 0]
    me = env.pos[i, env.fp + p]
    dist = float(np.hypot(*(ball - me)))
    mode = int(ri[K.RI_MODE])
    last = int(ri[K.RI_LAST])
    open_both = pre["mode"] == K.OPEN and mode == K.OPEN
    if ev == K.EV_OUT and g.holder[i] and int(ri[K.RI_TEAM]) != tp:
        r += PREMIOS["perdida"] + PREMIOS["avance"] * (ball[0] - g.gain_x[i]) * att
        st["perdidas"] += 1
        g.holder[i] = False
    elif ev == K.EV_OFFSIDE and (pre["offmask"] >> p) & 1 and dist <= OFFSIDE_PX:
        r += PREMIOS["offside"]
        st["offsides"] += 1
        g.holder[i] = False
    elif open_both and last >= 0 and last != pre["last"]:
        if g.holder[i]:
            r += PREMIOS["avance"] * (ball[0] - g.gain_x[i]) * att
            if (last >= 11) == (tp == 1):
                r += PREMIOS["pase"]
                st["pases"] += 1
            else:
                r += PREMIOS["perdida"]
                st["perdidas"] += 1
            g.holder[i] = False
        if last == p:
            g.holder[i] = True
            g.gain_x[i] = ball[0]
            if pre["last"] >= 0 and (pre["last"] >= 11) != (tp == 1):
                r += PREMIOS["recupera"]
                st["recuperas"] += 1
    elif mode != K.OPEN:
        g.holder[i] = False
    if mode == K.OPEN:
        st["ticks_abiertos"] += 1
        if dist <= CONTACT_PX:
            g.streak[i] += 1
            st["contacto"] += 1
            if g.streak[i] <= PREMIOS["contacto_max"]:
                r += PREMIOS["contacto"]
        else:
            g.streak[i] = 0
    if last == p and int(ri[K.RI_CURVE]) == C.PS_CURVE_TICKS and int(ri[K.RI_PS_RESET]) > 0 and pre["curve"] != C.PS_CURVE_TICKS:
        r += PREMIOS["cargada"]
        st["cargadas"] += 1
    return r


def rollout(g: Games, prot, pop, critic, T, temp, every=FRAME_SKIP):
    """T decisiones del protagonista por partido. Devuelve el lote y las estadísticas."""
    n = g.n
    buf = {k: [] for k in ("own", "ents", "act", "logp", "val", "rew", "done")}
    st = {k: 0 for k in ("goles_favor", "goles_contra", "pases", "perdidas", "recuperas", "offsides", "contacto",
                         "ticks_abiertos", "cargadas")}
    for _ in range(T):
        kk = MAX_BACK + g.k
        rows_o, rows_e, who = [], [], []
        for i, w in enumerate(g.W):
            slots = np.nonzero(w.pid[kk] >= 0)[0]
            o, e, _, _ = build(w, np.full(len(slots), kk), slots, g.D[i][slots], pop.extra)
            rows_o.append(o)
            rows_e.append(e)
            who += [(i, int(s)) for s in slots]
        ot, et = np.concatenate(rows_o), np.concatenate(rows_e)
        is_p = np.array([s == g.prot[i] for i, s in who])
        mp = torch.from_numpy(is_p)
        keys = [(g.episode, i, s) for i, s in who]
        fresh = near_rows([g.env.pos[i, g.env.fp + s] for i, s in who], [g.env.pos[i, 0] for i, s in who])
        gum = g.noise.get(keys, fresh)
        ku = g.noise.get_kick(keys, piece_rows([g.env.ri[i, K.RI_MODE] for i, s in who]))
        with torch.no_grad():
            lg = torch.zeros(len(who), 18)
            lg[~mp] = pop(torch.from_numpy(ot[~is_p]), torch.from_numpy(et[~is_p]))
            po = torch.from_numpy(np.ascontiguousarray(ot[is_p][:, :SELF_DIM]))
            pe = torch.from_numpy(et[is_p])
            lg[mp] = prot(po, pe)
            val = critic(po, pe)[:, 0]
        act = sample(lg, temp, gumbel=gum, kick_u=ku)
        lp = tempered_logp(lg[mp], temp).gather(1, torch.as_tensor(act[is_p]).unsqueeze(1))[:, 0]
        order = [i for (i, s), q in zip(who, is_p) if q]
        assert order == list(range(n)), "un protagonista por partido"
        buf["own"].append(po.numpy())
        buf["ents"].append(pe.numpy())
        buf["act"].append(act[is_p].astype(np.int64))
        buf["logp"].append(lp.numpy())
        buf["val"].append(val.numpy())
        for j, ((i, s), a) in enumerate(zip(who, act)):
            a = int(a)
            L = kk - int(g.D[i][s]) - 1
            if a >= 9:
                if s == g.prot[i]:
                    if protagonist_no_kick(g.W[i], L, s):
                        a %= 9
                elif kick_blocked(g.W[i], L, s, int(g.D[i][s]), True, True):
                    a %= 9
            if s >= 11:
                a = int(ROT_MOVE[a % 9] + 9 * (a // 9))
            g.acts[i, s] = a
            g.W[i].inp[kk, s] = bits_of_action(np.array([a]))[0]
        rew = np.zeros(n, np.float32)
        for f in range(every):
            kk = MAX_BACK + g.k
            if f > 0:
                for w in g.W:
                    w.inp[kk] = w.inp[kk - 1]
            pre = [dict(score=g.env.ri[i, [K.RI_SCORE0, K.RI_SCORE1]].copy(), mode=int(g.env.ri[i, K.RI_MODE]),
                        last=int(g.env.ri[i, K.RI_LAST]), offmask=int(g.env.ri[i, K.RI_OFF_MASK]),
                        curve=int(g.env.ri[i, K.RI_CURVE])) for i in range(n)]
            ev = g.env.step(g.acts, frames=1)
            for i, w in enumerate(g.W):
                store_all(g.env, i, w, kk + 1)
                w.score[kk + 1] = g.env.ri[i, [K.RI_SCORE0, K.RI_SCORE1]]
                rew[i] += tick_reward(g, i, pre[i], int(ev[i, 0]), st)
            g.k += 1
        done = g.k + every > g.H
        buf["rew"].append(rew)
        buf["done"].append(np.full(n, done))
        if done:
            g.reset()
    with torch.no_grad():
        kk = MAX_BACK + g.k
        rows_o, rows_e = [], []
        for i, w in enumerate(g.W):
            s = np.array([g.prot[i]])
            o, e, _, _ = build(w, np.full(1, kk), s, g.D[i][s], False)
            rows_o.append(o)
            rows_e.append(e)
        last_v = critic(torch.from_numpy(np.concatenate(rows_o)), torch.from_numpy(np.concatenate(rows_e)))[:, 0].numpy()
    out = {k: np.stack(v) for k, v in buf.items()}          # (T, n, ...)
    out["last_v"] = last_v
    return out, st


def replay_starts(matches, H, rng, per_match=200, min_open=300):
    """(partido, t0, lugar): momentos con al menos `min_open` ticks de juego abierto grabado por delante, con el jugador
    de campo más cercano a la pelota presente toda la ventana."""
    out = []
    for mi, m in enumerate(matches):
        op = m.phase == OPEN
        run = np.zeros(m.T + 1, np.int64)
        for t in range(m.T - 1, -1, -1):
            run[t] = run[t + 1] + 1 if op[t] else 0
        cand = np.nonzero((run[:m.T] >= min_open) & (np.arange(m.T) >= MAX_BACK) & (np.arange(m.T) < m.T - H - 1))[0]
        if not len(cand):
            continue
        got = 0
        for t0 in rng.choice(cand, size=min(per_match * 2, len(cand)), replace=False):
            ss = np.nonzero((m.pid[t0] >= 0) & (m.pid[t0] == m.pid[t0 + H]) & (m.pid[t0] == m.pid[t0 - MAX_BACK]))[0]
            gkp = np.r_[np.full(11, m.gk[t0, 0]), np.full(11, m.gk[t0, 1])]
            ss = ss[m.pid[t0, ss] != gkp[ss]]
            if not len(ss):
                continue
            d = np.hypot(*(m.pos[t0, ss].astype(np.float64) - m.ball[t0].astype(np.float64)).T)
            out.append((mi, int(t0), int(ss[int(np.argmin(d))])))
            got += 1
            if got >= per_match:
                break
    return out


class ReplayGames:
    """n ventanas grabadas en paralelo: el protagonista con la red, los otros 21 con sus teclas grabadas. Cada partido
    se reinicia por su cuenta al terminar su ventana (siempre en el borde de una decisión)."""

    def __init__(self, n, matches, starts, H, seed, renew=0.33, renew_kick=0.33):
        self.n, self.matches, self.starts, self.H = n, matches, starts, H
        self.rng = np.random.default_rng(seed)
        self.env = HX11Env(n, seed=seed)
        self.noise = StickyNoise(renew, seed, renew_kick)
        self.episode = np.zeros(n, np.int64)
        self.W, self.log, self.D, self.prot, self.team = [None] * n, [None] * n, [0] * n, [0] * n, [0] * n
        self.k = np.zeros(n, np.int64)
        self.acts = np.zeros((n, 22), np.int64)
        self.holder = [False] * n
        self.gain_x = np.zeros(n)
        self.streak = np.zeros(n, np.int64)
        for i in range(n):
            self.reset_one(i)

    def reset_one(self, i):
        mi, t0, s = self.starts[int(self.rng.integers(len(self.starts)))]
        m = self.matches[mi]
        w = window_sp(m, t0, self.H)
        load_state(self.env, i, w, MAX_BACK)
        self.W[i], self.log[i] = w, window_sp(m, t0, self.H)
        self.D[i] = int(self.rng.integers(D_MIN, D_MAX + 1))
        self.prot[i], self.team[i] = s, int(s >= 11)
        self.k[i] = 0
        self.episode[i] += 1
        self.holder[i] = False
        self.streak[i] = 0


def rollout_replay(g: ReplayGames, prot, critic, T, temp, every=FRAME_SKIP):
    """Como `rollout`, entre humanos grabados."""
    n = g.n
    buf = {k: [] for k in ("own", "ents", "act", "logp", "val", "rew", "done")}
    st = {k: 0 for k in ("goles_favor", "goles_contra", "pases", "perdidas", "recuperas", "offsides", "contacto",
                         "ticks_abiertos", "cargadas")}

    def observe():
        rows_o, rows_e = [], []
        for i in range(n):
            kk = MAX_BACK + int(g.k[i])
            o, e, _, _ = build(g.W[i], np.array([kk]), np.array([g.prot[i]]), np.array([g.D[i]]), False)
            rows_o.append(o)
            rows_e.append(e)
        return torch.from_numpy(np.concatenate(rows_o)), torch.from_numpy(np.concatenate(rows_e))

    for _ in range(T):
        po, pe = observe()
        keys = [(int(g.episode[i]), i) for i in range(n)]
        fresh = near_rows([g.env.pos[i, g.env.fp + g.prot[i]] for i in range(n)], [g.env.pos[i, 0] for i in range(n)])
        with torch.no_grad():
            lg = prot(po, pe)
            val = critic(po, pe)[:, 0]
        act = sample(lg, temp, gumbel=g.noise.get(keys, fresh),
                     kick_u=g.noise.get_kick(keys, piece_rows([g.env.ri[i, K.RI_MODE] for i in range(n)])))
        lp = tempered_logp(lg, temp).gather(1, torch.as_tensor(act).unsqueeze(1))[:, 0]
        buf["own"].append(po.numpy())
        buf["ents"].append(pe.numpy())
        buf["act"].append(act.astype(np.int64))
        buf["logp"].append(lp.numpy())
        buf["val"].append(val.numpy())
        held = np.array([int(ROT_MOVE[int(a) % 9] + 9 * (int(a) // 9)) if g.prot[i] >= 11 else int(a)
                         for i, a in enumerate(act)], np.int64)
        rew = np.zeros(n, np.float32)
        ended = np.zeros(n, bool)
        for f in range(every):
            for i in range(n):
                kk = MAX_BACK + int(g.k[i])
                g.acts[i] = action_of_bits(g.log[i].inp[kk])
                g.acts[i, g.prot[i]] = held[i]
                g.W[i].inp[kk, g.prot[i]] = bits_of_action(np.array([held[i]]))[0]
            pre = [dict(score=g.env.ri[i, [K.RI_SCORE0, K.RI_SCORE1]].copy(), mode=int(g.env.ri[i, K.RI_MODE]),
                        last=int(g.env.ri[i, K.RI_LAST]), offmask=int(g.env.ri[i, K.RI_OFF_MASK]),
                        curve=int(g.env.ri[i, K.RI_CURVE])) for i in range(n)]
            ev = g.env.step(g.acts, frames=1)
            for i in range(n):
                if ended[i]:
                    continue
                kk = MAX_BACK + int(g.k[i])
                store_all(g.env, i, g.W[i], kk + 1)
                rew[i] += tick_reward(g, i, pre[i], int(ev[i, 0]), st)
                g.k[i] += 1
                if g.env.ri[i, K.RI_MODE] != K.OPEN or g.k[i] + every > g.H:
                    ended[i] = True
        buf["rew"].append(rew)
        buf["done"].append(ended.copy())
        for i in np.nonzero(ended)[0]:
            g.reset_one(int(i))
    with torch.no_grad():
        po, pe = observe()
        last_v = critic(po, pe)[:, 0].numpy()
    out = {k: np.stack(v) for k, v in buf.items()}
    out["last_v"] = last_v
    return out, st


def gae(rew, val, done, last_v, gamma, lam):
    T = len(rew)
    adv = np.zeros_like(rew)
    nxt = last_v
    acc = np.zeros_like(last_v)
    for t in reversed(range(T)):
        nonterm = 1.0 - done[t].astype(np.float32)
        delta = rew[t] + gamma * nxt * nonterm - val[t]
        acc = delta + gamma * lam * nonterm * acc
        adv[t] = acc
        nxt = val[t]
    return adv, adv + val


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init", required=True, help="red del protagonista (y referencia congelada para la KL)")
    ap.add_argument("--poblacion", default=None, help="red de los otros 21 (entorno bots)")
    ap.add_argument("--entorno", default="bots", choices=["bots", "humanos"],
                    help="bots: población de imitadores; humanos: ventanas grabadas con los otros 21 repitiendo")
    ap.add_argument("--ventana", type=int, default=480, help="entorno humanos: ticks de cada ventana")
    ap.add_argument("--out", required=True)
    ap.add_argument("--split-file", default="runs/hx11_l0/split.json")
    ap.add_argument("--partidos", type=int, default=40)
    ap.add_argument("--n-env", type=int, default=12)
    ap.add_argument("--segundos", type=int, default=60, help="duración de cada partido")
    ap.add_argument("--T", type=int, default=128, help="decisiones del protagonista por partido y por iteración")
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--calentar", type=int, default=3, help="iteraciones con solo el crítico")
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--lr-critico", type=float, default=3e-4)
    ap.add_argument("--clip", type=float, default=0.1)
    ap.add_argument("--epocas", type=int, default=4)
    ap.add_argument("--lote", type=int, default=512)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--kl-objetivo", type=float, default=0.03, help="KL(π || π_ref) por decisión buscada (nats)")
    ap.add_argument("--beta", type=float, default=1.0, help="coeficiente inicial de la KL (adaptativo)")
    ap.add_argument("--guardar", type=int, default=5)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--premios", default="", help="premios a cambiar, p. ej. cargada=0.5,contacto=0.001 (sin comillas: "
                                                  "pasa por cmd en la PC prestada)")
    ap.add_argument("--reanudar", action="store_true")
    a = ap.parse_args()
    if a.premios:
        extra = {k.strip(): float(v) for k, v in (kv.split("=") for kv in a.premios.split(",") if kv.strip())}
        assert set(extra) <= set(PREMIOS), f"premios desconocidos: {set(extra) - set(PREMIOS)}"
        PREMIOS.update(extra)
    print("premios:", json.dumps(PREMIOS), flush=True)
    torch.set_num_threads(a.threads)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    sp = json.loads((ROOT / a.split_file).read_text(encoding="utf-8"))
    rng = np.random.default_rng(a.seed)
    names = sorted(rng.permutation(sp["train"])[:a.partidos].tolist())
    modos = json.loads((ROOT / "data" / "hx11" / "modos.json").read_text(encoding="utf-8"))
    matches = [load_match(nm, modos) for nm in names]
    H = a.segundos * 60 if a.entorno == "bots" else a.ventana
    starts = game_starts(matches, H, rng) if a.entorno == "bots" else replay_starts(matches, H, rng)
    print(f"datos: {len(matches)} partidos, {len(starts)} comienzos ({time.time() - t0:.0f} s)", flush=True)
    prot = load_net(a.init)
    ref = load_net(a.init).eval()
    pop = load_net(a.poblacion).eval() if a.entorno == "bots" else None
    assert not prot.extra, "el protagonista usa la observación de L0 (sin --extra)"
    critic = make_critic(prot)
    opt = torch.optim.Adam(prot.parameters(), lr=a.lr)
    opt_c = torch.optim.Adam(critic.parameters(), lr=a.lr_critico)
    beta, it0, hist = a.beta, 0, []
    ult = out / "ultimo.pt"
    if a.reanudar and ult.exists():
        ck = torch.load(ult, map_location="cpu", weights_only=False)
        prot.load_state_dict(ck["state"])
        critic.load_state_dict(ck["critico"])
        opt.load_state_dict(ck["opt"])
        opt_c.load_state_dict(ck["opt_c"])
        beta, it0, hist = ck["beta"], ck["paso"], ck["hist"]
        print(f"reanudado en la iteración {it0}", flush=True)
    games = (Games(a.n_env, matches, starts, H, a.seed + 1000 * it0) if a.entorno == "bots"
             else ReplayGames(a.n_env, matches, starts, H, a.seed + 1000 * it0))
    for it in range(it0, a.iters):
        ti = time.time()
        prot.eval()
        critic.eval()
        b, st = (rollout(games, prot, pop, critic, a.T, a.temp) if a.entorno == "bots"
                 else rollout_replay(games, prot, critic, a.T, a.temp))
        tr = time.time() - ti
        adv, ret = gae(b["rew"], b["val"], b["done"], b["last_v"], a.gamma, a.lam)
        N = a.T * a.n_env
        own = torch.from_numpy(b["own"].reshape(N, -1))
        ents = torch.from_numpy(b["ents"].reshape(N, *b["ents"].shape[2:]))
        act = torch.from_numpy(b["act"].reshape(N))
        logp_old = torch.from_numpy(b["logp"].reshape(N))
        adv_t = torch.from_numpy(adv.reshape(N))
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        ret_t = torch.from_numpy(ret.reshape(N))
        with torch.no_grad():
            lp_ref = torch.cat([tempered_logp(ref(own[j:j + 2048], ents[j:j + 2048]), a.temp)
                                for j in range(0, N, 2048)])
        prot.train()
        critic.train()
        train_pol = it >= a.calentar
        kls, clips, vls, ents_ = [], [], [], []
        for _ in range(a.epocas):
            perm = torch.randperm(N)
            for j in range(0, N, a.lote):
                mb = perm[j:j + a.lote]
                v = critic(own[mb], ents[mb])[:, 0]
                vl = ((v - ret_t[mb]) ** 2).mean()
                opt_c.zero_grad()
                vl.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                opt_c.step()
                vls.append(float(vl.detach()))
                if not train_pol:
                    continue
                lp = tempered_logp(prot(own[mb], ents[mb]), a.temp)
                lpa = lp.gather(1, act[mb].unsqueeze(1))[:, 0]
                ratio = torch.exp(lpa - logp_old[mb])
                pg = -torch.min(ratio * adv_t[mb], ratio.clamp(1 - a.clip, 1 + a.clip) * adv_t[mb]).mean()
                kl = (lp.exp() * (lp - lp_ref[mb])).sum(-1).mean()
                loss = pg + beta * kl
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(prot.parameters(), 1.0)
                opt.step()
                kls.append(float(kl.detach()))
                clips.append(float(((ratio - 1).abs() > a.clip).float().mean()))
                ents_.append(float(-(lp.exp() * lp).sum(-1).mean().detach()))
        kl_m = float(np.mean(kls)) if kls else 0.0
        if train_pol:
            if kl_m > 1.5 * a.kl_objetivo:
                beta *= 1.5
            elif kl_m < a.kl_objetivo / 1.5:
                beta /= 1.5
            beta = float(np.clip(beta, 1e-3, 100.0))
        mins = max(st["ticks_abiertos"], 1) / 3600 / a.n_env   # minutos de juego abierto por protagonista
        row = dict(iter=it + 1, muestras=(it + 1) * N, premio_decision=round(float(b["rew"].mean()), 5),
                   contacto_s_min=round(st["contacto"] / 60 / a.n_env / mins, 3),
                   cargadas_min=round(st["cargadas"] / a.n_env / mins, 3),
                   pases_min=round(st["pases"] / a.n_env / mins, 3),
                   perdidas_min=round(st["perdidas"] / a.n_env / mins, 3),
                   recuperas_min=round(st["recuperas"] / a.n_env / mins, 3),
                   offsides_10min=round(st["offsides"] / a.n_env / mins * 10, 3),
                   goles=[st["goles_favor"], st["goles_contra"]], kl_ref=round(kl_m, 4), beta=round(beta, 4),
                   clip=round(float(np.mean(clips)), 3) if clips else None,
                   entropia=round(float(np.mean(ents_)), 3) if ents_ else None,
                   v_loss=round(float(np.mean(vls)), 5), seg_rollout=round(tr), seg=round(time.time() - ti),
                   total=round(time.time() - t0))
        hist.append(row)
        print(json.dumps(row), flush=True)
        last = it + 1 == a.iters
        if (it + 1) % a.guardar == 0 or last:
            model = dict(state=prot.state_dict(), config=prot.config, paso=it + 1, args=vars(a), premios=PREMIOS)
            torch.save(model, out / "model.pt")
            torch.save(dict(model, critico=critic.state_dict(), opt=opt.state_dict(), opt_c=opt_c.state_dict(),
                            beta=beta, hist=hist, terminado=last), ult)
            (out / "historia.json").write_text(json.dumps(hist, indent=1), encoding="utf-8")
            if (it + 1) % (a.guardar * 4) == 0:
                torch.save(model, out / f"model_{it + 1}.pt")


if __name__ == "__main__":
    main()
