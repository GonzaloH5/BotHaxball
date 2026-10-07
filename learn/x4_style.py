"""Recompensa de estilo humano para el RL X4 (AMP: Peng et al. 2021, arXiv 2104.02180; GAIL: Ho y Ermon 2016,
arXiv 1606.03476).

Un discriminador mira el juego visto desde un equipo, en el marco del equipo (ataca hacia +x), con dos partes:
* posiciones: la pelota, sus 4 jugadores y los 4 rivales (posición y velocidad) hace 1 s, 0,5 s, 0,25 s y ahora, con
  los jugadores ordenados por distancia a la pelota (espaciado, apoyos, presión);
* cadena de toques: los últimos 6 toques de los últimos 4 s (de quién: propio o rival, el mismo jugador o un compañero;
  dónde; con qué velocidad salió la pelota; cuánto tardó desde el anterior; presión sobre el que tocó y su compañero más
  cercano) y el contexto actual. Con sólo posiciones el discriminador no distinguía cómo termina un duelo (premiaba
  igual "pasa" que "pierde"); con la cadena, pasar al salir del duelo paga más (prueba offline del 2026-10-07).
Aprende a separar las grabaciones humanas de Sanguchito (+1) del agente (−1) por mínimos cuadrados, con penalización de
gradiente sobre los humanos (AMP). Cada decisión cuyos últimos 2 s fueron juego abierto le da al equipo
r = max(0, 1 − ¼(D − 1)²), centrada con su media móvil: los saques y el saque inicial valen 0, así que el estilo no
empuja ni a buscar ni a evitar pelotas muertas. No es de suma cero: cada equipo cobra por su propio parecido.

Los rasgos se calculan igual para grabaciones (`episode_windows`) y para el entrenamiento (`History`): muestreo antes
de cada decisión, patadas de esa decisión y dueño de la pelota con la regla de `tools.x4_metrics._touch_owner`
(`test_history_matches_the_recorded_episode`). Si no, el discriminador aprendería la diferencia de cálculo.

Por qué (diagnóstico del 2026-10-07, `docs/PRELANZAMIENTO.md` §5): el ancla KL copia las decisiones de la imitación una
por una, con sus defectos, y se apaga con λ → 0,05; las métricas humanas sólo controlan. Ninguna premia que los
partidos del propio agente se parezcan a los humanos. En duelos 1v1 el agente pasa el 13% de las veces y pierde la
pelota el 61% (humanos: 22% y 49%; la imitación, 15% y 53%).

  python -m learn.x4_style --bank --out data/x4_style_bank.npz            # rasgos humanos (una vez)
  python -m learn.x4_style --probe --agent runs/x4_cola/rl_checkpoint/evals/ep_01950.npz,...   # ¿los separa?
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from env.rs4z import kernel as K

ROOT = Path(__file__).resolve().parent.parent
OFFSETS = (20, 10, 5, 0)              # muestreos hacia atrás (cada uno = 3 ticks = 1/20 s): 1 s, 0,5 s, 0,25 s y ahora
BODIES = 9                            # pelota + 8 lugares
POS_FEAT = len(OFFSETS) * BODIES * 4
KT, HOR = 6, 80                       # últimos 6 toques dentro de 80 muestreos (4 s)
EV_FEAT = 10                          # por toque: existe, propio/rival, mismo jugador, x, y, vx, vy, espera, presión, apoyo
CHAIN_FEAT = 1 + KT * EV_FEAT + 6     # + cuántos toques, + contexto (pelota y distancia del propio y del rival más cercanos)
FEAT = POS_FEAT + CHAIN_FEAT
OPEN = 40                             # muestreos de juego abierto que pide un momento (2 s)
DEPTH = OPEN + 1
SCALE = np.array([1150.0, 670.0, 6.0, 6.0], np.float32)     # x, y, vx, vy
TEAM = np.array([0, 0, 0, 0, 1, 1, 1, 1])
# columnas de y y vy (para el espejo vertical con que se aumentan los datos del discriminador)
Y_COLS = np.zeros(FEAT, bool)
Y_COLS[1:POS_FEAT:4] = True
Y_COLS[3:POS_FEAT:4] = True
for _k in range(KT):
    Y_COLS[POS_FEAT + 1 + _k * EV_FEAT + 4] = True
    Y_COLS[POS_FEAT + 1 + _k * EV_FEAT + 6] = True
Y_COLS[POS_FEAT + 1 + KT * EV_FEAT + 1] = True
Y_COLS[POS_FEAT + 1 + KT * EV_FEAT + 3] = True


# ------------------------------------------------------------------------------------- rasgos
def features(win, team):
    """Posiciones. win: (M, K, 9, 4) estados del mundo (pelota y los 8 lugares; x, y, vx, vy) en los momentos de
    OFFSETS, del más viejo al actual; team: (M,) 0 rojo, 1 azul. Devuelve (M, POS_FEAT) en el marco del equipo:
    pelota, sus 4 jugadores y los 4 rivales, ordenados por distancia a la pelota en el momento actual (el mismo orden en
    los K momentos)."""
    team = np.asarray(team, np.int64)
    w = np.array(win, np.float32, copy=True)
    M, Kn = w.shape[:2]
    s = np.where(team == 1, -1.0, 1.0).astype(np.float32)[:, None, None]
    w[..., 0] *= s
    w[..., 2] *= s
    now = w[:, -1]
    rows = np.arange(M)[:, None]

    def by_distance(idx):
        d = np.hypot(now[rows, idx, 0] - now[:, None, 0, 0], now[rows, idx, 1] - now[:, None, 0, 1])
        return np.take_along_axis(idx, np.argsort(d, 1, kind="stable"), 1)

    red, blue = np.arange(1, 5)[None].repeat(M, 0), np.arange(5, 9)[None].repeat(M, 0)
    mates = np.where(team[:, None] == 0, red, blue)
    rivals = np.where(team[:, None] == 0, blue, red)
    sel = np.concatenate([np.zeros((M, 1), np.int64), by_distance(mates), by_distance(rivals)], 1)
    out = w[np.arange(M)[:, None, None], np.arange(Kn)[None, :, None], sel[:, None, :]] / SCALE
    return out.reshape(M, -1)


def touch_owner(ball, pos, kicked, ball_r):
    """Dueño de la pelota de cada muestreo (lote): el lugar en contacto o que pateó (la patada manda, si no el más
    cercano); -1 nadie, -2 los dos equipos. La misma regla que `tools.x4_metrics._touch_owner`."""
    from tools.x4_metrics import CONTACT_EPS, PLAYER_R
    d = np.hypot(pos[..., 0] - ball[..., None, 0], pos[..., 1] - ball[..., None, 1])
    touch = (d <= PLAYER_R + np.asarray(ball_r)[..., None] + CONTACT_EPS) | kicked
    red, blue = touch[..., :4].any(-1), touch[..., 4:].any(-1)
    score = np.where(touch, np.where(kicked, -1.0, d), np.inf)
    owner = np.where(touch.any(-1), np.argmin(score, -1), -1)
    return np.where(red & blue, -2, owner), d


def event_attrs(pos_now, ball_now, ball_vel_next, slot, prev_slot):
    """(E, 7) rasgos de mundo de cada toque: x, y de la pelota; vx, vy de la pelota que sale (muestreo siguiente);
    presión (rival más cercano al que tocó); apoyo (compañero más cercano); si lo tocó el mismo jugador que el toque
    anterior."""
    E = len(slot)
    me = pos_now[np.arange(E), slot]
    d = np.hypot(pos_now[..., 0] - me[:, None, 0], pos_now[..., 1] - me[:, None, 1])
    same_team = TEAM[None, :] == TEAM[slot][:, None]
    press = np.where(~same_team, d, np.inf).min(1)
    other = same_team & (np.arange(8)[None, :] != slot[:, None])
    support = np.where(other, d, np.inf).min(1)
    return np.stack([ball_now[:, 0], ball_now[:, 1], ball_vel_next[:, 0], ball_vel_next[:, 1],
                     np.minimum(press, 600.0), np.minimum(support, 600.0), (prev_slot == slot).astype(float)], 1)


def chain_features(ev_slot, ev_attr, ev_wait, ev_age, ball_now, pos_now, team):
    """Cadena de toques. (M, CHAIN_FEAT) desde los últimos KT toques de cada fila (del más nuevo al más viejo;
    ev_slot −1 = no hay): ev_attr (M, KT, 7) de `event_attrs`, ev_wait (M, KT) muestreos desde el toque anterior,
    ev_age (M, KT) muestreos desde el toque hasta ahora (cuentan los de menos de HOR). ball_now (M, 4), pos_now
    (M, 8, 2); team (M,)."""
    M = len(team)
    team = np.asarray(team, np.int64)
    s = np.where(team == 1, -1.0, 1.0)[:, None]
    ok = (ev_slot >= 0) & (ev_age < HOR)
    out = np.zeros((M, CHAIN_FEAT), np.float32)
    out[:, 0] = ok.sum(1) / KT
    f = np.zeros((M, KT, EV_FEAT), np.float32)
    slot = np.maximum(ev_slot, 0)
    f[..., 0] = ok
    f[..., 1] = np.where(TEAM[slot] == team[:, None], 1.0, -1.0)
    f[..., 2] = ev_attr[..., 6]
    f[..., 3] = ev_attr[..., 0] * s / 1150
    f[..., 4] = ev_attr[..., 1] / 670
    f[..., 5] = ev_attr[..., 2] * s / 6
    f[..., 6] = ev_attr[..., 3] / 6
    f[..., 7] = np.minimum(ev_wait, HOR) / 20.0
    f[..., 8] = ev_attr[..., 4] / 300
    f[..., 9] = ev_attr[..., 5] / 300
    out[:, 1:1 + KT * EV_FEAT] = (f * ok[..., None]).reshape(M, -1)
    c = 1 + KT * EV_FEAT
    out[:, c] = ball_now[:, 0] * s[:, 0] / 1150
    out[:, c + 1] = ball_now[:, 1] / 670
    out[:, c + 2] = ball_now[:, 2] * s[:, 0] / 6
    out[:, c + 3] = ball_now[:, 3] / 6
    d = np.hypot(pos_now[..., 0] - ball_now[:, None, 0], pos_now[..., 1] - ball_now[:, None, 1])
    own = TEAM[None, :] == team[:, None]
    out[:, c + 4] = np.minimum(np.where(own, d, np.inf).min(1), 600.0) / 300
    out[:, c + 5] = np.minimum(np.where(~own, d, np.inf).min(1), 600.0) / 300
    return out


def open_play(env):
    """Juego abierto en cada partido de un `RS4ZEnv`: sin saque inicial, sin saque y sin cobro pendiente."""
    return (env.ri[:, K.RI_KO] == 0) & (env.ri[:, K.RI_TEAM] < 0) & (env.ri[:, K.RI_PEND] == 0)


class History:
    """Lo que necesita el discriminador de cada partido de un `RS4ZEnv` durante el entrenamiento: los últimos DEPTH
    estados antes de cada decisión (y si eran juego abierto) y los últimos KT toques. Uso: `prime(env)` antes de la
    primera decisión; por decisión, `feats, rows = observe(env, ev["kicked"])` justo después de `env.step`, y
    `reset(filas, env)` después de reiniciar partidos. Los rasgos son los del estado de antes del paso con las patadas del
    paso, como en las grabaciones."""

    def __init__(self, n):
        self.buf = np.zeros((n, DEPTH, BODIES, 4), np.float32)
        self.open = np.zeros((n, DEPTH), bool)
        self.i = 0
        self.ev_slot = np.full((n, KT), -1, np.int64)
        self.ev_attr = np.zeros((n, KT, 7))
        self.ev_wait = np.zeros((n, KT), np.int64)
        self.ev_step = np.zeros((n, KT), np.int64)
        self.prev_owner = np.full(n, -1, np.int64)
        self.step = 0
        self.primed = False

    @staticmethod
    def _state(env):
        s = np.zeros((env.N, BODIES, 4), np.float32)
        s[:, 0, :2] = env.pos[:, 0]
        s[:, 0, 2:] = env.vel[:, 0]
        s[:, 1:, :2] = env.player_pos
        s[:, 1:, 2:] = env.player_vel
        return s

    def prime(self, env):
        """Estado antes de la primera decisión."""
        self.buf[:, self.i] = self._state(env)
        self.open[:, self.i] = open_play(env)
        self.primed = True

    def observe(self, env, kicked):
        """Después de `env.step`: registra los toques del paso (con el estado de antes) y devuelve los rasgos de los
        partidos con 2 s de juego abierto, (2M, FEAT) (primero el rojo de cada fila, después el azul), y esas M filas.
        Guarda el estado nuevo para la decisión siguiente."""
        if not self.primed:
            raise RuntimeError("History.prime(env) antes de la primera decisión")
        N = len(self.buf)
        pre = self.buf[:, self.i].copy()
        post = self._state(env)
        kicked = np.asarray(kicked, bool)
        owner, _ = touch_owner(pre[:, 0, :2], pre[:, 1:, :2], kicked, env.radius[:, 0])
        safe = np.maximum(owner, 0)
        new = (owner >= 0) & ((owner != self.prev_owner) | kicked[np.arange(N), safe])
        rows = np.flatnonzero(new)
        if len(rows):
            had = self.ev_slot[rows, 0] >= 0
            wait = np.where(had, self.step - self.ev_step[rows, 0], HOR)
            attr = event_attrs(pre[rows, 1:, :2], pre[rows, 0, :2], post[rows, 0, 2:], owner[rows],
                               self.ev_slot[rows, 0])
            for a in (self.ev_slot, self.ev_attr, self.ev_wait, self.ev_step):
                a[rows, 1:] = a[rows, :-1]
            self.ev_slot[rows, 0] = owner[rows]
            self.ev_attr[rows, 0] = attr
            self.ev_wait[rows, 0] = wait
            self.ev_step[rows, 0] = self.step
        self.prev_owner = np.where(owner >= 0, owner, np.where(owner == -2, -1, self.prev_owner))
        valid = np.flatnonzero(self.open.all(1))
        feats = np.zeros((2 * len(valid), FEAT), np.float32)
        if len(valid):
            idx = [(self.i - o) % DEPTH for o in OFFSETS]
            win = self.buf[valid][:, idx]
            args = (self.ev_slot[valid], self.ev_attr[valid], self.ev_wait[valid], self.step - self.ev_step[valid],
                    pre[valid, 0], pre[valid, 1:, :2])
            M = len(valid)
            for k, team in enumerate((0, 1)):
                tm = np.full(M, team)
                feats[k * M:(k + 1) * M, :POS_FEAT] = features(win, tm)
                feats[k * M:(k + 1) * M, POS_FEAT:] = chain_features(*args, tm)
        self.step += 1
        self.i = (self.i + 1) % DEPTH
        self.buf[:, self.i] = post
        self.open[:, self.i] = open_play(env)
        return feats, valid

    def reset(self, rows, env):
        """Partidos que empiezan de nuevo: su historia y sus toques dejan de valer; el estado actual es el del arranque."""
        rows = np.atleast_1d(rows)
        if not len(rows):
            return
        self.open[rows] = False
        self.buf[rows, self.i] = self._state(env)[rows]
        self.open[rows, self.i] = open_play(env)[rows]
        self.ev_slot[rows] = -1
        self.prev_owner[rows] = -1


def episode_windows(ep, every=1):
    """(2M, FEAT) rasgos de los dos equipos (primero los M del rojo, después los del azul) de un
    `tools.x4_metrics.Episode` (grabación humana o partido grabado del agente; muestreo cada 3 ticks, estado antes de la
    decisión, patadas de la decisión) en los momentos con los últimos 2 s de juego abierto."""
    T = len(ep.ball)
    if T <= OPEN:
        return np.zeros((0, FEAT), np.float32)
    st = np.zeros((T, BODIES, 4), np.float32)
    st[:, 0] = ep.ball[:, :4]
    st[:, 1:, :2] = ep.pos
    st[:, 1:, 2:] = ep.vel
    bad = np.concatenate([[0], np.cumsum(~np.asarray(ep.open_play, bool))])
    t = np.arange(OPEN, T)
    t = t[bad[t + 1] - bad[t - OPEN] == 0][::every]
    if not len(t):
        return np.zeros((0, FEAT), np.float32)
    # toques de todo el tramo (la misma regla que History.observe)
    kicked = np.asarray(ep.kicked, bool)
    owner, _ = touch_owner(st[:, 0, :2], st[:, 1:, :2], kicked, np.full(T, ep.ball_r, np.float32))
    ev_t, ev_p, prev = [], [], -1
    for k in range(T):
        o = owner[k]
        if o >= 0 and (o != prev or kicked[k, o]):
            ev_t.append(k)
            ev_p.append(o)
        prev = o if o >= 0 else (-1 if o == -2 else prev)
    ev_t, ev_p = np.array(ev_t, np.int64), np.array(ev_p, np.int64)
    M = len(t)
    ev_slot = np.full((M, KT), -1, np.int64)
    ev_attr = np.zeros((M, KT, 7))
    ev_wait = np.zeros((M, KT), np.int64)
    ev_age = np.full((M, KT), HOR, np.int64)
    if len(ev_t):
        prev_p = np.r_[-1, ev_p[:-1]]
        attr = event_attrs(st[ev_t, 1:, :2], st[ev_t, 0, :2], st[np.minimum(ev_t + 1, T - 1), 0, 2:], ev_p, prev_p)
        wait = np.r_[HOR, np.diff(ev_t)]
        j = np.searchsorted(ev_t, t, side="right")          # los KT toques más nuevos hasta t (inclusive)
        for n in range(KT):
            k = j - 1 - n
            ok = k >= 0
            kk = np.maximum(k, 0)
            ev_slot[:, n] = np.where(ok, ev_p[kk], -1)
            ev_attr[:, n] = np.where(ok[:, None], attr[kk], 0.0)
            ev_wait[:, n] = np.where(ok, wait[kk], 0)
            ev_age[:, n] = np.where(ok, t - ev_t[kk], HOR)
    win = st[t[:, None] - np.asarray(OFFSETS)[None, :]]
    out = []
    for team in (0, 1):
        tm = np.full(M, team)
        out.append(np.concatenate([features(win, tm),
                                   chain_features(ev_slot, ev_attr, ev_wait, ev_age, st[t, 0], st[t, 1:, :2], tm)], 1))
    return np.concatenate(out)


# ------------------------------------------------------------------------------------- discriminador
class Discriminator(nn.Module):
    def __init__(self, mean, std, hidden=256):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32))
        self.net = nn.Sequential(nn.Linear(FEAT, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net((x - self.mean) / self.std).squeeze(-1)


def style_reward(d):
    """AMP: r = max(0, 1 − ¼(D − 1)²) ∈ [0, 1]; 1 cuando D da el valor humano (+1)."""
    return torch.clamp(1.0 - 0.25 * (d - 1.0) ** 2, min=0.0)


class Style:
    """Discriminador, su entrenamiento en línea y la recompensa centrada. `bank`: archivo de `--bank` (rasgos humanos
    del split de entrenamiento y una muestra del de desarrollo). lr 1e-3 y penalización 0,1: con los valores de AMP
    (1e-4 y 5) el discriminador casi no separaba al agente de los humanos (AUC 0,66; prueba offline del 2026-10-07)."""

    def __init__(self, bank, device, lr=1e-3, gp=0.1, steps=8, batch=4096, keep=65536, seed=0):
        d = np.load(bank)
        if d["x"].shape[1] != FEAT:
            raise SystemExit(f"{bank}: el banco tiene {d['x'].shape[1]} rasgos y el discriminador usa {FEAT}; "
                             "rearmarlo con `python -m learn.x4_style --bank`")
        self.human = torch.from_numpy(np.ascontiguousarray(d["x"]))            # float16 en CPU
        self.human_dev = torch.from_numpy(np.ascontiguousarray(d["x_dev"])) if "x_dev" in d.files else None
        self.dev = torch.device(device)
        self.D = Discriminator(d["mean"], d["std"]).to(self.dev)
        self.opt = torch.optim.Adam(self.D.parameters(), lr=lr)
        self.gp, self.steps, self.batch, self.keep = gp, steps, batch, keep
        self.gen = torch.Generator().manual_seed(seed)
        self.r_mean, self.r_w = 0.0, 0.0                                         # media móvil de r (EMA con corrección)
        self.agent = []
        self.flip = torch.from_numpy(np.where(Y_COLS, -1.0, 1.0).astype(np.float32)).to(self.dev)

    @property
    def baseline(self):
        return self.r_mean / self.r_w if self.r_w > 1e-6 else 0.0

    @torch.no_grad()
    def reward(self, feats):
        """r ∈ [0, 1] de cada momento (sin centrar)."""
        if not len(feats):
            return np.zeros(0, np.float32)
        x = torch.from_numpy(np.ascontiguousarray(feats, dtype=np.float32)).to(self.dev)
        return style_reward(self.D(x)).float().cpu().numpy()

    def add(self, feats, r):
        """Rasgos del agente de esta actualización (para entrenar el discriminador) y su r (para la media móvil)."""
        if len(feats):
            self.agent.append(np.asarray(feats, np.float16))
            b = 0.99
            self.r_mean = b * self.r_mean + (1 - b) * float(np.mean(r))
            self.r_w = b * self.r_w + (1 - b)

    def _augment(self, x):
        flip = torch.rand(len(x), 1, generator=self.gen).to(self.dev) < 0.5
        return torch.where(flip, x * self.flip, x)

    def fit(self):
        """Pasos del discriminador con los rasgos del agente juntados desde el último `fit`. Devuelve estadísticas."""
        if not self.agent:
            return {}
        agent = np.concatenate(self.agent)
        self.agent = []
        if len(agent) > self.keep:
            agent = agent[torch.randperm(len(agent), generator=self.gen)[:self.keep].numpy()]
        agent = torch.from_numpy(agent)
        sums = np.zeros(4)
        for _ in range(self.steps):
            ia = torch.randint(len(agent), (self.batch,), generator=self.gen)
            ih = torch.randint(len(self.human), (self.batch,), generator=self.gen)
            xa = self._augment(agent[ia].to(self.dev).float())
            xh = self._augment(self.human[ih].to(self.dev).float()).requires_grad_(True)
            dh, da = self.D(xh), self.D(xa)
            loss_ls = ((dh - 1) ** 2).mean() + ((da + 1) ** 2).mean()
            g = torch.autograd.grad(dh.sum(), xh, create_graph=True)[0]
            gp = (g ** 2).sum(-1).mean()
            loss = loss_ls + self.gp * gp
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            self.opt.step()
            sums += [float(loss_ls.detach()), float(gp.detach()), float(dh.detach().mean()), float(da.detach().mean())]
        s = sums / self.steps
        return dict(style_d_loss=round(s[0], 4), style_gp=round(s[1], 4), style_d_human=round(s[2], 3),
                    style_d_agent=round(s[3], 3), style_baseline=round(self.baseline, 4))

    @torch.no_grad()
    def score(self, feats, n=20000):
        """r medio de un conjunto de rasgos (muestra de hasta n)."""
        if feats is None or not len(feats):
            return None
        if len(feats) > n:
            feats = feats[torch.randperm(len(feats), generator=self.gen)[:n].numpy()]
        x = torch.as_tensor(np.asarray(feats, np.float32)).to(self.dev)
        return round(float(style_reward(self.D(x)).mean()), 4)

    def state_dict(self):
        return dict(D=self.D.state_dict(), opt=self.opt.state_dict(), r_mean=float(self.r_mean), r_w=float(self.r_w))

    def load_state_dict(self, sd):
        self.D.load_state_dict(sd["D"])
        self.opt.load_state_dict(sd["opt"])
        self.r_mean, self.r_w = float(sd["r_mean"]), float(sd["r_w"])


# ------------------------------------------------------------------------------------- banco humano
def human_windows(names, ticks_dir, every, map_name="sanguchito_rs_x4"):
    from tools import x4_metrics as XM
    from tools.x4_ticks import MAP_IDS
    out = []
    for name in names:
        path = Path(ticks_dir) / f"{Path(name).stem}.npz"
        if not path.exists():
            continue
        for ep in XM.episodes_from_ticks(path, map_id=MAP_IDS[map_name]):
            w = episode_windows(ep, every)
            if len(w):
                out.append(w.astype(np.float16))
    return np.concatenate(out) if out else np.zeros((0, FEAT), np.float16)


def build_bank(out, splits, ticks_dir, every=2, dev_max=50000, seed=0):
    from learn import x4_data as XD
    train = XD.split_names(splits, "train", None, ("sanguchito_rs_x4",))
    dev = XD.split_names(splits, "dev", None, ("sanguchito_rs_x4",))
    x = human_windows(train, ticks_dir, every)
    x_dev = human_windows(dev, ticks_dir, every)
    rng = np.random.default_rng(seed)
    if len(x_dev) > dev_max:
        x_dev = x_dev[rng.choice(len(x_dev), dev_max, replace=False)]
    xf = x.astype(np.float32)
    mean, std = xf.mean(0), np.maximum(xf.std(0), 1e-3)
    np.savez(out, x=x, x_dev=x_dev, mean=mean.astype(np.float32), std=std.astype(np.float32),
             meta=json.dumps(dict(offsets=OFFSETS, kt=KT, hor=HOR, open=OPEN, every=every, train=len(train),
                                  dev=len(dev))))
    return dict(rasgos=len(x), rasgos_dev=len(x_dev), grabaciones_train=len(train), grabaciones_dev=len(dev))


# ------------------------------------------------------------------------------------- prueba offline
def auc(pos, neg):
    sc = np.r_[pos, neg]
    ranks = np.empty(len(sc))
    ranks[np.argsort(sc)] = np.arange(1, len(sc) + 1)
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def probe(bank, agent_files, device="cpu", steps=4000, batch=1024, lr=1e-3, gp=0.1, seed=0):
    """Entrena un discriminador sólo con rasgos humanos y del agente (partidos guardados por las evaluaciones: la primera
    mitad de los archivos) y mide en datos que no vio: AUC humanos de desarrollo contra la segunda mitad, y r medio de
    cada lado."""
    from tools import x4_metrics as XM
    files = list(agent_files)
    held = files[len(files) // 2:] if len(files) > 1 else files
    fit_files = files[:len(files) // 2] if len(files) > 1 else files
    load = lambda fs: np.concatenate([episode_windows(e) for f in fs for e in XM.episodes_from_npz(f)]).astype(np.float16)
    torch.manual_seed(seed)
    st = Style(bank, device, lr=lr, gp=gp, steps=steps, batch=batch, keep=10 ** 9, seed=seed)
    xa, xa_held = load(fit_files), load(held)
    st.agent = [xa]
    stats = st.fit()
    with torch.no_grad():
        dh = st.D(st.human_dev.float().to(st.dev)).cpu().numpy()
        da = st.D(torch.as_tensor(xa_held.astype(np.float32)).to(st.dev)).cpu().numpy()
    rf = lambda d: float(np.clip(1 - 0.25 * (d - 1) ** 2, 0, None).mean())
    return dict(auc=round(auc(dh, da), 4), r_humanos_dev=round(rf(dh), 4), r_agente_apartado=round(rf(da), 4),
                rasgos_agente=len(xa), rasgos_agente_apartados=len(xa_held), **stats), st


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bank", action="store_true", help="armar el banco de rasgos humanos")
    ap.add_argument("--probe", action="store_true", help="¿el discriminador separa al agente de los humanos?")
    ap.add_argument("--out", default=str(ROOT / "data" / "x4_style_bank.npz"))
    ap.add_argument("--bank-path", default=str(ROOT / "data" / "x4_style_bank.npz"))
    ap.add_argument("--splits", default=str(ROOT / "reports" / "x4" / "splits.json"))
    ap.add_argument("--ticks-dir", default=str(ROOT / "data" / "x4_ticks"))
    ap.add_argument("--every", type=int, default=2, help="un momento cada N muestreos (los vecinos son casi iguales)")
    ap.add_argument("--agent", default="", help="partidos guardados del agente (evals/ep_*.npz), separados por coma")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(argv)
    if a.bank:
        print(json.dumps(build_bank(a.out, a.splits, a.ticks_dir, a.every)), flush=True)
    if a.probe:
        print(json.dumps(probe(a.bank_path, [f for f in a.agent.split(",") if f], a.device, a.steps)[0]), flush=True)


if __name__ == "__main__":
    main()
