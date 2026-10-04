"""Interfaz de RS-Pro sobre `env.rs4z.core.RS4ZEnv`: niveles, estilos, memoria y percepción retrasada.

Uso:
    bot = RSPro(env, seed=0)
    bot.configure(rows, team=0, level=5, style=STYLE_BALANCED)
    bot.sync(rows)                     # tras reiniciar partidos o colocar estados
    actions = bot.act(env, ctrl)       # ctrl: (N, 8) jugadores que controla el bot
    env.step(...); bot.push(env)       # registrar el estado nuevo para la percepción retrasada
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange

from env.rs4z import kernel as K
from env.rs4z.numba_cache import guard

guard(__file__, ["bots/rspro/brain.py", "bots/rspro/geom.py", "env/rs4z/kernel.py", "env/rs4z/contract.py"])

from . import brain as B
from .geom import HORIZON, TTR_TABLE

# Niveles: misma lógica con capacidades crecientes (ver auditoría). La reacción de L5 (~10 ticks,
# 167 ms) es la de un humano experto que anticipa; su fuerza viene de decidir mejor, no de reflejos.
LEVELS = {
    0: dict(delay=30, noise=26.0, aim_tol=32.0, aim_noise=12.0, options=0, temp=0.30, hyst=20.0, react=6.0, commit=3, speed=1.0),
    1: dict(delay=24, noise=18.0, aim_tol=27.0, aim_noise=8.0, options=1, temp=0.20, hyst=40.0, react=6.0, commit=4, speed=1.0),
    2: dict(delay=18, noise=12.0, aim_tol=22.0, aim_noise=5.5, options=2, temp=0.13, hyst=60.0, react=6.0, commit=4, speed=1.0),
    3: dict(delay=15, noise=8.0, aim_tol=18.0, aim_noise=3.5, options=3, temp=0.08, hyst=80.0, react=6.0, commit=5, speed=1.0),
    4: dict(delay=12, noise=5.0, aim_tol=15.0, aim_noise=2.2, options=4, temp=0.05, hyst=95.0, react=6.0, commit=5, speed=1.0),
    5: dict(delay=9, noise=3.0, aim_tol=13.0, aim_noise=1.3, options=5, temp=0.035, hyst=110.0, react=6.0, commit=6, speed=1.0),
}
STYLE_NAMES = ("press", "direct", "width", "risk", "tempo", "depth")
STYLE_BALANCED = np.full(6, 0.5)
# Región reservada para evaluación (nunca se muestrea al entrenar): presión alta + directo + estrecho.
RESERVED = dict(press=(0.75, 1.0), direct=(0.75, 1.0), width=(0.0, 0.25))


def is_reserved(style):
    s = np.asarray(style)
    return bool(s[0] >= RESERVED["press"][0] and s[1] >= RESERVED["direct"][0] and s[2] <= RESERVED["width"][1])


def sample_style(rng, reserved=False):
    """Estilo al azar; con reserved=False nunca cae en la región reservada para evaluar."""
    while True:
        s = rng.random(6)
        if reserved:
            s[0] = rng.uniform(*RESERVED["press"])
            s[1] = rng.uniform(*RESERVED["direct"])
            s[2] = rng.uniform(*RESERVED["width"])
            return s
        if not is_reserved(s):
            return s


BUFFER = 12   # decisiones de historia para la percepción retrasada (≥ 30 ticks)


@njit(cache=True, parallel=True)
def _act_all(seed, table, pos9, vel9, buf_pos, buf_vel, head, active, kick_cancel, ctrl, ri, rf, grav, lvl, sty,
             mem_i, mem_f, traj, out):
    N = pos9.shape[0]
    for job in prange(N * 2):
        n = job // 2
        team = job % 2
        any_ctrl = False
        for p in range(8):
            if ctrl[n, p] and active[n, p] and ((p < 4) == (team == 0)):
                any_ctrl = True
        if not any_ctrl:
            continue
        lag = int(lvl[n, team, B.L_DELAY]) // 3
        if lag > buf_pos.shape[1] - 1:
            lag = buf_pos.shape[1] - 1
        idx = (head - lag) % buf_pos.shape[1]
        acts = np.zeros(8, dtype=np.int64)
        B.decide(n, team, seed, table, pos9[n], vel9[n], active[n], kick_cancel[n], ctrl[n], ri[n, K.RI_TEAM],
                 ri[n, K.RI_KIND], ri[n, K.RI_TICKS], ri[n, K.RI_KO], ri[n, K.RI_KO_TEAM], rf[n, K.RF_SPOT_X],
                 rf[n, K.RF_SPOT_Y], grav[n, 0], grav[n, 1], ri[n, K.RI_GRAV], lvl[n, team], sty[n, team], mem_i[n, team],
                 mem_f[n, team], buf_pos[n, idx], buf_vel[n, idx], traj[n, team], acts)
        for p in range(8):
            if ctrl[n, p] and active[n, p] and ((p < 4) == (team == 0)):
                out[n, p] = acts[p]


class RSPro:
    def __init__(self, env, seed=0):
        N = env.N
        self.N = N
        self.seed = int(seed)
        self.lvl = np.zeros((N, 2, B.NLV))
        self.sty = np.tile(STYLE_BALANCED, (N, 2, 1))
        self.level = np.full((N, 2), 5, dtype=np.int64)
        self.mem_i = np.zeros((N, 2, B.MI), dtype=np.int64)
        self.mem_f = np.zeros((N, 2, B.MF))
        self.buf_pos = np.zeros((N, BUFFER, 9, 2))
        self.buf_vel = np.zeros((N, BUFFER, 9, 2))
        self.head = 0
        self.traj = np.zeros((N, 2, HORIZON + 1, 2))
        for t in (0, 1):
            self.configure(np.arange(N), t, 5)
        self.sync(env, np.arange(N))

    def configure(self, rows, team, level, style=None):
        rows = np.atleast_1d(np.asarray(rows, dtype=np.int64))
        cfg = LEVELS[int(level)]
        delay = 3 * int(round(cfg["delay"] / 3))      # la percepción va por decisiones de 3 ticks
        vec = np.array([delay, cfg["noise"], cfg["aim_tol"], cfg["aim_noise"], cfg["options"], cfg["temp"],
                        cfg["hyst"], cfg["react"], cfg["commit"], cfg["speed"]], dtype=np.float64)
        self.lvl[rows, team] = vec
        self.level[rows, team] = int(level)
        if style is not None:
            self.sty[rows, team] = np.asarray(style, dtype=np.float64)
        self.mem_i[rows, team] = 0
        self.mem_i[rows, team, B.M_TAKER] = -1
        self.mem_f[rows, team] = 0.0

    def _state(self, env):
        fp = env.fp
        idx = np.concatenate([[0], fp + np.arange(8)])
        return env.pos[:, idx], env.vel[:, idx]

    def sync(self, env, rows):
        """Llenar la historia de percepción con el estado actual (tras un reinicio o una colocación)."""
        rows = np.atleast_1d(np.asarray(rows, dtype=np.int64))
        pos, vel = self._state(env)
        self.buf_pos[rows] = pos[rows][:, None]
        self.buf_vel[rows] = vel[rows][:, None]
        self.mem_i[rows, :, B.M_PLAN:B.M_PLAN + 8] = 0

    def push(self, env):
        pos, vel = self._state(env)
        self.head = (self.head + 1) % BUFFER
        self.buf_pos[:, self.head] = pos
        self.buf_vel[:, self.head] = vel

    def act(self, env, ctrl, out=None):
        """Acciones (marco propio) para los jugadores `ctrl`; el resto de `out` no se toca."""
        if out is None:
            out = np.zeros((env.N, 8), dtype=np.int64)
        pos, vel = self._state(env)
        _act_all(self.seed, TTR_TABLE, np.ascontiguousarray(pos), np.ascontiguousarray(vel), self.buf_pos,
                 self.buf_vel, self.head, env.active, env.kick_cancel, np.asarray(ctrl, dtype=np.bool_), env.ri,
                 env.rf, env.grav, self.lvl, self.sty, self.mem_i, self.mem_f, self.traj, out)
        return out
