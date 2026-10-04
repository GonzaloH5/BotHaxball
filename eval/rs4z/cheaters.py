"""Políticas degeneradas para probar explotaciones (de rewards y del scripted).

Cada una reproduce una trampa observada o plausible: quedarse quieto, oscilar, que todos persigan,
dejar un delantero colgado, bloque dentro del arco, pelotazos, sacarla afuera, trabar saques.
Mismo protocolo de controlador que `eval/rs4z/runner.py`.
"""
from __future__ import annotations

import numpy as np

from env.rs4z import kernel as K
from sim.physics import MOVE_UNIT

SIGN = np.where(np.arange(8) < 4, 1.0, -1.0)


def _toward(env, tx, ty):
    """Acción de movimiento (marco propio) hacia (tx, ty) del mundo, por jugador: (N, 8)."""
    d = np.stack([tx - env.player_pos[..., 0], ty - env.player_pos[..., 1]], axis=-1)
    d[..., 0] *= SIGN
    n = np.maximum(np.hypot(d[..., 0], d[..., 1]), 1e-9)
    u = d / n[..., None]
    move = np.argmax(u @ MOVE_UNIT.T, axis=-1)
    move[n < 6.0] = 0
    return move


class Cheater:
    def __init__(self, kind, seed=0):
        self.kind = kind
        self.rng = np.random.default_rng(seed)
        self.t = 0

    def sync(self, env, rows):
        pass

    def push(self, env):
        self.t += 1

    def act(self, env, ctrl, out):
        k = self.kind
        N = env.N
        bx = np.broadcast_to(env.ball_pos[:, 0:1], (N, 8))
        by = np.broadcast_to(env.ball_pos[:, 1:2], (N, 8))
        dist = np.hypot(env.player_pos[..., 0] - bx, env.player_pos[..., 1] - by)
        near = dist < 30.0
        if k == "still":
            a = np.zeros((N, 8), dtype=np.int64)
        elif k == "oscillate":
            a = np.full((N, 8), 3 if (self.t // 4) % 2 == 0 else 7, dtype=np.int64)
        elif k == "all_chase":
            a = _toward(env, bx, by) + 9 * near
        elif k == "long_ball":
            # todos persiguen y patean siempre (pelotazo hacia donde estén mirando)
            a = _toward(env, bx, by) + 9
        elif k == "kick_out":
            # persigue y patea hacia la línea lateral más cercana
            side = np.where(env.ball_pos[:, 1:2] >= 0, 1.0, -1.0)
            tx = bx
            ty = by - side * 40.0          # pararse del lado de adentro para sacarla hacia afuera
            a = _toward(env, tx, ty) + 9 * near
        elif k == "goal_camp":
            gx = -1100.0 * SIGN[None, :] * np.ones((N, 1))
            gy = np.tile(np.array([-60.0, 0.0, 60.0, 0.0, -60.0, 0.0, 60.0, 0.0]), (N, 1))
            a = _toward(env, gx, gy) + 9 * near
        elif k == "park":
            # tres persiguen; el último del equipo se queda colgado arriba
            a = _toward(env, bx, by) + 9 * near
            hx = 900.0 * SIGN[None, :] * np.ones((N, 1))
            hang = _toward(env, hx, np.zeros((N, 8)))
            a[:, 3] = hang[:, 3]
            a[:, 7] = hang[:, 7]
        elif k == "stall_restarts":
            # juega como all_chase pero nunca ejecuta sus propios saques
            a = _toward(env, bx, by) + 9 * near
            own = env.ri[:, K.RI_TEAM][:, None] == np.where(np.arange(8) < 4, 0, 1)[None, :]
            ko_own = (env.ri[:, K.RI_KO][:, None] != 0) & (env.ri[:, K.RI_KO_TEAM][:, None] == np.where(np.arange(8) < 4, 0, 1)[None, :])
            a = np.where(own | ko_own, 0, a)
        elif k == "random":
            a = self.rng.integers(0, 18, (N, 8))
        else:
            raise ValueError(k)
        out[ctrl] = np.asarray(a)[ctrl]
        return out


KINDS = ("still", "oscillate", "all_chase", "long_ball", "kick_out", "goal_camp", "park", "stall_restarts", "random")
