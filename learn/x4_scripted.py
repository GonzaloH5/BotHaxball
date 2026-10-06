"""Política scripted X4 sobre la observación v3: referencia de fuerza independiente de los datos humanos.

Roles por distancia a la pelota dentro del equipo (lo más simple que juega un partido entero):
* el más cercano a la pelota presiona: va a un punto detrás de la pelota en la línea pelota→arco rival y
  patea cuando está alineado y en contacto (o despeja si la pelota está cerca del arco propio);
* el segundo apoya: se ubica 180 px detrás de la pelota, abierto hacia el lado contrario al que presiona;
* los otros dos cubren: uno entre la pelota y el arco propio (a 350 px), el otro de último hombre.
En saques del rival se aleja de la pelota; en saques propios va el más cercano.

No usa información que la política aprendida no tenga: lee sólo la observación v3 (marco propio).
Sirve como rival fijo del pool de evaluación (revisión §7: rivales fijos y diversos).
"""
from __future__ import annotations

import numpy as np

from env.rs4z import obs_v3 as O

S = O.SELF_DIM
E = O.ENT_DIM
# direcciones de movimiento (marco propio): índice → (dx, dy)
MOVES = np.array([(0, 0), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)], dtype=float)
_J = 27 + 9 * O.N_HIST + O.N_HIST   # inicio del bloque delay/restart/kickoff


def _move_toward(dx, dy, dead=4.0):
    if dx * dx + dy * dy < dead * dead:
        return 0
    v = np.array([dx, dy]) / np.hypot(dx, dy)
    return int(np.argmax(MOVES[1:] @ v) + 1)


class ScriptedPolicy:
    name = "scripted"
    spec = "scripted"

    def __init__(self, kick_dist=27.0):
        self.kick_dist = kick_dist

    def act_one(self, o):
        px, py = o[0] * O.SX, o[1] * O.SY
        bx, by = o[6] * O.SX, o[7] * O.SY
        bvx, bvy = o[8] * O.SVB, o[9] * O.SVB
        my_d = o[12] * O.SR
        mates = o[S:S + 3 * E].reshape(3, E)
        present = mates[:, 0] > 0
        mate_d = mates[:, 7] * O.SR
        rank = int((mate_d[present] < my_d).sum())          # 0 = el más cercano del equipo
        restart = o[_J + 6] > 0
        restart_own = o[_J + 7] > 0
        ko = o[_J + 13] > 0
        ko_own = o[_J + 14] > 0
        # pelota anticipada unos ticks
        fx, fy = bx + 6 * bvx, by + 6 * bvy
        goal_x = 1150.0
        if (restart and not restart_own) or (ko and not ko_own):
            # alejarse de la pelota hacia el propio arco
            tx, ty = bx - 220.0, by * 0.5
            return _move_toward(tx - px, ty - py)
        if rank == 0:
            # punto detrás de la pelota en la línea al arco rival (o hacia afuera si defiende cerca del arco)
            ax, ay = goal_x - fx, 0.0 - fy
            if fx < -700:
                ax, ay = 1.0, np.sign(fy) if abs(fy) > 1 else 1.0   # despejar hacia adelante y afuera
            n = np.hypot(ax, ay) + 1e-9
            ux, uy = ax / n, ay / n
            tx, ty = fx - ux * 22.0, fy - uy * 22.0
            move = _move_toward(tx - px, ty - py, dead=3.0)
            # alineado: el vector jugador→pelota apunta hacia el objetivo
            vx, vy = bx - px, by - py
            dist = np.hypot(vx, vy)
            aligned = dist > 1e-6 and (vx * ux + vy * uy) / dist > 0.8
            if dist < self.kick_dist and aligned:
                return _move_toward(ux, uy, dead=0.0) + 9
            if dist < self.kick_dist and not aligned:
                return move
            return move
        if rank == 1:
            side = -np.sign(by) if abs(by) > 30 else 1.0
            tx, ty = bx - 180.0, by + side * 160.0
        elif rank == 2:
            tx, ty = min(bx - 350.0, 200.0), by * 0.4
        else:
            tx, ty = max(-1000.0, min(bx - 600.0, -300.0)), by * 0.2
        tx = float(np.clip(tx, -1100.0, 1100.0))
        ty = float(np.clip(ty, -600.0, 600.0))
        return _move_toward(tx - px, ty - py, dead=12.0)

    def __call__(self, obs, rng=None):
        return np.array([self.act_one(o) if o[0] != 0 or o[1] != 0 or o[6] != 0 else 0 for o in obs], dtype=np.int64)
