"""Observación universal fusionada: mismos features, una llamada paralela por tarea.

Sin fastmath: física/normalización siguen en float64; salida float32 como antes.
El orden de entidades es estable (compañeros, rivales, ausentes al final).
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

from .geometry import RAY_DIRS, RAY_MAX, RAY_MIRROR, ray_distance


def observe_universal(env, indices=None):
    sim, rays = env.sim, env.rays
    rows = np.arange(env.N) if indices is None else np.asarray(indices, dtype=np.int64)
    # El bloque de reglas continúa perteneciendo al script original.
    rules = env.rules.features() if env.rules is not None else np.empty((0, 0, 0))
    expelled = env.rules.expelled if env.rules is not None else np.empty((0, 0), dtype=np.bool_)
    out = np.empty((len(rows), env.P, env.obs_dim), dtype=np.float32)
    _observe(rows, sim.ball_pos, sim.ball_vel, sim.player_pos, sim.player_vel,
             sim.player_team, env.sign, sim.kick_cancel, sim.kickoff, sim.kickoff_team,
             sim.touch, env.ticks, env.max_ticks, env.field_w, env.field_h, env.goal_x,
             sim.ps_on, sim.ps_par, sim.ps_comba, sim.ps_charge, sim.inv_env,
             sim.ball_grav, sim.ps_held, env._map_features(), rules, expelled,
             rays.red, rays.blue, rays.ball, rays.player_r, rays.ball_r, out)
    return out


@njit(cache=True, parallel=True)
def _observe(rows, bp, bv, pp, pv, team, sign, kick_cancel, kickoff, kickoff_team,
             touch, ticks, max_ticks, W, H, gx, ps_on, ps_par, ps_comba, ps_charge,
             inv, grav, held, map_feats, rule_feats, expelled, red_geometry,
             blue_geometry, ball_geometry, rp, rb, out):
    P = pp.shape[1]
    T = P // 2
    for i in prange(len(rows)):
        n = rows[i]
        ball_rays = np.empty(8)
        for k in range(8):
            ball_rays[k] = ray_distance(bp[n, 0], bp[n, 1], RAY_DIRS[k, 0], RAY_DIRS[k, 1],
                                        ball_geometry, rb, RAY_MAX)
        for p in range(P):
            row = out[i, p]
            # También escribe padding/flags de PS desactivado; nunca deja memoria sin inicializar.
            for j in range(len(row)):
                row[j] = 0.0
            s = sign[p]
            px, py = pp[n, p, 0] * s, pp[n, p, 1]
            bx, by = bp[n, 0] * s, bp[n, 1]
            dx, dy = bx - px, by - py
            row[0], row[1] = px / W, py / H
            row[2], row[3] = pv[n, p, 0] * s / 5.0, pv[n, p, 1] / 5.0
            row[4], row[5] = bx / W, by / H
            row[6], row[7] = bv[n, 0] * s / 5.0, bv[n, 1] / 5.0
            row[8], row[9] = dx / 400.0, dy / 400.0
            row[10] = math.sqrt(dx * dx + dy * dy) / 400.0
            row[11], row[12] = (gx - px) / W, -py / W
            row[13], row[14] = (-gx - px) / W, -py / W
            row[15], row[16] = (gx - bx) / W, -by / W
            row[17] = not kick_cancel[n, p]
            row[18] = kickoff[n]
            row[19] = kickoff[n] and kickoff_team[n] == team[p]
            row[20] = touch[n, p]
            row[21] = ticks[n] / max_ticks
            if ps_on:
                row[22] = ps_comba[n]
                row[23] = 1.0 - ps_charge[n] / ps_par[0] if ps_charge[n] > 0 else 0.0
                row[24] = (inv[n, 0] - ps_par[2]) / (ps_par[1] - ps_par[2])
                row[25], row[26] = grav[n, 0] * s / ps_par[5], grav[n, 1] / ps_par[5]
                row[27] = held[n] == p
            for j in range(10):
                row[28 + j] = map_feats[j]
            geometry = red_geometry if team[p] == 0 else blue_geometry
            for k in range(8):
                world_k = k if s > 0 else RAY_MIRROR[k]
                row[38 + k] = ray_distance(pp[n, p, 0], pp[n, p, 1],
                                          RAY_DIRS[world_k, 0], RAY_DIRS[world_k, 1],
                                          geometry, rp, RAY_MAX) / RAY_MAX
                row[46 + k] = ball_rays[world_k] / RAY_MAX
            row[54], row[55] = (T - 1) / 10.0, T / 10.0
            if rule_feats.shape[0]:
                for j in range(rule_feats.shape[2]):
                    row[56 + j] = rule_feats[n, p, j]
            e = 0
            for rival in range(2):
                for q in range(P):
                    if q == p or (team[q] != team[p]) != (rival == 1):
                        continue
                    if expelled.shape[0] and expelled[n, q]:
                        continue
                    j = 71 + 8 * e
                    row[j], row[j + 1] = 1.0, rival
                    # Mismo orden de operaciones del builder NumPy para las coordenadas propias.
                    row[j + 2], row[j + 3] = (pp[n, q, 0] * s - px) / 400.0, (pp[n, q, 1] - py) / 400.0
                    row[j + 4], row[j + 5] = pv[n, q, 0] * s / 5.0, pv[n, q, 1] / 5.0
                    row[j + 6], row[j + 7] = (bx - pp[n, q, 0] * s) / 400.0, (by - pp[n, q, 1]) / 400.0
                    e += 1
