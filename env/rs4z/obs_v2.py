"""Observación v2 del actor RS4-Z (sólo información pública de la sala) y features del crítico.

Marco propio: x espejado para el azul (ataca a +x), y sin espejar. Escala real (1150, 670).
Sin reloj, marcador, último toque real ni estado interno del árbitro (preferencia del usuario:
marcador y reloj sólo para el crítico). Escalado analítico fijo: no hay estadísticas que deriven
entre etapas del curriculum; `tests/rs4z/test_rs4z_obs.py` comprueba el rango.

Bloque propio (SELF_DIM) + 7 entidades × ENT_DIM (3 compañeros y 4 rivales, `present`=0 si no juegan).
El nombre de cada feature está en SELF_FEATURES / ENT_FEATURES; `deploy/rs4z/obs_v2.js` las
reproduce en el mismo orden (paridad en `deploy/test_obs_v2.js`).
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

from . import kernel as K

SX, SY = 1150.0, 670.0       # escala de posiciones
SR = 600.0                   # escala de vectores relativos
SVP, SVB = 3.0, 6.0          # velocidad de jugador / pelota
GOAL_X = 1150.0
N_HIST = 3                   # últimas decisiones propias en la obs (latencia: acciones ya comprometidas)
MAX_DELAY = 12.0

SELF_FEATURES = (
    ["x", "y", "vx", "vy", "kick_armed", "kicking",
     "ball_x", "ball_y", "ball_vx", "ball_vy", "ball_dx", "ball_dy", "ball_dist", "ball_rvx", "ball_rvy",
     "line_opp", "line_own", "line_top", "line_bottom",
     "opp_goal_dx", "opp_goal_dy", "own_goal_dx", "own_goal_dy",
     "ball_opp_goal_dx", "ball_opp_goal_dy", "ball_own_goal_dx", "ball_own_goal_dy"]
    + [f"hist{h}_move{m}" for h in range(N_HIST) for m in range(9)]
    + [f"hist{h}_kick" for h in range(N_HIST)]
    + ["delay", "n_mates", "n_rivals", "kick_strength", "ball_radius", "mass_phase",
       "restart_active", "restart_own", "restart_rival", "restart_lateral", "restart_corner",
       "restart_goal_kick", "restart_age", "kickoff", "kickoff_own", "kickoff_age"]
)
ENT_FEATURES = ("present", "dx", "dy", "vx", "vy", "ball_dx", "ball_dy", "ball_dist", "kicking")
SELF_DIM = len(SELF_FEATURES)
ENT_DIM = len(ENT_FEATURES)
N_ENT = 7
N_MATES = 3
OBS_DIM = SELF_DIM + N_ENT * ENT_DIM

CRITIC_FEATURES = ("clock_frac", "clock_left", "score_own", "score_rival", "score_diff", "last_touch_own",
                   "last_touch_rival", "restart_deadline_frac", "kickoff_wait_frac", "safety_frac",
                   "mass_phase", "delay_mean_rivals")
CRITIC_DIM = len(CRITIC_FEATURES)

_MIRROR_MOVE = np.array([0, 1, 8, 7, 6, 5, 4, 3, 2], dtype=np.int64)


@njit(cache=True)
def _clip1(v):
    return 1.0 if v > 1.0 else (0.0 if v < 0.0 else v)


@njit(cache=True, parallel=True)
def build_obs(pos, vel, kick_cancel, active, team, ri, rf, radius, act_hist, applied, delay, fp, out):
    """out[n, p, :] = observación del jugador p del partido n (ceros si no juega)."""
    N = pos.shape[0]
    P = active.shape[1]
    S = SELF_DIM
    for n in prange(N):
        n_red = 0
        n_blue = 0
        for p in range(P):
            if active[n, p]:
                if team[p] == 0:
                    n_red += 1
                else:
                    n_blue += 1
        rteam = ri[n, K.RI_TEAM]
        rkind = ri[n, K.RI_KIND]
        ko = ri[n, K.RI_KO] != 0
        for p in range(P):
            row = out[n, p]
            for j in range(row.shape[0]):
                row[j] = 0.0
            if not active[n, p]:
                continue
            s = 1.0 if team[p] == 0 else -1.0
            k = fp + p
            px = pos[n, k, 0] * s
            py = pos[n, k, 1]
            pvx = vel[n, k, 0] * s
            pvy = vel[n, k, 1]
            bx = pos[n, 0, 0] * s
            by = pos[n, 0, 1]
            bvx = vel[n, 0, 0] * s
            bvy = vel[n, 0, 1]
            dx = bx - px
            dy = by - py
            row[0] = px / SX
            row[1] = py / SY
            row[2] = pvx / SVP
            row[3] = pvy / SVP
            row[4] = 0.0 if kick_cancel[n, p] else 1.0
            row[5] = 1.0 if (applied[n, p] >= 9 and not kick_cancel[n, p]) else 0.0
            row[6] = bx / SX
            row[7] = by / SY
            row[8] = bvx / SVB
            row[9] = bvy / SVB
            row[10] = dx / SR
            row[11] = dy / SR
            row[12] = math.sqrt(dx * dx + dy * dy) / SR
            row[13] = (bvx - pvx) / SVB
            row[14] = (bvy - pvy) / SVB
            row[15] = (GOAL_X - px) / SX
            row[16] = (px + GOAL_X) / SX
            row[17] = (SY + py) / SY
            row[18] = (SY - py) / SY
            row[19] = (GOAL_X - px) / SX
            row[20] = -py / SY
            row[21] = (-GOAL_X - px) / SX
            row[22] = -py / SY
            row[23] = (GOAL_X - bx) / SX
            row[24] = -by / SY
            row[25] = (-GOAL_X - bx) / SX
            row[26] = -by / SY
            j = 27
            for h in range(N_HIST):
                a = act_hist[n, p, h]
                m = a % 9
                if s < 0:
                    m = _MIRROR_MOVE[m]
                row[j + h * 9 + m] = 1.0
            j += 9 * N_HIST
            for h in range(N_HIST):
                row[j + h] = 1.0 if act_hist[n, p, h] >= 9 else 0.0
            j += N_HIST
            row[j] = delay[n, p] / MAX_DELAY
            own_n = n_red if team[p] == 0 else n_blue
            riv_n = n_blue if team[p] == 0 else n_red
            row[j + 1] = (own_n - 1) / 3.0
            row[j + 2] = riv_n / 4.0
            row[j + 3] = (rf[n, K.RF_KSTR] - 5.8) / 0.1
            row[j + 4] = (radius[n, 0] - 8.1625) / 0.1625
            row[j + 5] = float(ri[n, K.RI_MASS])
            if rteam >= 0:
                row[j + 6] = 1.0
                row[j + 7] = 1.0 if rteam == team[p] else 0.0
                row[j + 8] = 1.0 if rteam != team[p] else 0.0
                row[j + 9] = 1.0 if rkind == 1 else 0.0
                row[j + 10] = 1.0 if rkind == 2 else 0.0
                row[j + 11] = 1.0 if rkind == 3 else 0.0
                row[j + 12] = _clip1(ri[n, K.RI_TICKS] / 600.0)
            if ko:
                row[j + 13] = 1.0
                row[j + 14] = 1.0 if ri[n, K.RI_KO_TEAM] == team[p] else 0.0
                row[j + 15] = _clip1(ri[n, K.RI_KO_TICKS] / 600.0)
            # entidades: compañeros (hasta 3) y luego rivales (hasta 4), por slot
            e_mate = 0
            e_riv = 0
            for q in range(P):
                if q == p or not active[n, q]:
                    continue
                if team[q] == team[p]:
                    if e_mate >= N_MATES:
                        continue
                    base = S + e_mate * ENT_DIM
                    e_mate += 1
                else:
                    base = S + (N_MATES + e_riv) * ENT_DIM
                    e_riv += 1
                kq = fp + q
                qx = pos[n, kq, 0] * s
                qy = pos[n, kq, 1]
                row[base] = 1.0
                row[base + 1] = (qx - px) / SR
                row[base + 2] = (qy - py) / SR
                row[base + 3] = vel[n, kq, 0] * s / SVP
                row[base + 4] = vel[n, kq, 1] / SVP
                row[base + 5] = (bx - qx) / SR
                row[base + 6] = (by - qy) / SR
                row[base + 7] = math.sqrt((bx - qx) ** 2 + (by - qy) ** 2) / SR
                row[base + 8] = 1.0 if (applied[n, q] >= 9 and not kick_cancel[n, q]) else 0.0


@njit(cache=True, parallel=True)
def build_critic(active, team, ri, delay, deadline, out):
    """Features privilegiadas sólo para el crítico (por jugador, en el marco de su equipo)."""
    N = active.shape[0]
    P = active.shape[1]
    for n in prange(N):
        length = ri[n, K.RI_LEN]
        clock = ri[n, K.RI_CLOCK]
        for p in range(P):
            row = out[n, p]
            for j in range(row.shape[0]):
                row[j] = 0.0
            if not active[n, p]:
                continue
            t = team[p]
            own = ri[n, K.RI_SCORE0] if t == 0 else ri[n, K.RI_SCORE1]
            riv = ri[n, K.RI_SCORE1] if t == 0 else ri[n, K.RI_SCORE0]
            row[0] = _clip1(clock / max(length, 1))
            left = length - clock
            row[1] = _clip1(left / 36000.0)
            row[2] = min(own, 10) / 5.0
            row[3] = min(riv, 10) / 5.0
            diff = own - riv
            row[4] = max(-3.0, min(3.0, float(diff))) / 3.0
            last = ri[n, K.RI_LAST]
            row[5] = 1.0 if last == t else 0.0
            row[6] = 1.0 if (last >= 0 and last != t) else 0.0
            if ri[n, K.RI_TEAM] >= 0 and deadline > 0:
                row[7] = _clip1(ri[n, K.RI_TICKS] / deadline)
            if ri[n, K.RI_KO] != 0 and deadline > 0:
                row[8] = _clip1(ri[n, K.RI_KO_TICKS] / deadline)
            if ri[n, K.RI_TEAM] >= 0:
                row[9] = _clip1(ri[n, K.RI_TICKS] / 900.0)
            row[10] = float(ri[n, K.RI_MASS])
            s = 0.0
            c = 0
            for q in range(P):
                if active[n, q] and team[q] != t:
                    s += delay[n, q]
                    c += 1
            row[11] = (s / c) / MAX_DELAY if c > 0 else 0.0


def observe(env, out=None):
    if out is None:
        out = np.empty((env.N, 8, OBS_DIM), dtype=np.float32)
    build_obs(env.pos, env.vel, env.kick_cancel, env.active, env.team, env.ri, env.rf, env.radius,
              env.act_hist, env._s_act, env.delay, env.fp, out)
    return out


def critic(env, out=None):
    if out is None:
        out = np.empty((env.N, 8, CRITIC_DIM), dtype=np.float32)
    build_critic(env.active, env.team, env.ri, env.delay, env.deadline, out)
    return out
