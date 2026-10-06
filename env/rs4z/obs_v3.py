"""Observación v3 del actor X4: obs v2 + acciones propias pendientes para latencias de sala + mapa.

Cambios respecto de v2 (`env/rs4z/obs_v2.py`), con su motivo:
* 5 decisiones propias en vez de 3 (`N_HIST`): con frame_skip 3 cubren 15 ticks. La latencia medida en sala
  es de 9–12 ticks (`reports/room_latency/`). El estado aumentado con las acciones pendientes hace exacto el
  MDP con retardo constante (Katsikopoulos & Engelbrecht 2003; revisión §5).
* Geometría del mapa (línea lateral, línea de gol) tomada del contrato del mapa, no fija: 2K23 tiene la
  lateral en 600 y Sanguchito/RS ONE en 670 (revisión §6: observación relativa a la geometría).
* Mapa como condición (one-hot rs_one / sanguchito_rs_x4 / haxarg_2k23; revisión §6, plan E1).
* Margen de contacto con la pelota (distancia − radios, /8 px) y "al alcance de patada" (margen < KICK_REACH = 4 px)
  para el jugador y cada entidad: con la distancia escalada a 600 px la zona donde se decide la patada ocupa
  0,007 de rango; la imitación con v2+ pateaba la mitad que los humanos en lazo cerrado.

Una sola función por muestra (`_featurize`) arma la observación tanto desde `RS4ZEnv` como desde las
grabaciones (`build_samples`, caché `tools.x4_ticks`): la paridad sim/datos es por construcción. Las
entidades (3 compañeros, 4 rivales) van en bloques iguales para un codificador de conjuntos.

Marco propio: x espejado para el azul (ataca a +x), y sin espejar. Sin reloj ni marcador (sólo crítico).
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

from . import contract as C
from . import kernel as K
from .numba_cache import guard

guard(__file__, ["env/rs4z/kernel.py", "env/rs4z/contract.py"])

SX = 1150.0                  # escala de x (línea de gol de RS ONE / Sanguchito / 2K23)
SY = 670.0                   # escala de y (lateral más ancha de los mapas soportados)
SR = 600.0                   # escala de vectores relativos
SVP, SVB = 3.0, 6.0          # velocidad de jugador / pelota
PLAYER_R = 15.0
KICK_REACH = 4.0             # sim/physics.KICK_REACH
GAP_SCALE = 8.0
N_HIST = 5                   # decisiones propias en la obs (15 ticks con frame_skip 3)
MAX_DELAY = 15.0
MAP_NAMES = ("rs_one", "sanguchito_rs_x4", "haxarg_2k23")
N_MAPS = len(MAP_NAMES)

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
    + [f"map_{m}" for m in MAP_NAMES]
    + ["ball_gap", "ball_in_reach"]
)
ENT_FEATURES = ("present", "dx", "dy", "vx", "vy", "ball_dx", "ball_dy", "ball_dist", "kicking", "ball_gap")
SELF_DIM = len(SELF_FEATURES)
ENT_DIM = len(ENT_FEATURES)
N_ENT = 7
N_MATES = 3
OBS_DIM = SELF_DIM + N_ENT * ENT_DIM
OBS_VERSION = "x4-obs-v3"

_MIRROR_MOVE = np.array([0, 1, 8, 7, 6, 5, 4, 3, 2], dtype=np.int64)
_TEAM = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)


def map_geometry(map_name):
    """(línea lateral, x de la línea de gol, índice del mapa) del contrato."""
    prm = C.params(map_name)
    return float(prm[C.PI["line_half_h"]]), float(C.LINE_W), MAP_NAMES.index(map_name)


@njit(cache=True, inline="always")
def _clip1(v):
    return 1.0 if v > 1.0 else (0.0 if v < 0.0 else v)


@njit(cache=True, inline="always")
def _gap(g):
    v = g / GAP_SCALE
    return -1.0 if v < -1.0 else (4.0 if v > 4.0 else v)


@njit(cache=True)
def _featurize(row, p, ball, ball_r, ppos, pvel, active, kcancel, kicking_now, hist, H, delay,
               rteam, rkind, rage, ko, ko_team, ko_age, mass_phase, kstr, map_idx, line_h, goal_x):
    """Observación del lugar `p` (0..3 rojo, 4..7 azul) en `row`.

    ball: (x, y, vx, vy); ppos/pvel: (8, 2) mundo; kicking_now[q]: patada aplicada en el último tick y no
    cancelada; hist[h]: decisión propia h (0 = la más nueva) en coordenadas del mundo, 0..17.
    """
    for j in range(row.shape[0]):
        row[j] = 0.0
    if not active[p]:
        return
    tp = 0 if p < 4 else 1
    n_red = 0
    n_blue = 0
    for q in range(8):
        if active[q]:
            if q < 4:
                n_red += 1
            else:
                n_blue += 1
    s = 1.0 if tp == 0 else -1.0
    px = ppos[p, 0] * s
    py = ppos[p, 1]
    pvx = pvel[p, 0] * s
    pvy = pvel[p, 1]
    bx = ball[0] * s
    by = ball[1]
    bvx = ball[2] * s
    bvy = ball[3]
    dx = bx - px
    dy = by - py
    row[0] = px / SX
    row[1] = py / SY
    row[2] = pvx / SVP
    row[3] = pvy / SVP
    row[4] = 0.0 if kcancel[p] else 1.0
    row[5] = 1.0 if kicking_now[p] else 0.0
    row[6] = bx / SX
    row[7] = by / SY
    row[8] = bvx / SVB
    row[9] = bvy / SVB
    row[10] = dx / SR
    row[11] = dy / SR
    row[12] = math.sqrt(dx * dx + dy * dy) / SR
    row[13] = (bvx - pvx) / SVB
    row[14] = (bvy - pvy) / SVB
    row[15] = (goal_x - px) / SX
    row[16] = (px + goal_x) / SX
    row[17] = (line_h + py) / SY
    row[18] = (line_h - py) / SY
    row[19] = (goal_x - px) / SX
    row[20] = -py / SY
    row[21] = (-goal_x - px) / SX
    row[22] = -py / SY
    row[23] = (goal_x - bx) / SX
    row[24] = -by / SY
    row[25] = (-goal_x - bx) / SX
    row[26] = -by / SY
    j = 27
    nh = H if H < N_HIST else N_HIST
    for h in range(nh):
        a = hist[h]
        m = a % 9
        if s < 0:
            m = _MIRROR_MOVE[m]
        row[j + h * 9 + m] = 1.0
    j += 9 * N_HIST
    for h in range(nh):
        row[j + h] = 1.0 if hist[h] >= 9 else 0.0
    j += N_HIST
    row[j] = delay / MAX_DELAY
    own_n = n_red if tp == 0 else n_blue
    riv_n = n_blue if tp == 0 else n_red
    row[j + 1] = (own_n - 1) / 3.0
    row[j + 2] = riv_n / 4.0
    row[j + 3] = (kstr - 5.8) / 0.1
    row[j + 4] = (ball_r - 8.1625) / 0.1625
    row[j + 5] = mass_phase
    if rteam >= 0:
        row[j + 6] = 1.0
        row[j + 7] = 1.0 if rteam == tp else 0.0
        row[j + 8] = 1.0 if rteam != tp else 0.0
        row[j + 9] = 1.0 if rkind == 1 else 0.0
        row[j + 10] = 1.0 if rkind == 2 else 0.0
        row[j + 11] = 1.0 if rkind == 3 else 0.0
        row[j + 12] = _clip1(rage / 600.0)
    if ko:
        row[j + 13] = 1.0
        row[j + 14] = 1.0 if ko_team == tp else 0.0
        row[j + 15] = _clip1(ko_age / 600.0)
    if map_idx >= 0:
        row[j + 16 + map_idx] = 1.0
    gap = math.sqrt(dx * dx + dy * dy) - PLAYER_R - ball_r
    row[j + 16 + N_MAPS] = _gap(gap)
    row[j + 17 + N_MAPS] = 1.0 if gap < KICK_REACH else 0.0
    S = j + 18 + N_MAPS
    e_mate = 0
    e_riv = 0
    for q in range(8):
        if q == p or not active[q]:
            continue
        tq = 0 if q < 4 else 1
        if tq == tp:
            if e_mate >= N_MATES:
                continue
            base = S + e_mate * 10
            e_mate += 1
        else:
            if e_riv >= 4:
                continue
            base = S + (N_MATES + e_riv) * 10
            e_riv += 1
        qx = ppos[q, 0] * s
        qy = ppos[q, 1]
        row[base] = 1.0
        row[base + 1] = (qx - px) / SR
        row[base + 2] = (qy - py) / SR
        row[base + 3] = pvel[q, 0] * s / SVP
        row[base + 4] = pvel[q, 1] / SVP
        row[base + 5] = (bx - qx) / SR
        row[base + 6] = (by - qy) / SR
        dq = math.sqrt((bx - qx) ** 2 + (by - qy) ** 2)
        row[base + 7] = dq / SR
        row[base + 8] = 1.0 if kicking_now[q] else 0.0
        row[base + 9] = _gap(dq - PLAYER_R - ball_r)


# ------------------------------------------------------------------------------------- simulador
@njit(cache=True, parallel=True, nogil=True)
def _build_env(pos, vel, kick_cancel, active, ri, rf, radius, inv, act_hist, applied, delay, fp, map_idx, line_h,
               goal_x, out):
    N = pos.shape[0]
    H = act_hist.shape[2]
    for n in prange(N):
        ball = np.empty(4)
        ball[0] = pos[n, 0, 0]
        ball[1] = pos[n, 0, 1]
        ball[2] = vel[n, 0, 0]
        ball[3] = vel[n, 0, 1]
        ppos = pos[n, fp:fp + 8]
        pvel = vel[n, fp:fp + 8]
        kick_now = np.zeros(8, dtype=np.bool_)
        for q in range(8):
            kick_now[q] = applied[n, q] >= 9 and not kick_cancel[n, q]
        rteam = ri[n, K.RI_TEAM]
        # fase de masa por el valor físico (Sanguchito siempre 0,5; RS ONE 0,5 → 0,3 con el saque)
        mphase = 0.0
        for q in range(8):
            if active[n, q]:
                mphase = 0.0 if abs(inv[n, fp + q] - 0.5) < 1e-6 else 1.0
                break
        for p in range(8):
            _featurize(out[n, p], p, ball, radius[n, 0], ppos, pvel, active[n], kick_cancel[n], kick_now,
                       act_hist[n, p], H, delay[n, p], rteam, ri[n, K.RI_KIND], ri[n, K.RI_TICKS],
                       ri[n, K.RI_KO] != 0, ri[n, K.RI_KO_TEAM], ri[n, K.RI_KO_TICKS], mphase,
                       rf[n, K.RF_KSTR], map_idx, line_h, goal_x, )


def observe(env, out=None, delay=None):
    """Observación v3 de todos los jugadores de `env` (N, 8, OBS_DIM). `delay` (N, 8): retardo que se le informa a la
    política (por defecto, el verdadero `env.delay`; en la sala el bot informa un valor fijo)."""
    if out is None:
        out = np.empty((env.N, 8, OBS_DIM), dtype=np.float32)
    line_h, goal_x, map_idx = map_geometry(env.map)
    d = env.delay if delay is None else np.ascontiguousarray(np.broadcast_to(delay, env.delay.shape), env.delay.dtype)
    _build_env(env.pos, env.vel, env.kick_cancel, env.active, env.ri, env.rf, env.radius, env.inv, env.act_hist,
               env._s_act, d, env.fp, map_idx, line_h, goal_x, out)
    return out


# ------------------------------------------------------------------------------------- grabaciones
@njit(cache=True, parallel=True, nogil=True)
def build_samples(t_obs, t_lab, slot, delay, ball, ball_r, pos, vel, inp, kicking, state, rkind, rteam, rage,
                  ko_team, ko_age, mass, kstr, map_idx, line_h_of_map, goal_x, out):
    """Observaciones de muestras de grabaciones (arrays del caché `x4_ticks` concatenados).

    Convención de índices (la del kernel): el registro f de una grabación es el estado después del tick f y
    la entrada que el host aplicó en ese tick. Una decisión tomada viendo el estado S_t con retardo D se
    aplica en los ticks que producen S_{t+D+1}..S_{t+D+3}. Así, para la etiqueta `t_lab[b]` (entrada del
    registro t_lab) con retardo D: t_obs = t_lab − D − 1, la feature de retardo es D y la decisión propia h
    es la entrada del registro t_lab − 3(h+1). "Patada en curso" = entrada del propio registro observado con
    patada y no cancelada (la acción aplicada en el tick que produjo S_t_obs, como `env._s_act`); patada
    cancelada = tecla apretada y el motor no la está usando.
    """
    B = t_obs.shape[0]
    for b in prange(B):
        t = t_obs[b]
        p = slot[b]
        active = np.ones(8, dtype=np.bool_)
        kc = np.zeros(8, dtype=np.bool_)
        kn = np.zeros(8, dtype=np.bool_)
        for q in range(8):
            held = (inp[t, q] & 16) != 0
            kc[q] = held and not kicking[t, q]
            kn[q] = held and not kc[q]
        hist = np.zeros(N_HIST, dtype=np.int64)
        for h in range(N_HIST):
            v = inp[t_lab[b] - 3 * (h + 1), p]
            ddx = ((v >> 3) & 1) - ((v >> 2) & 1)
            ddy = ((v >> 1) & 1) - (v & 1)
            m = _move_index(ddx, ddy)
            hist[h] = m + (9 if (v & 16) != 0 else 0)
        mi = map_idx[t]
        ko = state[t] == 0
        rt = rteam[t] if (rkind[t] > 0 and not ko) else -1
        _featurize(out[b], p, ball[t], ball_r[t], pos[t], vel[t], active, kc, kn, hist, N_HIST, delay[b],
                   rt, rkind[t], rage[t], ko, ko_team[t], ko_age[t], 0.0 if abs(mass[t] - 0.5) < 1e-6 else 1.0, kstr[t],
                   mi, line_h_of_map[mi], goal_x)


@njit(cache=True, inline="always")
def _move_index(dx, dy):
    # MOVE_DIRS de bridge/compare_sim: (0,0) (0,-1) (1,-1) (1,0) (1,1) (0,1) (-1,1) (-1,0) (-1,-1)
    if dx == 0:
        return 0 if dy == 0 else (1 if dy < 0 else 5)
    if dx > 0:
        return 2 if dy < 0 else (3 if dy == 0 else 4)
    return 8 if dy < 0 else (7 if dy == 0 else 6)


def line_h_table():
    return np.array([map_geometry(m)[0] for m in MAP_NAMES], dtype=np.float64)
