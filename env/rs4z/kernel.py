"""Kernel numba de RS4-Z: una llamada por decisión, todos los ticks y el árbitro dentro.

Por partido y por tick (orden del motor de HaxBall y del script de la sala):
  1. acciones efectivas (cola de latencia por jugador)
  2. [v1] protección emulada de saques (`RSOneReferee.protect`)
  3. física: entradas → integración → colisiones → saque inicial / gol (`sim/physics.step_batch`)
  4. árbitro post-tick: contacto exacto y último toque, impulsos y curva, liberación,
     salida y saque nuevo (`RSOneReferee.post_tick`) con las correcciones v2 por bandera
  5. reloj, plazos de entrenamiento y fin de partido

El modo v1 (flags = 0) es el árbitro histórico (sin las correcciones del script; no usar).
El modo v2 usa los mecanismos reales del script
(`env/rs4z/contract.py`): grupos c0/c1, discos del script y masa por fase.

Layout de discos por partido: [pelota, disco rojo, disco azul, disco ambos, discos del mapa..., 8 jugadores].
Jugadores 0..3 rojos, 4..7 azules; `active` desactiva los que no juegan (máscara y grupo 0).
Estado por partido: `ri` (enteros, índices RI_*), `rf` (reales, RF_*), `outside` (v1).
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

from sim.physics import KICK_REACH, _collide_env, _cross
from env.rs4z.numba_cache import guard

guard(__file__, ["sim/physics.py", "sim/stadium.py", "env/rs4z/contract.py"])
from sim.stadium import BLUEKO, C0, C1, PLAYER_MASK, REDKO

from .contract import (CORNER, FIX_CLOCK, FIX_ENGINE, FIX_LATERAL, FIX_MASS, FIX_SANGU, FIX_SEGBOOST, FIX_STRIP,
                       GOAL_KICK, LATERAL, PI, SD_BLUE, SD_BOTH, SD_RED)

# ---------------------------------------------------------------------------- estado entero
RI_TEAM = 0          # equipo que saca (-1 sin saque)
RI_KIND = 1          # LATERAL / CORNER / GOAL_KICK (0 sin saque)
RI_TICKS = 2         # ticks desde que empezó el saque
RI_BOOST = 3         # ticks hasta el impulso del saque
RI_GRAV = 4          # ticks restantes de curva
RI_ARMED = 5         # la salida se cobra al cruzar desde adentro
RI_LAST = 6          # último toque (equipo, -1 desconocido)
RI_MASS = 7          # fase de masa: 0 mapa (0,5), 1 script (0,3)
RI_KO = 8            # saque inicial pendiente
RI_KO_TEAM = 9       # equipo del saque inicial
RI_KO_TICKS = 10     # ticks esperando el saque inicial
RI_CLOCK = 11        # ticks de reloj del partido
RI_LEN = 12          # duración del partido (ticks de reloj)
RI_SCORE0 = 13
RI_SCORE1 = 14
RI_LAST_P = 15       # último jugador con contacto (-1)
RI_LAT_KICKED = 16   # patada durante el lateral, persiste hasta entrada/reposicionamiento
RI_C0 = 17           # pelota con c0 en su máscara (FIX_SEGBOOST): -1 saque activo, >0 ticks que faltan, 0 no
RI_GHOLD = 18        # ticks que la curva del script se mantiene antes de decaer
RI_PEND = 19         # cobro demorado: ticks hasta que la sala coloca la pelota (0 sin cobro pendiente)
RI_PEND_KIND = 20
RI_PEND_TEAM = 21
RI_SIZE = 22
# ---------------------------------------------------------------------------- estado real
RF_SPOT_X = 0
RF_SPOT_Y = 1
RF_BOOST = 2         # factor del impulso pendiente
RF_KSTR = 3          # kickStrength del partido (variante del mapa)
RF_BALL_INV = 4      # invMass de la pelota
RF_PEND_X = 5        # punto del cobro pendiente
RF_PEND_Y = 6
RF_SIZE = 7

# ---------------------------------------------------------------------------- parámetros
_LINE_H = PI["line_half_h"]
_LAT_Y = PI["lateral_ball_y"]
_LAT_MARGIN = PI["lateral_x_margin"]
_LAT_REL = PI["lateral_release_y"]
_LAT_RUN = PI["lateral_run_distance"]
_LAT_TIMEOUT = PI["lateral_timeout_ticks"]
_SPOT_REL = PI["spot_release_distance"]
_CX = PI["corner_x"]
_CY = PI["corner_y"]
_GX = PI["goal_kick_x"]
_GY = PI["goal_kick_y"]
_BAR_Y = PI["barrier_y"]
_PUSH_Y = PI["push_y"]
_PUSH_DX = PI["push_dx"]
_BOX_F = PI["box_front"]
_BOX_H = PI["box_half_h"]
_BOX_PUSH = PI["box_push_x"]
_PLAY_INV = PI["play_inv_mass"]
_PIECE_INV = PI["piece_inv_mass"]
_C_BOOST = PI["corner_boost"]
_C_DELAY = PI["corner_boost_delay"]
_G_BOOST = PI["goal_kick_boost"]
_G_DELAY = PI["goal_kick_boost_delay"]
_C_ALONG = PI["corner_gravity_along"]
_C_ALONG_V = PI["corner_gravity_along_per_speed"]
_C_PERP = PI["corner_gravity_perp"]
_G_GRAV = PI["goal_kick_gravity"]
_DECAY = PI["gravity_decay"]
_GRAV_T = PI["gravity_ticks"]
_SAFETY = PI["safety_ticks"]
_C_CLEAR = PI["corner_rival_clearance"]
_SPOT_CLEAR = PI["spot_clearance"]
_HOLD = PI["goal_kick_hold_ticks"]
_MAP_INV = PI["map_inv_mass"]
_CD_X = PI["corner_disc_x"]
_CD_Y = PI["corner_disc_y"]
_CD_R = PI["corner_disc_radius"]
_SPOT_R = PI["spot_disc_radius"]
_LAT_PUSH_MIN = PI["lat_push_min_y"]
_BOX_PAD = PI["box_push_pad"]
_C_SPEED = PI["corner_kick_speed"]
_G_SPEED = PI["goal_kick_speed"]
_C_GRAV_Y = PI["corner_grav_y"]
_C_GRAV_KVX = PI["corner_grav_kvx"]
_G_GRAV_KVY = PI["goal_kick_grav_kvy"]
_GRAV_HOLD = PI["grav_hold_ticks"]
_FORM_X = PI["form0_x"]  # form{i}_x, form{i}_y consecutivos
_G_GRAV_KVX = PI["goal_kick_grav_kvx"]
_GRAV_HOLD_GK = PI["grav_hold_ticks_gk"]
_G_TOUCH = PI["grav_touch_stop"]
_BALL_C0_T = PI["ball_c0_ticks"]
_PEND_LAT_LO = PI["pend_lat_min"]
_PEND_LAT_HI = PI["pend_lat_max"]
_PEND_C_LO = PI["pend_corner_min"]
_PEND_C_HI = PI["pend_corner_max"]
_PEND_G_LO = PI["pend_gk_min"]
_PEND_G_HI = PI["pend_gk_max"]
_PEND_LAT_DISC = PI["pend_lat_disc"]

GROUP_RED = 2
GROUP_BLUE = 4

# eventos por decisión (columnas de `ev`)
EV_GOAL = 0          # +1 rojo / -1 azul
EV_OUT = 1           # salida cobrada
EV_START = 2         # tipo del último saque iniciado
EV_EXEC = 3          # tipo del último saque ejecutado
EV_FORFEIT = 4       # equipo que perdió un saque por plazo (-1 si ninguno)
EV_KO_TAKEN = 5      # se ejecutó el saque inicial
EV_MATCH_END = 6
EV_TICKS = 7         # ticks simulados en la decisión
EV_SAFETY = 8        # liberación de seguridad (saque o saque inicial)
EV_FWHY = 9          # causa del saque perdido (EV_FORFEIT): FW_* (0 si ninguno)
EV_SIZE = 10
FW_BAD_THROW = 1     # lateral mal ejecutado (regla de la sala: corrida > 270 px o entrada sin patada)
FW_LAT_TIME = 2      # lateral vencido por tiempo (regla de la sala)
FW_LATE = 3          # plazo de entrenamiento de un saque
FW_KICKOFF = 4       # saque inicial vencido (plazo de entrenamiento)


# ============================================================================ utilidades
@njit(cache=True, inline="always")
def _sign(v):
    return 1.0 if v >= 0.0 else -1.0


@njit(cache=True, inline="always")
def _team_group(t):
    return GROUP_RED if t == 0 else GROUP_BLUE


@njit(cache=True)
def _circle_obstacle(pos, vel, fp, P, cx, cy, radius, sel):
    """`rs_one_referee._circle_obstacle` para un partido: borde del disco, sin velocidad hacia el centro."""
    for p in range(P):
        if not sel[p]:
            continue
        k = fp + p
        ox = pos[k, 0] - cx
        oy = pos[k, 1] - cy
        d = math.hypot(ox, oy)
        if d < radius:
            m = d if d > 1e-9 else 1e-9
            ux = ox / m
            uy = oy / m
            if d <= 1e-9:
                ux = 1.0
                uy = 0.0
            pos[k, 0] = cx + ux * radius
            pos[k, 1] = cy + uy * radius
            inward = vel[k, 0] * ux + vel[k, 1] * uy
            if inward > 0.0:
                inward = 0.0
            vel[k, 0] -= inward * ux
            vel[k, 1] -= inward * uy


# ============================================================================ física
@njit(cache=True)
def physics_tick(pos, vel, mask, group, radius, inv, bcoef, damping, kick_cancel, ri, act, move_unit,
                 fp, P, active, p_acc, p_kacc, p_kdamp, kstr, p_kback, grav,
                 v_pos, v_bcoef, v_group, v_mask,
                 s_p0, s_p1, s_curved, s_center, s_radius, s_t0, s_t1, s_bias, s_bcoef, s_group, s_mask,
                 p_normal, p_dist, p_bcoef, p_group, p_mask, g_p0, g_p1, g_team,
                 kicked, touch4, kinfo):
    """Un tick de un partido; copia del cuerpo de `sim.physics.step_batch` sin powershot.

    Devuelve el gol del tick (+1 rojo, -1 azul). Libera el saque inicial cuando la pelota se mueve.
    kinfo[p] = (distancia a la pelota, vx, vy, pelota − pateador en x, en y) antes del tick (script de Sanguchito).
    """
    K = pos.shape[0]
    # máscara de la pelota: el script le agrega c0 en córner y saque de arco (segmentos de impulso del mapa)
    if ri[RI_C0] != 0:
        mask[0] |= C0
    else:
        mask[0] &= ~C0
    # 1. entradas
    for p in range(P):
        k = fp + p
        a = act[p]
        kick_pressed = a >= 9
        if not kick_pressed:
            kick_cancel[p] = False
        is_kicking = kick_pressed and not kick_cancel[p]
        touch4[p] = False
        kicked[p] = False
        if active[p]:
            dx = pos[0, 0] - pos[k, 0]
            dy = pos[0, 1] - pos[k, 1]
            d = math.sqrt(dx * dx + dy * dy)
            if d - radius[k] - radius[0] < KICK_REACH:
                touch4[p] = True
                if is_kicking and d > 0.0:
                    kinfo[p, 0] = d
                    kinfo[p, 1] = vel[k, 0]
                    kinfo[p, 2] = vel[k, 1]
                    kinfo[p, 3] = dx         # pelota − pateador antes del tick (dirección que juzga la sala)
                    kinfo[p, 4] = dy
                    nx = dx / d
                    ny = dy / d
                    vel[0, 0] += nx * kstr * inv[0]
                    vel[0, 1] += ny * kstr * inv[0]
                    vel[k, 0] -= nx * p_kback * inv[k]
                    vel[k, 1] -= ny * p_kback * inv[k]
                    kick_cancel[p] = True
                    kicked[p] = True
        is_kicking = kick_pressed and not kick_cancel[p]
        acc = p_kacc if is_kicking else p_acc
        m = a % 9
        vel[k, 0] += move_unit[m, 0] * acc
        vel[k, 1] += move_unit[m, 1] * acc
    # 2. integración
    bx0 = pos[0, 0]
    by0 = pos[0, 1]
    for k in range(K):
        pos[k, 0] += vel[k, 0]
        pos[k, 1] += vel[k, 1]
        dmp = damping[k]
        if k >= fp:
            p = k - fp
            if act[p] >= 9 and not kick_cancel[p]:
                dmp = p_kdamp
        if k == 0:
            vel[0, 0] += grav[0]
            vel[0, 1] += grav[1]
        vel[k, 0] *= dmp
        vel[k, 1] *= dmp
    # 3. colisiones
    _collide_env(pos, vel, mask, radius, inv, bcoef, group,
                 v_pos, v_bcoef, v_group, v_mask,
                 s_p0, s_p1, s_curved, s_center, s_radius, s_t0, s_t1, s_bias, s_bcoef, s_group, s_mask,
                 p_normal, p_dist, p_bcoef, p_group, p_mask)
    # 4. estado
    goal = 0
    if ri[RI_KO] != 0:
        if vel[0, 0] != 0.0 or vel[0, 1] != 0.0:
            ri[RI_KO] = 0
            for p in range(P):
                mask[fp + p] = PLAYER_MASK if active[p] else 0
    else:
        qx = pos[0, 0]
        qy = pos[0, 1]
        mvx = qx - bx0
        mvy = qy - by0
        for g in range(g_p0.shape[0]):
            ax = g_p0[g, 0]
            ay = g_p0[g, 1]
            gx = g_p1[g, 0] - ax
            gy = g_p1[g, 1] - ay
            c1 = _cross(qx - ax, qy - ay, mvx, mvy) * _cross(qx - g_p1[g, 0], qy - g_p1[g, 1], mvx, mvy)
            c2 = _cross(bx0 - ax, by0 - ay, gx, gy) * _cross(qx - ax, qy - ay, gx, gy)
            if c1 <= 0.0 and c2 <= 0.0 and (mvx != 0.0 or mvy != 0.0):
                goal = -1 if g_team[g] == 0 else 1
                break
    return goal


# ============================================================================ árbitro
@njit(cache=True)
def clear_piece(pos, radius, group, fp, P, active, team, sd_home, flags):
    """Deshace los mecanismos v2 de un saque: grupos de equipo, discos del script fuera."""
    if flags & FIX_ENGINE:
        for p in range(P):
            group[fp + p] = _team_group(team[p]) if active[p] else 0
        for s in range(3):
            pos[1 + s, 0] = sd_home[s, 0]
            pos[1 + s, 1] = sd_home[s, 1]
            radius[1 + s] = 0.0


@njit(cache=True)
def start_piece(kind, taker, spot_x, spot_y, pos, vel, group, radius, inv, kick_cancel, grav, active,
                team, fp, P, prm, flags, sd_home, ri, rf, outside):
    """`RSOneReferee.start` (acciones del script al iniciar un saque) para un partido."""
    pr = radius[fp] if P > 0 else 15.0
    sx = _sign(spot_x)
    sy = _sign(spot_y)
    clear_piece(pos, radius, group, fp, P, active, team, sd_home, flags)
    pos[0, 0] = spot_x
    pos[0, 1] = spot_y
    vel[0, 0] = 0.0
    vel[0, 1] = 0.0
    inv[0] = rf[RF_BALL_INV]
    grav[0] = 0.0
    grav[1] = 0.0
    for p in range(P):
        kick_cancel[p] = False
        outside[p] = False
    ri[RI_ARMED] = 0
    ri[RI_BOOST] = 0
    ri[RI_GRAV] = 0
    if kind == LATERAL:
        if ri[RI_C0] < 0:
            ri[RI_C0] = 0
        for p in range(P):
            if not active[p] or team[p] == taker:
                continue
            k = fp + p
            if sy * pos[k, 1] > prm[_LAT_PUSH_MIN] and abs(pos[k, 0] - spot_x) < prm[_PUSH_DX]:
                pos[k, 1] = sy * prm[_PUSH_Y]  # el script sólo fija y
            if flags & FIX_ENGINE:
                group[k] = _team_group(team[p]) | C1
            else:
                outside[p] = sy * pos[k, 1] > prm[_BAR_Y] + pr
    else:
        for p in range(P):
            inv[fp + p] = prm[_PIECE_INV]
        if flags & FIX_SANGU:
            inv[0] = 0.0  # pelota fija hasta la patada válida del ejecutor
        if flags & FIX_SEGBOOST:
            ri[RI_C0] = -1
        if kind == GOAL_KICK:
            for p in range(P):
                if not active[p] or team[p] == taker:
                    continue
                k = fp + p
                pad = prm[_BOX_PAD]
                if sx * pos[k, 0] > prm[_BOX_F] - pad and abs(pos[k, 1]) < prm[_BOX_H] + pad:
                    pos[k, 0] = sx * prm[_BOX_PUSH]  # el script sólo fija x
                if flags & FIX_ENGINE:
                    group[k] = _team_group(team[p]) | C0
        if flags & FIX_ENGINE:
            if kind == GOAL_KICK:
                if prm[_SPOT_R] > 0.0:
                    pos[SD_BOTH, 0] = spot_x
                    pos[SD_BOTH, 1] = spot_y
                    radius[SD_BOTH] = prm[_SPOT_R]
            else:
                d = SD_RED if sx < 0 else SD_BLUE  # disco del equipo que defiende ese arco
                pos[d, 0] = sx * prm[_CD_X]
                pos[d, 1] = sy * prm[_CD_Y]
                radius[d] = prm[_CD_R]
                # la sala despeja el punto durante la demora del cobro (disco 3); acá, una vez
                _circle_obstacle(pos, vel, fp, P, spot_x, spot_y, prm[_SPOT_CLEAR], active)
        else:
            # Disco transitorio del script (radio 18): despeja el punto exacto del saque.
            _circle_obstacle(pos, vel, fp, P, spot_x, spot_y, prm[_SPOT_CLEAR], active)
    ri[RI_TEAM] = taker
    ri[RI_KIND] = kind
    ri[RI_TICKS] = 0
    rf[RF_SPOT_X] = spot_x
    rf[RF_SPOT_Y] = spot_y
    ri[RI_LAST] = -1
    ri[RI_LAST_P] = -1
    ri[RI_LAT_KICKED] = 0


@njit(cache=True)
def protect_v1(pos, vel, fp, P, active, team, prm, ri, rf, outside):
    """`RSOneReferee.protect` (sólo v1: emulación de las barreras del script)."""
    if ri[RI_TEAM] < 0:
        return
    pr = 15.0
    sy = 1.0 if rf[RF_SPOT_Y] >= 0 else -1.0
    sx = 1.0 if rf[RF_SPOT_X] >= 0 else -1.0
    kind = ri[RI_KIND]
    owner = ri[RI_TEAM]
    for p in range(P):
        if not active[p] or team[p] == owner:
            continue
        k = fp + p
        if kind == LATERAL:
            inner = (not outside[p]) and sy * pos[k, 1] > prm[_BAR_Y] - pr
            if inner:
                pos[k, 1] = sy * (prm[_BAR_Y] - pr)
            outer = outside[p] and sy * pos[k, 1] < prm[_BAR_Y] + pr
            if outer:
                pos[k, 1] = sy * (prm[_BAR_Y] + pr)
            if inner or outer:
                vel[k, 1] = 0.0
        elif kind == GOAL_KICK:
            if sx * pos[k, 0] > prm[_BOX_F] - pr and abs(pos[k, 1]) < prm[_BOX_H] + pr:
                pos[k, 0] = sx * (prm[_BOX_F] - pr)
                vel[k, 0] = 0.0
    if kind == GOAL_KICK and ri[RI_TICKS] < prm[_HOLD]:
        _circle_obstacle(pos, vel, fp, P, rf[RF_SPOT_X], rf[RF_SPOT_Y], prm[_SPOT_CLEAR], active)
    if kind == CORNER:
        sel = np.zeros(P, dtype=np.bool_)
        for p in range(P):
            sel[p] = active[p] and team[p] != owner
        _circle_obstacle(pos, vel, fp, P, rf[RF_SPOT_X], rf[RF_SPOT_Y], prm[_C_CLEAR], sel)


@njit(cache=True)
def _schedule(kind, pos, vel, grav, k_kicker, prm, ri, rf):
    """Impulso diferido y curva del córner / saque de arco (`RSOneReferee._schedule`)."""
    vx = vel[0, 0]
    vy = vel[0, 1]
    speed = math.hypot(vx, vy)
    if speed < 1e-6:
        return
    ux = vx / speed
    uy = vy / speed
    nx = -uy
    ny = ux
    pvx = vel[k_kicker, 0]
    pvy = vel[k_kicker, 1]
    if kind == CORNER:
        along = prm[_C_ALONG] + prm[_C_ALONG_V] * (pvx * ux + pvy * uy)
        perp = prm[_C_PERP] * (pvx * nx + pvy * ny)
        grav[0] = along * ux + perp * nx
        grav[1] = along * uy + perp * ny
        rf[RF_BOOST] = prm[_C_BOOST]
        ri[RI_BOOST] = int(prm[_C_DELAY])
    else:
        grav[0] = prm[_G_GRAV] * pvx
        grav[1] = prm[_G_GRAV] * pvy
        rf[RF_BOOST] = prm[_G_BOOST]
        ri[RI_BOOST] = int(prm[_G_DELAY])
    ri[RI_GRAV] = int(prm[_GRAV_T])


@njit(cache=True)
def _script_curve(kind, grav, p, kinfo, prm, ri, sy):
    """Curva que el script fija al patear un córner o saque de arco (Sanguchito y familia RS ONE):
    lineal en la velocidad del pateador antes del tick; córner con componente fija hacia la cancha."""
    if kind == CORNER:
        grav[0] = prm[_C_GRAV_KVX] * kinfo[p, 1]
        grav[1] = -sy * prm[_C_GRAV_Y]
        ri[RI_GHOLD] = int(prm[_GRAV_HOLD])
    else:
        grav[0] = prm[_G_GRAV_KVX] * kinfo[p, 1]
        grav[1] = prm[_G_GRAV_KVY] * kinfo[p, 2]
        ri[RI_GHOLD] = int(prm[_GRAV_HOLD_GK])
    ri[RI_BOOST] = 0
    ri[RI_GRAV] = int(prm[_GRAV_T])


@njit(cache=True)
def _sangu_kick(kind, pos, vel, grav, fp, p, kinfo, prm, ri, sy):
    """Patada válida del ejecutor en córner o saque de arco de Sanguchito: velocidad y curva del script."""
    k = fp + p
    s = prm[_C_SPEED] if kind == CORNER else prm[_G_SPEED]
    d0 = kinfo[p, 0]
    vel[0, 0] = s * (pos[0, 0] - pos[k, 0]) / d0
    vel[0, 1] = s * (pos[0, 1] - pos[k, 1]) / d0
    _script_curve(kind, grav, p, kinfo, prm, ri, sy)


@njit(cache=True)
def set_piece(side, rand_bit, pos, vel, group, radius, inv, kick_cancel, grav, active, team, fp, P,
              prm, flags, sd_home, ri, rf, outside, line_w, rand_u):
    """Salida cobrada: `RSOneReferee.set_piece` para un partido. Devuelve el tipo iniciado (0 si la sala
    demora la colocación: queda pendiente `RI_PEND` ticks, con el disco del punto puesto)."""
    bx = pos[0, 0]
    by = pos[0, 1]
    last = ri[RI_LAST]
    if last < 0:
        last = rand_bit
    sx = _sign(bx)
    sy = _sign(by)
    line_h = prm[_LINE_H]
    if flags & FIX_STRIP:
        is_lateral = side
    else:
        is_lateral = abs(by) > line_h and abs(bx) < line_w
    if is_lateral:
        kind = LATERAL
        taker = 1 - last
        margin = line_w - prm[_LAT_MARGIN]
        x = bx
        if x < -margin:
            x = -margin
        if x > margin:
            x = margin
        spot_x = x
        spot_y = sy * prm[_LAT_Y]
    else:
        defender = 1 if sx > 0 else 0  # el arco de +x es del azul
        if last == defender:
            kind = CORNER
            taker = 1 - defender
            spot_x = sx * prm[_CX]
            spot_y = sy * prm[_CY]
        else:
            kind = GOAL_KICK
            taker = defender
            spot_x = sx * prm[_GX]
            spot_y = sy * prm[_GY]
    if kind == LATERAL:
        lo = prm[_PEND_LAT_LO]
        hi = prm[_PEND_LAT_HI]
    elif kind == CORNER:
        lo = prm[_PEND_C_LO]
        hi = prm[_PEND_C_HI]
    else:
        lo = prm[_PEND_G_LO]
        hi = prm[_PEND_G_HI]
    delay = 0
    if hi > 0.0:
        delay = int(lo + rand_u * (hi - lo + 1.0))
        if delay > hi:
            delay = int(hi)
    if delay > 0:
        ri[RI_PEND] = delay
        ri[RI_PEND_KIND] = kind
        ri[RI_PEND_TEAM] = taker
        rf[RF_PEND_X] = spot_x
        rf[RF_PEND_Y] = spot_y
        ri[RI_ARMED] = 0
        if (flags & FIX_ENGINE) and prm[_SPOT_R] > 0.0 and (kind != LATERAL or prm[_PEND_LAT_DISC] > 0.0):
            pos[SD_BOTH, 0] = spot_x       # la sala despeja el punto mientras demora la colocación
            pos[SD_BOTH, 1] = spot_y
            radius[SD_BOTH] = prm[_SPOT_R]
        return 0
    start_piece(kind, taker, spot_x, spot_y, pos, vel, group, radius, inv, kick_cancel, grav, active,
                team, fp, P, prm, flags, sd_home, ri, rf, outside)
    return kind


@njit(cache=True)
def forfeit_piece(pos, vel, group, radius, inv, kick_cancel, grav, active, team, fp, P, prm, flags,
                  sd_home, ri, rf, outside):
    """Plazo de entrenamiento vencido: el saque pasa al rival (lateral → lateral rival; córner →
    saque de arco del defensor; saque de arco → córner del atacante). Devuelve el equipo que lo perdió."""
    owner = ri[RI_TEAM]
    kind = ri[RI_KIND]
    sx = _sign(rf[RF_SPOT_X])
    sy = _sign(rf[RF_SPOT_Y])
    if kind == LATERAL:
        new_kind = LATERAL
        x = rf[RF_SPOT_X]
        y = rf[RF_SPOT_Y]
    elif kind == CORNER:
        new_kind = GOAL_KICK
        x = sx * prm[_GX]
        y = sy * prm[_GY]
    else:
        new_kind = CORNER
        x = sx * prm[_CX]
        y = sy * prm[_CY]
    start_piece(new_kind, 1 - owner, x, y, pos, vel, group, radius, inv, kick_cancel, grav, active,
                team, fp, P, prm, flags, sd_home, ri, rf, outside)
    return owner


@njit(cache=True)
def post_tick(goal, kicked, contact, pos, vel, group, radius, inv, kick_cancel, grav, active, team,
              fp, P, prm, flags, sd_home, ri, rf, outside, line_w, goal_hh, rand_bit, deadline, ev, kinfo, rand_u):
    """`RSOneReferee.post_tick` para un partido, más las correcciones v2.

    Devuelve 1 si el saque activo venció su plazo de entrenamiento en este tick.
    """
    rb = radius[0]
    reach = radius[fp] + rb + 0.01
    t0 = False
    t1 = False
    any_contact = False
    best = -1
    best_d = 1e18
    for p in range(P):
        c = False
        if active[p]:
            dx = pos[fp + p, 0] - pos[0, 0]
            dy = pos[fp + p, 1] - pos[0, 1]
            d = math.hypot(dx, dy)
            c = d <= reach or kicked[p]
            if c and d < best_d:
                best_d = d
                best = p
        contact[p] = c
        if c:
            any_contact = True
            if team[p] == 0:
                t0 = True
            else:
                t1 = True
    if t0 and not t1:
        ri[RI_LAST] = 0
    elif t1 and not t0:
        ri[RI_LAST] = 1
    if best >= 0:
        ri[RI_LAST_P] = best
    owner = ri[RI_TEAM]
    active_sp = owner >= 0
    if active_sp:
        ri[RI_TICKS] += 1
    # 1. impulsos y curva de saques ya liberados
    pending = ri[RI_BOOST] > 0
    fire = False
    if pending:
        ri[RI_BOOST] -= 1
        fire = ri[RI_BOOST] == 0
        if fire:
            vel[0, 0] *= rf[RF_BOOST]
            vel[0, 1] *= rf[RF_BOOST]
    if ri[RI_C0] > 0:
        ri[RI_C0] -= 1           # el script le saca c0 a la pelota `ball_c0_ticks` después de la patada
    if ri[RI_GRAV] > 0 and (flags & (FIX_SANGU | FIX_SEGBOOST)):
        # curva del script: se mantiene RI_GHOLD ticks, después decae; en RS ONE un toque la corta
        ri[RI_GRAV] -= 1
        if ri[RI_GRAV] == 0 or goal != 0 or (prm[_G_TOUCH] > 0.0 and any_contact):
            grav[0] = 0.0
            grav[1] = 0.0
            ri[RI_GRAV] = 0
        elif prm[_GRAV_T] - ri[RI_GRAV] >= ri[RI_GHOLD]:
            grav[0] *= prm[_DECAY]
            grav[1] *= prm[_DECAY]
    elif ri[RI_GRAV] > 0:
        stop = (any_contact and not (pending or fire)) or goal != 0
        if stop:
            grav[0] = 0.0
            grav[1] = 0.0
            ri[RI_GRAV] = 0
        else:
            grav[0] *= prm[_DECAY]
            grav[1] *= prm[_DECAY]
            ri[RI_GRAV] -= 1
            if ri[RI_GRAV] == 0:
                grav[0] = 0.0
                grav[1] = 0.0
    # cobro demorado: la sala coloca la pelota al vencer la demora
    if ri[RI_PEND] > 0:
        ri[RI_PEND] -= 1
        if ri[RI_PEND] == 0:
            start_piece(ri[RI_PEND_KIND], ri[RI_PEND_TEAM], rf[RF_PEND_X], rf[RF_PEND_Y], pos, vel, group, radius, inv,
                        kick_cancel, grav, active, team, fp, P, prm, flags, sd_home, ri, rf, outside)
            ev[EV_START] = ri[RI_KIND]
    # 2. liberación del saque activo
    late = 0
    if active_sp:
        kind = ri[RI_KIND]
        own_kick = -1
        any_kick = False
        for p in range(P):
            if kicked[p]:
                any_kick = True
                if team[p] == owner and own_kick < 0:
                    own_kick = p
        lateral_in = kind == LATERAL and abs(pos[0, 1]) < prm[_LAT_REL]
        if kind == LATERAL and (flags & FIX_LATERAL):
            if any_kick:
                ri[RI_LAT_KICKED] = 1
            # El script verifica X mientras la pelota sigue fuera. Una entrada sin
            # patada es mal saque; las patadas cuentan también en ticks anteriores.
            too_far = abs(pos[0, 1]) > prm[_LAT_REL] and abs(pos[0, 0] - rf[RF_SPOT_X]) > prm[_LAT_RUN]
            # sólo si la pelota estaba colocada afuera (el script la pone en ±lateral_ball_y): un saque iniciado con la
            # pelota ya adentro (estado grabado a mitad de la colocación) no puede "entrar sin patada", y sin esta
            # condición el lateral rebotaba de un equipo al otro en cada decisión
            pushed_in = lateral_in and ri[RI_LAT_KICKED] == 0 and abs(rf[RF_SPOT_Y]) >= prm[_LAT_REL]
            timed_out = ri[RI_TICKS] >= prm[_LAT_TIMEOUT] and not lateral_in
            if too_far or pushed_in or timed_out:
                ev[EV_FORFEIT] = forfeit_piece(pos, vel, group, radius, inv, kick_cancel, grav,
                    active, team, fp, P, prm, flags, sd_home, ri, rf, outside)
                ev[EV_FWHY] = FW_BAD_THROW if (too_far or pushed_in) else FW_LAT_TIME
                ev[EV_START] = LATERAL
                return 0
        moved = math.hypot(pos[0, 0] - rf[RF_SPOT_X], pos[0, 1] - rf[RF_SPOT_Y]) > prm[_SPOT_REL]
        dead_ball = kind != LATERAL and (any_kick or moved)
        sangu_kick = -1
        if (flags & FIX_SANGU) and kind != LATERAL:
            # sólo cuenta la patada del ejecutor que manda la pelota hacia la cancha, juzgada con la posición del
            # pateador al patear (antes del tick): en x siempre; en el córner también en y. En 498 grabaciones esta
            # regla separa las 2746 patadas que liberan de las 346 ignoradas sin errores; con la posición después
            # del tick fallaba en 43 (`reports/x4/conformance_x4.md`).
            sx_spot = 1.0 if rf[RF_SPOT_X] >= 0.0 else -1.0
            sy_spot = 1.0 if rf[RF_SPOT_Y] >= 0.0 else -1.0
            for p in range(P):
                if kicked[p] and team[p] == owner and kinfo[p, 3] * sx_spot < 0.0:
                    if kind == CORNER and kinfo[p, 4] * sy_spot > 0.0:
                        continue
                    sangu_kick = p
                    break
            dead_ball = sangu_kick >= 0
        released = lateral_in or dead_ball or ri[RI_TICKS] >= prm[_SAFETY] or goal != 0
        executed = lateral_in or dead_ball or goal != 0
        if deadline > 0 and (not executed) and ri[RI_TICKS] == deadline:
            late = 1
        if (flags & FIX_MASS) and kind == LATERAL and own_kick >= 0:
            # F1: el script fija 0,3 con la patada del lateral
            for p in range(P):
                inv[fp + p] = prm[_PLAY_INV]
            ri[RI_MASS] = 1
        if released and (flags & FIX_SANGU) and kind != LATERAL:
            inv[0] = rf[RF_BALL_INV]
            if sangu_kick >= 0:
                _sangu_kick(kind, pos, vel, grav, fp, sangu_kick, kinfo, prm, ri,
                            1.0 if rf[RF_SPOT_Y] >= 0.0 else -1.0)
        elif released and (flags & FIX_SEGBOOST) and kind != LATERAL:
            # el impulso lo dan los segmentos c0 del mapa (motor); el script fija la curva y retira c0 después
            ri[RI_C0] = int(prm[_BALL_C0_T])
            if own_kick >= 0:
                _script_curve(kind, grav, own_kick, kinfo, prm, ri, 1.0 if rf[RF_SPOT_Y] >= 0.0 else -1.0)
        elif released and own_kick >= 0 and (kind == CORNER or kind == GOAL_KICK):
            _schedule(kind, pos, vel, grav, fp + own_kick, prm, ri, rf)
        if (flags & FIX_ENGINE) and kind == GOAL_KICK and ri[RI_TICKS] == int(prm[_HOLD]):
            # el disco del punto se retira 180 ticks después de colocar la pelota
            pos[SD_BOTH, 0] = sd_home[2, 0]
            pos[SD_BOTH, 1] = sd_home[2, 1]
            radius[SD_BOTH] = 0.0
        if released:
            if (flags & FIX_MASS) == 0 or kind != LATERAL:
                for p in range(P):
                    inv[fp + p] = prm[_PLAY_INV]
                ri[RI_MASS] = 1
            for p in range(P):
                outside[p] = False
            clear_piece(pos, radius, group, fp, P, active, team, sd_home, flags)
            ri[RI_TEAM] = -1
            ri[RI_KIND] = 0
            if executed:
                ev[EV_EXEC] = kind
            else:
                ev[EV_SAFETY] = 1
            if kind == LATERAL and abs(pos[0, 0]) > line_w + rb:
                # Lateral pateado por afuera hacia el córner que "entra" (|y| bajo el umbral) detrás de la línea de fondo:
                # la pelota nunca pisa la cancha, así que la salida no se volvía a vigilar y quedaba viva afuera hasta el
                # final del partido (8,5% del tiempo de las evaluaciones del 2026-10-06, con los jugadores sin saber qué
                # hacer). La sala lo cobra como salida por el fondo (saque de arco o córner según quién la tocó último):
                # los 5 casos de las 498 grabaciones de Sanguchito, un tick después de la liberación.
                ri[RI_ARMED] = 1
    # 3. salida en este mismo tick (sin gol): saque nuevo
    bx = pos[0, 0]
    by = pos[0, 1]
    line_h = prm[_LINE_H]
    inside = abs(by) <= line_h + rb and abs(bx) <= line_w + rb
    if inside and ri[RI_TEAM] < 0:
        ri[RI_ARMED] = 1
    if ri[RI_ARMED] != 0 and ri[RI_TEAM] < 0 and ri[RI_PEND] == 0 and goal == 0:
        if flags & FIX_STRIP:
            side = abs(by) > line_h + rb and abs(bx) <= line_w + rb
        else:
            side = abs(by) > line_h + rb and abs(bx) < line_w
        end = abs(bx) > line_w + rb and abs(by) > goal_hh
        if side or end:
            ev[EV_OUT] = 1
            ev[EV_START] = set_piece(side, rand_bit, pos, vel, group, radius, inv, kick_cancel, grav, active,
                                     team, fp, P, prm, flags, sd_home, ri, rf, outside, line_w, rand_u)
            if (flags & FIX_ENGINE) == 0:
                # rs_one_v1 protege dentro de set_piece y otra vez al final del post-tick
                protect_v1(pos, vel, fp, P, active, team, prm, ri, rf, outside)
    if (flags & FIX_ENGINE) == 0:
        protect_v1(pos, vel, fp, P, active, team, prm, ri, rf, outside)
    return late


# ============================================================================ saque inicial
@njit(cache=True)
def reset_kickoff(ko_team, pos, vel, mask, group, radius, inv, kick_cancel, grav, active, team,
                  spawn_rank, fp, P, prm, flags, sd_home, base_mask, d_pos, ri, rf, outside,
                  spawn_x, spawn_dy, mass_after_reset):
    """Reposicionamiento de HaxBall (gol, inicio de partido o saque inicial vencido)."""
    pos[0, 0] = 0.0
    pos[0, 1] = 0.0
    vel[0, 0] = 0.0
    vel[0, 1] = 0.0
    for s in range(3):
        pos[1 + s, 0] = sd_home[s, 0]
        pos[1 + s, 1] = sd_home[s, 1]
        vel[1 + s, 0] = 0.0
        vel[1 + s, 1] = 0.0
        radius[1 + s] = 0.0
    for d in range(d_pos.shape[0]):
        pos[4 + d, 0] = d_pos[d, 0]
        pos[4 + d, 1] = d_pos[d, 1]
        vel[4 + d, 0] = 0.0
        vel[4 + d, 1] = 0.0
    for k in range(fp):
        mask[k] = base_mask[k]
    ko_flag = REDKO if ko_team == 0 else BLUEKO
    for p in range(P):
        k = fp + p
        vel[k, 0] = 0.0
        vel[k, 1] = 0.0
        kick_cancel[p] = False
        outside[p] = False
        if active[p]:
            c = spawn_rank[p]
            if flags & FIX_SANGU:
                # el script de Sanguchito forma el rombo (posiciones del rojo; el azul, espejado en x)
                pos[k, 0] = prm[_FORM_X + 2 * c] * (1.0 if team[p] == 0 else -1.0)
                pos[k, 1] = prm[_FORM_X + 2 * c + 1]
            else:
                y = spawn_dy * ((c + 1) >> 1) * (1.0 if c % 2 == 1 else -1.0)
                pos[k, 0] = -spawn_x if team[p] == 0 else spawn_x
                pos[k, 1] = y
            mask[k] = PLAYER_MASK | ko_flag
            group[k] = _team_group(team[p])
        else:
            pos[k, 0] = 0.0
            pos[k, 1] = 5000.0 + 40.0 * p
            mask[k] = 0
            group[k] = 0
        inv[k] = mass_after_reset
    inv[0] = rf[RF_BALL_INV]
    grav[0] = 0.0
    grav[1] = 0.0
    ri[RI_TEAM] = -1
    ri[RI_KIND] = 0
    ri[RI_TICKS] = 0
    ri[RI_BOOST] = 0
    ri[RI_GRAV] = 0
    ri[RI_ARMED] = 1
    ri[RI_LAST] = -1
    ri[RI_LAST_P] = -1
    ri[RI_LAT_KICKED] = 0
    ri[RI_C0] = 0
    ri[RI_PEND] = 0
    ri[RI_MASS] = 0 if (flags & FIX_MASS) else 1
    ri[RI_KO] = 1
    ri[RI_KO_TEAM] = ko_team
    ri[RI_KO_TICKS] = 0


# ============================================================================ decisión
@njit(cache=True, parallel=True, nogil=True)
def decision_step(pos, vel, mask, group, radius, inv, bcoef, damping, kick_cancel, grav, active, team,
                  spawn_rank, ri, rf, outside, act_hist, delay, rand,
                  fp, frame_skip, prm, flags, deadline, ko_deadline, sd_home, base_mask, d_pos,
                  move_unit, p_acc, p_kacc, p_kdamp, p_kback, line_w, goal_hh, spawn_x, spawn_dy,
                  v_pos, v_bcoef, v_group, v_mask,
                  s_p0, s_p1, s_curved, s_center, s_radius, s_t0, s_t1, s_bias, s_bcoef, s_group, s_mask,
                  p_normal, p_dist, p_bcoef, p_group, p_mask, g_p0, g_p1, g_team,
                  ev, ev_kicked, ev_touch, scratch_act, scratch_kick, scratch_t4, scratch_contact, scratch_kinfo):
    """Avanza una decisión (frame_skip ticks) en todos los partidos.

    act_hist[n, p, h]: acción en coordenadas del mundo, h=0 la más nueva (una por decisión).
    delay[n, p]: retraso en ticks entre decidir y aplicar. rand[n]: uniforme para sorteos.
    Un gol corta la decisión y repone el saque inicial (saca el que recibió); el fin de partido también corta.
    """
    N = pos.shape[0]
    P = active.shape[1]
    H = act_hist.shape[2]
    map_inv = prm[_MAP_INV] if (flags & FIX_MASS) else prm[_PLAY_INV]
    for n in prange(N):
        for j in range(ev.shape[1]):
            ev[n, j] = 0
        ev[n, EV_FORFEIT] = -1
        for p in range(P):
            ev_kicked[n, p] = False
            ev_touch[n, p] = False
        rbit = 1 if rand[n] < 0.5 else 0
        ticks = 0
        for k in range(frame_skip):
            for p in range(P):
                if active[n, p]:
                    lag = delay[n, p] - k
                    h = 0
                    if lag > 0:
                        h = (lag + frame_skip - 1) // frame_skip
                        if h > H - 1:
                            h = H - 1
                    scratch_act[n, p] = act_hist[n, p, h]
                else:
                    scratch_act[n, p] = 0
            if (flags & FIX_ENGINE) == 0:
                protect_v1(pos[n], vel[n], fp, P, active[n], team, prm, ri[n], rf[n], outside[n])
            was_ko = ri[n, RI_KO] != 0
            g = physics_tick(pos[n], vel[n], mask[n], group[n], radius[n], inv[n], bcoef, damping,
                             kick_cancel[n], ri[n], scratch_act[n], move_unit, fp, P, active[n],
                             p_acc, p_kacc, p_kdamp, rf[n, RF_KSTR], p_kback, grav[n],
                             v_pos, v_bcoef, v_group, v_mask,
                             s_p0, s_p1, s_curved, s_center, s_radius, s_t0, s_t1, s_bias, s_bcoef, s_group,
                             s_mask, p_normal, p_dist, p_bcoef, p_group, p_mask, g_p0, g_p1, g_team,
                             scratch_kick[n], scratch_t4[n], scratch_kinfo[n])
            if was_ko and ri[n, RI_KO] == 0:
                ev[n, EV_KO_TAKEN] = 1
            late = post_tick(g, scratch_kick[n], scratch_contact[n], pos[n], vel[n], group[n], radius[n],
                             inv[n], kick_cancel[n], grav[n], active[n], team, fp, P, prm, flags, sd_home,
                             ri[n], rf[n], outside[n], line_w, goal_hh, rbit, deadline, ev[n], scratch_kinfo[n],
                             (rand[n] * 1000.0) % 1.0)
            for p in range(P):
                if scratch_kick[n, p]:
                    ev_kicked[n, p] = True
                if scratch_contact[n, p]:
                    ev_touch[n, p] = True
            ticks += 1
            # reloj (F2: congelado durante el saque inicial)
            if ri[n, RI_KO] != 0:
                ri[n, RI_KO_TICKS] += 1
                if (flags & FIX_CLOCK) == 0:
                    ri[n, RI_CLOCK] += 1
            else:
                ri[n, RI_KO_TICKS] = 0
                ri[n, RI_CLOCK] += 1
            if late:
                ev[n, EV_FORFEIT] = forfeit_piece(pos[n], vel[n], group[n], radius[n], inv[n], kick_cancel[n],
                                                  grav[n], active[n], team, fp, P, prm, flags, sd_home,
                                                  ri[n], rf[n], outside[n])
                ev[n, EV_START] = ri[n, RI_KIND]
                ev[n, EV_FWHY] = FW_LATE
            if ri[n, RI_KO] != 0 and ko_deadline > 0 and ri[n, RI_KO_TICKS] >= ko_deadline:
                loser = ri[n, RI_KO_TEAM]
                ev[n, EV_FORFEIT] = loser
                ev[n, EV_SAFETY] = 2
                ev[n, EV_FWHY] = FW_KICKOFF
                reset_kickoff(1 - loser, pos[n], vel[n], mask[n], group[n], radius[n], inv[n], kick_cancel[n],
                              grav[n], active[n], team, spawn_rank[n], fp, P, prm, flags, sd_home, base_mask,
                              d_pos, ri[n], rf[n], outside[n], spawn_x, spawn_dy, map_inv)
            if g != 0:
                ev[n, EV_GOAL] = g
                if g == 1:
                    ri[n, RI_SCORE0] += 1
                else:
                    ri[n, RI_SCORE1] += 1
                break
            if ri[n, RI_CLOCK] >= ri[n, RI_LEN]:
                break
        ev[n, EV_TICKS] = ticks
        if ri[n, RI_CLOCK] >= ri[n, RI_LEN]:
            ev[n, EV_MATCH_END] = 1
        elif ev[n, EV_GOAL] != 0:
            # saca el que recibió el gol (como HaxBall)
            reset_kickoff(1 if ev[n, EV_GOAL] == 1 else 0, pos[n], vel[n], mask[n], group[n], radius[n], inv[n],
                          kick_cancel[n], grav[n], active[n], team, spawn_rank[n], fp, P, prm, flags, sd_home,
                          base_mask, d_pos, ri[n], rf[n], outside[n], spawn_x, spawn_dy, map_inv)
