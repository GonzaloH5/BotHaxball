"""Kernel numba de HX11: física de HaxBall y árbitro del contrato HX11-1, por partido y por tick.

Por tick (orden del motor y del script de la sala, que actúa al final del tick; lo que deja vale para el siguiente):
  1. física: entradas (patada con la masa vigente de la pelota) → integración (gravedad de la pelota) → colisiones →
     saque inicial / gol (copia exacta de `sim.physics.step_batch` sin el powershot de otra sala)
  2. árbitro (`referee`):
     toques y posesión → offside (cobro al tocar un marcado; marcas nuevas en cada patada) → powershot (contacto,
     rampa de masa, comba de la patada cargada) → falta (contacto rival con el poseedor) → salida de la pelota →
     máquina de pelota parada (espera, colocación, barra, barreras, jugadores sin masa, rivales fantasma, patada,
     restauraciones con sus demoras) → gol (150 ticks) → saque inicial (arquero a ±1760).

Layout de discos por partido: [pelota, discos del mapa (`d_src`: 0 = barrera, 1 = disco del punto, ...), jugadores].
Jugadores 0..10 rojos (0 = arquero), 11..21 azules (11 = arquero). Arreglos por partido: pos, vel (N, K, 2); mask,
group (N, K) enteros; radius, inv (N, K); grav (N, 2). Estado entero `ri` (N, RI_SIZE) y real `rf` (N, RF_SIZE);
contacto de falta por rival `fs`, `fl` (N, P).
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

from sim.physics import KICK_REACH, _collide_env, _cross
from sim.stadium import BLUE, BLUEKO, C0, C1, PLAYER_MASK, RED, REDKO

from . import contract as C

# ---------------------------------------------------------------------------- modos
OPEN, OUT_PENDING, PIECE, FREEZE, GOAL, KICKOFF = 0, 1, 2, 3, 4, 5
# ---------------------------------------------------------------------------- estado entero
RI_MODE = 0
RI_KIND = 1          # tipo de la pelota parada pendiente o activa
RI_TEAM = 2          # equipo que saca (0 rojo, 1 azul)
RI_TICKS = 3         # ticks en el modo actual
RI_LAST = 4          # último jugador que tocó (-1)
RI_PS_CONTACT = 5    # ticks seguidos de contacto del poseedor (powershot)
RI_PS_T = 6          # ticks desde el inicio de la carga (-1 sin carga)
RI_PS_RESET = 7      # ticks hasta devolver la masa de la pelota a 1,65 tras una patada cargada (0 nada)
RI_CURVE = 8         # ticks de comba que quedan
RI_OFF_MASK = 9      # jugadores marcados en offside por la última patada (bits)
RI_EXEMPT = 10       # la próxima patada es la ejecución de un lateral / córner / saque de arco
RI_SCORE0 = 11
RI_SCORE1 = 12
RI_CLOCK = 13        # ticks de reloj
RI_KO_TEAM = 14      # equipo del saque inicial
RI_KO = 15           # saque inicial pendiente (la física lo libera al moverse la pelota)
RI_AFTER_KIND = 16   # tipo del saque recién pateado (restauraciones pendientes)
RI_AFTER_T = 17      # ticks desde la patada del saque (-1 nada pendiente)
RI_GHOSTS = 18       # jugadores sin masa
RI_GK_GHOST = 19     # rivales fantasma del saque de arco (0 no, 1 c0, 2 c1)
RI_SPOT_ON = 20      # disco del punto colocado
RI_BARRIER_ON = 21   # barrera colocada
RI_LAT_ARMED = 22    # después de un lateral: la pelota volvió a la cancha (si no, no se cobra otra salida)
RI_OFFSIDE_ON = 23   # modo offside de la sala
RI_FOULS_ON = 24     # modo faltas
RI_PS_ON = 25        # modo powershot
RI_PREV_IN_Y = 26    # la pelota estaba dentro de |y| <= 1200 el tick anterior
RI_DONE = 27         # partido terminado
RI_LEN = 28          # duración del partido en ticks de reloj
RI_EVENT = 29        # evento del tick (para el entorno y las pruebas; EV_*)
RI_HOLD = 30         # ticks del valor fijo de la barra en este saque (sorteado)
RI_LAT_INSIDE = 31   # lateral pendiente: ticks seguidos con la pelota ya adentro de la cancha (metida sin patear)
RI_LAT_THROWER = 32  # el que pateó el lateral (para saber si después la tocó otro antes de entrar)
RI_LAT_TOUCHED = 33  # un compañero del que sacó la tocó antes de entrar: el lateral ya cuenta como sacado
RI_SIZE = 34
# eventos del tick
EV_NONE, EV_GOAL_RED, EV_GOAL_BLUE, EV_OUT, EV_PLACE, EV_PIECE_KICK, EV_FOUL, EV_OFFSIDE, EV_KICKOFF, EV_TIMEOUT = range(10)
# ---------------------------------------------------------------------------- estado real
RF_SPOT_X = 0
RF_SPOT_Y = 1
RF_CROSS_X = 2       # x donde el centro de la pelota cruzó y = ±1200 por última vez
RF_GY = 3            # comba vigente
RF_GX = 4
RF_SIZE = 5

_TEAM_GROUP = np.array([RED, BLUE], dtype=np.int64)
_TEAM_FLAG = _TEAM_GROUP
BIG = 1e9


@njit(cache=True)
def physics_tick(pos, vel, mask, group, radius, inv, bcoef, damping, kick_cancel, act, move_unit, fp, P,
                 p_acc, p_kacc, p_kdamp, kstr, p_kback, grav,
                 v_pos, v_bcoef, v_group, v_mask,
                 s_p0, s_p1, s_curved, s_center, s_radius, s_t0, s_t1, s_bias, s_bcoef, s_group, s_mask,
                 p_normal, p_dist, p_bcoef, p_group, p_mask, g_p0, g_p1, g_team, ri, kicked, kvel):
    """Un tick de física de un partido (copia de `sim.physics.step_batch`). kvel[p] = velocidad del jugador al patear.
    Devuelve el gol del tick (+1 rojo, -1 azul)."""
    K = pos.shape[0]
    for p in range(P):
        k = fp + p
        a = act[p]
        kick_pressed = a >= 9
        if not kick_pressed:
            kick_cancel[p] = False
        is_kicking = kick_pressed and not kick_cancel[p]
        kicked[p] = False
        dx = pos[0, 0] - pos[k, 0]
        dy = pos[0, 1] - pos[k, 1]
        d = math.sqrt(dx * dx + dy * dy)
        if d - radius[k] - radius[0] < KICK_REACH:
            if is_kicking and d > 0.0:
                kvel[p, 0] = vel[k, 0]
                kvel[p, 1] = vel[k, 1]
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
    _collide_env(pos, vel, mask, radius, inv, bcoef, group,
                 v_pos, v_bcoef, v_group, v_mask,
                 s_p0, s_p1, s_curved, s_center, s_radius, s_t0, s_t1, s_bias, s_bcoef, s_group, s_mask,
                 p_normal, p_dist, p_bcoef, p_group, p_mask)
    goal = 0
    if ri[RI_KO] != 0:
        if vel[0, 0] != 0.0 or vel[0, 1] != 0.0:
            ri[RI_KO] = 0
            for p in range(P):
                mask[fp + p] = PLAYER_MASK
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


# ============================================================================ árbitro: ayudas
@njit(cache=True)
def _team_of(p, n_red):
    return 0 if p < n_red else 1


@njit(cache=True)
def _edge(pos, k):
    dx = pos[0, 0] - pos[k, 0]
    dy = pos[0, 1] - pos[k, 1]
    return math.sqrt(dx * dx + dy * dy) - C.PLAYER_R - C.BALL_R


@njit(cache=True)
def _restore_players(inv, group, fp, P, n_red):
    for p in range(P):
        inv[fp + p] = C.PLAYER_INV
        group[fp + p] = _TEAM_GROUP[_team_of(p, n_red)]


@njit(cache=True)
def _discs_off(pos, radius, mask, home):
    """Barrera (disco 1) y disco del punto (disco 2) a su lugar de reposo, fuera de juego."""
    for s in range(2):
        pos[1 + s, 0] = home[s, 0]
        pos[1 + s, 1] = home[s, 1]
    radius[1] = 0.0
    radius[2] = 0.0


@njit(cache=True)
def _place(pos, vel, inv, mask, group, radius, grav, ri, rf, kind, team, x, y, fp, P, n_red, home, hold, hold_cdf):
    """Colocación de la pelota parada: pelota fija en el punto, barreras, jugadores sin masa, rivales fantasma."""
    pos[0, 0] = x
    pos[0, 1] = y
    vel[0, 0] = 0.0
    vel[0, 1] = 0.0
    grav[0] = 0.0
    grav[1] = 0.0
    inv[0] = hold[kind]
    u = np.random.random()
    h = 0
    while h < hold_cdf.shape[1] - 1 and hold_cdf[kind, h] < u:
        h += 1
    ri[RI_HOLD] = h
    ri[RI_MODE] = PIECE
    ri[RI_KIND] = kind
    ri[RI_TEAM] = team
    ri[RI_TICKS] = 0
    ri[RI_PS_CONTACT] = 0
    ri[RI_PS_T] = -1
    ri[RI_PS_RESET] = 0
    ri[RI_CURVE] = 0
    ri[RI_OFF_MASK] = 0
    ri[RI_AFTER_T] = -1
    rf[RF_SPOT_X] = x
    rf[RF_SPOT_Y] = y
    # disco del punto (radio 17,5, choca con los dos equipos)
    pos[2, 0] = x
    pos[2, 1] = y
    radius[2] = 17.5
    ri[RI_SPOT_ON] = 1
    # barrera contra el rival (disco 1): aparece en el tick siguiente (la colocación es de este tick)
    ri[RI_BARRIER_ON] = 0
    if kind == C.CORNER or kind == C.GOAL_KICK or kind == C.FREE_KICK or kind == C.PENALTY:
        ri[RI_GHOSTS] = 1
        for p in range(P):
            inv[fp + p] = C.GHOST_INV
    if kind == C.GOAL_KICK:
        ri[RI_GK_GHOST] = 1 if x < 0 else 2
        for p in range(P):
            if _team_of(p, n_red) != team:
                group[fp + p] = C0 if x < 0 else C1
    ri[RI_LAT_INSIDE] = 0
    ri[RI_EVENT] = EV_PLACE


@njit(cache=True)
def _gk_clear_box(pos, vel, radius, fp, P, n_red, team, spot_x):
    """Saque de arco: los rivales que están dentro del área grande van a |x| = 1035 (justo afuera de la línea), con su
    misma y (la sala lo hace un tick después de la colocación; 328 de 328 en el banco, 2026-10-09)."""
    side = 1.0 if spot_x > 0 else -1.0
    for p in range(P):
        q = fp + p
        if _team_of(p, n_red) == team or radius[q] <= 0.0:
            continue
        if pos[q, 0] * side > C.BOX_X and abs(pos[q, 1]) < C.BOX_Y:
            pos[q, 0] = side * C.GK_CLEAR_X
            vel[q, 0] = 0.0
            vel[q, 1] = 0.0


@njit(cache=True)
def _barrier_on(pos, radius, mask, ri, rf):
    kind = ri[RI_KIND]
    x = rf[RF_SPOT_X]
    y = rf[RF_SPOT_Y]
    if kind == C.GOAL_KICK:
        x = x - C.GOAL_KICK_BARRIER_DX if x > 0 else x + C.GOAL_KICK_BARRIER_DX
    pos[1, 0] = x
    pos[1, 1] = y
    r = 373.0
    if kind == C.CORNER:
        r = 750.0
    radius[1] = r
    # en el penal la barrera deja afuera del área a los compañeros del que patea; en los demás, al rival
    mask[1] = _TEAM_FLAG[ri[RI_TEAM]] if kind == C.PENALTY else _TEAM_FLAG[1 - ri[RI_TEAM]]
    ri[RI_BARRIER_ON] = 1


@njit(cache=True)
def _freeze(pos, vel, grav, radius, ri, rf, kind, team, x, y):
    """Primera colocación de falta u offside: la pelota queda quieta en el punto (conserva su masa) y el disco del punto
    ya se pone [S4: aparece 1 tick después del congelamiento y dura hasta 120 después del tiro libre]."""
    pos[0, 0] = x
    pos[0, 1] = y
    pos[2, 0] = x
    pos[2, 1] = y
    radius[2] = 17.5
    ri[RI_SPOT_ON] = 1
    vel[0, 0] = 0.0
    vel[0, 1] = 0.0
    grav[0] = 0.0
    grav[1] = 0.0
    ri[RI_MODE] = FREEZE
    ri[RI_KIND] = kind
    ri[RI_TEAM] = team
    ri[RI_TICKS] = 0
    ri[RI_OFF_MASK] = 0
    ri[RI_CURVE] = 0
    rf[RF_SPOT_X] = x
    rf[RF_SPOT_Y] = y


@njit(cache=True)
def _offside_marks(pos, ri, kicker, team, fp, P, n_red):
    """Compañeros del que patea por delante del anteúltimo rival, en campo rival y por delante de la pelota (bits)."""
    sgn = 1.0 if team == 0 else -1.0
    best = -BIG
    second = -BIG
    for p in range(P):
        if _team_of(p, n_red) == team:
            continue
        x = sgn * pos[fp + p, 0]
        if x > best:
            second = best
            best = x
        elif x > second:
            second = x
    bx = sgn * pos[0, 0]
    marks = 0
    for p in range(P):
        if p == kicker or _team_of(p, n_red) != team:
            continue
        x = sgn * pos[fp + p, 0]
        if x > second and x > 0.0 and x > bx:
            marks |= 1 << p
    return marks


@njit(cache=True)
def _in_box(x, y, attack_team):
    """La pelota está en el área que defiende el rival de `attack_team`."""
    if abs(y) > C.BOX_HALF_H:
        return False
    return x >= C.BOX_X if attack_team == 0 else x <= -C.BOX_X


# ============================================================================ árbitro
@njit(cache=True)
def referee(pos, vel, mask, group, radius, inv, grav, ri, rf, fs, fl, kicked, kvel, goal, fp, P, n_red,
            home, spawn, bar, hold, hold_cdf, ramp):
    """Script de la sala al final del tick. Devuelve 1 si el partido terminó."""
    ri[RI_EVENT] = EV_NONE
    mode = ri[RI_MODE]
    ri[RI_TICKS] += 1
    # ------------------------------------------------------------- gol
    if mode != GOAL and goal != 0:
        if goal > 0:
            ri[RI_SCORE0] += 1
            ri[RI_EVENT] = EV_GOAL_RED
        else:
            ri[RI_SCORE1] += 1
            ri[RI_EVENT] = EV_GOAL_BLUE
        ri[RI_MODE] = GOAL
        ri[RI_TICKS] = 0
        ri[RI_KO_TEAM] = 1 if goal > 0 else 0
        ri[RI_PS_T] = -1
        ri[RI_PS_CONTACT] = 0
        ri[RI_CURVE] = 0
        grav[0] = 0.0
        grav[1] = 0.0
        return 0
    if mode == GOAL:
        if ri[RI_TICKS] >= C.GOAL_RESET_TICKS:
            # reposicionamiento de HaxBall y arqueros a (±1760, 0)
            pos[0, 0] = 0.0
            pos[0, 1] = 0.0
            vel[0, 0] = 0.0
            vel[0, 1] = 0.0
            inv[0] = C.BASE_BALL_INV
            for p in range(P):
                pos[fp + p, 0] = spawn[p, 0]
                pos[fp + p, 1] = spawn[p, 1]
                vel[fp + p, 0] = 0.0
                vel[fp + p, 1] = 0.0
            pos[fp, 0] = -C.GK_KICKOFF[0]
            pos[fp, 1] = C.GK_KICKOFF[1]
            pos[fp + n_red, 0] = C.GK_KICKOFF[0]
            pos[fp + n_red, 1] = C.GK_KICKOFF[1]
            _restore_players(inv, group, fp, P, n_red)
            ko = REDKO if ri[RI_KO_TEAM] == 0 else BLUEKO
            for p in range(P):
                mask[fp + p] = PLAYER_MASK | ko
            _discs_off(pos, radius, mask, home)
            ri[RI_KO] = 1
            ri[RI_MODE] = KICKOFF
            ri[RI_TICKS] = 0
            ri[RI_LAST] = -1
            ri[RI_OFF_MASK] = 0
            ri[RI_GHOSTS] = 0
            ri[RI_GK_GHOST] = 0
            ri[RI_SPOT_ON] = 0
            ri[RI_BARRIER_ON] = 0
            ri[RI_EVENT] = EV_KICKOFF
        return 0
    if mode == KICKOFF:
        if ri[RI_KO] != 0:
            return 0
        ri[RI_MODE] = OPEN
        ri[RI_TICKS] = 0
        mode = OPEN
    # ------------------------------------------------------------- reloj
    ri[RI_CLOCK] += 1
    if ri[RI_LEN] > 0 and ri[RI_CLOCK] >= ri[RI_LEN] and mode == OPEN:
        ri[RI_DONE] = 1
        return 1
    # ------------------------------------------------------------- toques
    toucher = -1
    kicker = -1
    best = BIG
    for p in range(P):
        if kicked[p]:
            kicker = p
    for p in range(P):
        e = _edge(pos, fp + p)
        if e <= C.TOUCH_EPS and e < best:
            best = e
            toucher = p
    if kicker >= 0:
        toucher = kicker
    # ------------------------------------------------------------- restauraciones después de la patada de un saque
    if ri[RI_AFTER_T] >= 0:
        ri[RI_AFTER_T] += 1
        t = ri[RI_AFTER_T]
        ak = ri[RI_AFTER_KIND]
        if t == C.GHOST_RESTORE_AFTER_KICK and (ri[RI_GHOSTS] != 0 or ri[RI_GK_GHOST] != 0):
            _restore_players(inv, group, fp, P, n_red)
            ri[RI_GHOSTS] = 0
            ri[RI_GK_GHOST] = 0
        # el lateral termina cuando la pelota vuelve a |y| < 1210: ahí se van la barrera y la masa de 1,8 [S4: |y| al
        # quitarse p10 1195, p90 1208 en 224 laterales]; en los demás saques, al tick siguiente de la patada
        if ak == C.LATERAL:
            done = abs(pos[0, 1]) < C.LATERAL_OUT_Y or t >= C.BAD_LATERAL_TICKS
        else:
            done = t >= 1
        if done:
            if ri[RI_BARRIER_ON] == 1:
                pos[1, 0] = home[0, 0]
                pos[1, 1] = home[0, 1]
                radius[1] = 0.0
                ri[RI_BARRIER_ON] = 0
            if ri[RI_PS_T] < 0:
                inv[0] = C.BASE_BALL_INV
            if t >= 1:
                ri[RI_AFTER_T] = -1
    # ------------------------------------------------------------- discos con tiempo fijo (punto y barrera del arco)
    if ri[RI_SPOT_ON] != 0 and (mode == PIECE or ri[RI_AFTER_T] >= 0):
        spot_ticks = 120
        if ri[RI_KIND] == C.LATERAL or ri[RI_AFTER_KIND] == C.LATERAL and mode != PIECE:
            spot_ticks = 30
        if ri[RI_KIND] == C.PENALTY and mode == PIECE:
            spot_ticks = C.PENALTY_DISC_TICKS
        if mode != PIECE or ri[RI_TICKS] >= spot_ticks:
            pos[2, 0] = home[1, 0]
            pos[2, 1] = home[1, 1]
            radius[2] = 0.0
            ri[RI_SPOT_ON] = 0
    if mode == PIECE and ri[RI_BARRIER_ON] == 1 and (
            ri[RI_KIND] == C.GOAL_KICK and ri[RI_TICKS] >= C.GOAL_KICK_BARRIER_TICKS
            or ri[RI_KIND] == C.PENALTY and ri[RI_TICKS] >= C.PENALTY_DISC_TICKS):
        pos[1, 0] = home[0, 0]
        pos[1, 1] = home[0, 1]
        radius[1] = 0.0
        ri[RI_BARRIER_ON] = 2  # ya se usó
    # ------------------------------------------------------------- pelota parada: espera y patada
    if mode == PIECE:
        kind = ri[RI_KIND]
        team = ri[RI_TEAM]
        if ri[RI_TICKS] >= 1 and ri[RI_BARRIER_ON] == 0:
            _barrier_on(pos, radius, mask, ri, rf)
            if kind == C.GOAL_KICK:
                _gk_clear_box(pos, vel, radius, fp, P, n_red, team, rf[RF_SPOT_X])
        # masa de la pelota: barra (valor fijo y onda de 150 ticks) o fija
        if kind == C.CORNER or kind == C.FREE_KICK or kind == C.GOAL_KICK:
            t = ri[RI_TICKS]
            if t < ri[RI_HOLD]:
                inv[0] = hold[kind]
            else:
                inv[0] = bar[kind, (t - ri[RI_HOLD]) % C.BAR_PERIOD]
        if kind == C.LATERAL:
            ri[RI_LAT_INSIDE] = ri[RI_LAT_INSIDE] + 1 if abs(pos[0, 1]) < C.LATERAL_OUT_Y else 0
        if kind == C.LATERAL and (ri[RI_LAT_INSIDE] > C.LATERAL_INSIDE_TICKS or (
                kicker >= 0 and _team_of(kicker, n_red) == team and abs(pos[0, 1] - vel[0, 1]) < C.LATERAL_OUT_Y)):
            # lateral pateado con la pelota ya adentro (antes de la patada), o metida sin patear y dejada adentro: mal
            # sacado, para el rival en el punto original
            ri[RI_LAT_INSIDE] = 0
            ri[RI_MODE] = OUT_PENDING
            ri[RI_TICKS] = 0
            ri[RI_KIND] = C.LATERAL
            ri[RI_TEAM] = 1 - team
            ri[RI_EVENT] = EV_OUT
            return 0
        if kicker >= 0 and _team_of(kicker, n_red) == team:
            ri[RI_MODE] = OPEN
            ri[RI_TICKS] = 0
            ri[RI_AFTER_KIND] = kind
            ri[RI_AFTER_T] = 0
            ri[RI_LAST] = kicker
            ri[RI_EXEMPT] = 1 if (kind == C.LATERAL or kind == C.CORNER or kind == C.GOAL_KICK) else 0
            ri[RI_LAT_ARMED] = 0 if kind == C.LATERAL else 1
            ri[RI_LAT_THROWER] = kicker
            ri[RI_LAT_TOUCHED] = 0
            ri[RI_EVENT] = EV_PIECE_KICK
            # comba de la pelota parada (ajustada al banco, 2026-10-09)
            vx = kvel[kicker, 0]
            vy = kvel[kicker, 1]
            if kind == C.CORNER:
                sx = 1.0 if rf[RF_SPOT_X] > 0 else -1.0
                sy = 1.0 if rf[RF_SPOT_Y] > 0 else -1.0
                rf[RF_GX] = -C.CORNER_CURVE_KX * vx + C.CORNER_CURVE_KY * sx * sy * vy - C.CORNER_CURVE_X0 * sx
                rf[RF_GY] = -C.CORNER_CURVE_Y * sy
                ri[RI_CURVE] = C.PS_CURVE_TICKS
            elif kind == C.FREE_KICK or kind == C.GOAL_KICK:
                rf[RF_GX] = 0.0
                rf[RF_GY] = -(C.FK_CURVE_K if kind == C.FREE_KICK else C.GK_CURVE_K) * vy
                ri[RI_CURVE] = C.PS_CURVE_TICKS
            _offside_kick(pos, ri, kicker, fp, P, n_red)
            for q in range(P):
                fs[q] = -1
                fl[q] = -1
            return 0
        elif ri[RI_TICKS] >= C.PIECE_TIMEOUT:
            # [supuesto] se entrega al rival en el mismo punto
            _place(pos, vel, inv, mask, group, radius, grav, ri, rf, kind, 1 - team, rf[RF_SPOT_X], rf[RF_SPOT_Y],
                   fp, P, n_red, home, hold, hold_cdf)
            ri[RI_EVENT] = EV_TIMEOUT
            return 0
        else:
            return 0
    # ------------------------------------------------------------- congelado de falta u offside
    if mode == FREEZE:
        if ri[RI_TICKS] >= C.FREEZE_TICKS:
            kind = ri[RI_KIND]
            x = rf[RF_SPOT_X]
            y = rf[RF_SPOT_Y]
            if kind == C.PENALTY:
                x = C.PENALTY_X if x > 0 else -C.PENALTY_X
                y = 0.0
            _place(pos, vel, inv, mask, group, radius, grav, ri, rf, kind, ri[RI_TEAM], x, y, fp, P, n_red, home,
                   hold, hold_cdf)
        return 0
    # ------------------------------------------------------------- salida pendiente
    if mode == OUT_PENDING:
        delay = C.LATERAL_DELAY if ri[RI_KIND] == C.LATERAL else C.END_LINE_DELAY
        if ri[RI_TICKS] >= delay:
            _place(pos, vel, inv, mask, group, radius, grav, ri, rf, ri[RI_KIND], ri[RI_TEAM], rf[RF_SPOT_X],
                   rf[RF_SPOT_Y], fp, P, n_red, home, hold, hold_cdf)
        return 0
    # ================================================================ juego abierto
    # ------------------------------------------------------------- offside: cobro al tocar un marcado
    if ri[RI_OFFSIDE_ON] != 0 and toucher >= 0 and (ri[RI_OFF_MASK] >> toucher) & 1:
        att = _team_of(toucher, n_red)
        dx = C.OFFSIDE_SPOT_DX if att == 0 else -C.OFFSIDE_SPOT_DX
        _freeze(pos, vel, grav, radius, ri, rf, C.FREE_KICK, 1 - att, pos[fp + toucher, 0] + dx, pos[fp + toucher, 1])
        ri[RI_EVENT] = EV_OFFSIDE
        for q in range(P):
            fs[q] = -1
            fl[q] = -1
        return 0
    if kicker >= 0:
        _offside_kick(pos, ri, kicker, fp, P, n_red)
    # ------------------------------------------------------------- posesión
    prev = ri[RI_LAST]
    if toucher >= 0 and toucher != prev:
        ri[RI_LAST] = toucher
        ri[RI_PS_CONTACT] = 0
        for r in range(P):
            fs[r] = -1
            fl[r] = -1
        if ri[RI_PS_T] >= 0:
            ri[RI_PS_T] = -1
            if ri[RI_PS_RESET] == 0:
                inv[0] = C.BASE_BALL_INV
    poss = ri[RI_LAST]
    # ------------------------------------------------------------- comba vigente
    if ri[RI_CURVE] > 0:
        # cualquier toque (también una patada) corta la comba vigente; la patada cargada pone la suya más abajo [S4]
        if toucher >= 0:
            ri[RI_CURVE] = 0
        else:
            ri[RI_CURVE] -= 1
    if ri[RI_CURVE] > 0:
        grav[0] = rf[RF_GX]
        grav[1] = rf[RF_GY]
        # entera los primeros CURVE_HOLD_TICKS ticks, después se atenúa
        if C.PS_CURVE_TICKS - ri[RI_CURVE] >= C.CURVE_HOLD_TICKS:
            rf[RF_GX] *= C.PS_CURVE_DECAY
            rf[RF_GY] *= C.PS_CURVE_DECAY
    else:
        grav[0] = 0.0
        grav[1] = 0.0
    # ------------------------------------------------------------- powershot
    if ri[RI_PS_RESET] > 0:
        ri[RI_PS_RESET] -= 1
        if ri[RI_PS_RESET] == 0:
            inv[0] = C.BASE_BALL_INV
    if ri[RI_PS_ON] != 0 and poss >= 0:
        if kicker >= 0 and ri[RI_PS_T] >= 0:
            # patada cargada: comba en y y masa a 1,65 en el tick siguiente
            rf[RF_GX] = 0.0
            rf[RF_GY] = -C.PS_CURVE_K * kvel[kicker, 1]
            ri[RI_CURVE] = C.PS_CURVE_TICKS
            ri[RI_PS_T] = -1
            ri[RI_PS_CONTACT] = 0
            ri[RI_PS_RESET] = C.PS_RESET_AFTER_KICK
        elif kicker >= 0:
            ri[RI_PS_CONTACT] = 0
        elif _edge(pos, fp + poss) <= C.PS_CONTACT_EPS:
            ri[RI_PS_CONTACT] += 1
            if ri[RI_PS_T] >= 0:
                ri[RI_PS_T] += 1
            elif ri[RI_PS_CONTACT] >= C.PS_DELAY:
                ri[RI_PS_T] = 0
            if ri[RI_PS_T] >= 0:
                inv[0] = ramp[min(ri[RI_PS_T], ramp.shape[0] - 1)]
        else:
            ri[RI_PS_CONTACT] = 0
            if ri[RI_PS_T] >= 0:
                ri[RI_PS_T] = -1
                inv[0] = C.BASE_BALL_INV
    # ------------------------------------------------------------- falta
    if ri[RI_FOULS_ON] != 0 and poss >= 0:
        if _edge(pos, fp + poss) <= C.FOUL_BALL_DIST:
            tp = _team_of(poss, n_red)
            now = ri[RI_CLOCK]
            for r in range(P):
                if _team_of(r, n_red) == tp:
                    continue
                dx = pos[fp + r, 0] - pos[fp + poss, 0]
                dy = pos[fp + r, 1] - pos[fp + poss, 1]
                if math.sqrt(dx * dx + dy * dy) <= 2 * C.PLAYER_R + C.FOUL_CONTACT_TOL:
                    if fs[r] < 0 or now - fl[r] > C.FOUL_GAP + 1:
                        fs[r] = now
                    fl[r] = now
                    if now - fs[r] >= C.FOUL_TICKS:
                        kind = C.PENALTY if _in_box(pos[0, 0], pos[0, 1], tp) else C.FREE_KICK
                        x = pos[fp + poss, 0]
                        y = pos[fp + poss, 1]
                        if kind == C.PENALTY:
                            x = C.PENALTY_X if tp == 0 else -C.PENALTY_X
                            y = 0.0
                        _freeze(pos, vel, grav, radius, ri, rf, kind, tp, x, y)
                        ri[RI_EVENT] = EV_FOUL
                        for q in range(P):
                            fs[q] = -1
                            fl[q] = -1
                        return 0
        else:
            for r in range(P):
                fs[r] = -1
                fl[r] = -1
    # ------------------------------------------------------------- salida de la pelota
    bx = pos[0, 0]
    by = pos[0, 1]
    inside_y = abs(by) <= C.LINE_H
    if ri[RI_PREV_IN_Y] != 0 and not inside_y:
        rf[RF_CROSS_X] = bx
    ri[RI_PREV_IN_Y] = 1 if inside_y else 0
    if ri[RI_LAT_ARMED] == 0:
        thrower = ri[RI_LAT_THROWER]
        if abs(by) < C.LATERAL_OUT_Y:
            # entró a la cancha: el lateral ya se sacó
            ri[RI_LAT_ARMED] = 1
        elif poss >= 0 and thrower >= 0 and poss != thrower and _team_of(poss, n_red) != _team_of(thrower, n_red):
            # la tocó un rival con la pelota afuera: salida contra el rival, el lateral vuelve al que sacaba, en el punto
            ri[RI_LAT_ARMED] = 1
            ri[RI_MODE] = OUT_PENDING
            ri[RI_TICKS] = 0
            ri[RI_KIND] = C.LATERAL
            ri[RI_TEAM] = _team_of(thrower, n_red)
            ri[RI_EVENT] = EV_OUT
            return 0
        elif thrower >= 0 and ri[RI_LAT_TOUCHED] == 0 and (
                (toucher >= 0 and _team_of(toucher, n_red) == _team_of(thrower, n_red)
                 and (toucher != thrower or ri[RI_TICKS] >= 3))
                or (poss >= 0 and poss != thrower and _team_of(poss, n_red) == _team_of(thrower, n_red))):
            # la volvió a tocar el que sacó, o un compañero: el lateral ya cuenta como sacado (puede entrar mucho
            # después; en el banco, 370 + 35 casos entraron hasta 418 ticks después de la patada)
            ri[RI_LAT_TOUCHED] = 1
            return 0
        elif ri[RI_TICKS] >= (C.BAD_LATERAL_TICKS_TOUCHED if ri[RI_LAT_TOUCHED] else C.BAD_LATERAL_TICKS) and poss >= 0:
            # lateral mal sacado (no entró): se repite para el rival del último que la tocó, en el punto original,
            # después de la demora de colocación (la sala: p50 51 ticks desde la patada)
            ri[RI_LAT_ARMED] = 1
            ri[RI_MODE] = OUT_PENDING
            ri[RI_TICKS] = 0
            ri[RI_KIND] = C.LATERAL
            ri[RI_TEAM] = 1 - _team_of(poss, n_red)
            ri[RI_EVENT] = EV_OUT
            return 0
        else:
            return 0
    out_side = abs(by) > C.LATERAL_OUT_Y
    out_end = abs(bx) > C.LINE_W + C.OUT_MARGIN and abs(by) > C.GOAL_HALF_H
    if poss >= 0 and (out_side or out_end):
        last_team = _team_of(poss, n_red)
        ri[RI_MODE] = OUT_PENDING
        ri[RI_TICKS] = 0
        ri[RI_TEAM] = 1 - last_team
        ri[RI_PS_T] = -1
        ri[RI_PS_CONTACT] = 0
        ri[RI_OFF_MASK] = 0
        ri[RI_EVENT] = EV_OUT
        for q in range(P):
            fs[q] = -1
            fl[q] = -1
        if out_side and abs(bx) <= C.LINE_W:
            ri[RI_KIND] = C.LATERAL
            rf[RF_CROSS_X] = bx
            rf[RF_SPOT_X] = bx
            rf[RF_SPOT_Y] = C.LATERAL_Y if by > 0 else -C.LATERAL_Y
        else:
            # arco de x > 0 lo defiende el azul (1)
            defender = 1 if bx > 0 else 0
            if last_team == defender:
                ri[RI_KIND] = C.CORNER
                rf[RF_SPOT_X] = C.CORNER_SPOT[0] if bx > 0 else -C.CORNER_SPOT[0]
                rf[RF_SPOT_Y] = C.CORNER_SPOT[1] if by > 0 else -C.CORNER_SPOT[1]
            else:
                ri[RI_KIND] = C.GOAL_KICK
                rf[RF_SPOT_X] = C.GOAL_KICK_SPOT[0] if bx > 0 else -C.GOAL_KICK_SPOT[0]
                rf[RF_SPOT_Y] = 0.0
    return 0


@njit(cache=True)
def _offside_kick(pos, ri, kicker, fp, P, n_red):
    """Una patada corta las marcas anteriores y, si no es la ejecución de un saque exento, marca a los adelantados."""
    exempt = ri[RI_EXEMPT] != 0
    ri[RI_EXEMPT] = 0
    ri[RI_OFF_MASK] = 0
    if not exempt:
        ri[RI_OFF_MASK] = _offside_marks(pos, ri, kicker, _team_of(kicker, n_red), fp, P, n_red)


# ============================================================================ paso de N partidos
@njit(cache=True, parallel=True)
def step_envs(pos, vel, mask, group, radius, inv, grav, kick_cancel, ri, rf, fs, fl, actions, frames,
              move_unit, fp, n_red, bcoef, damping, p_acc, p_kacc, p_kdamp, kstr, p_kback,
              v_pos, v_bcoef, v_group, v_mask,
              s_p0, s_p1, s_curved, s_center, s_radius, s_t0, s_t1, s_bias, s_bcoef, s_group, s_mask,
              p_normal, p_dist, p_bcoef, p_group, p_mask, g_p0, g_p1, g_team,
              home, spawn, bar, hold, hold_cdf, ramp, events):
    """Avanza `frames` ticks cada partido con las mismas acciones. events[n, f] = evento de cada tick."""
    N = pos.shape[0]
    P = actions.shape[1]
    for n in prange(N):
        kicked = np.zeros(P, np.bool_)
        kvel = np.zeros((P, 2))
        for f in range(frames):
            if ri[n, RI_DONE] != 0:
                events[n, f] = EV_NONE
                continue
            goal = physics_tick(pos[n], vel[n], mask[n], group[n], radius[n], inv[n], bcoef, damping,
                                kick_cancel[n], actions[n], move_unit, fp, P, p_acc, p_kacc, p_kdamp, kstr, p_kback,
                                grav[n], v_pos, v_bcoef, v_group, v_mask,
                                s_p0, s_p1, s_curved, s_center, s_radius, s_t0, s_t1, s_bias, s_bcoef, s_group,
                                s_mask, p_normal, p_dist, p_bcoef, p_group, p_mask, g_p0, g_p1, g_team, ri[n],
                                kicked, kvel)
            referee(pos[n], vel[n], mask[n], group[n], radius[n], inv[n], grav[n], ri[n], rf[n], fs[n], fl[n],
                    kicked, kvel, goal, fp, P, n_red, home, spawn, bar, hold, hold_cdf, ramp)
            events[n, f] = ri[n, RI_EVENT]
