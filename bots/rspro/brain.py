"""Decisión de RS-Pro por partido y equipo (numba). Todo en el marco del equipo: ataca hacia +x.

Flujo por decisión (`decide`):
  1. percepción: estado con retraso de reacción, ruido y anticipación (propios en tiempo real)
  2. trayectoria de la pelota e intercepción de cada jugador (cinemática exacta, `geom.py`)
  3. fase con histéresis: posesión propia / rival / disputa, saques y saque inicial
  4. roles por fase y asignación de costo mínimo con costo de cambio; los compañeros no controlados
     (aprendices o humanos) ocupan el rol más cercano y los bots llenan el resto
  5. objetivo de cada rol: candidatos puntuados (línea de pase, espacio, progreso, separación)
  6. portador: tiro / pase (al pie, adelantado, en profundidad) / pared / conducción / protección /
     despeje, por valor esperado con elección estocástica (temperatura del nivel)
  7. ejecución: llegada detrás de la pelota, frenado por velocidad deseada, patada sólo alineada
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit

from env.rs4z.numba_cache import guard

guard(__file__, ["bots/rspro/geom.py"])

from .geom import (A_KICK, A_P, GOAL_HH, GOAL_X, HORIZON, KICK_SPEED, LINE_H, LINE_W, Q_B, Q_P, R_B, R_P,
                   REACH, ball_travel_ticks, nrand, predict_ball, seg_dist, ttr, urand)

# ------------------------------------------------------------------ parámetros de nivel (lvl[...])
L_DELAY = 0        # retraso de reacción en ticks
L_NOISE = 1        # σ de ruido de posición percibida (px)
L_AIM_TOL = 2      # tolerancia angular para patear (grados)
L_AIM_NOISE = 3    # σ de error de dirección (grados)
L_OPTIONS = 4      # conjunto de decisiones 0..5
L_TEMP = 5         # temperatura de elección entre opciones
L_HYST = 6         # costo de cambiar de rol (px)
L_REACT = 7        # reacción que supone en los rivales al evaluar pases/tiros (ticks)
L_COMMIT = 8       # decisiones que sostiene un plan de portador
L_SPEED = 9        # factor de velocidad deseada (<1: jugador lento)
NLV = 10
# ------------------------------------------------------------------ estilo (sty[...]) en [0, 1]
S_PRESS, S_DIRECT, S_WIDTH, S_RISK, S_TEMPO, S_DEPTH = 0, 1, 2, 3, 4, 5
NST = 6
# ------------------------------------------------------------------ memoria entera por equipo
M_PHASE, M_PHASE_AGE = 0, 1
M_ROLE = 2         # 8 slots
M_PLAN = 10        # 8 slots: tipo de plan del portador
M_PLAN_UNTIL = 18  # 8 slots
M_TAKER = 26
M_WAIT = 27
M_TICK = 28
M_PARTNER = 29     # 8 slots
MI = 37
# ------------------------------------------------------------------ memoria real por equipo
F_PLAN_T = 0       # 8×2 objetivo del plan
F_PLAN_U = 16      # 8×2 dirección de patada del plan
F_SUP = 32         # 8×2 último punto de apoyo elegido (histéresis)
F_AIM = 48         # 8 error de puntería del plan (rad)
MF = 56

# fases
PH_CONTEST, PH_OWN, PH_OPP = 0, 1, 2
# roles
R_NONE, R_CARRIER, R_SUPPORT, R_WIDE, R_DEPTH, R_SAFETY, R_PRESS, R_COVER, R_MARK, R_LAST, R_TAKER = range(11)
# planes del portador
P_NONE, P_SHOT, P_PASS, P_DRIBBLE, P_SHIELD, P_CLEAR, P_ONETWO, P_RETURN, P_SELF = range(9)

MOVE = np.array([[0.0, 0.0], [0.0, -1.0], [1.0, -1.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [-1.0, 1.0],
                 [-1.0, 0.0], [-1.0, -1.0]])
MOVE_U = MOVE / np.maximum(np.sqrt((MOVE ** 2).sum(axis=1)), 1e-12)[:, None]
VMAX = 3.0          # desplazamiento por tick a máxima velocidad


@njit(cache=True, inline="always")
def _clip(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


@njit(cache=True, inline="always")
def _sigmoid(x):
    if x > 30.0:
        return 1.0
    if x < -30.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


# ================================================================== movimiento y patada
@njit(cache=True)
def move_action(px, py, vx, vy, tx, ty, arrive_speed, speed_factor, kicking):
    """Acción de movimiento 0..8 que mejor lleva la velocidad hacia la deseada (frena al llegar).

    arrive_speed: rapidez deseada al llegar (0 = detenerse en el punto).
    """
    dx = tx - px
    dy = ty - py
    d = math.sqrt(dx * dx + dy * dy)
    vmax = VMAX * speed_factor
    if d < 3.0 and arrive_speed <= 0.0:
        dvx, dvy = 0.0, 0.0
    else:
        # frenado: con la dirección opuesta se pierde ~a + 0,04 v por tick
        brake = math.sqrt(max(0.0, 2.0 * 0.11 * d)) + arrive_speed
        sp = min(vmax, brake)
        dvx = dx / max(d, 1e-9) * sp
        dvy = dy / max(d, 1e-9) * sp
    return vel_action(vx, vy, dvx, dvy, kicking)


@njit(cache=True)
def vel_action(vx, vy, dvx, dvy, kicking):
    """Acción 0..8 cuya velocidad al cabo de la decisión (3 ticks) más se acerca a la deseada."""
    a = A_KICK if kicking else A_P
    best = 0
    best_e = 1e18
    for m in range(9):
        nvx = (vx + a * MOVE_U[m, 0]) * Q_P
        nvy = (vy + a * MOVE_U[m, 1]) * Q_P
        # tres ticks con la misma acción (una decisión)
        for _ in range(2):
            nvx = (nvx + a * MOVE_U[m, 0]) * Q_P
            nvy = (nvy + a * MOVE_U[m, 1]) * Q_P
        e = (nvx - dvx) ** 2 + (nvy - dvy) ** 2
        if e < best_e:
            best_e = e
            best = m
    return best


@njit(cache=True)
def kick_ready(px, py, vx, vy, bx, by, bvx, bvy, ux, uy, tol_cos, move_m):
    """¿Patear ahora? Simula los 3 ticks de la decisión: el primer tick en alcance debe estar alineado."""
    a = A_KICK
    qx, qy, wx, wy = px, py, vx, vy
    cx, cy, cvx, cvy = bx, by, bvx, bvy
    for _ in range(3):
        dx = cx - qx
        dy = cy - qy
        d = math.sqrt(dx * dx + dy * dy)
        if d - R_P - R_B < 4.0:
            if d <= 1e-9:
                return False
            return (dx * ux + dy * uy) / d >= tol_cos
        wx = (wx + a * MOVE_U[move_m, 0])
        wy = (wy + a * MOVE_U[move_m, 1])
        qx += wx
        qy += wy
        wx *= Q_P
        wy *= Q_P
        cx += cvx
        cy += cvy
        cvx *= Q_B
        cvy *= Q_B
    return False


@njit(cache=True)
def approach_point(px, py, bx, by, ux, uy, vx=0.0, vy=0.0):
    """Punto de llegada para patear o conducir en dirección u.

    Navegación de órbita: si el jugador no está detrás de la pelota (respecto de u), avanza por un
    círculo de radio seguro alrededor de la pelota hacia el lado de atrás, por el camino angular más
    corto, sin tocarla; ya alineado (< ~26°), entra directo al punto de contacto.
    """
    contact = R_P + R_B - 1.0
    dx = px - bx
    dy = py - by
    d = math.sqrt(dx * dx + dy * dy)
    theta_p = math.atan2(dy, dx)
    theta_b = math.atan2(-uy, -ux)
    delta = theta_p - theta_b
    while delta > math.pi:
        delta -= 2.0 * math.pi
    while delta < -math.pi:
        delta += 2.0 * math.pi
    if abs(delta) < 0.45:
        return bx - ux * contact, by - uy * contact, 1.4
    if abs(delta) > math.pi - 0.2:
        # justo delante: elegir el lado hacia el que ya se mueve
        cross = dx * vy - dy * vx
        delta = math.pi - 0.2 if cross >= 0.0 else -(math.pi - 0.2)
    radius = max(contact + 14.0, min(d, contact + 45.0))
    step = min(abs(delta), 0.9)
    theta_t = theta_p - (step if delta > 0.0 else -step)
    return bx + radius * math.cos(theta_t), by + radius * math.sin(theta_t), 1.2


@njit(cache=True, inline="always")
def _wrap(a):
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


ORBIT_R = R_P + R_B + 15.0     # radio de la órbita alrededor de una pelota en movimiento (sin tocarla)


@njit(cache=True)
def orbit_action(px, py, vx, vy, bx, by, bvx, bvy, ux, uy, speed_factor):
    """Rodear una pelota en movimiento hasta su lado de atrás respecto de u, sin tocarla.

    Control de velocidad en el marco de la pelota: tangencial hacia el lado de atrás + corrección al radio
    ORBIT_R, más la velocidad de la pelota. Con puntos de paso alrededor de la posición predicha, el jugador
    que venía detrás de una pelota que se alejaba la chocaba una y otra vez en la dirección en que iba.
    """
    rx = px - bx
    ry = py - by
    d = math.sqrt(rx * rx + ry * ry) + 1e-9
    delta = _wrap(math.atan2(ry, rx) - math.atan2(-uy, -ux))
    if abs(delta) > math.pi - 0.2:
        # justo delante: girar hacia el lado al que ya se mueve respecto de la pelota
        cross = rx * (vy - bvy) - ry * (vx - bvx)
        delta = -(math.pi - 0.2) if cross > 0.0 else math.pi - 0.2
    rux = rx / d
    ruy = ry / d
    if delta > 0.0:
        tx = ruy            # sentido horario (el ángulo decrece)
        ty = -rux
    else:
        tx = -ruy
        ty = rux
    vt = 2.4 * min(1.0, 0.4 + abs(delta))
    vr = _clip(0.2 * (ORBIT_R - d), -2.6, 1.6)
    dvx = bvx + tx * vt + rux * vr
    dvy = bvy + ty * vt + ruy * vr
    vmax = VMAX * speed_factor
    m = math.sqrt(dvx * dvx + dvy * dvy)
    if m > vmax:
        dvx *= vmax / m
        dvy *= vmax / m
    return vel_action(vx, vy, dvx, dvy, False)


@njit(cache=True)
def approach_move(px, py, vx, vy, bx, by, bvx, bvy, cbx, cby, ux, uy, speed_factor, dribble):
    """Acción para quedar detrás de la pelota respecto de u (patear o conducir en dirección u).

    (bx, by): pelota ahora; (cbx, cby): donde estará al llegar. Pelota quieta o lenta: puntos de paso de
    `approach_point` (validados en saques). Pelota en movimiento y jugador fuera de línea: `orbit_action`
    alrededor de la posición actual (S5, 2026-10-04: el portador que venía detrás de una pelota que iba hacia
    su arco la empujaba con el cuerpo durante segundos; 38 de 41 goles de la red contra L5 fueron así).
    dribble: alineado, empujarla en la dirección u en vez de frenar en el punto de contacto.
    """
    if bvx * bvx + bvy * bvy > 0.36:
        delta = _wrap(math.atan2(py - by, px - bx) - math.atan2(-uy, -ux))
        if abs(delta) >= 0.45:
            return orbit_action(px, py, vx, vy, bx, by, bvx, bvy, ux, uy, speed_factor)
    ax, ay, sp = approach_point(px, py, cbx, cby, ux, uy, vx, vy)
    if dribble and sp > 1.3:
        return move_action(px, py, vx, vy, ax + ux * 30.0, ay + uy * 30.0, 2.6, speed_factor, False)
    return move_action(px, py, vx, vy, ax, ay, sp, speed_factor, False)


@njit(cache=True)
def setup_ticks(table, px, py, vx, vy, bx, by, ux, uy):
    """Ticks para quedar en posición de patear o conducir en dirección u: directo si ya está detrás de la
    pelota; si no, llegar al radio de órbita y recorrer el arco hasta el lado de atrás."""
    contact = R_P + R_B - 1.0
    delta = _wrap(math.atan2(py - by, px - bx) - math.atan2(-uy, -ux))
    if abs(delta) < 0.45:
        return ttr(table, px, py, vx, vy, bx - ux * contact, by - uy * contact, 4.0)
    return ttr(table, px, py, vx, vy, bx, by, ORBIT_R) + (abs(delta) - 0.45) * ORBIT_R / 2.2


@njit(cache=True)
def kick_normal(ux, uy, bvx, bvy):
    """Normal de patada n tal que v_pelota + 6,14·n apunta en la dirección deseada u."""
    a = bvx * ux + bvy * uy
    disc = a * a - (bvx * bvx + bvy * bvy) + KICK_SPEED * KICK_SPEED
    if disc <= 0.0:
        return ux, uy
    k = a + math.sqrt(disc)
    nx = (ux * k - bvx) / KICK_SPEED
    ny = (uy * k - bvy) / KICK_SPEED
    m = math.hypot(nx, ny) + 1e-12
    return nx / m, ny / m


@njit(cache=True)
def dribble_dir(table, bx, by, cur_x, cur_y, px_me, py_me, opp_x, opp_y, opp_vx, opp_vy, n_opp, react):
    """Mejor dirección de conducción cerca de la actual (±45°): espacio libre, avance y arco."""
    best_v = -1e18
    bux = cur_x
    buy = cur_y
    base = math.atan2(cur_y, cur_x)
    gdx = GOAL_X - bx
    gdy = -by
    gd = math.hypot(gdx, gdy) + 1e-9
    for j in range(7):
        ang = base + (j - 3) * (math.pi / 12.0)
        ux = math.cos(ang)
        uy = math.sin(ang)
        if ux < -0.1:
            continue
        tx = bx + 120.0 * ux
        ty = by + 120.0 * uy
        if abs(ty) > LINE_H - 60.0 or abs(tx) > LINE_W - 20.0:
            continue
        t_me = ttr(table, px_me, py_me, 0.0, 0.0, tx, ty, 10.0) * 1.25
        free = 99.0
        for o in range(n_opp):
            to = ttr(table, opp_x[o], opp_y[o], opp_vx[o], opp_vy[o], tx, ty, R_P + R_B) + react
            if to - t_me < free:
                free = to - t_me
        v = possession_value(tx, ty) + 0.08 * (ux * gdx + uy * gdy) / gd + 0.02 * _sigmoid((free - 3.0) / 3.0) * 5.0
        v -= 0.01 * abs(j - 3)        # suavidad: preferir no girar
        if v > best_v:
            best_v = v
            bux = ux
            buy = uy
    return bux, buy


@njit(cache=True)
def path_margin(table, bx, by, tx, ty, speed, opp_x, opp_y, opp_vx, opp_vy, opp_n, react):
    """Margen (ticks) del rival que mejor llega a interceptar la pelota en el trayecto B→T.

    Positivo: la pelota pasa antes de que llegue el rival más rápido. Muestra cada 40 px.
    """
    dx = tx - bx
    dy = ty - by
    dist = math.sqrt(dx * dx + dy * dy)
    if dist < 1e-6:
        return 99.0
    steps = int(dist / 40.0) + 1
    best = 99.0
    for k in range(1, steps + 1):
        f = k / steps
        sx = bx + dx * f
        sy = by + dy * f
        tb = ball_travel_ticks(dist * f, speed)
        if tb > 1e8:
            return -99.0
        for o in range(opp_n):
            to = ttr(table, opp_x[o], opp_y[o], opp_vx[o], opp_vy[o], sx, sy, R_P + R_B) + react
            m = to - tb
            if m < best:
                best = m
    return best


@njit(cache=True)
def threat(x, y):
    """Valor heurístico de tener la pelota en (x, y) (marco de equipo): cercanía y ángulo al arco rival."""
    dx = GOAL_X - x
    d = math.sqrt(dx * dx + y * y)
    ang = math.atan2(GOAL_HH, max(dx, 1.0))
    centrality = 1.0 - min(1.0, abs(y) / 700.0) * 0.5
    return math.exp(-d / 520.0) * (0.6 + 0.4 * ang / 1.2) * centrality


# ================================================================== decisión principal
@njit(cache=True)
def decide(n, team, seed, table, pos, vel, active, kick_cancel, ctrl, ri_team, ri_kind, ri_ticks, ri_ko,
           ri_ko_team, spot_x, spot_y, grav_x, grav_y, grav_left, lvl, sty, mem_i, mem_f, delayed_pos,
           delayed_vel, traj, out):
    """Acciones (marco propio) de los jugadores `ctrl` del equipo `team` en el partido n.

    pos/vel: estado real (9 entidades: pelota + 8 jugadores, mundo). delayed_*: estado percibido con
    retraso (mismo layout). traj: arreglo de trabajo (HORIZON+1, 2).
    """
    s = 1.0 if team == 0 else -1.0
    dec = mem_i[M_TICK]
    mem_i[M_TICK] = dec + 1
    delay = lvl[L_DELAY]
    noise = lvl[L_NOISE]
    opts = int(lvl[L_OPTIONS])
    react = lvl[L_REACT]
    spd = lvl[L_SPEED]
    # ---------------------------------------------------------------- 1. percepción (marco de equipo)
    px = np.zeros(8)
    py = np.zeros(8)
    pvx = np.zeros(8)
    pvy = np.zeros(8)
    mine = np.zeros(8, dtype=np.bool_)
    for p in range(8):
        mine[p] = (p < 4) == (team == 0)
        if ctrl[p] and mine[p]:
            x, y, vx_, vy_ = pos[1 + p, 0], pos[1 + p, 1], vel[1 + p, 0], vel[1 + p, 1]
        else:
            x = delayed_pos[1 + p, 0] + delayed_vel[1 + p, 0] * delay
            y = delayed_pos[1 + p, 1] + delayed_vel[1 + p, 1] * delay
            vx_, vy_ = delayed_vel[1 + p, 0], delayed_vel[1 + p, 1]
            if noise > 0.0:
                x += noise * nrand(seed, n, dec, 10 + p)
                y += noise * nrand(seed, n, dec, 20 + p)
        px[p] = x * s
        py[p] = y
        pvx[p] = vx_ * s
        pvy[p] = vy_
    # pelota percibida: estado retrasado avanzado `delay` ticks con su amortiguamiento
    bx0 = delayed_pos[0, 0]
    by0 = delayed_pos[0, 1]
    bvx0 = delayed_vel[0, 0]
    bvy0 = delayed_vel[0, 1]
    for _ in range(int(delay)):
        bx0 += bvx0
        by0 += bvy0
        bvx0 *= Q_B
        bvy0 *= Q_B
    if noise > 0.0:
        bx0 += 0.5 * noise * nrand(seed, n, dec, 30)
        by0 += 0.5 * noise * nrand(seed, n, dec, 31)
    bx = bx0 * s
    by = by0
    bvx = bvx0 * s
    bvy = bvy0
    t_end = predict_ball(bx, by, bvx, bvy, grav_x * s, grav_y, grav_left, 0.97, traj)
    t_end = max(1, min(t_end, HORIZON))
    # pelota camino a nuestro arco: tick y punto de cruce de la línea de gol
    save_t = -1
    save_y = 0.0
    if t_end < HORIZON and traj[t_end, 0] < -LINE_W and abs(traj[t_end, 1]) < GOAL_HH + 12.0:
        save_t = t_end
        save_y = traj[t_end, 1]
    # listas de propios y rivales activos
    own_idx = np.zeros(4, dtype=np.int64)
    opp_idx = np.zeros(4, dtype=np.int64)
    n_own = 0
    n_opp = 0
    for p in range(8):
        if not active[p]:
            continue
        if mine[p]:
            own_idx[n_own] = p
            n_own += 1
        else:
            opp_idx[n_opp] = p
            n_opp += 1
    opp_x = np.zeros(4)
    opp_y = np.zeros(4)
    opp_vx = np.zeros(4)
    opp_vy = np.zeros(4)
    for i in range(n_opp):
        q = opp_idx[i]
        opp_x[i] = px[q]
        opp_y[i] = py[q]
        opp_vx[i] = pvx[q]
        opp_vy[i] = pvy[q]
    # ---------------------------------------------------------------- 2. intercepciones
    t_int = np.full(8, 1e9)
    ix = np.zeros(8)
    iy = np.zeros(8)
    for p in range(8):
        if not active[p]:
            continue
        found = False
        # sólo antes de que la pelota salga o cruce una línea de gol (no se intercepta dentro del arco)
        for t in range(0, t_end, 2):
            tt = ttr(table, px[p], py[p], pvx[p], pvy[p], traj[t, 0], traj[t, 1], REACH - 6.0)
            if tt <= t:
                t_int[p] = t
                ix[p] = traj[t, 0]
                iy[p] = traj[t, 1]
                found = True
                break
        if not found:
            last = t_end - 1
            ix[p] = traj[last, 0]
            iy[p] = traj[last, 1]
            t_int[p] = HORIZON + ttr(table, px[p], py[p], pvx[p], pvy[p], ix[p], iy[p], REACH - 6.0)
        if not mine[p] or not ctrl[p]:
            t_int[p] += 0.0
    t_own = 1e9
    t_opp = 1e9
    win = -1
    opp_win = -1
    for i in range(n_own):
        p = own_idx[i]
        if t_int[p] < t_own:
            t_own = t_int[p]
            win = p
    for i in range(n_opp):
        p = opp_idx[i]
        if t_int[p] < t_opp:
            t_opp = t_int[p]
            opp_win = p
    margin = t_opp - t_own
    # ---------------------------------------------------------------- 3. fase con histéresis
    phase = mem_i[M_PHASE]
    if phase == PH_OWN:
        if margin < -3.0:
            phase = PH_OPP if margin < -8.0 else PH_CONTEST
    elif phase == PH_OPP:
        if margin > 3.0:
            phase = PH_OWN if margin > 8.0 else PH_CONTEST
    else:
        if margin > 6.0:
            phase = PH_OWN
        elif margin < -6.0:
            phase = PH_OPP
    if phase != mem_i[M_PHASE]:
        mem_i[M_PHASE_AGE] = 0
    else:
        mem_i[M_PHASE_AGE] += 1
    mem_i[M_PHASE] = phase
    restart_own = ri_team >= 0 and ri_team == team
    restart_opp = ri_team >= 0 and ri_team != team
    ko = ri_ko != 0
    if restart_own:
        phase = PH_OWN
    elif restart_opp:
        phase = PH_OPP
    # ---------------------------------------------------------------- 4. roles
    # rol especial: portador (posesión propia/disputa) o presionante (posesión rival)
    lead = -1
    if restart_own or (ko and ri_ko_team == team):
        # ejecutor: el propio que llega antes al punto (con histéresis)
        best_t = 1e9
        prev = mem_i[M_TAKER]
        for i in range(n_own):
            p = own_idx[i]
            tt = ttr(table, px[p], py[p], pvx[p], pvy[p], bx, by, REACH - 4.0)
            if p == prev:
                tt -= 25.0
            if tt < best_t:
                best_t = tt
                lead = p
        mem_i[M_TAKER] = lead
    else:
        mem_i[M_TAKER] = -1
        mem_i[M_WAIT] = 0
        if restart_opp or (ko and ri_ko_team != team):
            lead = -1
        else:
            lead = win
            if phase == PH_OPP and opts >= 3 and lead >= 0 and px[lead] > bx + 10.0:
                # relevo (L1+): el que llega antes quedó pasado (del lado equivocado de la pelota) y sólo
                # puede perseguir; presiona el mejor ubicado del lado del arco y el pasado se recupera
                best_t = 1e9
                alt = -1
                for i in range(n_own):
                    q = own_idx[i]
                    if q == lead or px[q] > bx - 20.0:
                        continue
                    tt = ttr(table, px[q], py[q], pvx[q], pvy[q], bx, by, REACH)
                    if tt < best_t:
                        best_t = tt
                        alt = q
                if alt >= 0:
                    lead = alt
    # conjunto de roles para el resto según fase y cantidad
    m = n_own - (1 if lead >= 0 else 0)
    roles = np.zeros(4, dtype=np.int64)
    if phase == PH_OWN:
        if m == 1:
            roles[0] = R_SUPPORT
        elif m == 2:
            roles[0] = R_SUPPORT
            roles[1] = R_SAFETY
        elif m >= 3:
            roles[0] = R_SUPPORT
            roles[1] = R_DEPTH if (opts >= 4 and sty[S_RISK] > 0.5) else R_WIDE
            roles[2] = R_SAFETY
    else:
        if m == 1:
            roles[0] = R_COVER if lead >= 0 else R_LAST
        elif m == 2:
            roles[0] = R_COVER if lead >= 0 else R_MARK
            roles[1] = R_LAST
        elif m >= 3:
            roles[0] = R_COVER if lead >= 0 else R_MARK
            roles[1] = R_MARK
            roles[2] = R_LAST
        if opts <= 1 and m >= 1:
            # L0–L1: forma básica, sin marcas
            for i in range(m):
                roles[i] = R_LAST if i == m - 1 else R_COVER
    # objetivos de cada rol (anclas)
    own_gx = -GOAL_X
    tgt_x = np.zeros(4)
    tgt_y = np.zeros(4)
    marked = np.zeros(4, dtype=np.bool_)
    side_y = 1.0 if by < 0.0 else -1.0     # lado con más espacio: el opuesto a la pelota
    width = 200.0 + 220.0 * sty[S_WIDTH]
    for r in range(m):
        role = roles[r]
        # anclas compactas (bloque de ~400 px de profundidad y ~260 de anchura, como los humanos)
        if role == R_SUPPORT:
            tgt_x[r] = bx + 30.0 + 70.0 * sty[S_RISK]
            tgt_y[r] = by + side_y * (150.0 + 60.0 * sty[S_WIDTH])
        elif role == R_WIDE:
            tgt_x[r] = bx + 90.0
            tgt_y[r] = by * 0.3 + side_y * (200.0 + 120.0 * sty[S_WIDTH])
        elif role == R_DEPTH:
            tgt_x[r] = min(GOAL_X - 160.0, bx + 260.0)
            tgt_y[r] = by * 0.5 + side_y * 90.0
        elif role == R_SAFETY:
            # seguridad (defensa en posesión): a una distancia de la pelota que permita llegar al corte;
            # con la pelota cerca del arco propio, cubre el arco (entre la pelota y el centro del arco)
            gdx = bx - own_gx
            gdy = by
            gd = math.sqrt(gdx * gdx + gdy * gdy) + 1e-9
            if gd < 520.0:
                depth_goal = _clip(0.4 * gd, 45.0, 170.0)
                tgt_x[r] = own_gx + gdx / gd * depth_goal
                tgt_y[r] = gdy / gd * depth_goal
            else:
                tgt_x[r] = max(own_gx + 220.0, bx - (230.0 + 120.0 * sty[S_DEPTH]))
                tgt_y[r] = by * 0.45
        elif role == R_COVER:
            gdx = own_gx - bx
            gdy = -by
            gd = math.sqrt(gdx * gdx + gdy * gdy) + 1e-9
            k = min(170.0, 0.4 * gd)
            tgt_x[r] = bx + gdx / gd * k
            tgt_y[r] = by + gdy / gd * k
        elif role == R_LAST:
            # último hombre: del lado del arco, sobre la línea pelota-arco, a distancia de la pelota
            # según la amenaza; sólo entre los postes si el tiro es inminente
            gdx = own_gx - bx
            gdy = -by
            gd = math.sqrt(gdx * gdx + gdy * gdy) + 1e-9
            if gd < 420.0:
                depth_goal = _clip(0.4 * gd, 55.0, 160.0)
                tgt_x[r] = own_gx - gdx / gd * depth_goal
                tgt_y[r] = -gdy / gd * depth_goal
            else:
                dist = _clip(0.4 * gd, 130.0, 240.0 + 130.0 * sty[S_DEPTH])
                tgt_x[r] = bx + gdx / gd * dist
                tgt_y[r] = by + gdy / gd * dist
        elif role == R_MARK:
            # rival peligroso más cercano a la jugada, sin marca (no el que tiene la pelota)
            best = -1
            best_v = -1e9
            for i in range(n_opp):
                q = opp_idx[i]
                if q == opp_win or marked[i]:
                    continue
                d_goal = math.hypot(opp_x[i] - own_gx, opp_y[i])
                v = -0.6 * d_goal - math.hypot(opp_x[i] - bx, opp_y[i] - by)
                if v > best_v:
                    best_v = v
                    best = i
            if best >= 0:
                marked[best] = True
                gdx = own_gx - opp_x[best]
                gdy = -opp_y[best]
                gd = math.sqrt(gdx * gdx + gdy * gdy) + 1e-9
                mx = opp_x[best] + gdx / gd * 45.0
                my = opp_y[best] + gdy / gd * 45.0
                tgt_x[r] = 0.75 * mx + 0.25 * bx
                tgt_y[r] = 0.75 * my + 0.25 * by
            else:
                tgt_x[r] = bx + (own_gx - bx) * 0.25
                tgt_y[r] = by * 0.6
        tgt_x[r] = _clip(tgt_x[r], -LINE_W + 40.0, LINE_W - 40.0)
        tgt_y[r] = _clip(tgt_y[r], -LINE_H + 45.0, LINE_H - 45.0)
    # asignación: propios sin el especial; costo = tiempo de llegada; histéresis sólo para controlados
    others = np.zeros(4, dtype=np.int64)
    k_o = 0
    for i in range(n_own):
        p = own_idx[i]
        if p != lead:
            others[k_o] = p
            k_o += 1
    assign = np.full(4, -1, dtype=np.int64)
    if k_o > 0:
        best_cost = 1e18
        perm = np.arange(k_o)
        best_perm = perm.copy()
        # permutaciones (k_o ≤ 3 → ≤ 6) por Heap
        c = np.zeros(k_o, dtype=np.int64)
        cost = 0.0
        for r in range(k_o):
            p = others[perm[r]]
            cst = math.hypot(px[p] - tgt_x[r], py[p] - tgt_y[r])
            if ctrl[p] and mem_i[M_ROLE + p] == roles[r]:
                cst -= lvl[L_HYST]
            cost += cst
        best_cost = cost
        best_perm[:] = perm
        i = 0
        while i < k_o:
            if c[i] < i:
                if i % 2 == 0:
                    perm[0], perm[i] = perm[i], perm[0]
                else:
                    perm[c[i]], perm[i] = perm[i], perm[c[i]]
                cost = 0.0
                for r in range(k_o):
                    p = others[perm[r]]
                    cst = math.hypot(px[p] - tgt_x[r], py[p] - tgt_y[r])
                    if ctrl[p] and mem_i[M_ROLE + p] == roles[r]:
                        cst -= lvl[L_HYST]
                    cost += cst
                if cost < best_cost:
                    best_cost = cost
                    best_perm[:] = perm
                c[i] += 1
                i = 0
            else:
                c[i] = 0
                i += 1
        for r in range(k_o):
            assign[r] = others[best_perm[r]]
    # ---------------------------------------------------------------- 5–7. acciones de controlados
    for r in range(k_o):
        p = assign[r]
        if p < 0 or not ctrl[p]:
            continue
        mem_i[M_ROLE + p] = roles[r]
        tx = tgt_x[r]
        ty = tgt_y[r]
        if roles[r] == R_SUPPORT or roles[r] == R_WIDE or roles[r] == R_DEPTH:
            tx, ty = _best_support(table, tx, ty, bx, by, px, py, p, own_idx, n_own, opp_x, opp_y, opp_vx,
                                   opp_vy, n_opp, react, roles[r], sty, mem_f[F_SUP + 2 * p], mem_f[F_SUP + 2 * p + 1])
            mem_f[F_SUP + 2 * p] = tx
            mem_f[F_SUP + 2 * p + 1] = ty
        mv = move_action(px[p], py[p], pvx[p], pvy[p], tx, ty, 0.0, spd, False)
        # despeje de oportunidad (L≥1): un defensor con la pelota encima y peligro → la saca
        out[p] = mv
        dxb = bx - px[p]
        dyb = by - py[p]
        if (opts >= 1 and phase != PH_OWN and not kick_cancel[p] and math.hypot(dxb, dyb) < REACH + 4.0
                and bx < -200.0 and lead != p):
            if safe_clear(px[p], py[p], bx, by, own_gx):
                out[p] = mv + 9
    if lead >= 0 and ctrl[lead]:
        mem_i[M_ROLE + lead] = R_TAKER if (restart_own or ko) else (R_CARRIER if phase != PH_OPP else R_PRESS)
        if restart_own or (ko and ri_ko_team == team):
            out[lead] = _restart_taker(n, team, seed, dec, table, lead, px, py, pvx, pvy, bx, by, bvx, bvy,
                                       own_idx, n_own, opp_x, opp_y, opp_vx, opp_vy, n_opp, ri_kind, ri_ticks,
                                       ko, kick_cancel, lvl, sty, mem_i, mem_f)
        else:
            # ¿queda alguien detrás? Sin cobertura, el que va a la pelota es el último hombre (L1+)
            d_bg = math.hypot(bx - own_gx, by)
            contain = False
            if opts >= 3 and opp_win >= 0:
                contain = True
                for i in range(n_own):
                    q = own_idx[i]
                    if q != lead and px[q] < bx - 30.0 and math.hypot(px[q] - own_gx, py[q]) < d_bg - 80.0:
                        contain = False
                        break
            pressing = phase == PH_OPP and t_own > t_opp + 4.0 and opp_win >= 0
            # el último hombre lejos de su arco no sale a una pelota que el rival todavía disputa: sólo con una
            # ventaja clara (20 ticks: llega con el rival a > 60 px). Si no, contiene. El RL aprendía a alejarse
            # un poco de la pelota para hacerlo salir y desbordarlo (2026-10-04).
            hold = contain and d_bg > 430.0 and t_own > 3.0 and t_opp - t_own < 20.0
            if pressing or hold:
                out[lead] = _press(n, seed, dec, lead, px, py, pvx, pvy, bx, by, bvx, bvy, own_gx, kick_cancel,
                                   lvl, sty, mem_i, traj, contain, opp_win)
            else:
                out[lead] = _carrier(n, team, seed, dec, table, lead, px, py, pvx, pvy, bx, by, bvx, bvy, ix[lead],
                                     iy[lead], t_int[lead], own_idx, n_own, opp_x, opp_y, opp_vx, opp_vy, n_opp,
                                     kick_cancel, lvl, sty, mem_i, mem_f, traj)
    # atajada: si la pelota va a cruzar nuestra línea de gol, el propio que mejor llega al punto de cruce
    # se interpone (cuerpo entre la pelota y el arco) y la despeja si la patada es segura
    if save_t >= 0 and not restart_opp and not ko:
        sx_ = -GOAL_X + R_P + 4.0
        best_s = -1
        best_tt = 1e9
        for i in range(n_own):
            p = own_idx[i]
            tt = ttr(table, px[p], py[p], pvx[p], pvy[p], sx_, save_y, 6.0)
            if tt < best_tt:
                best_tt = tt
                best_s = p
        vb = math.hypot(bvx, bvy) + 1e-9
        ubx = bvx / vb
        uby = bvy / vb
        trailing = False
        if best_s >= 0:
            rxs = px[best_s] - bx
            rys = py[best_s] - by
            lat = -rxs * uby + rys * ubx           # lado del jugador respecto de la marcha (+: izquierda)
            trailing = rxs * ubx + rys * uby < 0.0 and abs(lat) < R_P + R_B + 12.0
        if best_s >= 0 and ctrl[best_s] and trailing:
            # viene detrás de la pelota: ponerse delante exige atravesarla y la empujaba adentro (S5,
            # 2026-10-04). Rodearla por su lado y desviarla hacia el otro, afuera del arco
            nx_, ny_, ok = deflect_dir(px[best_s], py[best_s], bx, by, bvx, bvy)
            if not ok:
                sgn = 1.0 if lat >= 0.0 else -1.0
                nx_, ny_ = kick_normal(sgn * uby, -sgn * ubx, bvx, bvy)
            mv = approach_move(px[best_s], py[best_s], pvx[best_s], pvy[best_s], bx, by, bvx, bvy, bx, by, nx_, ny_,
                               spd, False)
            out[best_s] = mv
            mem_i[M_ROLE + best_s] = R_LAST
            mem_i[M_PLAN + best_s] = P_NONE
            if not kick_cancel[best_s] and math.hypot(bx - px[best_s], by - py[best_s]) < REACH + 3.0:
                if safe_deflect(px[best_s], py[best_s], bx, by, bvx, bvy):
                    out[best_s] = mv + 9
        elif best_s >= 0 and ctrl[best_s]:
            # punto de bloqueo: sobre la trayectoria, del lado del arco, alcanzable antes del cruce
            tx = sx_
            ty = save_y
            for t in range(2, save_t, 2):
                qx = traj[t, 0] + ubx * (R_P + R_B)
                qy = traj[t, 1] + uby * (R_P + R_B)
                if qx < sx_:
                    break
                tt = ttr(table, px[best_s], py[best_s], pvx[best_s], pvy[best_s], qx, qy, 4.0)
                if tt + 2.0 <= t:
                    tx = qx
                    ty = qy
                    break
            mv = move_action(px[best_s], py[best_s], pvx[best_s], pvy[best_s], tx, ty, 0.0, spd, False)
            out[best_s] = mv
            mem_i[M_ROLE + best_s] = R_LAST
            mem_i[M_PLAN + best_s] = P_NONE
            if not kick_cancel[best_s] and math.hypot(bx - px[best_s], by - py[best_s]) < REACH + 3.0:
                if safe_clear(px[best_s], py[best_s], bx, by, own_gx):
                    out[best_s] = mv + 9
    # restart rival / saque inicial rival: nadie del equipo patea antes de tiempo
    for p in range(8):
        if ctrl[p] and mine[p] and kick_cancel[p] and out[p] >= 9:
            out[p] -= 9  # soltar la tecla para volver a armar la patada


@njit(cache=True)
def safe_clear(px, py, bx, by, own_gx):
    """¿Es segura una patada defensiva (dirección jugador→pelota)? La trayectoria no puede pasar por el
    arco propio ni ir hacia atrás cerca del área."""
    dx = bx - px
    dy = by - py
    d = math.hypot(dx, dy)
    if d < 1e-6:
        return False
    ux = dx / d
    uy = dy / d
    need = 0.55 if bx < own_gx + 350.0 else -0.2
    if ux < need:
        return False
    if ux < 0.0:
        # hacia atrás: no puede cruzar la línea de gol propia entre los postes (con margen)
        t = (own_gx - bx) / ux
        y_at = by + uy * t
        if abs(y_at) < 124.0 + 60.0:
            return False
    return True


@njit(cache=True)
def safe_deflect(px, py, bx, by, bvx, bvy):
    """¿Patear ahora (dirección jugador→pelota) deja la pelota fuera del arco propio? Suma la velocidad que
    trae la pelota (desvío de una pelota que va hacia el arco)."""
    dx = bx - px
    dy = by - py
    d = math.hypot(dx, dy)
    if d < 1e-6:
        return False
    rvx = bvx + KICK_SPEED * dx / d
    rvy = bvy + KICK_SPEED * dy / d
    rv = math.hypot(rvx, rvy) + 1e-9
    return not crosses_own_goal(bx, by, rvx / rv, rvy / rv, 25.0)


@njit(cache=True)
def deflect_dir(px, py, bx, by, bvx, bvy):
    """Dirección de contacto (jugador→pelota) para despejar de primera: la de menor giro respecto de la actual
    cuya pelota resultante (velocidad que trae + patada) no va al arco propio; a igual giro, la que la manda
    hacia el lateral más cercano. Devuelve (nx, ny, encontrada)."""
    base = math.atan2(by - py, bx - px)
    side = 1.0 if by >= 0.0 else -1.0
    for k in range(9):
        best_s = -1e9
        bnx = 0.0
        bny = 0.0
        for sg in range(2):
            if k == 0 and sg == 1:
                continue
            ang = base + 0.2 * k * (1.0 if sg == 0 else -1.0)
            nx = math.cos(ang)
            ny = math.sin(ang)
            rvx = bvx + KICK_SPEED * nx
            rvy = bvy + KICK_SPEED * ny
            rv = math.hypot(rvx, rvy) + 1e-9
            if crosses_own_goal(bx, by, rvx / rv, rvy / rv, 40.0):
                continue
            sc = rvy * side / rv
            if sc > best_s:
                best_s = sc
                bnx = nx
                bny = ny
        if best_s > -1e8:
            return bnx, bny, True
    return 0.0, 0.0, False


@njit(cache=True)
def _press(n, seed, dec, me, px, py, pvx, pvy, bx, by, bvx, bvy, own_gx, kick_cancel, lvl, sty, mem_i, traj,
           contain, carrier):
    """Presionante sin la pelota: ir a la pelota por el lado del arco propio (bloquear la conducción) y
    disputarla con una patada de quite cuando no la manda hacia el arco propio.

    contain (último hombre, sin nadie detrás): no ir al contacto lejos del arco. Pararse sobre la línea
    pelota→arco a 70–110 px (a esa distancia tapa todo el ángulo de tiro) y retroceder con el poseedor;
    cerrar y disputar sólo en zona de tiro (< 430 px del arco). Así un regate en diagonal no lo deja pasado
    y el 2v1 se resuelve pasando. Una pelota suelta que el defensor gana no llega acá (la fase deja de ser
    de posesión rival y va a disputarla); una que el poseedor sigue ganando no es motivo para salir: el RL
    aprendió a alejarse de la pelota para hacerlo salir y desbordarlo (2026-10-04).

    En zona de tiro, el último hombre no se tira al contacto: tapa el ángulo a 30–50 px del lado del arco,
    leyendo la conducción del poseedor (su velocidad), y sólo mete la pierna si la pelota le queda al alcance.
    Ir al contacto lo dejaba pasado con un corte en diagonal (el RL ganaba así el 1v1 el 81–95%, S3 2026-10-04).
    """
    spd = lvl[L_SPEED]
    # pelota un poco adelantada (el poseedor la empuja)
    t = 4
    cbx = traj[t, 0]
    cby = traj[t, 1]
    if contain and carrier >= 0 and math.hypot(px[carrier] - bx, py[carrier] - by) < R_P + R_B + 20.0:
        # pelota conducida: leer hacia dónde va el poseedor (lo empuja con el cuerpo)
        cbx = bx + pvx[carrier] * 6.0
        cby = by + pvy[carrier] * 6.0
    gdx = own_gx - cbx
    gdy = -cby
    gd = math.sqrt(gdx * gdx + gdy * gdy) + 1e-9
    d_me = math.hypot(px[me] - cbx, py[me] - cby)
    arrive = 1.2
    if contain and gd > 430.0:
        off = 70.0 + _clip((gd - 430.0) / 320.0, 0.0, 1.0) * 40.0
        arrive = 0.0
    elif contain:
        # zona de tiro: tapar el ángulo sin tirarse (30 px junto al arco, 50 px al borde de la zona)
        off = 30.0 + 20.0 * _clip((gd - 200.0) / 230.0, 0.0, 1.0)
        arrive = 0.0
    elif d_me < 140.0:
        # cerca: contra la pelota del lado del arco
        off = R_P + R_B - 2.0
    else:
        # lejos: a una distancia que deje cerrar el ángulo
        off = min(60.0, 0.15 * d_me)
    tx = cbx + gdx / gd * off
    ty = cby + gdy / gd * off
    mv = move_action(px[me], py[me], pvx[me], pvy[me], tx, ty, arrive, spd, False)
    if not kick_cancel[me]:
        dx = bx - px[me]
        dy = by - py[me]
        d = math.hypot(dx, dy)
        if d - R_P - R_B < 4.0 + 3.0 and d > 1e-6:
            # quite: la dirección de la patada es jugador→pelota; nunca hacia el arco propio
            if safe_clear(px[me], py[me], bx, by, own_gx):
                return mv + 9
    return mv


@njit(cache=True)
def _best_support(table, ax, ay, bx, by, px, py, me, own_idx, n_own, opp_x, opp_y, opp_vx, opp_vy, n_opp,
                  react, role, sty, prev_x, prev_y):
    """Mejor punto de apoyo cerca del ancla: línea de pase abierta, progreso, separación, dentro de la cancha."""
    best_x = ax
    best_y = ay
    best_v = -1e18
    for ring in range(3):
        rad = 75.0 * ring
        n_ang = 1 if ring == 0 else 8
        for a in range(n_ang):
            ang = 2.0 * math.pi * a / n_ang
            cx = ax + rad * math.cos(ang)
            cy = ay + rad * math.sin(ang)
            if abs(cx) > LINE_W - 50.0 or abs(cy) > LINE_H - 50.0:
                continue
            d_ball = math.hypot(cx - bx, cy - by)
            v = 0.0
            # línea de pase desde la pelota
            if d_ball > 1.0:
                mg = path_margin(table, bx, by, cx, cy, KICK_SPEED, opp_x, opp_y, opp_vx, opp_vy, n_opp, react)
                v += 1.5 * _sigmoid((mg - 3.0) / 3.0)
            # distancia útil a la pelota (alcance de pase ~600)
            if role == 2:   # R_SUPPORT
                v -= abs(d_ball - 230.0) / 300.0
            else:
                v -= max(0.0, d_ball - 560.0) / 200.0
            v += 1.2 * threat(cx, cy) + 0.15 * (cx - bx) / 400.0
            # separación de compañeros
            for i in range(n_own):
                q = own_idx[i]
                if q == me:
                    continue
                dq = math.hypot(cx - px[q], cy - py[q])
                if dq < 150.0:
                    v -= (150.0 - dq) / 60.0
            # costo de llegar e histéresis: no cambiar de punto por diferencias mínimas
            v -= math.hypot(cx - px[me], cy - py[me]) / 900.0
            if math.hypot(cx - prev_x, cy - prev_y) < 60.0:
                v += 0.25
            if v > best_v:
                best_v = v
                best_x = cx
                best_y = cy
    return best_x, best_y


@njit(cache=True)
def crosses_own_goal(bx, by, ux, uy, margin):
    """¿La línea de la pelota desde (bx, by) en dirección u cruza el arco propio (x = -1162) con margen?"""
    if ux >= -1e-6:
        return False
    t = (-GOAL_X - bx) / ux
    y_at = by + uy * t
    return abs(y_at) < GOAL_HH + margin


@njit(cache=True)
def possession_value(x, y):
    """Valor de tener la pelota en (x, y) (marco de equipo): avanzar vale, cerca del arco vale más."""
    return 0.04 + 0.30 * _sigmoid(x / 330.0) + threat(x, y)


@njit(cache=True)
def loss_cost(x, y):
    """Costo de perderla en (x, y): el valor de posesión del rival en ese punto."""
    return possession_value(-x, y)


@njit(cache=True)
def _options(n, team, seed, dec, table, me, px, py, bx, by, bvx, bvy, own_idx, n_own, opp_x, opp_y, opp_vx,
             opp_vy, n_opp, lvl, sty, speed_mult, vals, ux_o, uy_o, kinds, tgt_x, tgt_y, partner, mvx=0.0, mvy=0.0):
    """Opciones del portador con su valor esperado: éxito × valor de la posesión siguiente − fracaso × valor
    para el rival donde la pierde. Devuelve la cantidad de opciones."""
    opts = int(lvl[L_OPTIONS])
    react = lvl[L_REACT]
    risk = sty[S_RISK]
    k = 0
    v0 = KICK_SPEED * speed_mult
    lose_here = loss_cost(bx, by)
    pressure = 99.0
    for o in range(n_opp):
        to = ttr(table, opp_x[o], opp_y[o], opp_vx[o], opp_vy[o], bx, by, R_P + R_B)
        if to < pressure:
            pressure = to
    # ---- tiro: puntos a lo largo del arco
    dgoal = math.hypot(GOAL_X - bx, by)
    if dgoal < 0.95 * v0 / (1.0 - Q_B):
        for j in range(7):
            gy = -GOAL_HH + 18.0 + j * (2.0 * GOAL_HH - 36.0) / 6.0
            if opts == 0 and j != 3:
                continue
            dx = GOAL_X + 5.0 - bx
            dy = gy - by
            d = math.hypot(dx, dy)
            ux = dx / d
            uy = dy / d
            # el error angular del nivel no puede sacar la pelota del arco: el ángulo libre hasta los
            # palos (visto desde la pelota) debe cubrir el error típico
            ta = math.atan2(uy, ux)
            a1 = math.atan2(GOAL_HH - 9.0 - by, GOAL_X - bx)
            a2 = math.atan2(-GOAL_HH + 9.0 - by, GOAL_X - bx)
            free = min(a1 - ta, ta - a2)
            if free < (0.35 * lvl[L_AIM_TOL] + 1.5 * lvl[L_AIM_NOISE]) * math.pi / 180.0 and j != 3:
                continue
            sp = v0 + bvx * ux + bvy * uy
            if sp < 1.0:
                continue
            mg = path_margin(table, bx, by, GOAL_X + 5.0, gy, sp, opp_x, opp_y, opp_vx, opp_vy, n_opp, react)
            p_goal = _sigmoid((mg - 1.0) / 2.0)
            # un tiro desviado o atajado suele quedar para el rival lejos de nuestro arco (costo bajo)
            # un tiro al arco también genera rebotes y córners (valor residual 0,1)
            vals[k] = p_goal * (1.0 + 0.35 * risk) + (1.0 - p_goal) * (0.1 - 0.6 * loss_cost(GOAL_X - 150.0, gy))
            ux_o[k] = ux
            uy_o[k] = uy
            kinds[k] = 1
            tgt_x[k] = GOAL_X + 5.0
            tgt_y[k] = gy
            partner[k] = -1
            k += 1
    # ---- pases (L2+): al pie, adelantado (L3+), en profundidad (L4+)
    if opts >= 2:
        for i in range(n_own):
            q = own_idx[i]
            if q == me:
                continue
            for variant in range(3):
                if variant == 1 and opts < 3:
                    continue
                if variant == 2 and opts < 4:
                    continue
                if variant == 0:
                    rx = px[q]
                    ry = py[q]
                elif variant == 1:
                    rx = px[q] + 90.0
                    ry = py[q] * 0.85
                else:
                    rx = min(GOAL_X - 120.0, px[q] + 220.0)
                    ry = py[q] * 0.7
                if abs(ry) > LINE_H - 40.0 or abs(rx) > LINE_W - 40.0:
                    continue
                dx = rx - bx
                dy = ry - by
                d = math.hypot(dx, dy)
                if d < 60.0:
                    continue
                ux = dx / d
                uy = dy / d
                sp = v0 + bvx * ux + bvy * uy
                if sp < 1.0:
                    continue
                tb = ball_travel_ticks(d, sp)
                if tb > 1e8:
                    continue
                t_recv = ttr(table, px[q], py[q], 0.0, 0.0, rx, ry, R_P + R_B)
                late = max(0.0, t_recv - tb - 4.0)
                mg = path_margin(table, bx, by, rx, ry, sp, opp_x, opp_y, opp_vx, opp_vy, n_opp, react)
                p = _sigmoid((mg - 1.5 - 2.5 * (1.0 - risk)) / 2.0) * math.exp(-late / 12.0)
                gain = possession_value(rx, ry) * (1.0 + 0.2 * sty[S_DIRECT] * min(1.0, d / 500.0))
                vals[k] = p * gain - (1.0 - p) * loss_cost(0.5 * (bx + rx), 0.5 * (by + ry))
                ux_o[k] = ux
                uy_o[k] = uy
                kinds[k] = 2
                tgt_x[k] = rx - px[q]       # objetivo relativo al receptor
                tgt_y[k] = ry - py[q]
                partner[k] = q
                k += 1
    # ---- conducción en 8 direcciones (marco de equipo)
    for j in range(8):
        ang = (j - 2) * math.pi / 4.0
        ux = math.cos(ang)
        uy = math.sin(ang)
        # conducir hacia adelante o de costado; hacia atrás se juega con pases a compañeros
        if (opts == 0 and ux < 0.5) or ux < -0.1:
            continue
        tx = bx + 130.0 * ux
        ty = by + 130.0 * uy
        if abs(tx) > LINE_W - 30.0 or abs(ty) > LINE_H - 90.0:
            continue
        t_me = ttr(table, px[me], py[me], 0.0, 0.0, tx, ty, 10.0) * 1.25
        free = 99.0
        for o in range(n_opp):
            to = ttr(table, opp_x[o], opp_y[o], opp_vx[o], opp_vy[o], tx, ty, R_P + R_B) + react
            if to - t_me < free:
                free = to - t_me
        # un duelo no se pierde sólo porque el rival llegue: piso de retención y la pérdida suele dejar
        # la pelota suelta (costo reducido); retroceder cuesta tiempo (el rival se reorganiza)
        p = 0.45 + 0.55 * _sigmoid((free - 2.0) / 3.0)
        keep = 0.92 - 0.12 * sty[S_TEMPO]      # conducir es más lento que pasar
        gdx_ = GOAL_X - bx
        gdy_ = -by
        toward = (ux * gdx_ + uy * gdy_) / (math.hypot(gdx_, gdy_) + 1e-9)
        vals[k] = p * keep * (possession_value(tx, ty) + 0.08 * toward) - (1.0 - p) * 0.6 * loss_cost(tx, ty)
        ux_o[k] = ux
        uy_o[k] = uy
        kinds[k] = 3
        tgt_x[k] = tx
        tgt_y[k] = ty
        partner[k] = -1
        k += 1
    # ---- autopase (L2+): patear hacia el espacio libre y correrla (la patada es siempre a fondo:
    # la pelota recorre ~420 px antes de frenar lo suficiente para alcanzarla)
    if opts >= 2:
        for j in range(3):
            ang = (j - 1) * math.pi / 7.0
            ux = math.cos(ang)
            uy = math.sin(ang)
            tx = bx + 420.0 * ux
            ty = by + 420.0 * uy
            if tx > GOAL_X - 80.0 or abs(ty) > LINE_H - 70.0:
                continue
            tb = ball_travel_ticks(420.0, v0)
            t_me = ttr(table, px[me], py[me], 0.0, 0.0, tx, ty, R_P + R_B)
            if t_me > tb + 30.0:
                continue
            mg = path_margin(table, bx, by, tx, ty, v0, opp_x, opp_y, opp_vx, opp_vy, n_opp, react)
            p = _sigmoid((mg - 6.0) / 3.0)
            vals[k] = p * 0.9 * possession_value(tx, ty) - (1.0 - p) * loss_cost(tx, ty)
            ux_o[k] = ux
            uy_o[k] = uy
            kinds[k] = 8
            tgt_x[k] = tx
            tgt_y[k] = ty
            partner[k] = -1
            k += 1
    # ---- despeje (L1+): bajo presión en campo propio, lejos de los rivales
    if opts >= 1 and bx < -550.0 and pressure < 7.0:
        for j in range(3):
            cy = (-1.0 + j) * 480.0
            tx = bx + 560.0
            dx = tx - bx
            dy = cy - by
            d = math.hypot(dx, dy)
            ux = dx / d
            uy = dy / d
            vals[k] = 0.5 * possession_value(tx, cy) - 0.5 * loss_cost(tx, cy)
            ux_o[k] = ux
            uy_o[k] = uy
            kinds[k] = 5
            tgt_x[k] = tx
            tgt_y[k] = cy
            partner[k] = -1
            k += 1
    # ---- despeje al costado (L1+): presionado cerca del arco propio, sacarla por el lateral (cuesta un lateral
    # rival lejos del arco). Con la pelota viniendo hacia el arco es la salida que menos rodeo exige
    if opts >= 1 and bx < -450.0 and pressure < 9.0:
        for j in range(2):
            sg = -1.0 if j == 0 else 1.0
            tx = bx + 220.0
            ty = sg * (LINE_H + 80.0)
            dx = tx - bx
            dy = ty - by
            d = math.hypot(dx, dy)
            vals[k] = -0.6 * loss_cost(tx, sg * LINE_H)
            ux_o[k] = dx / d
            uy_o[k] = dy / d
            kinds[k] = 5
            tgt_x[k] = tx
            tgt_y[k] = ty
            partner[k] = -1
            k += 1
    # ---- despeje de primera (L1+): en campo propio con presión, la patada de menor giro respecto de cómo llega
    # el portador (ver deflect_dir); vale como un despeje según dónde cae
    if opts >= 1 and bx < -200.0 and pressure < 12.0:
        nx_, ny_, ok = deflect_dir(px[me], py[me], bx, by, bvx, bvy)
        if ok:
            rvx = bvx + KICK_SPEED * nx_
            rvy = bvy + KICK_SPEED * ny_
            rv = math.hypot(rvx, rvy) + 1e-9
            tx = _clip(bx + 420.0 * rvx / rv, -LINE_W + 20.0, LINE_W - 20.0)
            ty = _clip(by + 420.0 * rvy / rv, -LINE_H + 20.0, LINE_H - 20.0)
            vals[k] = 0.5 * possession_value(tx, ty) - 0.5 * loss_cost(tx, ty)
            ux_o[k] = rvx / rv
            uy_o[k] = rvy / rv
            kinds[k] = 5
            tgt_x[k] = bx + 420.0 * rvx / rv
            tgt_y[k] = by + 420.0 * rvy / rv
            partner[k] = -1
            k += 1
    # ---- costo de preparación: para patear o conducir en dirección u hay que estar detrás de la pelota;
    # mientras el portador se acomoda, el rival llega (probabilidad de ejecutar antes de la presión)
    for j in range(k):
        if crosses_own_goal(bx, by, ux_o[j], uy_o[j], 90.0):
            vals[j] = -10.0      # nunca jugar hacia el arco propio (pase atrás al arquero incluido)
            continue
        ax, ay, sp_ = approach_point(px[me], py[me], bx, by, ux_o[j], uy_o[j], mvx, mvy)
        # acomodarse detrás de la pelota no puede exigir entrar al arco propio (empujaría la pelota adentro)
        cx_ = bx - ux_o[j] * (R_P + R_B)
        if cx_ < -GOAL_X + 25.0 or ax < -GOAL_X + 25.0:
            vals[j] = -10.0
            continue
        if kinds[j] == 3:
            setup = setup_ticks(table, px[me], py[me], mvx, mvy, bx, by, ux_o[j], uy_o[j])
        else:
            knx, kny = kick_normal(ux_o[j], uy_o[j], bvx, bvy)
            setup = setup_ticks(table, px[me], py[me], mvx, mvy, bx, by, knx, kny)
        p_exec = _sigmoid((pressure - setup - 2.0) / 3.0)
        vals[j] = p_exec * vals[j] - (1.0 - p_exec) * 0.6 * lose_here
    return k


@njit(cache=True)
def _plan_dir(me, plan, partner, px, py, bx, by, mem_f):
    """Dirección de patada actual de un plan guardado (el objetivo de un pase sigue al receptor)."""
    if plan == P_DRIBBLE:
        return mem_f[F_PLAN_U + 2 * me], mem_f[F_PLAN_U + 2 * me + 1]
    tx = mem_f[F_PLAN_T + 2 * me]
    ty = mem_f[F_PLAN_T + 2 * me + 1]
    if plan == P_PASS and partner >= 0:
        tx += px[partner]
        ty += py[partner]
    dx = tx - bx
    dy = ty - by
    d = math.hypot(dx, dy)
    if d < 1e-6:
        return mem_f[F_PLAN_U + 2 * me], mem_f[F_PLAN_U + 2 * me + 1]
    return dx / d, dy / d


@njit(cache=True)
def _carrier(n, team, seed, dec, table, me, px, py, pvx, pvy, bx, by, bvx, bvy, ix, iy, t_me, own_idx, n_own,
             opp_x, opp_y, opp_vx, opp_vy, n_opp, kick_cancel, lvl, sty, mem_i, mem_f, traj):
    """Acción del portador: elige jugada por valor esperado, la sostiene hasta ejecutarla (salvo que
    aparezca otra claramente mejor) y la ejecuta."""
    spd = lvl[L_SPEED]
    if t_me > 9.0:
        mem_i[M_PLAN + me] = P_NONE
        rx = px[me] - bx
        ry = py[me] - by
        d2 = rx * rx + ry * ry
        moving = bvx * bvx + bvy * bvy > 0.36
        if moving and d2 < 200.0 * 200.0:
            pressure = 1e9
            for o in range(n_opp):
                to = ttr(table, opp_x[o], opp_y[o], opp_vx[o], opp_vy[o], bx, by, R_P + R_B)
                if to < pressure:
                    pressure = to
            pressed = pressure < t_me + 12.0
            if pressed and d2 < 90.0 * 90.0 and rx * bvx + ry * bvy < 0.0:
                # viene detrás de una pelota que se aleja con un rival cerca: despejarla de primera con el menor
                # giro seguro (ir derecho la chocaba en la dirección en que iba, a menudo hacia el arco propio)
                nx_, ny_, ok = deflect_dir(px[me], py[me], bx, by, bvx, bvy)
                if ok:
                    mv = approach_move(px[me], py[me], pvx[me], pvy[me], bx, by, bvx, bvy, ix, iy, nx_, ny_, spd, False)
                    if not kick_cancel[me] and math.hypot(bx - px[me], by - py[me]) < REACH + 3.0:
                        if safe_deflect(px[me], py[me], bx, by, bvx, bvy):
                            return mv + 9
                    return mv
            if not pressed and bvx < -0.5 and rx > -10.0:
                # sin presión, pelota hacia nuestro arco y el portador no está del lado del arco: rodearla hasta
                # quedar en su camino, del lado del arco, y recibirla de frente (S5, 2026-10-04: perseguirla
                # desde atrás le daba tiempo al delantero rival y la empujaba hacia el arco)
                return approach_move(px[me], py[me], pvx[me], pvy[me], bx, by, bvx, bvy, ix, iy, 1.0, 0.0, spd, False)
        return move_action(px[me], py[me], pvx[me], pvy[me], ix, iy, 1.0, spd, False)
    plan = mem_i[M_PLAN + me]
    until = mem_i[M_PLAN_UNTIL + me]
    if plan == P_NONE or dec >= until:
        vals = np.zeros(64)
        ux_o = np.zeros(64)
        uy_o = np.zeros(64)
        kinds = np.zeros(64, dtype=np.int64)
        tgt_x = np.zeros(64)
        tgt_y = np.zeros(64)
        partner = np.full(64, -1, dtype=np.int64)
        k = _options(n, team, seed, dec, table, me, px, py, bx, by, bvx, bvy, own_idx, n_own, opp_x, opp_y,
                     opp_vx, opp_vy, n_opp, lvl, sty, 1.0, vals, ux_o, uy_o, kinds, tgt_x, tgt_y, partner,
                     pvx[me], pvy[me])
        # valor actual del plan guardado (la opción del mismo tipo y destino)
        cur = -1e18
        if plan != P_NONE:
            # el mismo plan reevaluado: mismo tipo y receptor, la dirección más parecida
            cu_x, cu_y = _plan_dir(me, plan, mem_i[M_PARTNER + me], px, py, bx, by, mem_f)
            best_cos = -2.0
            for j in range(k):
                if kinds[j] == plan and partner[j] == mem_i[M_PARTNER + me]:
                    cs = ux_o[j] * cu_x + uy_o[j] * cu_y
                    if cs > best_cos:
                        best_cos = cs
                        cur = vals[j]
            if best_cos < 0.8:
                cur = -1e18
        best = -1e18
        for j in range(k):
            if vals[j] > best:
                best = vals[j]
        # cambiar sólo si otra jugada es claramente mejor (compromiso: no ir y venir)
        if k > 0 and (plan == P_NONE or best > cur + 0.08 + 0.35 * abs(cur)):
            # elección estocástica (softmax con la temperatura del nivel)
            temp = max(lvl[L_TEMP], 1e-3)
            w = np.zeros(k)
            tot = 0.0
            for j in range(k):
                w[j] = math.exp((vals[j] - best) / temp)
                tot += w[j]
            r = urand(np.uint64(seed), n, dec, 1000 + me) * tot
            choice = k - 1
            acc = 0.0
            for j in range(k):
                acc += w[j]
                if r <= acc:
                    choice = j
                    break
            plan = kinds[choice]
            mem_i[M_PLAN + me] = plan
            mem_i[M_PARTNER + me] = partner[choice]
            mem_f[F_PLAN_T + 2 * me] = tgt_x[choice]
            mem_f[F_PLAN_T + 2 * me + 1] = tgt_y[choice]
            # error de puntería del nivel (fijo durante el plan)
            noise = lvl[L_AIM_NOISE] * math.pi / 180.0 * nrand(seed, n, dec, 2000 + me)
            c = math.cos(noise)
            s_ = math.sin(noise)
            ux, uy = ux_o[choice], uy_o[choice]
            mem_f[F_PLAN_U + 2 * me] = ux * c - uy * s_
            mem_f[F_PLAN_U + 2 * me + 1] = ux * s_ + uy * c
            mem_f[F_AIM + me] = noise
        mem_i[M_PLAN_UNTIL + me] = dec + int(lvl[L_COMMIT])
    if plan == P_NONE:
        plan = P_DRIBBLE
        mem_f[F_PLAN_U + 2 * me] = 1.0
        mem_f[F_PLAN_U + 2 * me + 1] = 0.0
    ux, uy = _plan_dir(me, plan, mem_i[M_PARTNER + me], px, py, bx, by, mem_f)
    if plan != P_DRIBBLE:
        # error de puntería del nivel: la dirección exacta rotada por el ángulo sorteado al planear
        a = mem_f[F_AIM + me]
        ux, uy = ux * math.cos(a) - uy * math.sin(a), ux * math.sin(a) + uy * math.cos(a)
    tb = int(min(max(t_me, 0.0), HORIZON))
    cbx = traj[tb, 0]
    cby = traj[tb, 1]
    if plan == P_DRIBBLE:
        ux, uy = dribble_dir(table, bx, by, ux, uy, px[me], py[me], opp_x, opp_y, opp_vx, opp_vy, n_opp, lvl[L_REACT])
        mem_f[F_PLAN_U + 2 * me] = ux
        mem_f[F_PLAN_U + 2 * me + 1] = uy
        # alineado detrás de la pelota: empujarla en la dirección de conducción
        return approach_move(px[me], py[me], pvx[me], pvy[me], bx, by, bvx, bvy, cbx, cby, ux, uy, spd, True)
    # la patada se suma a la velocidad que trae la pelota: apuntar con la normal que la compensa
    q = Q_B ** max(t_me, 0.0)
    ux, uy = kick_normal(ux, uy, bvx * q, bvy * q)
    mv = approach_move(px[me], py[me], pvx[me], pvy[me], bx, by, bvx, bvy, cbx, cby, ux, uy, spd, False)
    tol_deg = lvl[L_AIM_TOL]
    if plan == P_SHOT:
        # al tirar, la tolerancia no puede exceder el ángulo libre (visto desde la pelota) hasta el palo más cercano
        ta = math.atan2(uy, ux)
        a1 = math.atan2(GOAL_HH - 9.0 - by, GOAL_X - bx)
        a2 = math.atan2(-GOAL_HH + 9.0 - by, GOAL_X - bx)
        free = min(abs(a1 - ta), abs(ta - a2)) * 180.0 / math.pi
        tol_deg = max(2.0, min(tol_deg, free))
    tol = math.cos(tol_deg * math.pi / 180.0)
    if not kick_cancel[me] and kick_ready(px[me], py[me], pvx[me], pvy[me], bx, by, bvx, bvy, ux, uy, tol, mv):
        cdx = bx - px[me]
        cdy = by - py[me]
        cd = math.hypot(cdx, cdy) + 1e-9
        if not crosses_own_goal(bx, by, cdx / cd, cdy / cd, 60.0):
            mem_i[M_PLAN + me] = P_NONE
            return mv + 9
    return mv


@njit(cache=True)
def _restart_taker(n, team, seed, dec, table, me, px, py, pvx, pvy, bx, by, bvx, bvy, own_idx, n_own, opp_x,
                   opp_y, opp_vx, opp_vy, n_opp, kind, ticks, ko, kick_cancel, lvl, sty, mem_i, mem_f):
    """Ejecutor de saque (lateral, córner, saque de arco o saque inicial): espera apoyos y patea."""
    spd = lvl[L_SPEED]
    mem_i[M_WAIT] += 1
    wait = mem_i[M_WAIT]
    # impulso real de cada saque (contrato): córner ×1,98, saque de arco ×2,71
    mult = 1.0
    min_wait = 20
    if kind == 2:
        mult = 1.98
        min_wait = 45
    elif kind == 3:
        mult = 2.71
        min_wait = 62     # el disco del punto bloquea 180 ticks
    elif kind == 1:
        min_wait = 25
    if ko:
        min_wait = 15
    plan_ok = mem_i[M_PLAN + me] != P_NONE and dec < mem_i[M_PLAN_UNTIL + me]
    if not plan_ok:
        vals = np.zeros(64)
        ux_o = np.zeros(64)
        uy_o = np.zeros(64)
        kinds = np.zeros(64, dtype=np.int64)
        tgt_x = np.zeros(64)
        tgt_y = np.zeros(64)
        partner = np.full(64, -1, dtype=np.int64)
        k = _options(n, team, seed, dec, table, me, px, py, bx, by, 0.0, 0.0, own_idx, n_own, opp_x, opp_y,
                     opp_vx, opp_vy, n_opp, lvl, sty, mult, vals, ux_o, uy_o, kinds, tgt_x, tgt_y, partner)
        best = -1
        best_v = -1e18
        for j in range(k):
            v = vals[j]
            if kinds[j] == 3:
                v *= 0.6 if not ko else 1.0   # conducir un saque parado vale menos (salvo saque inicial)
            if kinds[j] == 5:
                v *= 0.5
            v += 0.05 * nrand(seed, n, dec, 3000 + j) * lvl[L_TEMP]
            if v > best_v:
                best_v = v
                best = j
        if best < 0:
            ux, uy = 1.0, 0.0
            plan_kind = P_DRIBBLE
        else:
            ux, uy = ux_o[best], uy_o[best]
            plan_kind = kinds[best]
        mem_i[M_PLAN + me] = plan_kind
        mem_i[M_PLAN_UNTIL + me] = dec + 6
        mem_f[F_PLAN_U + 2 * me] = ux
        mem_f[F_PLAN_U + 2 * me + 1] = uy
    ux = mem_f[F_PLAN_U + 2 * me]
    uy = mem_f[F_PLAN_U + 2 * me + 1]
    if kind == 1:
        # lateral: la pelota está afuera (|y| = 688); se patea hacia adentro de la cancha
        inward = -1.0 if by > 0.0 else 1.0
        if uy * inward < 0.25:
            uy = inward * 0.35
            nrm = math.hypot(ux, uy)
            ux /= nrm
            uy /= nrm
    ax, ay, sp = approach_point(px[me], py[me], bx, by, ux, uy, pvx[me], pvy[me])
    ready = wait * 3 >= min_wait
    if not ready:
        # acercarse sin tocar: quedarse a 40 px del punto de contacto
        ax = bx - ux * (R_P + R_B + 28.0)
        ay = by - uy * (R_P + R_B + 28.0)
        sp = 0.0
    mv = move_action(px[me], py[me], pvx[me], pvy[me], ax, ay, sp, spd, False)
    tol = math.cos(lvl[L_AIM_TOL] * math.pi / 180.0)
    if ready and not kick_cancel[me] and kick_ready(px[me], py[me], pvx[me], pvy[me], bx, by, 0.0, 0.0, ux, uy, tol, mv):
        mem_i[M_PLAN + me] = P_NONE
        return mv + 9
    return mv
