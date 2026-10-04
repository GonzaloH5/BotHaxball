"""Geometría y cinemática exactas de RS ONE para RS-Pro (marco de equipo: ataca hacia +x).

* Jugador: cada tick v += a·u; x += v; v *= 0,96 (a = 0,12; pateando 0,07). Con aceleración
  constante hacia el objetivo, el desplazamiento tras t ticks partiendo de velocidad u0 es
  s(t) = u0(1−q^t)/(1−q) + a/(1−q)·(t − q(1−q^t)/(1−q)), q = 0,96 (verificado contra el motor).
* Pelota: x += v; v *= 0,99 (+ gravedad de saques) → recorrido v0(1−0,99^t)/0,01; alcance de una
  patada ≈ 614 px (6,14 px/tick); córner ×1,98 y saque de arco ×2,71 a los 2–3 ticks.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit

Q_P = 0.96
A_P = 0.12
A_KICK = 0.07
Q_B = 0.99
KICK_SPEED = 5.85 * 1.05
R_P = 15.0
R_B = 8.325
REACH = R_P + R_B + 4.0        # distancia de centros para poder patear
LINE_W, LINE_H = 1150.0, 670.0
GOAL_X, GOAL_HH = 1162.0, 124.0
POST_Y = 124.0
HORIZON = 120                  # ticks de predicción de pelota

# tabla de tiempo de llegada 1D: T[d, u0] (ticks), d en pasos de D_STEP, u0 en [-3, 3]
D_STEP = 4.0
D_MAX = 3200.0
U_MIN, U_MAX, U_STEP = -3.0, 3.0, 0.1


def _build_ttr_table():
    nd = int(D_MAX / D_STEP) + 1
    nu = int(round((U_MAX - U_MIN) / U_STEP)) + 1
    table = np.zeros((nd, nu), dtype=np.float64)
    q, a = Q_P, A_P
    t = np.arange(0, 1201, dtype=np.float64)
    for j in range(nu):
        u0 = U_MIN + j * U_STEP
        s = u0 * (1 - q ** t) / (1 - q) + a / (1 - q) * (t - q * (1 - q ** t) / (1 - q))
        # primera vez que s ≥ d (s es creciente a partir de su mínimo)
        smax = np.maximum.accumulate(s)
        for i in range(nd):
            d = i * D_STEP
            k = int(np.searchsorted(smax, d))
            if k >= len(t):
                table[i, j] = 1200.0
            elif k == 0:
                table[i, j] = 0.0
            else:
                # interpolación lineal entre k-1 y k
                s0, s1 = smax[k - 1], smax[k]
                frac = (d - s0) / (s1 - s0) if s1 > s0 else 0.0
                table[i, j] = (k - 1) + frac
    return table


TTR_TABLE = _build_ttr_table()


@njit(cache=True)
def ttr1d(table, d, u0):
    """Ticks para recorrer d (px) acelerando a fondo, con velocidad inicial u0 a lo largo."""
    if d <= 0.0:
        return 0.0
    fi = d / D_STEP
    if fi >= table.shape[0] - 1:
        fi = table.shape[0] - 1.0001
    fj = (u0 - U_MIN) / U_STEP
    if fj < 0.0:
        fj = 0.0
    if fj > table.shape[1] - 1.0001:
        fj = table.shape[1] - 1.0001
    i = int(fi)
    j = int(fj)
    di = fi - i
    dj = fj - j
    return ((table[i, j] * (1 - di) + table[i + 1, j] * di) * (1 - dj)
            + (table[i, j + 1] * (1 - di) + table[i + 1, j + 1] * di) * dj)


@njit(cache=True)
def ttr(table, px, py, vx, vy, tx, ty, slack):
    """Tiempo (ticks) para que un jugador en (px,py) con velocidad (vx,vy) llegue a `slack` px de (tx,ty).

    Descompone la velocidad en la línea de visión; la deriva perpendicular (que decae ×0,96 por
    tick, hasta 24·|w| px) se suma a la distancia efectiva con una iteración.
    """
    dx = tx - px
    dy = ty - py
    d = math.sqrt(dx * dx + dy * dy)
    if d <= slack:
        return 0.0
    ux = dx / d
    uy = dy / d
    along = vx * ux + vy * uy
    perp = -vx * uy + vy * ux
    t = ttr1d(table, d - slack, along)
    drift = abs(perp) * (1.0 - Q_P ** t) / (1.0 - Q_P)
    eff = math.sqrt((d - slack) * (d - slack) + drift * drift)
    return ttr1d(table, eff, along)


@njit(cache=True)
def ball_travel_ticks(dist, v0):
    """Ticks para que una pelota con rapidez v0 (sin gravedad) recorra dist; 1e9 si no llega."""
    if dist <= 0.0:
        return 0.0
    rng = v0 / (1.0 - Q_B)
    if dist >= rng * 0.999:
        return 1e9
    # dist = v0 (1 - q^t)/(1-q) → q^t = 1 - dist (1-q)/v0
    return math.log(1.0 - dist * (1.0 - Q_B) / v0) / math.log(Q_B)


@njit(cache=True)
def predict_ball(bx, by, vx, vy, gx, gy, gravity_ticks, decay, out):
    """Trayectoria de la pelota sola (amortiguamiento + gravedad de saque), out[t] = posición en el tick t.

    Devuelve el tick en que sale de la cancha o cruza la línea de gol (o HORIZON). Los postes y la
    red no se modelan: el último tramo se usa sólo para decidir, nunca para el árbitro.
    """
    x = bx
    y = by
    ux = vx
    uy = vy
    g0 = gx
    g1 = gy
    gl = gravity_ticks
    end = out.shape[0]
    out[0, 0] = x
    out[0, 1] = y
    for t in range(1, out.shape[0]):
        x += ux
        y += uy
        if gl > 0:
            ux += g0
            uy += g1
            g0 *= decay
            g1 *= decay
            gl -= 1
        ux *= Q_B
        uy *= Q_B
        out[t, 0] = x
        out[t, 1] = y
        if end == out.shape[0] and (abs(x) > LINE_W + R_B or abs(y) > LINE_H + R_B):
            end = t
    return end


@njit(cache=True)
def seg_dist(px, py, ax, ay, bx, by):
    """Distancia de un punto al segmento AB y parámetro de proyección en [0, 1]."""
    dx = bx - ax
    dy = by - ay
    l2 = dx * dx + dy * dy
    if l2 <= 1e-12:
        return math.hypot(px - ax, py - ay), 0.0
    u = ((px - ax) * dx + (py - ay) * dy) / l2
    if u < 0.0:
        u = 0.0
    elif u > 1.0:
        u = 1.0
    cx = ax + u * dx
    cy = ay + u * dy
    return math.hypot(px - cx, py - cy), u


# ---------------------------------------------------------------------- RNG determinista
@njit(cache=True, inline="always")
def _mix(z):
    z = (z + np.uint64(0x9E3779B97F4A7C15))
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


@njit(cache=True)
def urand(seed, a, b, c):
    """Uniforme [0,1) determinista a partir de (semilla, partido, tick, canal): no depende de hilos."""
    z = _mix(np.uint64(seed) ^ _mix(np.uint64(a) * np.uint64(0x100000001B3) ^ _mix(np.uint64(b) ^ _mix(np.uint64(c)))))
    return (z >> np.uint64(11)) * (1.0 / 9007199254740992.0)


@njit(cache=True)
def nrand(seed, a, b, c):
    """Normal estándar (Box-Muller) determinista."""
    u1 = urand(seed, a, b, 2 * c)
    u2 = urand(seed, a, b, 2 * c + 1)
    if u1 < 1e-12:
        u1 = 1e-12
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
