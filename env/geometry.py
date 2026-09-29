"""Distancias a paredes (raycasts) para la observación universal.

El bot "ve" la forma de cualquier mapa midiendo, en 8 direcciones, cuánto puede avanzar un disco antes
de chocar contra la geometría fija del estadio: segmentos (los curvos se discretizan en tramos rectos),
planos y discos fijos (postes). Se respetan las reglas de colisión de HaxBall (cGroup/cMask), así que
los rayos del jugador ven lo que frena a un jugador y los de la pelota lo que frena a la pelota.

Simplificaciones: las paredes se tratan como de dos caras (se ignora bias) y los discos que se mueven
(invMass > 0) no cuentan como pared.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

from sim.stadium import BLUE, PLAYER_MASK, RED, Stadium

N_RAYS = 8
RAY_MAX = 400.0  # distancia máxima que se mide (unidades del mapa)
# direcciones en coordenadas HaxBall (y hacia abajo), empezando hacia +x y girando
RAY_DIRS = np.array([[math.cos(2 * math.pi * k / N_RAYS), math.sin(2 * math.pi * k / N_RAYS)]
                     for k in range(N_RAYS)])
# espejo en x de las direcciones (para el marco propio del azul): k -> (N/2 - k) mod N
RAY_MIRROR = np.array([(N_RAYS // 2 - k) % N_RAYS for k in range(N_RAYS)])


def _arc_points(st: Stadium, s: int, n: int = 12) -> np.ndarray:
    """Puntos de un segmento curvo, del lado donde está el arco (misma prueba que el motor)."""
    c, r = st.s_center[s], st.s_radius[s]
    p0, p1 = st.s_p0[s], st.s_p1[s]
    a0 = math.atan2(p0[1] - c[1], p0[0] - c[0])
    a1 = math.atan2(p1[1] - c[1], p1[0] - c[0])
    best = None
    for direction in (1.0, -1.0):
        span = (a1 - a0) * direction % (2 * math.pi)
        mid = a0 + direction * span / 2
        m = np.array([math.cos(mid), math.sin(mid)])
        in0 = m @ st.s_t0[s] > 0
        in1 = m @ st.s_t1[s] > 0
        inside = (in0 and in1) if st.s_curved[s] == 1 else (in0 or in1)
        if inside:
            best = (direction, span)
            break
    if best is None:
        return np.array([p0, p1])
    direction, span = best
    ang = a0 + direction * np.linspace(0.0, span, n + 1)
    return c[None] + r * np.stack([np.cos(ang), np.sin(ang)], axis=1)


def obstacles(st: Stadium, group: int, mask: int):
    """Geometría fija con la que choca un disco de (group, mask): (seg_a, seg_b, circ_c, circ_r, pl_n, pl_d)."""
    def hits(g, m):
        return (group & m) != 0 and (mask & g) != 0

    a, b = [], []
    for s in range(len(st.s_p0)):
        if not hits(st.s_group[s], st.s_mask[s]):
            continue
        if st.s_curved[s] == 0:
            a.append(st.s_p0[s])
            b.append(st.s_p1[s])
        else:
            pts = _arc_points(st, s)
            a.extend(pts[:-1])
            b.extend(pts[1:])
    cc, cr = [], []
    for v in range(len(st.v_pos)):  # vértices: discos de radio 0
        if hits(st.v_group[v], st.v_mask[v]):
            cc.append(st.v_pos[v])
            cr.append(0.0)
    for d in range(len(st.d_pos)):
        if st.d_invmass[d] == 0.0 and hits(st.d_group[d], st.d_mask[d]):
            cc.append(st.d_pos[d])
            cr.append(st.d_radius[d])
    pn, pd = [], []
    for p in range(len(st.p_normal)):
        if hits(st.p_group[p], st.p_mask[p]):
            pn.append(st.p_normal[p])
            pd.append(st.p_dist[p])
    f = lambda x, sh: np.array(x, dtype=np.float64).reshape(sh)  # noqa: E731
    return (f(a, (-1, 2)), f(b, (-1, 2)), f(cc, (-1, 2)), f(cr, (-1,)), f(pn, (-1, 2)), f(pd, (-1,)))


class StadiumRays:
    """Obstáculos precalculados de un estadio para pelota, jugador rojo y jugador azul."""

    def __init__(self, st: Stadium):
        self.ball = obstacles(st, st.ball["cGroup"], st.ball["cMask"])
        self.red = obstacles(st, RED, PLAYER_MASK)
        self.blue = obstacles(st, BLUE, PLAYER_MASK)
        self.ball_r = float(st.ball["radius"])
        self.player_r = float(st.player["radius"])

    def cast(self, origins: np.ndarray, which: str, sign: np.ndarray | None = None) -> np.ndarray:
        """origins (M,2) en coordenadas mundo -> distancias (M, N_RAYS) en [0, RAY_MAX], en el marco propio
        (para sign=-1 los rayos se ordenan como si el mapa estuviera espejado en x)."""
        obs = getattr(self, which)
        r = self.ball_r if which == "ball" else self.player_r
        out = _cast(np.ascontiguousarray(origins, dtype=np.float64), RAY_DIRS, *obs, r, RAY_MAX)
        if sign is not None:
            blue = sign < 0
            out[blue] = out[blue][:, RAY_MIRROR]
        return out


@njit(cache=True, parallel=True)
def _cast(orig, dirs, seg_a, seg_b, circ_c, circ_r, pl_n, pl_d, radius, maxd):
    M = orig.shape[0]
    R = dirs.shape[0]
    out = np.full((M, R), maxd)
    for m in prange(M):
        ox = orig[m, 0]
        oy = orig[m, 1]
        for k in range(R):
            dx = dirs[k, 0]
            dy = dirs[k, 1]
            best = maxd + radius
            # segmentos: intersección rayo-segmento
            for s in range(seg_a.shape[0]):
                ax = seg_a[s, 0]
                ay = seg_a[s, 1]
                ex = seg_b[s, 0] - ax
                ey = seg_b[s, 1] - ay
                den = dx * ey - dy * ex
                if abs(den) < 1e-12:
                    continue
                wx = ax - ox
                wy = ay - oy
                t = (wx * ey - wy * ex) / den
                u = (wx * dy - wy * dx) / den
                if t > 0.0 and 0.0 <= u <= 1.0 and t < best:
                    best = t
            # discos fijos y vértices: rayo contra círculo inflado por el radio del disco
            for c in range(circ_c.shape[0]):
                fx = ox - circ_c[c, 0]
                fy = oy - circ_c[c, 1]
                rr = circ_r[c] + radius
                b = fx * dx + fy * dy
                cc = fx * fx + fy * fy - rr * rr
                disc = b * b - cc
                if disc < 0.0:
                    continue
                t = -b - math.sqrt(disc)
                if t > 0.0 and t + radius < best:
                    best = t + radius  # se resta el radio al final como con las paredes
            # planos: el disco queda del lado de la normal (dot(p, n) >= dist + radio)
            for p in range(pl_n.shape[0]):
                nx = pl_n[p, 0]
                ny = pl_n[p, 1]
                vn = dx * nx + dy * ny
                if vn >= 0.0:
                    continue
                t = (pl_d[p] - (ox * nx + oy * ny)) / vn
                if t > 0.0 and t < best:
                    best = t
            d = best - radius
            out[m, k] = min(max(d, 0.0), maxd)
    return out
