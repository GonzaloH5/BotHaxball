"""Motor de física de HaxBall, en lote (N partidos a la vez) y compilado con numba.

Orden por tick (igual que HaxBall):
  1. inputs de jugadores (patada + aceleración)
  2. integración: pos += vel; vel *= damping
  3. colisiones disco-disco, disco-plano, disco-segmento, disco-vértice
  4. estado del partido (saque inicial / detección de gol)

Layout de discos por partido: [pelota, discos del estadio..., jugadores...]
Acción por jugador: entero 0..17 -> mover = a % 9 (0 quieto, 1..8 direcciones), patear = a // 9.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

from .stadium import BLUEKO, PLAYER_MASK, REDKO, Stadium, load_stadium

N_ACTIONS = 18
# dirección por acción de movimiento (coordenadas HaxBall, y hacia abajo)
_S = 1.0
MOVE_DIRS = np.array([
    [0, 0],            # 0 quieto
    [0, -1],           # 1 arriba
    [1, -1],           # 2 arriba-derecha
    [1, 0],            # 3 derecha
    [1, 1],            # 4 abajo-derecha
    [0, 1],            # 5 abajo
    [-1, 1],           # 6 abajo-izquierda
    [-1, 0],           # 7 izquierda
    [-1, -1],          # 8 arriba-izquierda
], dtype=np.float64)
MOVE_UNIT = MOVE_DIRS / np.maximum(np.linalg.norm(MOVE_DIRS, axis=1, keepdims=True), 1e-12)

KICK_REACH = 4.0


@njit(cache=True, inline="always")
def _bounce(vx, vy, nx, ny, b):
    vn = vx * nx + vy * ny
    if vn < 0.0:
        f = vn * (1.0 + b)
        vx -= nx * f
        vy -= ny * f
    return vx, vy


@njit(cache=True)
def _collide_env(pos, vel, mask, radius, invmass, bcoef, group,
                 v_pos, v_bcoef, v_group, v_mask,
                 s_p0, s_p1, s_curved, s_center, s_radius, s_t0, s_t1, s_bias, s_bcoef, s_group, s_mask,
                 p_normal, p_dist, p_bcoef, p_group, p_mask):
    K = pos.shape[0]
    for i in range(K):
        gi = group[i]
        mi = mask[i]
        ri = radius[i]
        bi = bcoef[i]
        # disco - disco
        for j in range(i + 1, K):
            if (gi & mask[j]) == 0 or (mi & group[j]) == 0:
                continue
            ims = invmass[i] + invmass[j]
            if ims == 0.0:
                continue
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            rs = ri + radius[j]
            d2 = dx * dx + dy * dy
            if d2 > rs * rs or d2 == 0.0:
                continue
            d = math.sqrt(d2)
            nx = dx / d
            ny = dy / d
            fa = invmass[i] / ims
            pen = rs - d
            pos[i, 0] += nx * pen * fa
            pos[i, 1] += ny * pen * fa
            pos[j, 0] -= nx * pen * (1.0 - fa)
            pos[j, 1] -= ny * pen * (1.0 - fa)
            rvn = (vel[i, 0] - vel[j, 0]) * nx + (vel[i, 1] - vel[j, 1]) * ny
            if rvn < 0.0:
                f = rvn * (1.0 + bi * bcoef[j])
                vel[i, 0] -= nx * f * fa
                vel[i, 1] -= ny * f * fa
                vel[j, 0] += nx * f * (1.0 - fa)
                vel[j, 1] += ny * f * (1.0 - fa)
        if invmass[i] == 0.0:
            continue
        # disco - plano
        for p in range(p_normal.shape[0]):
            if (gi & p_mask[p]) == 0 or (mi & p_group[p]) == 0:
                continue
            nx = p_normal[p, 0]
            ny = p_normal[p, 1]
            dist = p_dist[p] - (pos[i, 0] * nx + pos[i, 1] * ny) + ri
            if dist > 0.0:
                pos[i, 0] += nx * dist
                pos[i, 1] += ny * dist
                vel[i, 0], vel[i, 1] = _bounce(vel[i, 0], vel[i, 1], nx, ny, bi * p_bcoef[p])
        # disco - segmento
        for s in range(s_p0.shape[0]):
            if (gi & s_mask[s]) == 0 or (mi & s_group[s]) == 0:
                continue
            px = pos[i, 0]
            py = pos[i, 1]
            if s_curved[s] == 0:
                sx = s_p1[s, 0] - s_p0[s, 0]
                sy = s_p1[s, 1] - s_p0[s, 1]
                if sx * (px - s_p0[s, 0]) + sy * (py - s_p0[s, 1]) <= 0.0:
                    continue
                if sx * (px - s_p1[s, 0]) + sy * (py - s_p1[s, 1]) >= 0.0:
                    continue
                sl = math.sqrt(sx * sx + sy * sy)
                nx = -sy / sl
                ny = sx / sl
                dist = nx * (px - s_p1[s, 0]) + ny * (py - s_p1[s, 1])
            else:
                cx = px - s_center[s, 0]
                cy = py - s_center[s, 1]
                in0 = cx * s_t0[s, 0] + cy * s_t0[s, 1] > 0.0
                in1 = cx * s_t1[s, 0] + cy * s_t1[s, 1] > 0.0
                inside = (in0 and in1) if s_curved[s] == 1 else (in0 or in1)
                if not inside:
                    continue
                cl = math.sqrt(cx * cx + cy * cy)
                if cl == 0.0:
                    continue
                nx = cx / cl
                ny = cy / cl
                dist = cl - s_radius[s]
            bias = s_bias[s]
            if bias == 0.0:
                if dist < 0.0:
                    dist = -dist
                    nx = -nx
                    ny = -ny
            elif bias < 0.0:
                bias = -bias
                dist = -dist
                nx = -nx
                ny = -ny
            if dist < -bias:
                continue
            if dist < ri:
                pen = ri - dist
                pos[i, 0] += nx * pen
                pos[i, 1] += ny * pen
                vel[i, 0], vel[i, 1] = _bounce(vel[i, 0], vel[i, 1], nx, ny, bi * s_bcoef[s])
        # disco - vértice
        for v in range(v_pos.shape[0]):
            if (gi & v_mask[v]) == 0 or (mi & v_group[v]) == 0:
                continue
            dx = pos[i, 0] - v_pos[v, 0]
            dy = pos[i, 1] - v_pos[v, 1]
            d2 = dx * dx + dy * dy
            if d2 > ri * ri or d2 == 0.0:
                continue
            d = math.sqrt(d2)
            nx = dx / d
            ny = dy / d
            pos[i, 0] += nx * (ri - d)
            pos[i, 1] += ny * (ri - d)
            vel[i, 0], vel[i, 1] = _bounce(vel[i, 0], vel[i, 1], nx, ny, bi * v_bcoef[v])


@njit(cache=True, inline="always")
def _cross(ax, ay, bx, by):
    return ax * by - ay * bx


# Parámetros del powershot (script de la sala, ball-physics.js). Tiempos en ticks (60/s).
PS_CHARGE, PS_POWER_INV, PS_BASE_INV, PS_CURVE, PS_INV_RESET, PS_GRAV, PS_MAX_LAT, PS_EVERY = range(8)


@njit(cache=True, parallel=True)
def step_batch(pos, vel, mask, kick_cancel, kickoff, kickoff_team, actions, move_unit,
               radius, inv_env, bcoef, damping, group, first_player, player_team,
               p_acc, p_kacc, p_kdamp, p_kstr, p_kback,
               v_pos, v_bcoef, v_group, v_mask,
               s_p0, s_p1, s_curved, s_center, s_radius, s_t0, s_t1, s_bias, s_bcoef, s_group, s_mask,
               p_normal, p_dist, p_bcoef, p_group, p_mask,
               g_p0, g_p1, g_team,
               goal_out, touch_out, kick_out,
               ball_grav, ps_on, ps_par, ps_held, ps_charge, ps_comba, ps_grav_left, ps_inv_left,
               ps_gtick, ps_pending, ps_has_pending, ps_kick_out,
               frames=1, window_goal=None, window_kicked=None, last_touch=None):
    """Avanza frames ticks por partido con una única barrera paralela.

    goal_out[n]: +1 si anotó rojo, -1 si anotó azul, 0 nada.
    touch_out[n, p]: el jugador p tocó (o pateó) la pelota este tick.
    kick_out[n, p]: el jugador p pateó la pelota este tick.
    inv_env[n, k]: invMass por partido (la de la pelota cambia con el powershot).
    ball_grav[n]: gravedad de la pelota (curva del powershot).
    ps_*: estado del powershot (sólo si ps_on). ps_kick_out[n, p]: pateó un powershot.
    Los eventos anteriores describen el último tick. Los buffers opcionales window_*
    conservan el primer gol y todas las patadas; last_touch conserva el último equipo.
    Sin buffers y con frames=1 es la ruta original; reglas/lag llaman esa variante.
    """
    N = pos.shape[0]
    K = pos.shape[1]
    P = actions.shape[1]
    for n in prange(N):
        if window_goal is not None:
            window_goal[n] = 0
        if window_kicked is not None:
            for p in range(P):
                window_kicked[n, p] = False
        for _ in range(frames):
            goal_out[n] = 0
            inv = inv_env[n]
            # 1. inputs
            for p in range(P):
                k = first_player + p
                a = actions[n, p]
                kick_pressed = a >= 9
                if not kick_pressed:
                    kick_cancel[n, p] = False
                is_kicking = kick_pressed and not kick_cancel[n, p]
                touch_out[n, p] = False
                kick_out[n, p] = False
                ps_kick_out[n, p] = False
                # la pelota es el disco 0 (único con flag KICK)
                dx = pos[n, 0, 0] - pos[n, k, 0]
                dy = pos[n, 0, 1] - pos[n, k, 1]
                d = math.sqrt(dx * dx + dy * dy)
                if d - radius[k] - radius[0] < KICK_REACH:
                    touch_out[n, p] = True
                    if is_kicking and d > 0.0:
                        nx = dx / d
                        ny = dy / d
                        # HaxBall: impulso = kickStrength * dirección * invMass de la pelota
                        vel[n, 0, 0] += nx * p_kstr * inv[0]
                        vel[n, 0, 1] += ny * p_kstr * inv[0]
                        vel[n, k, 0] -= nx * p_kback * inv[k]
                        vel[n, k, 1] -= ny * p_kback * inv[k]
                        kick_cancel[n, p] = True
                        kick_out[n, p] = True
                        if ps_on and ps_comba[n]:
                            # curva: gravedad perpendicular al tiro, proporcional a la velocidad
                            # lateral del jugador respecto de la pelota (setDiscProps del script)
                            lat = nx * vel[n, k, 1] - ny * vel[n, k, 0]
                            ci = min(1.0, max(-1.0, lat / ps_par[PS_MAX_LAT]))
                            sx = vel[n, 0, 0]
                            sy = vel[n, 0, 1]
                            sp = math.sqrt(sx * sx + sy * sy)
                            gx_ = 0.0
                            gy_ = 0.0
                            if sp > 0.1:
                                gx_ = sy / sp * ci * ps_par[PS_GRAV]
                                gy_ = -sx / sp * ci * ps_par[PS_GRAV]
                            ps_pending[n, 0] = gx_
                            ps_pending[n, 1] = gy_
                            ps_has_pending[n] = True
                            ps_grav_left[n] = int(ps_par[PS_CURVE])
                            ps_inv_left[n] = int(ps_par[PS_INV_RESET])
                            ps_comba[n] = False
                            ps_held[n] = -1
                            ps_charge[n] = -1
                            ps_kick_out[n, p] = True
                is_kicking = kick_pressed and not kick_cancel[n, p]
                acc = p_kacc if is_kicking else p_acc
                m = a % 9
                vel[n, k, 0] += move_unit[m, 0] * acc
                vel[n, k, 1] += move_unit[m, 1] * acc
            # 2. integración: pos += v; v = (v + gravedad) * damping
            bx0 = pos[n, 0, 0]
            by0 = pos[n, 0, 1]
            for k in range(K):
                pos[n, k, 0] += vel[n, k, 0]
                pos[n, k, 1] += vel[n, k, 1]
                dmp = damping[k]
                if k >= first_player:
                    p = k - first_player
                    if actions[n, p] >= 9 and not kick_cancel[n, p]:
                        dmp = p_kdamp
                if k == 0:
                    vel[n, 0, 0] += ball_grav[n, 0]
                    vel[n, 0, 1] += ball_grav[n, 1]
                vel[n, k, 0] *= dmp
                vel[n, k, 1] *= dmp
            # 3. colisiones
            _collide_env(pos[n], vel[n], mask[n], radius, inv, bcoef, group,
                         v_pos, v_bcoef, v_group, v_mask,
                         s_p0, s_p1, s_curved, s_center, s_radius, s_t0, s_t1, s_bias, s_bcoef, s_group, s_mask,
                         p_normal, p_dist, p_bcoef, p_group, p_mask)
            # 4. estado
            if kickoff[n]:
                if vel[n, 0, 0] != 0.0 or vel[n, 0, 1] != 0.0:
                    kickoff[n] = False
                    for p in range(P):
                        mask[n, first_player + p] = PLAYER_MASK
            else:
                qx = pos[n, 0, 0]
                qy = pos[n, 0, 1]
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
                        # entró en el arco de g_team -> anota el otro equipo
                        goal_out[n] = -1 if g_team[g] == 0 else 1
                        break
            # 5. script de la sala (onGameTick), después de la física
            if ps_on:
                ps_gtick[n] += 1
                if ps_has_pending[n]:  # runAfterGameTick: la curva arranca en el tick siguiente al tiro
                    ball_grav[n, 0] = ps_pending[n, 0]
                    ball_grav[n, 1] = ps_pending[n, 1]
                    ps_has_pending[n] = False
                elif ps_grav_left[n] > 0:
                    ps_grav_left[n] -= 1
                    if ps_grav_left[n] == 0:
                        ball_grav[n, 0] = 0.0
                        ball_grav[n, 1] = 0.0
                if ps_inv_left[n] > 0:
                    ps_inv_left[n] -= 1
                    if ps_inv_left[n] == 0:
                        inv[0] = ps_par[PS_BASE_INV]
                if ps_charge[n] > 0:
                    ps_charge[n] -= 1
                    if ps_charge[n] == 0:
                        # el timer dispara aunque ya no la tenga: la pelota queda cargada
                        ps_comba[n] = True
                        inv[0] = ps_par[PS_POWER_INV]
                        ps_charge[n] = -1
                if ps_gtick[n] % int(ps_par[PS_EVERY]) == 0:
                    holder = -1
                    for p in range(P):  # el script se queda con el ÚLTIMO jugador que la toca
                        k = first_player + p
                        dx = pos[n, 0, 0] - pos[n, k, 0]
                        dy = pos[n, 0, 1] - pos[n, k, 1]
                        if math.sqrt(dx * dx + dy * dy) <= radius[k] + radius[0] + 0.1:
                            holder = p
                    if holder >= 0:
                        if ps_held[n] < 0:
                            ps_held[n] = holder
                            ps_comba[n] = False
                            ps_charge[n] = int(ps_par[PS_CHARGE])
                    elif ps_held[n] >= 0:
                        # la perdió: se cancela la carga (ojo: si ya estaba cargada, invMass NO vuelve)
                        ps_held[n] = -1
                        ps_comba[n] = False
                        ps_charge[n] = -1
            if window_goal is not None and window_goal[n] == 0:
                window_goal[n] = goal_out[n]
            if window_kicked is not None:
                for p in range(P):
                    window_kicked[n, p] |= kick_out[n, p]
            if last_touch is not None:
                for p in range(P):
                    if touch_out[n, p]:
                        last_touch[n] = player_team[p]


class BatchSim:
    """N partidos simultáneos de `n_red` vs `n_blue` en un estadio."""

    def __init__(self, n_envs: int, n_red: int = 1, n_blue: int = 1,
                 stadium: Stadium | str = "classic", seed: int | None = None,
                 powershot: bool | dict = False):
        st = load_stadium(stadium) if isinstance(stadium, str) else stadium
        self.st = st
        self.N = n_envs
        self.n_red = n_red
        self.n_blue = n_blue
        self.P = n_red + n_blue
        D = st.d_pos.shape[0]
        self.first_player = 1 + D
        self.K = 1 + D + self.P
        self.rng = np.random.default_rng(seed)
        pl, b = st.player, st.ball

        self.player_team = np.array([0] * n_red + [1] * n_blue, dtype=np.int64)
        self.radius = np.concatenate([[b["radius"]], st.d_radius, np.full(self.P, pl["radius"])])
        self.invmass = np.concatenate([[b["invMass"]], st.d_invmass, np.full(self.P, pl["invMass"])])
        self.bcoef = np.concatenate([[b["bCoef"]], st.d_bcoef, np.full(self.P, pl["bCoef"])])
        self.damping = np.concatenate([[b["damping"]], st.d_damping, np.full(self.P, pl["damping"])])
        from .stadium import BLUE, RED
        self.group = np.concatenate([[b["cGroup"]], st.d_group,
                                     [RED if t == 0 else BLUE for t in self.player_team]]).astype(np.int64)
        self.base_mask = np.concatenate([[b["cMask"]], st.d_mask, np.full(self.P, PLAYER_MASK)]).astype(np.int64)

        self.pos = np.zeros((n_envs, self.K, 2))
        self.vel = np.zeros((n_envs, self.K, 2))
        self.mask = np.tile(self.base_mask, (n_envs, 1))
        self.kick_cancel = np.zeros((n_envs, self.P), dtype=np.bool_)
        self.kickoff = np.zeros(n_envs, dtype=np.bool_)
        self.kickoff_team = np.zeros(n_envs, dtype=np.int64)
        self.goal = np.zeros(n_envs, dtype=np.int64)
        self.touch = np.zeros((n_envs, self.P), dtype=np.bool_)
        self.kicked = np.zeros((n_envs, self.P), dtype=np.bool_)
        # invMass por partido (la de la pelota cambia con el powershot) y gravedad de la pelota
        self.inv_env = np.tile(self.invmass, (n_envs, 1))
        self.ball_grav = np.zeros((n_envs, 2))
        # powershot + curva del script de la sala (ball-physics.js); tiempos en ticks
        self.ps_on = bool(powershot)
        cfg = dict(charge=96, power_inv=2.3, base_inv=float(b["invMass"]), curve=84, inv_reset=48,
                   grav=0.1, max_lat=3.0, every=30)
        if isinstance(powershot, dict):
            cfg.update(powershot)
        self.ps_cfg = cfg
        self.ps_par = np.array([cfg["charge"], cfg["power_inv"], cfg["base_inv"], cfg["curve"],
                                cfg["inv_reset"], cfg["grav"], cfg["max_lat"], cfg["every"]], dtype=np.float64)
        self.ps_held = np.full(n_envs, -1, dtype=np.int64)
        self.ps_charge = np.full(n_envs, -1, dtype=np.int64)
        self.ps_comba = np.zeros(n_envs, dtype=np.bool_)
        self.ps_grav_left = np.zeros(n_envs, dtype=np.int64)
        self.ps_inv_left = np.zeros(n_envs, dtype=np.int64)
        self.ps_gtick = np.zeros(n_envs, dtype=np.int64)
        self.ps_pending = np.zeros((n_envs, 2))
        self.ps_has_pending = np.zeros(n_envs, dtype=np.bool_)
        self.ps_kicked = np.zeros((n_envs, self.P), dtype=np.bool_)
        self.reset_all()

    def _reset_ball_state(self, idx) -> None:
        """Estado del script (powershot) y propiedades de la pelota como al iniciar el partido."""
        self.inv_env[idx, 0] = self.invmass[0]
        self.ball_grav[idx] = 0.0
        self.ps_held[idx] = -1
        self.ps_charge[idx] = -1
        self.ps_comba[idx] = False
        self.ps_grav_left[idx] = 0
        self.ps_inv_left[idx] = 0
        self.ps_gtick[idx] = 0
        self.ps_has_pending[idx] = False

    # ------------------------------------------------------------------ reset
    def spawn_positions(self) -> np.ndarray:
        """Posiciones de saque de HaxBall: x = ±spawnDistance, y = 0, +55, -55, +110..."""
        out = np.zeros((self.P, 2))
        counts = [0, 0]
        points = (self.st.red_spawn, self.st.blue_spawn)
        for p, t in enumerate(self.player_team):
            c = counts[t]
            pts = points[t]
            if pts is not None and c < len(pts):
                # el mapa define puntos de saque propios (HaxEleven, Real Futsal x7)
                out[p] = pts[c]
            else:
                y = 55.0 * ((c + 1) >> 1) * (1 if c % 2 == 1 else -1)  # 0, +55, -55, +110 (medido en salas reales)
                out[p] = (-self.st.spawn_distance if t == 0 else self.st.spawn_distance, y)
            counts[t] += 1
        return out

    def reset_kickoff(self, idx, kickoff_team=None) -> None:
        """Saque desde el centro (como en HaxBall) para los partidos `idx`."""
        idx = np.atleast_1d(np.asarray(idx))
        if len(idx) == 0:
            return
        fp = self.first_player
        self.pos[idx, 0] = 0.0
        self.vel[idx, 0] = 0.0
        self.pos[idx, 1:fp] = self.st.d_pos
        self.vel[idx, 1:fp] = 0.0
        self.pos[idx, fp:] = self.spawn_positions()
        self.vel[idx, fp:] = 0.0
        self.kick_cancel[idx] = False
        if kickoff_team is not None:
            self.kickoff_team[idx] = kickoff_team
        self.kickoff[idx] = True
        ko = np.where(self.kickoff_team[idx] == 0, REDKO, BLUEKO)
        self.mask[idx] = self.base_mask
        self.mask[idx, fp:] = PLAYER_MASK | ko[:, None]
        self._reset_ball_state(idx)

    def reset_random(self, idx) -> None:
        """Posiciones aleatorias (sin saque): diversidad de situaciones para entrenar."""
        idx = np.atleast_1d(np.asarray(idx))
        if len(idx) == 0:
            return
        n = len(idx)
        fp = self.first_player
        st = self.st
        W, H = st.field_half_w, st.field_half_h
        self.pos[idx, 1:fp] = st.d_pos
        self.vel[idx, 1:fp] = 0.0
        self.pos[idx, 0, 0] = self.rng.uniform(-0.85 * W, 0.85 * W, n)
        self.pos[idx, 0, 1] = self.rng.uniform(-0.85 * H, 0.85 * H, n)
        self.vel[idx, 0] = self.rng.normal(0, 1.5, (n, 2))
        for p in range(self.P):
            self.pos[idx, fp + p, 0] = self.rng.uniform(-W, W, n)
            self.pos[idx, fp + p, 1] = self.rng.uniform(-H, H, n)
        self.vel[idx, fp:] = self.rng.normal(0, 0.8, (n, self.P, 2))
        self.kick_cancel[idx] = False
        self.kickoff[idx] = False
        self.mask[idx] = self.base_mask
        self._reset_ball_state(idx)

    def reset_all(self) -> None:
        self.reset_kickoff(np.arange(self.N), kickoff_team=0)

    # ------------------------------------------------------------------ step
    def step(self, actions: np.ndarray) -> np.ndarray:
        """actions: (N, P) enteros 0..17. Devuelve goal (N,) con +1 rojo / -1 azul / 0."""
        step_batch(*self._step_args(actions))
        return self.goal

    def step_frames(self, actions, frames, last_touch):
        """Sólo sin callbacks/reglas por tick ni cambios de acción dentro de la ventana."""
        goal = np.empty(self.N, dtype=np.int64)
        kicked = np.empty((self.N, self.P), dtype=np.bool_)
        step_batch(*self._step_args(actions), frames, goal, kicked, last_touch)
        return goal, kicked

    def _step_args(self, actions):
        st, pl = self.st, self.st.player
        return (self.pos, self.vel, self.mask, self.kick_cancel, self.kickoff, self.kickoff_team,
                   np.ascontiguousarray(actions, dtype=np.int64), MOVE_UNIT,
                   self.radius, self.inv_env, self.bcoef, self.damping, self.group,
                   self.first_player, self.player_team,
                   float(pl["acceleration"]), float(pl["kickingAcceleration"]), float(pl["kickingDamping"]),
                   float(pl["kickStrength"]), float(pl["kickback"]),
                   st.v_pos, st.v_bcoef, st.v_group, st.v_mask,
                   st.s_p0, st.s_p1, st.s_curved, st.s_center, st.s_radius, st.s_t0, st.s_t1,
                   st.s_bias, st.s_bcoef, st.s_group, st.s_mask,
                   st.p_normal, st.p_dist, st.p_bcoef, st.p_group, st.p_mask,
                   st.g_p0, st.g_p1, st.g_team,
                   self.goal, self.touch, self.kicked,
                   self.ball_grav, self.ps_on, self.ps_par, self.ps_held, self.ps_charge, self.ps_comba,
                   self.ps_grav_left, self.ps_inv_left, self.ps_gtick, self.ps_pending,
                   self.ps_has_pending, self.ps_kicked)

    # vistas cómodas
    @property
    def ball_pos(self):
        return self.pos[:, 0]

    @property
    def ball_vel(self):
        return self.vel[:, 0]

    @property
    def player_pos(self):
        return self.pos[:, self.first_player:]

    @property
    def player_vel(self):
        return self.vel[:, self.first_player:]
