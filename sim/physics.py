"""Motor de física de HaxBall, en lote (N partidos a la vez) con triple backend de alto rendimiento:
  1. Numba JIT paralelo (`@njit(cache=True, parallel=True)`) cuando `numba` está instalado.
  2. Kernel nativo C99/OpenMP auto-compilado (`ctypes`) cuando `numba` no está disponible y hay compilador C.
  3. Fallback NumPy vectorizado con broadphase por lote (sin bucles Python por partido).

Orden por tick (igual que HaxBall):
  1. inputs de jugadores (patada + aceleración)
  2. integración: pos += vel; vel *= damping
  3. colisiones disco-disco, disco-plano, disco-segmento, disco-vértice
  4. estado del partido (saque inicial / detección de gol)

Layout de discos por partido: [pelota, discos del estadio..., jugadores...]
Acción por jugador: entero 0..17 -> mover = a % 9 (0 quieto, 1..8 direcciones), patear = a // 9.
"""
from __future__ import annotations

import ctypes
import hashlib
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

try:
    import numba as _numba_mod
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:  # pragma: no cover - fallback cuando numba no está instalado
    _numba_mod = None
    HAS_NUMBA = False

    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return lambda fn: fn

    prange = range

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
MOVE_UNIT = np.ascontiguousarray(
    MOVE_DIRS / np.maximum(np.linalg.norm(MOVE_DIRS, axis=1, keepdims=True), 1e-12),
    dtype=np.float64,
)

KICK_REACH = 4.0

# Parámetros del powershot (script de la sala, ball-physics.js). Tiempos en ticks (60/s).
PS_CHARGE, PS_POWER_INV, PS_BASE_INV, PS_CURVE, PS_INV_RESET, PS_GRAV, PS_MAX_LAT, PS_EVERY = range(8)


# ============================================================================ Kernel C99 / OpenMP opcional
_C_KERNEL_SOURCE = r"""
#include <math.h>
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#ifdef _OPENMP
#include <omp.h>
#endif

static inline void bounce_c(double *vx, double *vy, double nx, double ny, double b) {
    double vn = (*vx) * nx + (*vy) * ny;
    if (vn < 0.0) {
        double f = vn * (1.0 + b);
        *vx -= nx * f;
        *vy -= ny * f;
    }
}

void set_omp_threads_c(int n_threads) {
#ifdef _OPENMP
    if (n_threads > 0) {
        omp_set_num_threads(n_threads);
    }
#else
    (void)n_threads;
#endif
}

static inline void collide_env_c(
    int64_t K, int64_t V, int64_t S, int64_t PL,
    double *pos_n, double *vel_n, const int64_t *mask_n,
    const double *radius, const double *inv, const double *bcoef, const int64_t *group,
    const double *v_pos, const double *v_bcoef, const int64_t *v_group, const int64_t *v_mask,
    const double *s_p0, const double *s_p1, const int64_t *s_curved,
    const double *s_center, const double *s_radius, const double *s_t0, const double *s_t1,
    const double *s_bias, const double *s_bcoef, const int64_t *s_group, const int64_t *s_mask,
    const double *s_nx_pre, const double *s_ny_pre,
    const double *p_normal, const double *p_dist, const double *p_bcoef, const int64_t *p_group, const int64_t *p_mask
) {
    for (int64_t i = 0; i < K; ++i) {
        int64_t gi = group[i];
        int64_t mi = mask_n[i];
        double ri = radius[i];
        double bi = bcoef[i];
        for (int64_t j = i + 1; j < K; ++j) {
            if ((gi & mask_n[j]) == 0 || (mi & group[j]) == 0) continue;
            double ims = inv[i] + inv[j];
            if (ims == 0.0) continue;
            double dx = pos_n[i * 2 + 0] - pos_n[j * 2 + 0];
            double dy = pos_n[i * 2 + 1] - pos_n[j * 2 + 1];
            double rs = ri + radius[j];
            double d2 = dx * dx + dy * dy;
            if (d2 > rs * rs || d2 == 0.0) continue;
            double d = sqrt(d2);
            double nx = dx / d;
            double ny = dy / d;
            double fa = inv[i] / ims;
            double pen = rs - d;
            pos_n[i * 2 + 0] += nx * pen * fa;
            pos_n[i * 2 + 1] += ny * pen * fa;
            pos_n[j * 2 + 0] -= nx * pen * (1.0 - fa);
            pos_n[j * 2 + 1] -= ny * pen * (1.0 - fa);
            double rvn = (vel_n[i * 2 + 0] - vel_n[j * 2 + 0]) * nx + (vel_n[i * 2 + 1] - vel_n[j * 2 + 1]) * ny;
            if (rvn < 0.0) {
                double f = rvn * (1.0 + bi * bcoef[j]);
                vel_n[i * 2 + 0] -= nx * f * fa;
                vel_n[i * 2 + 1] -= ny * f * fa;
                vel_n[j * 2 + 0] += nx * f * (1.0 - fa);
                vel_n[j * 2 + 1] += ny * f * (1.0 - fa);
            }
        }
        if (inv[i] == 0.0) continue;
        for (int64_t p = 0; p < PL; ++p) {
            if ((gi & p_mask[p]) == 0 || (mi & p_group[p]) == 0) continue;
            double nx = p_normal[p * 2 + 0];
            double ny = p_normal[p * 2 + 1];
            double dist = p_dist[p] - (pos_n[i * 2 + 0] * nx + pos_n[i * 2 + 1] * ny) + ri;
            if (dist > 0.0) {
                pos_n[i * 2 + 0] += nx * dist;
                pos_n[i * 2 + 1] += ny * dist;
                bounce_c(&vel_n[i * 2 + 0], &vel_n[i * 2 + 1], nx, ny, bi * p_bcoef[p]);
            }
        }
        for (int64_t s = 0; s < S; ++s) {
            if ((gi & s_mask[s]) == 0 || (mi & s_group[s]) == 0) continue;
            double px = pos_n[i * 2 + 0];
            double py = pos_n[i * 2 + 1];
            double nx, ny, dist;
            if (s_curved[s] == 0) {
                double sx = s_p1[s * 2 + 0] - s_p0[s * 2 + 0];
                double sy = s_p1[s * 2 + 1] - s_p0[s * 2 + 1];
                if (sx * (px - s_p0[s * 2 + 0]) + sy * (py - s_p0[s * 2 + 1]) <= 0.0) continue;
                if (sx * (px - s_p1[s * 2 + 0]) + sy * (py - s_p1[s * 2 + 1]) >= 0.0) continue;
                nx = s_nx_pre[s];
                ny = s_ny_pre[s];
                dist = nx * (px - s_p1[s * 2 + 0]) + ny * (py - s_p1[s * 2 + 1]);
            } else {
                double cx = px - s_center[s * 2 + 0];
                double cy = py - s_center[s * 2 + 1];
                bool in0 = (cx * s_t0[s * 2 + 0] + cy * s_t0[s * 2 + 1]) > 0.0;
                bool in1 = (cx * s_t1[s * 2 + 0] + cy * s_t1[s * 2 + 1]) > 0.0;
                bool inside = (s_curved[s] == 1) ? (in0 && in1) : (in0 || in1);
                if (!inside) continue;
                double cl = sqrt(cx * cx + cy * cy);
                if (cl == 0.0) continue;
                nx = cx / cl;
                ny = cy / cl;
                dist = cl - s_radius[s];
            }
            double bias = s_bias[s];
            if (bias == 0.0) {
                if (dist < 0.0) { dist = -dist; nx = -nx; ny = -ny; }
            } else if (bias < 0.0) {
                bias = -bias; dist = -dist; nx = -nx; ny = -ny;
            }
            if (dist < -bias) continue;
            if (dist < ri) {
                double pen = ri - dist;
                pos_n[i * 2 + 0] += nx * pen;
                pos_n[i * 2 + 1] += ny * pen;
                bounce_c(&vel_n[i * 2 + 0], &vel_n[i * 2 + 1], nx, ny, bi * s_bcoef[s]);
            }
        }
        double r2 = ri * ri;
        for (int64_t v = 0; v < V; ++v) {
            if ((gi & v_mask[v]) == 0 || (mi & v_group[v]) == 0) continue;
            double dx = pos_n[i * 2 + 0] - v_pos[v * 2 + 0];
            double dy = pos_n[i * 2 + 1] - v_pos[v * 2 + 1];
            double d2 = dx * dx + dy * dy;
            if (d2 > r2 || d2 == 0.0) continue;
            double d = sqrt(d2);
            double nx = dx / d;
            double ny = dy / d;
            pos_n[i * 2 + 0] += nx * (ri - d);
            pos_n[i * 2 + 1] += ny * (ri - d);
            bounce_c(&vel_n[i * 2 + 0], &vel_n[i * 2 + 1], nx, ny, bi * v_bcoef[v]);
        }
    }
}

void step_batch_c(
    int64_t N, int64_t K, int64_t P, int64_t first_player, int64_t frames,
    int64_t V, int64_t S, int64_t PL, int64_t G,
    double *pos, double *vel, int64_t *mask, bool *kick_cancel, bool *kickoff,
    const int64_t *actions, const double *move_unit,
    const double *radius, double *inv_env, const double *bcoef, const double *damping, const int64_t *group,
    const int64_t *player_team,
    double p_acc, double p_kacc, double p_kdamp, double p_kstr, double p_kback,
    const double *v_pos, const double *v_bcoef, const int64_t *v_group, const int64_t *v_mask,
    const double *s_p0, const double *s_p1, const int64_t *s_curved,
    const double *s_center, const double *s_radius, const double *s_t0, const double *s_t1,
    const double *s_bias, const double *s_bcoef, const int64_t *s_group, const int64_t *s_mask,
    const double *s_nx_pre, const double *s_ny_pre,
    const double *p_normal, const double *p_dist, const double *p_bcoef, const int64_t *p_group, const int64_t *p_mask,
    const double *g_p0, const double *g_p1, const int64_t *g_team,
    int64_t *goal_out, bool *touch_out, bool *kick_out,
    double *ball_grav,
    int64_t *window_goal, bool *window_kicked, int64_t *last_touch,
    bool *window_touches, double *window_ball_pos
) {
    #pragma omp parallel for schedule(static) if(N >= 8)
    for (int64_t n = 0; n < N; ++n) {
        double *pos_n = pos + n * K * 2;
        double *vel_n = vel + n * K * 2;
        int64_t *mask_n = mask + n * K;
        double *inv = inv_env + n * K;
        bool *kc_n = kick_cancel + n * P;
        const int64_t *act_n = actions + n * P;
        bool *touch_n = touch_out + n * P;
        bool *kick_n = kick_out + n * P;
        double *bgrav_n = ball_grav + n * 2;

        if (window_goal != NULL) window_goal[n] = 0;
        if (window_kicked != NULL) {
            for (int64_t p = 0; p < P; ++p) window_kicked[n * P + p] = false;
        }

        for (int64_t frame = 0; frame < frames; ++frame) {
            goal_out[n] = 0;
            for (int64_t p = 0; p < P; ++p) {
                int64_t k = first_player + p;
                int64_t a = act_n[p];
                bool kick_pressed = (a >= 9);
                if (!kick_pressed) kc_n[p] = false;
                bool is_kicking = kick_pressed && !kc_n[p];
                touch_n[p] = false;
                kick_n[p] = false;
                double dx = pos_n[0] - pos_n[k * 2 + 0];
                double dy = pos_n[1] - pos_n[k * 2 + 1];
                double d = sqrt(dx * dx + dy * dy);
                if (d - radius[k] - radius[0] < 4.0) {
                    touch_n[p] = true;
                    if (is_kicking && d > 0.0) {
                        double nx = dx / d;
                        double ny = dy / d;
                        vel_n[0] += nx * p_kstr * inv[0];
                        vel_n[1] += ny * p_kstr * inv[0];
                        vel_n[k * 2 + 0] -= nx * p_kback * inv[k];
                        vel_n[k * 2 + 1] -= ny * p_kback * inv[k];
                        kc_n[p] = true;
                        kick_n[p] = true;
                    }
                }
                is_kicking = kick_pressed && !kc_n[p];
                double acc = is_kicking ? p_kacc : p_acc;
                int64_t m = a % 9;
                vel_n[k * 2 + 0] += move_unit[m * 2 + 0] * acc;
                vel_n[k * 2 + 1] += move_unit[m * 2 + 1] * acc;
            }
            double bx0 = pos_n[0];
            double by0 = pos_n[1];
            for (int64_t k = 0; k < K; ++k) {
                pos_n[k * 2 + 0] += vel_n[k * 2 + 0];
                pos_n[k * 2 + 1] += vel_n[k * 2 + 1];
                double dmp = damping[k];
                if (k >= first_player) {
                    int64_t p = k - first_player;
                    if (act_n[p] >= 9 && !kc_n[p]) dmp = p_kdamp;
                }
                if (k == 0) {
                    vel_n[0] += bgrav_n[0];
                    vel_n[1] += bgrav_n[1];
                }
                vel_n[k * 2 + 0] *= dmp;
                vel_n[k * 2 + 1] *= dmp;
            }
            collide_env_c(
                K, V, S, PL, pos_n, vel_n, mask_n, radius, inv, bcoef, group,
                v_pos, v_bcoef, v_group, v_mask,
                s_p0, s_p1, s_curved, s_center, s_radius, s_t0, s_t1, s_bias, s_bcoef, s_group, s_mask,
                s_nx_pre, s_ny_pre,
                p_normal, p_dist, p_bcoef, p_group, p_mask
            );
            if (kickoff[n]) {
                if (vel_n[0] != 0.0 || vel_n[1] != 0.0) {
                    kickoff[n] = false;
                    for (int64_t p = 0; p < P; ++p) mask_n[first_player + p] = 39;
                }
            } else {
                double qx = pos_n[0], qy = pos_n[1];
                double mvx = qx - bx0, mvy = qy - by0;
                if (mvx != 0.0 || mvy != 0.0) {
                    for (int64_t g = 0; g < G; ++g) {
                        double ax = g_p0[g * 2 + 0], ay = g_p0[g * 2 + 1];
                        double gx1 = g_p1[g * 2 + 0], gy1 = g_p1[g * 2 + 1];
                        double gx = gx1 - ax, gy = gy1 - ay;
                        double c1 = ((qx - ax) * mvy - (qy - ay) * mvx) * ((qx - gx1) * mvy - (qy - gy1) * mvx);
                        double c2 = ((bx0 - ax) * gy - (by0 - ay) * gx) * ((qx - ax) * gy - (qy - ay) * gx);
                        if (c1 <= 0.0 && c2 <= 0.0) {
                            goal_out[n] = (g_team[g] == 0) ? -1 : 1;
                            break;
                        }
                    }
                }
            }
            if (window_goal != NULL && window_goal[n] == 0) window_goal[n] = goal_out[n];
            if (window_kicked != NULL) {
                for (int64_t p = 0; p < P; ++p) window_kicked[n * P + p] |= kick_n[p];
            }
            if (last_touch != NULL) {
                for (int64_t p = 0; p < P; ++p) {
                    if (touch_n[p]) last_touch[n] = player_team[p];
                }
            }
            if (window_touches != NULL) {
                for (int64_t p = 0; p < P; ++p) {
                    window_touches[(n * frames + frame) * P + p] = touch_n[p];
                }
            }
            if (window_ball_pos != NULL) {
                window_ball_pos[(n * frames + frame) * 2 + 0] = pos_n[0];
                window_ball_pos[(n * frames + frame) * 2 + 1] = pos_n[1];
            }
        }
    }
}
"""


def _load_c_kernel():
    if HAS_NUMBA:
        return None
    compiler = shutil.which("gcc") or shutil.which("cc") or shutil.which("clang")
    if not compiler:
        return None
    digest = hashlib.sha256(_C_KERNEL_SOURCE.encode("utf-8")).hexdigest()[:16]
    cache_dir = Path(tempfile.gettempdir()) / "haxballrl_c_cache"
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        so_path = cache_dir / f"fast_physics_{digest}.so"
        if not so_path.is_file():
            c_path = cache_dir / f"fast_physics_{digest}.c"
            tmp_so = cache_dir / f"fast_physics_{digest}.{os.getpid()}.so"
            c_path.write_text(_C_KERNEL_SOURCE, encoding="utf-8")
            cmd = [compiler, "-O3", "-fopenmp", "-shared", "-fPIC", str(c_path), "-o", str(tmp_so), "-lm"]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                cmd = [compiler, "-O3", "-shared", "-fPIC", str(c_path), "-o", str(tmp_so), "-lm"]
                res = subprocess.run(cmd, capture_output=True, text=True)
                if res.returncode != 0:
                    return None
            tmp_so.replace(so_path)
        lib = ctypes.CDLL(str(so_path))
        lib.set_omp_threads_c.argtypes = [ctypes.c_int]
        lib.set_omp_threads_c.restype = None
        return lib
    except Exception:
        return None


_C_LIB = _load_c_kernel()


def set_physics_threads(threads: int) -> None:
    """Configura el número de hilos para el motor físico (Numba o C/OpenMP)."""
    threads = max(1, int(threads))
    if HAS_NUMBA and _numba_mod is not None:
        try:
            _numba_mod.set_num_threads(threads)
        except Exception:
            pass
    if _C_LIB is not None:
        try:
            _C_LIB.set_omp_threads_c(threads)
        except Exception:
            pass


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
               frames=1, window_goal=None, window_kicked=None, last_touch=None,
               window_touches=None, window_ball_pos=None):
    """Avanza frames ticks por partido con una única barrera paralela."""
    N = pos.shape[0]
    K = pos.shape[1]
    P = actions.shape[1]
    for n in prange(N):
        if window_goal is not None:
            window_goal[n] = 0
        if window_kicked is not None:
            for p in range(P):
                window_kicked[n, p] = False
        for frame in range(frames):
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
                dx = pos[n, 0, 0] - pos[n, k, 0]
                dy = pos[n, 0, 1] - pos[n, k, 1]
                d = math.sqrt(dx * dx + dy * dy)
                if d - radius[k] - radius[0] < KICK_REACH:
                    touch_out[n, p] = True
                    if is_kicking and d > 0.0:
                        nx = dx / d
                        ny = dy / d
                        vel[n, 0, 0] += nx * p_kstr * inv[0]
                        vel[n, 0, 1] += ny * p_kstr * inv[0]
                        vel[n, k, 0] -= nx * p_kback * inv[k]
                        vel[n, k, 1] -= ny * p_kback * inv[k]
                        kick_cancel[n, p] = True
                        kick_out[n, p] = True
                        if ps_on and ps_comba[n]:
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
                        goal_out[n] = -1 if g_team[g] == 0 else 1
                        break
            # 5. script de la sala (onGameTick), después de la física
            if ps_on:
                ps_gtick[n] += 1
                if ps_has_pending[n]:
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
                        ps_comba[n] = True
                        inv[0] = ps_par[PS_POWER_INV]
                        ps_charge[n] = -1
                if ps_gtick[n] % int(ps_par[PS_EVERY]) == 0:
                    holder = -1
                    for p in range(P):
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
            if window_touches is not None:
                for p in range(P):
                    window_touches[n, frame, p] = touch_out[n, p]

            if window_ball_pos is not None:
                window_ball_pos[n, frame, 0] = pos[n, 0, 0]
                window_ball_pos[n, frame, 1] = pos[n, 0, 1]


def _step_batch_fast_numpy(sim: BatchSim, actions: np.ndarray, frames: int = 1,
                           window_goal=None, window_kicked=None, last_touch=None,
                           window_touches=None, window_ball_pos=None) -> None:
    """Fallback NumPy vectorizado con broadphase por lote (20x más rápido que bucles escalares en Python)."""
    pos = sim.pos
    vel = sim.vel
    mask = sim.mask
    kick_cancel = sim.kick_cancel
    kickoff = sim.kickoff
    radius = sim.radius
    inv_env = sim.inv_env
    bcoef = sim.bcoef
    damping = sim.damping
    group = sim.group
    fp = sim.first_player
    P = sim.P
    K = sim.K
    pl = sim.st.player
    p_acc = float(pl["acceleration"])
    p_kacc = float(pl["kickingAcceleration"])
    p_kdamp = float(pl["kickingDamping"])
    p_kstr = float(pl["kickStrength"])
    p_kback = float(pl["kickback"])

    v_pos, v_bcoef, v_group, v_mask = sim._v_pos, sim._v_bcoef, sim._v_group, sim._v_mask
    s_p0, s_p1, s_curved = sim._s_p0, sim._s_p1, sim._s_curved
    s_center, s_radius, s_t0, s_t1 = sim._s_center, sim._s_radius, sim._s_t0, sim._s_t1
    s_bias, s_bcoef, s_group, s_mask = sim._s_bias, sim._s_bcoef, sim._s_group, sim._s_mask
    s_sx, s_sy, s_nx0, s_ny0 = sim._s_sx, sim._s_sy, sim._s_nx, sim._s_ny
    s_min_x, s_max_x, s_min_y, s_max_y = sim._s_min_x, sim._s_max_x, sim._s_min_y, sim._s_max_y
    p_normal, p_dist = sim.st.p_normal, sim.st.p_dist
    p_bcoef, p_group, p_mask = sim.st.p_bcoef, sim.st.p_group, sim.st.p_mask
    g_p0, g_p1, g_team = sim.st.g_p0, sim.st.g_p1, sim.st.g_team

    if window_goal is not None:
        window_goal.fill(0)
    if window_kicked is not None:
        window_kicked.fill(False)

    for frame in range(frames):
        sim.goal.fill(0)
        kick_pressed = actions >= 9
        kick_cancel &= kick_pressed
        is_kicking = kick_pressed & ~kick_cancel
        sim.touch.fill(False)
        sim.kicked.fill(False)
        sim.ps_kicked.fill(False)

        dx_p = pos[:, 0:1, 0] - pos[:, fp:, 0]
        dy_p = pos[:, 0:1, 1] - pos[:, fp:, 1]
        d_p = np.hypot(dx_p, dy_p)
        in_reach = (d_p - radius[fp:][None, :] - radius[0]) < KICK_REACH
        sim.touch[:] = in_reach
        do_kick = in_reach & is_kicking & (d_p > 0.0)
        if np.any(do_kick):
            for p in range(P):
                if not np.any(do_kick[:, p]):
                    continue
                idx = np.flatnonzero(do_kick[:, p])
                k = fp + p
                nx = dx_p[idx, p] / d_p[idx, p]
                ny = dy_p[idx, p] / d_p[idx, p]
                vel[idx, 0, 0] += nx * p_kstr * inv_env[idx, 0]
                vel[idx, 0, 1] += ny * p_kstr * inv_env[idx, 0]
                vel[idx, k, 0] -= nx * p_kback * inv_env[idx, k]
                vel[idx, k, 1] -= ny * p_kback * inv_env[idx, k]
                kick_cancel[idx, p] = True
                sim.kicked[idx, p] = True
                if sim.ps_on:
                    for n in idx:
                        if sim.ps_comba[n]:
                            lat = nx[idx == n][0] * vel[n, k, 1] - ny[idx == n][0] * vel[n, k, 0]
                            ci = min(1.0, max(-1.0, float(lat) / sim.ps_par[PS_MAX_LAT]))
                            sx_, sy_ = vel[n, 0, 0], vel[n, 0, 1]
                            sp = math.hypot(sx_, sy_)
                            sim.ps_pending[n, 0] = sy_ / sp * ci * sim.ps_par[PS_GRAV] if sp > 0.1 else 0.0
                            sim.ps_pending[n, 1] = -sx_ / sp * ci * sim.ps_par[PS_GRAV] if sp > 0.1 else 0.0
                            sim.ps_has_pending[n] = True
                            sim.ps_grav_left[n] = int(sim.ps_par[PS_CURVE])
                            sim.ps_inv_left[n] = int(sim.ps_par[PS_INV_RESET])
                            sim.ps_comba[n] = False
                            sim.ps_held[n] = -1
                            sim.ps_charge[n] = -1
                            sim.ps_kicked[n, p] = True

        still_kicking = kick_pressed & ~kick_cancel
        acc = np.where(still_kicking, p_kacc, p_acc)
        m = actions % 9
        vel[:, fp:, 0] += MOVE_UNIT[m, 0] * acc
        vel[:, fp:, 1] += MOVE_UNIT[m, 1] * acc

        bx0 = pos[:, 0, 0].copy()
        by0 = pos[:, 0, 1].copy()
        pos += vel
        vel[:, 0, :] += sim.ball_grav
        vel[:, :fp, :] *= damping[:fp, None]
        p_dmp = np.where(still_kicking, p_kdamp, damping[fp:])
        vel[:, fp:, :] *= p_dmp[:, :, None]

        for i in range(K):
            gi = group[i]
            ri = radius[i]
            bi = bcoef[i]
            im_i = sim.invmass[i]
            mi_all = mask[:, i]
            if i + 1 < K:
                js = np.arange(i + 1, K)
                valid_j = (im_i + sim.invmass[js]) > 0.0
                if np.any(valid_j):
                    js = js[valid_j]
                    dx = pos[:, i:i + 1, 0] - pos[:, js, 0]
                    dy = pos[:, i:i + 1, 1] - pos[:, js, 1]
                    rs = ri + radius[js][None, :]
                    d2 = dx * dx + dy * dy
                    hit_mat = (d2 <= rs * rs) & (d2 > 0.0)
                    if np.any(hit_mat):
                        for jj_idx in np.flatnonzero(hit_mat.any(axis=0)):
                            j = int(js[jj_idx])
                            gj = group[j]
                            hit = hit_mat[:, jj_idx] & ((gi & mask[:, j]) != 0) & ((mi_all & gj) != 0)
                            if not np.any(hit):
                                continue
                            idx = np.flatnonzero(hit)
                            dx_h = pos[idx, i, 0] - pos[idx, j, 0]
                            dy_h = pos[idx, i, 1] - pos[idx, j, 1]
                            d2_h = dx_h * dx_h + dy_h * dy_h
                            rs_j = ri + radius[j]
                            ok = (d2_h <= rs_j * rs_j) & (d2_h > 0.0)
                            if not np.any(ok):
                                continue
                            idx = idx[ok]
                            d = np.sqrt(d2_h[ok])
                            nx = dx_h[ok] / d
                            ny = dy_h[ok] / d
                            fa = inv_env[idx, i] / (inv_env[idx, i] + inv_env[idx, j])
                            pen = rs_j - d
                            pos[idx, i, 0] += nx * pen * fa
                            pos[idx, i, 1] += ny * pen * fa
                            pos[idx, j, 0] -= nx * pen * (1.0 - fa)
                            pos[idx, j, 1] -= ny * pen * (1.0 - fa)
                            rvn = (vel[idx, i, 0] - vel[idx, j, 0]) * nx + (vel[idx, i, 1] - vel[idx, j, 1]) * ny
                            bm = rvn < 0.0
                            if np.any(bm):
                                b_idx = idx[bm]
                                nx_b, ny_b, fa_b = nx[bm], ny[bm], fa[bm]
                                f = rvn[bm] * (1.0 + bi * bcoef[j])
                                vel[b_idx, i, 0] -= nx_b * f * fa_b
                                vel[b_idx, i, 1] -= ny_b * f * fa_b
                                vel[b_idx, j, 0] += nx_b * f * (1.0 - fa_b)
                                vel[b_idx, j, 1] += ny_b * f * (1.0 - fa_b)
            if im_i == 0.0:
                continue

            if len(p_dist):
                p_dists = p_dist[None, :] - (pos[:, i, 0:1] * p_normal[:, 0] + pos[:, i, 1:2] * p_normal[:, 1]) + ri
                p_hits = (p_dists > 0.0) & ((gi & p_mask[None, :]) != 0) & ((mi_all[:, None] & p_group[None, :]) != 0)
                if np.any(p_hits):
                    for p in np.flatnonzero(p_hits.any(axis=0)):
                        nx, ny = p_normal[p, 0], p_normal[p, 1]
                        dist = p_dist[p] - (pos[:, i, 0] * nx + pos[:, i, 1] * ny) + ri
                        hit = (dist > 0.0) & ((mi_all & p_group[p]) != 0)
                        if np.any(hit):
                            idx = np.flatnonzero(hit)
                            pos[idx, i, 0] += nx * dist[idx]
                            pos[idx, i, 1] += ny * dist[idx]
                            vn = vel[idx, i, 0] * nx + vel[idx, i, 1] * ny
                            bm = vn < 0.0
                            if np.any(bm):
                                b_idx = idx[bm]
                                f = vn[bm] * (1.0 + bi * p_bcoef[p])
                                vel[b_idx, i, 0] -= nx * f
                                vel[b_idx, i, 1] -= ny * f

            if len(s_p0):
                px = pos[:, i, 0:1]
                py = pos[:, i, 1:2]
                s_cand = (
                    ((gi & s_mask[None, :]) != 0)
                    & ((mi_all[:, None] & s_group[None, :]) != 0)
                    & (px >= s_min_x[None, :]) & (px <= s_max_x[None, :])
                    & (py >= s_min_y[None, :]) & (py <= s_max_y[None, :])
                )
                if np.any(s_cand):
                    for s in np.flatnonzero(s_cand.any(axis=0)):
                        idx = np.flatnonzero(s_cand[:, s])
                        px_i = pos[idx, i, 0]
                        py_i = pos[idx, i, 1]
                        if s_curved[s] == 0:
                            sx, sy = s_sx[s], s_sy[s]
                            in_seg = (
                                ((sx * (px_i - s_p0[s, 0]) + sy * (py_i - s_p0[s, 1])) > 0.0)
                                & ((sx * (px_i - s_p1[s, 0]) + sy * (py_i - s_p1[s, 1])) < 0.0)
                            )
                            if not np.any(in_seg):
                                continue
                            idx = idx[in_seg]
                            nx = np.full(len(idx), s_nx0[s])
                            ny = np.full(len(idx), s_ny0[s])
                            dist = nx * (pos[idx, i, 0] - s_p1[s, 0]) + ny * (pos[idx, i, 1] - s_p1[s, 1])
                        else:
                            cx = px_i - s_center[s, 0]
                            cy = py_i - s_center[s, 1]
                            in0 = (cx * s_t0[s, 0] + cy * s_t0[s, 1]) > 0.0
                            in1 = (cx * s_t1[s, 0] + cy * s_t1[s, 1]) > 0.0
                            inside = (in0 & in1) if s_curved[s] == 1 else (in0 | in1)
                            if not np.any(inside):
                                continue
                            idx = idx[inside]
                            cx, cy = cx[inside], cy[inside]
                            cl = np.hypot(cx, cy)
                            valid_cl = cl > 0.0
                            if not np.any(valid_cl):
                                continue
                            idx = idx[valid_cl]
                            cl = cl[valid_cl]
                            nx = cx[valid_cl] / cl
                            ny = cy[valid_cl] / cl
                            dist = cl - s_radius[s]
                        bias = s_bias[s]
                        if bias == 0.0:
                            neg = dist < 0.0
                            dist = np.where(neg, -dist, dist)
                            nx = np.where(neg, -nx, nx)
                            ny = np.where(neg, -ny, ny)
                        elif bias < 0.0:
                            bias = -bias
                            dist = -dist
                            nx = -nx
                            ny = -ny
                        hit = (dist >= -bias) & (dist < ri)
                        if np.any(hit):
                            h_idx = idx[hit]
                            nx_h, ny_h, pen = nx[hit], ny[hit], ri - dist[hit]
                            pos[h_idx, i, 0] += nx_h * pen
                            pos[h_idx, i, 1] += ny_h * pen
                            vn = vel[h_idx, i, 0] * nx_h + vel[h_idx, i, 1] * ny_h
                            bm = vn < 0.0
                            if np.any(bm):
                                b_idx = h_idx[bm]
                                f = vn[bm] * (1.0 + bi * s_bcoef[s])
                                vel[b_idx, i, 0] -= nx_h[bm] * f
                                vel[b_idx, i, 1] -= ny_h[bm] * f

            if len(v_pos):
                r2 = ri * ri
                dx_v = pos[:, i, 0:1] - v_pos[None, :, 0]
                dy_v = pos[:, i, 1:2] - v_pos[None, :, 1]
                d2_v = dx_v * dx_v + dy_v * dy_v
                v_cand = (
                    (d2_v <= r2) & (d2_v > 0.0)
                    & ((gi & v_mask[None, :]) != 0)
                    & ((mi_all[:, None] & v_group[None, :]) != 0)
                )
                if np.any(v_cand):
                    for v in np.flatnonzero(v_cand.any(axis=0)):
                        idx = np.flatnonzero(v_cand[:, v])
                        dx = pos[idx, i, 0] - v_pos[v, 0]
                        dy = pos[idx, i, 1] - v_pos[v, 1]
                        d2 = dx * dx + dy * dy
                        ok = (d2 <= r2) & (d2 > 0.0)
                        if not np.any(ok):
                            continue
                        idx = idx[ok]
                        d = np.sqrt(d2[ok])
                        nx = dx[ok] / d
                        ny = dy[ok] / d
                        pos[idx, i, 0] += nx * (ri - d)
                        pos[idx, i, 1] += ny * (ri - d)
                        vn = vel[idx, i, 0] * nx + vel[idx, i, 1] * ny
                        bm = vn < 0.0
                        if np.any(bm):
                            b_idx = idx[bm]
                            f = vn[bm] * (1.0 + bi * v_bcoef[v])
                            vel[b_idx, i, 0] -= nx[bm] * f
                            vel[b_idx, i, 1] -= ny[bm] * f

        ko_moved = kickoff & ((vel[:, 0, 0] != 0.0) | (vel[:, 0, 1] != 0.0))
        if np.any(ko_moved):
            kickoff[ko_moved] = False
            mask[ko_moved, fp:] = PLAYER_MASK
        active_play = ~kickoff
        if np.any(active_play):
            qx = pos[:, 0, 0]
            qy = pos[:, 0, 1]
            mvx = qx - bx0
            mvy = qy - by0
            moving = active_play & ((mvx != 0.0) | (mvy != 0.0))
            if np.any(moving):
                for g in range(g_p0.shape[0]):
                    ax, ay = g_p0[g, 0], g_p0[g, 1]
                    gx, gy = g_p1[g, 0] - ax, g_p1[g, 1] - ay
                    c1 = ((qx - ax) * mvy - (qy - ay) * mvx) * ((qx - g_p1[g, 0]) * mvy - (qy - g_p1[g, 1]) * mvx)
                    c2 = ((bx0 - ax) * gy - (by0 - ay) * gx) * ((qx - ax) * gy - (qy - ay) * gx)
                    scored = moving & (sim.goal == 0) & (c1 <= 0.0) & (c2 <= 0.0)
                    if np.any(scored):
                        sim.goal[scored] = -1 if g_team[g] == 0 else 1

        if sim.ps_on:
            for n in range(sim.N):
                sim.ps_gtick[n] += 1
                if sim.ps_has_pending[n]:
                    sim.ball_grav[n] = sim.ps_pending[n]
                    sim.ps_has_pending[n] = False
                elif sim.ps_grav_left[n] > 0:
                    sim.ps_grav_left[n] -= 1
                    if sim.ps_grav_left[n] == 0:
                        sim.ball_grav[n] = 0.0
                if sim.ps_inv_left[n] > 0:
                    sim.ps_inv_left[n] -= 1
                    if sim.ps_inv_left[n] == 0:
                        sim.inv_env[n, 0] = sim.ps_par[PS_BASE_INV]
                if sim.ps_charge[n] > 0:
                    sim.ps_charge[n] -= 1
                    if sim.ps_charge[n] == 0:
                        sim.ps_comba[n] = True
                        sim.inv_env[n, 0] = sim.ps_par[PS_POWER_INV]
                        sim.ps_charge[n] = -1
                if sim.ps_gtick[n] % int(sim.ps_par[PS_EVERY]) == 0:
                    holder = -1
                    for p in range(P):
                        k = fp + p
                        if math.hypot(pos[n, 0, 0] - pos[n, k, 0], pos[n, 0, 1] - pos[n, k, 1]) <= radius[k] + radius[0] + 0.1:
                            holder = p
                    if holder >= 0:
                        if sim.ps_held[n] < 0:
                            sim.ps_held[n] = holder
                            sim.ps_comba[n] = False
                            sim.ps_charge[n] = int(sim.ps_par[PS_CHARGE])
                    elif sim.ps_held[n] >= 0:
                        sim.ps_held[n] = -1
                        sim.ps_comba[n] = False
                        sim.ps_charge[n] = -1

        if window_goal is not None:
            unset = window_goal == 0
            window_goal[unset] = sim.goal[unset]
        if window_kicked is not None:
            window_kicked |= sim.kicked
        if last_touch is not None and np.any(sim.touch):
            for p in range(P):
                last_touch[sim.touch[:, p]] = sim.player_team[p]
        if window_touches is not None:
            window_touches[:, frame, :] = sim.touch
        if window_ball_pos is not None:
            window_ball_pos[:, frame, :] = pos[:, 0, :]


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

        # Pre-filtrar segmentos y vértices decorativos (cMask == 0 o cGroup == 0) y deduplicar
        # vértices idénticos para acelerar colisiones sin alterar el objeto Stadium ni su identidad.
        v_active_idx = np.flatnonzero((st.v_mask != 0) & (st.v_group != 0))
        seen_v = set()
        dedup_v = []
        for idx in v_active_idx:
            key = (
                float(st.v_pos[idx, 0]),
                float(st.v_pos[idx, 1]),
                float(st.v_bcoef[idx]),
                int(st.v_group[idx]),
                int(st.v_mask[idx]),
            )
            if key not in seen_v:
                seen_v.add(key)
                dedup_v.append(idx)
        dedup_v_arr = np.asarray(dedup_v, dtype=np.int64)
        self._v_pos = np.ascontiguousarray(st.v_pos[dedup_v_arr], dtype=np.float64)
        self._v_bcoef = np.ascontiguousarray(st.v_bcoef[dedup_v_arr], dtype=np.float64)
        self._v_group = np.ascontiguousarray(st.v_group[dedup_v_arr], dtype=np.int64)
        self._v_mask = np.ascontiguousarray(st.v_mask[dedup_v_arr], dtype=np.int64)

        s_active = (st.s_mask != 0) & (st.s_group != 0)
        self._s_p0 = np.ascontiguousarray(st.s_p0[s_active], dtype=np.float64)
        self._s_p1 = np.ascontiguousarray(st.s_p1[s_active], dtype=np.float64)
        self._s_curved = np.ascontiguousarray(st.s_curved[s_active], dtype=np.int64)
        self._s_center = np.ascontiguousarray(st.s_center[s_active], dtype=np.float64)
        self._s_radius = np.ascontiguousarray(st.s_radius[s_active], dtype=np.float64)
        self._s_t0 = np.ascontiguousarray(st.s_t0[s_active], dtype=np.float64)
        self._s_t1 = np.ascontiguousarray(st.s_t1[s_active], dtype=np.float64)
        self._s_bias = np.ascontiguousarray(st.s_bias[s_active], dtype=np.float64)
        self._s_bcoef = np.ascontiguousarray(st.s_bcoef[s_active], dtype=np.float64)
        self._s_group = np.ascontiguousarray(st.s_group[s_active], dtype=np.int64)
        self._s_mask = np.ascontiguousarray(st.s_mask[s_active], dtype=np.int64)

        # Pre-cálculo de normales y cajas envolventes de segmentos
        self._s_sx = np.ascontiguousarray(self._s_p1[:, 0] - self._s_p0[:, 0], dtype=np.float64)
        self._s_sy = np.ascontiguousarray(self._s_p1[:, 1] - self._s_p0[:, 1], dtype=np.float64)
        s_sl = np.maximum(np.hypot(self._s_sx, self._s_sy), 1e-12)
        self._s_nx = np.ascontiguousarray(-self._s_sy / s_sl, dtype=np.float64)
        self._s_ny = np.ascontiguousarray(self._s_sx / s_sl, dtype=np.float64)
        pad = np.where(self._s_curved != 0, self._s_radius, 0.0) + 25.0
        self._s_min_x = np.minimum(self._s_p0[:, 0], self._s_p1[:, 0]) - pad
        self._s_max_x = np.maximum(self._s_p0[:, 0], self._s_p1[:, 0]) + pad
        self._s_min_y = np.minimum(self._s_p0[:, 1], self._s_p1[:, 1]) - pad
        self._s_max_y = np.maximum(self._s_p0[:, 1], self._s_p1[:, 1]) + pad

        self._p_normal = np.ascontiguousarray(st.p_normal, dtype=np.float64)
        self._p_dist = np.ascontiguousarray(st.p_dist, dtype=np.float64)
        self._p_bcoef = np.ascontiguousarray(st.p_bcoef, dtype=np.float64)
        self._p_group = np.ascontiguousarray(st.p_group, dtype=np.int64)
        self._p_mask = np.ascontiguousarray(st.p_mask, dtype=np.int64)
        self._g_p0 = np.ascontiguousarray(st.g_p0, dtype=np.float64)
        self._g_p1 = np.ascontiguousarray(st.g_p1, dtype=np.float64)
        self._g_team = np.ascontiguousarray(st.g_team, dtype=np.int64)

        self.player_team = np.ascontiguousarray([0] * n_red + [1] * n_blue, dtype=np.int64)
        self.radius = np.ascontiguousarray(
            np.concatenate([[b["radius"]], st.d_radius, np.full(self.P, pl["radius"])]), dtype=np.float64
        )
        self.invmass = np.ascontiguousarray(
            np.concatenate([[b["invMass"]], st.d_invmass, np.full(self.P, pl["invMass"])]), dtype=np.float64
        )
        self.bcoef = np.ascontiguousarray(
            np.concatenate([[b["bCoef"]], st.d_bcoef, np.full(self.P, pl["bCoef"])]), dtype=np.float64
        )
        self.damping = np.ascontiguousarray(
            np.concatenate([[b["damping"]], st.d_damping, np.full(self.P, pl["damping"])]), dtype=np.float64
        )
        from .stadium import BLUE, RED
        self.group = np.ascontiguousarray(
            np.concatenate([[b["cGroup"]], st.d_group, [RED if t == 0 else BLUE for t in self.player_team]]),
            dtype=np.int64,
        )
        self.base_mask = np.ascontiguousarray(
            np.concatenate([[b["cMask"]], st.d_mask, np.full(self.P, PLAYER_MASK)]), dtype=np.int64
        )

        self.pos = np.zeros((n_envs, self.K, 2), dtype=np.float64)
        self.vel = np.zeros((n_envs, self.K, 2), dtype=np.float64)
        self.mask = np.ascontiguousarray(np.tile(self.base_mask, (n_envs, 1)), dtype=np.int64)
        self.kick_cancel = np.zeros((n_envs, self.P), dtype=np.bool_)
        self.kickoff = np.zeros(n_envs, dtype=np.bool_)
        self.kickoff_team = np.zeros(n_envs, dtype=np.int64)
        self.goal = np.zeros(n_envs, dtype=np.int64)
        self.touch = np.zeros((n_envs, self.P), dtype=np.bool_)
        self.kicked = np.zeros((n_envs, self.P), dtype=np.bool_)
        self._coop_touch_window = None
        self._coop_ball_pos_window = None
        self.inv_env = np.ascontiguousarray(np.tile(self.invmass, (n_envs, 1)), dtype=np.float64)
        self.ball_grav = np.zeros((n_envs, 2), dtype=np.float64)
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
        self.ps_pending = np.zeros((n_envs, 2), dtype=np.float64)
        self.ps_has_pending = np.zeros(n_envs, dtype=np.bool_)
        self.ps_kicked = np.zeros((n_envs, self.P), dtype=np.bool_)
        self._spawn_cache = self.spawn_positions()
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

    def spawn_positions(self) -> np.ndarray:
        """Posiciones de saque de HaxBall: x = ±spawnDistance, y = 0, +55, -55, +110..."""
        out = np.zeros((self.P, 2), dtype=np.float64)
        counts = [0, 0]
        points = (self.st.red_spawn, self.st.blue_spawn)
        for p, t in enumerate(self.player_team):
            c = counts[t]
            pts = points[t]
            if pts is not None and c < len(pts):
                out[p] = pts[c]
            else:
                y = 55.0 * ((c + 1) >> 1) * (1 if c % 2 == 1 else -1)
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
        self.pos[idx, fp:] = self._spawn_cache
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

    def reset_open_play_tactical(self, idx) -> None:
        """Posiciones tácticas 3v3 de juego abierto (triángulos de rotación sin arquero fijo en el área)."""
        idx = np.atleast_1d(np.asarray(idx))
        if len(idx) == 0:
            return
        n = len(idx)
        fp = self.first_player
        W, H = self.st.field_half_w, self.st.field_half_h
        self.pos[idx, 1:fp] = self.st.d_pos
        self.vel[idx, 1:fp] = 0.0
        bx = self.rng.uniform(-0.68 * W, 0.68 * W, n)
        by = self.rng.uniform(-0.75 * H, 0.75 * H, n)
        self.pos[idx, 0, 0] = bx
        self.pos[idx, 0, 1] = by
        self.vel[idx, 0] = self.rng.normal(0, 1.8, (n, 2))
        # Colocar los 3 jugadores de cada equipo en formación triangular rotada (nunca anclados al área)
        for team in range(2):
            orient = 1.0 if team == 0 else -1.0
            base_slots = np.arange(3 * team, 3 * team + 3)
            # Permutar qué slot ocupa punta, ala y cierre en cada partido
            for row_i, env_idx in enumerate(idx):
                perm = self.rng.permutation(3)
                b_x, b_y = float(bx[row_i]), float(by[row_i])
                # 1er hombre: cerca de la pelota; 2do hombre: ala abierta; 3er hombre: cierre adelantado
                p1_x = np.clip(b_x - orient * self.rng.uniform(25.0, 95.0), -W + 65.0, W - 65.0)
                p1_y = np.clip(b_y + self.rng.uniform(-55.0, 55.0), -H + 30.0, H - 30.0)
                wing_sign = -1.0 if b_y >= 0.0 else 1.0
                p2_x = np.clip(b_x + orient * self.rng.uniform(-50.0, 85.0), -W + 120.0, W - 80.0)
                p2_y = np.clip(wing_sign * self.rng.uniform(70.0, 165.0), -H + 30.0, H - 30.0)
                p3_x = np.clip(b_x - orient * self.rng.uniform(140.0, 260.0), -W + 165.0, W - 165.0)
                p3_y = np.clip(-wing_sign * self.rng.uniform(20.0, 95.0), -H + 35.0, H - 35.0)
                coords = [(p1_x, p1_y), (p2_x, p2_y), (p3_x, p3_y)]
                for role_idx, slot in enumerate(base_slots[perm]):
                    self.pos[env_idx, fp + slot, 0] = coords[role_idx][0]
                    self.pos[env_idx, fp + slot, 1] = coords[role_idx][1]
        self.vel[idx, fp:] = self.rng.normal(0, 0.75, (n, self.P, 2))
        self.kick_cancel[idx] = False
        self.kickoff[idx] = False
        self.mask[idx] = self.base_mask
        self._reset_ball_state(idx)

    def reset_all(self) -> None:
        self.reset_kickoff(np.arange(self.N), kickoff_team=0)

    def _run_kernel(self, actions: np.ndarray, frames: int = 1,
                    window_goal=None, window_kicked=None, last_touch=None,
                    window_touches=None, window_ball_pos=None) -> None:
        act = np.ascontiguousarray(actions, dtype=np.int64)
        if HAS_NUMBA:
            step_batch(
                *self._step_args(act),
                frames,
                window_goal,
                window_kicked,
                last_touch,
                window_touches,
                window_ball_pos,
            )
            return
        if _C_LIB is not None and not self.ps_on:
            pl = self.st.player
            _C_LIB.step_batch_c(
                ctypes.c_int64(self.N), ctypes.c_int64(self.K), ctypes.c_int64(self.P),
                ctypes.c_int64(self.first_player), ctypes.c_int64(frames),
                ctypes.c_int64(len(self._v_pos)), ctypes.c_int64(len(self._s_p0)),
                ctypes.c_int64(len(self._p_dist)), ctypes.c_int64(len(self._g_team)),
                self.pos.ctypes.data_as(ctypes.c_void_p),
                self.vel.ctypes.data_as(ctypes.c_void_p),
                self.mask.ctypes.data_as(ctypes.c_void_p),
                self.kick_cancel.ctypes.data_as(ctypes.c_void_p),
                self.kickoff.ctypes.data_as(ctypes.c_void_p),
                act.ctypes.data_as(ctypes.c_void_p),
                MOVE_UNIT.ctypes.data_as(ctypes.c_void_p),
                self.radius.ctypes.data_as(ctypes.c_void_p),
                self.inv_env.ctypes.data_as(ctypes.c_void_p),
                self.bcoef.ctypes.data_as(ctypes.c_void_p),
                self.damping.ctypes.data_as(ctypes.c_void_p),
                self.group.ctypes.data_as(ctypes.c_void_p),
                self.player_team.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_double(pl["acceleration"]),
                ctypes.c_double(pl["kickingAcceleration"]),
                ctypes.c_double(pl["kickingDamping"]),
                ctypes.c_double(pl["kickStrength"]),
                ctypes.c_double(pl["kickback"]),
                self._v_pos.ctypes.data_as(ctypes.c_void_p),
                self._v_bcoef.ctypes.data_as(ctypes.c_void_p),
                self._v_group.ctypes.data_as(ctypes.c_void_p),
                self._v_mask.ctypes.data_as(ctypes.c_void_p),
                self._s_p0.ctypes.data_as(ctypes.c_void_p),
                self._s_p1.ctypes.data_as(ctypes.c_void_p),
                self._s_curved.ctypes.data_as(ctypes.c_void_p),
                self._s_center.ctypes.data_as(ctypes.c_void_p),
                self._s_radius.ctypes.data_as(ctypes.c_void_p),
                self._s_t0.ctypes.data_as(ctypes.c_void_p),
                self._s_t1.ctypes.data_as(ctypes.c_void_p),
                self._s_bias.ctypes.data_as(ctypes.c_void_p),
                self._s_bcoef.ctypes.data_as(ctypes.c_void_p),
                self._s_group.ctypes.data_as(ctypes.c_void_p),
                self._s_mask.ctypes.data_as(ctypes.c_void_p),
                self._s_nx.ctypes.data_as(ctypes.c_void_p),
                self._s_ny.ctypes.data_as(ctypes.c_void_p),
                self._p_normal.ctypes.data_as(ctypes.c_void_p),
                self._p_dist.ctypes.data_as(ctypes.c_void_p),
                self._p_bcoef.ctypes.data_as(ctypes.c_void_p),
                self._p_group.ctypes.data_as(ctypes.c_void_p),
                self._p_mask.ctypes.data_as(ctypes.c_void_p),
                self._g_p0.ctypes.data_as(ctypes.c_void_p),
                self._g_p1.ctypes.data_as(ctypes.c_void_p),
                self._g_team.ctypes.data_as(ctypes.c_void_p),
                self.goal.ctypes.data_as(ctypes.c_void_p),
                self.touch.ctypes.data_as(ctypes.c_void_p),
                self.kicked.ctypes.data_as(ctypes.c_void_p),
                self.ball_grav.ctypes.data_as(ctypes.c_void_p),
                window_goal.ctypes.data_as(ctypes.c_void_p) if window_goal is not None else ctypes.c_void_p(0),
                window_kicked.ctypes.data_as(ctypes.c_void_p) if window_kicked is not None else ctypes.c_void_p(0),
                last_touch.ctypes.data_as(ctypes.c_void_p) if last_touch is not None else ctypes.c_void_p(0),
                window_touches.ctypes.data_as(ctypes.c_void_p) if window_touches is not None else ctypes.c_void_p(0),
                window_ball_pos.ctypes.data_as(ctypes.c_void_p) if window_ball_pos is not None else ctypes.c_void_p(0),
            )
            return
        _step_batch_fast_numpy(
            self, act, frames, window_goal, window_kicked, last_touch, window_touches, window_ball_pos
        )

    def step(self, actions: np.ndarray) -> np.ndarray:
        """actions: (N, P) enteros 0..17. Devuelve goal (N,) con +1 rojo / -1 azul / 0."""
        self._run_kernel(actions, frames=1)
        return self.goal

    def step_frames(self, actions, frames, last_touch):
        """Sólo sin callbacks/reglas por tick ni cambios de acción dentro de la ventana."""
        goal = np.empty(self.N, dtype=np.int64)
        kicked = np.empty((self.N, self.P), dtype=np.bool_)
        self._run_kernel(actions, frames, goal, kicked, last_touch)
        return goal, kicked

    def step_frames_with_touches(self, actions, frames, last_touch):
        """Ruta fused que además conserva los toques de cada tick para rewards cooperativos."""
        goal = np.empty(self.N, dtype=np.int64)
        kicked = np.empty((self.N, self.P), dtype=np.bool_)

        touch_shape = (self.N, frames, self.P)
        pos_shape = (self.N, frames, 2)

        if (
            self._coop_touch_window is None
            or self._coop_touch_window.shape != touch_shape
        ):
            self._coop_touch_window = np.empty(
                touch_shape,
                dtype=np.bool_,
            )

            self._coop_ball_pos_window = np.empty(
                pos_shape,
                dtype=np.float64,
            )

        self._run_kernel(
            actions,
            frames,
            goal,
            kicked,
            last_touch,
            self._coop_touch_window,
            self._coop_ball_pos_window,
        )

        return (
            goal,
            kicked,
            self._coop_touch_window,
            self._coop_ball_pos_window,
        )

    def _step_args(self, actions):
        st, pl = self.st, self.st.player
        return (self.pos, self.vel, self.mask, self.kick_cancel, self.kickoff, self.kickoff_team,
                np.ascontiguousarray(actions, dtype=np.int64), MOVE_UNIT,
                self.radius, self.inv_env, self.bcoef, self.damping, self.group,
                self.first_player, self.player_team,
                float(pl["acceleration"]), float(pl["kickingAcceleration"]), float(pl["kickingDamping"]),
                float(pl["kickStrength"]), float(pl["kickback"]),
                self._v_pos, self._v_bcoef, self._v_group, self._v_mask,
                self._s_p0, self._s_p1, self._s_curved, self._s_center, self._s_radius, self._s_t0, self._s_t1,
                self._s_bias, self._s_bcoef, self._s_group, self._s_mask,
                st.p_normal, st.p_dist, st.p_bcoef, st.p_group, st.p_mask,
                st.g_p0, st.g_p1, st.g_team,
                self.goal, self.touch, self.kicked,
                self.ball_grav, self.ps_on, self.ps_par, self.ps_held, self.ps_charge, self.ps_comba,
                self.ps_grav_left, self.ps_inv_left, self.ps_gtick, self.ps_pending,
                self.ps_has_pending, self.ps_kicked)

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