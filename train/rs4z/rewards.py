"""Registro de rewards RS4-Z. Todo término arranca en 0; el curriculum fija coeficientes por etapa.

Objetivo: goles ±1 (de equipo, suma cero). Desde S5, resultado del partido ±0,3 al terminal de fin
(suma cero; sólo el crítico ve marcador y reloj). En ejercicios, el resultado del ejercicio (+1/0/-1)
reemplaza al gol (no se suman los dos).

Auxiliares: sólo potenciales (PBRS) F = γ·c·Φ(s') − c·Φ(s), con Φ(s') = 0 en terminales reales, de suma
cero entre equipos y centrados. Preservan la política óptima (Ng et al. 1999) y los equilibrios de Nash
del juego estocástico (Devlin & Kudenko 2011); el retiro de c entre rollouts es PBRS dinámico (Devlin &
Kudenko 2012). Ver la tabla de justificación, escala, exploit y retiro en `TERMS`.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numba import njit, prange

from env.rs4z import kernel as K

ROOT = Path(__file__).resolve().parent.parent.parent
XT_PATH = ROOT / "reports" / "rs4z" / "xt.json"
TEAM = np.array([0, 0, 0, 0, 1, 1, 1, 1])

TERMS = {
    "goal": dict(kind="sparse", scale=1.0, why="objetivo: diferencia de gol", exploit="ninguno (suma cero)",
                 retire="nunca"),
    "result": dict(kind="terminal", scale=0.3, why="jugar para ganar; el crítico ve marcador y reloj",
                   exploit="suma cero: no repite la trampa anti-0-0 (que penalizaba a ambos)", retire="nunca"),
    "drill": dict(kind="terminal", scale=1.0, why="define la tarea del ejercicio",
                  exploit="éxitos auditados: 'recuperar' exige toque propio sostenido 1 s; sacarla no cuenta",
                  retire="cuando el ejercicio sale de la mezcla"),
    "threat": dict(kind="pbrs", scale=0.25, why="señal densa de progreso: amenaza esperada humana (xT)",
                   exploit="pelotazos sin control: un ciclo no rinde (PBRS); suma cero",
                   retire="lineal a 0 al aprobar la compuerta de goles de S5"),
    "access": dict(kind="pbrs", scale=0.05, why="presión/posicionamiento: el equipo que llega antes a la pelota",
                   exploit="que todos persigan no mejora el margen de equipo (sólo cuenta el mínimo)",
                   retire="a 0 al salir de S4"),
    "ball": dict(kind="pbrs", scale=0.5, why="exploración en 1v0: acercarse a la pelota",
                 exploit="quedarse cerca sin jugar no rinde (PBRS); sólo en ejercicios 1v0",
                 retire="a 0 al salir de S1"),
    "crowd": dict(kind="pbrs", scale=0.02, why="contingencia contra la aglomeración",
                  exploit="sólo penaliza estar a <120 px de un compañero: no premia colgarse",
                  retire="apagado salvo que el detector falle en S4/S5"),
}


# ----------------------------------------------------------------------------- xT
class XT:
    """Tabla de amenaza esperada: E[goles a favor − en contra en 10 s | pelota, posesión], marco del poseedor."""

    def __init__(self, path=XT_PATH):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.grid = np.asarray(data["grid"], dtype=np.float64)       # (nx, ny)
        self.x0, self.dx = data["x0"], data["dx"]
        self.y0, self.dy = data["y0"], data["dy"]
        self.mean = data["mean"]
        self.meta = {k: v for k, v in data.items() if k != "grid"}

    def value(self, x, y):
        return _xt_lookup(self.grid, self.x0, self.dx, self.y0, self.dy, np.asarray(x, dtype=np.float64),
                          np.asarray(y, dtype=np.float64)) - self.mean


@njit(cache=True)
def _xt_lookup(grid, x0, dx, y0, dy, x, y):
    out = np.empty(x.shape)
    nx, ny = grid.shape
    for i in range(x.size):
        fx = (x.flat[i] - x0) / dx - 0.5
        fy = (y.flat[i] - y0) / dy - 0.5
        fx = min(max(fx, 0.0), nx - 1.0001)
        fy = min(max(fy, 0.0), ny - 1.0001)
        ix = int(fx)
        iy = int(fy)
        ax = fx - ix
        ay = fy - iy
        out.flat[i] = ((grid[ix, iy] * (1 - ax) + grid[ix + 1, iy] * ax) * (1 - ay)
                       + (grid[ix, iy + 1] * (1 - ax) + grid[ix + 1, iy + 1] * ax) * ay)
    return out


# ----------------------------------------------------------------------------- potenciales
@njit(cache=True, parallel=True)
def _access(pos, vel, active, fp, out):
    """Φ_access por partido para el rojo: tanh((t_azul − t_rojo)/10) con tiempo de llegada a la pelota."""
    N = pos.shape[0]
    for n in prange(N):
        best = np.array([1e9, 1e9])
        for p in range(8):
            if not active[n, p]:
                continue
            k = fp + p
            dx = pos[n, 0, 0] - pos[n, k, 0]
            dy = pos[n, 0, 1] - pos[n, k, 1]
            d = math.sqrt(dx * dx + dy * dy)
            along = (vel[n, k, 0] * dx + vel[n, k, 1] * dy) / max(d, 1e-9)
            # tiempo aproximado de llegada (velocidad máxima 3 px/tick, arranque desde `along`)
            t = max(0.0, d - 23.0) / 3.0 + max(0.0, (3.0 - along)) * 6.0
            team = 0 if p < 4 else 1
            if t < best[team]:
                best[team] = t
        if best[0] > 1e8 and best[1] > 1e8:
            out[n] = 0.0
        else:
            out[n] = math.tanh((min(best[1], 400.0) - min(best[0], 400.0)) / 10.0)


@njit(cache=True, parallel=True)
def _crowd(pos, active, fp, out):
    N = pos.shape[0]
    for n in prange(N):
        for t in range(2):
            acc = 0.0
            for a in range(4):
                pa = 4 * t + a
                if not active[n, pa]:
                    continue
                for b in range(a + 1, 4):
                    pb = 4 * t + b
                    if not active[n, pb]:
                        continue
                    d = math.hypot(pos[n, fp + pa, 0] - pos[n, fp + pb, 0], pos[n, fp + pa, 1] - pos[n, fp + pb, 1])
                    if d < 120.0:
                        acc -= 1.0 - d / 120.0
            out[n, t] = acc


@dataclass
class Coefs:
    goal: float = 1.0
    result: float = 0.0
    drill: float = 1.0
    threat: float = 0.0
    access: float = 0.0
    ball: float = 0.0
    crowd: float = 0.0


class Rewards:
    def __init__(self, env, gamma, xt=None):
        self.env = env
        self.gamma = gamma
        self.xt = xt
        self.coefs = Coefs()
        self._acc = np.zeros(env.N)
        self._crowd = np.zeros((env.N, 2))

    def potentials(self, ball_task):
        """Φ total por partido y equipo (N, 2), ya multiplicado por los coeficientes vigentes."""
        env, c = self.env, self.coefs
        phi = np.zeros((env.N, 2))
        if c.threat and self.xt is not None:
            last = env.ri[:, K.RI_LAST]
            bx, by = env.ball_pos[:, 0], env.ball_pos[:, 1]
            v_red = self.xt.value(bx, by)       # rojo en posesión (marco rojo)
            v_blue = self.xt.value(-bx, by)     # azul en posesión (marco azul)
            red = np.where(last == 0, v_red, np.where(last == 1, -v_blue, 0.0))
            phi[:, 0] += c.threat * red
            phi[:, 1] -= c.threat * red
        if c.access:
            _access(env.pos, env.vel, env.active, env.fp, self._acc)
            phi[:, 0] += c.access * self._acc
            phi[:, 1] -= c.access * self._acc
        if c.ball:
            # sólo en ejercicios sin rivales: −distancia del jugador más cercano del equipo a la pelota
            d = np.hypot(*(env.player_pos - env.ball_pos[:, None]).transpose(2, 0, 1))
            d = np.where(env.active, d, np.inf)
            for t in (0, 1):
                m = d[:, TEAM == t].min(axis=1)
                phi[:, t] += np.where(ball_task & np.isfinite(m), -c.ball * np.minimum(m, 2300.0) / 2300.0, 0.0)
        if c.crowd:
            _crowd(env.pos, env.active, env.fp, self._crowd)
            phi += c.crowd * self._crowd
        return phi

    def shaping(self, phi_before, phi_after, terminal):
        """F por partido y equipo: γ·Φ(s') − Φ(s), con Φ(s') = 0 en terminales reales."""
        nxt = np.where(terminal[:, None], 0.0, phi_after)
        return self.gamma * nxt - phi_before


def per_player(team_values):
    """(N, 2) por equipo → (N, 8) por jugador."""
    return team_values[:, TEAM]
