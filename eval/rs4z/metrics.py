"""Métricas de juego y detectores de comportamiento, idénticas para simulación y grabaciones humanas.

Se alimentan con muestras a resolución de decisión (cada 3 ticks): posiciones del mundo, quién tocó la
pelota, patadas, goles y estado de saque. Cada equipo se mide en su propio marco (ataca hacia +x).

Definiciones (las mismas se usan para la referencia humana, `tools/rs4z_human_reference.py`):
* pase: dos toques consecutivos de jugadores distintos del mismo equipo, con ≥ 60 px de recorrido
  de la pelota entre ellos; pérdida: el toque siguiente es de un rival. Un saque corta la cadena.
* tiro al arco: patada tras la cual la pelota va hacia el arco rival y su línea pasa a < 30 px del arco.
* estructura (juego abierto, 4 jugadores activos): 3+ detrás de la pelota, dispersión (distancia media
  al centroide), profundidad (rango en x), anchura (rango en y), perfil de distancias 1.º–4.º a la pelota.
* detectores: aglomeración (≥3 a < 150 px de la pelota), colgado (> 800 px de la pelota), flotación
  junto al arco (≥2 a < 250 px del arco propio con la pelota en campo rival), quietud (velocidad < 0,1),
  oscilación (inversiones de velocidad > 90° con rapidez > 0,8, por jugador y minuto).
"""
from __future__ import annotations

import numpy as np

GOAL_X, GOAL_HH = 1162.0, 124.0
TEAM = np.array([0, 0, 0, 0, 1, 1, 1, 1])
SIGN = np.array([1.0, -1.0])
TICKS_PER_SAMPLE = 3


class Metrics:
    """Acumulador para N partidos simultáneos (o una grabación con N=1)."""

    def __init__(self, n):
        self.n = n
        self.prev_toucher = np.full(n, -1, dtype=np.int64)
        self.prev_touch_ball = np.zeros((n, 2))
        self.prev_vel = np.zeros((n, 8, 2))
        self.has_prev_vel = np.zeros(n, dtype=bool)
        self.restart_age = np.full(n, -1, dtype=np.int64)
        self.restart_kind = np.zeros(n, dtype=np.int64)
        z2 = lambda: np.zeros(2)
        self.c = dict(samples=z2(), open_samples=z2(), full_samples=z2(), passes=z2(), turnovers=z2(), shots=z2(),
                      goals=z2(), behind3=z2(), spread=z2(), depth=z2(), width=z2(), ball_x=z2(), clump=z2(),
                      parked=z2(), goal_hug=z2(), idle=z2(), reversals=z2(), player_samples=z2(),
                      rank=np.zeros((2, 4)))
        self.restart_durations = {1: [], 2: [], 3: []}

    def add(self, ball, players, vel, active, toucher, kicked, goal, restart_kind, restart_started, in_play):
        """Una muestra.

        ball (n,2), players/vel (n,8,2) en el mundo; active (n,8); toucher (n,) jugador que tocó la
        pelota en esta muestra (-1 ninguno); kicked (n,8); goal (n,) +1 rojo/-1 azul/0; restart_kind (n,)
        saque activo (0 ninguno); restart_started (n,) bool; in_play (n,) juego abierto (sin saque inicial
        ni saque activo).
        """
        c = self.c
        n = self.n
        # ---------------------------------------------------------------- toques: pases y pérdidas
        for i in range(n):
            if restart_started[i]:
                self.prev_toucher[i] = -1
            q = int(toucher[i])
            if q >= 0:
                p = int(self.prev_toucher[i])
                if p >= 0 and q != p:
                    travel = np.hypot(*(ball[i] - self.prev_touch_ball[i]))
                    if TEAM[q] == TEAM[p]:
                        if travel >= 60.0:
                            c["passes"][TEAM[p]] += 1
                    else:
                        c["turnovers"][TEAM[p]] += 1
                self.prev_toucher[i] = q
                self.prev_touch_ball[i] = ball[i]
            if goal[i] != 0:
                c["goals"][0 if goal[i] > 0 else 1] += 1
                self.prev_toucher[i] = -1
        # ---------------------------------------------------------------- tiros al arco
        for t in (0, 1):
            k = (kicked[:, TEAM == t]).any(axis=1)
            if not k.any():
                continue
            s = SIGN[t]
            bx = ball[:, 0] * s
            by = ball[:, 1]
            bvx = np.zeros(n)  # la velocidad de la pelota se aproxima con la dirección jugador→pelota
            for i in np.flatnonzero(k):
                kicker = np.flatnonzero(kicked[i] & (TEAM == t))[0]
                d = (ball[i] - players[i, kicker]) * np.array([s, 1.0])
                if d[0] <= 0:
                    continue
                y_at = by[i] + d[1] / d[0] * (GOAL_X - bx[i])
                if abs(y_at) < GOAL_HH + 30.0 and GOAL_X - bx[i] < 700.0:
                    c["shots"][t] += 1
        # ---------------------------------------------------------------- saques
        for i in range(n):
            if restart_started[i]:
                self.restart_age[i] = 0
                self.restart_kind[i] = restart_kind[i]
            elif self.restart_age[i] >= 0:
                if restart_kind[i] == 0:
                    if self.restart_kind[i] in self.restart_durations:
                        self.restart_durations[self.restart_kind[i]].append(self.restart_age[i] * TICKS_PER_SAMPLE)
                    self.restart_age[i] = -1
                else:
                    self.restart_age[i] += 1
        # ---------------------------------------------------------------- estructura y detectores
        speed = np.hypot(vel[..., 0], vel[..., 1])
        for t in (0, 1):
            s = SIGN[t]
            sel = TEAM == t
            act = active[:, sel]
            c["samples"][t] += n
            open_ = in_play & act.any(axis=1)
            if not open_.any():
                continue
            x = players[:, sel, 0] * s
            y = players[:, sel, 1]
            bx = ball[:, 0] * s
            by = ball[:, 1]
            d = np.hypot(x - bx[:, None], y - by[:, None])
            d = np.where(act, d, np.nan)
            c["open_samples"][t] += open_.sum()
            # detectores (por equipo y muestra)
            near = (d < 150.0) & act
            c["clump"][t] += ((near.sum(axis=1) >= 3) & open_).sum()
            own_goal = np.hypot(x + GOAL_X, y)
            hug = ((own_goal < 250.0) & act).sum(axis=1) >= 2
            c["goal_hug"][t] += (hug & (bx > 0.0) & open_).sum()
            per = act & open_[:, None]
            c["player_samples"][t] += per.sum()
            c["parked"][t] += ((d > 800.0) & per).sum()
            c["idle"][t] += ((speed[:, sel] < 0.1) & per).sum()
            if self.has_prev_vel.any():
                v0 = self.prev_vel[:, sel]
                v1 = vel[:, sel]
                dot = (v0 * v1).sum(axis=-1)
                rev = (dot < 0) & (np.hypot(v0[..., 0], v0[..., 1]) > 0.8) & (np.hypot(v1[..., 0], v1[..., 1]) > 0.8)
                c["reversals"][t] += (rev & per & self.has_prev_vel[:, None]).sum()
            # estructura humana: sólo con 4 jugadores activos
            full = open_ & (act.sum(axis=1) == 4)
            if full.any():
                xf, yf, bxf = x[full], y[full], bx[full]
                cx, cy = xf.mean(axis=1, keepdims=True), yf.mean(axis=1, keepdims=True)
                c["full_samples"][t] += full.sum()
                c["behind3"][t] += ((xf < bxf[:, None]).sum(axis=1) >= 3).sum()
                c["spread"][t] += np.hypot(xf - cx, yf - cy).mean(axis=1).sum()
                c["depth"][t] += (xf.max(axis=1) - xf.min(axis=1)).sum()
                c["width"][t] += (yf.max(axis=1) - yf.min(axis=1)).sum()
                c["ball_x"][t] += bxf.sum()
                c["rank"][t] += np.sort(d[full], axis=1).sum(axis=0)
        self.prev_vel[:] = vel
        self.has_prev_vel[:] = True

    def summary(self, team=None):
        """Resumen por equipo (o promedio de ambos con team=None)."""
        c = self.c
        teams = (0, 1) if team is None else (team,)
        out = {}
        tot = lambda k: sum(c[k][t] for t in teams)
        minutes = tot("samples") / len(teams) * TICKS_PER_SAMPLE / 3600.0
        full = max(tot("full_samples"), 1)
        ps = max(tot("player_samples"), 1)
        opens = max(tot("open_samples"), 1)
        out["minutes"] = minutes
        out["passes_per_min"] = tot("passes") / max(minutes * len(teams), 1e-9)
        out["turnovers_per_min"] = tot("turnovers") / max(minutes * len(teams), 1e-9)
        out["pass_turnover_ratio"] = tot("passes") / max(tot("turnovers"), 1)
        out["shots_per_min"] = tot("shots") / max(minutes * len(teams), 1e-9)
        out["goals_per_min"] = tot("goals") / max(minutes * len(teams), 1e-9)
        for k in ("behind3", "spread", "depth", "width", "ball_x"):
            out[k] = tot(k) / full
        rank = sum(c["rank"][t] for t in teams) / full
        out.update({f"dist_rank{i + 1}": float(rank[i]) for i in range(4)})
        out["clump"] = tot("clump") / opens
        out["goal_hug"] = tot("goal_hug") / opens
        out["parked"] = tot("parked") / ps
        out["idle"] = tot("idle") / ps
        out["reversals_per_player_min"] = tot("reversals") / max(ps * TICKS_PER_SAMPLE / 3600.0, 1e-9)
        for kind, name in ((1, "lateral"), (2, "corner"), (3, "goal_kick")):
            v = self.restart_durations[kind]
            out[f"{name}_ticks_p50"] = float(np.median(v)) if v else float("nan")
        return {k: float(v) for k, v in out.items()}
