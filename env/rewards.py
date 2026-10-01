"""Recompensas.

- Gol: +1 al equipo que anota, -1 al que recibe (se comparte entre compañeros).
- Shaping basado en potencial (Ng et al. 1999: r = γΦ(s') - Φ(s)), que NO cambia la política
  óptima y por eso es resistente al reward hacking. Se multiplica por `shaping_coef`, que el
  entrenamiento baja a 0 con el tiempo.
    Φ_ball  = progreso de la pelota hacia el arco rival (en coordenadas del equipo)
    Φ_near  = -distancia jugador-pelota (sólo al principio sirve; peso chico)
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations

import numpy as np

_COVER_ASSIGNMENTS = np.array(list(permutations(range(4), 3)))


@dataclass
class RewardConfig:
    goal: float = 1.0
    shaping_coef: float = 1.0      # multiplicador global del shaping (se decae desde el trainer)
    w_ball_progress: float = 0.25   # Φ_ball
    w_near_ball: float = 0.05       # Φ_near
    kick_to_goal: float = 0.02      # bonus pequeño por patear con la pelota saliendo hacia el arco
    kickoff_stall: float = 0.5      # penalidad al equipo que no saca a tiempo (evita "trabar" el partido)
    w_spread: float = 0.3           # Φ de separación entre compañeros (sólo con equipos >1)
    out_penalty: float = 0.1        # pelota afuera: -out_penalty al equipo que la tocó último
    kickoff_approach: float = 1.0   # durante el saque propio: premio por acercarse a la pelota
                                    # (progreso en fracción de spawnDistance; llegar ~ +1). No decae.
    gamma: float = 0.995
    w_defense_support: float = 0.0  # opt-in por tarea; cobertura, no persecución colectiva
    defense_shaping_floor: float = 0.0
    rs4_tactical_coef: float = 0.0  # opt-in sólo rs_one 4v4; guía conjunta acotada

    # Cooperación global: sólo se usa con equipos de 2+ jugadores.
    team_pass_success: float = 0.0015
    team_pass_value: float = 0.003
    team_pass_chain: float = 0.0015
    team_pass_min_dist_frac: float = 0.04
    team_pass_hold_ticks: int = 12
    team_pass_possession_cap: float = 0.01
    team_pass_return_min_usefulness: float = 0.25
    team_spread_floor: float = 0.05

    # Córners: premio pequeño por ejecutar el saque hacia dentro de la cancha.
    # Se deja en 0 globalmente y env/tasks.py lo activa sólo en tareas configuradas.
    corner_execute: float = 0.0
    corner_hold_ticks: int = 12
    corner_min_inward_frac: float = 0.04

def potentials(ball_x_own, ball_y_own, player_ball_dist, goal_x: float, field_w: float):
    """Φ por agente. `ball_*_own` en coords del equipo del agente (ataca hacia +x)."""
    # distancia de la pelota al arco rival vs al propio, en [−1, 1]
    d_opp = np.hypot(goal_x - ball_x_own, ball_y_own)
    d_own = np.hypot(-goal_x - ball_x_own, ball_y_own)
    phi_ball = (d_own - d_opp) / (2 * goal_x)
    phi_near = -player_ball_dist / (2 * field_w)
    return phi_ball, phi_near


def defense_support_potential(players, ball, goal_x, field_h):
    """Cobertura 6v6 en coordenadas propias (+x ataca), en [0,1].

    Reserva al más cercano al arco como arquero y al siguiente más cercano a
    la pelota como presionante. Tres jugadores DIFERENTES cubren carriles entre
    pelota y arco; queda un jugador libre. No fija identidades ni usa posesión
    privada. Es una guía geométrica, no una táctica óptima demostrada.
    """
    n, team_size, _ = players.shape
    if team_size != 6:
        return np.zeros(n)
    danger = np.clip((-ball[:, 0] / goal_x - 0.10) / 0.45, 0.0, 1.0)
    keeper_dist = np.linalg.norm(players - np.array([-goal_x, 0.0]), axis=-1)
    keeper = keeper_dist.argmin(axis=1)
    ball_dist = np.linalg.norm(players - ball[:, None], axis=-1)
    rows = np.arange(n)
    ball_dist[rows, keeper] = np.inf
    pressure = ball_dist.argmin(axis=1)
    available = (np.arange(6)[None, :] != keeper[:, None]) & (
        np.arange(6)[None, :] != pressure[:, None])
    cover = players[available].reshape(n, 4, 2)
    targets = np.zeros((n, 3, 2))
    targets[..., 0] = np.clip((ball[:, 0] - goal_x) * 0.5,
                            -0.78 * goal_x, -0.20 * goal_x)[:, None]
    targets[..., 1] = np.clip(0.35 * ball[:, None, 1] +
                             field_h * np.array([-0.28, 0.0, 0.28]),
                             -0.75 * field_h, 0.75 * field_h)
    distance = np.linalg.norm(cover[:, :, None] - targets[:, None], axis=-1)
    # 4P3 = 24 asignaciones: un único jugador nunca llena los tres carriles.
    costs = distance[:, _COVER_ASSIGNMENTS, np.arange(3)].mean(axis=-1).min(axis=1)
    support = np.exp(-costs / (0.25 * goal_x))
    keeper_target = np.column_stack((np.full(n, -0.92 * goal_x),
                                    np.clip(0.25 * ball[:, 1], -0.16 * field_h, 0.16 * field_h)))
    keeper_score = np.exp(-np.linalg.norm(players[rows, keeper] - keeper_target, axis=-1)
                          / (0.18 * goal_x))
    return danger * (0.75 * support + 0.25 * keeper_score)
