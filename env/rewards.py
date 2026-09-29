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

import numpy as np


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


def potentials(ball_x_own, ball_y_own, player_ball_dist, goal_x: float, field_w: float):
    """Φ por agente. `ball_*_own` en coords del equipo del agente (ataca hacia +x)."""
    # distancia de la pelota al arco rival vs al propio, en [−1, 1]
    d_opp = np.hypot(goal_x - ball_x_own, ball_y_own)
    d_own = np.hypot(-goal_x - ball_x_own, ball_y_own)
    phi_ball = (d_own - d_opp) / (2 * goal_x)
    phi_near = -player_ball_dist / (2 * field_w)
    return phi_ball, phi_near
