"""Guía geométrica RS4: roles intercambiables y amenaza, NO probabilidad de gol.

Sólo geometría observable. Sin historial de posesión privado, RNG ni roles fijos.
"""
import math
from itertools import permutations

import numpy as np
from numba import njit

ASSIGNMENTS = np.array(list(permutations(range(4))), dtype=np.int64)


@njit(cache=True)
def _threat(ball, attackers, defenders, gx, gh):
    bx, by = ball[0], ball[1]
    dx = max(gx - bx, 1e-6)
    opening = max(0.0, math.atan2(gh - by, dx) - math.atan2(-gh - by, dx))
    angle = min(1.0, opening / (math.pi / 2))
    proximity = math.exp(-math.hypot(gx - bx, by) / (0.50 * gx))
    near_a, near_d = 1e30, 1e30
    for p in range(4):
        near_a = min(near_a, math.hypot(attackers[p, 0] - bx, attackers[p, 1] - by))
        near_d = min(near_d, math.hypot(defenders[p, 0] - bx, defenders[p, 1] - by))
    control = 1.0 / (1.0 + math.exp(max(-10.0, min(10.0, (near_a - near_d) / (0.10 * gx)))))
    openness = 0.0
    for lane in range(3):
        goal_y = (lane - 1) * 0.7 * gh
        vx, vy = gx - bx, goal_y - by
        length2 = max(vx * vx + vy * vy, 1e-9)
        coverage = 0.0
        for p in range(4):
            t = ((defenders[p, 0] - bx) * vx + (defenders[p, 1] - by) * vy) / length2
            if 0.05 < t <= 1.0:
                distance = math.hypot(defenders[p, 0] - bx - t * vx,
                                      defenders[p, 1] - by - t * vy)
                coverage = max(coverage, math.exp(-distance / (0.055 * gx)))
        openness += (1.0 - coverage) / 3
    return proximity * angle * (0.25 + 0.75 * control) * (0.20 + 0.80 * openness)


@njit(cache=True)
def components(player_pos, player_team, ball_pos, gx, fh, gh):
    """(N,2,3): estructura, amenaza rival, peligro propio; cada término en [0,1]."""
    n = len(ball_pos)
    result = np.empty((n, 2, 3), dtype=np.float64)
    for row in range(n):
        for team in range(2):
            sign = 1.0 if team == 0 else -1.0
            own, opp = np.empty((4, 2)), np.empty((4, 2))
            a, b = 0, 0
            for p in range(8):
                if player_team[p] == team:
                    own[a, 0], own[a, 1] = sign * player_pos[row, p, 0], player_pos[row, p, 1]
                    a += 1
                else:
                    opp[b, 0], opp[b, 1] = sign * player_pos[row, p, 0], player_pos[row, p, 1]
                    b += 1
            ball = np.array([sign * ball_pos[row, 0], ball_pos[row, 1]])
            near_a, near_b = 1e30, 1e30
            for p in range(4):
                near_a = min(near_a, math.hypot(own[p, 0] - ball[0], own[p, 1] - ball[1]))
                near_b = min(near_b, math.hypot(opp[p, 0] - ball[0], opp[p, 1] - ball[1]))
            control = 1.0 / (1.0 + math.exp(max(-10.0, min(10.0, (near_a - near_b) / (0.10 * gx)))))
            advance = min(1.0, max(0.0, 0.5 + 0.5 * ball[0] / gx))
            attack = 0.6 * control + 0.4 * advance
            targets = np.empty((4, 2))
            # Arquero/líbero, presión/conductor, dos apoyos/coberturas diagonales.
            targets[0, 0] = (-0.94 + 0.10 * attack) * gx
            targets[0, 1] = max(-0.6 * gh, min(0.6 * gh, 0.30 * ball[1]))
            targets[1, 0] = max(-0.90 * gx, min(0.90 * gx, ball[0]))
            targets[1, 1] = max(-0.90 * fh, min(0.90 * fh, ball[1]))
            support_x = max(-0.78 * gx, min(0.82 * gx, ball[0] + (-0.25 + 0.35 * attack) * gx))
            width = (0.20 + 0.16 * attack) * fh
            for role in (2, 3):
                targets[role, 0] = support_x
                targets[role, 1] = max(-0.80 * fh, min(0.80 * fh, 0.40 * ball[1] + (2 * role - 5) * width))
            costs = np.empty((4, 4))
            weights = (0.30, 0.20, 0.25, 0.25)
            for p in range(4):
                for role in range(4):
                    distance = math.hypot(own[p, 0] - targets[role, 0], own[p, 1] - targets[role, 1])
                    # Zona, no punto exacto. Un jugador no puede llenar dos roles.
                    costs[p, role] = weights[role] * (1 - math.exp(-max(0.0, distance - 0.06 * gx) / (0.22 * gx)))
            best = 1e30
            for assignment in ASSIGNMENTS:
                cost = 0.0
                for role in range(4):
                    cost += costs[assignment[role], role]
                best = min(best, cost)
            result[row, team, 0] = 1.0 - best
            result[row, team, 1] = _threat(ball, own, opp, gx, gh)
            result[row, team, 2] = _threat(ball * np.array([-1., 1.]),
                                          opp * np.array([-1., 1.]), own * np.array([-1., 1.]), gx, gh)
    return result
