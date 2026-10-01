"""Geometría de rewards sin temporales N x T x T ni equipos de hilos pequeños.

Mismas distancias y reducción por equipo; float64, sin fastmath ni RNG.
"""
import math

import numpy as np
from numba import njit


@njit(cache=True)
def team_ball_dist(players, ball, team):
    n, p, _ = players.shape
    out = np.empty((n, p), dtype=np.float64)
    for row in range(n):
        nearest = np.full(2, np.inf)
        for player in range(p):
            dx = players[row, player, 0] - ball[row, 0]
            dy = players[row, player, 1] - ball[row, 1]
            distance = math.sqrt(dx * dx + dy * dy)
            nearest[team[player]] = min(nearest[team[player]], distance)
        for player in range(p):
            out[row, player] = nearest[team[player]]
    return out


@njit(cache=True)
def spread_potential(players, team, field_w):
    n, p, _ = players.shape
    out = np.zeros((n, p), dtype=np.float64)
    if p <= 2:
        return out
    cap = 0.25 * field_w
    for row in range(n):
        sums = np.zeros(2)
        counts = np.zeros(2, dtype=np.int64)
        for player in range(p):
            nearest = np.inf
            for other in range(p):
                if other != player and team[other] == team[player]:
                    dx = players[row, player, 0] - players[row, other, 0]
                    dy = players[row, player, 1] - players[row, other, 1]
                    nearest = min(nearest, math.sqrt(dx * dx + dy * dy))
            sums[team[player]] += min(nearest, cap) / cap
            counts[team[player]] += 1
        for player in range(p):
            out[row, player] = sums[team[player]] / counts[team[player]]
    return out
