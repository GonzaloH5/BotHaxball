"""Callbacks RS4 compilados: mismas condiciones, float64 sin fastmath ni RNG."""
import math

import numpy as np
from numba import njit


@njit(cache=True)
def record_defensive_touch(touches, positions, ball, teams, kickoff, owner,
                           open_before, ticks, goal_x, context_team, context_tick):
    for row in range(len(touches)):
        count, player = 0, -1
        for p in range(len(teams)):
            if touches[row, p]:
                count += 1
                player = p
        if count == 0:
            continue  # conservar el contexto cuando no hubo un nuevo contacto
        context_team[row] = -1
        if count != 1 or kickoff[row] or owner[row] >= 0 or not open_before[row]:
            continue
        team = teams[player]
        own_x = ball[row, 0] * (1 if team == 0 else -1)
        if not own_x < -0.55 * goal_x:
            continue
        pressure = np.inf
        for p in range(len(teams)):
            if teams[p] != team:
                dx = positions[row, p, 0] - ball[row, 0]
                dy = positions[row, p, 1] - ball[row, 1]
                # Igual que np.linalg.norm(..., axis=1), sin reasociar operaciones.
                distance = math.sqrt(dx * dx + dy * dy)
                pressure = min(pressure, distance)
        if pressure <= 0.12 * goal_x:
            context_team[row] = team
            context_tick[row] = ticks[row]


@njit(cache=True)
def restart_potential(positions, teams, owner, origin, goal_x, coefficient):
    result = np.zeros((len(owner), len(teams)), dtype=np.float64)
    if coefficient == 0:
        return result
    for row in range(len(owner)):
        if owner[row] < 0:
            continue
        nearest = np.inf
        for p in range(len(teams)):
            if teams[p] == owner[row]:
                dx = positions[row, p, 0] - origin[row, 0]
                dy = positions[row, p, 1] - origin[row, 1]
                nearest = min(nearest, math.sqrt(dx * dx + dy * dy))
        value = coefficient * (1 - min(1.0, max(0.0, nearest / (2 * goal_x))))
        for p in range(len(teams)):
            if teams[p] == owner[row]:
                result[row, p] = value
    return result
