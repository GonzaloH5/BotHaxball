"""Guía geométrica RS4: roles intercambiables y amenaza, NO probabilidad de gol.

Sólo geometría observable. Sin historial de posesión privado, RNG ni roles fijos.
"""
import math
from itertools import permutations

import numpy as np
from numba import njit

ASSIGNMENTS = np.array(list(permutations(range(4))), dtype=np.int64)


@njit(cache=True)
def _receiver_scores(ball, own, opp, gx):
    scores = np.zeros(4)
    for p in range(4):
        dx, dy = own[p, 0] - ball[0], own[p, 1] - ball[1]
        length = math.hypot(dx, dy)
        if length < .08 * gx:
            continue  # carrier/crowd cannot also count as a receiver
        lane = 1.0
        for q in range(4):
            t = ((opp[q, 0] - ball[0]) * dx + (opp[q, 1] - ball[1]) * dy) / (length * length)
            if .05 < t <= 1.1:
                distance = math.hypot(opp[q, 0] - ball[0] - min(t, 1.) * dx,
                                      opp[q, 1] - ball[1] - min(t, 1.) * dy)
                lane = min(lane, 1. - math.exp(-distance / (.045 * gx)))
        scores[p] = lane * math.exp(-length / (.8 * gx))
    return scores


@njit(cache=True)
def passing_outlets(ball, own, opp, gx):
    """Two distinct, reachable passing lanes; separation alone is not support."""
    scores = _receiver_scores(ball, own, opp, gx)
    best = 0.0
    for p in range(4):
        for q in range(p + 1, 4):
            separation = min(1., math.hypot(own[p, 0] - own[q, 0], own[p, 1] - own[q, 1]) / (.22 * gx))
            best = max(best, math.sqrt(scores[p] * scores[q]) * separation)
    return best


@njit(cache=True)
def penetrating_outlet(ball, own, opp, gx, gh):
    """Reachable off-ball receiver near goal, ahead or available for a cutback.

    Mere area occupancy is insufficient: a marked/blocked receiver scores less.
    This is a bounded geometric potential, never a per-tick entry bonus.
    """
    scores = _receiver_scores(ball, own, opp, gx)
    best = 0.0
    for p in range(4):
        depth = min(1., max(0., (own[p, 0] / gx - .50) / .35))
        central = math.exp(-max(0., abs(own[p, 1]) - gh) / max(gh, .05 * gx))
        space = 1.0
        for q in range(4):
            distance = math.hypot(own[p, 0] - opp[q, 0], own[p, 1] - opp[q, 1])
            space = min(space, 1. - math.exp(-distance / (.06 * gx)))
        best = max(best, scores[p] * depth * central * space)
    return best


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
def dynamic_targets(ball, control, gx, fh, gh, wing=1.0):
    """Último hombre móvil, portador/presión, balance y profundidad.

    Control geométrico, NO posesión real. Evaluar ambos wings evita sesgo vertical.
    El bloque sólo sube mucho si pelota y ventaja de acceso permiten atacar.
    """
    advance = min(1.0, max(0.0, .5 + .5 * ball[0] / gx))
    attack = control * advance
    targets = np.empty((4, 2))
    targets[0, 0] = max(-.96 * gx, min((-.94 + .72 * attack) * gx, ball[0] - .48 * gx))
    targets[0, 1] = max(-.6 * gh, min(.6 * gh, .22 * ball[1]))
    targets[1, 0] = max(-.9 * gx, min(.9 * gx, ball[0] - .06 * (1 - control) * gx))
    targets[1, 1] = max(-.9 * fh, min(.9 * fh, ball[1]))
    targets[2, 0] = max(-.82 * gx, min(.75 * gx, ball[0] - (.30 - .14 * control) * gx))
    targets[3, 0] = max(-.78 * gx, min(.88 * gx, ball[0] + (.02 + .36 * control) * gx))
    width = (.16 + .18 * control) * fh
    targets[2, 1] = max(-.8 * fh, min(.8 * fh, .40 * ball[1] - wing * width))
    targets[3, 1] = max(-.8 * fh, min(.8 * fh, .40 * ball[1] + wing * width))
    return targets


@njit(cache=True)
def _formation_costs(own, targets, gx, version):
    costs = np.empty((4, 4))
    weights = (0.20, 0.25, 0.25, 0.30) if version == 2 else (0.30, 0.20, 0.25, 0.25)
    for p in range(4):
        for role in range(4):
            distance = math.hypot(own[p, 0] - targets[role, 0], own[p, 1] - targets[role, 1])
            costs[p, role] = weights[role] * (1 - math.exp(-max(0.0, distance - 0.06 * gx) / (0.22 * gx)))
    return costs


@njit(cache=True)
def _assignment_score(costs):
    best = 1e30
    for assignment in ASSIGNMENTS:
        cost = 0.0
        for role in range(4):
            cost += costs[assignment[role], role]
        best = min(best, cost)
    return 1.0 - best


@njit(cache=True)
def _formation_score(own, targets, gx, version):
    return _assignment_score(_formation_costs(own, targets, gx, version))


@njit(cache=True)
def _formation_v2_score(own, left, right, gx):
    left_costs = _formation_costs(own, left, gx, 2)
    right_costs = left_costs.copy()
    # GK y presión son idénticos entre bandas: sólo cambian balance y profundidad.
    for p in range(4):
        for role in (2, 3):
            weight = 0.25 if role == 2 else 0.30
            distance = math.hypot(own[p, 0] - right[role, 0], own[p, 1] - right[role, 1])
            right_costs[p, role] = weight * (1 - math.exp(-max(0.0, distance - 0.06 * gx) / (0.22 * gx)))
    return max(_assignment_score(left_costs), _assignment_score(right_costs))


@njit(cache=True)
def components(player_pos, player_team, ball_pos, gx, fh, gh, version=1):
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
            if version >= 2:
                left = dynamic_targets(ball, control, gx, fh, gh, -1.0)
                right = dynamic_targets(ball, control, gx, fh, gh, 1.0)
                # Only with a credible access advantage in the attacking half:
                # one runner offers a finishing lane instead of a second wide
                # rebound station. Keeper/balance targets remain unchanged.
                penetration = (min(1., max(0., (control - .55) / .30))
                               * min(1., max(0., (ball[0] / gx - .25) / .35)))
                if version >= 4:
                    for target in (left, right):
                        finishing_y = max(-gh, min(gh, target[3, 1]))
                        target[3, 1] += penetration * (finishing_y - target[3, 1])
                result[row, team, 0] = _formation_v2_score(own, left, right, gx)
                if version >= 3:
                    # Geometry constraints rather than identities: one accesses
                    # the ball, a different player covers depth, and two offer
                    # distinct lateral/progressive outlets. The assignment is
                    # recomputed from positions every decision.
                    pressure = 1.0 - math.exp(-near_a / (.14 * gx))
                    depth = 0.0
                    for p in range(4):
                        if own[p, 0] < ball[0] - .14 * gx:
                            depth = max(depth, math.exp(-abs(own[p, 1] - .35 * ball[1]) / (.45 * fh)))
                    outlets = passing_outlets(ball, own, opp, gx)
                    coverage = (1 - control) * depth + control * (.5 * depth + .5 * outlets)
                    if version >= 4:
                        runner = penetrating_outlet(ball, own, opp, gx, gh)
                        coverage += control * penetration * (.35 * runner - .25 * depth - .10 * outlets)
                    result[row, team, 0] = (.55 * result[row, team, 0]
                                            + .25 * coverage + .20 * (1 - pressure))
            else:
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
                result[row, team, 0] = _formation_score(own, targets, gx, version)
            result[row, team, 1] = _threat(ball, own, opp, gx, gh)
            result[row, team, 2] = _threat(ball * np.array([-1., 1.]),
                                          opp * np.array([-1., 1.]), own * np.array([-1., 1.]), gx, gh)
    return result
