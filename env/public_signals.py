"""Public RS4 cue decoder and sampled geometric contact history.

No referee, true last_touch, kick events or restart labels enter this tracker.
The simulator's renderer supplies colors just as a room does. Contact confidence
is deliberately imperfect: proximity is evidence, not an oracle of possession.
"""
import numpy as np
from numba import njit

VERSION = 1
START, WIDTH = 56, 15
RED_COLORS = (0xFF0000, 0xE56E56, 0xEC7458, 0xFF3F34)
BLUE_COLORS = (0x0000FF, 0x5689E5, 0x48BEF9, 0x0FBCF9)
FEATURES = ("restart_visible", "restart_own", "restart_rival", "cue_conflict",
            "ball_cue", "barrier_cue", "lateral_geometry", "corner_geometry",
            "goal_kick_geometry", "contact_own", "contact_rival", "contact_confidence",
            "contact_age", "restart_age", "public_schema")


def color_team(colors):
    colors = np.asarray(colors)
    return np.where(colors == -2, -2, np.where(np.isin(colors, RED_COLORS), 0,
                    np.where(np.isin(colors, BLUE_COLORS), 1, -1))).astype(np.int64)


@njit(cache=True)
def _color(color):
    if color == -2:
        return -2
    if color == 0xFF0000 or color == 0xE56E56 or color == 0xEC7458 or color == 0xFF3F34:
        return 0
    if color == 0x0000FF or color == 0x5689E5 or color == 0x48BEF9 or color == 0x0FBCF9:
        return 1
    return -1


@njit(cache=True)
def _decode(bv, ball_colors, barrier_colors):
    # This is called twice per environment decision: avoid np.isin/norm
    # temporaries and Python dispatch for every palette element.
    ball = np.empty(len(bv), np.int64)
    barrier = np.empty(len(bv), np.int64)
    for row in range(len(bv)):
        ball[row] = _color(ball_colors[row]) if np.hypot(bv[row, 0], bv[row, 1]) <= .05 else -1
        barrier[row] = _color(barrier_colors[row])
    return ball, barrier


@njit(cache=True)
def _sample(bp, bv, pp, teams, radius, ball_radius, ball_cue, barrier_cue, dt, memory):
    for row in range(len(bp)):
        memory[row, 1] += dt
        # A repositioning jump is not evidence of a player touching the ball.
        jump = memory[row, 7] > 0 and np.hypot(bp[row, 0] - memory[row, 5], bp[row, 1] - memory[row, 6]) > 80 + dt * 15
        if jump:
            memory[row, 0], memory[row, 1], memory[row, 2] = -1, 600, 0
        close0, close1 = False, False
        for player in range(len(teams)):
            close = np.hypot(pp[row, player, 0] - bp[row, 0], pp[row, player, 1] - bp[row, 1]) <= radius + ball_radius + 3
            if close:
                if teams[player] == 0:
                    close0 = True
                else:
                    close1 = True
        if not jump:
            if close0 and close1:
                memory[row, 0], memory[row, 1], memory[row, 2] = -1, 0, 0
            elif close0 or close1:
                memory[row, 0], memory[row, 1], memory[row, 2] = (0 if close0 else 1), 0, .6
        a, b = ball_cue[row], barrier_cue[row]
        conflict = b == -2 or (a >= 0 and b >= 0 and a != b)
        owner = -1 if conflict else (a if a >= 0 else b)
        if owner >= 0 and owner == memory[row, 3]:
            memory[row, 4] += dt
        else:
            memory[row, 4] = 0
        memory[row, 3] = owner
        memory[row, 5], memory[row, 6], memory[row, 7] = bp[row, 0], bp[row, 1], 1


@njit(cache=True)
def _features(bp, teams, ball_cue, barrier_cue, gx, fh, memory):
    output = np.empty((len(bp), len(teams), WIDTH), np.float32)
    for row in range(len(bp)):
        a, b = ball_cue[row], barrier_cue[row]
        conflict = b == -2 or (a >= 0 and b >= 0 and a != b)
        owner = -1 if conflict else (a if a >= 0 else b)
        visible = a >= 0 or b >= 0 or b == -2
        by, bx = abs(bp[row, 1]), abs(bp[row, 0])
        kind = 0
        if visible:
            kind = 2 if by >= .85 * fh and bx >= .85 * gx else (1 if by >= .85 * fh else (3 if bx >= .65 * gx else 0))
        confidence = memory[row, 2] * max(0., 1 - memory[row, 1] / 600)
        for player in range(len(teams)):
            team = teams[player]
            output[row, player, 0] = visible
            output[row, player, 1] = owner == team
            output[row, player, 2] = owner >= 0 and owner != team
            output[row, player, 3] = conflict
            output[row, player, 4] = a >= 0
            output[row, player, 5] = b >= 0 or b == -2
            output[row, player, 6] = kind == 1
            output[row, player, 7] = kind == 2
            output[row, player, 8] = kind == 3
            output[row, player, 9] = memory[row, 0] == team and confidence > 0
            output[row, player, 10] = memory[row, 0] >= 0 and memory[row, 0] != team and confidence > 0
            output[row, player, 11] = confidence
            output[row, player, 12] = min(1., memory[row, 1] / 600)
            output[row, player, 13] = min(1., memory[row, 4] / 600)
            output[row, player, 14] = 1
    return output


class PublicSignalTracker:
    def __init__(self, n):
        self.memory = np.zeros((n, 8), dtype=np.float64)
        self.reset(np.arange(n))

    def reset(self, rows):
        self.memory[rows] = 0
        self.memory[rows, 0] = -1
        self.memory[rows, 1] = 600
        self.memory[rows, 3] = -1

    @staticmethod
    def decode(bp, bv, ball_colors, barrier_colors):
        # A permanently colored moving ball must not fabricate a restart.
        return _decode(bv, ball_colors, barrier_colors)

    def sample(self, bp, bv, pp, teams, radius, ball_radius, ball_colors, barrier_colors, dt):
        a, b = self.decode(bp, bv, ball_colors, barrier_colors)
        _sample(bp, bv, pp, teams, radius, ball_radius, a, b, dt, self.memory)

    def features(self, bp, bv, teams, ball_colors, barrier_colors, gx, fh):
        a, b = self.decode(bp, bv, ball_colors, barrier_colors)
        return _features(bp, teams, a, b, gx, fh, self.memory)
