"""Callbacks pequeños del rollout: mismas reglas/rewards, float64 sin fastmath.

Se ejecutan tick por tick. No consumen RNG ni omiten eventos de cooperación.
"""
import math

import numpy as np
from numba import njit


@njit(cache=True)
def protect_pieces(pp, pv, teams, owner, kind, origin, distance, front, side):
    for n in range(len(owner)):
        if owner[n] < 0:
            continue
        sx = 1.0 if origin[n, 0] >= 0 else -1.0
        for p in range(len(teams)):
            if teams[p] == owner[n]:
                continue
            if kind[n] == 3 and sx * pp[n, p, 0] > front and abs(pp[n, p, 1]) < side:
                pp[n, p, 0] = sx * front
                pv[n, p, 0] = pv[n, p, 1] = 0.0
            dx, dy = pp[n, p, 0] - origin[n, 0], pp[n, p, 1] - origin[n, 1]
            norm = np.hypot(dx, dy)
            if norm < distance:
                denominator = max(norm, 1e-9)
                ux, uy = dx / denominator, dy / denominator
                if norm <= 1e-6:
                    ux, uy = -sx, 0.0
                pp[n, p, 0] = origin[n, 0] + ux * distance
                pp[n, p, 1] = origin[n, 1] + uy * distance
                pv[n, p, 0] = pv[n, p, 1] = 0.0


@njit(cache=True)
def piece_pre(actions, pp, pv, teams, owner, kind, origin, distance, front, side):
    protect_pieces(pp, pv, teams, owner, kind, origin, distance, front, side)
    out = actions.copy()
    for n in range(len(owner)):
        if owner[n] >= 0:
            for p in range(len(teams)):
                if teams[p] != owner[n]:
                    out[n, p] %= 9
    return out


@njit(cache=True)
def piece_post(goal, ball, velocity, pp, pv, teams, kicked, owner, kind, ticks,
               limit, origin, kick_speed, distance, front, side):
    previous = owner.copy()
    timed_out = np.zeros(len(owner), dtype=np.bool_)
    for n in range(len(owner)):
        if owner[n] < 0:
            continue
        ticks[n] += 1
        own_kick = False
        for p in range(len(teams)):
            if kicked[n, p] and teams[p] == owner[n]:
                own_kick = True
        if own_kick and kind[n] == 3 and kick_speed > 0:
            speed = math.sqrt(velocity[n, 0] ** 2 + velocity[n, 1] ** 2)
            if speed > 1e-6:
                scale = max(1.0, kick_speed / speed)
                velocity[n, 0] *= scale
                velocity[n, 1] *= scale
        timed_out[n] = not own_kick and ticks[n] >= limit[n] and goal[n] == 0
        if own_kick or timed_out[n] or goal[n] != 0:
            owner[n] = -1
            kind[n] = 0
        else:
            ball[n, 0], ball[n, 1] = origin[n, 0], origin[n, 1]
            velocity[n, 0] = velocity[n, 1] = 0.0
    protect_pieces(pp, pv, teams, owner, kind, origin, distance, front, side)
    return timed_out, previous


@njit(cache=True)
def advance_passes(pp, teams, sender, receiver, pending_team, age, progress,
                   last_sender, last_receiver, spent, ticks, hold, field_w,
                   success, value, chain_reward, return_threshold, cap, bonus, events, participant=0.0):
    n_rows, players, _ = pp.shape
    for n in range(n_rows):
        if pending_team[n] < 0:
            continue
        age[n] += ticks
        if age[n] < hold:
            continue
        s, r, t = sender[n], receiver[n], pending_team[n]
        receiver_space, sender_space = np.inf, np.inf
        for p in range(players):
            if teams[p] != t:
                dx, dy = pp[n, p, 0] - pp[n, r, 0], pp[n, p, 1] - pp[n, r, 1]
                receiver_space = min(receiver_space, math.sqrt(dx * dx + dy * dy))
                dx, dy = pp[n, p, 0] - pp[n, s, 0], pp[n, p, 1] - pp[n, s, 1]
                sender_space = min(sender_space, math.sqrt(dx * dx + dy * dy))
        space_gain = 0.0
        if receiver_space != np.inf:
            space_gain = min(1.0, max(0.0, (receiver_space - sender_space) / (0.15 * field_w)))
        usefulness = 0.65 * progress[n] + 0.35 * space_gain
        returned = last_sender[n] == r and last_receiver[n] == s
        amount = (success if usefulness > 0.05 else 0.0) + value * usefulness
        if returned and usefulness < return_threshold:
            amount = 0.0
        chain = (not returned and last_receiver[n] == s and last_sender[n] >= 0
                 and last_sender[n] != r and usefulness > 0.10)
        if chain:
            amount += chain_reward
            events[2, n, t] += 1
        paid = min(amount, max(0.0, cap - spent[n, t]))
        if paid > 0:
            for p in range(players):
                if teams[p] == t:
                    bonus[n, p] += paid
            spent[n, t] += paid
            credit = min(participant, max(0.0, cap - spent[n, t]))
            bonus[n, s] += credit
            bonus[n, r] += credit
            spent[n, t] += credit
        events[0, n, t] += 1
        if usefulness > 0.10:
            events[1, n, t] += 1
        last_sender[n], last_receiver[n] = s, r
        sender[n] = receiver[n] = pending_team[n] = -1
        age[n] = 0
        progress[n] = 0.0
    return bonus, events


@njit(cache=True)
def touch_passes(touch, ball, teams, kickoff, piece_owner, last_player, last_pos,
                 sender, receiver, pending_team, age, progress, last_sender,
                 last_receiver, spent, field_w, min_fraction, bonus, events):
    n_rows, players = touch.shape
    if players <= 2:
        return bonus, events
    for n in range(n_rows):
        count, player = 0, -1
        for p in range(players):
            if touch[n, p]:
                count += 1
                player = p
        if count == 0:
            continue
        previous = last_player[n]
        if count != 1:
            if previous >= 0 and touch[n, previous]:
                last_pos[n] = ball[n]
            continue
        if previous == player:
            last_pos[n] = ball[n]
            continue
        if previous >= 0:
            t, prev_team = teams[player], teams[previous]
            if t == prev_team:
                dx, dy = ball[n, 0] - last_pos[n, 0], ball[n, 1] - last_pos[n, 1]
                travel = math.sqrt(dx * dx + dy * dy)
                if travel >= min_fraction * field_w and not kickoff[n] and piece_owner[n] < 0:
                    sign = 1.0 if t == 0 else -1.0
                    progress[n] = min(1.0, max(0.0, sign * dx / (0.25 * field_w)))
                    sender[n], receiver[n], pending_team[n] = previous, player, t
                    age[n] = 0
            else:
                events[3, n, prev_team] += 1
                last_sender[n] = last_receiver[n] = -1
                sender[n] = receiver[n] = pending_team[n] = -1
                age[n] = 0
                spent[n, t] = 0.0
        last_player[n] = player
        last_pos[n] = ball[n]
    return bonus, events
