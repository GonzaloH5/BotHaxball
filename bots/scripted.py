"""Bot por reglas, vectorizado. Sirve de baseline y de primer rival para el RL.

Trabaja en el marco propio del jugador (ataca hacia +x) y devuelve acciones 0..17 en ese marco,
igual que el agente de RL, así que se puede mezclar con él en HaxballEnv.

Estrategia:
  - Si está "detrás" de la pelota respecto al arco rival y bien alineado: ir a la pelota y patear.
  - Si no: ir al punto detrás de la pelota (rodeándola para no hacer gol en contra).
  - Si el rival está más cerca de la pelota y ésta está en campo propio: volver a cubrir el arco.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit

from sim.physics import MOVE_UNIT


def _dir_to_action(d: np.ndarray, stop_eps: float = 2.0) -> np.ndarray:
    """Vector deseado (…,2) -> acción de movimiento 0..8 (la dirección más parecida)."""
    norm = np.linalg.norm(d, axis=-1)
    unit = d / np.maximum(norm, 1e-9)[..., None]
    scores = unit @ MOVE_UNIT[1:].T
    move = 1 + np.argmax(scores, axis=-1)
    return np.where(norm < stop_eps, 0, move)


def scripted_actions(env, players=None, eps=0.0, rng=None, env_indices=None):
    """Calcula sólo partidos seleccionados; conserva los sorteos originales de ruido."""
    if not getattr(env, "optimize_rollout", True):
        actions = _scripted_reference(env, players, eps, rng)
        return actions if env_indices is None else actions[env_indices]
    sim = env.sim
    players = np.arange(sim.P) if players is None else np.asarray(players, dtype=np.int64)
    rows = np.arange(sim.N) if env_indices is None else np.asarray(env_indices, dtype=np.int64)
    actions = _scripted_kernel(rows, players, sim.ball_pos, sim.ball_vel, sim.player_pos,
                               sim.player_team, env.sign, env.goal_x, sim.st.field_half_h,
                               float(sim.st.player["radius"] + sim.st.ball["radius"]), env.T)
    actions = _restart_actions(env, rows, players, actions)
    if eps > 0:
        rng = rng or np.random.default_rng()
        # Antes se sorteaban N x jugadores y después se seleccionaban las filas.
        # Mantener esos sorteos evita cambiar las semillas/currículo por esta optimización.
        shape = (sim.N, len(players))
        random_mask = (rng.random(shape) < eps)[rows]
        random_actions = rng.integers(0, 18, shape)[rows]
        actions = np.where(random_mask, random_actions, actions)
    return actions


def _restart_actions(env, rows, players, actions):
    """Árbitro disponible sólo para el baseline: no añade entradas a la política RL.

    Un único ejecutor por equipo; en lateral patea hacia adentro, no hacia el arco.
    Sustituye roles/defensa de juego abierto únicamente mientras espera un saque.
    """
    sim = env.sim
    owner = np.where(sim.kickoff, sim.kickoff_team, -1).astype(np.int64)
    kind = np.where(sim.kickoff, 6, 0).astype(np.int64)
    excluded = np.zeros((sim.N, sim.P), dtype=np.bool_)
    forced = np.full(sim.N, -1, dtype=np.int64)
    rules = env.rules
    if rules is not None:
        waiting = ((rules.status == 1) & ~rules.throw_kicked) | rules.sp_waiting
        active = waiting & (rules.sp_team >= 0)
        owner[active], kind[active] = rules.sp_team[active], rules.status[active]
        excluded = rules.expelled
        forced = np.where(rules.status == 5, rules.pen_kicker, -1)
    elif env.out_of_bounds:
        active = env.setpiece_team >= 0
        owner[active], kind[active] = env.setpiece_team[active], env.setpiece_kind[active]
    if not (owner[rows] >= 0).any():
        return actions
    return _restart_kernel(rows, players, actions, sim.ball_pos, sim.player_pos,
                           sim.player_team, env.sign, owner, kind, excluded, forced,
                           env.goal_x, float(sim.st.player["radius"] + sim.st.ball["radius"]))


@njit(cache=True, nogil=True)
def _restart_kernel(rows, players, actions, bp, pp, team, sign, owner, kind, excluded, forced, gx, radii):
    for i in range(len(rows)):
        n = rows[i]
        if owner[n] < 0:
            continue
        taker, best = -1, math.inf
        for q in range(pp.shape[1]):
            if team[q] != owner[n] or excluded[n, q]:
                continue
            d = (pp[n, q, 0] - bp[n, 0]) ** 2 + (pp[n, q, 1] - bp[n, 1]) ** 2
            if d < best:
                taker, best = q, d
        if forced[n] >= 0 and not excluded[n, forced[n]]:
            taker = forced[n]
        for j in range(len(players)):
            p = players[j]
            if excluded[n, p]:
                actions[i, j] = 0
                continue
            if team[p] != owner[n]:
                actions[i, j] = 0 if kind[n] == 6 else actions[i, j] % 9
                continue
            if p != taker:
                actions[i, j] %= 9
                continue
            bx, by = bp[n, 0] * sign[p], bp[n, 1]
            px, py = pp[n, p, 0] * sign[p], pp[n, p, 1]
            tx, ty = gx - bx, -by
            if kind[n] == 1:  # lateral: tiro perpendicular a la banda hacia adentro
                tx, ty = 0.0, -1.0 if by > 0 else 1.0
            elif kind[n] == 2:  # córner: hacia la cancha, no hacia afuera del fondo
                tx, ty = -bx, -by
            length = max(math.hypot(tx, ty), 1e-9)
            tx, ty = tx / length, ty / length
            dx, dy = bx - px, by - py
            distance = math.hypot(dx, dy)
            alignment = (dx * tx + dy * ty) / max(distance, 1e-9)
            target_x, target_y = bx - tx * (radii + 2), by - ty * (radii + 2)
            if alignment > 0.8:
                target_x, target_y = bx, by
            elif alignment < 0 and distance < radii * 2.5:
                # Rodear, sin atravesar/push-ear la pelota desde el lado incorrecto.
                side = 1.0 if (px - bx) * -ty + (py - by) * tx >= 0 else -1.0
                target_x += -ty * side * (radii + 12)
                target_y += tx * side * (radii + 12)
            dx, dy = target_x - px, target_y - py
            norm = max(math.hypot(dx, dy), 1e-9)
            move, score = 0, -math.inf
            if norm >= 1.0:
                for m in range(1, 9):
                    candidate = (dx * MOVE_UNIT[m, 0] + dy * MOVE_UNIT[m, 1]) / norm
                    if candidate > score:
                        move, score = m, candidate
            # Mantener X antes del contacto: frame_skip puede cruzar el alcance de
            # patada entre decisiones; el motor decide cuándo el tiro es posible.
            kick = alignment > 0.65 and distance < radii + 8.0
            actions[i, j] = move + 9 * kick
    return actions


@njit(cache=True, nogil=True)
def _six_cover_target(ordinal, bx, by, gx, H, radii):
    """Roles estables por identidad: cuatro carriles de apoyo y un portero.

    El quinto jugador de campo (el más cercano) presiona en el llamador.
    No cambiar carril con el ranking de distancias evita cruces/oscilaciones.
    """
    if ordinal == 5:
        x = -gx + max(60.0, .06 * gx)
        y = min(max(by * .35, -.16 * H), .16 * H)
    else:
        depth = .12 if ordinal in (0, 4) else (.30 if ordinal in (1, 3) else .45)
        x = max(bx - depth * gx, -.78 * gx)  # no ocupar el mismo fondo que el portero
        y = by * .20 + (ordinal - 2) * .25 * H
    margin = radii + 20.0
    x = min(max(x, -gx + margin), gx - margin)
    y = min(max(y, -H + margin), H - margin)
    return x, y


@njit(cache=True, nogil=True)
def _scripted_kernel(rows, players, ball_pos, ball_vel, pp, team, sign, gx, H, radii, T):
    out = np.empty((len(rows), len(players)), dtype=np.int64)
    for i in range(len(rows)):
        n = rows[i]
        distances = np.empty(pp.shape[1])
        for q in range(pp.shape[1]):
            dx, dy = pp[n, q, 0] - ball_pos[n, 0], pp[n, q, 1] - ball_pos[n, 1]
            distances[q] = math.sqrt(dx * dx + dy * dy)
        for j in range(len(players)):
            p = players[j]
            s = sign[p]
            bx, by = ball_pos[n, 0] * s, ball_pos[n, 1]
            fx = bx + ball_vel[n, 0] * s * 6.0
            fy = by + ball_vel[n, 1] * 6.0
            px, py = pp[n, p, 0] * s, pp[n, p, 1]
            dx, dy = gx - fx, -fy
            norm = max(math.sqrt(dx * dx + dy * dy), 1e-9)
            tx, ty = dx / norm, dy / norm
            dx, dy = fx - px, fy - py
            distance = math.sqrt(dx * dx + dy * dy)
            norm = max(distance, 1e-9)
            aligned = (dx / norm) * tx + (dy / norm) * ty > 0.8
            if aligned:
                target_x, target_y = fx, fy
            elif px > fx - 5.0:
                delta = py - fy + 1e-6
                side = 1.0 if delta > 0 else (-1.0 if delta < 0 else 0.0)
                target_x, target_y = min(fx - 10.0, px), fy + side * (radii + 25.0)
            else:
                target_x, target_y = fx - tx * (radii + 6.0), fy - ty * (radii + 6.0)
            opponent = math.inf
            rank = 0
            for q in range(pp.shape[1]):
                if team[q] != team[p]:
                    opponent = min(opponent, distances[q])
                elif distances[q] < distances[p] or (distances[q] == distances[p] and q < p):
                    rank += 1
            if T == 6:
                # Portero fijo; el más cercano ENTRE LOS CINCO DE CAMPO nunca
                # abandona la presión sólo porque el rival llega antes.
                ordinal, keeper = 0, -1
                for q in range(pp.shape[1]):
                    if team[q] == team[p]:
                        keeper = q
                        if q < p:
                            ordinal += 1

                field_rank = 0
                for q in range(pp.shape[1]):
                    if team[q] == team[p] and q != keeper:
                        if distances[q] < distances[p] or (distances[q] == distances[p] and q < p):
                            field_rank += 1

                if p == keeper or field_rank > 0:
                    target_x, target_y = _six_cover_target(
                        ordinal, bx, by, gx, H, radii
                    )

            elif T != 4 and opponent + 20.0 < distance and bx < 0:
                target_x, target_y = -gx + 30.0, by * 0.4

            if T == 4:
                if rank == 0:
                    # El más cercano mantiene la lógica ofensiva.
                    pass

                elif rank == 1:
                    # Apoyo lateral.
                    lateral = (-1.0 if by > 0 else 1.0) * 0.30 * H
                    target_x = bx - 0.20 * gx
                    target_y = by * 0.35 + lateral

                elif rank == 2:
                    # Defensor, sin hundirse hasta el arco.
                    target_x = max(-0.55 * gx, bx - 0.35 * gx)
                    target_y = by * 0.25

                else:
                    # Portero.
                    target_x = -gx + 35.0
                    target_y = min(max(by * 0.20, -0.22 * H), 0.22 * H)

            elif T != 6 and rank == 1:
                lateral = (-1.0 if by > 0 else 1.0) * 0.28 * H
                target_x = bx - 0.18 * gx
                target_y = by * 0.4 + lateral

            elif T != 6 and rank >= 2:
                target_x = min(bx - 0.25 * gx, -0.55 * gx)
                target_y = by * 0.35

            dx, dy = target_x - px, target_y - py
            norm = math.sqrt(dx * dx + dy * dy)
            denominator = max(norm, 1e-9)
            ux, uy = dx / denominator, dy / denominator

            best, move = -math.inf, 1
            for m in range(1, 9):
                score = ux * MOVE_UNIT[m, 0] + uy * MOVE_UNIT[m, 1]
                if score > best:
                    best, move = score, m

            if norm < 2.0:
                move = 0

            out[i, j] = move + 9 * (aligned and distance < radii + 8.0)

    return out


def _scripted_reference(env, players: np.ndarray | None = None, eps: float = 0.0,
                     rng: np.random.Generator | None = None) -> np.ndarray:
    """Acciones (N, len(players)) en marco propio para los jugadores `players` de `env`."""
    rng = rng or np.random.default_rng()
    sim = env.sim
    P = sim.P
    players = np.arange(P) if players is None else np.asarray(players)
    sign = env.sign[players]
    st = sim.st
    r_p = st.player["radius"]
    r_b = st.ball["radius"]
    gx = env.goal_x

    ball = np.repeat(sim.ball_pos[:, None], len(players), 1).copy()
    ball[..., 0] *= sign
    bvel = np.repeat(sim.ball_vel[:, None], len(players), 1).copy()
    bvel[..., 0] *= sign
    me = sim.player_pos[:, players].copy()
    me[..., 0] *= sign

    # anticipar un poco la pelota
    ball_f = ball + bvel * 6.0
    goal = np.array([gx, 0.0])
    to_goal = goal - ball_f
    to_goal /= np.maximum(np.linalg.norm(to_goal, axis=-1, keepdims=True), 1e-9)
    behind = ball_f - to_goal * (r_p + r_b + 6.0)

    to_ball = ball_f - me
    dist_ball = np.linalg.norm(to_ball, axis=-1)
    unit_tb = to_ball / np.maximum(dist_ball, 1e-9)[..., None]
    aligned = np.sum(unit_tb * to_goal, axis=-1) > 0.8

    # rodear la pelota: si estoy delante de ella, desviarme lateralmente
    ahead = me[..., 0] > ball_f[..., 0] - 5.0
    side = np.sign(me[..., 1] - ball_f[..., 1] + 1e-6)
    detour = behind.copy()
    detour[..., 1] = ball_f[..., 1] + side * (r_p + r_b + 25.0)
    detour[..., 0] = np.minimum(ball_f[..., 0] - 10.0, me[..., 0])
    target = np.where(aligned[..., None], ball_f, np.where(ahead[..., None], detour, behind))

    # defensa: el rival más cercano a la pelota llega antes y la pelota está en mi mitad
    team = sim.player_team[players]
    d_all = np.linalg.norm(sim.player_pos - sim.ball_pos[:, None], axis=-1)  # (N, P)
    opp_best = np.stack([np.min(d_all[:, sim.player_team != t], axis=1) for t in team], axis=1)
    defend = (opp_best + 20.0 < dist_ball) & (ball[..., 0] < 0)
    guard = np.stack([np.full(ball.shape[:-1], -gx + 30.0), ball[..., 1] * 0.4], axis=-1)
    if env.T != 6:
        target = np.where(defend[..., None], guard, target)

    # roles por equipo: el más cercano a la pelota va a buscarla; el 2º se abre para apoyar
    # (atrás y hacia el centro) y el resto cubre el arco. Sin esto los tres se amontonan.
    if env.T == 6:
        for j, p in enumerate(players):
            mates = np.flatnonzero(sim.player_team == sim.player_team[p])
            keeper = mates[-1]
            ordinal = int(np.flatnonzero(mates == p)[0])
            field = mates[:-1]
            dp = d_all[:, p][:, None]
            rank = ((d_all[:, field] < dp) | ((d_all[:, field] == dp) & (field[None, :] < p))).sum(axis=1)
            for n in range(sim.N):
                if p == keeper or rank[n] > 0:
                    target[n, j] = _six_cover_target.py_func(ordinal, ball[n, j, 0], ball[n, j, 1],
                                                            gx, st.field_half_h, r_p + r_b)
    elif env.T > 1:
        rank = np.zeros((sim.N, len(players)), dtype=np.int64)
        for j, p in enumerate(players):
            mates = np.where(sim.player_team == sim.player_team[p])[0]
            dp = d_all[:, p][:, None]
            dm = d_all[:, mates]
            rank[:, j] = ((dm < dp) | ((dm == dp) & (mates[None, :] < p))).sum(axis=1)
        lateral = np.where(ball[..., 1] > 0, -1.0, 1.0) * 0.28 * st.field_half_h
        support = np.stack([ball[..., 0] - 0.18 * gx, ball[..., 1] * 0.4 + lateral], axis=-1)
        cover = np.stack([np.minimum(ball[..., 0] - 0.25 * gx, -0.55 * gx), ball[..., 1] * 0.35], axis=-1)
        target = np.where((rank == 1)[..., None], support, np.where((rank >= 2)[..., None], cover, target))

    move = _dir_to_action(target - me)
    kick = aligned & (dist_ball < r_p + r_b + 8.0)
    act = move + 9 * kick
    act = _restart_actions(env, np.arange(sim.N), players, act)

    if eps > 0:
        rnd = rng.random(act.shape) < eps
        act = np.where(rnd, rng.integers(0, 18, act.shape), act)
    return act
