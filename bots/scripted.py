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
def _clamp(v, lo, hi):
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


@njit(cache=True, nogil=True)
def _keeper_target(bx, by, gx, H, radii):
    """
    Portero dinámico.

    Cerca de nuestro arco se queda profundo.
    Cuando la pelota está lejos, adelanta la línea para no vivir
    permanentemente dentro del arco.
    """
    phase = (bx + gx) / max(2.0 * gx, 1e-9)
    phase = _clamp(phase, 0.0, 1.0)

    base = max(35.0, 0.05 * gx)

    # Puede adelantar hasta ~18% de la cancha cuando atacamos.
    x = -gx + base + phase * 0.18 * gx

    # Nunca convertirse en mediocampista.
    x = min(x, -0.62 * gx)

    # Sigue lateralmente la pelota, pero sin pegarse a los palos.
    y = _clamp(by * 0.35, -0.30 * H, 0.30 * H)

    return x, y


@njit(cache=True, nogil=True)
def _support_target(
    support_rank,
    support_count,
    lane_ordinal,
    field_count,
    bx,
    by,
    gx,
    H,
    radii,
    defending,
):
    """
    Posición de los jugadores que no son presionador ni portero.

    support_rank:
        orden por cercanía a la pelota entre los apoyos.

    lane_ordinal:
        carril estable por identidad para evitar que todos persigan
        exactamente el mismo punto.
    """

    if support_count <= 1:
        rel = 0.0
    else:
        rel = support_rank / float(support_count - 1)

    # El apoyo más cercano queda relativamente cerca de la jugada.
    # Los siguientes forman líneas progresivamente más profundas.
    depth = 0.18 + 0.28 * rel

    if defending:
        depth += 0.07

    target_x = bx - depth * gx

    # Los jugadores de campo nunca deben hundirse hasta el arco.
    min_x = -0.72 * gx
    max_x = gx - max(50.0, radii + 20.0)
    target_x = _clamp(target_x, min_x, max_x)

    # Carriles estables.
    #
    # field_count = jugadores de campo totales, incluido el presionador.
    if field_count <= 1:
        lane = 0.0
    else:
        center = (field_count - 1) * 0.5
        lane_step = 0.55 * H / float(field_count - 1)
        lane = (lane_ordinal - center) * lane_step

    if defending:
        # En defensa convergen algo hacia la pelota,
        # pero conservando separación lateral.
        target_y = by * 0.38 + lane * 0.75
    else:
        target_y = by * 0.28 + lane

    margin = radii + 18.0
    target_y = _clamp(target_y, -H + margin, H - margin)

    return target_x, target_y


@njit(cache=True, nogil=True)
def _scripted_kernel(
    rows,
    players,
    ball_pos,
    ball_vel,
    pp,
    team,
    sign,
    gx,
    H,
    radii,
    T,
):
    out = np.empty((len(rows), len(players)), dtype=np.int64)

    for i in range(len(rows)):
        n = rows[i]

        # ---------------------------------------------------------
        # Distancia de TODOS los jugadores a la pelota.
        # ---------------------------------------------------------
        distances = np.empty(pp.shape[1])

        for q in range(pp.shape[1]):
            dx = pp[n, q, 0] - ball_pos[n, 0]
            dy = pp[n, q, 1] - ball_pos[n, 1]
            distances[q] = math.sqrt(dx * dx + dy * dy)

        for j in range(len(players)):
            p = players[j]
            my_team = team[p]
            s = sign[p]

            bx = ball_pos[n, 0] * s
            by = ball_pos[n, 1]

            px = pp[n, p, 0] * s
            py = pp[n, p, 1]

            # -----------------------------------------------------
            # Identidad dentro del equipo.
            # -----------------------------------------------------
            ordinal = 0
            team_count = 0
            keeper = -1

            for q in range(pp.shape[1]):
                if team[q] != my_team:
                    continue

                if q < p:
                    ordinal += 1

                team_count += 1
                keeper = q

            # En 1v1 y 2v2 no hay portero fijo.
            if T < 3:
                keeper = -1

            # -----------------------------------------------------
            # Presionador:
            # jugador de campo más cercano a la pelota.
            # -----------------------------------------------------
            presser = -1
            presser_distance = math.inf

            for q in range(pp.shape[1]):
                if team[q] != my_team:
                    continue

                if q == keeper:
                    continue

                if distances[q] < presser_distance:
                    presser = q
                    presser_distance = distances[q]

            # -----------------------------------------------------
            # Portero-líbero.
            #
            # Si la pelota está prácticamente encima del arco y
            # el portero llega MUCHO antes, él se convierte
            # temporalmente en el presionador.
            # -----------------------------------------------------
            if keeper >= 0:
                keeper_bx = ball_pos[n, 0] * sign[keeper]

                if (
                    keeper_bx < -0.72 * gx
                    and distances[keeper] + 25.0 < presser_distance
                ):
                    presser = keeper
                    presser_distance = distances[keeper]

            # -----------------------------------------------------
            # Mejor rival respecto de la pelota.
            # -----------------------------------------------------
            opponent_best = math.inf

            for q in range(pp.shape[1]):
                if team[q] != my_team:
                    if distances[q] < opponent_best:
                        opponent_best = distances[q]

            defending = (
                bx < 0.0
                and opponent_best + 20.0 < presser_distance
            )

            # -----------------------------------------------------
            # Objetivo ofensivo para EL presionador.
            # -----------------------------------------------------
            fx = bx + ball_vel[n, 0] * s * 6.0
            fy = by + ball_vel[n, 1] * 6.0

            goal_dx = gx - fx
            goal_dy = -fy
            goal_norm = max(
                math.sqrt(goal_dx * goal_dx + goal_dy * goal_dy),
                1e-9,
            )

            tx = goal_dx / goal_norm
            ty = goal_dy / goal_norm

            ball_dx = fx - px
            ball_dy = fy - py

            distance = math.sqrt(
                ball_dx * ball_dx + ball_dy * ball_dy
            )

            ball_norm = max(distance, 1e-9)

            aligned = (
                (ball_dx / ball_norm) * tx
                + (ball_dy / ball_norm) * ty
                > 0.8
            )

            # Objetivo ofensivo original.
            if aligned:
                attack_x = fx
                attack_y = fy

            elif px > fx - 5.0:
                delta = py - fy + 1e-6

                if delta > 0:
                    side = 1.0
                elif delta < 0:
                    side = -1.0
                else:
                    side = 0.0

                attack_x = min(fx - 10.0, px)
                attack_y = fy + side * (radii + 25.0)

            else:
                attack_x = fx - tx * (radii + 6.0)
                attack_y = fy - ty * (radii + 6.0)

            # -----------------------------------------------------
            # ROLE SELECTION
            # -----------------------------------------------------

            if p == presser:
                # Exactamente uno va a disputar la pelota.
                target_x = attack_x
                target_y = attack_y

            elif p == keeper:
                # Portero estable.
                target_x, target_y = _keeper_target(
                    bx,
                    by,
                    gx,
                    H,
                    radii,
                )

            else:
                # -------------------------------------------------
                # Jugador de apoyo / cobertura.
                #
                # Ranking solamente entre jugadores que NO son
                # presionador ni portero.
                # -------------------------------------------------
                support_rank = 0
                support_count = 0

                for q in range(pp.shape[1]):
                    if team[q] != my_team:
                        continue

                    if q == presser or q == keeper:
                        continue

                    if q == p:
                        continue

                    support_count += 1

                    if (
                        distances[q] < distances[p]
                        or (
                            distances[q] == distances[p]
                            and q < p
                        )
                    ):
                        support_rank += 1

                # incluirnos
                support_count += 1

                # Carril estable entre los jugadores de campo.
                lane_ordinal = 0
                field_count = 0

                for q in range(pp.shape[1]):
                    if team[q] != my_team or q == keeper:
                        continue

                    if q < p:
                        lane_ordinal += 1

                    field_count += 1

                target_x, target_y = _support_target(
                    support_rank,
                    support_count,
                    lane_ordinal,
                    field_count,
                    bx,
                    by,
                    gx,
                    H,
                    radii,
                    defending,
                )

            # -----------------------------------------------------
            # Convertir objetivo en acción 0..8.
            # -----------------------------------------------------
            dx = target_x - px
            dy = target_y - py

            norm = math.sqrt(dx * dx + dy * dy)
            denominator = max(norm, 1e-9)

            ux = dx / denominator
            uy = dy / denominator

            best = -math.inf
            move = 1

            for m in range(1, 9):
                score = (
                    ux * MOVE_UNIT[m, 0]
                    + uy * MOVE_UNIT[m, 1]
                )

                if score > best:
                    best = score
                    move = m

            if norm < 2.0:
                move = 0

            # Cualquier jugador puede despejar/rematar si la pelota
            # accidentalmente llega a sus pies y está bien orientado.
            kick = aligned and distance < radii + 8.0

            out[i, j] = move + 9 * kick

    return out


def _scripted_reference(
    env,
    players: np.ndarray | None = None,
    eps: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Versión Python de exactamente la misma política táctica
    utilizada por el kernel optimizado.
    """
    rng = rng or np.random.default_rng()

    sim = env.sim

    players = (
        np.arange(sim.P, dtype=np.int64)
        if players is None
        else np.asarray(players, dtype=np.int64)
    )

    rows = np.arange(sim.N, dtype=np.int64)

    actions = _scripted_kernel.py_func(
        rows,
        players,
        sim.ball_pos,
        sim.ball_vel,
        sim.player_pos,
        sim.player_team,
        env.sign,
        env.goal_x,
        sim.st.field_half_h,
        float(
            sim.st.player["radius"]
            + sim.st.ball["radius"]
        ),
        env.T,
    )

    actions = _restart_actions(
        env,
        rows,
        players,
        actions,
    )

    if eps > 0:
        rnd = rng.random(actions.shape) < eps

        actions = np.where(
            rnd,
            rng.integers(0, 18, actions.shape),
            actions,
        )

    return actions