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


def scripted_actions(env, players=None, eps=0.0, rng=None, env_indices=None, policy="r2", style=-1):
    """Acciones del rival por reglas.

    ``policy="r2"`` conserva exactamente el baseline histórico.
    ``policy="r3"`` activa la política táctica nueva (posesión, intercepción,
    presión selectiva, cobertura, marcaje, pases y estilos). ``style=-1`` mezcla
    de forma estable tres estilos por entorno; 0/1/2 fuerzan balanceado/agresivo/conservador.
    """
    policy = str(policy).lower()
    if policy not in ("r2", "r3"):
        raise ValueError(f"scripted policy desconocida: {policy}")
    if style not in (-1, 0, 1, 2):
        raise ValueError("scripted style debe ser -1, 0, 1 o 2")

    if not getattr(env, "optimize_rollout", True):
        actions = _scripted_reference(env, players, eps, rng, policy=policy, style=style)
        return actions if env_indices is None else actions[env_indices]

    sim = env.sim
    players = np.arange(sim.P) if players is None else np.asarray(players, dtype=np.int64)
    rows = np.arange(sim.N) if env_indices is None else np.asarray(env_indices, dtype=np.int64)

    if policy == "r3":
        pl = sim.st.player
        ball = sim.st.ball
        p_damping = float(pl["damping"])
        p_acc = float(pl["acceleration"])
        player_speed = p_acc / max(1.0 - p_damping, 1e-6)
        tactical = _scripted_r3_kernel(
            rows, players, sim.ball_pos, sim.ball_vel, sim.player_pos, sim.player_vel,
            sim.player_team, env.sign, env.goal_x, sim.st.field_half_h,
            float(pl["radius"] + ball["radius"]), env.T, float(ball["damping"]),
            player_speed, int(env.frame_skip), env.scripted_style, int(style),
        )
        baseline = _scripted_kernel(rows, players, sim.ball_pos, sim.ball_vel, sim.player_pos,
                                    sim.player_team, env.sign, env.goal_x, sim.st.field_half_h,
                                    float(pl["radius"] + ball["radius"]), env.T)
        actions = _r3_hybrid_actions(rows, players, baseline, tactical, sim.ball_pos,
                                     sim.player_pos, sim.player_team, env.T,
                                     env.scripted_style, int(style))
    else:
        actions = _scripted_kernel(rows, players, sim.ball_pos, sim.ball_vel, sim.player_pos,
                                   sim.player_team, env.sign, env.goal_x, sim.st.field_half_h,
                                   float(sim.st.player["radius"] + sim.st.ball["radius"]), env.T)

    # Los saques siguen usando el árbitro/baseline ya probado.
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
def _six_cover_target(ordinal, bx, by, gx, H, radii):
    """Contrato histórico de R2 para los tests/herramientas de equipos 6v6.

    R2 evolucionó hacia los helpers generales ``_keeper_target`` y
    ``_support_target``, pero esta función pública de facto seguía siendo usada
    por diagnósticos. Mantenerla evita que una mejora de R3 rompa esas
    herramientas y conserva los carriles dorados originales.
    """
    if ordinal == 5:
        x = -gx + max(60.0, .06 * gx)
        y = min(max(by * .35, -.16 * H), .16 * H)
    else:
        depth = .12 if ordinal in (0, 4) else (.30 if ordinal in (1, 3) else .45)
        x = max(bx - depth * gx, -.78 * gx)
        y = by * .20 + (ordinal - 2) * .25 * H
    margin = radii + 20.0
    x = min(max(x, -gx + margin), gx - margin)
    y = min(max(y, -H + margin), H - margin)
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

    # El apoyo más cercano queda cerca de la jugada.
    # Los demás forman profundidades progresivamente mayores.
    depth = 0.18 + 0.28 * rel

    if defending:
        depth += 0.07

    target_x = bx - depth * gx

    if defending:
        # IMPORTANTE:
        # cada apoyo tiene un límite defensivo diferente.
        # Evita que todos colapsen en la misma línea vertical
        # cuando la pelota está cerca del arco.
        min_depth = 0.52 + 0.22 * rel
        min_x = -min_depth * gx
    else:
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



# ============================================================================
# R3: rival táctico. R2 queda intacto arriba.
# ============================================================================

@njit(cache=True, inline="always")
def _point_segment_distance(px, py, ax, ay, bx, by):
    """Distancia de P al segmento AB (sin allocations; usable desde numba)."""
    abx, aby = bx - ax, by - ay
    den = abx * abx + aby * aby
    if den <= 1e-12:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * abx + (py - ay) * aby) / den
    t = _clamp(t, 0.0, 1.0)
    qx, qy = ax + t * abx, ay + t * aby
    return math.hypot(px - qx, py - qy)


@njit(cache=True, inline="always")
def _r3_ball_future(bx, by, bvx, bvy, damping, ticks):
    """Predicción sin colisiones; respeta el damping geométrico del motor."""
    if ticks <= 0:
        return bx, by
    if abs(1.0 - damping) < 1e-9:
        travel = float(ticks)
    else:
        travel = (1.0 - damping ** ticks) / (1.0 - damping)
    return bx + bvx * travel, by + bvy * travel


@njit(cache=True, inline="always")
def _r3_intercept(px, py, pvx, pvy, bx, by, bvx, bvy, ball_damping,
                  player_speed, frame_skip, radii):
    """Busca el primer punto futuro razonablemente alcanzable por el jugador.

    No intenta clonar toda la física: usa la trayectoria amortiguada real de la
    pelota y una cota conservadora de desplazamiento del jugador. El objetivo es
    interceptar en vez de perseguir la posición vieja de la pelota.
    """
    step = max(frame_skip, 1)
    now_speed = math.hypot(pvx, pvy)
    # 36 ticks = 0.6 s. Más horizonte tiende a sobrepredecir por paredes/choques.
    best_x, best_y = bx, by
    best_t = 40.0
    for ticks in range(step, 37, step):
        fx, fy = _r3_ball_future(bx, by, bvx, bvy, ball_damping, ticks)
        dist = math.hypot(fx - px, fy - py)
        # Aceleración implícita: parte de la velocidad actual y converge al cap.
        avg_speed = min(player_speed, now_speed + 0.55 * player_speed)
        reach = radii + avg_speed * ticks * 0.78
        if dist <= reach:
            return fx, fy, float(ticks)
        best_x, best_y = fx, fy
    return best_x, best_y, best_t


@njit(cache=True, inline="always")
def _r3_keeper_target(bx, by, gx, H, radii, style):
    """Keeper R3: misma base robusta de R2, con altura de línea por estilo."""
    x, y = _keeper_target(bx, by, gx, H, radii)
    if style == 1:       # agresivo: líbero un poco más alto
        x += 0.035 * gx
    elif style == 2:     # conservador: protege algo más profundo
        x -= 0.025 * gx
    x = _clamp(x, -gx + max(25.0, radii), -0.56 * gx)
    return x, y


@njit(cache=True, inline="always")
def _r3_base_roles(team, my_team, team_size):
    """Selección estable de arquero; el resto de roles depende de la jugada."""
    keeper, count = -1, 0
    for q in range(len(team)):
        if team[q] == my_team:
            count += 1
            keeper = q
    if team_size < 3:
        keeper = -1
    return keeper, count


@njit(cache=True, inline="always")
def _r3_possession(distances, team, my_team, control):
    """Devuelve posesión (-1 rival, 0 dividida, 1 propia) y jugadores cercanos."""
    own_best, opp_best = math.inf, math.inf
    own_near, opp_near = -1, -1
    for q in range(len(team)):
        d = distances[q]
        if team[q] == my_team:
            if d < own_best:
                own_best, own_near = d, q
        elif d < opp_best:
            opp_best, opp_near = d, q
    if own_best <= control and opp_best > control:
        possession = 1
    elif opp_best <= control and own_best > control:
        possession = -1
    elif own_best + 18.0 < opp_best:
        possession = 1
    elif opp_best + 18.0 < own_best:
        possession = -1
    else:
        possession = 0
    return possession, own_near, opp_near, own_best, opp_best


@njit(cache=True, inline="always")
def _r3_direct_press(possession, team_size, presser_t, style, bx, gx):
    """Decide presión directa frente a contención estructural."""
    if possession != -1 or team_size < 3:
        return True
    late = presser_t > (16.0 if style == 1 else (10.0 if style == 2 else 13.0))
    dangerous = bx < (-0.25 * gx if style == 2 else -0.38 * gx)
    return not (late and dangerous)


@njit(cache=True, inline="always")
def _r3_pass_score(forward, space, lane, length, style):
    """Utilidad de pase verificable: progreso, espacio, carril y coste de distancia."""
    pass_bias = 0.22 if style == 2 else (-0.10 if style == 1 else 0.05)
    return 1.25 * forward + 0.85 * space + 0.95 * lane - 0.35 * length + pass_bias


@njit(cache=True, inline="always")
def _r3_pass_threshold(style):
    return 0.72 if style == 1 else (0.42 if style == 2 else 0.56)


@njit(cache=True, nogil=True)
def _r3_hybrid_actions(rows, players, baseline, tactical, ball_pos, player_pos, team, team_size,
                       style_ids, style_mode):
    """Protege los contratos fuertes de R2 y limita R3 a roles seguros.

    En 1v1 no hay cooperación que añadir. En 2v2 R3 ya supera a R2 en la
    matriz fija. En 3v3+ el portero y el disputador conservan la acción R2;
    apoyos/marcadores usan la capa táctica R3. Las combinaciones tamaño/estilo
    que no superan el gate conservan R2 completo.
    """
    if team_size == 1:
        return baseline
    if team_size == 2:
        return tactical
    out = tactical.copy()
    for i in range(len(rows)):
        n = rows[i]
        style = style_ids[n] if style_mode < 0 else style_mode
        for j in range(len(players)):
            p = players[j]
            # Los estilos que no superaron la matriz conservan R2 completo;
            # nunca se introducen como rival sólo por ser "nuevos".
            if (team_size == 3 and style == 0) or (team_size >= 5 and style != 2):
                out[i, j] = baseline[i, j]
                continue
            my_team = team[p]
            keeper = -1
            for q in range(len(team)):
                if team[q] == my_team:
                    keeper = q
            presser, best = -1, math.inf
            for q in range(len(team)):
                if team[q] != my_team or q == keeper:
                    continue
                dx = player_pos[n, q, 0] - ball_pos[n, 0]
                dy = player_pos[n, q, 1] - ball_pos[n, 1]
                distance = dx * dx + dy * dy
                if distance < best:
                    presser, best = q, distance
            if p == keeper or p == presser:
                out[i, j] = baseline[i, j]
    return out


@njit(cache=True, nogil=True)
def _scripted_r3_kernel(rows, players, ball_pos, ball_vel, pp, pv, team, sign,
                        gx, H, radii, T, ball_damping, player_speed, frame_skip,
                        style_ids, style_mode):
    """R3 completo: siete capas tácticas sobre el mismo espacio de 18 acciones.

    1) posesión propia/rival/dividida
    2) intercepción predictiva
    3) presión inteligente
    4) bloqueo de línea de tiro
    5) marcaje + cierre de líneas de pase
    6) pase/conducción/remate
    7) tres estilos tácticos estables por entorno
    """
    out = np.empty((len(rows), len(players)), dtype=np.int64)
    P = pp.shape[1]

    for i in range(len(rows)):
        n = rows[i]
        style = style_ids[n] if style_mode < 0 else style_mode  # 0 balance, 1 agresivo, 2 conservador

        # Distancias absolutas a pelota: compartidas por todos los jugadores del row.
        distances = np.empty(P)
        for q in range(P):
            dx = pp[n, q, 0] - ball_pos[n, 0]
            dy = pp[n, q, 1] - ball_pos[n, 1]
            distances[q] = math.hypot(dx, dy)

        for j in range(len(players)):
            p = players[j]
            my_team = team[p]
            s = sign[p]
            bx, by = ball_pos[n, 0] * s, ball_pos[n, 1]
            bvx, bvy = ball_vel[n, 0] * s, ball_vel[n, 1]
            px, py = pp[n, p, 0] * s, pp[n, p, 1]
            pvx, pvy = pv[n, p, 0] * s, pv[n, p, 1]

            # -----------------------------------------------------------------
            # Identidad/roles base. Conserva la compatibilidad de R2: en 3v3+
            # el último slot del equipo es keeper; en 1v1/2v2 no hay fijo.
            # -----------------------------------------------------------------
            keeper, team_count = _r3_base_roles(team, my_team, T)

            # -----------------------------------------------------------------
            # 1) POSESIÓN: contacto claro gana; si no, carrera a la pelota.
            # -----------------------------------------------------------------
            control = radii + 10.0
            possession, own_near, opp_near, own_best, opp_best = _r3_possession(
                distances, team, my_team, control)

            # -----------------------------------------------------------------
            # 2) INTERCEPCIÓN: el presionador se elige por tiempo estimado de
            # llegada a la trayectoria, no por distancia instantánea.
            # -----------------------------------------------------------------
            presser = -1
            presser_t = math.inf
            presser_ix, presser_iy = bx, by
            for q in range(P):
                if team[q] != my_team or q == keeper:
                    continue
                qx, qy = pp[n, q, 0] * s, pp[n, q, 1]
                qvx, qvy = pv[n, q, 0] * s, pv[n, q, 1]
                ix, iy, it = _r3_intercept(qx, qy, qvx, qvy, bx, by, bvx, bvy,
                                           ball_damping, player_speed, frame_skip, radii)
                # Si tenemos control, quien ya está sobre la pelota conserva la jugada.
                if possession == 1 and q == own_near:
                    it -= 6.0
                if it < presser_t:
                    presser, presser_t = q, it
                    presser_ix, presser_iy = ix, iy

            # Keeper-líbero sólo en emergencia real.
            if keeper >= 0 and bx < -0.78 * gx:
                kx, ky = pp[n, keeper, 0] * s, pp[n, keeper, 1]
                kvx, kvy = pv[n, keeper, 0] * s, pv[n, keeper, 1]
                kix, kiy, kit = _r3_intercept(kx, ky, kvx, kvy, bx, by, bvx, bvy,
                                              ball_damping, player_speed, frame_skip, radii)
                if kit + 4.0 < presser_t:
                    presser, presser_t = keeper, kit
                    presser_ix, presser_iy = kix, kiy

            # -----------------------------------------------------------------
            # 3) PRESIÓN INTELIGENTE: si el rival controla en zona peligrosa y
            # llegar tarde rompería la estructura, el presser contiene primero.
            # Agresivo presiona casi siempre; conservador contiene antes.
            # -----------------------------------------------------------------
            direct_press = _r3_direct_press(possession, T, presser_t, style, bx, gx)

            # -----------------------------------------------------------------
            # 4) SHOT BLOCKER: jugador libre más apto para ocupar el segmento
            # pelota -> centro del arco propio. No persigue la pelota.
            # -----------------------------------------------------------------
            block_x = bx + (-gx - bx) * (0.38 if style == 2 else 0.32)
            block_y = by * (0.58 if style == 2 else 0.64)
            shot_blocker, block_best = -1, math.inf
            if possession != 1 and T >= 2:
                for q in range(P):
                    if team[q] != my_team or q == keeper or q == presser:
                        continue
                    qx, qy = pp[n, q, 0] * s, pp[n, q, 1]
                    d = math.hypot(qx - block_x, qy - block_y)
                    if d < block_best:
                        shot_blocker, block_best = q, d

            # -----------------------------------------------------------------
            # Elegir objetivo según rol.
            # -----------------------------------------------------------------
            target_x, target_y = px, py
            aim_x, aim_y = gx, 0.0
            wants_kick = False
            kick = False

            if p == keeper and p != presser:
                target_x, target_y = _r3_keeper_target(bx, by, gx, H, radii, style)

            elif p == presser:
                # -------------------------------------------------------------
                # 6) ATAQUE: con posesión evalúa tiro, pase o conducción.
                # -------------------------------------------------------------
                if possession == 1 or distances[p] < radii + 22.0:
                    fx, fy = _r3_ball_future(bx, by, bvx, bvy, ball_damping,
                                             max(frame_skip, 1) * 2)

                    # Línea de tiro: rivales cerca del segmento pelota->arco.
                    shot_clearance = math.inf
                    for q in range(P):
                        if team[q] == my_team:
                            continue
                        qx, qy = pp[n, q, 0] * s, pp[n, q, 1]
                        dline = _point_segment_distance(qx, qy, fx, fy, gx, 0.0)
                        if dline < shot_clearance:
                            shot_clearance = dline

                    shot_zone = fx > (0.22 * gx if style == 1 else (0.42 * gx if style == 2 else 0.32 * gx))
                    shoot = shot_zone and shot_clearance > (radii * (0.65 if style == 1 else 0.9))

                    # Mejor pase: progreso + espacio + carril libre - distancia.
                    best_pass = -1
                    best_pass_score = -math.inf
                    for q in range(P):
                        if team[q] != my_team or q == p or q == keeper:
                            continue
                        qx, qy = pp[n, q, 0] * s, pp[n, q, 1]
                        # No regalar la pelota muy hacia atrás salvo estilo conservador.
                        if qx < fx - (0.22 * gx if style == 2 else 0.10 * gx):
                            continue
                        nearest_opp = math.inf
                        lane_clear = math.inf
                        for r in range(P):
                            if team[r] == my_team:
                                continue
                            rx, ry = pp[n, r, 0] * s, pp[n, r, 1]
                            od = math.hypot(rx - qx, ry - qy)
                            if od < nearest_opp:
                                nearest_opp = od
                            ld = _point_segment_distance(rx, ry, fx, fy, qx, qy)
                            if ld < lane_clear:
                                lane_clear = ld
                        forward = (qx - fx) / max(gx, 1.0)
                        space = min(nearest_opp, 0.45 * gx) / max(0.45 * gx, 1.0)
                        lane = min(lane_clear, 0.30 * H) / max(0.30 * H, 1.0)
                        length = math.hypot(qx - fx, qy - fy) / max(gx, 1.0)
                        score = _r3_pass_score(forward, space, lane, length, style)
                        if score > best_pass_score:
                            best_pass_score, best_pass = score, q

                    pass_threshold = _r3_pass_threshold(style)
                    do_pass = (not shoot and best_pass >= 0 and best_pass_score > pass_threshold)
                    if do_pass:
                        qx = pp[n, best_pass, 0] * s
                        qy = pp[n, best_pass, 1]
                        qvx = pv[n, best_pass, 0] * s
                        qvy = pv[n, best_pass, 1]
                        aim_x, aim_y = qx + qvx * 4.0, qy + qvy * 4.0
                        wants_kick = True
                    else:
                        aim_x, aim_y = gx, 0.0
                        # Conducción por cuerpo hasta zona de tiro; luego remate.
                        wants_kick = shoot

                    adx, ady = aim_x - fx, aim_y - fy
                    an = max(math.hypot(adx, ady), 1e-9)
                    tx, ty = adx / an, ady / an
                    ball_dx, ball_dy = fx - px, fy - py
                    distance = math.hypot(ball_dx, ball_dy)
                    bn = max(distance, 1e-9)
                    alignment = (ball_dx / bn) * tx + (ball_dy / bn) * ty

                    if alignment > 0.82:
                        target_x, target_y = fx, fy
                    elif px > fx - tx * 3.0:
                        # Rodear la pelota por el lado más corto sin empujarla mal.
                        cross = (px - fx) * -ty + (py - fy) * tx
                        side = 1.0 if cross >= 0.0 else -1.0
                        target_x = fx - tx * (radii + 5.0) - ty * side * (radii + 18.0)
                        target_y = fy - ty * (radii + 5.0) + tx * side * (radii + 18.0)
                    else:
                        target_x = fx - tx * (radii + 5.0)
                        target_y = fy - ty * (radii + 5.0)

                    # Al perder la pelota en campo propio, despejar es mejor que driblar.
                    emergency = bx < -0.58 * gx and possession != 1
                    kick = (wants_kick or emergency) and alignment > 0.70 and distance < radii + 8.0
                else:
                    # Sin control: ir al punto de intercepción, no a la pelota actual.
                    target_x, target_y = presser_ix, presser_iy
                    kick = False

                if possession == -1 and not direct_press:
                    # Contención: ponerse en la línea rival->arco hasta que la carrera sea favorable.
                    target_x = bx + (-gx - bx) * 0.20
                    target_y = by * 0.78
                    kick = False

            elif p == shot_blocker:
                target_x, target_y = block_x, block_y
                kick = False

            else:
                # -------------------------------------------------------------
                # 5) MARCAJE / PASES: si defendemos, cada apoyo toma una amenaza
                # distinta y ocupa su línea de pase/tiro. En posesión se abre.
                # -------------------------------------------------------------
                if possession != 1:
                    # Rango estable de este defensor entre los libres.
                    my_rank = 0
                    free_count = 0
                    for q in range(P):
                        if team[q] != my_team or q == keeper or q == presser or q == shot_blocker:
                            continue
                        if q < p:
                            my_rank += 1
                        free_count += 1

                    opp_count = 0
                    for q in range(P):
                        if team[q] != my_team:
                            opp_count += 1
                    desired_rank = my_rank % max(opp_count, 1)

                    mark = -1
                    for cand in range(P):
                        if team[cand] == my_team:
                            continue
                        cx, cy = pp[n, cand, 0] * s, pp[n, cand, 1]
                        cball = math.hypot(cx - bx, cy - by)
                        cgoal = math.hypot(cx + gx, cy)
                        cscore = -0.65 * cgoal - 0.35 * cball
                        rank = 0
                        for other in range(P):
                            if team[other] == my_team or other == cand:
                                continue
                            ox, oy = pp[n, other, 0] * s, pp[n, other, 1]
                            oball = math.hypot(ox - bx, oy - by)
                            ogoal = math.hypot(ox + gx, oy)
                            oscore = -0.65 * ogoal - 0.35 * oball
                            if oscore > cscore or (oscore == cscore and other < cand):
                                rank += 1
                        if rank == desired_rank:
                            mark = cand
                            break

                    if mark >= 0:
                        mx, my = pp[n, mark, 0] * s, pp[n, mark, 1]
                        if possession == -1:
                            # Rival controla: negar pase desde pelota al receptor.
                            target_x = mx * 0.68 + bx * 0.32
                            target_y = my * 0.68 + by * 0.32
                        else:
                            # Dividida: un poco más del lado del arco.
                            target_x = mx * 0.72 + (-gx) * 0.28
                            target_y = my * 0.72
                        if style == 2:
                            target_x -= 0.04 * gx
                        elif style == 1:
                            target_x += 0.025 * gx
                    else:
                        target_x, target_y = _support_target(0, 1, 0, max(T - 1, 1),
                                                             bx, by, gx, H, radii, True)
                else:
                    # En ataque: anchura/profundidad de R2, modificada por estilo y
                    # con separación de la pelota para convertirse en opción de pase.
                    support_rank = 0
                    support_count = 0
                    lane_ordinal = 0
                    field_count = 0
                    for q in range(P):
                        if team[q] != my_team or q == keeper:
                            continue
                        if q < p:
                            lane_ordinal += 1
                        field_count += 1
                        if q == presser:
                            continue
                        if q != p:
                            support_count += 1
                            if distances[q] < distances[p] or (distances[q] == distances[p] and q < p):
                                support_rank += 1
                    support_count += 1
                    target_x, target_y = _support_target(support_rank, support_count,
                                                         lane_ordinal, field_count,
                                                         bx, by, gx, H, radii, False)
                    if style == 1:
                        target_x += 0.08 * gx
                    elif style == 2:
                        target_x -= 0.05 * gx
                    # Abrirse del lado contrario de la pelota crea un pase real.
                    if abs(target_y - by) < 0.16 * H:
                        target_y += (-1.0 if by > 0 else 1.0) * 0.18 * H
                    margin = radii + 18.0
                    target_y = _clamp(target_y, -H + margin, H - margin)
                kick = False

            # -----------------------------------------------------------------
            # Convertir target en una de las 9 direcciones, y opcionalmente kick.
            # -----------------------------------------------------------------
            dx, dy = target_x - px, target_y - py
            norm = math.hypot(dx, dy)
            if norm < 2.0:
                move = 0
            else:
                ux, uy = dx / norm, dy / norm
                move, best = 1, -math.inf
                for m in range(1, 9):
                    score = ux * MOVE_UNIT[m, 0] + uy * MOVE_UNIT[m, 1]
                    if score > best:
                        move, best = m, score

            # Cualquier no-presionador puede despejar una pelota que le cae encima
            # en campo propio; evita congelarse por una asignación de rol imperfecta.
            if p != presser and distances[p] < radii + 7.0 and bx < -0.45 * gx:
                ball_dx, ball_dy = bx - px, by - py
                dn = max(math.hypot(ball_dx, ball_dy), 1e-9)
                gdx, gdy = gx - bx, -by
                gn = max(math.hypot(gdx, gdy), 1e-9)
                clear_align = (ball_dx / dn) * (gdx / gn) + (ball_dy / dn) * (gdy / gn)
                if clear_align > 0.55:
                    kick = True

            out[i, j] = move + 9 * int(kick)

    return out


def _scripted_reference(
    env,
    players: np.ndarray | None = None,
    eps: float = 0.0,
    rng: np.random.Generator | None = None,
    policy: str = "r2",
    style: int = -1,
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

    if policy == "r3":
        pl = sim.st.player
        ball = sim.st.ball
        p_damping = float(pl["damping"])
        player_speed = float(pl["acceleration"]) / max(1.0 - p_damping, 1e-6)
        tactical = _scripted_r3_kernel.py_func(
            rows, players, sim.ball_pos, sim.ball_vel, sim.player_pos, sim.player_vel,
            sim.player_team, env.sign, env.goal_x, sim.st.field_half_h,
            float(pl["radius"] + ball["radius"]), env.T, float(ball["damping"]),
            player_speed, int(env.frame_skip), env.scripted_style, int(style),
        )
        baseline = _scripted_kernel.py_func(
            rows, players, sim.ball_pos, sim.ball_vel, sim.player_pos, sim.player_team,
            env.sign, env.goal_x, sim.st.field_half_h,
            float(pl["radius"] + ball["radius"]), env.T,
        )
        actions = _r3_hybrid_actions.py_func(rows, players, baseline, tactical,
                                             sim.ball_pos, sim.player_pos, sim.player_team, env.T,
                                             env.scripted_style, int(style))
    else:
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
