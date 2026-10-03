"""Árbitro de Real Soccer ONE 4v4 reconstruido de grabaciones reales (contrato RS4-b1).

Fuente: `reports/rs4_b1/rules_audit.json` (61 partidos, 3708 laterales, 486 córners,
415 saques de arco) y `reports/rs4_b1/restart_kicks.json`, generados desde el motor
original de HaxBall con las acciones del script de la sala (`disc_props` con valores).

Diferencias con el árbitro simplificado anterior (que sigue disponible por defecto):

* Masa inversa de jugadores 0,3 (la fija el script; el mapa dice 0,5). Con 0,3 el
  simulador reproduce 60 ticks reales con error p90 de 0,065 px; con 0,5, 12,6 px.
* Cancha de 670 de medio alto para salidas (el valor 600 queda sólo como escala de
  observación, para no cambiar lo que ven las políticas ni los datos humanos).
* Lateral: la pelota queda afuera en (x, ±688) sin clavarse; el rival no puede pasar
  la línea de ±555 y los que estaban dentro de ±270 px del punto se mueven a ±540.
  Se libera cuando la pelota vuelve hacia la cancha (|y| < 682; p99 real 682,0): antes,
  el que saca puede llevarla por fuera de la línea.
* Córner (±1140, ±660) y saque de arco (±1030, ±180): todos los jugadores quedan casi
  sin masa (invMass 100000) hasta liberar; en el saque de arco los rivales salen del
  área (x=±825). En el saque de arco nadie puede acercarse a menos de 33 px del punto
  durante 180 ticks (disco de radio 18 del script; mediana real 179). La patada se duplica
  a los 2 ticks (córner, ×1,98) o ×2,71 a los 3
  ticks (saque de arco) y la pelota recibe gravedad que decae ×0,97 por tick. Se liberan
  con la patada (el contacto previo no cuenta: los jugadores casi sin masa no la mueven).
* No hay vencimiento de saques en la sala: el simulador libera por seguridad a los
  900 ticks sin fin de episodio. Para entrenar puede fijarse un plazo (`deadline`, p. ej. 600
  ticks) que multa al equipo que no sacó; la evaluación no lo usa.

El último toque es el contacto físico exacto (distancia <= suma de radios + 0,01) o una
patada, evaluado tick a tick como el script (acierta el dueño del lateral en el 98,6% de
3708 casos; el alcance de patada de 4 px usado por el simulador acertaba el 87,7%).

Aproximaciones documentadas (no medidas como regla del script): los rivales no se
acercan a menos de 100 px de un córner (p10 observado: 162 px) y la salida se cobra en el
mismo tick del cruce (la sala tarda de 1 a ~9 ticks, sin regla simple).
"""
from __future__ import annotations

import numpy as np

CONTRACT = dict(
    version="RS4-b1-referee-1",
    line_half_h=670.0, lateral_ball_y=688.0, lateral_x_margin=10.0, lateral_release_y=682.0,
    spot_release_distance=3.0,
    corner=(1140.0, 660.0), goal_kick=(1030.0, 180.0),
    barrier_y=555.0, push_y=540.0, push_dx=270.0,
    box_front=840.0, box_half_h=320.0, box_push_x=825.0,
    play_inv_mass=0.3, piece_inv_mass=100000.0,
    corner_boost=1.98, corner_boost_delay=2, goal_kick_boost=2.71, goal_kick_boost_delay=3,
    corner_gravity_along=0.0438, corner_gravity_along_per_speed=-0.0173, corner_gravity_perp=-0.0169,
    goal_kick_gravity=-0.03, gravity_decay=0.97, gravity_ticks=143,
    safety_ticks=900, corner_rival_clearance=100.0, spot_clearance=33.0, goal_kick_hold_ticks=180,
)

LATERAL, CORNER, GOAL_KICK = 1, 2, 3


def _circle_obstacle(pos, vel, center, radius, mask):
    """Choque inelástico con un disco fijo: corrige la posición al borde y sólo anula la
    componente de velocidad hacia el centro (el jugador puede seguir deslizándose)."""
    center = np.asarray(center, dtype=np.float64).reshape(-1, 2)
    offset = pos - center[:, None, :]
    distance = np.hypot(offset[..., 0], offset[..., 1])
    close = mask & (distance < radius)
    if not close.any():
        return
    rows, players = np.nonzero(close)
    unit = offset[rows, players] / np.maximum(distance[rows, players], 1e-9)[:, None]
    degenerate = distance[rows, players] <= 1e-9
    unit[degenerate] = (1.0, 0.0)
    pos[rows, players] = center[rows] + unit * radius
    inward = np.minimum(0.0, (vel[rows, players] * unit).sum(axis=1))
    vel[rows, players] -= inward[:, None] * unit


class RSOneReferee:
    def __init__(self, env, contract=None):
        self.env = env
        self.c = dict(CONTRACT if contract is None else contract)
        n, p = env.N, env.P
        self.outside = np.zeros((n, p), dtype=bool)
        self.boost_left = np.zeros(n, dtype=np.int64)
        self.boost_factor = np.ones(n)
        self.gravity_left = np.zeros(n, dtype=np.int64)
        # La sala cobra salida al cruzar desde adentro: tras liberar un lateral con la pelota
        # todavía fuera de la línea (|y|≈681) no vuelve a cobrar hasta que regrese a la cancha.
        self.armed = np.ones(n, dtype=bool)
        self.contact = np.zeros((n, p), dtype=bool)   # contacto exacto del último tick
        self.out_tick = np.zeros(n, dtype=bool)       # salidas cobradas durante la decisión
        # Plazo de ENTRENAMIENTO (ticks; 0 = sin plazo, como la sala y la evaluación). Al cumplirse
        # sin ejecutar, el equipo que saca recibe la multa de saque trabado (rcfg.rs4_restart_stall) y el
        # juego sigue: sin él, esperar la liberación de seguridad era gratis y la política dejó de
        # patear córners (A1: 64% liberados por seguridad). La liberación sigue en safety_ticks.
        self.deadline = 0
        sim = env.sim
        fp = sim.first_player
        sim.invmass[fp:] = self.c["play_inv_mass"]
        sim.inv_env[:, fp:] = self.c["play_inv_mass"]

    # ------------------------------------------------------------------ estado
    def reset_rows(self, idx):
        idx = np.asarray(idx, dtype=np.int64)
        sim = self.env.sim
        sim.inv_env[idx, sim.first_player:] = self.c["play_inv_mass"]
        self.outside[idx] = False
        self.boost_left[idx] = 0
        self.boost_factor[idx] = 1.0
        self.gravity_left[idx] = 0
        self.armed[idx] = True
        sim.ball_grav[idx] = 0.0

    @property
    def busy(self):
        """Siempre tick a tick: contacto exacto, salidas e impulsos necesitan su tick
        (costo medido: +0,28 ms por decisión de 1152 jugadores, ~2% del entrenamiento)."""
        return True

    def out_of_bounds(self):
        env = self.env
        b = env.sim.ball_pos
        r = env.sim.st.ball["radius"]
        line_h, line_w, goal_h = self.c["line_half_h"], env.field_w, env.sim.st.goal_half_height
        inside = (np.abs(b[:, 1]) <= line_h + r) & (np.abs(b[:, 0]) <= line_w + r)
        self.armed |= inside & (env.setpiece_team < 0)
        side = self.armed & (np.abs(b[:, 1]) > line_h + r) & (np.abs(b[:, 0]) < line_w)
        end = self.armed & (np.abs(b[:, 0]) > line_w + r) & (np.abs(b[:, 1]) > goal_h)
        return side, end

    # ------------------------------------------------------------------ saques
    def set_piece(self, idx):
        env, sim, c = self.env, self.env.sim, self.c
        line_h, line_w = c["line_half_h"], env.field_w
        for n in np.asarray(idx, dtype=np.int64):
            bx, by = sim.pos[n, 0]
            last = env.last_touch[n]
            if last < 0:
                last = int(env.rng.integers(0, 2))
            sx = 1.0 if bx >= 0 else -1.0
            sy = 1.0 if by >= 0 else -1.0
            if abs(by) > line_h and abs(bx) < line_w:
                kind, taker = LATERAL, 1 - last
                margin = line_w - c["lateral_x_margin"]
                spot = (float(np.clip(bx, -margin, margin)), sy * c["lateral_ball_y"])
            else:
                defender = 1 if sx > 0 else 0  # el arco de +x es del azul
                if last == defender:
                    kind, taker = CORNER, 1 - defender
                    spot = (sx * c["corner"][0], sy * c["corner"][1])
                else:
                    kind, taker = GOAL_KICK, defender
                    spot = (sx * c["goal_kick"][0], sy * c["goal_kick"][1])
            self.start(n, kind, taker, spot)
        self.protect()

    def start(self, n, kind, taker, spot):
        """Acciones del script al iniciar un saque (también usadas por la prueba de conformidad)."""
        env, sim, c = self.env, self.env.sim, self.c
        fp = sim.first_player
        sx = 1.0 if spot[0] >= 0 else -1.0
        sy = 1.0 if spot[1] >= 0 else -1.0
        sim.pos[n, 0] = spot
        sim.vel[n, 0] = 0.0
        sim._reset_ball_state([n])
        sim.kick_cancel[n] = False
        self.armed[n] = False
        self.boost_left[n] = 0
        self.gravity_left[n] = 0
        rivals = sim.player_team != taker
        pos = sim.player_pos[n]
        radius = sim.st.player["radius"]
        self.outside[n] = False
        if kind == LATERAL:
            near = rivals & (sy * pos[:, 1] > c["push_y"]) & (np.abs(pos[:, 0] - spot[0]) < c["push_dx"])
            pos[near, 1] = sy * c["push_y"]  # el script sólo fija y; no toca la velocidad
            self.outside[n] = rivals & (sy * pos[:, 1] > c["barrier_y"] + radius)
        else:
            sim.inv_env[n, fp:] = c["piece_inv_mass"]
            if kind == GOAL_KICK:
                inside = rivals & (sx * pos[:, 0] > c["box_front"] - radius) & (np.abs(pos[:, 1]) < c["box_half_h"] + radius)
                pos[inside, 0] = sx * c["box_push_x"]  # el script sólo fija x
            # Disco transitorio del script (radio 18): despeja el punto exacto del saque.
            _circle_obstacle(sim.player_pos[n:n + 1], sim.player_vel[n:n + 1], np.asarray([spot], dtype=np.float64),
                             c["spot_clearance"], np.ones((1, env.P), dtype=bool))
        env._begin_set_piece(n, taker, kind, spot, c["safety_ticks"])

    def protect(self):
        env, sim, c = self.env, self.env.sim, self.c
        active = env.setpiece_team >= 0
        if not active.any():
            return
        radius = sim.st.player["radius"]
        pos, vel = sim.player_pos, sim.player_vel
        rivals = active[:, None] & (sim.player_team[None, :] != env.setpiece_team[:, None])
        sy = np.where(env.setpiece_pos[:, 1] >= 0, 1.0, -1.0)[:, None]
        sx = np.where(env.setpiece_pos[:, 0] >= 0, 1.0, -1.0)[:, None]
        kind = env.setpiece_kind[:, None]
        # Lateral: segmento c1 en ±555 (choca por ambos lados).
        lateral = rivals & (kind == LATERAL)
        inner = lateral & ~self.outside & (sy * pos[..., 1] > c["barrier_y"] - radius)
        pos[..., 1] = np.where(inner, sy * (c["barrier_y"] - radius), pos[..., 1])
        outer = lateral & self.outside & (sy * pos[..., 1] < c["barrier_y"] + radius)
        pos[..., 1] = np.where(outer, sy * (c["barrier_y"] + radius), pos[..., 1])
        vel[..., 1] = np.where(inner | outer, 0.0, vel[..., 1])
        # Saque de arco: segmentos c0 del área grande.
        goal_kick = rivals & (kind == GOAL_KICK)
        invaded = goal_kick & (sx * pos[..., 0] > c["box_front"] - radius) & (np.abs(pos[..., 1]) < c["box_half_h"] + radius)
        pos[..., 0] = np.where(invaded, sx * (c["box_front"] - radius), pos[..., 0])
        vel[..., 0] = np.where(invaded, 0.0, vel[..., 0])
        # Saque de arco: el disco del script (radio 18) despeja el punto durante 180 ticks.
        held = active & (env.setpiece_kind == GOAL_KICK) & (env.setpiece_ticks < c["goal_kick_hold_ticks"])
        if held.any():
            _circle_obstacle(pos, vel, env.setpiece_pos, c["spot_clearance"],
                             np.broadcast_to(held[:, None], pos.shape[:2]))
        # Córner: distancia observada de los rivales (aproximación documentada).
        corner = rivals & (kind == CORNER)
        if corner.any():
            _circle_obstacle(pos, vel, env.setpiece_pos, c["corner_rival_clearance"], corner)

    def pre_tick(self, actions):
        self.protect()
        return actions

    def post_tick(self, goal):
        env, sim, c = self.env, self.env.sim, self.c
        # Último toque como el script: contacto físico exacto o patada, después de la física.
        offset = sim.player_pos - sim.ball_pos[:, None, :]
        reach = sim.st.player["radius"] + sim.st.ball["radius"] + 0.01
        self.contact = (np.hypot(offset[..., 0], offset[..., 1]) <= reach) | sim.kicked
        for team in (0, 1):
            touched = self.contact[:, sim.player_team == team].any(axis=1)
            other = self.contact[:, sim.player_team != team].any(axis=1)
            env.last_touch[touched & ~other] = team
        owner = env.setpiece_team.copy()
        active = owner >= 0
        env.setpiece_ticks[active] += 1
        contact = self.contact
        # 1. Impulsos y efecto de saques ya liberados.
        pending = self.boost_left > 0
        self.boost_left[pending] -= 1
        fire = pending & (self.boost_left == 0)
        if fire.any():
            sim.vel[fire, 0] *= self.boost_factor[fire, None]
        spinning = self.gravity_left > 0
        if spinning.any():
            stop = spinning & ((contact.any(axis=1) & ~(pending | fire)) | (goal != 0))
            sim.ball_grav[stop] = 0.0
            self.gravity_left[stop] = 0
            keep = spinning & ~stop
            sim.ball_grav[keep] *= c["gravity_decay"]
            self.gravity_left[keep] -= 1
            sim.ball_grav[keep & (self.gravity_left == 0)] = 0.0
        # 2. Liberación del saque activo (reglas medidas en las grabaciones).
        own = owner[:, None] == sim.player_team[None, :]
        kind = env.setpiece_kind
        ball = sim.ball_pos
        lateral_in = (kind == LATERAL) & (np.abs(ball[:, 1]) < c["lateral_release_y"])
        moved = np.hypot(ball[:, 0] - env.setpiece_pos[:, 0], ball[:, 1] - env.setpiece_pos[:, 1]) > c["spot_release_distance"]
        dead_ball = (kind != LATERAL) & (sim.kicked.any(axis=1) | moved)
        released = active & (lateral_in | dead_ball | (env.setpiece_ticks >= c["safety_ticks"]) | (goal != 0))
        executed = active & (lateral_in | dead_ball | (goal != 0))
        late = (active & ~executed & (env.setpiece_ticks == self.deadline) if self.deadline > 0
                else np.zeros(env.N, dtype=bool))
        kicked_by_taker = released & (sim.kicked & own).any(axis=1)
        for n in np.flatnonzero(kicked_by_taker & np.isin(env.setpiece_kind, (CORNER, GOAL_KICK))):
            kicker = int(np.flatnonzero(sim.kicked[n] & own[n])[0])
            self._schedule(n, int(env.setpiece_kind[n]), sim.player_vel[n, kicker])
        if released.any():
            sim.inv_env[released, sim.first_player:] = c["play_inv_mass"]
            self.outside[released] = False
            env.setpiece_team[released] = -1
            env.setpiece_kind[released] = 0
        # 3. Salida en este mismo tick (sin gol): nuevo saque.
        side, end = self.out_of_bounds()
        new = (side | end) & (env.setpiece_team < 0) & (goal == 0)
        if new.any():
            self.out_tick |= new
            self.set_piece(np.flatnonzero(new))
        self.protect()
        return late, owner

    def _schedule(self, n, kind, player_velocity):
        sim, c = self.env.sim, self.c
        velocity = sim.vel[n, 0].copy()
        speed = float(np.hypot(*velocity))
        if speed < 1e-6:
            return
        u = velocity / speed
        normal = np.array([-u[1], u[0]])
        pv = np.asarray(player_velocity, dtype=np.float64)
        if kind == CORNER:
            along = c["corner_gravity_along"] + c["corner_gravity_along_per_speed"] * float(pv @ u)
            gravity = along * u + c["corner_gravity_perp"] * float(pv @ normal) * normal
            self.boost_factor[n], self.boost_left[n] = c["corner_boost"], c["corner_boost_delay"]
        else:
            gravity = c["goal_kick_gravity"] * pv
            self.boost_factor[n], self.boost_left[n] = c["goal_kick_boost"], c["goal_kick_boost_delay"]
        sim.ball_grav[n] = gravity
        self.gravity_left[n] = c["gravity_ticks"]

    def after_fused(self, frames, touched):
        """Ruta fusionada (sin saques ni impulsos pendientes): decaimiento por decisión."""
        sim, c = self.env.sim, self.c
        spinning = self.gravity_left > 0
        if not spinning.any():
            return
        stop = spinning & touched
        sim.ball_grav[stop] = 0.0
        self.gravity_left[stop] = 0
        keep = spinning & ~stop
        sim.ball_grav[keep] *= c["gravity_decay"] ** frames
        self.gravity_left[keep] = np.maximum(0, self.gravity_left[keep] - frames)
        sim.ball_grav[keep & (self.gravity_left == 0)] = 0.0
