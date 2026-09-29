"""Entorno vectorizado multi-agente de HaxBall.

Cada partido tiene P = 2·n_per_team jugadores y *todos* son agentes. Cada agente ve el
partido desde su equipo: para el azul se espeja el eje x, así que siempre "ataca hacia +x"
y un único modelo sirve para ambos lados. Las acciones también están en ese marco propio.

API (todo en lote):
    obs = env.reset()                         # (N, P, OBS_DIM)
    obs, rew, done, info = env.step(actions)  # actions (N, P) en 0..17
    info["truncated"] (N,) bool, info["final_obs"] (N, P, OBS_DIM) obs antes del reset
"""
from __future__ import annotations

import numpy as np

from sim.physics import N_ACTIONS, BatchSim

from .pegeche import N_RULE_FEATS, PegecheRules
from .rewards import RewardConfig, potentials, defense_support_potential

# espejo en x de las 9 direcciones de movimiento (ver MOVE_DIRS en sim/physics.py)
_MIRROR_MOVE = np.array([0, 1, 8, 7, 6, 5, 4, 3, 2])
MIRROR_ACTION = np.concatenate([_MIRROR_MOVE, _MIRROR_MOVE + 9])

POS_SCALE = 400.0
VEL_SCALE = 5.0


N_PS_FEATS = 6  # entradas extra con powershot: cargada, progreso de carga, invMass, gravedad (2), soy quien carga


def obs_dim(n_per_team: int, powershot: bool = False, layout: str = "flat", max_entities: int = 0) -> int:
    if layout == "universal":
        return U_SELF_DIM + U_ENT_DIM * max(max_entities, 2 * n_per_team - 1)
    if layout == "entities":
        return self_dim(powershot) + ENT_DIM * (2 * n_per_team - 1)
    return 22 + 4 * (n_per_team - 1) + 6 * n_per_team + (N_PS_FEATS if powershot else 0)


# Layout "universal" (un solo modelo para cualquier mapa y formato, ver train/model.py::SetActorCritic):
#   [bloque propio (U_SELF_DIM) | max_entities x U_ENT_DIM]
#   bloque propio: posiciones normalizadas por el tamaño de la cancha, descriptor del mapa, reglas,
#   distancias a paredes (8 rayos desde el jugador y 8 desde la pelota), powershot y estado del script
#   de la sala (env/pegeche.py::features: pelota parada, slide, falta para pedir...; ceros si no hay).
#   entidad = [presente, es_rival, pos relativa (2), velocidad (2), pelota - entidad (2)], rellenadas con
#   ceros (presente = 0) hasta max_entities: el ancho de la obs no depende del formato. Los expulsados
#   (reglas Pegeche) no se ven: van al final como ausentes, igual que en la sala (pasan a espectadores).
U_ENT_DIM = 8
N_MAP_FEATS = 10
U_SELF_DIM = 17 + 5 + N_PS_FEATS + N_MAP_FEATS + 16 + 2 + N_RULE_FEATS  # = 71


# Layout "entities" (sirve para cualquier tamaño de equipo, ver train/model.py::EntityActorCritic):
#   [bloque propio (self_dim) | compañeros (T-1) x ENT_DIM | rivales T x ENT_DIM]
#   entidad = [es_rival, pos relativa (2), velocidad (2), pelota - entidad (2)]
ENT_DIM = 7


def self_dim(powershot: bool = False) -> int:
    return 22 + (N_PS_FEATS if powershot else 0)


class HaxballEnv:
    def __init__(self, n_envs: int, n_per_team: int = 1, stadium: str = "classic",
                 frame_skip: int = 3, max_ticks: int = 60 * 60 * 2,
                 random_reset_prob: float = 0.5, reward: RewardConfig | None = None,
                 seed: int | None = None, action_delay_max: int = 0,
                 kickoff_timeout: int = 0, powershot: bool | dict = False,
                 out_of_bounds: bool = False, obs_layout: str = "flat", max_entities: int = 0,
                 rules: str | None = None, optimize_rollout: bool = True):
        self.sim = BatchSim(n_envs, n_per_team, n_per_team, stadium, seed=seed, powershot=powershot)
        self.N = n_envs
        self.T = n_per_team
        self.P = 2 * n_per_team
        self.frame_skip = frame_skip
        self.max_ticks = max_ticks
        self.random_reset_prob = random_reset_prob
        self.rcfg = reward or RewardConfig()
        self.rng = np.random.default_rng(seed)
        self.obs_layout = obs_layout
        self.max_entities = max(max_entities, 2 * n_per_team - 1)
        self.obs_dim = obs_dim(n_per_team, self.sim.ps_on, obs_layout, self.max_entities)
        self.self_dim = U_SELF_DIM if obs_layout == "universal" else self_dim(self.sim.ps_on)
        if obs_layout == "universal":
            from .geometry import StadiumRays
            self.rays = StadiumRays(self.sim.st)
        self.n_actions = N_ACTIONS
        st = self.sim.st
        self.goal_x = st.goal_x
        self.field_w = st.field_half_w
        # Real Soccer ONE: el saque de arco real tiene impulso extra.
        self.goal_kick_speed = 10.5 if stadium == "rs_one" else 0.0
        # signo de x por jugador: +1 rojo (ataca a +x), -1 azul
        self.sign = np.where(self.sim.player_team == 0, 1.0, -1.0)
        self.ticks = np.zeros(self.N, dtype=np.int64)
        self.score = np.zeros((self.N, 2), dtype=np.int64)  # acumulado (estadísticas)
        self._phi = None
        # Retraso de acciones (lag de red): cada partido sortea d en [0, action_delay_max] ticks
        # al reiniciarse, y la acción elegida en t recién se aplica en t + d (antes sigue la anterior).
        self.action_delay_max = action_delay_max
        # Si el equipo que saca no toca la pelota en `kickoff_timeout` ticks (0 = sin límite), recibe
        # -kickoff_stall y el episodio termina. Sin esto el agente aprende a no sacar nunca: la pelota
        # queda quieta, el rival no puede entrar al círculo y no le pueden hacer gol.
        self.kickoff_timeout = kickoff_timeout
        self.kickoff_ticks = np.zeros(self.N, dtype=np.int64)
        self.kickoff_limit = np.full(self.N, kickoff_timeout, dtype=np.int64)
        self.delay = np.zeros(self.N, dtype=np.int64)
        n_hist = 1 + -(-action_delay_max // frame_skip)  # ceil
        self.act_hist = np.zeros((n_hist, self.N, self.P), dtype=np.int64)  # [0] = la más nueva
        # Pelota afuera (sin pelotas paradas todavía): termina el episodio y el equipo que la tocó
        # último recibe -out_penalty (en la sala real pierde la posesión).
        self.out_of_bounds = out_of_bounds
        self.last_touch = np.full(self.N, -1, dtype=np.int64)  # equipo que tocó último (-1 nadie)
        self.field_h = st.field_half_h
        self.stuck_ticks = np.zeros(self.N, dtype=np.int64)  # pelota casi quieta pegada a una línea
        self.setpiece_dist = 50.0      # los rivales del que saca se corren a esta distancia
        self.stuck_limit = 180         # 3 s trabada contra una línea => se cobra como salida
        # Estado privado del árbitro simplificado: nunca se añade a las observaciones.
        self.setpiece_team = np.full(self.N, -1, dtype=np.int64)
        self.setpiece_kind = np.zeros(self.N, dtype=np.int64)  # 1 lateral, 2 córner, 3 saque de arco
        self.setpiece_ticks = np.zeros(self.N, dtype=np.int64)
        self.setpiece_pos = np.zeros((self.N, 2))
        self.setpiece_timeout = 420    # protección finita (7 s); no réplica del script de cada sala
        self.setpiece_limit = np.full(self.N, self.setpiece_timeout, dtype=np.int64)
        # Script completo de la sala (rules="pegeche", env/pegeche.py): reemplaza las pelotas paradas
        # simplificadas de arriba por las reales y suma slide, faltas, penales y tarjetas.
        if rules not in (None, "pegeche"):
            raise ValueError(f"reglas desconocidas: {rules}")
        if rules is not None and obs_layout != "universal":
            raise ValueError("las reglas del script sólo están en la obs universal")
        self.rules = PegecheRules(self.sim, self.rng) if rules == "pegeche" else None
        if self.rules is not None:
            self.out_of_bounds = True
        self.optimize_rollout = optimize_rollout

    # ----------------------------------------------------------------- reset
    def _reset_envs(self, idx, kickoff_team=None):
        if len(idx) == 0:
            return
        rnd = self.rng.random(len(idx)) < self.random_reset_prob
        self.sim.reset_random(idx[rnd])
        ko = idx[~rnd]
        if kickoff_team is None:
            kt = self.rng.integers(0, 2, len(ko))
        else:
            kt = kickoff_team[~rnd]
        self.sim.reset_kickoff(ko, kickoff_team=kt)
        self.kickoff_limit[idx] = self.kickoff_timeout
        if self.kickoff_timeout > 0:
            for n in ko:
                travel = self._restart_travel_ticks(
                    n, self.sim.kickoff_team[n]
                )
                self.kickoff_limit[n] = max(
                    self.kickoff_timeout,
                    travel + self.kickoff_timeout
                )
        self.ticks[idx] = 0
        self.kickoff_ticks[idx] = 0
        self.last_touch[idx] = -1
        self.stuck_ticks[idx] = 0
        self.setpiece_team[idx] = -1
        self.setpiece_kind[idx] = 0
        self.setpiece_ticks[idx] = 0
        self.setpiece_limit[idx] = self.setpiece_timeout
        self.act_hist[:, idx] = 0
        if self.rules is not None:
            self.rules.reset(idx)
        if self.action_delay_max > 0:
            self.delay[idx] = self.rng.integers(0, self.action_delay_max + 1, len(idx))

    def _set_piece(self, idx) -> None:
        """Pelota parada simplificada (lateral / córner / saque de arco).
        Protección hasta la patada del equipo que saca o el timeout; no expone estado privado."""
        sim = self.sim
        r = sim.st.ball["radius"]
        W, H, GH = self.field_w, self.field_h, sim.st.goal_half_height
        for n in idx:
            bx, by = sim.pos[n, 0]
            lt = self.last_touch[n]
            if lt < 0:
                lt = int(self.rng.integers(0, 2))
            sx = 1.0 if bx >= 0 else -1.0
            sy = 1.0 if by >= 0 else -1.0
            if abs(by) > H - r - 12 and abs(bx) < W - 2 * r:      # lateral
                kind = 1
                taker = 1 - lt
                nb = (np.clip(bx, -W + 2 * r, W - 2 * r), sy * (H - r - 1))
            else:                                                 # línea de fondo
                defender = 1 if sx > 0 else 0                     # el arco de +x es del azul
                if lt == defender:                                # la tocó el que defiende: córner
                    kind = 2
                    taker = 1 - defender
                    nb = (sx * (W - r - 1), sy * (H - r - 1))
                else:                                             # saque de arco
                    kind = 3
                    taker = defender
                    nb = (
                        sx * 1030.0,
                        sy * 180.0
                    )   
            sim.pos[n, 0] = nb
            sim.vel[n, 0] = 0.0
            sim._reset_ball_state([n])
            sim.kick_cancel[n] = False
            self.setpiece_team[n] = taker
            self.setpiece_kind[n] = kind
            self.setpiece_ticks[n] = 0
            self.setpiece_pos[n] = nb
            # En mapas grandes el ejecutor puede necesitar más de 7 s sólo para llegar.
            self.setpiece_limit[n] = max(self.setpiece_timeout,
                                         self._restart_travel_ticks(n, taker) + 180)
            self.last_touch[n] = -1
            self.stuck_ticks[n] = 0
        self._protect_setpieces()

    def _restart_travel_ticks(self, n, team):
        st = self.sim.st
        eligible = self.sim.player_team == team
        distance = np.linalg.norm(self.sim.player_pos[n, eligible] - self.sim.ball_pos[n], axis=1)
        if not len(distance):
            return 0
        speed = st.player["acceleration"] * st.player["damping"] / max(1.0 - st.player["damping"], 1e-6)
        return int(np.ceil(distance.min() / max(speed, 0.1))) + 30

    def _protect_setpieces(self):
        """Aplica barreras antes/después de cada tick, no sólo al colocar la pelota.

        Área rectangular aproximada para el perfil real (no se conoce su script):
        proporciones 840/1150 y 320/600, como la geometría RS de referencia.
        """
        sim = self.sim
        pp, pv = sim.player_pos, sim.player_vel
        active = self.setpiece_team >= 0
        if not active.any():
            return
        rp = sim.st.player["radius"]
        distance = max(self.setpiece_dist, rp + sim.st.ball["radius"] + 5.0)
        rival = active[:, None] & (sim.player_team[None, :] != self.setpiece_team[:, None])
        sx = np.where(self.setpiece_pos[:, 0] >= 0, 1.0, -1.0)
        front = self.field_w * (840.0 / 1150) - rp
        side = self.field_h * (320.0 / 600) + rp
        invaded = (rival & (self.setpiece_kind[:, None] == 3)
                   & (sx[:, None] * pp[..., 0] > front) & (np.abs(pp[..., 1]) < side))
        rows, players = np.nonzero(invaded)
        pp[rows, players, 0] = sx[rows] * front
        pv[invaded] = 0.0
        d = pp - self.setpiece_pos[:, None, :]
        norm = np.hypot(d[..., 0], d[..., 1])
        near = rival & (norm < distance)
        rows, players = np.nonzero(near)
        u = d[near] / np.maximum(norm[near], 1e-9)[:, None]
        coincident = norm[near] <= 1e-6
        u[coincident, 0] = -sx[rows[coincident]]
        u[coincident, 1] = 0.0
        pp[near] = self.setpiece_pos[rows] + u * distance
        pv[near] = 0.0

    def _setpiece_pre_tick(self, actions):
        self._protect_setpieces()
        blocked = ((self.setpiece_team[:, None] >= 0)
                   & (self.setpiece_team[:, None] != self.sim.player_team[None, :]))
        return np.where(blocked, actions % 9, actions)

    def _setpiece_post_tick(self, goal):
        active = self.setpiece_team >= 0
        self.setpiece_ticks[active] += 1

        own = self.setpiece_team[:, None] == self.sim.player_team[None, :]

        # El equipo encargado realmente pateó.
        kicked_by_taker = active & (self.sim.kicked & own).any(axis=1)

        # Saque de arco = kind 3.
        goal_kick = kicked_by_taker & (self.setpiece_kind == 3)

        # Real Soccer ONE tiene un saque de arco mucho más potente que un kick normal.
        if self.goal_kick_speed > 0 and goal_kick.any():
            v = self.sim.vel[goal_kick, 0]
            speed = np.linalg.norm(v, axis=1)

            scale = np.ones_like(speed)
            valid = speed > 1e-6

            scale[valid] = np.maximum(
                1.0,
                self.goal_kick_speed / speed[valid]
            )

            self.sim.vel[goal_kick, 0] *= scale[:, None]

        released = active & (
            kicked_by_taker
            | (self.setpiece_ticks >= self.setpiece_limit)
            | (goal != 0)
        )

    self.setpiece_team[released] = -1
    self.setpiece_kind[released] = 0

    waiting = active & ~released

    self.sim.pos[waiting, 0] = self.setpiece_pos[waiting]
    self.sim.vel[waiting, 0] = 0.0

    self._protect_setpieces()

    def reset(self) -> np.ndarray:
        self._reset_envs(np.arange(self.N))
        self._phi = self._potentials()
        return self.observe()

    # ----------------------------------------------------------------- obs
    def _own(self, xy):
        """(N, P, 2) coordenadas mundo -> marco propio de cada agente (espejo x para azul)."""
        out = xy.copy()
        out[..., 0] *= self.sign
        return out

    def _map_features(self) -> np.ndarray:
        """Descriptor del mapa y de las reglas (igual para todos los agentes del entorno)."""
        st = self.sim.st
        return np.array([st.field_half_w / 1000, st.field_half_h / 1000, st.goal_half_height / 100,
                         self.goal_x / st.field_half_w, st.ball["radius"] / 10, st.player["radius"] / 15,
                         st.kickoff_radius / 100, float(self.sim.ps_on), float(self.out_of_bounds),
                         self.T / 11], dtype=np.float64)

    def _observe_universal(self) -> np.ndarray:
        from .geometry import N_RAYS, RAY_MAX
        sim, st = self.sim, self.sim.st
        N, P, T = self.N, self.P, self.T
        W, H = st.field_half_w, st.field_half_h
        fs = np.array([W, H])
        bp = self._own(np.broadcast_to(sim.ball_pos[:, None, :], (N, P, 2)))
        bv = self._own(np.broadcast_to(sim.ball_vel[:, None, :], (N, P, 2)))
        pp_w, pv_w = sim.player_pos, sim.player_vel
        pp, pv = self._own(pp_w), self._own(pv_w)
        rel_ball = bp - pp
        opp_goal = np.array([self.goal_x, 0.0])
        team = sim.player_team
        feats = [pp / fs, pv / VEL_SCALE, bp / fs, bv / VEL_SCALE,
                 rel_ball / POS_SCALE, np.linalg.norm(rel_ball, axis=-1, keepdims=True) / POS_SCALE,
                 (opp_goal - pp) / W, (-opp_goal - pp) / W, (opp_goal - bp) / W]
        # flags (mismos que los otros layouts)
        my_ko = sim.kickoff[:, None] & (sim.kickoff_team[:, None] == team[None, :])
        feats += [(~sim.kick_cancel).astype(np.float64)[..., None],
                  np.broadcast_to(sim.kickoff[:, None, None], (N, P, 1)).astype(np.float64),
                  my_ko[..., None].astype(np.float64), sim.touch.astype(np.float64)[..., None],
                  np.broadcast_to((self.ticks / self.max_ticks)[:, None, None], (N, P, 1))]
        if sim.ps_on:
            cfg = sim.ps_cfg
            prog = np.where(sim.ps_charge > 0, 1.0 - sim.ps_charge / cfg["charge"], 0.0)
            invb = (sim.inv_env[:, 0] - cfg["base_inv"]) / (cfg["power_inv"] - cfg["base_inv"])
            feats += [np.broadcast_to(sim.ps_comba[:, None, None], (N, P, 1)).astype(np.float64),
                      np.broadcast_to(prog[:, None, None], (N, P, 1)),
                      np.broadcast_to(invb[:, None, None], (N, P, 1)),
                      self._own(np.broadcast_to(sim.ball_grav[:, None, :], (N, P, 2))) / cfg["grav"],
                      (sim.ps_held[:, None] == np.arange(P)[None, :]).astype(np.float64)[..., None]]
        else:
            feats.append(np.zeros((N, P, N_PS_FEATS)))
        feats.append(np.broadcast_to(self._map_features(), (N, P, N_MAP_FEATS)))
        # rayos: desde cada jugador (contra lo que frena a su equipo) y desde la pelota
        sgn = np.broadcast_to(self.sign[None, :], (N, P))
        rays_p = np.zeros((N, P, N_RAYS))
        for t, which in ((0, "red"), (1, "blue")):
            sel = team == t
            o = pp_w[:, sel].reshape(-1, 2)
            rays_p[:, sel] = self.rays.cast(o, which, sgn[:, sel].reshape(-1)).reshape(N, sel.sum(), N_RAYS)
        rb = self.rays.cast(sim.ball_pos, "ball")                                # (N, R) mundo
        from .geometry import RAY_MIRROR
        rays_b = np.where((self.sign < 0)[None, :, None], rb[:, None, RAY_MIRROR], rb[:, None, :])
        feats += [rays_p / RAY_MAX, rays_b / RAY_MAX]
        feats.append(np.broadcast_to(np.array([(T - 1) / 10, T / 10]), (N, P, 2)))
        feats.append(self.rules.features() if self.rules is not None else np.zeros((N, P, N_RULE_FEATS)))
        own = np.concatenate(feats, axis=-1)
        assert own.shape[-1] == U_SELF_DIM, own.shape

        # entidades: compañeros y rivales en el marco propio, rellenadas hasta max_entities
        E = self.max_entities
        ent = np.zeros((N, P, E, U_ENT_DIM))
        for p in range(P):
            others = [q for q in range(P) if q != p]
            others.sort(key=lambda q: team[q] != team[p])  # compañeros primero
            q = np.array(others, dtype=np.int64)
            s = self.sign[p]
            op = pp_w[:, q].copy(); op[..., 0] *= s
            ov = pv_w[:, q].copy(); ov[..., 0] *= s
            k = len(q)
            ent[:, p, :k, 0] = 1.0
            ent[:, p, :k, 1] = (team[q] != team[p]).astype(np.float64)[None, :]
            ent[:, p, :k, 2:4] = (op - pp[:, p, None]) / POS_SCALE
            ent[:, p, :k, 4:6] = ov / VEL_SCALE
            ent[:, p, :k, 6:8] = (bp[:, p, None] - op) / POS_SCALE
            if self.rules is not None and self.rules.expelled.any():
                ent[:, p, :k][self.rules.expelled[:, q]] = 0.0
        if self.rules is not None and self.rules.expelled.any():
            order = np.argsort(-ent[..., 0], axis=2, kind="stable")  # presentes primero
            ent = np.take_along_axis(ent, order[..., None], axis=2)
        return np.concatenate([own, ent.reshape(N, P, -1)], axis=-1).astype(np.float32)

    def observe(self, indices=None) -> np.ndarray:
        if self.obs_layout == "universal":
            if self.optimize_rollout:
                from .observation import observe_universal
                return observe_universal(self, indices)
            if indices is not None:
                return self._observe_universal()[indices]
            return self._observe_universal()
        if indices is not None:
            return self.observe()[indices]
        sim = self.sim
        N, P, T = self.N, self.P, self.T
        bp = np.broadcast_to(sim.ball_pos[:, None, :], (N, P, 2))
        bv = np.broadcast_to(sim.ball_vel[:, None, :], (N, P, 2))
        bp, bv = self._own(bp), self._own(bv)
        pp_w, pv_w = sim.player_pos, sim.player_vel
        pp, pv = self._own(pp_w), self._own(pv_w)

        rel_ball = bp - pp
        dist_ball = np.linalg.norm(rel_ball, axis=-1, keepdims=True)
        opp_goal = np.array([self.goal_x, 0.0])
        own_goal = -opp_goal

        feats = [pp / POS_SCALE, pv / VEL_SCALE,
                 bp / POS_SCALE, bv / VEL_SCALE, rel_ball / POS_SCALE, dist_ball / POS_SCALE,
                 (opp_goal - pp) / POS_SCALE, (own_goal - pp) / POS_SCALE,
                 (opp_goal - bp) / POS_SCALE]
        n_self_base = len(feats)

        # compañeros y rivales, en el marco del agente
        team = sim.player_team
        mates, opps = [], []
        for p in range(P):
            same = [q for q in range(P) if team[q] == team[p] and q != p]
            other = [q for q in range(P) if team[q] != team[p]]
            mates.append(same)
            opps.append(other)
        mates = np.array(mates, dtype=np.int64).reshape(P, T - 1)
        opps = np.array(opps, dtype=np.int64).reshape(P, T)
        sgn = self.sign[None, :, None, None]
        if T > 1:
            mp = pp_w[:, mates].copy(); mp[..., 0] *= sgn[..., 0]
            mv = pv_w[:, mates].copy(); mv[..., 0] *= sgn[..., 0]
            feats += [((mp - pp[:, :, None]) / POS_SCALE).reshape(N, P, -1),
                      (mv / VEL_SCALE).reshape(N, P, -1)]
        op = pp_w[:, opps].copy(); op[..., 0] *= sgn[..., 0]
        ov = pv_w[:, opps].copy(); ov[..., 0] *= sgn[..., 0]
        feats += [((op - pp[:, :, None]) / POS_SCALE).reshape(N, P, -1),
                  (ov / VEL_SCALE).reshape(N, P, -1),
                  ((bp[:, :, None] - op) / POS_SCALE).reshape(N, P, -1)]

        can_kick = (~sim.kick_cancel).astype(np.float64)[..., None]
        kickoff = np.broadcast_to(sim.kickoff[:, None, None], (N, P, 1)).astype(np.float64)
        my_ko = (sim.kickoff[:, None] & (sim.kickoff_team[:, None] == team[None, :]))[..., None]
        touching = sim.touch.astype(np.float64)[..., None]
        tfrac = np.broadcast_to((self.ticks / self.max_ticks)[:, None, None], (N, P, 1))
        flags = [can_kick, kickoff, my_ko.astype(np.float64), touching, tfrac]
        feats += flags
        ps_feats = []
        if sim.ps_on:
            cfg = sim.ps_cfg
            comba = np.broadcast_to(sim.ps_comba[:, None, None], (N, P, 1)).astype(np.float64)
            charging = sim.ps_charge > 0
            prog = np.where(charging, 1.0 - sim.ps_charge / cfg["charge"], 0.0)
            prog = np.broadcast_to(prog[:, None, None], (N, P, 1))
            invb = (sim.inv_env[:, 0] - cfg["base_inv"]) / (cfg["power_inv"] - cfg["base_inv"])
            invb = np.broadcast_to(invb[:, None, None], (N, P, 1))
            grav = self._own(np.broadcast_to(sim.ball_grav[:, None, :], (N, P, 2))) / cfg["grav"]
            mine = (sim.ps_held[:, None] == np.arange(P)[None, :]).astype(np.float64)[..., None]
            ps_feats = [comba, prog, invb, grav, mine]
            feats += ps_feats
        if self.obs_layout == "entities":
            ents = []
            if T > 1:
                ents.append(np.concatenate([np.zeros((N, P, T - 1, 1)), (mp - pp[:, :, None]) / POS_SCALE,
                                            mv / VEL_SCALE, (bp[:, :, None] - mp) / POS_SCALE], axis=-1))
            ents.append(np.concatenate([np.ones((N, P, T, 1)), (op - pp[:, :, None]) / POS_SCALE,
                                        ov / VEL_SCALE, (bp[:, :, None] - op) / POS_SCALE], axis=-1))
            ent = np.concatenate(ents, axis=2).reshape(N, P, -1)
            blocks = feats[:n_self_base] + flags + ps_feats + [ent]
            return np.concatenate(blocks, axis=-1).astype(np.float32)
        return np.concatenate(feats, axis=-1).astype(np.float32)

    # ----------------------------------------------------------------- step
    def _potentials(self):
        sim = self.sim
        bx = sim.ball_pos[:, None, 0] * self.sign[None, :]
        by = np.broadcast_to(sim.ball_pos[:, None, 1], bx.shape)
        d = self._team_ball_dist()
        phi_ball, phi_near = potentials(bx, by, d, self.goal_x, self.field_w)
        return phi_ball, phi_near, self._spread_potential(), self._defense_potential()

    def _defense_potential(self):
        out = np.zeros((self.N, self.P))
        if self.rcfg.w_defense_support == 0 or self.T != 6:
            return out
        for team, sign in ((0, 1.0), (1, -1.0)):
            sel = self.sim.player_team == team
            players = self.sim.player_pos[:, sel].copy()
            ball = self.sim.ball_pos.copy()
            players[..., 0] *= sign
            ball[..., 0] *= sign
            out[:, sel] = defense_support_potential(players, ball, self.goal_x, self.field_h)[:, None]
        # No pagar colocación en saques: están protegidos por reglas distintas.
        out[self.sim.kickoff | (self.setpiece_team >= 0)] = 0.0
        return out

    def _spread_potential(self):
        """Φ de separación: distancia media al compañero más cercano (tope 25% del ancho), en [0,1].
        Premia no amontonarse; por ser de potencial no cambia la política óptima."""
        if self.T < 2:
            return np.zeros((self.N, self.P))
        pp = self.sim.player_pos
        team = self.sim.player_team
        cap = 0.25 * self.field_w
        out = np.zeros((self.N, self.P))
        for t in (0, 1):
            sel = np.where(team == t)[0]
            q = pp[:, sel]                                           # (N, T, 2)
            dd = np.linalg.norm(q[:, :, None] - q[:, None], axis=-1)  # (N, T, T)
            dd[:, np.arange(len(sel)), np.arange(len(sel))] = np.inf
            m = np.minimum(dd.min(axis=2), cap) / cap
            out[:, sel] = m.mean(axis=1, keepdims=True)
        return out

    def _team_ball_dist(self):
        """Distancia a la pelota por agente; con compañeros, la del más cercano de su equipo
        (así el premio de acercarse lo cobra el equipo y no corren todos detrás de la pelota)."""
        sim = self.sim
        d = np.linalg.norm(sim.player_pos - sim.ball_pos[:, None], axis=-1)
        if self.T > 1:
            team = sim.player_team
            for t in (0, 1):
                sel = team == t
                d[:, sel] = d[:, sel].min(axis=1, keepdims=True)
        return d

    def step(self, actions: np.ndarray):
        actions = np.asarray(actions, dtype=np.int64)
        world_act = actions.copy()
        blue = self.sim.player_team == 1
        world_act[:, blue] = MIRROR_ACTION[actions[:, blue]]

        if self.action_delay_max > 0:
            self.act_hist = np.roll(self.act_hist, 1, axis=0)
        self.act_hist[0] = world_act

        # para el premio de saque: quién saca y a qué distancia estaba antes de moverse
        own_ko = self.sim.kickoff[:, None] & (self.sim.kickoff_team[:, None] == self.sim.player_team[None, :])
        d_before = self._team_ball_dist()

        goal = np.zeros(self.N, dtype=np.int64)
        kicked = np.zeros((self.N, self.P), dtype=bool)
        rules = self.rules
        if rules is not None:
            rules.begin_step()
        simplified = self.out_of_bounds and rules is None
        fused = (self.optimize_rollout and rules is None and self.action_delay_max == 0
                 and not (simplified and (self.setpiece_team >= 0).any()))
        if fused:
            goal, kicked = self.sim.step_frames(world_act, self.frame_skip, self.last_touch)
        for k in range(0 if fused else self.frame_skip):
            if self.action_delay_max > 0:
                # acción vigente en este tick: la decisión de hace ceil((d - k) / frame_skip) ventanas
                lag = self.delay - k
                h = np.where(lag <= 0, 0, -(-lag // self.frame_skip))
                world_act = self.act_hist[h, np.arange(self.N)]
            if rules is not None:
                world_act = rules.pre_tick(world_act)
            tick_act = self._setpiece_pre_tick(world_act) if simplified else world_act
            g = self.sim.step(tick_act)
            goal = np.where(goal == 0, g, goal)
            if rules is not None:
                rules.post_tick(world_act, goal)
            elif simplified:
                self._setpiece_post_tick(g)
            kicked |= self.sim.kicked
            tch = self.sim.touch
            if tch.any():
                for t in (0, 1):
                    self.last_touch[tch[:, self.sim.player_team == t].any(axis=1)] = t
        self.ticks += self.frame_skip

        rc = self.rcfg
        team_sign = self.sign[None, :]                  # +1 rojo, -1 azul
        rew = rc.goal * goal[:, None] * team_sign       # gol del rojo = +1 rojo / -1 azul

        # shaping por potencial (sólo si no hubo gol: el estado terminal tiene Φ = 0)
        phi_ball, phi_near, phi_spread, phi_defense = self._potentials()
        phi0_ball, phi0_near, phi0_spread, phi0_defense = self._phi
        scored = goal != 0
        g = rc.gamma
        sh = rc.w_ball_progress * (np.where(scored[:, None], 0.0, g * phi_ball) - phi0_ball)
        sh += rc.w_near_ball * (np.where(scored[:, None], 0.0, g * phi_near) - phi0_near)
        sh += rc.w_spread * (np.where(scored[:, None], 0.0, g * phi_spread) - phi0_spread)
        # bonus: patada que manda la pelota hacia el arco rival
        bvx_own = self.sim.ball_vel[:, None, 0] * team_sign
        sh += rc.kick_to_goal * (kicked & (bvx_own > 1.0))
        rew = rew + rc.shaping_coef * sh
        defense_delta = np.where(scored[:, None], 0.0, g * phi_defense) - phi0_defense
        rew += rc.w_defense_support * max(rc.shaping_coef, rc.defense_shaping_floor) * defense_delta

        # saque propio: premio por progreso hacia la pelota. No es de potencial a propósito: si el saque
        # se corta por tiempo no hay "reembolso", así que trabar queda neto negativo frente a sacar.
        if rc.kickoff_approach > 0:
            d_after = self._team_ball_dist()
            rew = rew + rc.kickoff_approach * own_ko * (d_before - d_after) / self.sim.st.spawn_distance

        self.kickoff_ticks = np.where(self.sim.kickoff, self.kickoff_ticks + self.frame_skip, 0)
        stall = np.zeros(self.N, dtype=bool)
        if self.kickoff_timeout > 0:
            stall = self.sim.kickoff & (self.kickoff_ticks >= self.kickoff_limit)
            staller = self.sim.kickoff_team[:, None] == self.sim.player_team[None, :]
            rew = rew - rc.kickoff_stall * (stall[:, None] & staller)

        out = np.zeros(self.N, dtype=bool)
        if rules is not None:
            # las salidas las cobra el script (env/pegeche.py); pierde la posesión el que tocó último
            out = rules.ev["out"] > 0
            loser = (rules.out_loser[:, None] == self.sim.player_team[None, :]) & out[:, None]
            rew = rew - rc.out_penalty * loser
        elif self.out_of_bounds:
            b = self.sim.ball_pos
            r = self.sim.st.ball["radius"]
            W, H, GH = self.field_w, self.field_h, self.sim.st.goal_half_height
            # misma regla que ref.js: la pelota cruzó ENTERA la línea (centro + radio)
            side = (np.abs(b[:, 1]) > H + r) & (np.abs(b[:, 0]) < W)
            end = (np.abs(b[:, 0]) > W + r) & (np.abs(b[:, 1]) > GH)
            # anti-traba: casi quieta y pegada a una línea (esquinas, banderín) => se cobra salida
            near = (np.abs(b[:, 1]) > H - r - 12) | ((np.abs(b[:, 0]) > W - r - 12) & (np.abs(b[:, 1]) > GH))
            slow = np.linalg.norm(self.sim.ball_vel, axis=1) < 0.3
            self.stuck_ticks = np.where(near & slow & ~scored & (self.setpiece_team < 0),
                                        self.stuck_ticks + self.frame_skip, 0)
            stuck = self.stuck_ticks >= self.stuck_limit
            out = ~scored & (side | end | stuck)
            loser = (self.last_touch[:, None] == self.sim.player_team[None, :]) & out[:, None]
            rew = rew - rc.out_penalty * loser
            if out.any():
                self._set_piece(np.where(out)[0])

        timeout = self.ticks >= self.max_ticks
        done = scored | timeout | stall
        truncated = timeout & ~scored & ~stall
        self.score[goal == 1, 0] += 1
        self.score[goal == -1, 1] += 1

        final_obs = self.observe()
        idx = np.where(done)[0]
        if len(idx):
            # recibe el saque el equipo que recibió el gol (como en HaxBall)
            # tras un gol saca el que lo recibió (como en HaxBall); si no, al azar
            kt = np.where(goal[idx] == 1, 1, np.where(goal[idx] == -1, 0, self.rng.integers(0, 2, len(idx))))
            self._reset_envs(idx, kickoff_team=kt)
        # Sin reset/set-piece, ya tenemos exactamente estos potenciales.
        self._phi = (self._potentials() if len(idx) or out.any() or not self.optimize_rollout
                     else (phi_ball, phi_near, phi_spread, phi_defense))
        if len(idx) and self.optimize_rollout:
            obs = final_obs.copy()
            obs[idx] = self.observe(idx)
        else:
            obs = self.observe() if len(idx) else final_obs
        info = {"goal": goal, "truncated": truncated, "stall": stall, "out": out,
                "ps_kicked": self.sim.ps_kicked.copy(), "final_obs": final_obs, "kicked": kicked}
        if rules is not None:
            info["rules"] = {k: v.copy() for k, v in rules.ev.items()}
        return obs, rew.astype(np.float32), done, info
