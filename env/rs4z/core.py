"""Entorno vectorizado RS4-Z (Real Soccer ONE, contrato RS4-Z-2).

Siempre 8 lugares de jugador (0..3 rojos, 4..7 azules) con un plantel `active` por partido:
1v0, 2v1, 4v3, 4v4… conviven en el mismo lote. Las acciones entran en el marco propio de cada
jugador (x espejado para azul, como la política) y se convierten al mundo acá.

Semántica (ver `reports/rs4z/auditoria_pre_entrenamiento.md`):
* Un gol no termina el episodio: repone el saque inicial (saca el que recibió) y sigue.
* El fin de partido (`EV_MATCH_END`) es terminal real; el que llama decide el reinicio.
* Plazo de entrenamiento `deadline` (600): el saque vencido pasa al rival. Evaluación: 0.
* Reloj congelado durante el saque inicial (F2). Masa 0,5 tras reposicionar (F1).
* Latencia por jugador en ticks (`delay`), con historial de decisiones.
"""
from __future__ import annotations

import math

import numpy as np

from sim.physics import MOVE_UNIT
from sim.stadium import BALL, BLUE, PLAYER_MASK, RED, load_stadium

from . import contract as C
from . import kernel as K

MIRROR_MOVE = np.array([0, 1, 8, 7, 6, 5, 4, 3, 2])
MIRROR_ACTION = np.concatenate([MIRROR_MOVE, MIRROR_MOVE + 9])
TEAM = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
SIGN = np.where(TEAM == 0, 1.0, -1.0)
P = 8


class RS4ZEnv:
    def __init__(self, n_envs: int, *, contract: str = "v2_lateral", frame_skip: int = 3, seed: int | None = None,
                 deadline: int = C.TRAINING_DEADLINE, kickoff_deadline: int = C.TRAINING_DEADLINE,
                 max_delay: int = 12, stadium: str | None = None, map: str = "rs_one"):
        if contract not in C.FLAGS:
            raise ValueError(f"contrato desconocido: {contract}")
        self.N = int(n_envs)
        self.contract = contract
        # el script propio de un mapa (p. ej. Sanguchito) se suma al contrato pedido
        self.flags = C.FLAGS[contract] | C.MAPS[map].get("flags", 0)
        self.frame_skip = int(frame_skip)
        self.deadline = int(deadline)
        self.kickoff_deadline = int(kickoff_deadline)
        self.rng = np.random.default_rng(seed)
        self.map = map
        st = self.st = load_stadium(stadium or C.MAPS[map]["stadium"])
        self.prm = C.params(map)
        D = st.d_pos.shape[0]
        self.fp = fp = 4 + D
        self.K = Kd = fp + P
        pl, b = st.player, st.ball
        self.team = TEAM.copy()
        self.sign = SIGN.copy()
        # propiedades por disco comunes a todos los partidos
        sd_group = np.full(3, BALL, dtype=np.int64)
        sd_mask = np.array([RED, BLUE, RED | BLUE], dtype=np.int64)
        self.bcoef = np.concatenate([[b["bCoef"]], np.full(3, self.prm[C.PI["script_disc_bcoef"]]), st.d_bcoef,
                                     np.full(P, pl["bCoef"])])
        self.damping = np.concatenate([[b["damping"]], np.full(3, 0.99), st.d_damping, np.full(P, pl["damping"])])
        team_group = np.where(TEAM == 0, RED, BLUE)
        self.base_group = np.concatenate([[b["cGroup"]], sd_group, st.d_group, team_group]).astype(np.int64)
        self.base_mask = np.concatenate([[b["cMask"]], sd_mask, st.d_mask, np.full(P, PLAYER_MASK)]).astype(np.int64)
        self.base_radius = np.concatenate([[b["radius"]], np.zeros(3), st.d_radius, np.full(P, pl["radius"])])
        self.base_inv = np.concatenate([[b["invMass"]], np.zeros(3), st.d_invmass,
                                        np.full(P, self.prm[C.PI["map_inv_mass"]])])
        self.sd_home = np.array(C.MAPS[map].get("sd_home", C.SD_HOME), dtype=np.float64)
        self.d_pos = st.d_pos.copy()
        self.p_acc = float(pl["acceleration"])
        self.p_kacc = float(pl["kickingAcceleration"])
        self.p_kdamp = float(pl["kickingDamping"])
        self.p_kback = float(pl["kickback"])
        self.line_w = float(C.LINE_W)
        self.goal_hh = float(st.goal_half_height)
        # estado por partido
        N = self.N
        self.pos = np.zeros((N, Kd, 2))
        self.vel = np.zeros((N, Kd, 2))
        self.mask = np.tile(self.base_mask, (N, 1))
        self.group = np.tile(self.base_group, (N, 1))
        self.radius = np.tile(self.base_radius, (N, 1))
        self.inv = np.tile(self.base_inv, (N, 1))
        self.kick_cancel = np.zeros((N, P), dtype=np.bool_)
        self.grav = np.zeros((N, 2))
        self.active = np.ones((N, P), dtype=np.bool_)
        self.spawn_rank = np.tile(np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int64), (N, 1))
        self.ri = np.zeros((N, K.RI_SIZE), dtype=np.int64)
        self.rf = np.zeros((N, K.RF_SIZE))
        self.rf[:, K.RF_KSTR] = pl["kickStrength"]
        self.rf[:, K.RF_BALL_INV] = b["invMass"]
        self.outside = np.zeros((N, P), dtype=np.bool_)
        self.max_delay = int(max_delay)
        self.H = max(3, 1 + -(-self.max_delay // self.frame_skip))  # ≥3: la obs v2 lleva 3 decisiones
        self.act_hist = np.zeros((N, P, self.H), dtype=np.int64)
        self.delay = np.zeros((N, P), dtype=np.int64)
        self.ev = np.zeros((N, K.EV_SIZE), dtype=np.int64)
        self.ev_kicked = np.zeros((N, P), dtype=np.bool_)
        self.ev_touch = np.zeros((N, P), dtype=np.bool_)
        self._s_act = np.zeros((N, P), dtype=np.int64)
        self._s_kick = np.zeros((N, P), dtype=np.bool_)
        self._s_t4 = np.zeros((N, P), dtype=np.bool_)
        self._s_contact = np.zeros((N, P), dtype=np.bool_)
        self._s_kinfo = np.zeros((N, P, 3))
        self.episode = np.zeros(N, dtype=np.int64)
        self.start_match(np.arange(N))

    # ------------------------------------------------------------------ vistas
    @property
    def ball_pos(self):
        return self.pos[:, 0]

    @property
    def ball_vel(self):
        return self.vel[:, 0]

    @property
    def player_pos(self):
        return self.pos[:, self.fp:]

    @property
    def player_vel(self):
        return self.vel[:, self.fp:]

    @property
    def kickoff(self):
        return self.ri[:, K.RI_KO] != 0

    @property
    def restart_team(self):
        return self.ri[:, K.RI_TEAM]

    @property
    def restart_kind(self):
        return self.ri[:, K.RI_KIND]

    @property
    def score(self):
        return self.ri[:, [K.RI_SCORE0, K.RI_SCORE1]]

    @property
    def clock(self):
        return self.ri[:, K.RI_CLOCK]

    # ------------------------------------------------------------------ partidos
    def start_match(self, rows, *, active=None, kickoff_team=None, match_ticks=None, kick_strength=None,
                    ball_radius=None, delay=None, spawn_rank=None):
        """Partido nuevo en `rows`: plantel, variante del mapa, latencia, reloj y saque inicial.

        active: (len(rows), 8) bool; por defecto 4v4. spawn_rank: orden de spawn dentro de cada equipo
        (por defecto aleatorio entre los activos, F12). match_ticks: duración en ticks de reloj.
        """
        rows = np.atleast_1d(np.asarray(rows, dtype=np.int64))
        n = len(rows)
        if n == 0:
            return
        act = np.ones((n, P), dtype=bool) if active is None else np.asarray(active, dtype=bool).reshape(n, P)
        self.active[rows] = act
        if spawn_rank is None:
            rank = np.zeros((n, P), dtype=np.int64)
            for i in range(n):
                for t in (0, 1):
                    idx = np.flatnonzero(act[i] & (TEAM == t))
                    rank[i, idx] = self.rng.permutation(len(idx))
        else:
            rank = np.asarray(spawn_rank, dtype=np.int64).reshape(n, P)
        self.spawn_rank[rows] = rank
        self.rf[rows, K.RF_KSTR] = self.st.player["kickStrength"] if kick_strength is None else kick_strength
        br = self.st.ball["radius"] if ball_radius is None else ball_radius
        self.radius[rows, 0] = br
        self.delay[rows] = 0 if delay is None else np.asarray(delay, dtype=np.int64).reshape(n, P)
        self.act_hist[rows] = 0
        self.ri[rows] = 0
        self.ri[rows, K.RI_LEN] = 10 ** 12 if match_ticks is None else match_ticks
        kt = (self.rng.integers(0, 2, n) if kickoff_team is None
              else np.broadcast_to(np.asarray(kickoff_team, dtype=np.int64), (n,)))
        self.episode[rows] += 1
        self.reset_kickoff(rows, kt)

    def reset_kickoff(self, rows, kickoff_team):
        rows = np.atleast_1d(np.asarray(rows, dtype=np.int64))
        kt = np.broadcast_to(np.asarray(kickoff_team, dtype=np.int64), rows.shape)
        mass = self.prm[C.PI["map_inv_mass"]] if self.flags & C.FIX_MASS else self.prm[C.PI["play_inv_mass"]]
        for i, n in enumerate(rows):
            K.reset_kickoff(int(kt[i]), self.pos[n], self.vel[n], self.mask[n], self.group[n], self.radius[n],
                            self.inv[n], self.kick_cancel[n], self.grav[n], self.active[n], self.team,
                            self.spawn_rank[n], self.fp, P, self.prm, self.flags, self.sd_home, self.base_mask,
                            self.d_pos, self.ri[n], self.rf[n], self.outside[n], C.SPAWN_X, C.SPAWN_DY, mass)

    def start_restart(self, row, kind, taker, spot):
        """Iniciar un saque como lo hace el script (para ejercicios y estados grabados)."""
        n = int(row)
        self.ri[n, K.RI_KO] = 0
        for p in range(P):
            self.mask[n, self.fp + p] = PLAYER_MASK if self.active[n, p] else 0
        K.start_piece(int(kind), int(taker), float(spot[0]), float(spot[1]), self.pos[n], self.vel[n],
                      self.group[n], self.radius[n], self.inv[n], self.kick_cancel[n], self.grav[n],
                      self.active[n], self.team, self.fp, P, self.prm, self.flags, self.sd_home,
                      self.ri[n], self.rf[n], self.outside[n])
        if not self.flags & C.FIX_ENGINE:
            K.protect_v1(self.pos[n], self.vel[n], self.fp, P, self.active[n], self.team, self.prm,
                         self.ri[n], self.rf[n], self.outside[n])

    def place(self, row, *, ball_pos, ball_vel, player_pos, player_vel, kick_held=None, last_touch=-1,
              mass_phase=1):
        """Colocar un estado de juego abierto (sin saque) en el partido `row`.

        Jugadores inactivos se ignoran. mass_phase: 0 masa del mapa (0,5), 1 del script (0,3).
        """
        n = int(row)
        fp = self.fp
        self.ri[n, K.RI_KO] = 0
        self.ri[n, K.RI_TEAM] = -1
        self.ri[n, K.RI_KIND] = 0
        self.ri[n, K.RI_TICKS] = 0
        self.ri[n, K.RI_BOOST] = 0
        self.ri[n, K.RI_GRAV] = 0
        self.ri[n, K.RI_ARMED] = 1
        self.ri[n, K.RI_LAST] = last_touch
        self.ri[n, K.RI_LAST_P] = -1
        self.grav[n] = 0.0
        self.outside[n] = False
        K.clear_piece(self.pos[n], self.radius[n], self.group[n], fp, P, self.active[n], self.team,
                      self.sd_home, self.flags if self.flags & C.FIX_ENGINE else C.FIX_ENGINE)
        self.pos[n, 0] = ball_pos
        self.vel[n, 0] = ball_vel
        pp = np.asarray(player_pos, dtype=np.float64)
        pv = np.asarray(player_vel, dtype=np.float64)
        for p in range(P):
            k = fp + p
            if self.active[n, p]:
                self.pos[n, k] = pp[p]
                self.vel[n, k] = pv[p]
                self.mask[n, k] = PLAYER_MASK
            else:
                self.pos[n, k] = (0.0, 5000.0 + 40.0 * p)
                self.vel[n, k] = 0.0
                self.mask[n, k] = 0
        self.kick_cancel[n] = False if kick_held is None else np.asarray(kick_held, dtype=bool)
        phase = int(mass_phase) if self.flags & C.FIX_MASS else 1
        self.ri[n, K.RI_MASS] = phase
        mass = self.prm[C.PI["play_inv_mass"]] if phase else self.prm[C.PI["map_inv_mass"]]
        self.inv[n, fp:] = mass
        self.inv[n, 0] = self.rf[n, K.RF_BALL_INV]

    # ------------------------------------------------------------------ paso
    def step(self, actions):
        """actions: (N, 8) enteros 0..17 en el marco propio. Devuelve los eventos de la decisión."""
        own = np.asarray(actions, dtype=np.int64)
        world = own.copy()
        world[:, 4:] = MIRROR_ACTION[own[:, 4:]]
        world[~self.active] = 0
        if self.H > 1:
            self.act_hist[:, :, 1:] = self.act_hist[:, :, :-1]
        self.act_hist[:, :, 0] = world
        rand = self.rng.random(self.N)
        st = self.st
        K.decision_step(self.pos, self.vel, self.mask, self.group, self.radius, self.inv, self.bcoef, self.damping,
                        self.kick_cancel, self.grav, self.active, self.team, self.spawn_rank, self.ri, self.rf,
                        self.outside, self.act_hist, self.delay, rand,
                        self.fp, self.frame_skip, self.prm, self.flags, self.deadline, self.kickoff_deadline,
                        self.sd_home, self.base_mask, self.d_pos, MOVE_UNIT, self.p_acc, self.p_kacc,
                        self.p_kdamp, self.p_kback, self.line_w, self.goal_hh, C.SPAWN_X, C.SPAWN_DY,
                        st.v_pos, st.v_bcoef, st.v_group, st.v_mask,
                        st.s_p0, st.s_p1, st.s_curved, st.s_center, st.s_radius, st.s_t0, st.s_t1, st.s_bias,
                        st.s_bcoef, st.s_group, st.s_mask,
                        st.p_normal, st.p_dist, st.p_bcoef, st.p_group, st.p_mask, st.g_p0, st.g_p1, st.g_team,
                        self.ev, self.ev_kicked, self.ev_touch, self._s_act, self._s_kick, self._s_t4,
                        self._s_contact, self._s_kinfo)
        return self.events()

    def events(self):
        ev = self.ev
        return dict(goal=ev[:, K.EV_GOAL].copy(), out=ev[:, K.EV_OUT] != 0, restart_start=ev[:, K.EV_START].copy(),
                    restart_exec=ev[:, K.EV_EXEC].copy(), forfeit=ev[:, K.EV_FORFEIT].copy(),
                    kickoff_taken=ev[:, K.EV_KO_TAKEN] != 0, match_end=ev[:, K.EV_MATCH_END] != 0,
                    ticks=ev[:, K.EV_TICKS].copy(), safety=ev[:, K.EV_SAFETY].copy(),
                    kicked=self.ev_kicked.copy(), touched=self.ev_touch.copy())

    # ------------------------------------------------------------------ utilidades
    def snapshot(self):
        names = ("pos", "vel", "mask", "group", "radius", "inv", "kick_cancel", "grav", "active", "spawn_rank",
                 "ri", "rf", "outside", "act_hist", "delay")
        return {k: getattr(self, k).copy() for k in names}

    def restore(self, snap):
        for k, v in snap.items():
            getattr(self, k)[...] = v
