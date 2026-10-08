"""Entorno vectorizado HX11: N partidos 11v11 en el estadio actual de HaxEleven con el árbitro del contrato HX11-1.

    env = HX11Env(256)
    env.reset()
    events = env.step(actions, frames=3)   # actions (N, 22) en 0..17, como sim.physics

Jugadores 0..10 rojos (0 = arquero) y 11..21 azules (11 = arquero). Estado legible en `ri`/`rf` (índices de
`env.hx11.kernel`). Las observaciones y la latencia de sala son de la etapa siguiente (P, docs/PLAN_HAXELEVEN.md).
"""
from __future__ import annotations

import numpy as np

from sim.physics import MOVE_UNIT, BatchSim
from sim.stadium import BLUEKO, PLAYER_MASK, REDKO, load_stadium

from . import contract as C
from . import kernel as K


class HX11Env:
    def __init__(self, n_envs: int, n_red: int = 11, n_blue: int = 11, match_ticks: int = 0, seed: int | None = None,
                 offside: bool = True, fouls: bool = True, powershot: bool = True):
        st = load_stadium(C.STADIUM)
        self.st = st
        self.N, self.n_red, self.P = n_envs, n_red, n_red + n_blue
        ref = BatchSim(1, n_red, n_blue, st)  # mismas propiedades por disco que el simulador base
        self.fp = ref.first_player
        self.Kd = ref.K
        self.spawn = ref.spawn_positions()
        tile = lambda a: np.tile(a, (n_envs, 1))
        self.base_radius = ref.radius.astype(np.float64)
        self.base_inv = ref.invmass.astype(np.float64)
        self.base_group = ref.group.astype(np.int64)
        self.radius = tile(self.base_radius)
        self.inv = tile(self.base_inv)
        self.group = tile(self.base_group)
        self.base_mask = ref.base_mask.astype(np.int64)
        self.mask = tile(self.base_mask)
        self.bcoef = ref.bcoef.astype(np.float64)
        self.damping = ref.damping.astype(np.float64)
        self.home = st.d_pos[:2].astype(np.float64).copy()  # reposo de la barrera y del disco del punto
        self.pos = np.zeros((n_envs, self.Kd, 2))
        self.vel = np.zeros((n_envs, self.Kd, 2))
        self.grav = np.zeros((n_envs, 2))
        self.kick_cancel = np.zeros((n_envs, self.P), np.bool_)
        self.ri = np.zeros((n_envs, K.RI_SIZE), np.int64)
        self.rf = np.zeros((n_envs, K.RF_SIZE))
        self.fs = np.full((n_envs, self.P), -1, np.int64)
        self.fl = np.full((n_envs, self.P), -1, np.int64)
        t = C.load_tables()
        self.bar, self.hold, self.hold_ticks, self.ramp = t["bar"], t["hold"], t["hold_ticks"], t["ramp"]
        self.match_ticks = match_ticks
        self.modes = (int(offside), int(fouls), int(powershot))
        self.rng = np.random.default_rng(seed)
        self.reset()

    # ------------------------------------------------------------------ reset
    def reset(self, idx=None, kickoff_team=0):
        """Saque inicial (como después de un gol): posiciones de los puntos del mapa, arqueros a (±1760, 0)."""
        idx = np.arange(self.N) if idx is None else np.atleast_1d(np.asarray(idx))
        fp, nr = self.fp, self.n_red
        self.pos[idx] = 0.0
        self.vel[idx] = 0.0
        self.pos[idx, 1:fp] = self.st.d_pos
        self.pos[idx, fp:] = self.spawn
        self.pos[idx, fp] = (-C.GK_KICKOFF[0], C.GK_KICKOFF[1])
        self.pos[idx, fp + nr] = (C.GK_KICKOFF[0], C.GK_KICKOFF[1])
        self.radius[idx] = self.base_radius
        self.radius[idx, 1:3] = 0.0       # barrera y disco del punto fuera de juego
        self.inv[idx] = self.base_inv
        self.inv[idx, 0] = C.BASE_BALL_INV
        self.inv[idx, fp:] = C.PLAYER_INV
        self.group[idx] = self.base_group
        self.mask[idx] = self.base_mask
        ko = REDKO if kickoff_team == 0 else BLUEKO
        self.mask[idx, fp:] = PLAYER_MASK | ko
        self.grav[idx] = 0.0
        self.kick_cancel[idx] = False
        self.ri[idx] = 0
        self.rf[idx] = 0.0
        self.fs[idx] = -1
        self.fl[idx] = -1
        r = self.ri
        r[idx, K.RI_MODE] = K.KICKOFF
        r[idx, K.RI_KO] = 1
        r[idx, K.RI_KO_TEAM] = kickoff_team
        r[idx, K.RI_LAST] = -1
        r[idx, K.RI_PS_T] = -1
        r[idx, K.RI_AFTER_T] = -1
        r[idx, K.RI_LAT_ARMED] = 1
        r[idx, K.RI_PREV_IN_Y] = 1
        r[idx, K.RI_OFFSIDE_ON], r[idx, K.RI_FOULS_ON], r[idx, K.RI_PS_ON] = self.modes
        r[idx, K.RI_LEN] = self.match_ticks

    # ------------------------------------------------------------------ paso
    def step(self, actions, frames: int = 1) -> np.ndarray:
        """Avanza `frames` ticks con las acciones (N, P). Devuelve los eventos (N, frames) (EV_* del kernel)."""
        st, pl = self.st, self.st.player
        events = np.zeros((self.N, frames), np.int64)
        K.step_envs(self.pos, self.vel, self.mask, self.group, self.radius, self.inv, self.grav, self.kick_cancel,
                    self.ri, self.rf, self.fs, self.fl, np.ascontiguousarray(actions, dtype=np.int64), frames,
                    MOVE_UNIT, self.fp, self.n_red, self.bcoef, self.damping,
                    float(pl["acceleration"]), float(pl["kickingAcceleration"]), float(pl["kickingDamping"]),
                    float(pl["kickStrength"]), float(pl["kickback"]),
                    st.v_pos, st.v_bcoef, st.v_group, st.v_mask,
                    st.s_p0, st.s_p1, st.s_curved, st.s_center, st.s_radius, st.s_t0, st.s_t1, st.s_bias,
                    st.s_bcoef, st.s_group, st.s_mask, st.p_normal, st.p_dist, st.p_bcoef, st.p_group, st.p_mask,
                    st.g_p0, st.g_p1, st.g_team, self.home, self.spawn, self.bar, self.hold, self.hold_ticks,
                    self.ramp, events)
        return events

    # ------------------------------------------------------------------ vistas
    @property
    def ball_pos(self):
        return self.pos[:, 0]

    @property
    def player_pos(self):
        return self.pos[:, self.fp:]

    def mode(self):
        return self.ri[:, K.RI_MODE]
