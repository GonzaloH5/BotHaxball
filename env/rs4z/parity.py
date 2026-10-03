"""Paridad tick a tick entre RS4-Z en modo v1 y el oráculo `HaxballEnv(referee="rs_one_v1")`.

Ambos parten del mismo estado y reciben las mismas acciones (marco propio de cada jugador) con
`frame_skip=1`. Los sorteos de último toque desconocido se fijan a "rojo" en los dos. Se usa en
`tests/rs4z/test_rs4z_referee_parity.py` y en `tools/rs4z_parity.py` (corrida larga).
"""
from __future__ import annotations

import numpy as np

from .core import RS4ZEnv
from . import kernel as K


class _ZeroBitRNG:
    """Generador del oráculo: `integers(0, 2)` sin tamaño devuelve 0 (rojo); el resto delega."""

    def __init__(self, seed):
        self._g = np.random.default_rng(seed)

    def integers(self, low, high=None, size=None, *args, **kwargs):
        if size is None:
            return 0
        return self._g.integers(low, high, size, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._g, name)


class _OnesRNG:
    def __init__(self, seed):
        self._g = np.random.default_rng(seed)

    def random(self, n=None):
        return np.ones(n) if n is not None else 1.0

    def __getattr__(self, name):
        return getattr(self._g, name)


def make_pair(n_envs, seed=0):
    from env.haxball_env import HaxballEnv
    old = HaxballEnv(n_envs, 4, "rs_one", frame_skip=1, max_ticks=10 ** 9, random_reset_prob=0.0, seed=seed,
                     out_of_bounds=True, obs_layout="universal", max_entities=7, rules=None,
                     kickoff_timeout=0, referee="rs_one_v1")
    old.rng = _ZeroBitRNG(seed)
    old.reset()
    new = RS4ZEnv(n_envs, contract="v1", frame_skip=1, seed=seed, deadline=0, kickoff_deadline=0, max_delay=0)
    new.rng = _OnesRNG(seed)
    new.spawn_rank[:] = np.array([0, 1, 2, 3, 0, 1, 2, 3])  # orden fijo de spawn del oráculo
    copy_state(old, new)
    return old, new


def _old_index(new, old):
    """Índices de discos del oráculo para cada disco RS4-Z que existe en ambos."""
    D = old.sim.st.d_pos.shape[0]
    new_idx = np.concatenate([[0], 4 + np.arange(D), new.fp + np.arange(8)])
    old_idx = np.concatenate([[0], 1 + np.arange(D), old.sim.first_player + np.arange(8)])
    return new_idx, old_idx


def copy_state(old, new):
    s = old.sim
    ni, oi = _old_index(new, old)
    new.pos[:, ni] = s.pos[:, oi]
    new.vel[:, ni] = s.vel[:, oi]
    new.mask[:, ni] = s.mask[:, oi]
    new.inv[:, ni] = s.inv_env[:, oi]
    new.kick_cancel[:] = s.kick_cancel
    new.grav[:] = s.ball_grav
    ref = old._rs1
    new.ri[:, K.RI_TEAM] = old.setpiece_team
    new.ri[:, K.RI_KIND] = old.setpiece_kind
    new.ri[:, K.RI_TICKS] = old.setpiece_ticks
    new.rf[:, K.RF_SPOT_X] = old.setpiece_pos[:, 0]
    new.rf[:, K.RF_SPOT_Y] = old.setpiece_pos[:, 1]
    new.ri[:, K.RI_BOOST] = ref.boost_left
    new.rf[:, K.RF_BOOST] = ref.boost_factor
    new.ri[:, K.RI_GRAV] = ref.gravity_left
    new.ri[:, K.RI_ARMED] = ref.armed
    new.ri[:, K.RI_LAST] = old.last_touch
    new.ri[:, K.RI_KO] = s.kickoff
    new.ri[:, K.RI_KO_TEAM] = s.kickoff_team
    new.outside[:] = ref.outside


def compare(old, new):
    """Máxima diferencia de estado continuo y lista de discrepancias discretas."""
    s = old.sim
    ni, oi = _old_index(new, old)
    cont = max(float(np.abs(new.pos[:, ni] - s.pos[:, oi]).max()),
               float(np.abs(new.vel[:, ni] - s.vel[:, oi]).max()),
               float(np.abs(new.inv[:, ni] - s.inv_env[:, oi]).max()),
               float(np.abs(new.grav - s.ball_grav).max()))
    ref = old._rs1
    discrete = {
        "setpiece_team": new.ri[:, K.RI_TEAM] != old.setpiece_team,
        "setpiece_kind": new.ri[:, K.RI_KIND] != old.setpiece_kind,
        "setpiece_ticks": (new.ri[:, K.RI_TICKS] != old.setpiece_ticks) & (old.setpiece_team >= 0),
        "last_touch": new.ri[:, K.RI_LAST] != old.last_touch,
        "kickoff": (new.ri[:, K.RI_KO] != 0) != s.kickoff,
        "kick_cancel": (new.kick_cancel != s.kick_cancel).any(axis=1),
        "mask": (new.mask[:, ni] != s.mask[:, oi]).any(axis=1),
        "boost": new.ri[:, K.RI_BOOST] != ref.boost_left,
        "gravity_left": new.ri[:, K.RI_GRAV] != ref.gravity_left,
    }
    return cont, {k: int(v.sum()) for k, v in discrete.items() if v.any()}


def chase_actions(env_new, rng, p_random=0.35, p_kick=0.6):
    """Acciones mixtas: ir hacia la pelota y patear cerca (genera salidas, saques y goles) + azar."""
    from sim.physics import MOVE_UNIT
    N = env_new.N
    bp = env_new.ball_pos
    pp = env_new.player_pos
    d = bp[:, None, :] - pp
    own = d * np.array([1.0, 1.0])
    own[..., 0] *= env_new.sign[None, :]
    dist = np.hypot(d[..., 0], d[..., 1])
    unit = own / np.maximum(dist[..., None], 1e-9)
    move = np.argmax(unit @ MOVE_UNIT.T, axis=-1)
    kick = (dist < 40) & (rng.random((N, 8)) < p_kick)
    act = move + 9 * kick
    rand = rng.random((N, 8)) < p_random
    act[rand] = rng.integers(0, 18, int(rand.sum()))
    return act.astype(np.int64)


def run(n_envs=64, ticks=1500, seed=0, tol=1e-9, resync=True):
    """Paridad del árbitro v1.

    resync=True: antes de cada tick se copia el estado del oráculo (error de 1 tick, sin acumular).
    resync=False: trayectorias libres; las diferencias de redondeo entre numpy (oráculo) y numba
    (~1e-17 en la emulación de barreras) crecen por el caos de las colisiones, así que se informa
    hasta dónde coinciden los eventos discretos.
    """
    old, new = make_pair(n_envs, seed)
    rng = np.random.default_rng(seed + 1)
    worst = 0.0
    events = dict(goals=0, outs=0, restarts={1: 0, 2: 0, 3: 0})
    for t in range(ticks):
        if resync:
            copy_state(old, new)
        act = chase_actions(new, rng)
        ev = new.step(act)
        old.step(act)
        cont, disc = compare(old, new)
        worst = max(worst, cont)
        events["goals"] += int((ev["goal"] != 0).sum())
        events["outs"] += int(ev["out"].sum())
        for k in (1, 2, 3):
            events["restarts"][k] += int((ev["restart_start"] == k).sum())
        if cont > tol or disc:
            return dict(ok=False, tick=t, worst=cont, discrete=disc, events=events)
    return dict(ok=True, ticks=ticks, worst=worst, events=events)
