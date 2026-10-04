"""Ejecutor de partidos RS4-Z con controladores intercambiables y métricas comunes.

Un controlador implementa `act(env, ctrl, out)` (acciones en el marco propio para los jugadores
`ctrl`), `sync(env, rows)` (tras reinicios/colocaciones) y `push(env)` (después de cada paso).
"""
from __future__ import annotations

import numpy as np

from env.rs4z import contract as C
from env.rs4z import kernel as K
from env.rs4z.core import RS4ZEnv

from .metrics import Metrics

RED = np.array([True, True, True, True, False, False, False, False])
BLUE = ~RED


def env_sample(env, ev, metrics):
    """Agregar una decisión del entorno a `metrics`."""
    toucher = np.where(ev["touched"].any(axis=1), env.ri[:, K.RI_LAST_P], -1)
    in_play = (env.ri[:, K.RI_KO] == 0) & (env.ri[:, K.RI_TEAM] < 0)
    started = ev["restart_start"] > 0
    metrics.add(env.ball_pos, env.player_pos, env.player_vel, env.active, toucher, ev["kicked"], ev["goal"],
                env.ri[:, K.RI_KIND], started, in_play)


def play(red, blue, n_envs=32, match_ticks=10800, seed=0, active=None, deadline=0,
         kickoff_deadline=C.KICKOFF_SAFETY, max_delay=12, delay=None, kick_strength=None, ball_radius=None,
         contract="v2", red_ctrl=None, blue_ctrl=None, env=None, starts=None):
    """Juega n_envs partidos simultáneos de `match_ticks` ticks de reloj; devuelve resultados y métricas.

    red/blue: controladores. red_ctrl/blue_ctrl: máscaras (8,) de los lugares que controla cada uno
    (por defecto, su equipo). starts: función opcional starts(env, rows) que coloca estados iniciales.
    """
    if env is None:
        env = RS4ZEnv(n_envs, contract=contract, seed=seed, deadline=deadline, kickoff_deadline=kickoff_deadline,
                      max_delay=max_delay)
    rows = np.arange(env.N)
    rng = np.random.default_rng(seed)
    env.start_match(rows, active=active, kickoff_team=rng.integers(0, 2, env.N), match_ticks=match_ticks,
                    delay=delay, kick_strength=kick_strength, ball_radius=ball_radius)
    if starts is not None:
        starts(env, rows)
    red_ctrl = RED if red_ctrl is None else np.asarray(red_ctrl, dtype=bool)
    blue_ctrl = BLUE if blue_ctrl is None else np.asarray(blue_ctrl, dtype=bool)
    ctrl_r = np.broadcast_to(red_ctrl, (env.N, 8)) & env.active
    ctrl_b = np.broadcast_to(blue_ctrl, (env.N, 8)) & env.active
    for c in {id(red): red, id(blue): blue}.values():
        c.sync(env, rows)
    metrics = Metrics(env.N)
    finished = np.zeros(env.N, dtype=bool)
    final = np.zeros((env.N, 2), dtype=np.int64)
    events = dict(forfeits=np.zeros(env.N), safety=np.zeros(env.N), kickoff_safety=np.zeros(env.N))
    out = np.zeros((env.N, 8), dtype=np.int64)
    steps = 0
    while not finished.all():
        out[:] = 0
        red.act(env, ctrl_r, out)
        blue.act(env, ctrl_b, out)
        ev = env.step(out)
        steps += 1
        for c in {id(red): red, id(blue): blue}.values():
            c.push(env)
        live = ~finished
        env_sample(env, ev, metrics)
        events["forfeits"] += (ev["forfeit"] >= 0) & live
        events["safety"] += (ev["safety"] == 1) & live
        events["kickoff_safety"] += (ev["safety"] == 2) & live
        reset = np.flatnonzero((ev["goal"] != 0) | (ev["safety"] == 2))
        if len(reset):
            for c in {id(red): red, id(blue): blue}.values():
                c.sync(env, reset)
        end = ev["match_end"] & live
        final[end] = env.score[end]
        finished |= end
        if steps > 10 * match_ticks:
            raise RuntimeError("los partidos no terminan (reloj congelado sin saque inicial)")
    return dict(score=final, metrics=metrics, events=events, env=env)


def points(score):
    """Puntos del rojo por partido: 1 ganar, 0,5 empatar, 0 perder."""
    return (score[:, 0] > score[:, 1]) + 0.5 * (score[:, 0] == score[:, 1])
