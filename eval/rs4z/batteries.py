"""Baterías de habilidad y partidos de compuerta RS4-Z, reproducibles.

Cada episodio de una batería tiene su semilla (tarea, índice), así que el candidato y la referencia
(RS-Pro L5 jugando en los lugares del aprendiz) enfrentan exactamente los mismos estados iniciales.
Los rivales y compañeros scripted tienen nivel y estilo fijos por batería.
"""
from __future__ import annotations

import hashlib

import numpy as np

from bots.rspro.policy import STYLE_BALANCED, RSPro, sample_style
from env.rs4z.core import RS4ZEnv
from env.rs4z.drills import TASK_NAMES, TASKS, Drills, team_slots

from .metrics import Metrics
from .runner import env_sample

BATTERIES = {
    "control_battery": dict(tasks=("touch", "empty_goal", "carry_goal", "receive_shot"), level=0),
    "duel_battery": dict(tasks=("shot_vs_last", "attack_1v1", "defend_1v1", "loose_ball"), level=3),
    # defensores L5: contra L3 gambetear solo sigue siendo razonable y la medición de "elige pasar" es ambigua
    "passing_battery": dict(tasks=("attack_2v1", "one_two", "through_ball", "redirect", "attack_2v2"), level=5),
    "defense_battery": dict(tasks=("defend_2v2", "defend_4v4", "defend_restart"), level=4),
    "situation_battery": dict(tasks=("situation_attack", "restart_attack", "transition"), level=4),
}


def _seed(*parts):
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "little") & (2 ** 63 - 1)


def run_battery(candidate, name, episodes=256, n_envs=128, seed=0, state_bank=None, latency=0):
    """Tasa de éxito por tarea de `candidate` (controlador con act/sync/push) en la batería `name`.

    Devuelve {tarea: dict(success=…, conceded=…, episodes=…)}.
    """
    spec = BATTERIES[name]
    out = {}
    for task in spec["tasks"]:
        out[task] = _run_task(candidate, task, spec["level"], episodes, n_envs, seed, state_bank, latency)
    return out


def _run_task(candidate, task, level, episodes, n_envs, seed, bank, latency):
    n_envs = min(n_envs, episodes)
    env = RS4ZEnv(n_envs, seed=_seed(seed, task), deadline=600, kickoff_deadline=600)
    drills = Drills(env, np.random.default_rng(0), state_bank=bank)
    bot = RSPro(env, seed=_seed(seed, task, "bot"))
    next_ep = 0
    ep_of_row = np.full(n_envs, -1, dtype=np.int64)
    results = np.full(episodes, np.nan)
    passed_ball = np.zeros(episodes, dtype=bool)      # el episodio tuvo un pase entre aprendices
    last_touch = np.full(n_envs, -1, dtype=np.int64)   # último jugador que tocó la pelota en la fila
    had_pass = np.zeros(n_envs, dtype=bool)

    def start(rows):
        nonlocal next_ep
        for n in rows:
            if next_ep >= episodes:
                ep_of_row[n] = -1          # la fila sigue jugando, pero ya no cuenta
                continue
            drills.rng = np.random.default_rng(_seed(seed, task, next_ep))
            lt = int(drills.rng.integers(0, 2))
            drills.start([n], task, 1.0, learner_team=lt)
            for t in (0, 1):
                bot.configure([n], t, level, sample_style(np.random.default_rng(_seed(seed, task, next_ep, t))))
            bot.sync(env, [n])
            candidate.sync(env, [n])
            env.delay[n] = latency
            ep_of_row[n] = next_ep
            last_touch[n] = -1
            had_pass[n] = False
            next_ep += 1

    start(np.arange(n_envs))
    out = np.zeros((n_envs, 8), dtype=np.int64)
    guard = 0
    while (ep_of_row >= 0).any():
        learner, scripted = drills.controllers()
        live = ep_of_row >= 0
        learner &= live[:, None]
        scripted &= live[:, None]
        out[:] = 0
        bot.act(env, scripted, out)
        candidate.act(env, learner, out)
        ev = env.step(out)
        bot.push(env)
        candidate.push(env)
        # pase: toques consecutivos de dos aprendices distintos sin un toque rival en el medio
        for n in np.flatnonzero(ev["touched"].any(axis=1) & live):
            for q in np.flatnonzero(ev["touched"][n]):
                if learner[n, q] and last_touch[n] >= 0 and last_touch[n] != q and learner[n, last_touch[n]]:
                    had_pass[n] = True
                last_touch[n] = q
        done, outcome, truncated = drills.check(ev)
        fin = np.flatnonzero(done & live)
        for n in fin:
            results[ep_of_row[n]] = np.nan if truncated[n] else outcome[n]
            passed_ball[ep_of_row[n]] = had_pass[n]
        goal_rows = np.flatnonzero((ev["goal"] != 0) & ~done)
        if len(goal_rows):
            bot.sync(env, goal_rows)
        if len(fin):
            start(fin)
        guard += 1
        if guard > 200000:
            raise RuntimeError("batería sin terminar")
    ok = ~np.isnan(results)
    valid = results[ok]
    wins = ok & (results > 0.5)
    return dict(success=float((valid > 0.5).mean()) if len(valid) else float("nan"),
                conceded=float((valid < -0.5).mean()) if len(valid) else float("nan"),
                episodes=int(len(valid)), successes=int(wins.sum()),
                pass_share=float(passed_ball[wins].mean()) if wins.any() else float("nan"))


class RSProCandidate:
    """RS-Pro L5 (o el nivel indicado) jugando en los lugares del aprendiz: referencia de calibración."""

    def __init__(self, level=5, seed=0):
        self.level = level
        self.seed = seed
        self.bot = None

    def sync(self, env, rows):
        if self.bot is None or self.bot.N != env.N:
            self.bot = RSPro(env, seed=self.seed)
            for t in (0, 1):
                self.bot.configure(np.arange(env.N), t, self.level, STYLE_BALANCED)
        self.bot.sync(env, rows)

    def push(self, env):
        self.bot.push(env)

    def act(self, env, ctrl, out):
        return self.bot.act(env, ctrl, out)
