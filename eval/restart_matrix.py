"""Escenarios forzados de saques para medir ejecución útil de una política."""
from __future__ import annotations

import numpy as np

from env.tasks import load_catalog, make_env
from .agents import reset_agents


def _setup(env, team: int, kind: int, side: int) -> np.ndarray:
    sim = env.sim
    sim.kickoff[:] = False
    sim.mask[:] = sim.base_mask
    sim.vel[:] = 0
    for p in range(env.P):
        sim.player_pos[:, p] = (-150 + 300 * sim.player_team[p], 50 * p - 125)
    rows = np.arange(env.N)
    if kind == 6:
        env._reset_envs(rows, kickoff_team=np.full(env.N, team, dtype=np.int64))
        return sim.ball_pos.copy()
    sx = -1 if team == 0 else 1
    for n in rows:
        origin = np.array([-sx * env.field_w * .8, side * (env.field_h + 20.)])
        if env.rules is not None:
            method = {1: env.rules._throw_in, 2: env.rules._corner, 3: env.rules._goal_kick}[kind]
            if kind != 1:
                origin = np.array([(-sx if kind == 2 else sx) * (env.field_w + 20.), side * 200.])
            method(n, team, origin)
        else:
            sim.pos[n, 0] = (origin if kind == 1 else
                             ((-sx if kind == 2 else sx) * (env.field_w + 20.), side * 200.))
            env.last_touch[n] = 1 - team
            env._set_piece([n])
    env._phi = env._potentials()
    return sim.ball_pos.copy()


def evaluate_restarts(agent, opponent, task="rs4_4v4", trials=8, seed=0, policy_name="candidate"):
    """Evalúa laterales, córners, saques de arco y centrales, mitad por color/lado.

    Un saque general es útil si el dueño lo patea, lo libera y la pelota viaja.
    Un córner usa el contrato más estricto del entorno: dirección interior y
    continuidad ofensiva durante la ventana configurada.
    """
    t = load_catalog()[task]
    totals = {str(kind): {"attempts": 0, "useful": 0, "timeouts": 0} for kind in (1, 2, 3, 6)}
    for kind in (1, 2, 3, 6):
        for team in (0, 1):
            for side in (-1, 1):
                env = make_env(t, trials, max(t.n_entities, 15), seed=seed + kind * 100 + team * 10 + side,
                               random_reset_prob=0, optimize_rollout=True, corner_curriculum=False)
                env.reset()
                origin = _setup(env, team, kind, side)
                obs = env.observe()
                reset_agents((agent, opponent), env)
                useful = np.zeros(trials, dtype=bool)
                expired = np.zeros(trials, dtype=bool)
                kicked = np.zeros(trials, dtype=bool)
                limit = int(max(env.kickoff_limit.max(), env.setpiece_limit.max(), 420))
                red, blue = np.arange(env.T), np.arange(env.T, env.P)
                owner_players = env.sim.player_team == team
                for _ in range((limit + env.frame_skip - 1) // env.frame_skip + 70):
                    actions = np.empty((trials, env.P), dtype=np.int64)
                    if team == 0:
                        actions[:, red] = agent(env, obs, red)
                        actions[:, blue] = opponent(env, obs, blue)
                    else:
                        actions[:, red] = opponent(env, obs, red)
                        actions[:, blue] = agent(env, obs, blue)
                    obs, _, done, info = env.step(actions)
                    kicked |= info["kicked"][:, owner_players].any(axis=1)
                    expired |= info["events"]["restart_timeouts"][:, team] > 0
                    if kind == 2:
                        useful |= info["events"]["corner_successes"][:, team] > 0
                    else:
                        released = (~env.sim.kickoff if kind == 6 else env.setpiece_team < 0)
                        travel = np.linalg.norm(env.sim.ball_pos - origin, axis=1)
                        useful |= kicked & released & (travel >= env.rcfg.team_pass_min_dist_frac * env.field_w)
                    if done.any():
                        reset_agents((agent, opponent), env, done)
                    if (useful | expired).all():
                        break
                row = totals[str(kind)]
                row["attempts"] += trials
                row["useful"] += int(useful.sum())
                row["timeouts"] += int(expired.sum())
    attempts = sum(row["attempts"] for row in totals.values())
    useful = sum(row["useful"] for row in totals.values())
    timeouts = sum(row["timeouts"] for row in totals.values())
    corner = totals["2"]
    return {"policy": policy_name, "task": task, "by_kind": totals, "attempts": attempts,
            "useful": useful, "useful_fraction": useful / max(attempts, 1),
            "timeout_fraction": timeouts / max(attempts, 1),
            "corner_useful_fraction": corner["useful"] / max(corner["attempts"], 1)}

