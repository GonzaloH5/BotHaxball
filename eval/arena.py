"""Torneo entre agentes con tabla Elo.

python -m eval.arena runs/classic_1v1/latest.pt scripted
python -m eval.arena runs/classic_1v1/ckpt_*.pt scripted --games 256 --minutes 3

Cada partido es de verdad (saque desde el centro, se sigue jugando tras el gol) y dura
`--minutes` minutos de juego. Cada cruce se juega con ambos agentes de rojo y de azul.
"""
from __future__ import annotations

import argparse
import glob
import itertools

import numpy as np

from env.haxball_env import HaxballEnv
from env.rewards import RewardConfig

from .agents import env_config, env_kwargs, make_agent, reset_agents


def play(agent_red, agent_blue, n_games=128, minutes=3.0, n_per_team=1, stadium="classic",
         frame_skip=3, seed=0, env_kw=None, return_details=False):
    """Devuelve (victorias rojo, empates, victorias azul, goles rojo, goles azul)."""
    ticks = int(minutes * 60 * 60)
    if n_games < 1 or frame_skip < 1 or ticks < frame_skip:
        raise ValueError("Se requiere al menos un partido y una decisión por partido")
    env = HaxballEnv(n_games, n_per_team, stadium, frame_skip, max_ticks=10**9,
                     random_reset_prob=0.0, reward=RewardConfig(shaping_coef=0.0), seed=seed,
                     **(env_kw or {}))
    obs = env.reset()
    env.sim.reset_kickoff(np.arange(n_games), kickoff_team=0)
    obs = env.observe()
    reset_agents((agent_red, agent_blue), env)
    red = np.arange(env.T)
    blue = np.arange(env.T, env.P)
    for _ in range(ticks // frame_skip):
        a = np.empty((n_games, env.P), dtype=np.int64)
        a[:, red] = agent_red(env, obs, red)
        a[:, blue] = agent_blue(env, obs, blue)
        obs, _, done, _ = env.step(a)
        if done.any():
            reset_agents((agent_red, agent_blue), env, done)
    r, b = env.score[:, 0], env.score[:, 1]
    if return_details:
        return {"wins": int((r > b).sum()), "draws": int((r == b).sum()), "losses": int((b > r).sum()),
                "goals_for": int(r.sum()), "goals_against": int(b.sum()),
                "scoreless_games": int(((r + b) == 0).sum())}
    return int((r > b).sum()), int((r == b).sum()), int((b > r).sum()), int(r.sum()), int(b.sum())


def elo_from_results(names, results, iters=200):
    """Elo por máxima verosimilitud simple (iterativo) a partir de puntos (1 / 0.5 / 0)."""
    elo = {n: 1000.0 for n in names}
    for _ in range(iters):
        for (a, b), (wa, d, wb) in results.items():
            n = wa + d + wb
            if n == 0:
                continue
            ea = 1 / (1 + 10 ** ((elo[b] - elo[a]) / 400))
            sa = (wa + 0.5 * d) / n
            k = 8.0
            elo[a] += k * (sa - ea)
            elo[b] -= k * (sa - ea)
    mean = np.mean(list(elo.values()))
    return {k: v - mean + 1000 for k, v in elo.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("agents", nargs="+", help="rutas .pt (acepta comodines), 'scripted', 'scripted:0.3', 'random'")
    ap.add_argument("--games", type=int, default=128)
    ap.add_argument("--minutes", type=float, default=3.0)
    ap.add_argument("--n-per-team", type=int, default=None, help="por defecto, el del checkpoint")
    ap.add_argument("--stadium", default=None, help="por defecto, el del checkpoint")
    ap.add_argument("--task", default=None, help="tarea de train/tasks.yaml (estadio + formato + reglas)")
    ap.add_argument("--greedy", action="store_true", help="acción más probable en vez de muestrear")
    args = ap.parse_args()

    specs = []
    for s in args.agents:
        specs += sorted(glob.glob(s)) if any(c in s for c in "*?[") else [s]
    agents = {s: make_agent(s, args.greedy) for s in specs}
    cfg = env_config(specs, args.stadium, args.n_per_team, args.task)
    args.stadium, args.n_per_team = cfg["stadium"], cfg["n_per_team"]
    ekw = env_kwargs(cfg)
    results = {}
    for a, b in itertools.combinations(specs, 2):
        w1, d1, l1, g1a, g1b = play(agents[a], agents[b], args.games // 2, args.minutes,
                                    args.n_per_team, args.stadium, env_kw=ekw)
        w2, d2, l2, g2a, g2b = play(agents[b], agents[a], args.games // 2, args.minutes,
                                    args.n_per_team, args.stadium, seed=1, env_kw=ekw)
        wa, d, wb = w1 + l2, d1 + d2, l1 + w2
        results[(a, b)] = (wa, d, wb)
        print(f"{a}  vs  {b}:  {wa}G {d}E {wb}P   goles {g1a + g2b}-{g1b + g2a}", flush=True)
    elo = elo_from_results(specs, results)
    print("\nElo:")
    for n, e in sorted(elo.items(), key=lambda x: -x[1]):
        print(f"  {e:7.0f}  {n}")


if __name__ == "__main__":
    main()
