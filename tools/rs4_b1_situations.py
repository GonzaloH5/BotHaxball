"""Medir la señal de gol de las situaciones técnicas reales con una política (PLAN_RS4.md 3).

Antes de comparar candidatos el plan exige al menos 1 gol cada 10 situaciones de ataque con la
política A. Cada fila arranca en un estado real (escenario "recorded" de env/rs4_v3.py: mezcla
ataque/juego abierto/saque y espejo al azar), la misma política controla a los ocho jugadores y la
situación termina como en el entrenamiento: gol, saque nuevo o 12 s.

Se informa por tipo de situación junto a la tasa humana del mismo banco. La tasa humana cuenta
goles dentro de 12 s aunque haya un saque en el medio (el banco no guarda el corte), así que es una
cota superior de lo comparable.

  python -m tools.rs4_b1_situations --policy runs/rs4_b1/inputs/v3_2568M.pt \\
      --out reports/rs4_b1/situations_v3_2568M.json --situations 3000 --workers 8
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
POOLS = ("attack", "open", "restart")
THRESHOLD = 0.10  # goles del atacante por situación de ataque (PLAN_RS4.md 3)


def human_rates(path):
    from env.rs4_states import StateBank
    bank = StateBank(path)
    a, pools = bank.arrays, bank.pools()
    scorer, last, taker = a["goal_team"], a["last_touch"], a["taker"]
    out = {}
    for pool, ids in pools.items():
        attacker = taker[ids] if pool == "restart" else last[ids]
        out[pool] = dict(states=int(len(ids)), goals_per_100=100 * float((scorer[ids] >= 0).mean()),
                         attacker_goals_per_100=100 * float(((scorer[ids] >= 0) & (scorer[ids] == attacker)).mean()))
    return out


def _worker(job):
    import torch
    torch.set_num_threads(1)
    from env.rs4_v3 import RS4ScenarioEnv
    from eval.rs4_b1 import Policy, environment
    spec, states, seed, quota, greedy, n, ticks, mix = job
    env = RS4ScenarioEnv(environment(n, seed), {"guide_coef": 0., "drill_fraction": .4,
                                               "scenario_weights": {"recorded": 1},
                                               "recorded": {"path": states, "ticks": ticks, "mix": mix}})
    env.reset()
    rows = np.arange(n)
    env.assign(rows, scenario="recorded", team=0)
    policy = Policy(spec)
    policy.reset(n, env.P)
    rng = np.random.default_rng(seed)
    everyone = np.ones((n, env.P), dtype=bool)
    obs = env.observe()
    results = []
    while len(results) < quota:
        actions = policy.act(env.base, obs, everyone, rng, greedy=greedy)
        obs, _, _, info = env.step(actions)
        finished = np.array(sorted({r["row"] for r in info["scenario_result"]}), dtype=np.int64)
        results += [{key: r[key] for key in ("pool", "attacker", "goal_team", "ticks", "truncated", "state")}
                    for r in info["scenario_result"]]
        if len(finished):
            # step reasigna por cuota (partidos completos); aquí todas las filas son situaciones.
            env.assign(finished, scenario="recorded", team=0)
            policy.reset(n, env.P, rows=finished)
            obs = obs.copy()
            obs[finished] = env.base.observe(finished)
    return results


def summarize(results):
    out = {}
    for pool in POOLS:
        rows = [r for r in results if r["pool"] == pool]
        if not rows:
            continue
        goals = np.array([r["goal_team"] >= 0 for r in rows])
        attacker = np.array([r["goal_team"] >= 0 and r["goal_team"] == r["attacker"] for r in rows])
        out[pool] = dict(situations=len(rows), goals_per_100=100 * float(goals.mean()),
                         attacker_goals_per_100=100 * float(attacker.mean()),
                         defender_goals_per_100=100 * float((goals & ~attacker).mean()),
                         cut_by_restart_or_limit=float(np.mean([r["truncated"] for r in rows])),
                         mean_seconds=float(np.mean([r["ticks"] for r in rows])) / 60)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", required=True, help="checkpoint o scripted:r3:<estilo>")
    ap.add_argument("--states", default=str(ROOT / "data" / "rs4_states" / "entrenamiento.npz"))
    ap.add_argument("--situations", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--rows", type=int, default=32, help="situaciones simultáneas por proceso")
    ap.add_argument("--ticks", type=int, default=720)
    ap.add_argument("--mix", default='{"attack": 0.4, "open": 0.3, "restart": 0.3}')
    ap.add_argument("--greedy", action="store_true", help="greedy en lugar de muestrear como en el entrenamiento")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    mix = json.loads(a.mix)
    quota = -(-a.situations // a.workers)
    jobs = [(a.policy, a.states, a.seed + 101 * i, quota, a.greedy, a.rows, a.ticks, mix) for i in range(a.workers)]
    started = time.time()
    if a.workers > 1:
        from multiprocessing import get_context
        with get_context("spawn").Pool(a.workers) as pool:
            parts = pool.map(_worker, jobs)
    else:
        parts = [_worker(job) for job in jobs]
    results = [r for part in parts for r in part]
    summary = summarize(results)
    attack = summary.get("attack", {}).get("attacker_goals_per_100", 0.0) / 100
    report = dict(version="RS4-b1-situations-1", policy=a.policy, states=a.states, greedy=a.greedy, seed=a.seed,
                  ticks=a.ticks, mix=mix, seconds=round(time.time() - started, 1), summary=summary,
                  human=human_rates(a.states), threshold_attacker_goals_per_attack=THRESHOLD,
                  enough_goal_signal=bool(attack >= THRESHOLD), results=results)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    labels = {"attack": "ataque", "open": "juego abierto", "restart": "saque"}
    for pool, row in summary.items():
        human = report["human"][pool]
        print(f"{labels[pool]:>13}: {row['situations']} situaciones | goles {row['goals_per_100']:.1f}/100 "
              f"(atacante {row['attacker_goals_per_100']:.1f}, defensor {row['defender_goals_per_100']:.1f}) "
              f"| {row['mean_seconds']:.1f} s promedio | humanos: atacante {human['attacker_goals_per_100']:.1f}/100")
    print(f"señal de gol {'SUFICIENTE' if report['enough_goal_signal'] else 'INSUFICIENTE'}: "
          f"{100 * attack:.1f} goles del atacante cada 100 ataques (umbral {100 * THRESHOLD:.0f}) | {report['seconds']} s")


if __name__ == "__main__":
    main()
