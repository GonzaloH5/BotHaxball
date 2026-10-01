"""Gate reproducible antes de permitir que R3 entre al entrenamiento.

Compara R3 contra el contrato R2 en 1v1--7v7, ambos colores y los tres
estilos. Además fuerza todos los saques para ambos equipos.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from bots.scripted import scripted_actions
from env.tasks import load_catalog, make_env
from eval.agents import make_agent
from eval.matrix import eval_task
from eval.protocol import protocol_id, source_fingerprint

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SUITE = ROOT / "eval" / "scripted_r3_validation.yaml"


def _source_hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in ("bots/scripted.py", "env/haxball_env.py", "env/pegeche.py")}


def _forced_restart(task, kind, trials, seed):
    catalog = load_catalog()
    t = catalog[task]
    rows = trials * 4  # ambos equipos x ambos lados
    env = make_env(t, rows, max(t.n_entities, 15), seed=seed,
                   random_reset_prob=0, optimize_rollout=True)
    env.reset()
    sim = env.sim
    sim.kickoff[:] = False
    sim.mask[:] = sim.base_mask
    sim.vel[:] = 0
    teams = np.tile(np.repeat([0, 1], trials), 2)
    sides = np.repeat([-1, 1], 2 * trials)
    for p in range(env.P):
        sim.player_pos[:, p] = (-150 + 300 * sim.player_team[p], 50 * p - 125)
    all_rows = np.arange(rows)
    if kind == 6:
        env._reset_envs(all_rows, kickoff_team=teams)
    else:
        for n, (team, side) in enumerate(zip(teams, sides)):
            sx = -1 if team == 0 else 1
            origin = np.array([-sx * env.field_w * .8, side * (env.field_h + 20.)])
            if env.rules is not None:
                method = {1: env.rules._throw_in, 2: env.rules._corner, 3: env.rules._goal_kick}[kind]
                if kind != 1:
                    origin = np.array([(-sx if kind == 2 else sx) * (env.field_w + 20.), side * 200.])
                method(n, int(team), origin)
            else:
                sim.pos[n, 0] = (origin if kind == 1 else
                                 ((-sx if kind == 2 else sx) * (env.field_w + 20.), side * 200.))
                env.last_touch[n] = 1 - team
                env._set_piece([n])
    env._phi = env._potentials()
    executed = np.zeros(rows, dtype=bool)
    timed_out = np.zeros(rows, dtype=bool)
    limit = int(max(env.kickoff_limit.max(), env.setpiece_limit.max(), 420))
    for _ in range((limit + env.frame_skip - 1) // env.frame_skip + 5):
        actions = scripted_actions(env, policy="r3", style=-1)
        _, _, _, info = env.step(actions)
        for n, team in enumerate(teams):
            own = sim.player_team == team
            executed[n] |= bool(info["kicked"][n, own].any())
            timed_out[n] |= bool(info["events"]["restart_timeouts"][n, team])
        if (executed | timed_out).all():
            break
    return int(executed.sum()), int(timed_out.sum()), rows


def run_gate(suite):
    catalog = load_catalog()
    results = {}
    total_points = total_games = 0.0
    for ti, name in enumerate(suite["tasks"]):
        task = catalog[name]
        for style in suite["styles"]:
            result = eval_task(make_agent(f"scripted:r3:{style}"), make_agent("scripted:r2"), task,
                               int(suite["games_per_style"]), float(suite["minutes"]),
                               seed=int(suite["seed"]) + 100 * ti + int(style))
            results[f"{name}/style_{style}"] = result
            total_points += result["points"] * result["games"]
            total_games += result["games"]
            print(f"{name} estilo {style}: {result['points']:.3f} puntos "
                  f"({result['wins']}-{result['draws']}-{result['losses']})", flush=True)
    executed = timeouts = attempts = 0
    for ti, name in enumerate(suite["restart_tasks"]):
        for kind in (1, 2, 3, 6):
            ok, expired, n = _forced_restart(name, kind, int(suite["restart_trials"]),
                                             int(suite["seed"]) + 10000 + 10 * ti + kind)
            results[f"restart/{name}/{kind}"] = {"attempts": n, "executed": ok, "timeouts": expired}
            executed += ok
            timeouts += expired
            attempts += n
    overall = total_points / max(total_games, 1)
    restart_execution = executed / max(attempts, 1)
    restart_timeout = timeouts / max(attempts, 1)
    gates = suite["gates"]
    reasons = []
    if overall < gates["min_points_overall"]:
        reasons.append(f"R3 no supera globalmente a R2: {overall:.3f}")
    for name, result in results.items():
        if not name.startswith("restart/") and result["points"] < gates["min_points_per_cell"]:
            reasons.append(f"{name}: {result['points']:.3f} puntos")
    if restart_execution < gates["min_restart_execution"]:
        reasons.append(f"ejecución de saques {restart_execution:.3%}")
    if restart_timeout > gates["max_restart_timeout"]:
        reasons.append(f"timeouts de saques {restart_timeout:.3%}")
    return {"protocol_id": protocol_id({"suite": suite, "sources": source_fingerprint(list(catalog.values()))}),
            "source_sha256": _source_hashes(), "suite": suite, "results": results,
            "summary": {"points": overall, "restart_execution": restart_execution,
                        "restart_timeout": restart_timeout, "restart_attempts": attempts},
            "gate": {"passed": not reasons, "reasons": reasons}}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite", default=str(DEFAULT_SUITE))
    ap.add_argument("--out", default=str(ROOT / "reports" / "scripted_r3_gate.json"))
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    suite = yaml.safe_load(Path(args.suite).read_text(encoding="utf-8"))
    report = run_gate(suite)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False))
    print(json.dumps(report["gate"], ensure_ascii=False))
    raise SystemExit(0 if report["gate"]["passed"] else 1)


if __name__ == "__main__":
    main()

