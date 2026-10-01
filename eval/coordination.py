"""Evaluación completa de cooperación y gates de la fase colectiva.

Ejemplo:
python -m eval.coordination runs/candidate/latest.pt \
  --baseline runs/baselines/multi_it027572_2872839296/baseline.pt \
  --out reports/coordination_025m.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import yaml

from env.tasks import load_catalog
from .agents import make_agent
from .matrix import eval_task
from .protocol import file_hash, opponent_identity, protocol_id, source_fingerprint
from .restart_matrix import evaluate_restarts

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SUITE = ROOT / "eval" / "coordination_validation.yaml"


def aggregate(rows, minutes):
    keys = ("games", "wins", "draws", "losses", "goals_for", "goals_against", "scoreless_games")
    out = {key: sum(row[key] for row in rows) for key in keys}
    out["points"] = (out["wins"] + .5 * out["draws"]) / max(out["games"], 1)
    out["scoreless_fraction"] = out["scoreless_games"] / max(out["games"], 1)
    names = rows[0].get("events", {}) if rows else {}
    out["events"] = {name: sum(row.get("events", {}).get(name, 0) for row in rows) for name in names}
    denom = out["games"] * minutes
    out.update({name + "_per_minute": value / max(denom, 1e-9) for name, value in out["events"].items()})
    second = (out["wins"] + .25 * out["draws"]) / max(out["games"], 1)
    variance = max(0.0, second - out["points"] ** 2)
    out["points_ci95"] = 1.96 * math.sqrt(variance / max(out["games"], 1))
    return out


def _run(agent, opponent, names, games, minutes, seed):
    catalog = load_catalog()
    return {name: eval_task(agent, opponent(), catalog[name], games, minutes, seed + i * 101)
            for i, name in enumerate(names)}


def assess(candidate, baseline, restarts, suite, human):
    g = suite["gates"]
    reasons = []
    core = {"futsal_1v1", "big_3v3", "futsal_3v3"}
    for task in suite["r2_tasks"]:
        drop = baseline["r2"][task]["points"] - candidate["r2"][task]["points"]
        allowed = g["max_core_r2_drop"] if task in core else g["max_retention_r2_drop"]
        if drop > allowed:
            reasons.append(f"{task}/R2 perdió {drop:.3f} puntos (máx {allowed:.3f})")
    for task in suite["r3_tasks"]:
        row = candidate["r3_aggregate"][task]
        if row["points"] < g["min_r3_points"]:
            reasons.append(f"{task}/R3: {row['points']:.3f} puntos")
    rs = candidate["r3_aggregate"]["rs4_4v4"]
    if rs["scoreless_fraction"] >= g["max_rs4_scoreless"]:
        reasons.append(f"RS4: {rs['scoreless_fraction']:.1%} de 0-0")
    if human is None:
        reasons.append("falta calibración humana reservada")
    else:
        per_task = human.get("tasks", {})
        for task in suite["r3_tasks"]:
            row = candidate["r3_aggregate"][task]
            base = baseline["r3_aggregate"][task]
            human_task = suite.get("human_metric_aliases", {}).get(task, task)
            hm = per_task.get(human_task)
            if hm is None:
                reasons.append(f"falta calibración humana de {task}")
                continue
            required = max(g["progressive_multiplier"] * base["progressive_passes_per_minute"],
                           hm["progressive_passes_per_minute_p25"])
            if row["progressive_passes_per_minute"] < required:
                reasons.append(f"{task}: pases progresivos {row['progressive_passes_per_minute']:.3f} < {required:.3f}")
            if row["pass_chains_per_minute"] < hm["pass_chains_per_minute_p25"]:
                reasons.append(f"{task}: cadenas {row['pass_chains_per_minute']:.3f} bajo p25 humano")
            allowed_turnovers = (base["turnovers_per_minute"] + g["max_turnover_increase_per_minute"])
            if row["turnovers_per_minute"] > allowed_turnovers:
                reasons.append(f"{task}: pérdidas inmediatas aumentaron")
    if restarts["useful_fraction"] < g["min_restart_useful"]:
        reasons.append(f"reinicios útiles {restarts['useful_fraction']:.1%}")
    if restarts["corner_useful_fraction"] < g["min_corner_useful"]:
        reasons.append(f"córners útiles {restarts['corner_useful_fraction']:.1%}")
    if restarts["timeout_fraction"] > g["max_restart_timeout"]:
        reasons.append(f"timeouts de reinicio {restarts['timeout_fraction']:.1%}")
    return {"passed": not reasons, "reasons": reasons}


def run(model_path, baseline_path, suite, historical=()):
    games, minutes, seed = int(suite["games"]), float(suite["minutes"]), int(suite["seed"])
    if games < 2 or games % 2:
        raise ValueError("games debe ser par y >= 2")
    candidate_agent = make_agent(str(model_path), suite.get("greedy", False))
    baseline_agent = make_agent(str(baseline_path), suite.get("greedy", False))
    make_r2 = lambda: make_agent("scripted:r2")
    candidate = {"r2": _run(candidate_agent, make_r2, suite["r2_tasks"], games, minutes, seed)}
    baseline = {"r2": _run(baseline_agent, make_r2, suite["r2_tasks"], games, minutes, seed)}
    candidate["r3"], baseline["r3"] = {}, {}
    for style in suite["r3_styles"]:
        maker = lambda style=style: make_agent(f"scripted:r3:{style}")
        candidate["r3"][str(style)] = _run(candidate_agent, maker, suite["r3_tasks"], games, minutes,
                                                    seed + 1000 + 100 * style)
        baseline["r3"][str(style)] = _run(baseline_agent, maker, suite["r3_tasks"], games, minutes,
                                                   seed + 1000 + 100 * style)
    candidate["r3_aggregate"], baseline["r3_aggregate"] = {}, {}
    for task in suite["r3_tasks"]:
        candidate["r3_aggregate"][task] = aggregate(
            [candidate["r3"][str(s)][task] for s in suite["r3_styles"]], minutes)
        baseline["r3_aggregate"][task] = aggregate(
            [baseline["r3"][str(s)][task] for s in suite["r3_styles"]], minutes)
    history = {}
    for path in (baseline_path, *historical):
        identity = file_hash(path)[:12]
        opponent = lambda path=path: make_agent(str(path), suite.get("greedy", False))
        history[identity] = {"path": str(path), "results": _run(candidate_agent, opponent,
                              suite["historical_tasks"], games, minutes, seed + 5000)}
    restarts = evaluate_restarts(candidate_agent, make_agent("scripted:r3:0"),
                                 suite["restart_task"], int(suite["restart_trials"]), seed + 9000)
    human_path = Path(suite["human_metrics"])
    if not human_path.is_absolute():
        human_path = ROOT / human_path
    human = json.loads(human_path.read_text(encoding="utf-8")) if human_path.exists() else None
    gate = assess(candidate, baseline, restarts, suite, human)
    tasks = [load_catalog()[name] for name in sorted(set(suite["r2_tasks"] + suite["r3_tasks"]))]
    protocol = {"suite": suite, "candidate": file_hash(model_path), "baseline": file_hash(baseline_path),
                "historical": [opponent_identity(str(p)) for p in historical],
                "sources": source_fingerprint(tasks)}
    return {"protocol_id": protocol_id({k: v for k, v in protocol.items() if k != "candidate"}),
            "candidate_sha256": protocol["candidate"], "baseline_sha256": protocol["baseline"],
            "minutes": minutes,
            "candidate": candidate, "baseline": baseline, "historical": history,
            "restarts": restarts, "human_metrics": str(human_path), "gate": gate}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model")
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--historical", action="append", default=[])
    ap.add_argument("--suite", default=str(DEFAULT_SUITE))
    ap.add_argument("--previous", help="evaluación anterior de 25M para confirmar dos gates consecutivos")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    suite = yaml.safe_load(Path(args.suite).read_text(encoding="utf-8"))
    report = run(args.model, args.baseline, suite, args.historical)
    streak = 1 if report["gate"]["passed"] else 0
    if args.previous:
        previous = json.loads(Path(args.previous).read_text(encoding="utf-8"))
        if previous["protocol_id"] != report["protocol_id"]:
            raise ValueError("La evaluación anterior usa otro protocolo")
        if previous.get("gate", {}).get("passed") and report["gate"]["passed"]:
            streak = int(previous.get("gate", {}).get("consecutive_passes", 1)) + 1
    report["gate"]["consecutive_passes"] = streak
    report["gate"]["early_stop"] = streak >= 2
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["gate"], ensure_ascii=False))
    raise SystemExit(0 if report["gate"]["passed"] else 1)


if __name__ == "__main__":
    main()

