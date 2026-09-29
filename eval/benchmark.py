"""Partidos completos con protocolo fijo, ambos colores y referencia por contenido.

python -m eval.benchmark candidate.pt --reference frozen.pt --out report.json
Una aprobación en simulación no certifica rendimiento en una sala real.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from env.tasks import load_catalog
from .agents import make_agent
from .matrix import eval_task
from .protocol import file_hash, opponent_identity, protocol_id, source_fingerprint

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SUITE = ROOT / "eval" / "futsal_validation.yaml"


def aggregate(rows):
    counts = {k: sum(r[k] for r in rows) for k in
              ("games", "wins", "draws", "losses", "goals_for", "goals_against", "scoreless_games")}
    counts["points"] = (counts["wins"] + 0.5 * counts["draws"]) / counts["games"]
    counts["scoreless_fraction"] = counts["scoreless_games"] / counts["games"]
    return counts


def assess(results, suite, has_reference):
    reasons = []
    gates = suite["gates"]
    if not has_reference:
        reasons.append("Falta un checkpoint de referencia congelado (--reference)")
    for name, result in results.items():
        threshold = gates["min_points_vs_reference"] if name.endswith("/reference") else gates["min_points_vs_scripted"]
        if result["games"] < gates["min_games"]:
            reasons.append(f"{name}: muestra insuficiente")
        if result["points"] < threshold:
            reasons.append(f"{name}: puntos {result['points']:.3f} < {threshold}")
        if result["scoreless_fraction"] > gates["max_scoreless_fraction"]:
            reasons.append(f"{name}: demasiados partidos sin goles")
    return {"passed": not reasons, "reasons": reasons,
            "scope": "simulation_only", "room_validation": "pending"}


def run_benchmark(model_path, suite, reference=None):
    games = int(suite["games_per_seed"])
    seeds = suite["seeds"]
    if games < 2 or games % 2 or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("games_per_seed debe ser par >= 2; seeds debe contener semillas distintas")
    if not suite["tasks"] or len(set(suite["tasks"])) != len(suite["tasks"]):
        raise ValueError("tasks debe contener tareas distintas")
    model_hash = file_hash(model_path)
    catalog = load_catalog()
    tasks = [catalog[n] for n in suite["tasks"]]
    opponents = {"scripted": "scripted"}
    if reference:
        if file_hash(reference) == model_hash:
            raise ValueError("La referencia debe ser un checkpoint distinto del candidato")
        opponents["reference"] = str(reference)
    identities = {name: opponent_identity(spec) for name, spec in opponents.items()}
    protocol = {"suite": suite, "tasks": [vars(t) for t in tasks],
                "opponents": identities, "sources": source_fingerprint(tasks)}
    agent = make_agent(str(model_path), suite.get("greedy", False))
    results = {}
    for task in tasks:
        for name, spec in opponents.items():
            rows = []
            for seed in seeds:
                opp = make_agent(spec, suite.get("greedy", False))
                rows.append(eval_task(agent, opp, task, games, suite["minutes"], seed=int(seed),
                                      action_delay_max=suite.get("action_delay_max", 0)))
            key = f"{task.name}/{name}"
            results[key] = aggregate(rows)
            results[key]["per_seed"] = rows
            print(f"{key}: {results[key]['points']:.3f} puntos, "
                  f"{results[key]['wins']}-{results[key]['draws']}-{results[key]['losses']} G-E-P", flush=True)
    # Detectar si otro proceso reemplazó un checkpoint mientras se evaluaba.
    if file_hash(model_path) != model_hash or any(opponent_identity(s) != identities[n] for n, s in opponents.items()):
        raise RuntimeError("Un checkpoint cambió durante la evaluación; repetir con copias congeladas")
    return {"candidate_sha256": model_hash, "protocol_id": protocol_id(protocol), "protocol": protocol,
            "results": results, "gate": assess(results, suite, reference is not None)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model")
    ap.add_argument("--reference")
    ap.add_argument("--suite", default=str(DEFAULT_SUITE))
    ap.add_argument("--out", required=True)
    ap.add_argument("--compare", help="reporte previo con exactamente el mismo protocolo")
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    suite = yaml.safe_load(Path(a.suite).read_text(encoding="utf-8"))
    report = run_benchmark(a.model, suite, a.reference)
    if a.compare:
        previous = json.loads(Path(a.compare).read_text(encoding="utf-8"))
        if previous["protocol_id"] != report["protocol_id"]:
            raise ValueError("No son comparables: cambió el protocolo o el contenido de la referencia")
        report["point_changes"] = {name: r["points"] - previous["results"][name]["points"]
                                   for name, r in report["results"].items()}
        for name, delta in report["point_changes"].items():
            if delta < -suite["gates"]["max_point_drop"]:
                report["gate"]["passed"] = False
                report["gate"]["reasons"].append(f"{name}: regresión de {-delta:.3f} puntos")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # No sobrescribir resultados previos ni archivos de entrenamiento.
    with out.open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report["gate"], ensure_ascii=False))
    raise SystemExit(0 if report["gate"]["passed"] else 1)


if __name__ == "__main__":
    main()
