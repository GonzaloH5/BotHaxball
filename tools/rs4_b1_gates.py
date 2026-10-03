"""Condiciones para ampliar un candidato (PLAN_RS4.md 4) a partir de informes de tools.evaluate_rs4_b1.

  python -m tools.rs4_b1_gates --reference reports/rs4_b1/eval/champion_final.json \\
      --candidate reports/rs4_b1/eval/A_seed1_final.json reports/rs4_b1/eval/A_seed2_final.json

1. Saques: >= 95% de la batería técnica resuelta (sin automatismos: la política ejecuta).
2. Sin retroceso: ninguna celda con diferencia < -5 pp y su intervalo de 90% sin cero.
   Celdas con < 32 partidos no deciden (se informan como "ampliar").
3. Mejora: resultado equilibrado de una instancia >= +5 pp con intervalo de 95% positivo
   (bootstrap pareado por estado inicial).
4. Sin atajos: autogoles, saques trabados, jugador colgado y dos arqueros no empeoran más que
   la tolerancia; >= 95% de trayectorias distintas.
5. Reproducible: todas las semillas de entrenamiento cumplen 1-4.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

TOLERANCE = dict(own_goals_per_game=0.05, kickoff_stalls_per_game=0.02, parked_fraction=0.05,
                 two_keepers_fraction=0.05)
MIN_CELL_GAMES = 32


def _rows(report):
    out = {}
    for cell in report["cells"]:
        for i in range(len(cell["points"])):
            key = (cell["teammate"], cell["rival"], cell["color"][i], cell["slot"][i])
            out[key + (cell["game"][i],)] = cell["points"][i]
    return out


def _same_protocol(a, b):
    fields = ("protocol", "contract", "states", "state_ids", "seed", "games_per_cell", "greedy")
    return [f for f in fields if a.get(f) != b.get(f)]


def evaluate(reference, candidate, rng, cell_level="color_slot"):
    """cell_level: "color_slot" (compañero, rival, color, puesto: 72 celdas de 32 partidos, regla original)
    o "pair" (compañero x rival: 9 celdas de 256 partidos, intervalo por bootstrap agrupado por estado
    inicial). Con 72 celdas, un candidato de igual fuerza muestra algún retroceso falso en ~2/3 de las
    evaluaciones (ver reports/rs4_b1/gate_power.md)."""
    issues = _same_protocol(reference, candidate)
    if issues:
        return dict(valid=False, reason=f"protocolo distinto: {issues}")
    ref, cand = _rows(reference), _rows(candidate)
    keys = sorted(set(ref) & set(cand))
    games = sorted({k[-1] for k in keys})
    cells = defaultdict(list)
    for k in keys:
        cells[k[:-1]].append(k)
    regression_cells = cells
    if cell_level == "pair":
        regression_cells = defaultdict(list)
        for k in keys:
            regression_cells[k[:2]].append(k)
    elif cell_level != "color_slot":
        raise ValueError("cell_level debe ser color_slot o pair")
    # 3. resultado equilibrado pareado: bootstrap sobre estados iniciales (juegos)
    def balanced(selection):
        diffs = []
        for cell, members in cells.items():
            chosen = [m for m in members if m[-1] in selection] if selection is not None else members
            weights = [selection.count(m[-1]) for m in chosen] if selection is not None else [1] * len(chosen)
            if sum(weights):
                diffs.append(np.average([cand[m] - ref[m] for m in chosen], weights=weights))
        return float(np.mean(diffs))
    point = balanced(None)
    boots = []
    for _ in range(1000):
        sample = list(rng.choice(games, len(games), replace=True))
        boots.append(balanced(sample))
    low, high = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    improvement = dict(difference=point, ci95=(low, high), passed=point >= 0.05 and low > 0)
    # 2. retroceso por celda
    regressions, undecided = [], []
    for cell, members in regression_cells.items():
        diff = np.array([cand[m] - ref[m] for m in members])
        if len(diff) < MIN_CELL_GAMES:
            undecided.append("|".join(map(str, cell)))
            continue
        # bootstrap agrupado por estado inicial (en "color_slot" cada grupo tiene un único partido)
        groups = np.array([m[-1] for m in members])
        unique = np.unique(groups)
        sums = np.array([diff[groups == g].sum() for g in unique])
        counts = np.array([(groups == g).sum() for g in unique])
        draws = rng.integers(0, len(unique), (1000, len(unique)))
        b = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
        upper = float(np.percentile(b, 95))
        if diff.mean() < -0.05 and upper < 0:
            regressions.append(dict(cell="|".join(map(str, cell)), difference=float(diff.mean()), ci90_upper=upper))
    no_regression = dict(passed=not regressions and not undecided, regressions=regressions,
                         undecided_cells=len(undecided), note="celdas con < 32 partidos: ampliar antes de decidir" if undecided else None)
    # 1. saques
    battery = (candidate.get("restart_battery") or {}).get("all", {}).get("success")
    restarts = dict(success=battery, passed=battery is not None and battery >= 0.95)
    # 4. atajos
    cs, rs = candidate["summary"], reference["summary"]
    shortcuts = {k: dict(candidate=cs[k], reference=rs[k], passed=cs[k] <= rs[k] + tol) for k, tol in TOLERANCE.items()}
    shortcuts["unique_trajectories"] = dict(candidate=cs["unique_trajectories"], passed=cs["unique_trajectories"] >= 0.95)
    shortcuts_ok = all(v["passed"] for v in shortcuts.values())
    return dict(valid=True, restarts=restarts, no_regression=no_regression, improvement=improvement,
                shortcuts=dict(passed=shortcuts_ok, detail=shortcuts),
                passed=restarts["passed"] and no_regression["passed"] and improvement["passed"] and shortcuts_ok)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--candidate", nargs="+", required=True, help="un informe por semilla de entrenamiento")
    ap.add_argument("--cell-level", choices=("color_slot", "pair"), default="color_slot",
                    help="celdas del criterio de retroceso (ver evaluate)")
    ap.add_argument("--out")
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    reference = json.loads(Path(a.reference).read_text(encoding="utf-8"))
    results = {path: evaluate(reference, json.loads(Path(path).read_text(encoding="utf-8")), rng, a.cell_level)
               for path in a.candidate}
    decision = dict(reference=a.reference, cell_level=a.cell_level, seeds=results,
                    reproducible=len(results) >= 2 and all(r.get("passed") for r in results.values()),
                    passed=len(results) >= 2 and all(r.get("passed") for r in results.values()))
    text = json.dumps(decision, indent=1, ensure_ascii=False)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
