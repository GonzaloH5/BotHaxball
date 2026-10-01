"""Comparar especialista y generalista congelado en RS4 contra los mismos rivales.

python -m tools.evaluate_rs4 --run rs4 --games 128 --minutes 2
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from .prepare_rs4 import ROOT, TASK, file_hash


def evaluate(run="rs4", checkpoint=None, opponents=None, games=128, minutes=2.0, seed=51):
    if not run or run in (".", "..") or any(char in run for char in '/\\:'):
        raise ValueError("run debe ser un nombre simple, sin rutas")
    if games < 2 or games % 2 or not math.isfinite(minutes) or minutes * 3600 < 3:
        raise ValueError("games debe ser par y >= 2; minutes debe permitir una decisión")
    directory = ROOT / "runs" / run
    manifest = json.loads((directory / "specialization.json").read_text(encoding="utf-8"))
    parent = directory / "parent.pt"
    if file_hash(parent) != manifest["source_sha256"]:
        raise ValueError("La referencia parent.pt cambió; no se compara con una baseline distinta")
    candidate = Path(checkpoint).resolve() if checkpoint else directory / "latest.pt"
    if not candidate.is_file():
        raise ValueError(f"No existe {candidate}")
    opponents = opponents or ["scripted:r3", str(parent)]
    reports = directory / "evaluations"
    reports.mkdir(exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="rs4_", dir=reports))
    with tempfile.TemporaryDirectory(prefix="_rs4_eval_", dir=directory) as temporary:
        temporary = Path(temporary)
        baseline, specialist = temporary / "generalist.pt", temporary / "specialist.pt"
        shutil.copy2(parent, baseline)
        shutil.copy2(candidate, specialist)
        if file_hash(baseline) != manifest["source_sha256"]:
            raise ValueError("La referencia cambió durante la copia; no se evalúa")
        frozen_opponents = []
        identities = []
        for i, rival in enumerate(opponents):
            if rival.startswith("scripted"):
                frozen_opponents.append(rival)
                identities.append(dict(rival=rival))
            else:
                source = Path(rival).resolve()
                frozen = temporary / f"opponent_{i}.pt"
                shutil.copy2(source, frozen)
                frozen_opponents.append(str(frozen))
                identities.append(dict(rival=str(source), sha256=file_hash(frozen)))
        summary = dict(task=TASK, games=games, minutes=minutes, seed=seed,
                       generalist_sha256=file_hash(baseline), specialist_sha256=file_hash(specialist),
                       candidate=str(candidate), opponents=identities, results=[])
        # eval.matrix alterna rojo/azul y fija RNG de políticas y entorno por semilla.
        # Sin greedy: conservar la evaluación estocástica en ambos candidatos.
        process_env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", NUMBA_NUM_THREADS="2")
        for i, rival in enumerate(frozen_opponents):
            rows = {}
            for label, model in (("generalist", baseline), ("specialist", specialist)):
                report = output / f"{i}_{label}.json"
                command = [sys.executable, "-m", "eval.matrix", str(model), "--tasks", TASK,
                           "--vs", rival, "--games", str(games), "--minutes", str(minutes),
                           "--seed", str(seed), "--out", str(report)]
                subprocess.run(command, cwd=ROOT, env=process_env, check=True)
                rows[label] = json.loads(report.read_text(encoding="utf-8"))["runs"][-1]["tasks"][TASK]
            rows["opponent"] = identities[i]
            rows["points_delta"] = rows["specialist"]["points"] - rows["generalist"]["points"]
            summary["results"].append(rows)
        (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return output, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="rs4")
    parser.add_argument("--checkpoint", help="Checkpoint especializado concreto; default: latest.pt")
    parser.add_argument("--opponents", nargs="+", help="Rivales comunes; default: scripted:r3 y parent.pt")
    parser.add_argument("--games", type=int, default=128)
    parser.add_argument("--minutes", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=51)
    parser.add_argument("--styles", action="store_true", help="Evaluar R3 0/1/2 por separado y parent.pt")
    args = parser.parse_args()
    if args.styles and args.opponents:
        parser.error("Usar --styles o --opponents, no ambos")
    if args.styles:
        args.opponents = ["scripted:r3:0", "scripted:r3:1", "scripted:r3:2",
                          str(ROOT / "runs" / args.run / "parent.pt")]
    output, summary = evaluate(args.run, args.checkpoint, args.opponents, args.games, args.minutes, args.seed)
    for row in summary["results"]:
        print(f"{row['opponent']['rival']}: generalista {row['generalist']['points']:.3f}, "
              f"especialista {row['specialist']['points']:.3f}, delta {row['points_delta']:+.3f}")
    print(f"Resultados: {output / 'summary.json'}")
    print("Repetir con otras semillas; una diferencia pequeña no demuestra superioridad.")


if __name__ == "__main__":
    main()
