"""Evaluar un candidato RS4 con el protocolo fijo RS4-b1 (eval/rs4_b1.py).

  python -m tools.evaluate_rs4_b1 --candidate runs/rs4_v3_public/champion.pt \\
      --out reports/rs4_b1/eval/champion_dev.json --games 8 --workers 9

Compañeros y rivales por defecto: familias RESERVADAS para evaluación (no se usan para entrenar;
ver EVAL_TEAMMATES/EVAL_RIVALS y train/config_rs4_b1*.yaml). Los mismos estados y semillas se
usan para todos los candidatos, de modo que la comparación es pareada.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ARCHIVE = "runs/pod_archive_20261003/runs"
EVAL_TEAMMATES = ["scripted:r3:2", f"{ARCHIVE}/bc_rs4_v2_20261001/bc.pt", f"{ARCHIVE}/rs4_v5d/eval_50M.pt"]
EVAL_RIVALS = [f"{ARCHIVE}/rs4_v3_public/champion.pt", "scripted:r3:1", f"{ARCHIVE}/rs4_v5d/eval_300M.pt"]


def _resolve(spec):
    if spec.startswith("scripted"):
        return spec
    path = Path(spec)
    path = path if path.is_absolute() else ROOT / path
    if not path.exists() and spec.startswith(ARCHIVE):
        # En el Pod los originales están en runs/ (el archivo local es una copia).
        path = ROOT / ("runs" + spec[len(ARCHIVE):])
    return str(path)


def _cell_job(job):
    import torch
    torch.set_num_threads(1)
    from env.rs4_states import StateBank
    from eval.rs4_b1 import Policy, play_cell
    candidate, teammate, rival, states, state_ids, seed, greedy = job
    bank = StateBank(states)
    result = play_cell(Policy(candidate), Policy(teammate), Policy(rival), bank, state_ids, seed, greedy)
    result.update(teammate=teammate, rival=rival)
    return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in result.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--teammates", nargs="*", default=EVAL_TEAMMATES)
    ap.add_argument("--rivals", nargs="*", default=EVAL_RIVALS)
    ap.add_argument("--states", default=str(ROOT / "data" / "rs4_states" / "desarrollo.npz"))
    ap.add_argument("--games", type=int, default=8, help="juegos por celda (x2 colores x4 puestos)")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--sample", action="store_true", help="muestrear acciones en lugar de greedy")
    ap.add_argument("--restart-battery", action="store_true")
    ap.add_argument("--battery-rival", default=f"{ARCHIVE}/rs4_v3_public/champion.pt")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from env.rs4_states import StateBank, KIND_OPEN
    from eval.rs4_b1 import PROTOCOL, sha256
    from tools.rs4_freeze import fingerprint
    candidate = _resolve(a.candidate)
    bank = StateBank(a.states)
    open_ids = bank.select([KIND_OPEN])
    state_ids = np.sort(np.random.default_rng(a.seed).choice(open_ids, a.games, replace=False))
    jobs = [(candidate, _resolve(t), _resolve(r), a.states, state_ids, a.seed + 101 * i, not a.sample)
            for i, (t, r) in enumerate((t, r) for t in a.teammates for r in a.rivals)]
    started = time.time()
    cells = []
    if a.workers > 1:
        from multiprocessing import get_context
        with get_context("spawn").Pool(min(a.workers, len(jobs))) as pool:
            cells = pool.map(_cell_job, jobs)
    else:
        cells = [_cell_job(job) for job in jobs]
    battery = None
    if a.restart_battery:
        import torch
        torch.set_num_threads(4)
        from eval.rs4_b1 import Policy, restart_battery
        battery = restart_battery(Policy(candidate), Policy(_resolve(a.battery_rival)), bank, a.seed)
    code, _ = fingerprint()
    contract = json.loads((ROOT / "reports" / "rs4_b1" / "contract.json").read_text(encoding="utf-8"))["version"]
    report = dict(protocol=PROTOCOL, contract=contract, code_fingerprint=code[:16], candidate=a.candidate,
                  candidate_sha256=sha256(candidate) if not candidate.startswith("scripted") else candidate,
                  states=Path(a.states).name, state_ids=state_ids.tolist(), seed=a.seed, games_per_cell=a.games,
                  greedy=not a.sample, seconds=time.time() - started, cells=cells, restart_battery=battery)
    report["summary"] = summarize(report)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["summary"], indent=1, ensure_ascii=False))


def summarize(report):
    rows = []
    for cell in report["cells"]:
        for i in range(len(cell["points"])):
            rows.append(dict(teammate=Path(cell["teammate"]).parent.name or cell["teammate"],
                             rival=Path(cell["rival"]).parent.name or cell["rival"],
                             color=cell["color"][i], slot=cell["slot"][i], game=cell["game"][i],
                             points=cell["points"][i], goal_diff=cell["goals_for"][i] - cell["goals_against"][i],
                             goals_for=cell["goals_for"][i], goals_against=cell["goals_against"][i],
                             shots=cell["shots"][i], passes=cell["passes"][i], turnovers=cell["turnovers"][i],
                             own_goals=cell["own_goals"][i], kickoff_stalls=cell["kickoff_stalls"][i],
                             parked=cell["parked"][i] / cell["decisions"], idle=cell["idle"][i] / cell["decisions"],
                             keepers=cell["keepers"][i] / cell["decisions"], trace=cell["trace"][i]))
    groups = {}
    for r in rows:
        groups.setdefault((r["teammate"], r["rival"], r["color"], r["slot"]), []).append(r)
    cell_means = {"|".join(map(str, k)): float(np.mean([r["points"] for r in v])) for k, v in groups.items()}
    unique = len({r["trace"] for r in rows}) / max(len(rows), 1)
    mean = lambda key: float(np.mean([r[key] for r in rows]))
    by_pair = {}
    for r in rows:
        by_pair.setdefault(f"{r['teammate']} + {r['rival']}", []).append(r["points"])
    return dict(balanced_points=float(np.mean(list(cell_means.values()))), games=len(rows),
                goals_per_game=mean("goals_for") + mean("goals_against"), goal_diff=mean("goal_diff"),
                shots_per_game=mean("shots"), passes_per_game=mean("passes"), turnovers_per_game=mean("turnovers"),
                own_goals_per_game=mean("own_goals"), kickoff_stalls_per_game=mean("kickoff_stalls"),
                parked_fraction=mean("parked"), idle_fraction=mean("idle"), two_keepers_fraction=mean("keepers"),
                unique_trajectories=unique, points_by_pair={k: float(np.mean(v)) for k, v in by_pair.items()},
                restart_battery=report.get("restart_battery"))


if __name__ == "__main__":
    main()
