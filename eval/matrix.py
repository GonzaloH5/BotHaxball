"""Matriz de evaluación por tarea (mapa × formato): detecta si el bot se encasilla en una modalidad.

python -m eval.matrix runs/multi/latest.pt                       # tareas activas del checkpoint vs bot
python -m eval.matrix runs/multi/latest.pt --tasks all --vs runs/multi/fin_etapa0.pt
python -m eval.matrix runs/multi/latest.pt --games 64 --minutes 2 --out runs/multi/eval_matrix.json

Para cada tarea juega `--games` partidos contra el bot scripteado (y opcionalmente contra otro checkpoint),
la mitad de rojo y la mitad de azul. Informa puntos (victoria 1, empate 0.5) y goles, y guarda un JSON.
Con --best, compara contra la mejor marca guardada en ese JSON y marca las regresiones.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from env.tasks import load_catalog

from .agents import make_agent
from .arena import play
from .protocol import protocol_id, source_fingerprint, opponent_identity


def eval_task(agent, opp, task, games, minutes, seed=0, action_delay_max=0):
    if games < 2 or games % 2 or not math.isfinite(minutes) or minutes * 3600 < 3:
        raise ValueError("games debe ser par y >= 2; minutes debe permitir al menos una decisión")
    kw = {"powershot": task.powershot, "out_of_bounds": task.out_of_bounds, "obs_layout": "universal",
          "rules": task.rules, "action_delay_max": action_delay_max}
    half = games // 2
    # Semillas de política y de rivales, además de las del entorno. Restaurar el RNG
    # de torch permite usar esta evaluación dentro de un entrenamiento sin alterarlo.
    legs = []
    for leg, (red, blue) in enumerate(((agent, opp), (opp, agent))):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + leg)
            for offset, player in enumerate((agent, opp)):
                if hasattr(player, "rng"):
                    player.rng = np.random.default_rng(seed + leg + 10000 * (offset + 1))
            legs.append(play(red, blue, half, minutes, task.n_per_team, task.stadium,
                             seed=seed + leg, env_kw=kw, return_details=True))
    a, b = legs
    wins, draws, losses = a["wins"] + b["losses"], a["draws"] + b["draws"], a["losses"] + b["wins"]
    n = wins + draws + losses
    return {"points": (wins + 0.5 * draws) / n, "games": n, "wins": wins, "draws": draws, "losses": losses,
            "goals_for": a["goals_for"] + b["goals_against"],
            "goals_against": a["goals_against"] + b["goals_for"],
            "scoreless_games": a["scoreless_games"] + b["scoreless_games"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--tasks", default=None, help="'all', lista separada por comas, o las activas del checkpoint")
    ap.add_argument("--vs", default="scripted", help="rival: 'scripted', 'scripted:0.3' o un .pt")
    ap.add_argument("--games", type=int, default=32)
    ap.add_argument("--minutes", type=float, default=2.0)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="JSON de salida (por defecto, eval_matrix.json junto al modelo)")
    ap.add_argument("--best", action="store_true", help="comparar con la mejor marca guardada y actualizarla")
    ap.add_argument("--drop", type=float, default=0.15, help="caída de puntos que cuenta como regresión")
    a = ap.parse_args()

    catalog = load_catalog()
    ck = torch.load(a.model, map_location="cpu", weights_only=False)
    if ck.get("env", {}).get("obs_layout") != "universal":
        raise SystemExit(f"{a.model} no es un modelo multi-tarea (obs '{ck.get('env', {}).get('obs_layout', 'flat')}'): "
                         "la matriz evalúa modelos de train.multitask / train.bc (p. ej. runs/multi/latest.pt). "
                         "Para modelos de un solo mapa usar eval.arena.")
    if a.tasks in (None, ""):
        names = ck.get("env", {}).get("tasks") or list(catalog)
    elif a.tasks == "all":
        names = list(catalog)
    else:
        names = a.tasks.split(",")
    agent = make_agent(a.model, a.greedy)
    opp = make_agent(a.vs, a.greedy)
    out = Path(a.out) if a.out else Path(a.model).with_name("eval_matrix.json")
    prev = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    best = prev.get("best", {})

    rows = {}
    print(f"{a.model} vs {a.vs} | {a.games} partidos de {a.minutes:g} min por tarea\n")
    print(f"{'tarea':12s} {'mapa':15s} {'fmt':>5s} {'puntos':>7s} {'G-E-P':>10s} {'goles':>9s}  nota")
    regressions = []
    for n in names:
        t = catalog[n]
        r = eval_task(agent, opp, t, a.games, a.minutes, seed=a.seed)
        rows[n] = r
        # No comparar una marca obtenida con otra duración, semilla, modo o rival.
        protocol = {"task": vars(t), "games": a.games, "minutes": a.minutes,
                    "seed": a.seed, "greedy": a.greedy, "opponent": opponent_identity(a.vs),
                    "sources": source_fingerprint([t])}
        key = protocol_id(protocol)
        r["protocol"] = protocol
        note = ""
        if a.best and key in best:
            if r["points"] < best[key] - a.drop:
                note = f"REGRESIÓN (mejor {best[key]:.2f})"
                regressions.append(n)
            elif r["points"] > best[key]:
                note = f"nuevo récord (antes {best[key]:.2f})"
        if a.best:
            best[key] = max(best.get(key, 0.0), r["points"])
        print(f"{n:12s} {t.stadium:15s} {t.n_per_team}v{t.n_per_team:<3d} {r['points']:7.2f} "
              f"{r['wins']:3d}-{r['draws']:2d}-{r['losses']:3d} {r['goals_for']:4d}-{r['goals_against']:<4d}  {note}", flush=True)
    pts = [r["points"] for r in rows.values()]
    print(f"\npromedio {sum(pts) / len(pts):.2f} | peor tarea {min(rows, key=lambda k: rows[k]['points'])} "
          f"({min(pts):.2f})" + (f" | regresiones: {', '.join(regressions)}" if regressions else ""))
    prev.setdefault("runs", []).append({"model": a.model, "vs": a.vs, "games": a.games, "minutes": a.minutes,
                                        "iteration": ck.get("iteration"), "steps": ck.get("steps"), "tasks": rows})
    prev["best"] = best
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(f"guardado en {out}")


if __name__ == "__main__":
    main()
