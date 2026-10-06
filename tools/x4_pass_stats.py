"""Pases y posesión agregados, con intervalos de confianza: humanos (por partición) contra una política en self-play.

`tools.x4_metrics` compara distribuciones de valores por tramo; con pocos partidos simulados las medianas por tramo son
ruidosas. Acá se suman eventos y minutos (tasa agregada = Σ eventos / Σ minutos) y el intervalo del 90% sale de un
bootstrap por unidad independiente (grabación para humanos, partido para la simulación). Mismas definiciones de pase,
pérdida y posesión (`tools.x4_metrics.possession_sequence`) y mismo muestreo cada 3 ticks.

  python -m tools.x4_pass_stats --policy runs/x4_bc/final_sangu_rsone/best.pt --matches 24 --minutes 3 \\
      --out reports/x4/pass_stats_bc_final.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tools import x4_metrics as XM

ROOT = Path(__file__).resolve().parent.parent


def unit_counts(episodes):
    """Totales de un conjunto de tramos (una unidad de bootstrap)."""
    tot = dict(minutes=0.0, open_minutes=0.0, passes=0, losses=0, goals=0, kicks=0.0, possessions=0, pass_len=[])
    for ep in episodes:
        c = XM.episode_counts(ep)
        for k in ("minutes", "open_minutes", "passes", "losses", "goals", "kicks", "possessions"):
            tot[k] += c[k]
        tot["pass_len"] += c["pass_len"]
    return tot


def rates(units):
    m = sum(u["minutes"] for u in units)
    om = sum(u["open_minutes"] for u in units)
    p = sum(u["passes"] for u in units)
    l = sum(u["losses"] for u in units)
    poss = sum(u["possessions"] for u in units)
    lens = [x for u in units for x in u["pass_len"]]
    return dict(
        pases_por_min=p / m if m else float("nan"),
        pases_por_min_juego_abierto=p / om if om else float("nan"),
        perdidas_por_min=l / m if m else float("nan"),
        pases_sobre_pases_mas_perdidas=p / (p + l) if p + l else float("nan"),
        pases_por_posesion=p / poss if poss else float("nan"),
        patadas_por_min_y_jugador=sum(u["kicks"] for u in units) / 8 / m if m else float("nan"),
        goles_por_min=sum(u["goals"] for u in units) / m if m else float("nan"),
        largo_de_pase_p50=float(np.median(lens)) if lens else float("nan"),
    )


def bootstrap(units, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    point = rates(units)
    draws = {k: [] for k in point}
    for _ in range(n):
        pick = [units[i] for i in rng.integers(0, len(units), len(units))]
        r = rates(pick)
        for k, v in r.items():
            draws[k].append(v)
    return {k: dict(valor=round(point[k], 4), ic90=[round(float(np.nanpercentile(draws[k], 5)), 4),
                                                     round(float(np.nanpercentile(draws[k], 95)), 4)])
            for k in point}


def human_units(split, map_name="sanguchito_rs_x4"):
    from tools.x4_ticks import MAP_IDS
    rows = json.loads((ROOT / "reports" / "x4" / "splits.json").read_text(encoding="utf-8"))["recordings"]
    units = []
    for name, r in sorted(rows.items()):
        if r["split"] != split or map_name not in (r.get("maps") or {}):
            continue
        path = ROOT / "data" / "x4_ticks" / f"{Path(name).stem}.npz"
        if path.exists():
            eps = XM.episodes_from_ticks(str(path), map_id=MAP_IDS[map_name])
            if eps:
                units.append(unit_counts(eps))
    return units


def policy_units(spec, matches, minutes, seed, threads):
    import torch
    from learn.x4_eval import Policy, play
    torch.set_num_threads(threads)
    pol = Policy(spec)
    r = play(pol, pol, map_name="sanguchito_rs_x4", matches=matches, minutes=minutes, seed=seed, record=matches)
    return [unit_counts([ep]) for ep in r["episodes"]], dict(safety=int(r["safety"].sum()), finished=r["finished"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", action="append", default=[], help="checkpoint (se puede repetir)")
    ap.add_argument("--matches", type=int, default=24)
    ap.add_argument("--minutes", type=float, default=3.0)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--splits", default="train,test")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    report = {}
    for sp in a.splits.split(","):
        u = human_units(sp)
        report[f"humanos_{sp}"] = dict(unidades=len(u), minutos=round(sum(x["minutes"] for x in u), 1), **bootstrap(u))
        print(f"humanos {sp}: {len(u)} grabaciones", flush=True)
    for spec in a.policy:
        u, extra = policy_units(spec, a.matches, a.minutes, seed=11, threads=a.threads)
        report[spec] = dict(unidades=len(u), minutos=round(sum(x["minutes"] for x in u), 1), **extra, **bootstrap(u))
        print(f"{spec}: {len(u)} partidos", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    keys = list(next(iter(report.values())).keys())
    for k in [k for k in keys if isinstance(report[next(iter(report))][k], dict)]:
        print(f"{k:32s} " + "  ".join(f"{name[:22]:>22s}: {v[k]['valor']:7.3f} [{v[k]['ic90'][0]:.3f}, {v[k]['ic90'][1]:.3f}]"
                                         for name, v in report.items()))


if __name__ == "__main__":
    main()
