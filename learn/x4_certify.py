"""Certificación de juego colectivo de un checkpoint contra humanos reales (gate final antes de llamarlo competitivo).

Criterio pre-registrado (docs/PRELANZAMIENTO.md §Gates):
1. Cadena de pase en self-play (`--matches` partidos de `--minutes`, latencia de sala) contra la referencia humana de
   Sanguchito (partición de prueba, `reports/x4/pass_chain_human.json`): todas las métricas de eficacia ≥ promedio humano
   y todas las de estilo dentro de p10–p90 humano (`tools.x4_pass_chain.gate`). Se informa también contra la liga
   HAXARG de RS ONE (equipos de torneo, misma cancha y física), que es la vara de "jugador competente".
2. Fuerza: no pierde contra la imitación humana (`vs_bc.score` ≥ 0,5 con `--strength-matches` por lado).
3. Parecido humano: W1 normalizada media ≤ 1 en las métricas clave (como en el trainer).
4. Informativo: sondas 2v1 y 3v2 desde estados humanos (la política juega en inferioridad numérica reducida y los
   humanos, desde el mismo estado, 4v4: no es una comparación directa) y el gate contra la liga.

Sale con código 0 si aprueba todo, 1 si no. El reporte deja cada métrica con su intervalo y la razón agente/humano.

  python -m learn.x4_certify --ckpt runs/x4_ppo/lam02/best_pase.pt --device cuda --out reports/x4/cert_best_pase.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--bc", default=str(ROOT / "runs" / "x4_bc" / "final_sangu_rsone" / "best.pt"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--matches", type=int, default=64)
    ap.add_argument("--minutes", type=float, default=3.0)
    ap.add_argument("--strength-matches", type=int, default=64)
    ap.add_argument("--probe-n", type=int, default=256)
    ap.add_argument("--epv", default=str(ROOT / "runs" / "x4_epv" / "epv.pt"))
    ap.add_argument("--reference", default=str(ROOT / "reports" / "x4" / "pass_chain_human.json"))
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20261006)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    from learn.x4_epv import EPV
    from learn.x4_eval import Policy, human_compare, play
    from tools import x4_pass_chain as PC
    epv = EPV(a.epv, a.device)
    ref = json.loads(Path(a.reference).read_text(encoding="utf-8"))
    me = Policy(a.ckpt, device=a.device)
    bc = Policy(a.bc, device=a.device)
    report = dict(ckpt=a.ckpt, matches=a.matches, minutes=a.minutes)

    # 1. cadena de pase en self-play
    units, r = PC.policy_units(me, None, a.matches, a.minutes, seed=a.seed, epv=epv)
    m = PC.bootstrap(units, n=1000, seed=a.seed)
    report["cadena_pase"] = dict(metricas=m, safety=int(r["safety"].sum()))
    report["gate_sanguchito"] = PC.gate(m, ref["sanguchito_test"]["metricas"], ref["sanguchito_test"]["banda"])
    if "liga_rs_one" in ref:
        report["gate_liga"] = PC.gate(m, ref["liga_rs_one"]["metricas"], ref["liga_rs_one"]["banda"])
    hum = human_compare(r["episodes"]) or {}
    key = ("passes_per_min", "possession_s", "pass_length", "depth", "width", "dist_ball_2", "still_frac",
           "key_changes_per_s", "kickoff_wait_s", "restart_s_lateral")
    w1 = hum.get("w1_norm", {})
    vals = [w1[k] for k in key if w1.get(k) is not None]
    report["human_w1_mean"] = round(float(np.mean(vals)), 3) if vals else None

    # 2. fuerza contra la BC (los dos lados)
    w = d = l = 0
    for side, (red, blue) in enumerate(((me, bc), (bc, me))):
        rr = play(red, blue, matches=a.strength_matches, minutes=a.minutes, seed=a.seed + 1 + side, record=0)
        g = rr["goals"] if side == 0 else rr["goals"][:, ::-1]
        diff = g[:, 0] - g[:, 1]
        w += int((diff > 0).sum()); d += int((diff == 0).sum()); l += int((diff < 0).sum())
    report["vs_bc"] = dict(wins=w, draws=d, losses=l, score=round((w + 0.5 * d) / max(1, w + d + l), 3))

    # 4. sondas (informativas)
    from learn import x4_data as XD
    from learn import x4_probes as PR
    names = XD.split_names(ROOT / "reports" / "x4" / "splits.json", "test", None, ("sanguchito_rs_x4",))
    data = XD.load(names[:40], maps=("sanguchito_rs_x4",))
    rng = np.random.default_rng(0)
    probes = {}
    for name, (na, nd) in (("2v1", (2, 1)), ("3v2", (3, 2))):
        states = PR.pick_states(data, na, nd, a.probe_n, rng)
        probes[name] = PR.run(me.model, data, states, seed=1 + na)
        probes[name + "_humanos"] = PR.human_baseline(data, states)
    report["sondas"] = probes

    checks = dict(
        cadena_pase_vs_sanguchito=report["gate_sanguchito"]["aprobado"],
        fuerza_vs_bc=report["vs_bc"]["score"] >= 0.5,
        parecido_humano=report["human_w1_mean"] is not None and report["human_w1_mean"] <= 1.0,
    )
    report["checks"] = checks
    # las sondas son informativas: la política juega 2v1/3v2 y los humanos, desde el mismo estado, 4v4
    report["informativo"] = dict(cadena_pase_vs_liga=report.get("gate_liga", {}).get("aprobado"),
                                 indice_vs_liga=report.get("gate_liga", {}).get("indice_cadena"),
                                 sonda_2v1=[probes["2v1"]["tasa_pase"], probes["2v1_humanos"]["tasa_pase"]],
                                 sonda_3v2=[probes["3v2"]["tasa_pase"], probes["3v2_humanos"]["tasa_pase"]])
    report["aprobado"] = all(checks.values())
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    g = report["gate_sanguchito"]
    print(json.dumps(dict(aprobado=report["aprobado"], checks=checks, indice_cadena=g["indice_cadena"],
                          por_etapa=g["por_etapa"], fallan=[k for k, v in g["metricas"].items() if not v["ok"]],
                          vs_bc=report["vs_bc"], **report["informativo"]), ensure_ascii=False, indent=1))
    sys.exit(0 if report["aprobado"] else 1)


if __name__ == "__main__":
    main()
