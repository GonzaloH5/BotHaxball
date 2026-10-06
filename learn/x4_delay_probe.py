"""Retardo efectivo de los humanos: verosimilitud de sus acciones según el retardo de la observación.

Un modelo de imitación entrenado con retardo aleatorio (y el retardo como feature) se evalúa con el estado
de t − d para varios d. Con d menor que el retardo real del humano el modelo ve información posterior a
la que el humano usó (casi un superconjunto: la NLL se mantiene plana); con d mayor le falta información
que el humano sí tenía (la NLL sube). El codo de la curva estima el retardo efectivo (ping + reacción;
plan E1, "latencia de los humanos").

Por jugador (grabación × id, con ≥ `--min-samples` muestras de cambio de tecla) se toma el mayor d cuya NLL
no supera el mínimo + `--eps`.

  python -m learn.x4_delay_probe --ckpt runs/x4_bc/probe_d0_24/best.pt --out reports/x4/human_delay.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from learn import x4_data as XD
from learn.x4_policy import SetPolicy

ROOT = Path(__file__).resolve().parent.parent


@torch.no_grad()
def nll_matrix(model, data, t, p, delays, batch=8192):
    out = np.zeros((len(delays), len(t)), np.float32)
    for k, d in enumerate(delays):
        for i in range(0, len(t), batch):
            tt, pp = t[i:i + batch], p[i:i + batch]
            obs = torch.from_numpy(XD.featurize(data, tt, pp, np.full(len(tt), d)))
            y = torch.from_numpy(data.label[tt, pp].astype(np.int64))
            out[k, i:i + batch] = F.cross_entropy(model(obs), y, reduction="none").numpy()
    return out


def knee(curve, delays, eps):
    ok = np.flatnonzero(curve <= curve.min() + eps)
    return int(delays[ok.max()])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--splits", default=str(ROOT / "reports" / "x4" / "splits.json"))
    ap.add_argument("--split", default="dev")
    ap.add_argument("--maps", default="sanguchito_rs_x4")
    ap.add_argument("--delays", default="0,3,6,9,12,15,18,21,24")
    ap.add_argument("--n", type=int, default=200000)
    ap.add_argument("--eps", type=float, default=0.01)
    ap.add_argument("--min-samples", type=int, default=150)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    maps = tuple(a.maps.split(","))
    data = XD.load(XD.split_names(a.splits, a.split, None, maps), maps=maps)
    ck = torch.load(a.ckpt, map_location="cpu")
    model = SetPolicy(hidden=ck.get("hidden", 256))
    model.load_state_dict(ck["model"])
    model.eval()
    delays = np.array([int(x) for x in a.delays.split(",")])
    t, p = XD.Sampler(data).fixed(a.n, seed=7)
    changed = data.label[t, p] != data.label[t - 3, p]
    M = nll_matrix(model, data, t, p, delays)
    pop = {int(d): dict(nll=float(M[k].mean()), nll_change=float(M[k, changed].mean()),
                        nll_hold=float(M[k, ~changed].mean())) for k, d in enumerate(delays)}
    curve_change = M[:, changed].mean(1)
    rec = data.rec_of_tick[t]
    pid = data.pid[t, p]
    players = {}
    for key in set(zip(rec[changed].tolist(), pid[changed].tolist())):
        m = changed & (rec == key[0]) & (pid == key[1])
        if m.sum() < a.min_samples:
            continue
        c = M[:, m].mean(1)
        players[f"{data.names[key[0]]}#{key[1]}"] = dict(n=int(m.sum()), knee=knee(c, delays, a.eps),
                                                        curve=[round(float(x), 4) for x in c])
    knees = np.array([v["knee"] for v in players.values()]) if players else np.array([])
    report = dict(version="x4-delay-probe-1", ckpt=a.ckpt, split=a.split, maps=maps, samples=int(len(t)),
                  frac_change=float(changed.mean()), delays=delays.tolist(), population=pop,
                  population_knee_change=knee(curve_change, delays, a.eps), eps=a.eps,
                  players_knee=dict(n=int(len(knees)), hist={int(d): int((knees == d).sum()) for d in delays},
                                    p25=float(np.percentile(knees, 25)) if len(knees) else None,
                                    p50=float(np.percentile(knees, 50)) if len(knees) else None,
                                    p75=float(np.percentile(knees, 75)) if len(knees) else None),
                  players=players)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "players"}, indent=1))


if __name__ == "__main__":
    main()
