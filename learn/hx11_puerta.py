"""Puerta P3 de L0 por puesto (revisión de la auditoría): lazo cerrado no reactivo en partidos de PRUEBA.

Dos juegos de ventanas de 5 s:
* **cerca:** el jugador de campo más cercano a la pelota (¿interviene como un humano de su puesto?);
* **azar:** un jugador cualquiera (¿se ubica como un humano de su puesto, también lejos de la jugada?).

Para cada línea del puesto (ARQ, DEF, MCD, MED, DEL) y cada métrica, el bot y el humano se miden en las mismas
ventanas. Las tasas son totales sobre el tiempo total; las distancias, promedios. Se informa la razón bot/humano con su
IC 90% por bootstrap de ventanas.

**Criterio (fijado antes de mirar):** la métrica pasa si el IC 90% de la razón se superpone con [0,75; 1,33]. El puesto
pasa si pasan todas sus métricas. Es un criterio de diagnóstico para decidir el piloto en sala (C0), no una prueba de
nivel humano.

  python -m learn.hx11_puerta --model runs/hx11_l0/l0_sinhist_full/model.pt [--temp 0.7]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from learn import hx11_closed_loop as CL
from learn.hx11_data import l0_split, load_match
from learn.hx11_policy import SetPolicy

ROOT = Path(__file__).resolve().parent.parent
LO, HI = 0.75, 1.33
METRICS = {"cerca": ("patadas_efectivas_min", "toques_min", "vel", "dist_pelota"),
           "azar": ("dist_ancla", "dist_pelota", "vel", "toques_min")}
RATES = ("patadas_efectivas_min", "toques_min")


def stat(rows, key, who):
    T = np.array([r["ticks"] for r in rows], np.float64)
    v = np.array([r[f"{key}_{who}"] if r.get(f"{key}_{who}") is not None else np.nan for r in rows], np.float64)
    ok = np.isfinite(v)
    if key in RATES:
        return float((v[ok] * T[ok]).sum() / max(T[ok].sum(), 1))
    return float(np.nanmean(v))


def ratio_ci(rows, key, rng, n=500):
    base = stat(rows, key, "bot"), stat(rows, key, "humano")
    rs = []
    for _ in range(n):
        idx = rng.integers(0, len(rows), len(rows))
        sub = [rows[i] for i in idx]
        h = stat(sub, key, "humano")
        if h > 0:
            rs.append(stat(sub, key, "bot") / h)
    lo, hi = (np.percentile(rs, [5, 95]) if rs else (np.nan, np.nan))
    return base, (base[0] / base[1] if base[1] > 0 else np.nan), (float(lo), float(hi))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--every", type=int, default=CL.FRAME_SKIP)
    ap.add_argument("--matches", type=int, default=20)
    ap.add_argument("--cerca", type=int, default=1600)
    ap.add_argument("--azar", type=int, default=1600)
    ap.add_argument("--out", default="reports/hx11/l0_puerta")
    a = ap.parse_args()
    torch.set_num_threads(6)
    modos = json.loads((ROOT / "data" / "hx11" / "modos.json").read_text(encoding="utf-8"))
    rng = np.random.default_rng(0)
    names = sorted(rng.permutation(l0_split()["test"])[:a.matches].tolist())
    matches = [load_match(n, modos) for n in names]
    owners = [np.load(ROOT / "data" / "hx11" / "d1" / f"{n}.npz")["owner"] for n in names]
    ck = torch.load(a.model, map_location="cpu", weights_only=False)
    net = SetPolicy(**ck["config"])
    net.load_state_dict(ck["state"])
    net.eval()
    net.hold = False
    res = {}
    for kind, n in (("cerca", a.cerca), ("azar", a.azar)):
        rng = np.random.default_rng(1 if kind == "cerca" else 2)
        per = max(1, n // len(matches))
        sl = [(mi, t0, s) for mi, m in enumerate(matches)
              for t0, s in CL.starts(m, owners[mi], per, rng, near=kind == "cerca")]
        rows = CL.rollout(net, matches, owners, sl, n_env=64, seed=0, temp_move=a.temp, every=a.every)
        res[kind] = {}
        for line in ("ARQ", "DEF", "MCD", "MED", "DEL"):
            rr = [r for r in rows if r.get("linea") == line]
            if len(rr) < 40:
                continue
            res[kind][line] = {"ventanas": len(rr)}
            for key in METRICS[kind]:
                (b, h), r, (lo, hi) = ratio_ci(rr, key, np.random.default_rng(3))
                res[kind][line][key] = dict(bot=round(b, 3), humano=round(h, 3), razon=round(r, 3),
                                            ic90=[round(lo, 3), round(hi, 3)], pasa=bool(hi >= LO and lo <= HI))
        print(kind, json.dumps({l: {k: (v["razon"], v["pasa"]) for k, v in d.items() if k != "ventanas"}
                                for l, d in res[kind].items()}, ensure_ascii=False), flush=True)
    Path(a.out + ".json").write_text(json.dumps(dict(modelo=a.model, temp=a.temp, cada=a.every, partidos=names,
                                                     resultado=res), indent=1), encoding="utf-8")
    md = ["# Puerta P3 de L0 por puesto (partidos de prueba)", "",
          f"Con [`learn/hx11_puerta.py`](../../learn/hx11_puerta.py). Modelo `{a.model}`, temperatura {a.temp}, "
          f"decisión cada {a.every} ticks, {len(names)} partidos de prueba. Razón bot/humano en las mismas ventanas, "
          f"con IC 90%; **pasa** si el IC se superpone con [{LO}; {HI}].", ""]
    for kind, d in res.items():
        md += [f"## Ventanas {'cerca de la pelota' if kind == 'cerca' else 'al azar'}", "",
               "| Línea | Ventanas | Métrica | Bot | Humano | Razón | IC 90% | ¿Pasa? |", "|---|---|---|---|---|---|---|---|"]
        for line, m in d.items():
            for key, v in m.items():
                if key == "ventanas":
                    continue
                md.append(f"| {line} | {m['ventanas']} | {key} | {v['bot']} | {v['humano']} | {v['razon']} | "
                          f"{v['ic90'][0]}–{v['ic90'][1]} | {'sí' if v['pasa'] else '**no**'} |")
        md.append("")
    Path(a.out + ".md").write_text("\n".join(md) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
