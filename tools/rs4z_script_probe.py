"""Mediciones del script de la sala en las grabaciones (RS ONE, 2K23, Sanguchito), para fijar el contrato.

Por saque (pelota coloreada y colocada → pelota blanca):
* demora del cobro: ticks entre la salida de la pelota y su colocación;
* máscara de la pelota (cMask con c0) al empezar y cuándo vuelve a 63 respecto de la liberación;
* liberación del lateral: |y| de la pelota en el tick de liberación y en el anterior;
* curva tras la patada: g0 contra la velocidad del pateador; ticks que se mantiene; decaimiento por tick;
  duración; si un toque la corta (gravedad a 0 antes del final con un jugador a distancia de toque).

  python -m tools.rs4z_script_probe --data data/rs4_jsonl --pattern 'HBReplay*' --map rs_one --out probe_rsone.json
"""
from __future__ import annotations

import argparse
import collections
import functools
import glob
import json
import math
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from tools.rs4z_conformance import _read, in_spans, stadium_frames
from tools.rs4z_restart_conformance import infer_taker, real_restarts

ROOT = Path(__file__).resolve().parent.parent
C0 = 1 << 28


def _props(e):
    f1 = ["x", "y", "vx", "vy", "gx", "gy", "r", "bC", "im", "dmp"]
    f2 = ["col", "cM", "cG"]
    d = {f1[i]: v for i, v in enumerate(e.get("data1") or []) if v is not None}
    d.update({f2[i]: v for i, v in enumerate(e.get("data2") or []) if v is not None})
    return d


def probe_file(path, map_name):
    from env.rs4z import contract as C
    prm = C.params(map_name)
    pi = C.PI
    ticks, events = _read(path)
    spans = stadium_frames(path, map_name)
    ticks = [t for t in ticks if in_spans(t["frame"], spans)]
    bf = {t["frame"]: t for t in ticks}
    out = []
    restarts = real_restarts(ticks, events)
    for r in restarts:
        if r["end"] is None or r["start"] not in bf:
            continue
        x, y = r["spot"]
        kind = (1 if abs(abs(y) - prm[pi["lateral_ball_y"]]) < 1 else 3 if abs(abs(x) - prm[pi["goal_kick_x"]]) < 1 else 2)
        team_of = {p[0]: p[1] for p in bf[r["start"]]["players"]}
        taker = infer_taker(r, kind, events, team_of)
        row = dict(kind=kind, start=r["start"], end=r["end"][0], dur=r["end"][1] - r["start"])
        # demora del cobro
        rb = bf[r["start"]]["ball"][4]
        lw, lh = C.LINE_W, prm[pi["line_half_h"]]
        f = r["start"]
        while f - 1 in bf:
            bx, by = bf[f - 1]["ball"][:2]
            if abs(bx) <= lw + rb and abs(by) <= lh + rb:
                break
            f -= 1
        row["delay"] = r["start"] - f
        # máscara de la pelota al empezar
        row["mask_c0"] = any(e["name"] == "disc_props" and not e.get("kind") and e.get("id") == 0
                             and (_props(e).get("cM") or 0) & C0 for e in events.get(r["start"], []))
        if r["end"][0] != "released":
            out.append(row)
            continue
        rel = r["end"][1]
        if kind == 1 and rel in bf and rel - 1 in bf:
            row["lat_rel_y"] = abs(bf[rel]["ball"][1])
            row["lat_prev_y"] = abs(bf[rel - 1]["ball"][1])
        # vuelta de la máscara
        for fr in range(rel, rel + 400):
            if any(e["name"] == "disc_props" and not e.get("kind") and e.get("id") == 0 and _props(e).get("cM") == 63
                   for e in events.get(fr, [])):
                row["mask_restore"] = fr - rel
                break
        # curva: g0 y la velocidad del pateador antes del tick de la patada
        kick = [e for e in events.get(rel, []) if e["name"] == "kick"] or [e for e in events.get(rel - 1, []) if e["name"] == "kick"]
        if kick and kind != 1 and rel in bf:
            kf = kick[0]["frame"]
            pl = [p for p in bf[kf]["players"] if p[0] == kick[0]["playerId"]] if kf in bf else []
            if pl:
                row["kv"] = (pl[0][6], pl[0][7])
                row["kicker_team_ok"] = taker is None or (pl[0][1] - 1) == taker
            series = []
            for fr in range(rel, rel + 200):
                if fr in bf:
                    series.append((fr - rel, bf[fr]["ball"][6], bf[fr]["ball"][7]))
            row["grav"] = series
            # contacto: primer tick con un jugador a distancia de toque
            for fr in range(rel + 2, rel + 200):
                t = bf.get(fr)
                if t is None:
                    break
                bx, by = t["ball"][:2]
                if any(math.hypot(p[4] - bx, p[5] - by) <= 15 + rb + 0.01 for p in t["players"]):
                    row["touch"] = fr - rel
                    break
            row["speeds"] = [float(np.hypot(*bf[kf + k]["ball"][2:4])) for k in range(0, 6) if kf + k in bf]
        out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/rs4_jsonl")
    ap.add_argument("--pattern", default="*")
    ap.add_argument("--map", default="rs_one")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    files = sorted(glob.glob(str(ROOT / args.data / args.pattern / "*.jsonl.gz")))
    rows = []
    with Pool(args.workers) as pool:
        for r in pool.imap_unordered(functools.partial(probe_file, map_name=args.map), files):
            rows += r
    Path(args.out).write_text(json.dumps(rows), encoding="utf-8")
    print(len(files), "archivos,", len(rows), "saques ->", args.out)


if __name__ == "__main__":
    main()
