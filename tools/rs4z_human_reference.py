"""Referencia humana RS4-Z: las grabaciones reales medidas con `eval/rs4z/metrics.py` (mismas
definiciones que la simulación). Sólo tramos 4v4 con plantel estable; una muestra cada 3 ticks.

  python -m tools.rs4z_human_reference --out reports/rs4z/human_reference.json [--workers 6]
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from tools.rs4z_conformance import _read

ROOT = Path(__file__).resolve().parent.parent
REACH = 15.0 + 8.325 + 0.01


def restart_kinds(ticks, events):
    """frame → tipo de saque activo (1 lateral, 2 córner, 3 saque de arco, 0 ninguno), por el script."""
    kind, out = 0, {}
    ordered = sorted(events)
    j = 0
    for tick in ticks:
        fr = tick["frame"]
        while j < len(ordered) and ordered[j] <= fr:
            for e in events[ordered[j]]:
                if e["name"] == "disc_props" and not e.get("kind") and e.get("id") == 0:
                    d1 = e.get("data1") or [None] * 10
                    d2 = e.get("data2") or [None] * 3
                    if d1[0] is not None and d1[1] is not None:
                        x, y = abs(d1[0]), abs(d1[1])
                        if abs(y - 688.0) < 2.0:
                            kind = 1
                        elif abs(x - 1140.0) < 2.0 and abs(y - 660.0) < 2.0:
                            kind = 2
                        elif abs(x - 1030.0) < 2.0 and abs(y - 180.0) < 2.0:
                            kind = 3
                    if d2[0] is not None and d2[0] == 0xFFFFFF:
                        kind = 0
                elif e["name"] in ("positions_reset", "game_start"):
                    kind = 0
            j += 1
        out[fr] = kind
    return out


def measure(path):
    from eval.rs4z.metrics import Metrics
    ticks, events = _read(path)
    kinds = restart_kinds(ticks, events)
    kicks = collections.defaultdict(list)
    goals = {}
    for fr, evs in events.items():
        for e in evs:
            if e["name"] == "kick":
                kicks[fr].append(e.get("playerId"))
            elif e["name"] == "goal":
                goals[fr] = e
    m = Metrics(1)
    order = None
    window = []
    prev_kind = 0
    for t in ticks:
        bad = any(v is None for v in t["ball"][:4]) or any(v is None for p in t["players"] for v in p[4:8])
        if bad or len(t["players"]) != 8 or t["state"] not in (0, 1):
            order = None
            window = []
            continue
        ids = [p[0] for p in sorted(t["players"], key=lambda p: (p[1], p[0]))]
        if order != ids:
            order = ids
            m.prev_toucher[:] = -1
            m.has_prev_vel[:] = False
            window = []
        window.append(t)
        if len(window) < 3:
            continue
        slot = {pid: i for i, pid in enumerate(order)}
        toucher = -1
        kicked = np.zeros((1, 8), dtype=bool)
        goal = 0
        for w in window:
            bx, by = w["ball"][0], w["ball"][1]
            best = 1e9
            for p in w["players"]:
                d = np.hypot(p[4] - bx, p[5] - by)
                if d <= REACH and d < best:
                    best = d
                    toucher = slot[p[0]]
            for pid in kicks.get(w["frame"], []):
                if pid in slot:
                    kicked[0, slot[pid]] = True
                    toucher = slot[pid]
            if w["frame"] in goals:
                # equipo que anotó: el que tocó último (gol en contra incluido lo cuenta el árbitro de la sala)
                goal = 1 if w["ball"][0] > 0 else -1
        last = window[-1]
        players = np.zeros((1, 8, 2))
        vel = np.zeros((1, 8, 2))
        for p in last["players"]:
            players[0, slot[p[0]]] = (p[4], p[5])
            vel[0, slot[p[0]]] = (p[6], p[7])
        kind = kinds.get(last["frame"], 0)
        started = kind != 0 and prev_kind == 0
        prev_kind = kind
        in_play = last["state"] == 1 and kind == 0
        m.add(np.array([last["ball"][:2]]), players, vel, np.ones((1, 8), dtype=bool), np.array([toucher]), kicked,
              np.array([goal]), np.array([kind]), np.array([started]), np.array([in_play]))
        window = []
    return Path(path).parent.name, m.summary(), m.c["open_samples"].sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/rs4z/human_reference.json")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    files = sorted(glob.glob(str(ROOT / "data/rs4_jsonl/*/*.jsonl.gz")))
    rows = {}
    with Pool(args.workers) as pool:
        for name, summary, samples in pool.imap_unordered(measure, files):
            if samples > 2000:
                rows[name] = summary
    keys = sorted(next(iter(rows.values())).keys())
    agg = {}
    for k in keys:
        vals = np.array([r[k] for r in rows.values()], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if len(vals):
            agg[k] = dict(mean=float(vals.mean()), p10=float(np.percentile(vals, 10)), p50=float(np.median(vals)),
                          p90=float(np.percentile(vals, 90)))
    report = dict(version="RS4-Z-human-reference-1", recordings=len(rows), metrics=agg, per_recording=rows)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(dict(recordings=len(rows), metrics={k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in agg.items()}), indent=1))


if __name__ == "__main__":
    main()
