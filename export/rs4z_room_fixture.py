"""Fixture de paridad del bot de sala RS4-Z: lo que ve la sala tick a tick y la observación del simulador.

Toma una ventana continua de una grabación real (con saques y saque inicial) y escribe, por tick, el
cuadro público que recibiría `deploy/rs4z/room_state.js` (posiciones, input e isKicking del tick
anterior, color de la pelota, fase de masa, saque inicial) y, para cada decisión y lugar, la observación
v2 que calcula `tools/rs4z_bc_dataset.extract` cargando ese estado en el simulador. La prueba
`deploy/test_rs4z_room_state.js` reconstruye las observaciones en Node y las compara.

  python -m export.rs4z_room_fixture --out deploy/rs4z/fixture_room_state.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT = "data/rs4_jsonl/HBReplay-2026-05-03-22h26m-69f7fe8c03a974.47171323/replay_HBReplay-2026-05-03-22h26m-69f7fe8c03a974.47171323.jsonl.gz"


def main():
    from tools.rs4_rules_audit import load
    from tools.rs4z_bc_dataset import _kick_strength, extract
    from tools.rs4z_conformance import mass_phases
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recording", default=DEFAULT)
    ap.add_argument("--start", type=int, default=0, help="primer cuadro (0: elegir uno con saque inicial)")
    ap.add_argument("--ticks", type=int, default=5400)
    ap.add_argument("--out", default="deploy/rs4z/fixture_room_state.json")
    a = ap.parse_args()
    path = ROOT / a.recording
    header, ticks, events = load(path)
    by = {t["frame"]: t for t in ticks}
    obs, act, meta = extract(str(path), 0)
    phase = mass_phases([dict(frame=t["frame"]) for t in ticks], events)
    ks = _kick_strength(path.parent) or 5.85
    color = {}
    current = 0xFFFFFF
    for t in ticks:
        for e in events.get(t["frame"], []):
            if e["name"] == "disc_props" and not e.get("kind") and e.get("id") == 0:
                d2 = e.get("data2") or [None] * 3
                if d2[0] is not None:
                    current = d2[0]
        color[t["frame"]] = current
    start = a.start
    if not start:
        # primer saque inicial después de un gol, con margen previo
        resets = sorted(f for f, evs in events.items() if any(e["name"] == "goal" for e in evs))
        start = max(1, resets[0] - 1500) if resets else ticks[0]["frame"]
    frames = []
    for f in range(start, start + a.ticks):
        t, prev = by.get(f), by.get(f - 1)
        if t is None or prev is None:
            continue
        prev_by = {p["id"]: p for p in prev["players"]}
        discs = t["discs"]
        ball = discs[0]
        m = phase.get(f)
        inv = 0.5 if m == 0.5 else 0.3
        players = []
        for p in t["players"]:
            if p["team"] not in (1, 2):
                continue
            d = discs[p["disc"]]
            q = prev_by.get(p["id"], p)
            players.append(dict(id=p["id"], team=p["team"] - 1, x=d["x"], y=d["y"], vx=d["vx"], vy=d["vy"],
                                input=q["input"], isKicking=bool(q["kicking"]), invMass=inv, nowInput=p["input"]))
        frames.append(dict(frame=f, state=t["state"], kickoffTeam=int(t.get("ko") or 1) - 1, kickStrength=ks,
                           ball=dict(x=ball["x"], y=ball["y"], vx=ball["vx"], vy=ball["vy"], r=ball["r"],
                                     color=color[f]),
                           players=players))
    sel = (meta[:, 1] >= start + 60) & (meta[:, 1] < start + a.ticks)
    rows = [dict(frame=int(mm[1]), slot=int(mm[2]), state=int(mm[3]), obs=[round(float(v), 6) for v in o])
            for o, mm in zip(obs[sel].astype(np.float32), meta[sel])]
    out = dict(recording=a.recording, start=start, ticks=a.ticks, frames=frames, rows=rows,
               note="input/isKicking de cada jugador son los del tick anterior (lo que la sala ya aplicó); "
                    "nowInput es el del cuadro (para el historial propio en decisiones)")
    Path(a.out).write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    kinds = np.bincount(meta[sel, 3], minlength=5)
    print(f"{len(frames)} cuadros, {len(rows)} filas; estados abierto/lateral/córner/arco/inicial = {kinds.tolist()}")


if __name__ == "__main__":
    main()
