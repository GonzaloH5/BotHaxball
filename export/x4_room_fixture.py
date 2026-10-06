"""Fixture de paridad del bot de sala con observación v3: lo que ve la sala tick a tick y la obs de Python.

Toma una ventana continua 4v4 de una grabación del caché `tools.x4_ticks` (por defecto Sanguchito, con saque
inicial y saques) y escribe:
* `frames`: por tick, el cuadro público que recibe `deploy/rs4z/room_state.js` (posiciones, input e
  isKicking del registro, color de la pelota según el saque, masa, saque inicial) y `nextInput` de cada
  jugador (la entrada del registro siguiente: con retardo 0 es la decisión que el jugador "toma" viendo
  este cuadro, convención de `env/rs4z/obs_v3.build_samples`);
* `rows`: para cada decisión (cuadros múltiplos de 3) y lugar, la observación v3 de `learn.x4_data.featurize`
  con retardo 0.
`deploy/test_x4_room_state.js` reconstruye las observaciones en Node y las compara.

  python -m export.x4_room_fixture --out deploy/rs4z/fixture_x4_room_state.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
COLORS = {0: 0xFF0000, 1: 0x0000FF}
WHITE = 0xFFFFFF


def main():
    from learn import x4_data as XD
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recording", default="SanguREC-29-9-2026-18h12m.hbr2")
    ap.add_argument("--map", default="sanguchito_rs_x4")
    ap.add_argument("--ticks", type=int, default=2400)
    ap.add_argument("--out", default=str(ROOT / "deploy" / "rs4z" / "fixture_x4_room_state.json"))
    a = ap.parse_args()
    data = XD.load([a.recording], maps=(a.map,))
    T = data.ticks
    # ventana: empieza en un saque inicial y contiene saques
    ko_starts = np.flatnonzero((data.state == 0) & (np.r_[1, data.state[:-1]] != 0))
    valid = set(data.valid.tolist())
    start = None
    for s in ko_starts:
        rng = np.arange(s + XD.MAX_BACK, min(T - 3, s + a.ticks))
        if len(rng) > a.ticks // 2 and all(int(t) in valid for t in rng[::3][:50]) and (data.rkind[rng] > 0).any():
            start = int(s)
            break
    if start is None:
        raise SystemExit("no encontré una ventana con saque inicial y saques")
    first = start - XD.MAX_BACK if start >= XD.MAX_BACK else 0
    frames = []
    for t in range(first, min(T - 2, start + a.ticks)):
        color = COLORS[int(data.rteam[t])] if data.rkind[t] > 0 and data.rteam[t] >= 0 and data.state[t] == 1 else WHITE
        players = []
        for p in range(8):
            players.append(dict(id=int(data.pid[t, p]), team=0 if p < 4 else 1,
                                x=float(data.pos[t, p, 0]), y=float(data.pos[t, p, 1]),
                                vx=float(data.vel[t, p, 0]), vy=float(data.vel[t, p, 1]),
                                input=int(data.inp[t, p]), isKicking=bool(data.kicking[t, p]),
                                invMass=float(data.mass[t]) if data.mass[t] > 0 else 0.3,
                                nextInput=int(data.inp[t + 1, p])))
        frames.append(dict(frame=t, state=int(data.state[t]), kickoffTeam=int(max(0, data.ko_team[t])),
                           kickStrength=float(data.kstr[t]), stadiumName="SANGUCHITO RS X4",
                           ball=dict(x=float(data.ball[t, 0]), y=float(data.ball[t, 1]), vx=float(data.ball[t, 2]),
                                     vy=float(data.ball[t, 3]), r=float(data.ball_r[t]), color=color),
                           players=players))
    rows = []
    ts = [t for t in range(start, min(T - 2, start + a.ticks)) if t % 3 == 0 and t in valid]
    for t in ts:
        obs = XD.featurize(data, np.full(8, t + 1), np.arange(8), np.zeros(8, np.int64))
        for p in range(8):
            rows.append(dict(frame=t, slot=p, obs=[round(float(v), 6) for v in obs[p]]))
    out = dict(recording=a.recording, map=a.map, start=start, first=first, frames=frames, rows=rows)
    Path(a.out).write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    print(f"{len(frames)} cuadros, {len(rows)} observaciones -> {a.out}")


if __name__ == "__main__":
    main()
