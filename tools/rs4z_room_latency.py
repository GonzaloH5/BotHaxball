"""Latencia real de los bots en una sala: trazas de decisión de los bots × grabación del host.

Cada línea de traza (`deploy/bot.js --trace`) tiene el frame de la sala que el bot creía estar viendo, su
observación (posición propia y de la pelota) y la acción elegida. La grabación del host tiene, por frame,
la pelota, los jugadores y la entrada aplicada de cada uno. Por decisión:

* frame observado: el frame de la grabación cuyo estado coincide con lo que vio el bot (pelota + él mismo);
* frame aplicado: el primer frame en que el host aplica la tecla de esa acción para ese jugador;
* latencia total = aplicado − observado (ticks del host). Sólo cuenta decisiones que cambian de tecla.

  python -m tools.rs4z_room_latency --recording data/haxarg_jsonl/<rec>/<rec>.jsonl.gz \\
      --traces '<carpeta>/trace_*.jsonl' --out latency.json
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
from pathlib import Path

import numpy as np

from tools.rs4z_conformance import _read

SX, SY = 1150.0, 670.0
MOVE_DIRS = [(0, 0), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]
MIRROR_MOVE = [0, 1, 8, 7, 6, 5, 4, 3, 2]


def encode(action, team):
    """Acción de la política (marco propio) → bits de entrada de HaxBall (arriba 1, abajo 2, izq 4, der 8, patada 16)."""
    move = action % 9
    if team == 1:
        move = MIRROR_MOVE[move]
    dx, dy = MOVE_DIRS[move]
    bits = (1 if dy < 0 else 0) | (2 if dy > 0 else 0) | (4 if dx < 0 else 0) | (8 if dx > 0 else 0)
    return bits | (16 if action >= 9 else 0)


def load_trace(path):
    out = []
    for line in open(path, encoding="utf-8"):
        d = json.loads(line)
        sign = 1.0 if d["team"] == 0 else -1.0
        o = d["obs"]
        out.append(dict(frame=d["frame"], team=d["team"], action=d["action"],
                        self=(sign * o[0] * SX, o[1] * SY), ball=(sign * o[6] * SX, o[7] * SY)))
    return out


def analyze(recording, trace_paths, window=60):
    ticks, _ = _read(recording)
    frames = np.array([t["frame"] for t in ticks])
    ball = np.array([t["ball"][:2] for t in ticks])
    idx_of = {f: i for i, f in enumerate(frames)}
    traces = {Path(p).name: load_trace(p) for p in trace_paths}
    # desfase global sala → grabación: moda de (frame del bot − frame de la grabación con la misma pelota)
    cands = collections.Counter()
    for tr in traces.values():
        for d in tr[::25]:
            err = np.hypot(ball[:, 0] - d["ball"][0], ball[:, 1] - d["ball"][1])
            hits = np.flatnonzero(err < 0.5)
            if 0 < len(hits) <= 3:
                for h in hits:
                    cands[d["frame"] - int(frames[h])] += 1
    if not cands:
        raise SystemExit("no se pudo alinear la traza con la grabación")
    offset = cands.most_common(1)[0][0]
    report = dict(recording=Path(recording).name, offset=offset, bots={})
    for name, tr in traces.items():
        team = tr[0]["team"]
        # jugador de la grabación: el del mismo equipo más cercano a la posición propia observada
        votes = collections.Counter()
        obs_frame = []
        for d in tr:
            c = d["frame"] - offset
            lo, hi = c - window, c + window
            sl = [i for i in range(max(0, idx_of.get(lo, 0)), min(len(ticks), (idx_of.get(hi, len(ticks) - 1)) + 1))]
            if not sl:
                obs_frame.append(None)
                continue
            e = np.hypot(ball[sl, 0] - d["ball"][0], ball[sl, 1] - d["ball"][1])
            best = sl[int(np.argmin(e))]
            if e.min() > 1.0:
                obs_frame.append(None)
                continue
            obs_frame.append(best)
            t = ticks[best]
            mates = [(np.hypot(p[4] - d["self"][0], p[5] - d["self"][1]), p[0]) for p in t["players"] if p[1] - 1 == team]
            if mates:
                dist, pid = min(mates)
                if dist < 2.0:
                    votes[pid] += 1
        if not votes:
            continue
        pid = votes.most_common(1)[0][0]
        lags, obs_lag = [], []
        prev = None
        for d, k in zip(tr, obs_frame):
            if k is None:
                prev = d
                continue
            obs_lag.append(d["frame"] - offset - int(frames[k]))
            want = encode(d["action"], team)
            if prev is not None and encode(prev["action"], team) != want:
                for j in range(max(0, k - 2), min(len(ticks), k + 90)):
                    p = next((q for q in ticks[j]["players"] if q[0] == pid), None)
                    if p is None:
                        break
                    if p[2] == want:
                        lags.append(int(frames[j]) - int(frames[k]))
                        break
            prev = d
        lags = np.array(lags)
        # alineación por correlación: con retraso L, la tecla del host en el frame f es la de la última
        # decisión cuyo frame observado + L ≤ f. Se elige el L de máxima coincidencia (global y por ventana).
        dec = sorted((int(frames[k]), encode(d["action"], team)) for d, k in zip(tr, obs_frame) if k is not None)
        rec_in = {}
        for t in ticks:
            p = next((q for q in t["players"] if q[0] == pid), None)
            if p is not None:
                rec_in[t["frame"]] = p[2]
        dfr = np.array([a for a, _ in dec])
        dkey = np.array([b for _, b in dec])
        fr_all = np.array(sorted(rec_in))
        fr_all = fr_all[(fr_all > np.percentile(dfr, 1) + 45) & (fr_all < np.percentile(dfr, 99))]
        rin = np.array([rec_in[f] for f in fr_all])

        def agree(L, sel=slice(None)):
            j = np.searchsorted(dfr + L, fr_all[sel], side="right") - 1
            ok = j >= 0
            return float((dkey[j[ok]] == rin[sel][ok]).mean()) if ok.any() else 0.0

        curve = {L: agree(L) for L in range(0, 41)}
        best = max(curve, key=curve.get)
        win = []
        for a in range(0, len(fr_all), 1800):
            sel = slice(a, min(a + 1800, len(fr_all)))
            c = {L: agree(L, sel) for L in range(0, 41)}
            bl = max(c, key=c.get)
            win.append((int(fr_all[a]), bl, round(c[bl], 3)))
        report["bots"][name] = dict(corr_best_lag=best, corr_agree=round(curve[best], 3),
                                    corr_curve={L: round(v, 3) for L, v in curve.items()}, corr_windows=win,player_id=pid, decisions=len(tr), matched=int(sum(k is not None for k in obs_frame)),
                                    changes=int(len(lags)),
                                    total_lag=dict(collections.Counter(lags.tolist())),
                                    total_p10_50_90=np.percentile(lags, [10, 50, 90]).tolist() if len(lags) else None,
                                    view_lag=dict(collections.Counter(obs_lag)))
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recording", required=True)
    ap.add_argument("--traces", required=True, help="glob de trazas de los bots")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rep = analyze(args.recording, sorted(glob.glob(args.traces)))
    Path(args.out).write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in rep.items() if k != "bots"}))
    for name, b in rep["bots"].items():
        print(name, "id", b["player_id"], "decisiones", b["decisions"], "alineadas", b["matched"], "cambios", b["changes"],
              "latencia p10/p50/p90", b["total_p10_50_90"], "| correlación: L*", b["corr_best_lag"], "acuerdo",
              b["corr_agree"], "ventanas (frame, L*, acuerdo):", b["corr_windows"])


if __name__ == "__main__":
    main()
