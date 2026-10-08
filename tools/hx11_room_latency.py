"""Latencia real de los bots en una sala HaxEleven (docs/PLAN_HAXELEVEN.md, S6): trazas × grabación del host.

Cada línea de traza de `bridge/hx11_probe.js` con una tecla nueva trae el frame del cliente, la posición propia y la de
la pelota que el bot veía al decidir. La grabación del host (convertida con `tools.hx11_npz`) tiene por tick la verdad.
Por decisión:

* tick observado: el tick de la grabación cuyo estado coincide con lo que vio el bot (posición propia + pelota),
  buscado cerca del frame del cliente con el desfase cliente→grabación estimado en las primeras coincidencias;
* tick aplicado: el primer tick, desde el observado, en que el host aplica esa tecla para ese jugador;
* latencia = aplicado − observado, en ticks del host. También se informa el eco que vio el bot (lo que mide
  `bridge/latency_probe.js`, que subestima: el cliente adelanta su reloj).

  python -m tools.hx11_room_latency --recording data/hx11/room/<grabación> --meta data/hx11/room/meta.jsonl \\
      --traces 'runs/hx11_room/<sesión>/trace_*.jsonl' --out reports/hx11/s6_latencia.json
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
from pathlib import Path

import numpy as np

from tools.hx11_npz import load

MATCH_TOL = 0.5      # px: suma de errores de posición propia y de la pelota para aceptar una coincidencia


def bot_traces(paths):
    """{nombre del bot: [registros con tecla]} (la traza se identifica por su línea "joined" y el nombre del archivo)."""
    out = collections.defaultdict(list)
    for p in paths:
        for line in open(p, encoding="utf-8"):
            o = json.loads(line)
            if "key" in o and o.get("me") and o.get("ball"):
                out[Path(p).stem.removeprefix("trace_")].append(o)
    return out


def latency(rec_base, meta_path, trace_paths, window=300):
    z, events, header = load(rec_base)
    meta = json.loads(Path(meta_path).read_text(encoding="utf-8").splitlines()[0])
    frames = z["frame"].astype(np.int64)
    pid = list(z["pid"])
    names = {int(k): v for k, v in meta["names"].items()}
    res = {}
    for bot, recs in bot_traces(trace_paths).items():
        want = {"lat1": "[BOT] lat-1", "lat2": "[BOT] lat-2", "holder": "[BOT] holder", "presser": "[BOT] presser"}.get(bot, bot)
        cols = [pid.index(i) for i, n in names.items() if n == want and i in pid]
        if not cols:
            continue
        samples, offsets = [], []
        for r in recs:
            me, ball = np.array(r["me"]), np.array(r["ball"])
            best = None
            # todas las columnas de ese nombre (un bot que se reconecta tiene otro id)
            for c in cols:
                pp = z["pl_pos"][:, c]
                ok = ~np.isnan(pp[:, 0])
                if not ok.any():
                    continue
                if offsets:
                    guess = r["frame"] + int(np.median(offsets))
                    sel = np.where(ok & (np.abs(frames - guess) <= window))[0]
                else:
                    sel = np.where(ok)[0]
                if not len(sel):
                    continue
                err = np.abs(pp[sel] - me).sum(axis=1) + np.abs(z["ball_pos"][sel] - ball).sum(axis=1)
                j = int(np.argmin(err))
                if best is None or err[j] < best[0]:
                    best = (float(err[j]), int(sel[j]), c)
            if best is None or best[0] > MATCH_TOL:
                continue
            _, j_obs, c = best
            offsets.append(int(frames[j_obs]) - r["frame"])
            inp = z["pl_input"][:, c]
            app = next((j for j in range(j_obs, min(len(frames), j_obs + 120)) if int(inp[j]) == r["key"]), None)
            if app is None:
                continue
            samples.append(dict(obs=int(frames[j_obs]), applied=int(frames[app]), lat=int(frames[app] - frames[j_obs]),
                                client_frame=r["frame"], match_err=round(best[0], 3)))
        lats = np.array([s["lat"] for s in samples])
        res[bot] = dict(decisions=len(recs), matched=len(samples),
                        latency=dict(p10=float(np.percentile(lats, 10)), p50=float(np.median(lats)),
                                     p90=float(np.percentile(lats, 90)), min=int(lats.min()), max=int(lats.max()),
                                     hist=dict(collections.Counter(lats.tolist())))
                        if len(lats) else None,
                        client_minus_recording_offset=dict(p10=float(np.percentile(offsets, 10)) if offsets else None,
                                                           p90=float(np.percentile(offsets, 90)) if offsets else None))
    allv = [v for r in res.values() if r and r["latency"] for v, n in r["latency"]["hist"].items() for _ in range(n)]
    if allv:
        a = np.array(allv)
        res["todos"] = dict(n=len(a), p10=float(np.percentile(a, 10)), p50=float(np.median(a)),
                            p90=float(np.percentile(a, 90)), hist=dict(sorted(collections.Counter(a.tolist()).items())))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recording", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--traces", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = latency(a.recording, a.meta, sorted(glob.glob(a.traces)))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
