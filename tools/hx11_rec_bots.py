"""Diagnóstico de un partido de bots en una sala real (deploy/hx11/partido.js) a partir de la grabación del host.

Pasa la grabación por el mismo D0 que el banco (bridge/hx11_ticks.js → tools.hx11_npz → tools.hx11_d0.build), le
pone a cada bot el ancla de su rol (por el nombre: "R2 DEF-I", "A10 DEL-CI"…) y mide, por jugador y por equipo, lo mismo
que el lazo reactivo del simulador (learn/hx11_selfplay.py) contra humanos del banco (partidos de prueba):

* por jugador, en juego abierto: parte del tiempo quieto, cambios de tecla por segundo (la carga que el host reenvía a
  todos), distancia a la pelota, distancia a su zona (centroide de campo + ancla), patadas y toques por minuto;
* por equipo: las métricas de `game_metrics` (largo, ancho, amontonamiento, el más cercano a la pelota…).

  python -m tools.hx11_rec_bots <grabación.hbr2> [--rojo 4-3-3] [--azul 4-4-2] [--humanos 8] [--out reports/hx11/x.json]
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

import numpy as np

from env.hx11 import contract as HC
from learn.hx11_data import LINES, Match, l0_split, load_match
from learn.hx11_selfplay import game_metrics
from tools.hx11_d0 import OPEN, build
from tools.hx11_npz import convert

ROOT = Path(__file__).resolve().parent.parent
NAME = re.compile(r"^([RA])(\d+)\s+(\S+)$")


def match_from_rec(path):
    """Match (como el del banco) y nombres por id de jugador."""
    p = subprocess.run(["node", str(ROOT / "bridge" / "hx11_ticks.js"), str(path), "--max-minutes", "0", "--stdout"],
                       cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", check=True)
    arrays = convert(p.stdout.splitlines(), None, max_players=48, return_arrays=True)
    _, events, header = arrays
    out, _ = build(arrays, Path(path).stem)
    names = {q["id"]: q["name"] for q in (header or {}).get("players", [])}
    for e in events:
        if e["name"] == "join":
            names[e["id"]] = e.get("player")
    gk = np.where(out["gk_pid"] >= 0, out["gk_pid"], out["gk_inferred"]).astype(np.int32)
    T = len(out["frame"])
    m = Match(name=Path(path).stem, T=T, pos=out["pos"], vel=out["vel"], ball=out["ball_pos"], bvel=out["ball_vel"],
              binv=out["ball_inv"], bg=out["ball_g"], inp=out["input"], kick=out["kick"], score=out["score"],
              pid=out["slot_pid"], gk=gk, phase=out["phase"], pkind=out["piece_kind"], pteam=out["piece_team"],
              pticks=out["phase_ticks"], spot=out["spot"], offside=out["offside_on"], fouls=0, anchor={}, line={})
    return m, names


def player_stats(m: Match):
    """Por (lugar, id) en juego abierto: quieto, cambios de tecla/s, distancia a la pelota, a la zona, patadas y toques."""
    op = np.nonzero(m.phase == OPEN)[0]
    op = op[op > 0]
    pos = m.pos.astype(np.float64) / 10
    ball = m.ball.astype(np.float64) / 10
    # último toque (para el ancla con o sin pelota)
    gap = np.hypot(*(pos - ball[:, None]).transpose(2, 0, 1)) - HC.PLAYER_R - HC.BALL_R
    gap[m.pid < 0] = np.inf
    touch = gap <= 0.6
    poss = np.full(m.T, -1)
    last = -1
    for t in range(m.T):
        tt = np.nonzero(touch[t])[0]
        if len(tt):
            last = int(tt[0] >= 11)
        poss[t] = last
    rows = []
    for s in range(22):
        team = int(s >= 11)
        att = 1.0 if team == 0 else -1.0
        for pp in np.unique(m.pid[op, s]):
            if pp < 0:
                continue
            ts = op[m.pid[op, s] == pp]
            if len(ts) < 600:
                continue
            sp = np.hypot(*(pos[ts, s] - pos[ts - 1, s]).T)
            ch = (m.inp[ts, s] != m.inp[ts - 1, s]).sum()
            sl = np.arange(team * 11, team * 11 + 11)
            f = (m.pid[ts][:, sl] >= 0) & (m.pid[ts][:, sl] != m.gk[ts, team][:, None])
            q = pos[ts][:, sl] * att
            cen = (q * f[..., None]).sum(1) / np.maximum(f.sum(1), 1)[:, None]
            rel = pos[ts, s] * att - cen
            a = [m.anchor.get((int(pp), h)) for h in (0, 1)]
            a = [x for x in a if x is not None]
            zone = np.nan
            if a:
                a = np.mean(a, 0)
                anc = np.where((poss[ts] == team)[:, None], a[None, 0:2], a[None, 2:4])
                zone = float(np.hypot(*(rel - anc).T).mean())
            mins = len(ts) / 3600
            kk = (np.maximum(m.inp[ts, s].astype(np.int64), 0) >> 4) & 1
            kp = (np.maximum(m.inp[ts - 1, s].astype(np.int64), 0) >> 4) & 1
            rows.append(dict(lugar=s, id=int(pp), es_arquero=bool((m.gk[ts, team] == pp).mean() > 0.5),
                             linea=LINES[m.line.get(int(pp), 3)], minutos=round(mins, 2),
                             quieto=float((sp < 0.25).mean()), cambios_s=float(ch / (len(ts) / 60)),
                             dist_pelota_p50=float(np.median(np.hypot(*(pos[ts, s] - ball[ts]).T))),
                             dist_zona=zone, patadas_min=float(((kk == 1) & (kp == 0) & (gap[ts - 1, s] <= 4.0)).sum() / mins),
                             toques_min=float(touch[ts, s].sum() / 60 / mins)))
    return rows


def agg(rows, key):
    out = {}
    for ln in LINES:
        rr = [r for r in rows if r["linea"] == ln and not np.isnan(r[key])]
        if rr:
            w = np.array([r["minutos"] for r in rr])
            out[ln] = round(float(np.average([r[key] for r in rr], weights=w)), 3)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rec")
    ap.add_argument("--rojo", default="4-3-3")
    ap.add_argument("--azul", default="4-4-2")
    ap.add_argument("--humanos", type=int, default=8)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    vocab = json.loads((ROOT / "reports" / "hx11" / "pos_vocab.json").read_text(encoding="utf-8"))
    m, names = match_from_rec(a.rec)
    for pid, nm in names.items():
        g = NAME.match((nm or "").strip())
        if not g:
            continue
        key = f"{a.rojo if g.group(1) == 'R' else a.azul}:{g.group(3)}"
        v = vocab.get("roles", {}).get(key)
        if v:
            m.anchor[(int(pid), 0)] = m.anchor[(int(pid), 1)] = np.array(v["ancla"], np.float32)
            m.line[int(pid)] = int(v["linea_idx"])
    rows = player_stats(m)
    for r in rows:
        # solo los nombres de los bots (R2 DEF-I…): el reporte se versiona y no lleva nombres de jugadores
        nm = (names.get(r["id"]) or "").strip()
        r["nombre"] = nm if NAME.match(nm) else "humano"
    team = game_metrics(m, 1, m.T)
    # humanos: partidos de prueba del banco
    hum_rows, hum_team = [], []
    for n in l0_split()["test"][:a.humanos]:
        hm = load_match(n)
        hum_rows += player_stats(hm)
        hum_team.append(game_metrics(hm, 1, hm.T))
    keys = ["quieto", "cambios_s", "dist_pelota_p50", "dist_zona", "patadas_min", "toques_min"]
    res = dict(grabacion=Path(a.rec).name, minutos_juego_abierto=round(float((m.phase == OPEN).sum() / 3600), 2),
               jugadores=len(rows),
               por_linea={k: dict(bots=agg(rows, k), humanos=agg(hum_rows, k)) for k in keys},
               equipo=dict(bots={k: round(float(v), 3) for k, v in team.items()},
                           humanos={k: round(float(np.nanmean([h[k] for h in hum_team])), 3) for k in team}),
               por_bot=sorted(({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()} for r in rows),
                              key=lambda r: r["lugar"]))
    txt = json.dumps(res, ensure_ascii=False, indent=1)
    if a.out:
        Path(a.out).write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
