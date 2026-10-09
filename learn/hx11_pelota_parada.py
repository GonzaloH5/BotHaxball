"""Pelotas paradas del lazo reactivo (22 bots, `learn/hx11_selfplay.py`) contra humanos del banco.

Dos cosas que el usuario vio en los partidos simulados (2026-10-09):
* **saque de arco:** un delantero rival queda encima del arquero. Se mide, en los ticks de saque de arco, la parte con
  algún rival dentro del área grande y la distancia del rival más cercano a la pelota;
* **lateral mal sacado:** el lateral necesita ángulo para entrar a la cancha; apurado, no entra y la sala lo repite para
  el rival en el mismo punto. Se cuentan los laterales y los que se repiten así.

  python -m learn.hx11_pelota_parada runs/hx11_l0/l0_sinhist_ancho/ver_actual/trayectorias.json [más…] [--humanos 20]
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from env.hx11 import contract as C
from learn.hx11_data import l0_split, load_match
from tools.hx11_d0 import PIECE

BOX_X, BOX_Y = 1050.0, 675.0      # área grande del estadio (líneas de stadiums/haxeleven_x11.hbs)


def stats(phase, kind, team, spot, pos, ball, on):
    """Conteos de una secuencia: pos (T, 22, 2) y ball (T, 2) en px, spot (T, 2) en px, on (T, 22)."""
    T = len(phase)
    start = np.flatnonzero((phase == PIECE) & (np.r_[-1, phase[:-1]] != PIECE))
    lat = [t for t in start if kind[t] == C.LATERAL]
    rep = 0
    for a, b in zip(lat[:-1], lat[1:]):
        # repetido para el rival en el mismo punto, sin otra pelota parada en el medio
        between = [t for t in start if a < t < b]
        if not between and b - a < 600 and team[a] != team[b] and np.hypot(*(spot[a] - spot[b])) < 40:
            rep += 1
    gk = (phase == PIECE) & (kind == C.GOAL_KICK)
    inside, near = [], []
    for t in np.flatnonzero(gk):
        tm = int(team[t])
        if tm < 0:
            continue
        side = -1.0 if ball[t, 0] < 0 else 1.0                  # lado del arco que saca
        rivals = np.arange(11, 22) if tm == 0 else np.arange(11)
        rv = rivals[on[t, rivals]]
        if not len(rv):
            continue
        q = pos[t, rv]
        inside.append(bool(((q[:, 0] * side > BOX_X) & (np.abs(q[:, 1]) < BOX_Y)).any()))
        near.append(float(np.hypot(*(q - ball[t]).T).min()))
    return dict(laterales=len(lat), laterales_repetidos=rep, ticks_saque_arco=int(gk.sum()),
                saque_arco_rival_en_area=inside, saque_arco_rival_mas_cercano=near, minutos=T / 3600)


def summarize(rows):
    lat = sum(r["laterales"] for r in rows)
    rep = sum(r["laterales_repetidos"] for r in rows)
    ins = [x for r in rows for x in r["saque_arco_rival_en_area"]]
    near = [x for r in rows for x in r["saque_arco_rival_mas_cercano"]]
    mins = sum(r["minutos"] for r in rows)
    return dict(minutos=round(mins, 1), laterales=lat, laterales_por_10min=round(lat / mins * 10, 2),
                laterales_mal_sacados=rep, parte_mal_sacados=round(rep / max(lat, 1), 3),
                ticks_saque_arco=len(ins), parte_con_rival_en_area=round(float(np.mean(ins)), 3) if ins else None,
                rival_mas_cercano_px_p50=round(float(np.median(near)), 1) if near else None)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trayectorias", nargs="+")
    ap.add_argument("--humanos", type=int, default=20)
    a = ap.parse_args()
    bots = []
    for f in a.trayectorias:
        for g in json.loads(open(f, encoding="utf-8").read()):
            if "pkind" not in g:
                continue
            pos = np.array(g["pos"], np.float64) / 10
            on = np.repeat(np.array(g["pid"], bool)[None], len(pos), 0)
            bots.append(stats(np.array(g["phase"]), np.array(g["pkind"]), np.array(g["pteam"]),
                              np.array(g["spot"], np.float64) / 10, pos, np.array(g["ball"], np.float64) / 10, on))
    hum = []
    for n in l0_split()["test"][:a.humanos]:
        m = load_match(n)
        hum.append(stats(m.phase, m.pkind.astype(np.int64), m.pteam.astype(np.int64), m.spot.astype(np.float64) / 10,
                         m.pos.astype(np.float64) / 10, m.ball.astype(np.float64) / 10, m.pid >= 0))
    print(json.dumps(dict(bots=summarize(bots) if bots else None, humanos=summarize(hum)), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
