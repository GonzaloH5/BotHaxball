"""Paridad del cerebro de sala con el entrenamiento: un partido grabado entra tick a tick como si fuera la sala.

Para el mismo jugador y el mismo tick se comparan las probabilidades de la política con la observación del cerebro
(ventana viva + fase causal) y con la del entrenamiento (`hx11_data.build` sobre D0). Se le dan al cerebro la misma
ancla y la misma línea del jugador, los arqueros y el offside de D0, así que la única diferencia esperable es la fase
detectada en vivo (97% de acuerdo con D0). La comparación es en probabilidades porque el orden de las entidades puede
cambiar (lugares asignados en vivo) y la red es invariante a ese orden.

  python -m learn.hx11_cerebro_paridad --partido <nombre de prueba> --ticks 6000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from learn.hx11_cerebro import Cerebro
from learn.hx11_data import KICKOFF, LINES, build, l0_split, load_match
from tools.hx11_d0 import load_d0

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="runs/hx11_l0/l0_sinhist_full/model.pt")
    ap.add_argument("--partido", default=None)
    ap.add_argument("--ticks", type=int, default=6000)
    ap.add_argument("--delay", type=int, default=20)
    a = ap.parse_args()
    name = a.partido or l0_split()["test"][0]
    m = load_match(name, json.loads((ROOT / "data" / "hx11" / "modos.json").read_text(encoding="utf-8")))
    o, meta = load_d0(ROOT / "data" / "hx11" / "d0" / name)
    place = {d[0]: (d[4], d[5]) for d in meta["decisions"]}
    goals = np.r_[False, (np.diff(o["score"].astype(np.int64), axis=0) > 0).any(1)]
    vocab = json.loads((ROOT / "reports" / "hx11" / "pos_vocab.json").read_text(encoding="utf-8"))
    c = Cerebro(a.model, "DFC", vocab, temp=1.0, every=1)
    c.w.fouls = m.fouls
    # el bot: un jugador de campo presente casi todo el partido
    t0 = 600
    me = int(next(p for p in m.pid[t0] if p >= 0 and p not in (m.gk[t0, 0], m.gk[t0, 1])))
    c.anchor = None
    diffs, phase_eq = [], []
    for t in range(min(a.ticks, m.T - a.delay - 2)):
        players = [[int(m.pid[t, s]), int(s >= 11), m.pos[t, s, 0] / 10, m.pos[t, s, 1] / 10, m.vel[t, s, 0] / 1000,
                    m.vel[t, s, 1] / 1000, int(m.inp[t, s])] for s in range(22) if m.pid[t, s] >= 0]
        fr = dict(t=t, state=0 if m.phase[t] == KICKOFF else 1, score=o["score"][t].tolist(),
                  ball=[m.ball[t, 0] / 10, m.ball[t, 1] / 10, m.bvel[t, 0] / 1000, m.bvel[t, 1] / 1000,
                        float(m.binv[t]), float(m.bg[t, 0]), float(m.bg[t, 1])],
                  players=players, placement=place.get(t), goal=bool(goals[t]),
                  kicks=[int(m.pid[t, s]) for s in np.nonzero(m.kick[t])[0]], gk=m.gk[t].tolist(),
                  offside=bool(m.offside[t]), me=me)
        c.push(fr)
        if t < 400 or t % 30 or me not in c.slot_of or m.phase[t] != 0:
            continue
        s_live = c.slot_of[me]
        s_ref = int(np.nonzero(m.pid[t] == me)[0][0])
        half = int((t + a.delay + 1) >= m.T // 2)
        c.anchor = m.anchor.get((me, half), np.zeros(4, np.float32))
        c.line = m.line.get(me, 3)
        # misma ancla en las dos mitades para la referencia también
        w = c.w
        L = c.n - 1
        tt = L + a.delay + 1
        w.pid[L + 1:tt + 1] = w.pid[L]
        w.inp[L + 1:tt + 1, s_live] = w.inp[L, s_live]
        w.anchor[(me, 0)] = w.anchor[(me, 1)] = c.anchor
        w.line[me] = c.line
        w.T = 10 ** 9
        own_l, ents_l, _, _ = build(w, np.array([tt]), np.array([s_live]), np.array([a.delay]), c.net.extra)
        w.T = 4096
        own_r, ents_r, _, _ = build(m, np.array([t + a.delay + 1]), np.array([s_ref]), np.array([a.delay]), c.net.extra)
        own_r[:, 55:59] = c.anchor / 1000
        with torch.no_grad():
            p_l = torch.softmax(c.net(torch.from_numpy(own_l), torch.from_numpy(ents_l)), -1).numpy()
            p_r = torch.softmax(c.net(torch.from_numpy(own_r), torch.from_numpy(ents_r)), -1).numpy()
        diffs.append(float(np.abs(p_l - p_r).max()))
        phase_eq.append(bool(np.array_equal(own_l[:, 31:51], own_r[:, 31:51])))
    d = np.array(diffs)
    eq = np.array(phase_eq)
    print(json.dumps(dict(partido=name, comparaciones=len(d), fase_igual=round(float(eq.mean()), 4),
                          dif_max_prob_con_fase_igual=round(float(d[eq].max()), 6) if eq.any() else None,
                          dif_p50_con_fase_igual=round(float(np.median(d[eq])), 8) if eq.any() else None,
                          dif_max_prob_total=round(float(d.max()), 6)), ensure_ascii=False))


if __name__ == "__main__":
    main()
