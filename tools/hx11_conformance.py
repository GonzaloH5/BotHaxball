"""Conformidad del simulador con las grabaciones HaxEleven (docs/PLAN_HAXELEVEN.md, S1 y luego S4).

`fisica`: física de HaxBall en juego abierto. Carga el estado grabado de un tick en un BatchSim de un partido
(posiciones, velocidades y las PROPIEDADES que dejó el script de la sala: invMass de la pelota con la carga del
powershot, invMass/bCoef/cMask/cGroup de jugadores, barreras, gravedad de la pelota), avanza k ticks con las entradas
grabadas aplicando en cada tick las propiedades grabadas, y mide el error contra el juego. Se excluyen las transiciones
en que el script mueve un disco simulado (colocaciones de saque), los goles, los cambios de plantel y los tramos con
gravedad en un jugador (el motor sólo modela la de la pelota). `--align` elige qué propiedades valen en la transición
t -> t+1: las del tick t ("t") o las del t+1 ("t1").

`bench`: decisiones de partido por segundo del motor con 11v11 en el estadio (y 4v4 de referencia).

  node bridge/hx11_ticks.js <grabación.hbr2> --out data/hx11/ticks --max-minutes 3
  python -m tools.hx11_conformance fisica data/hx11/ticks/*.jsonl --out reports/hx11/s1_fisica.json
  python -m tools.hx11_conformance bench
"""
from __future__ import annotations

import argparse
import collections
import json
import time
from pathlib import Path

import numpy as np

from sim.physics import MOVE_DIRS, BatchSim
from sim.stadium import load_stadium

STADIUM = "haxeleven_x11"
HORIZONS = (1, 10, 30, 60)
PROPS = ("radius", "invMass", "bCoef", "damping", "cMask", "cGroup", "gx", "gy")

# entrada de HaxBall: 1 arriba, 2 abajo, 4 izquierda, 8 derecha, 16 patear -> acción 0..17 del sim
_DIR2MOVE = {(int(x), int(y)): i for i, (x, y) in enumerate(MOVE_DIRS)}


def input_to_action(inp: int) -> int:
    dx = ((inp >> 3) & 1) - ((inp >> 2) & 1)
    dy = ((inp >> 1) & 1) - (inp & 1)
    return _DIR2MOVE[(dx, dy)] + (9 if inp & 16 else 0)


def players_dict(p):
    """Jugador del tick en el formato compacto [id, equipo, input, patada activa, disco] o el viejo (dict)."""
    if isinstance(p, dict):
        return p
    return dict(id=p[0], team=p[1], input=p[2], kicking=bool(p[3]), disc=p[4])


class Recording:
    """Ticks de `bridge/hx11_ticks.js` con el estado completo reconstruido (los discos quietos y las propiedades sin
    cambio no se repiten en el archivo)."""

    def __init__(self, path):
        self.path = Path(path)
        self.header, ticks, self.events = None, [], []
        for line in open(path, encoding="utf-8"):
            try:
                o = json.loads(line)
            except json.JSONDecodeError:  # última línea cortada
                continue
            if o["type"] == "tick":
                o["players"] = [players_dict(p) for p in o["players"]]
                ticks.append(o)
            elif o["type"] == "header":
                self.header = o
            else:
                self.events.append(o)
        self.ticks = ticks
        T = len(ticks)
        nd = max(t["nd"] for t in ticks)
        self.pos = np.full((T, nd, 2), np.nan)
        self.vel = np.full((T, nd, 2), np.nan)
        self.prop = {k: np.full((T, nd), np.nan) for k in PROPS}
        cur_pv = np.full((nd, 4), np.nan)
        cur_pr = {k: np.full(nd, np.nan) for k in PROPS}
        for j, t in enumerate(ticks):
            for src in (t["m"], t["d"]):
                for i, pv in src.items():
                    cur_pv[int(i)] = [np.nan if v is None else v for v in pv]
            for i, ch in t["p"].items():
                for k, v in ch.items():
                    if k in cur_pr:
                        cur_pr[k][int(i)] = np.nan if v is None else v
            self.pos[j] = cur_pv[:, :2]
            self.vel[j] = cur_pv[:, 2:]
            for k in PROPS:
                self.prop[k][j] = cur_pr[k]
        self.frame = np.array([t["frame"] for t in ticks])
        # discos que el script coloca o lanza (setDiscProperties con x, y, vx o vy), por frame
        self.teleports = collections.defaultdict(set)
        for e in self.events:
            if e["name"] == "disc_props" and e.get("d1") and any(v is not None for v in e["d1"][:4]):
                self.teleports[e["frame"]].add(("p" if e["player"] else "d", e["id"]))


def roster(t):
    return tuple(sorted((p["id"], p["team"]) for p in t["players"]))


class Replayer:
    """BatchSim de un partido alineado con un tramo de plantel fijo de la grabación."""

    def __init__(self, st, rec: Recording, j0: int, align: str):
        self.rec, self.align = rec, align
        t0 = rec.ticks[j0]
        # mismo orden de discos que el juego: HaxBall resuelve los choques de a pares en ese orden y, con varios
        # jugadores en contacto a la vez (cuerpear), otro orden da otro resultado (hasta 0,07 px por tick)
        self.players = sorted(t0["players"], key=lambda p: p["disc"])
        n_red = sum(p["team"] == 1 for p in self.players)
        self.sim = BatchSim(1, n_red, len(self.players) - n_red, st)
        self.sim.player_team[:] = [0 if p["team"] == 1 else 1 for p in self.players]
        self.stadium_idx = [0] + [1 + int(i) for i in st.d_src]
        self.pids = [p["id"] for p in self.players]

    def disc_idx(self, j):
        by_id = {p["id"]: p["disc"] for p in self.rec.ticks[j]["players"]}
        return np.array(self.stadium_idx + [by_id[i] for i in self.pids])

    def set_props(self, j):
        s, r = self.sim, self.rec
        idx = self.disc_idx(j)
        s.radius[:] = r.prop["radius"][j, idx]
        s.inv_env[0] = r.prop["invMass"][j, idx]
        s.bcoef[:] = r.prop["bCoef"][j, idx]
        s.damping[:] = r.prop["damping"][j, idx]
        s.group[:] = r.prop["cGroup"][j, idx].astype(np.int64)
        s.mask[0] = r.prop["cMask"][j, idx].astype(np.int64)
        s.ball_grav[0] = [r.prop["gx"][j, 0], r.prop["gy"][j, 0]]

    def load(self, j):
        s, r = self.sim, self.rec
        idx = self.disc_idx(j)
        s.pos[0] = r.pos[j, idx]
        s.vel[0] = r.vel[j, idx]
        by_id = {p["id"]: p for p in r.ticks[j]["players"]}
        # patada "gastada": tecla apretada pero isKicking=false
        s.kick_cancel[0] = [bool(by_id[i]["input"] & 16) and not by_id[i]["kicking"] for i in self.pids]
        s.kickoff[0] = False  # las máscaras del saque inicial vienen de la grabación
        self.set_props(j)

    def step(self, j):
        """Transición j -> j+1 con la entrada grabada en j y las propiedades de j (o de j+1)."""
        self.set_props(j if self.align == "t" else j + 1)
        by_id = {p["id"]: p for p in self.rec.ticks[j]["players"]}
        self.sim.step(np.array([[input_to_action(by_id[i]["input"]) for i in self.pids]]))

    def error(self, j):
        idx = self.disc_idx(j)
        real = self.rec.pos[j, idx]
        p = self.sim.pos[0]
        fp = self.sim.first_player
        return float(np.linalg.norm(p[0] - real[0])), float(np.linalg.norm(p[fp:] - real[fp:], axis=1).max())


def window_ok(rec, rp, i, k, states, reasons):
    """Tramo i..i+k comparable: frames seguidos, mismo plantel y marcador, estado pedido, sin colocaciones del script
    sobre discos simulados y sin gravedad en jugadores."""
    t = rec.ticks
    if t[i]["state"] not in states:
        reasons["estado"] += 1
        return False
    sim_discs = set(rp.stadium_idx)
    for j in range(i, i + k):
        a, b = t[j], t[j + 1]
        if b["frame"] - a["frame"] != 1:
            reasons["frames"] += 1
            return False
        if b["score"] != t[i]["score"] or roster(b) != roster(t[i]):
            reasons["gol/plantel"] += 1
            return False
        for kind, did in rec.teleports.get(b["frame"], set()) | rec.teleports.get(a["frame"], set()):
            if kind == "p" or did in sim_discs:
                reasons["colocación del script"] += 1
                return False
    idx = rp.disc_idx(i)[rp.sim.first_player:]
    g = np.abs(rec.prop["gx"][i:i + k + 1][:, idx]) + np.abs(rec.prop["gy"][i:i + k + 1][:, idx])
    if np.nanmax(g) > 0:
        reasons["gravedad en jugador"] += 1
        return False
    if np.isnan(rec.pos[i:i + k + 1][:, rp.disc_idx(i)]).any():
        reasons["disco sin posición"] += 1
        return False
    return True


def physics(recs, align, states=(1,)):
    st = load_stadium(STADIUM)
    out = {}
    worst = []
    for k in HORIZONS:
        eb, ep, reasons, n_props = [], [], collections.Counter(), 0
        for rec in recs:
            rps = {}
            for i in range(0, len(rec.ticks) - k - 1, max(1, k // 3)):
                key = roster(rec.ticks[i])
                if key not in rps:
                    rps[key] = Replayer(st, rec, i, align)
                rp = rps[key]
                if not window_ok(rec, rp, i, k, states, reasons):
                    continue
                rp.load(i)
                for j in range(i, i + k):
                    rp.step(j)
                e_b, e_p = rp.error(i + k)
                eb.append(e_b)
                ep.append(e_p)
                if k == 1:
                    changed = any(not np.array_equal(rec.prop[p][i], rec.prop[p][i + 1], equal_nan=True)
                                  for p in PROPS)
                    n_props += changed
                    worst.append((max(e_b, e_p), rec.path.name, int(rec.frame[i + 1]), e_b, e_p, changed))
        eb, ep = np.array(eb), np.array(ep)
        out[k] = dict(n=len(eb), reasons=dict(reasons),
                      ball=dict(p50=float(np.median(eb)), p90=float(np.percentile(eb, 90)), max=float(eb.max()))
                      if len(eb) else None,
                      players=dict(p50=float(np.median(ep)), p90=float(np.percentile(ep, 90)), max=float(ep.max()))
                      if len(ep) else None)
        if k == 1:
            out[k]["transitions_with_prop_change"] = n_props
    worst.sort(key=lambda x: -x[0])
    out["worst_1tick"] = [dict(err=round(w[0], 4), file=w[1], frame=w[2], ball=round(w[3], 4), players=round(w[4], 4),
                               prop_change=w[5]) for w in worst[:15]]
    return out


def bench(n_envs=256, steps=600):
    res = {}
    for name, (nr, nb, stadium) in {"11v11 haxeleven_x11": (11, 11, STADIUM),
                                     "4v4 sanguchito_rs_x4": (4, 4, "sanguchito_rs_x4")}.items():
        sim = BatchSim(n_envs, nr, nb, stadium, seed=0)
        rng = np.random.default_rng(0)
        acts = rng.integers(0, 18, size=(steps, n_envs, nr + nb))
        sim.step(acts[0])  # compilación
        t0 = time.perf_counter()
        for a in acts:
            sim.step(a)
        dt = time.perf_counter() - t0
        res[name] = dict(n_envs=n_envs, steps=steps, ticks_per_s=round(n_envs * steps / dt),
                         player_ticks_per_s=round(n_envs * steps * (nr + nb) / dt))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fisica")
    f.add_argument("recordings", nargs="+")
    f.add_argument("--align", choices=("t", "t1", "ambos"), default="ambos")
    f.add_argument("--out", default=None)
    b = sub.add_parser("bench")
    b.add_argument("--envs", type=int, default=256)
    b.add_argument("--steps", type=int, default=600)
    a = ap.parse_args()
    if a.cmd == "fisica":
        recs = [Recording(p) for p in a.recordings]
        res = {al: dict(open_play=physics(recs, al), kickoff=physics(recs, al, states=(0,)))
               for al in (("t", "t1") if a.align == "ambos" else (a.align,))}
        res = dict(stadium=STADIUM, recordings=[Path(p).name for p in a.recordings], results=res)
        txt = json.dumps(res, indent=1, ensure_ascii=False)
        if a.out:
            Path(a.out).write_text(txt, encoding="utf-8")
        for al, r in res["results"].items():
            for part in ("open_play", "kickoff"):
                for k in HORIZONS:
                    x = r[part][k]
                    if x["ball"]:
                        print(f"[{al}] {part:9s} {k:3d} ticks n={x['n']:6d} | pelota p50 {x['ball']['p50']:.4f} "
                              f"p90 {x['ball']['p90']:.4f} máx {x['ball']['max']:.3f} | jugadores p50 "
                              f"{x['players']['p50']:.4f} p90 {x['players']['p90']:.4f} máx {x['players']['max']:.3f}")
                print(f"[{al}] {part} 1 tick: {r[part][1].get('transitions_with_prop_change')} con cambio de "
                      f"propiedades; excluidos {r[part][1]['reasons']}")
    else:
        print(json.dumps(bench(a.envs, a.steps), indent=1))


if __name__ == "__main__":
    main()
