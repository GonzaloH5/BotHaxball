"""Partido HaxEleven en arreglos por entidad (docs/PLAN_HAXELEVEN.md, S2): lo que lee la arqueología del script.

Lee el JSONL de `bridge/hx11_ticks.js` (archivo o entrada estándar) y guarda `<nombre>.npz` + `<nombre>.events.json`:

* `frame`, `state`, `score`, `time` por tick.
* Pelota: `ball_pos`, `ball_vel` (T, 2) y sus propiedades `ball_<prop>` (T,).
* Jugadores por id (`pid`, en orden de aparición): `pl_pos`, `pl_vel` (T, P, 2), `pl_input`, `pl_kick`, `pl_team`, `pl_disc`
  (T, P; equipo 0 = no está en la cancha) y propiedades `pl_<prop>` (T, P).
* Discos del script (`disc_ids`: barreras, adornos de la pelota, barra de fuerza): `disc_pos` (T, D, 2) y `disc_<prop>`.
* Eventos (setDiscProperties, anuncios, patadas, goles, cambios de equipo, pausas): en el JSON, con su frame. Los anuncios
  traen nombres: todo queda en data/ (no versionado).

Posiciones y propiedades en float32 (resolución ~2e-4 px en la cancha): sirve para analizar reglas, no para re-simular
exacto (para eso, el JSONL y `tools.hx11_conformance`).

  node bridge/hx11_ticks.js <grabación.hbr2> --max-minutes 0 --stdout | python -m tools.hx11_npz - --out data/hx11/npz/<nombre>
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np

PROPS = ("radius", "invMass", "bCoef", "damping", "cMask", "cGroup", "gx", "gy", "color")
DISC_IDS = (1, 2, 19, 20, 21, 22, 26, 27) + tuple(range(28, 47))


class _Blocks:
    """Arreglos que crecen por bloques de ticks (el largo del partido no se sabe hasta el final)."""

    def __init__(self, specs, block=8192):
        self.specs, self.block, self.n, self.parts, self.cur = specs, block, 0, {k: [] for k in specs}, None
        self._new()

    def _new(self):
        self.cur = {k: np.full((self.block,) + shape, fill, dtype) for k, (shape, dtype, fill) in self.specs.items()}
        self.j = 0

    def row(self):
        if self.j == self.block:
            for k, v in self.cur.items():
                self.parts[k].append(v)
            self._new()
        self.j += 1
        self.n += 1
        return self.cur, self.j - 1

    def arrays(self):
        return {k: np.concatenate(self.parts[k] + [self.cur[k][:self.j]]) for k in self.specs}


def convert(lines, out_base: Path, max_players=96):
    f32 = np.float32
    D, P = len(DISC_IDS), max_players
    specs = dict(frame=((), np.int32, 0), state=((), np.int8, 0), score=((2,), np.int16, 0), time=((), f32, 0),
                 ball_pos=((2,), f32, np.nan), ball_vel=((2,), f32, np.nan),
                 pl_pos=((P, 2), f32, np.nan), pl_vel=((P, 2), f32, np.nan), pl_input=((P,), np.int16, -1),
                 pl_kick=((P,), np.int8, 0), pl_team=((P,), np.int8, 0), pl_disc=((P,), np.int16, -1),
                 disc_pos=((D, 2), f32, np.nan))
    for k in PROPS:
        dt = np.float64 if k == "color" else f32
        specs[f"ball_{k}"] = ((), dt, np.nan)
        specs[f"pl_{k}"] = ((P,), dt, np.nan)
        # los adornos de la pelota tienen invMass 1e250: no entra en float32
        specs[f"disc_{k}"] = ((D,), np.float64 if k in ("color", "invMass") else f32, np.nan)
    blk = _Blocks(specs)
    header, events, pid = None, [], []
    col, dcol = {}, {d: i for i, d in enumerate(DISC_IDS)}
    # estado vigente por índice de disco (los quietos y las propiedades sin cambio no se repiten en el JSONL)
    cur_pv, cur_pr = {}, {}
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            t = json.loads(line)
        except json.JSONDecodeError:
            continue
        kind = t.get("type")
        if kind == "header":
            header = t
            continue
        if kind == "event":
            events.append(t)
            continue
        if kind != "tick":
            continue
        for src in (t["m"], t["d"]):
            for i, pv in src.items():
                cur_pv[int(i)] = pv
        for i, ch in t["p"].items():
            cur_pr.setdefault(int(i), {}).update(ch)
        a, j = blk.row()
        a["frame"][j], a["state"][j], a["score"][j], a["time"][j] = t["frame"], t["state"], t["score"], t["time"] or 0
        bp = cur_pv.get(0)
        if bp:
            a["ball_pos"][j], a["ball_vel"][j] = bp[:2], bp[2:]
        for k, v in cur_pr.get(0, {}).items():
            if k in PROPS and v is not None:
                a[f"ball_{k}"][j] = v
        for p in t["players"]:
            if p[0] not in col:
                if len(pid) == P:
                    raise SystemExit(f"más de {P} jugadores distintos: subir max_players")
                col[p[0]] = len(pid)
                pid.append(p[0])
            c = col[p[0]]
            a["pl_team"][j, c], a["pl_input"][j, c], a["pl_kick"][j, c], a["pl_disc"][j, c] = p[1], p[2], p[3], p[4]
            pv = cur_pv.get(p[4])
            if pv:
                a["pl_pos"][j, c], a["pl_vel"][j, c] = pv[:2], pv[2:]
            for k, v in cur_pr.get(p[4], {}).items():
                if k in PROPS and v is not None:
                    a[f"pl_{k}"][j, c] = v
        for d, c in dcol.items():
            pv = cur_pv.get(d)
            if pv:
                a["disc_pos"][j, c] = pv[:2]
            for k, v in cur_pr.get(d, {}).items():
                if k in PROPS and v is not None:
                    a[f"disc_{k}"][j, c] = v
    arr = blk.arrays()
    n = len(pid)
    for k in list(arr):
        if k.startswith("pl_"):
            arr[k] = arr[k][:, :n]
    arr["pid"] = np.array(pid, np.int32)
    arr["disc_ids"] = np.array(DISC_IDS, np.int16)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(out_base) + ".npz", **arr)
    Path(str(out_base) + ".events.json").write_text(
        json.dumps(dict(header=header, events=events)), encoding="utf-8")  # ascii: hay nombres con surrogates sueltos
    return blk.n, n


class _Lazy(dict):
    """Arreglos del .npz que se leen recién cuando se usan (un partido entero pesa ~150 MB descomprimido)."""

    def __init__(self, path):
        super().__init__()
        self._npz = np.load(path)

    def __missing__(self, k):
        v = self[k] = self._npz[k]
        return v

    def __contains__(self, k):
        return k in self._npz.files


def load(base):
    """(arreglos, eventos, header) de un partido convertido."""
    base = str(base).removesuffix(".npz")
    z = _Lazy(base + ".npz")
    ev = json.loads(Path(base + ".events.json").read_text(encoding="utf-8"))
    return z, ev["events"], ev["header"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help="JSONL de bridge/hx11_ticks.js, o - para la entrada estándar")
    ap.add_argument("--out", required=True, help="ruta base de salida (sin extensión)")
    a = ap.parse_args()
    lines = (io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8", errors="replace") if a.src == "-"
             else open(a.src, encoding="utf-8", errors="replace"))
    T, P = convert(lines, Path(a.out))
    print(f"{a.out}: {T} ticks, {P} jugadores")


if __name__ == "__main__":
    main()
