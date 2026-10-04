"""Tabla de amenaza esperada (xT) para el potencial Φ_threat de RS4-Z, ajustada con humanos.

Muestra estados en juego (4v4, sin saque activo) cada 15 ticks de las grabaciones de la partición de
ENTRENAMIENTO (`reports/rs4_b1/splits.json`; desarrollo y prueba no se usan). Posesión = equipo del
último contacto (exacto, tick a tick). Etiqueta = goles del poseedor − goles del rival en los
siguientes 600 ticks (10 s). Promedio por celda en el marco del poseedor, simetrizado en y, suavizado
y encogido hacia la media global con pseudo-conteos.

  python -m tools.rs4z_xt --out reports/rs4z/xt.json
"""
from __future__ import annotations

import argparse
import bisect
import glob
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from tools.rs4z_conformance import _read
from tools.rs4z_human_reference import restart_kinds

ROOT = Path(__file__).resolve().parent.parent
NX, NY = 23, 14
X0, DX = -1150.0, 100.0
Y0, DY = -670.0, 1340.0 / NY
HORIZON = 600
STRIDE = 15
REACH = 15.0 + 8.325 + 0.01


def train_dirs():
    splits = json.loads((ROOT / "reports/rs4_b1/splits.json").read_text(encoding="utf-8"))
    names = set()
    for sess in splits["sessions"].values():
        if sess["split"] == "entrenamiento":
            for rec in sess["recordings"]:
                names.add(Path(rec).stem)
    dirs = []
    for d in sorted(glob.glob(str(ROOT / "data/rs4_jsonl/*"))):
        if Path(d).name in names:
            dirs.append(d)
    return dirs


def scan(directory):
    files = glob.glob(str(Path(directory) / "*.jsonl.gz"))
    if not files:
        return None
    ticks, events = _read(files[0])
    kinds = restart_kinds(ticks, events)
    goals = sorted((fr, e["team"]) for fr, evs in events.items() for e in evs if e["name"] == "goal")
    goal_frames = [g[0] for g in goals]
    sums = np.zeros((NX, NY))
    counts = np.zeros((NX, NY))
    last = -1
    for i, t in enumerate(ticks):
        if t["state"] != 1 or len(t["players"]) != 8:
            if t["state"] != 1:
                last = -1
            continue
        bx, by = t["ball"][0], t["ball"][1]
        if bx is None or by is None:
            continue
        touch0 = touch1 = False
        for p in t["players"]:
            if p[4] is None:
                continue
            if (p[4] - bx) ** 2 + (p[5] - by) ** 2 <= REACH * REACH:
                if p[1] == 1:
                    touch0 = True
                else:
                    touch1 = True
        if touch0 and not touch1:
            last = 0
        elif touch1 and not touch0:
            last = 1
        if i % STRIDE or last < 0 or kinds.get(t["frame"], 0) != 0:
            continue
        fr = t["frame"]
        j = bisect.bisect_right(goal_frames, fr)
        label = 0.0
        while j < len(goals) and goals[j][0] <= fr + HORIZON:
            scorer = 0 if goals[j][1] == 1 else 1
            label += 1.0 if scorer == last else -1.0
            j += 1
        s = 1.0 if last == 0 else -1.0
        x = bx * s
        ix = int(np.clip((x - X0) // DX, 0, NX - 1))
        iy = int(np.clip((by - Y0) // DY, 0, NY - 1))
        sums[ix, iy] += label
        counts[ix, iy] += 1
    return sums, counts


def smooth(grid):
    k = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float64)
    pad = np.pad(grid, 1, mode="edge")
    out = np.zeros_like(grid)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            out[i, j] = (pad[i:i + 3, j:j + 3] * k).sum() / k.sum()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/rs4z/xt.json")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--prior", type=float, default=40.0, help="pseudo-conteos hacia la media global")
    args = ap.parse_args()
    dirs = train_dirs()
    sums = np.zeros((NX, NY))
    counts = np.zeros((NX, NY))
    with Pool(args.workers) as pool:
        for r in pool.imap_unordered(scan, dirs):
            if r is not None:
                sums += r[0]
                counts += r[1]
    # simetría en y
    sums = sums + sums[:, ::-1]
    counts = counts + counts[:, ::-1]
    mean = sums.sum() / max(counts.sum(), 1)
    shrunk = (sums + args.prior * mean) / (counts + args.prior)
    grid = smooth(smooth(shrunk))
    weighted_mean = float((grid * counts).sum() / max(counts.sum(), 1))
    report = dict(version="RS4-Z-xT-1", recordings=len(dirs), samples=int(counts.sum() / 2), horizon_ticks=HORIZON,
                  stride=STRIDE, x0=X0, dx=DX, y0=Y0, dy=DY, mean=weighted_mean, raw_mean=float(mean),
                  grid=grid.tolist(), counts=(counts / 2).tolist(),
                  note="valor esperado de (goles a favor - en contra) en 10 s para el equipo en posesión, marco del poseedor")
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report), encoding="utf-8")
    print(f"grabaciones {len(dirs)}, muestras {report['samples']}, media {weighted_mean:.4f}")
    xs = [0, 5, 11, 17, 20, 22]
    for ix in xs:
        print(f"x≈{X0 + (ix + 0.5) * DX:6.0f}:", " ".join(f"{grid[ix, iy]:+.3f}" for iy in range(0, NY, 2)))


if __name__ == "__main__":
    main()
