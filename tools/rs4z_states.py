"""Banco de estados humanos RS4-Z: el banco RS4-b1 (`data/rs4_states/*.npz`) más la fase de masa (F1).

Cada estado guarda su grabación y cuadro; la fase sale de los eventos del script (0: masa del mapa 0,5
tras un reposicionamiento, 1: masa del script 0,3). Escribe `data/rs4_states/rs4z_<partición>.npz`.

  python -m tools.rs4z_states
"""
from __future__ import annotations

import argparse
import glob
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from tools.rs4z_conformance import _read, mass_phases

ROOT = Path(__file__).resolve().parent.parent


def _phases_for(args):
    name, frames = args
    hits = glob.glob(str(ROOT / "data" / "rs4_jsonl" / name / "*.jsonl.gz"))
    if not hits:
        return name, None
    ticks, events = _read(hits[0])
    phase = mass_phases(ticks, events)
    out = []
    for fr in frames:
        m = phase.get(int(fr))
        out.append(1 if m is None or m < 0.4 else 0)   # pieza (None) o 0,3 → fase de script
    return name, np.array(out, dtype=np.int8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    for split in ("entrenamiento", "desarrollo", "prueba"):
        path = ROOT / "data" / "rs4_states" / f"{split}.npz"
        if not path.exists():
            continue
        with np.load(path, allow_pickle=False) as z:
            data = {k: z[k] for k in z.files}
        names = [str(Path(r).stem) for r in data["recordings"]]
        jobs = []
        for i, name in enumerate(names):
            sel = np.flatnonzero(data["recording"] == i)
            jobs.append((name, data["frame"][sel]))
        mass = np.ones(len(data["kind"]), dtype=np.int8)
        with Pool(args.workers) as pool:
            for name, ph in pool.imap(_phases_for, jobs):
                if ph is None:
                    continue
                i = names.index(name)
                mass[np.flatnonzero(data["recording"] == i)] = ph
        data["mass_phase"] = mass
        out = path.with_name(f"rs4z_{split}.npz")
        np.savez_compressed(out, **data)
        print(split, len(mass), "estados; fase de mapa (0,5):", float((mass == 0).mean()).__round__(3))


if __name__ == "__main__":
    main()
