"""Auditar la alineación temporal entre estado y tecla en las grabaciones RS4 (PLAN_RS4.md 2.3).

Hoy conviven etiquetas de la misma tecla (`act`, secuencias de tools/build_rs4_sequences.py)
y desplazadas (`act_lag6`, train/bc.py). Esta auditoría mide en el motor original qué tecla
registrada mueve al jugador entre el tick t y el t+1: para cada desfase L se predice

    v(t+1) = (v(t) + aceleración(L) * dirección(tecla(t+L))) * amortiguación(L)

y se cuenta en qué fracción de los ticks la predicción coincide con la velocidad grabada
(tolerancia 1e-3 px/tick). El desfase causal es el de mayor coincidencia; las colisiones
explican el resto. Se informa también la coincidencia sólo en los ticks donde la tecla cambia,
que es donde los desfases se distinguen.

  python -m tools.rs4_alignment_audit --workers 8
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
LAGS = tuple(range(-3, 7))
TOLERANCE = 1e-3
# Física de jugadores del contrato (reports/rs4_b1/contract.json; valores por defecto de HaxBall).
ACCELERATION, KICKING_ACCELERATION, DAMPING, KICKING_DAMPING = 0.12, 0.07, 0.96, 0.96


def direction(inputs):
    dx = ((inputs >> 3) & 1).astype(np.float64) - ((inputs >> 2) & 1)
    dy = ((inputs >> 1) & 1).astype(np.float64) - (inputs & 1)
    norm = np.hypot(dx, dy)
    norm[norm == 0] = 1.0
    return np.stack([dx / norm, dy / norm], axis=-1)


def audit_recording(path):
    """Coincidencias por desfase en una grabación: {lag: [aciertos, total, aciertos_cambio, total_cambio]}."""
    from tools.rs4_jsonl_cache import open_ticks
    frames, velocity, inputs = {}, {}, {}
    with open_ticks(path) as handle:
        for line in handle:
            if not line.startswith(('{"type":"tick"', '{"type": "tick"')):
                continue
            item = json.loads(line)
            if item.get("state") != 1:
                continue
            frame = int(item["frame"])
            for p in item["players"]:
                disc = p.get("disc", -1)
                if disc is None or disc < 0 or disc >= len(item["discs"]):
                    continue
                d = item["discs"][disc]
                key = (p["id"], frame)
                velocity[key] = (d["vx"], d["vy"])
                inputs[key] = int(p.get("input") or 0)
                frames.setdefault(p["id"], []).append(frame)
    counts = {lag: np.zeros(4, dtype=np.int64) for lag in LAGS}
    for player, player_frames in frames.items():
        f = np.asarray(sorted(set(player_frames)), dtype=np.int64)
        consecutive = f[np.isin(f + 1, f)]
        if not len(consecutive):
            continue
        v0 = np.array([velocity[(player, t)] for t in consecutive])
        v1 = np.array([velocity[(player, t + 1)] for t in consecutive])
        keys = {lag: np.array([inputs.get((player, t + lag), -1) for t in consecutive]) for lag in LAGS}
        known = np.all([keys[lag] >= 0 for lag in LAGS], axis=0)
        same = np.all([keys[lag] == keys[0] for lag in LAGS], axis=0)
        for lag in LAGS:
            k = keys[lag]
            kick = (k & 16) > 0
            accel = np.where(kick, KICKING_ACCELERATION, ACCELERATION)[:, None]
            damp = np.where(kick, KICKING_DAMPING, DAMPING)[:, None]
            predicted = (v0 + accel * direction(np.maximum(k, 0))) * damp
            hit = np.hypot(*(predicted - v1).T) < TOLERANCE
            counts[lag] += [int((hit & known).sum()), int(known.sum()),
                            int((hit & known & ~same).sum()), int((known & ~same).sum())]
    return {lag: value.tolist() for lag, value in counts.items()}


def _job(args):
    name, path = args
    try:
        return name, audit_recording(path)
    except Exception as error:  # una grabación dañada no detiene la auditoría
        print(f"error {name}: {error}", flush=True)
        return name, None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default=str(ROOT / "data" / "rs4_jsonl"))
    ap.add_argument("--splits", default=str(ROOT / "reports" / "rs4_b1" / "splits.json"))
    ap.add_argument("--split", default="entrenamiento")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default=str(ROOT / "reports" / "rs4_b1" / "alignment_audit.json"))
    a = ap.parse_args()
    splits = json.loads(Path(a.splits).read_text(encoding="utf-8"))
    jobs = [(name, str(Path(a.cache) / row["jsonl"].replace("\\", "/"))) for name, row in splits["recordings"].items()
            if row["only_rs_one"] and row["split"] == a.split]
    started = time.time()
    if a.workers > 1:
        from multiprocessing import get_context
        with get_context("spawn").Pool(a.workers) as pool:
            results = pool.map(_job, jobs)
    else:
        results = [_job(job) for job in jobs]
    total = {lag: np.zeros(4, dtype=np.int64) for lag in LAGS}
    for _, counts in results:
        for lag, value in (counts or {}).items():
            total[int(lag)] += value
    table = {lag: dict(match=float(v[0] / max(v[1], 1)), match_on_key_changes=float(v[2] / max(v[3], 1)),
                       ticks=int(v[1]), key_change_ticks=int(v[3])) for lag, v in total.items()}
    causal = max(table, key=lambda lag: table[lag]["match_on_key_changes"])
    report = dict(version="RS4-b1-alignment-1", split=a.split, recordings=sum(c is not None for _, c in results),
                  tolerance=TOLERANCE, lags=table, causal_lag=causal, seconds=round(time.time() - started, 1),
                  meaning=(f"la tecla registrada en el tick t+{causal} es la que mueve al jugador de t a t+1"
                           if causal >= 0 else f"la tecla del tick t{causal} mueve al jugador de t a t+1"))
    Path(a.out).write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    for lag, row in table.items():
        print(f"desfase {lag:+d}: coincide {100 * row['match']:.1f}% | en cambios de tecla "
              f"{100 * row['match_on_key_changes']:.1f}% ({row['key_change_ticks']} ticks)")
    print(f"desfase causal {causal:+d}: {report['meaning']}")


if __name__ == "__main__":
    main()
