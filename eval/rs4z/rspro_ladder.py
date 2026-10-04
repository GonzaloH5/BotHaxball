"""Validación de RS-Pro: escalera de niveles, métricas de L5 contra L5 y pruebas de explotación.

  python -m eval.rs4z.rspro_ladder --games 256 --out reports/rs4z/rspro_ladder.json

* Escalera: cada nivel Lk contra Lk-1 (ambos colores, estilos de entrenamiento al azar): Lk debe sacar
  ≥ 60% de los puntos con IC95 > 50%; también L5 contra L0..L3 (transitividad).
* L5 contra L5: métricas de estilo y detectores contra la referencia humana (diagnóstico).
* Tramposos: cada política degenerada contra L5; L5 debe sacar ≥ 60% de los puntos.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from bots.rspro.controller import RSProController
from bots.rspro.policy import sample_style
from eval.rs4z.cheaters import KINDS, Cheater
from eval.rs4z.runner import play, points

ROOT = Path(__file__).resolve().parent.parent.parent


def _bootstrap(values, reps=2000, seed=0):
    rng = np.random.default_rng(seed)
    v = np.asarray(values, dtype=np.float64)
    means = v[rng.integers(0, len(v), (reps, len(v)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def match_levels(level_a, level_b, games, seed, minutes=3.0):
    """level_a juega de rojo en la mitad de los partidos y de azul en la otra; puntos de level_a."""
    rng = np.random.default_rng(seed)
    half = games // 2
    pts = []
    goals = []
    for flip in (False, True):
        levels = np.zeros((half, 2), dtype=np.int64)
        levels[:, 0] = level_b if flip else level_a
        levels[:, 1] = level_a if flip else level_b
        styles = np.stack([[sample_style(rng) for _ in range(2)] for _ in range(half)])
        ctl = RSProController(levels=levels, styles=styles, seed=seed + (7 if flip else 0))
        r = play(ctl, ctl, n_envs=half, match_ticks=int(minutes * 3600), seed=seed + (1 if flip else 0))
        p = points(r["score"])
        pts.append(1.0 - p if flip else p)
        goals.append(r["score"].sum(axis=1))
    pts = np.concatenate(pts)
    lo, hi = _bootstrap(pts, seed=seed)
    return dict(points=float(pts.mean()), ci95=(lo, hi), games=int(len(pts)), goals_per_match=float(np.concatenate(goals).mean()))


def mirror_metrics(level, games, seed, minutes=3.0):
    rng = np.random.default_rng(seed)
    levels = np.full((games, 2), level, dtype=np.int64)
    styles = np.stack([[sample_style(rng) for _ in range(2)] for _ in range(games)])
    ctl = RSProController(levels=levels, styles=styles, seed=seed)
    r = play(ctl, ctl, n_envs=games, match_ticks=int(minutes * 3600), seed=seed)
    out = r["metrics"].summary()
    out["goals_per_match"] = float(r["score"].sum(axis=1).mean())
    out["zero_zero"] = float((r["score"].sum(axis=1) == 0).mean())
    out["forfeits"] = float(r["events"]["forfeits"].sum())
    out["safety_releases"] = float(r["events"]["safety"].sum())
    return out


def cheater_probe(kind, games, seed, minutes=3.0):
    half = games // 2
    pts = []
    for flip in (False, True):
        bot = RSProController(level=5, seed=seed)
        cheat = Cheater(kind, seed=seed)
        red, blue = (cheat, bot) if flip else (bot, cheat)
        r = play(red, blue, n_envs=half, match_ticks=int(minutes * 3600), seed=seed + (1 if flip else 0))
        p = points(r["score"])
        pts.append(1.0 - p if flip else p)
    pts = np.concatenate(pts)
    return dict(points_l5=float(pts.mean()), ci95=_bootstrap(pts, seed=seed))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="reports/rs4z/rspro_ladder.json")
    ap.add_argument("--skip-cheaters", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    report = dict(version="RS4-Z-rspro-ladder-1", games=args.games, ladder={}, transitive={}, mirror={}, cheaters={})
    for k in range(1, 6):
        report["ladder"][f"L{k}_vs_L{k - 1}"] = match_levels(k, k - 1, args.games, args.seed + 10 * k)
        print(f"L{k} vs L{k - 1}", report["ladder"][f"L{k}_vs_L{k - 1}"], flush=True)
    for k in range(0, 4):
        report["transitive"][f"L5_vs_L{k}"] = match_levels(5, k, args.games // 2, args.seed + 100 + k)
        print(f"L5 vs L{k}", report["transitive"][f"L5_vs_L{k}"], flush=True)
    for lv in (3, 5):
        report["mirror"][f"L{lv}"] = mirror_metrics(lv, args.games // 2, args.seed + 200 + lv)
        print(f"mirror L{lv}", json.dumps({k: round(v, 3) for k, v in report["mirror"][f"L{lv}"].items()}), flush=True)
    if not args.skip_cheaters:
        for kind in KINDS:
            report["cheaters"][kind] = cheater_probe(kind, args.games // 2, args.seed + 300)
            print("cheater", kind, report["cheaters"][kind], flush=True)
    report["seconds"] = time.time() - t0
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
