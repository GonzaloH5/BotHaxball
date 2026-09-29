"""Mide en qué se va el tiempo de una iteración de entrenamiento (para decidir si conviene una GPU).

python -m tools.profile_train [--config train/config.yaml] [--iters 6] [--override ppo.device=cuda ...]

Separa: física (sim.step), armado de obs (env.observe), recompensas/potenciales, inferencia de la red
en el rollout (act, sin el bot scripteado), bot scripteado, update de PPO y el resto (GAE, buffers, env).
La primera iteración se descarta (compilación de numba, calentamiento).
"""
from __future__ import annotations

import argparse
import shutil
import time
from collections import defaultdict
from pathlib import Path

import yaml

import train.ppo_selfplay as tp

ROOT = Path(__file__).resolve().parent.parent


class _Stop(Exception):
    pass


def timed(acc, key, fn):
    def wrapper(*a, **k):
        t = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            acc[key] += time.perf_counter() - t
    return wrapper


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "train" / "config.yaml"))
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--override", nargs="*", default=[])
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    for ov in args.override:
        k, v = ov.split("=", 1)
        d = cfg
        *path, last = k.split(".")
        for part in path:
            d = d[part]
        d[last] = yaml.safe_load(v)

    run = "_profile_tmp"
    trainer = tp.Trainer(cfg, run, resume=False)
    acc = defaultdict(float)
    per_iter = []

    env = trainer.env
    env.sim.step = timed(acc, "física (sim.step)", env.sim.step)
    env.observe = timed(acc, "obs (env.observe)", env.observe)
    env._potentials = timed(acc, "recompensas (potenciales)", env._potentials)
    tp.scripted_actions = timed(acc, "bot scripteado", tp.scripted_actions)
    trainer.act = timed(acc, "_act_total", trainer.act)
    orig_update = trainer.update
    state = {"it": 0, "t": time.perf_counter(), "samples": 0}

    def update(obs, *a, **k):
        acc_before = dict(acc)
        t = time.perf_counter()
        out = orig_update(obs, *a, **k)
        acc["update PPO"] += time.perf_counter() - t
        now = time.perf_counter()
        per_iter.append((now - state["t"], dict(acc), len(obs)))
        state["t"] = now
        state["it"] += 1
        if state["it"] > args.iters:
            raise _Stop
        return out

    trainer.update = update
    print(f"perfilando {args.iters} iteraciones (+1 de calentamiento) | dispositivo {trainer.device} | "
          f"{trainer.N} partidos {trainer.T}v{trainer.T} en {trainer.env_settings()['stadium']}", flush=True)
    try:
        trainer._train_phase()
    except _Stop:
        pass
    finally:
        trainer.writer.close()
        shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)

    # descartar la primera iteración (calentamiento): restar sus acumulados
    (_, acc0, _), rest = per_iter[0], per_iter[1:]
    wall = sum(w for w, _, _ in rest)
    final = rest[-1][1]
    tot = {k: final.get(k, 0.0) - acc0.get(k, 0.0) for k in final}
    n = len(rest)
    samples = sum(s for _, _, s in rest)
    tot["inferencia de la red (act)"] = tot.pop("_act_total", 0.0) - tot.get("bot scripteado", 0.0)
    tot["resto (GAE, buffers, env)"] = wall - sum(tot.values())
    print(f"\n{'parte':32s} {'s/iter':>8s} {'%':>6s}")
    for k, v in sorted(tot.items(), key=lambda kv: -kv[1]):
        print(f"{k:32s} {v / n:8.2f} {100 * v / wall:6.1f}")
    print(f"{'TOTAL':32s} {wall / n:8.2f}  | {samples / wall:,.0f} muestras/s ({samples // n:,} por iteración)")


if __name__ == "__main__":
    main()
