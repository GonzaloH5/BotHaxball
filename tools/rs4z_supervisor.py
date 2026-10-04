"""Supervisor del entrenamiento largo RS4-Z: corre el principal y, desde S6, sus exploiters.

  python -m tools.rs4z_supervisor --run rs4z_main [--envs 1024] [--exploiter-every 1e9] [--exploiter-samples 2e8]

- El principal (`train.rs4z.run --run NAME --resume`) escribe `runs/rs4z/NAME/status.json` en cada guardado.
- En S6/S7, cada `--exploiter-every` muestras del principal se copia su `latest.pt` a `exploiter_base_K.pt` y
  se entrena un exploiter (AlphaStar) contra esa copia congelada, en paralelo y con menos hilos. Al terminar,
  el exploiter se evalúa contra el principal y entra a su liga sólo si le saca ≥ 60% de los puntos
  (`exploiters.json`; el principal lo incorpora en su siguiente guardado).
- Un exploiter a la vez. Si el principal termina, se espera al exploiter en curso y se sale.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable


def _status(run_dir):
    try:
        return json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _launch(args, log, threads, gpu=None):
    env = dict(os.environ, OMP_NUM_THREADS="1", NUMBA_NUM_THREADS=str(threads))
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    fh = open(log, "a", encoding="utf-8")
    return subprocess.Popen([PY, "-u", "-m", "train.rs4z.run", *args], cwd=ROOT, env=env, stdout=fh,
                            stderr=subprocess.STDOUT), fh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--envs", type=int, default=1024)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--main-threads", type=int, default=8)
    ap.add_argument("--exploiter-threads", type=int, default=4)
    ap.add_argument("--main-gpu", type=int, default=None, help="GPU del principal (por defecto, la que vea)")
    ap.add_argument("--exploiter-gpu", type=int, default=None, help="GPU de los exploiters (otra libera al principal)")
    ap.add_argument("--exploiter-envs", type=int, default=256)
    ap.add_argument("--exploiter-every", type=float, default=1e9)
    ap.add_argument("--exploiter-samples", type=float, default=2e8)
    ap.add_argument("--poll", type=float, default=60.0)
    ap.add_argument("--extra", default="", help="argumentos adicionales para el principal (entre comillas)")
    args = ap.parse_args()

    run_dir = ROOT / "runs" / "rs4z" / args.run
    logs = ROOT / "pod_logs"
    logs.mkdir(exist_ok=True)
    state_path = run_dir / "supervisor.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else dict(next_at=None, k=0)

    main_args = ["--run", args.run, "--envs", str(args.envs), "--device", args.device, "--resume", *args.extra.split()]
    main, main_fh = _launch(main_args, logs / f"{args.run}.log", args.main_threads, args.main_gpu)
    print(f"principal pid {main.pid}", flush=True)
    exploiter = None
    try:
        while True:
            time.sleep(args.poll)
            st = _status(run_dir)
            if exploiter is not None and exploiter[0].poll() is not None:
                print(f"exploiter {state['k']} terminó con código {exploiter[0].returncode}", flush=True)
                exploiter[1].close()
                exploiter = None
            if main.poll() is not None:
                print(f"principal terminó con código {main.returncode}", flush=True)
                break
            if st is None or st["stage"] not in ("S6", "S7") or exploiter is not None:
                continue
            if state["next_at"] is None:
                state["next_at"] = st["samples"]            # el primero apenas empieza S6
            if st["samples"] < state["next_at"]:
                continue
            state["k"] += 1
            k = state["k"]
            base = run_dir / f"exploiter_base_{k:02d}.pt"
            shutil.copyfile(run_dir / "latest.pt", base)
            ex_args = ["--run", f"{args.run}_exploiter_{k:02d}", "--exploiter-of", str(base), "--envs",
                       str(args.exploiter_envs), "--device", args.device, "--samples", str(args.exploiter_samples)]
            exploiter = _launch(ex_args, logs / f"{args.run}_exploiter_{k:02d}.log", args.exploiter_threads,
                                args.exploiter_gpu)
            state["next_at"] = st["samples"] + args.exploiter_every
            state_path.write_text(json.dumps(state), encoding="utf-8")
            print(f"exploiter {k} contra {st['samples'] / 1e9:.2f}B (pid {exploiter[0].pid})", flush=True)
    finally:
        if exploiter is not None:
            exploiter[0].wait()
            exploiter[1].close()
        main_fh.close()


if __name__ == "__main__":
    main()
