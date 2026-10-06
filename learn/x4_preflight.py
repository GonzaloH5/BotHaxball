"""Preflight del pod antes de una corrida larga de RL: verifica lo necesario y corre una mini-corrida real.

Chequeos (cada uno OK / FALLA, código de salida ≠ 0 si alguno falla):
1. dispositivo (CUDA disponible si se pidió, nombre y memoria de la GPU);
2. datos: caché `data/x4_ticks`, particiones `reports/x4/splits.json`, grabaciones de entrenamiento de Sanguchito;
3. referencia humana `data/human_metrics_sanguchito.samples.npz` (sin ella best.pt nunca se guarda);
4. checkpoint de la imitación: carga con weights_only y la dimensión de la observación coincide con obs v3;
5. mini-corrida del trainer en el dispositivo (StateBank, rollouts con shaping de valor, actualización, snapshot,
   evaluación 0 y periódica con la cadena de pase, reanudación),
   con throughput, RAM y memoria de GPU, y una estimación del tiempo por actualización con la configuración real.

  python -m learn.x4_preflight --bc runs/x4_bc/final_sangu_rsone/best.pt --device cuda
"""
from __future__ import annotations

import argparse
import json
import resource
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bc", default=str(ROOT / "runs" / "x4_bc" / "final_sangu_rsone" / "best.pt"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--envs", type=int, default=128, help="partidos de la mini-corrida")
    ap.add_argument("--target-envs", type=int, default=1024, help="partidos de la corrida real (para la estimación)")
    ap.add_argument("--target-rollout", type=int, default=64)
    ap.add_argument("--scale-test", choices=("auto", "yes", "no"), default="auto",
                    help="una actualización sintética con el buffer de la corrida real (auto: sólo en cuda)")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    report, ok = {}, True

    def check(name, cond, detail):
        nonlocal ok
        report[name] = dict(ok=bool(cond), detail=detail)
        print(f"{'OK   ' if cond else 'FALLA'} {name}: {detail}", flush=True)
        ok &= bool(cond)

    # 1. dispositivo
    if a.device.startswith("cuda"):
        avail = torch.cuda.is_available()
        detail = (f"{torch.cuda.get_device_name(0)}, {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GB"
                  if avail else "torch.cuda.is_available() es False")
        check("dispositivo", avail, detail)
        if not avail:
            return finish(report, ok, a)
    else:
        check("dispositivo", True, "cpu (sólo para probar; el RL de verdad necesita la GPU)")

    # 1b. hilos, disco y versión del código
    import os
    import shutil
    import subprocess
    try:
        import numba
        nth = numba.get_num_threads()
    except Exception:  # noqa: BLE001
        nth = None
    try:
        cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        cpus = os.cpu_count()
    check("hilos", nth is not None and nth <= max(1, cpus) and os.environ.get("NUMBA_NUM_THREADS") is not None,
          f"numba {nth} hilos (NUMBA_NUM_THREADS={os.environ.get('NUMBA_NUM_THREADS')}), CPUs visibles {cpus}, "
          f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}")
    free = shutil.disk_usage(ROOT).free / 2**30
    check("disco", free >= 5.0, f"{free:.1f} GB libres (hacen falta ≥ 5: checkpoints, evals/ y logs)")
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "env", "learn", "tools"], cwd=ROOT,
                           capture_output=True, text=True).stdout.strip()
    report["codigo"] = dict(head=subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                                text=True).stdout.strip(), cambios_sin_commitear=dirty)
    print(("AVISO cambios sin commitear:\n" + dirty) if dirty else "código: sin cambios sin commitear", flush=True)

    # 2. datos
    ticks = sorted((ROOT / "data" / "x4_ticks").glob("*.npz"))
    check("cache_x4_ticks", len(ticks) >= 600, f"{len(ticks)} grabaciones en data/x4_ticks (se esperan ~647)")
    splits = ROOT / "reports" / "x4" / "splits.json"
    check("particiones", splits.exists(), str(splits))
    if splits.exists():
        from learn import x4_data as XD
        names = XD.split_names(splits, "train", None, ("sanguchito_rs_x4",))
        have = [n for n in names if (ROOT / "data" / "x4_ticks" / f"{Path(n).stem}.npz").exists()]
        check("entrenamiento_sanguchito", len(have) >= 300, f"{len(have)} de {len(names)} grabaciones con caché")

    # 3. referencias humanas y valor de posesión
    from learn.x4_eval import HUMAN_SAMPLES
    check("referencia_humana", HUMAN_SAMPLES.exists(), str(HUMAN_SAMPLES))
    pc_ref = ROOT / "reports" / "x4" / "pass_chain_human.json"
    ok_ref = pc_ref.exists() and "sanguchito_test" in json.loads(pc_ref.read_text(encoding="utf-8"))
    check("referencia_cadena_pase", ok_ref, str(pc_ref))
    epv_path = ROOT / "runs" / "x4_epv" / "epv.pt"
    try:
        from learn.x4_epv import EPV
        e = EPV(epv_path)
        v = e.phi(np.array([[1050.0, 0, 0, 0], [-1050.0, 0, 0, 0]]),
                  np.tile(np.array([[1020.0, 0], [-600, 200], [-600, -200], [-900, 0],
                                    [300, 0], [600, 200], [600, -200], [-1020, 0]]), (2, 1, 1)),
                  np.zeros((2, 8, 2)), np.zeros(2, int), np.full(2, -1), np.zeros(2, bool))
        check("valor_de_posesion", v[0] > 0 and v[1] < 0, f"{epv_path}: φ(pelota en el arco azul) = {v[0]:.3f}, "
                                                       f"en el rojo = {v[1]:.3f}")
    except Exception as e:  # noqa: BLE001
        check("valor_de_posesion", False, f"{type(e).__name__}: {e}")

    # 4. checkpoint de la imitación
    from env.rs4z import obs_v3
    from learn.x4_policy import SetPolicy
    try:
        ck = torch.load(a.bc, map_location="cpu")          # weights_only=True por defecto
        m = SetPolicy(hidden=ck.get("hidden", 256))
        m.load_state_dict(ck["model"])
        out = m(torch.zeros(2, obs_v3.OBS_DIM))
        check("checkpoint_bc", out.shape == (2, 18) and ck.get("obs_version") == obs_v3.OBS_VERSION,
              f"{a.bc}: obs {ck.get('obs_version')}, dim {ck.get('obs_dim')} (esperada {obs_v3.OBS_DIM})")
    except Exception as e:  # noqa: BLE001
        check("checkpoint_bc", False, f"{type(e).__name__}: {e}")
    if not ok:
        return finish(report, ok, a)

    # 5. mini-corrida
    from learn.x4_ppo import Trainer, parse_args
    with tempfile.TemporaryDirectory(prefix="x4-preflight-") as tmp:
        args = ["--bc", a.bc, "--out", tmp, "--device", a.device, "--envs", str(a.envs), "--rollout", "16",
                "--updates", "3", "--critic-warmup", "1", "--eval-every", "3", "--eval-matches", "2",
                "--eval-minutes", "0.3", "--eval-selfplay", "2", "--bank-recordings", "20", "--snapshot-every", "2"]
        try:
            t0 = time.time()
            tr = Trainer(parse_args(args))
            t_init = time.time() - t0
            if a.device.startswith("cuda"):
                torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            tr.train()
            t_train = time.time() - t0
            rows = [json.loads(l) for l in (Path(tmp) / "log.jsonl").read_text().splitlines()]
            evals = (Path(tmp) / "eval.jsonl").read_text().splitlines() if (Path(tmp) / "eval.jsonl").exists() else []
            last = rows[-1]
            per_dec = (last["rollout_s"] + last["learn_s"]) / (a.envs * 16)
            est = per_dec * a.target_envs * a.target_rollout   # lineal: cota pesimista (la GPU escala mejor)
            detail = dict(init_s=round(t_init, 1), train_s=round(t_train, 1), dps_ultima=last["dps"],
                          kl_bc=last["kl_bc"], evaluaciones=len(evals),
                          ram_pico_gb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2),
                          gpu_pico_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2) if a.device.startswith("cuda") else None,
                          seg_por_actualizacion_estimado=round(est, 1),
                          horas_3000_actualizaciones_estimado=round(est * 3000 / 3600, 1))
            ev_rows = [json.loads(l) for l in evals]
            pcs = [r.get("cadena_pase", {}).get("indice") for r in ev_rows]
            detail.update(indice_cadena_pase=pcs, epv_shaping_abs=last.get("epv_shaping_abs"))
            check("mini_corrida", len(rows) == 3 and len(evals) == 2 and ev_rows[0].get("baseline")
                  and all(p is not None for p in pcs) and np.isfinite(last["kl_bc"]), detail)
            tr2 = Trainer(parse_args(args[:-6] + ["--updates", "4", "--resume", "--eval-every", "0",
                                                  "--bank-recordings", "0"]))
            check("reanudacion", tr2.update == 3, f"reanuda en la actualización {tr2.update}")
            if a.scale_test == "yes" or (a.scale_test == "auto" and a.device.startswith("cuda")):
                scale_test(tr2, a, check)
                real_update(a, check)
        except SystemExit as e:
            check("mini_corrida", False, f"SystemExit: {e}")
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            check("mini_corrida", False, f"{type(e).__name__}: {e}")
    return finish(report, ok, a)


def scale_test(tr, a, check):
    """Una actualización (`learn`) con un buffer sintético del tamaño de la corrida real: mide el pico de RAM y de memoria
    de GPU de la parte más grande (los tensores de un rollout completo suben juntos al dispositivo)."""
    from learn.x4_ppo import CRITIC_IN
    from env.rs4z import obs_v3
    T, N = a.target_rollout, a.target_envs
    rng = np.random.default_rng(0)
    try:
        if a.device.startswith("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        u = np.float32(np.log(1 / 18))
        buf = [(rng.standard_normal((T, N, 8, obs_v3.OBS_DIM), np.float32) * 0.1,
                rng.standard_normal((T, N, 8, CRITIC_IN), np.float32) * 0.1,
                rng.integers(0, 18, (T, N, 8)), np.full((T, N, 8), u, np.float32),
                np.full((T, N, 8, 18), u, np.float32), np.zeros((T, N, 8), np.float32),
                rng.standard_normal((T, N, 8)).astype(np.float32) * 0.01, rng.random((T, N)) < 0.001,
                np.ones((T, N, 8), bool), np.full((T, N), 0.2, np.float32), np.zeros((N, 8), np.float32))]
        t0 = time.time()
        info = tr.learn(buf)
        dt = time.time() - t0
        detail = dict(muestras=info["samples"], learn_s=round(dt, 1),
                      ram_pico_gb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2),
                      gpu_pico_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2) if a.device.startswith("cuda") else None)
        frac = (torch.cuda.max_memory_allocated() / torch.cuda.get_device_properties(0).total_memory
                if a.device.startswith("cuda") else 0.0)
        detail["gpu_fraccion"] = round(frac, 2)
        check("escala_real", np.isfinite(info["kl_bc"]) and frac < 0.85, detail)
    except Exception as e:  # noqa: BLE001  (p. ej. falta de memoria: bajar --rollout a 32)
        check("escala_real", False, f"{type(e).__name__}: {e} (probar --rollout 32 en la corrida real)")


def real_update(a, check):
    """Una actualización y una evaluación con la configuración real (1024 × 64, StateBank de 200 grabaciones,
    evaluación completa): tiempo por actualización y por evaluación, memoria de GPU reservada y horas estimadas."""
    from learn.x4_ppo import Trainer, parse_args
    with tempfile.TemporaryDirectory(prefix="x4-preflight-real-") as tmp:
        try:
            if a.device.startswith("cuda"):
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            tr = Trainer(parse_args(["--bc", a.bc, "--out", tmp, "--device", a.device, "--envs", str(a.target_envs),
                                     "--rollout", str(a.target_rollout), "--updates", "2", "--critic-warmup", "1",
                                     "--eval-every", "0"]))
            t_init = time.time() - t0
            times = []
            for u in range(2):
                tr.update = u + 1
                t1 = time.time()
                buf, _ = tr.rollout()
                t2 = time.time()
                tr.learn(buf, critic_only=u == 0)
                times.append((t2 - t1, time.time() - t2))
            tr.a.eval_every = 25
            tr.update = 25
            t3 = time.time()
            ev = tr.evaluate()
            t_eval = time.time() - t3
            per_update = sum(times[-1])
            hours = (3000 * per_update + 120 * t_eval) / 3600
            free, total = torch.cuda.mem_get_info() if a.device.startswith("cuda") else (0, 1)
            used = (total - free) / total if a.device.startswith("cuda") else 0.0
            detail = dict(init_s=round(t_init, 1), rollout_s=round(times[-1][0], 2), learn_s=round(times[-1][1], 2),
                          eval_s=round(t_eval, 1), horas_3000_con_evaluaciones=round(hours, 2),
                          gpu_reservada_gb=round(torch.cuda.max_memory_reserved() / 2**30, 2) if a.device.startswith("cuda") else None,
                          gpu_usada_frac=round(used, 2), cadena_pase=(ev.get("cadena_pase") or {}).get("indice"))
            check("actualizacion_real", used < 0.85 and hours < 24, detail)
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            check("actualizacion_real", False, f"{type(e).__name__}: {e} (probar --rollout 32)")


def finish(report, ok, a):
    if a.out:
        Path(a.out).write_text(json.dumps(dict(ok=ok, checks=report), indent=1, ensure_ascii=False), encoding="utf-8")
    print("PREFLIGHT " + ("OK" if ok else "FALLA"), flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
