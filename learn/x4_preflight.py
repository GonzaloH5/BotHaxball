"""Preflight del pod antes de una corrida larga de RL: verifica lo necesario y corre una mini-corrida real.

Chequeos (cada uno OK / FALLA, código de salida ≠ 0 si alguno falla):
1. dispositivo (CUDA disponible si se pidió, nombre y memoria de la GPU);
2. datos: caché `data/x4_ticks`, particiones `reports/x4/splits.json`, grabaciones de entrenamiento de Sanguchito;
3. referencia humana `data/human_metrics_sanguchito.samples.npz` (sin ella best.pt nunca se guarda);
4. checkpoint de la imitación: carga con weights_only y la dimensión de la observación coincide con obs v3;
5. mini-corrida del trainer en el dispositivo (StateBank, rollouts, actualización, snapshot, evaluación y reanudación),
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

    # 3. referencia humana
    from learn.x4_eval import HUMAN_SAMPLES
    check("referencia_humana", HUMAN_SAMPLES.exists(), str(HUMAN_SAMPLES))

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
            check("mini_corrida", len(rows) == 3 and len(evals) == 1 and np.isfinite(last["kl_bc"]), detail)
            tr2 = Trainer(parse_args(args[:-6] + ["--updates", "4", "--resume", "--eval-every", "0",
                                                  "--bank-recordings", "0"]))
            check("reanudacion", tr2.update == 3, f"reanuda en la actualización {tr2.update}")
            if a.scale_test == "yes" or (a.scale_test == "auto" and a.device.startswith("cuda")):
                scale_test(tr2, a, check)
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
        check("escala_real", np.isfinite(info["kl_bc"]), detail)
    except Exception as e:  # noqa: BLE001  (p. ej. falta de memoria: bajar --rollout a 32)
        check("escala_real", False, f"{type(e).__name__}: {e} (probar --rollout 32 en la corrida real)")


def finish(report, ok, a):
    if a.out:
        Path(a.out).write_text(json.dumps(dict(ok=ok, checks=report), indent=1, ensure_ascii=False), encoding="utf-8")
    print("PREFLIGHT " + ("OK" if ok else "FALLA"), flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
