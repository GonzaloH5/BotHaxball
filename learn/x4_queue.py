"""Cola pre-registrada del pod: preflight → RL principal → (recuperación si se corta) → extensión → certificación.

Corre sola durante horas y no deja decisiones para después del lanzamiento: cada paso tiene su condición y su
acción escritas acá (docs/PRELANZAMIENTO.md §Plan de corrida). Es idempotente: guarda el estado en
`<root>/queue_state.json` y, si se relanza (corte de luz, reinicio del pod), saltea lo terminado y reanuda la corrida en
curso con `--resume`. Cada proceso tiene su bucle de reintentos acotado.

Pasos:
1. preflight (learn/x4_preflight.py). Si falla, la cola se detiene: no se gasta GPU con un sistema roto.
2. Selección temprana del shaping (A/B a escala real, ~10% del cómputo): `rl_epv` (shaping de potencial con el valor de
   posesión) y `rl_checkpoint` (franjas de GRF) en paralelo hasta la actualización `--ab-updates` (300) con la mitad de
   partidos y de hilos cada uno. Gana el de mayor índice de la cadena de pase en su última evaluación, salvo que pierda
   contra la BC (< 0,45) o tenga deriva; con diferencia menor a 0,03, el EPV (no cambia la política óptima). En CPU, a una
   escala 200 veces menor, los dos quedaron dentro del ruido (reports/x4/ab_cpu_reeval.json).
3. `principal` = el ganador, reanudado con todos los partidos hasta `--updates` (3000). El brazo de pases se decide adentro,
   en la actualización 1000 (`--auto-pass-arm-update`).
4. Si `principal` se cortó por deriva (stopped.json): `recuperacion` desde su best_pase.pt / best.pt (o la BC) con
   λ 0,4 y lr 1e-4, `--updates` (3000). Si también se corta, la cola termina en "revisar" sin más cómputo.
5. Si la corrida terminó sin corte y ningún checkpoint aprobó el gate de pases pero el índice de la cadena sigue
   subiendo (pendiente positiva en las últimas 8 evaluaciones), una sola extensión hasta `--extend-to` (6000).
6. Certificación (learn/x4_certify.py) de los candidatos: los `pase_aprobado_*.pt` (el último), best_pase.pt y best.pt.
   El primero que aprueba se exporta a ONNX como `deploy/rs4z/x4_rl.onnx`; si ninguno aprueba, se exporta igual el de
   mayor índice de la cadena para pruebas, marcado como NO competitivo en `queue_state.json`.

  tmux new -s cola
  export NUMBA_NUM_THREADS=8 OMP_NUM_THREADS=1
  python -m learn.x4_queue --root runs/x4_cola --device cuda 2>&1 | tee -a runs/x4_cola.console.log
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BC = ROOT / "runs" / "x4_bc" / "final_sangu_rsone" / "best.pt"


class Queue:
    def __init__(self, a):
        self.a = a
        self.root = Path(a.root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / "queue_state.json"
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else dict(done={}, log=[])

    def note(self, msg, **kw):
        row = dict(time=time.strftime("%Y-%m-%dT%H:%M:%S"), msg=msg, **kw)
        self.state["log"].append(row)
        self.save()
        print("COLA " + json.dumps(row, ensure_ascii=False), flush=True)

    def save(self):
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=1, ensure_ascii=False))
        tmp.replace(self.state_path)

    def run(self, name, cmd, retries=5, ok_codes=(0,)):
        """Corre `cmd` con reintentos; devuelve el código de salida final."""
        if name in self.state["done"]:
            return self.state["done"][name]
        code = None
        for i in range(retries):
            self.note(f"inicia {name} (intento {i + 1})", cmd=" ".join(map(str, cmd)))
            code = subprocess.run([sys.executable, "-m", *map(str, cmd)], cwd=ROOT).returncode
            if code in ok_codes:
                break
            out = Path(str(cmd[cmd.index("--out") + 1])) if "--out" in cmd else None
            if out is not None and (out / "stopped.json").exists():
                break                       # cortada por deriva: no se reintenta (el trainer no reanuda)
            self.note(f"{name} terminó con código {code}; reintento en 60 s")
            time.sleep(60)
        self.state["done"][name] = code
        self.save()
        return code

    def rl_cmd(self, name, extra, envs=None, updates=None):
        a = self.a
        cmd = ["learn.x4_ppo", "--bc", a.bc, "--out", self.root / name, "--device", a.device, "--envs", envs or a.envs,
               "--rollout", a.rollout, "--updates", updates or a.updates, "--lambda-dist", "0.2", "--lambda-decay",
               "0.9995", "--lambda-min", "0.05", "--critic-warmup", "20", "--human-starts", "0.4", "--pool-frac", "0.2",
               "--eval-every", "50", "--resume"]
        # los extras pisan a los valores por defecto (p. ej. --lambda-dist 0.4 en la recuperación)
        for i in range(0, len(extra), 2):
            k = extra[i]
            if k in cmd and i + 1 < len(extra) and not str(extra[i + 1]).startswith("--"):
                cmd[cmd.index(k) + 1] = extra[i + 1]
            else:
                cmd += extra[i:i + 2]
        return cmd

    def run_parallel(self, jobs):
        """Corre varios entrenamientos a la vez, cada uno con su parte de los hilos de numba, con reintentos."""
        import os
        pending = [(n, c) for n, c in jobs if n not in self.state["done"]]
        if not pending:
            return
        total = int(os.environ.get("NUMBA_NUM_THREADS", "8"))
        env = dict(os.environ, NUMBA_NUM_THREADS=str(max(1, total // len(pending))))
        tries = {n: 0 for n, _ in pending}
        procs = {}
        while pending or procs:
            for n, c in list(pending):
                self.note(f"inicia {n} en paralelo (intento {tries[n] + 1})", cmd=" ".join(map(str, c)))
                procs[n] = (subprocess.Popen([sys.executable, "-m", *map(str, c)], cwd=ROOT, env=env), c)
                pending.remove((n, c))
            time.sleep(30)
            for n, (p, c) in list(procs.items()):
                code = p.poll()
                if code is None:
                    continue
                del procs[n]
                tries[n] += 1
                if code == 0 or (self.root / n / "stopped.json").exists() or tries[n] >= 5:
                    self.state["done"][n] = code
                    self.save()
                else:
                    self.note(f"{n} terminó con código {code}; reintento")
                    pending.append((n, c))

    def last_eval(self, name):
        ev = [e for e in self.evals(name) if not e.get("baseline")]
        return ev[-1] if ev else None

    def choose_shaping(self):
        """Regla pre-registrada del A/B temprano (paso 2)."""
        if "shaping_elegido" in self.state:
            return self.state["shaping_elegido"]
        rows = {}
        for kind in ("epv", "checkpoint"):
            e = self.last_eval(f"rl_{kind}")
            ok = e is not None and not e.get("drift") and e["vs_bc"]["score"] >= 0.45 \
                and not (self.root / f"rl_{kind}" / "stopped.json").exists()
            idx = (e or {}).get("cadena_pase", {}).get("indice")
            rows[kind] = dict(valido=bool(ok), indice=idx, vs_bc=(e or {}).get("vs_bc", {}).get("score"))
        valid = {k: v for k, v in rows.items() if v["valido"] and v["indice"] is not None}
        if not valid:
            pick = "epv" if not (self.root / "rl_epv" / "stopped.json").exists() else "checkpoint"
        elif len(valid) == 1:
            pick = next(iter(valid))
        else:
            pick = "checkpoint" if valid["checkpoint"]["indice"] > valid["epv"]["indice"] + 0.03 else "epv"
        self.state["shaping_elegido"] = pick
        self.state["ab_temprano"] = rows
        self.note("A/B temprano del shaping", elegido=pick, brazos=rows)
        return pick

    def evals(self, name):
        p = self.root / name / "eval.jsonl"
        return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []

    def main(self):
        a = self.a
        # 1. preflight
        code = self.run("preflight", ["learn.x4_preflight", "--bc", a.bc, "--device", a.device,
                                      "--out", self.root / "preflight.json"], retries=1)
        if code != 0:
            self.note("preflight con FALLA: la cola se detiene (ver preflight.json)")
            return 1
        # 2. A/B temprano del shaping, en paralelo con la mitad de los partidos cada uno
        half = max(64, a.envs // 2)
        self.run_parallel([(f"rl_{k}", self.rl_cmd(f"rl_{k}", ["--shaping-kind", k], envs=half, updates=a.ab_updates))
                           for k in ("epv", "checkpoint")])
        pick = self.choose_shaping()
        # 3. principal: el ganador sigue con todos los partidos (misma carpeta, --resume)
        main = f"rl_{pick}"
        if (self.root / main / "stopped.json").exists():
            self.note(f"el ganador {main} quedó cortado por deriva en el A/B: va a la recuperación")
        else:
            self.run(main + "_full", self.rl_cmd(main, ["--shaping-kind", pick]))
        chosen = [main]
        if (self.root / main / "stopped.json").exists():
            # 3. recuperación pre-registrada
            init = next((p for p in (self.root / main / "best_pase.pt", self.root / main / "best.pt") if p.exists()), None)
            extra = ["--lambda-dist", "0.4", "--lr", "1e-4", "--shaping-kind", pick] + (["--init", init] if init else [])
            rec = "rl_recuperacion"
            self.note("principal cortado por deriva: recuperación", init=str(init))
            self.run(rec, self.rl_cmd(rec, extra))
            chosen.append(rec)
            if (self.root / rec / "stopped.json").exists():
                self.note("la recuperación también se cortó: la cola termina en REVISAR (sin más cómputo)")
                self.state["veredicto"] = "revisar"
                self.certify(chosen)
                return 2
        else:
            # 4. extensión única si no hay aprobado y el índice sigue subiendo
            last = chosen[-1]
            ev = [e for e in self.evals(last) if (e.get("cadena_pase") or {}).get("indice") is not None]
            approved = list((self.root / last).glob("pase_aprobado_*.pt"))
            if not approved and len(ev) >= 8 and "extension" not in self.state:
                y = np.array([e["cadena_pase"]["indice"] for e in ev[-8:]])
                slope = float(np.polyfit(np.arange(len(y)), y, 1)[0])
                self.state["extension"] = dict(pendiente=slope, decide=slope > 0)
                self.save()
                if slope > 0:
                    self.note("sin aprobado y con el índice subiendo: extensión", pendiente=slope)
                    self.run(last + "_ext", self.rl_cmd(last, ["--shaping-kind", pick], updates=a.extend_to))
        return self.certify(chosen)

    def certify(self, runs):
        a = self.a
        cands = []
        for r in runs:
            d = self.root / r
            ap = sorted(d.glob("pase_aprobado_*.pt"))
            cands += ([ap[-1]] if ap else []) + [p for p in (d / "best_pase.pt", d / "best.pt") if p.exists()]
        results = []
        for c in cands:
            name = f"cert_{c.parent.name}_{c.stem}"
            out = self.root / f"{name}.json"
            code = self.run(name, ["learn.x4_certify", "--ckpt", c, "--bc", a.bc, "--device", a.device, "--out", out],
                            retries=1, ok_codes=(0, 1))
            rep = json.loads(out.read_text()) if out.exists() else {}
            results.append((c, code, rep))
            if code == 0:
                break
        approved = [(c, rep) for c, code, rep in results if code == 0]
        if approved:
            c, rep = approved[0]
            self.state["veredicto"] = "competitivo_en_pases"
        elif results:
            c, rep = max(results, key=lambda x: (x[2].get("gate_sanguchito") or {}).get("indice_cadena") or -1)[::2]
            self.state["veredicto"] = self.state.get("veredicto") or "no_competitivo"
        else:
            self.note("no hay candidatos para certificar")
            return 2
        self.state["elegido"] = str(c)
        self.save()
        self.run("export", ["export.to_onnx_x4", "--ckpt", c, "--out", ROOT / "deploy" / "rs4z" / "x4_rl.onnx"], retries=1)
        self.note("fin de la cola", veredicto=self.state["veredicto"], elegido=str(c))
        return 0 if approved else 2


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(ROOT / "runs" / "x4_cola"))
    ap.add_argument("--bc", default=str(BC))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--envs", type=int, default=1024)
    ap.add_argument("--rollout", type=int, default=64)
    ap.add_argument("--updates", type=int, default=3000)
    ap.add_argument("--extend-to", type=int, default=6000)
    ap.add_argument("--ab-updates", type=int, default=300, help="actualizaciones del A/B temprano del shaping")
    a = ap.parse_args()
    sys.exit(Queue(a).main())


if __name__ == "__main__":
    main()
