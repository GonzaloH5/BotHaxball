"""Cola pre-registrada del pod: preflight → RL principal → (recuperación si se corta) → extensión → certificación.

Corre sola durante horas y no deja decisiones para después del lanzamiento: cada paso tiene su condición y su
acción escritas acá (docs/PRELANZAMIENTO.md §Plan de corrida). Es idempotente: guarda el estado en
`<root>/queue_state.json` y, si se relanza (corte de luz, reinicio del pod), saltea lo terminado y reanuda la corrida en
curso con `--resume`. Cada proceso tiene su bucle de reintentos acotado.

Pasos:
1. preflight (learn/x4_preflight.py). Si falla, la cola se detiene: no se gasta GPU con un sistema roto.
2. Selección temprana del shaping (A/B a escala real, ~10% del cómputo): `rl_epv` (shaping de potencial con el valor de
   posesión) y `rl_checkpoint` (franjas de GRF) en paralelo hasta la actualización `--ab-updates` (300) con la mitad de
   partidos y de hilos cada uno. Gana el de mayor índice de la cadena de pase (media de sus últimas 3 evaluaciones: una
   sola tiene un IC90 de ±0,04–0,06, reports/x4/ab_cpu_reeval.json), salvo que en su última evaluación pierda contra la
   BC (< 0,45) o tenga deriva; con diferencia menor a 0,03, el EPV (no cambia la política óptima). En CPU, a una escala
   200 veces menor, los dos quedaron dentro del ruido.
3. `principal` = el ganador, reanudado con todos los partidos hasta `--updates` (6000). El brazo de pases se decide adentro,
   una sola vez, en la primera evaluación desde la actualización 1000 (`--auto-pass-arm-update`).
4. Si `principal` se cortó por deriva (stopped.json): `recuperacion` desde su best_pase.pt / best.pt (o la BC) con
   λ 0,4 y lr 1e-4, `--updates` (6000), con el brazo de pases ya activo si el principal lo había activado. Si también se
   corta, la cola termina en "revisar" sin más cómputo.
5. Si la corrida terminó sin corte y ningún checkpoint aprobó el gate de pases pero el índice de la cadena sigue
   subiendo (pendiente positiva en las últimas 8 evaluaciones), una sola extensión hasta `--extend-to` (9000).
6. Certificación (learn/x4_certify.py) de los candidatos: los `pase_aprobado_*.pt` (el último), best_pase.pt y best.pt.
   Un candidato aprueba si pasa la certificación y una confirmación con otras semillas (`--confirm-seed`): con varios
   candidatos y una sola certificación cada uno, un agente algo por debajo del humano promedio tenía varias chances de
   pasar por ruido. El primero que aprueba se exporta a ONNX como `deploy/rs4z/x4_rl.onnx`; si ninguno aprueba, se
   exporta igual el de mayor índice de la cadena para pruebas, marcado como NO competitivo en `queue_state.json`.

Fallas técnicas: si un entrenamiento termina con error después de sus reintentos sin llegar a su última actualización
(y sin corte por deriva), queda `fallo_tecnico` en `queue_state.json`, no se extiende y, si nada aprueba, el veredicto
es "revisar" (no "no_competitivo": no se llegó a medir lo planeado). Una certificación que se cae (código distinto de
0 y 3) se reintenta una vez y, si vuelve a fallar, no cuenta como rechazo sino como error.

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
CERT_NOT_APPROVED = 3      # learn/x4_certify.py NOT_APPROVED: "no aprueba"; cualquier otro código ≠ 0 es una falla
AB_LAST = 3                # evaluaciones finales de cada brazo del A/B que se promedian


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
            if i + 1 < retries:
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
        extra = list(extra) + a.rl_extra.split()
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

    def progress(self, name):
        """Última actualización que el trainer dejó registrada en <root>/<name>/log.jsonl (0 si ninguna)."""
        p = self.root / name / "log.jsonl"
        last = 0
        if p.exists():
            for line in p.read_text().splitlines():
                try:
                    last = max(last, int(json.loads(line).get("update", 0)))
                except (ValueError, AttributeError):
                    continue                    # línea cortada por un corte de luz
        return last

    def check_finished(self, step, name, target, code):
        """¿El entrenamiento `name` llegó a `target` o se cortó por deriva? Si no, deja constancia de la falla técnica."""
        if (self.root / name / "stopped.json").exists():
            return True
        done = self.progress(name)
        if done >= target:
            return True
        self.state.setdefault("fallo_tecnico", {})[step] = dict(codigo=code, actualizacion=done, objetivo=target)
        self.note(f"{step} terminó con error sin llegar a la actualización {target} (quedó en {done}): fallo técnico, "
                  "se certifica lo que haya y el veredicto será 'revisar' si nada aprueba", codigo=code)
        return False

    def choose_shaping(self):
        """Regla pre-registrada del A/B temprano (paso 2)."""
        if "shaping_elegido" in self.state:
            return self.state["shaping_elegido"]
        rows = {}
        for kind in ("epv", "checkpoint"):
            e = self.last_eval(f"rl_{kind}")
            ok = e is not None and not e.get("drift") and e["vs_bc"]["score"] >= 0.45 \
                and not (self.root / f"rl_{kind}" / "stopped.json").exists()
            # media de las últimas evaluaciones con índice: una sola es demasiado ruidosa para una diferencia de 0,03
            last = [x["cadena_pase"]["indice"] for x in self.evals(f"rl_{kind}")
                    if not x.get("baseline") and (x.get("cadena_pase") or {}).get("indice") is not None][-AB_LAST:]
            idx = round(float(np.mean(last)), 4) if last else None
            rows[kind] = dict(valido=bool(ok), indice=idx, evaluaciones=len(last),
                              vs_bc=(e or {}).get("vs_bc", {}).get("score"))
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
        """Filas de <root>/<name>/eval.jsonl, una por actualización: si el trainer se cayó entre escribir una evaluación y
        guardar last.pt, al reanudar la repite; queda la última."""
        p = self.root / name / "eval.jsonl"
        if not p.exists():
            return []
        rows = {}
        for line in p.read_text().splitlines():
            try:
                e = json.loads(line)
            except ValueError:
                continue                        # línea cortada por un corte de luz
            rows[(e.get("update"), bool(e.get("baseline")))] = e
        return sorted(rows.values(), key=lambda e: (e.get("update") or 0, not e.get("baseline")))

    def main(self):
        a = self.a
        # 1. preflight
        code = self.run("preflight", ["learn.x4_preflight", "--bc", a.bc, "--device", a.device,
                                      "--out", self.root / "preflight.json", *a.preflight_extra.split()], retries=1)
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
        finished = True
        if (self.root / main / "stopped.json").exists():
            self.note(f"el ganador {main} quedó cortado por deriva en el A/B: va a la recuperación")
        else:
            code = self.run(main + "_full", self.rl_cmd(main, ["--shaping-kind", pick]))
            finished = self.check_finished(main + "_full", main, a.updates, code)
        chosen = [main]
        if (self.root / main / "stopped.json").exists():
            # 3. recuperación pre-registrada
            init = next((p for p in (self.root / main / "best_pase.pt", self.root / main / "best.pt") if p.exists()), None)
            extra = ["--lambda-dist", "0.4", "--lr", "1e-4", "--shaping-kind", pick] + (["--init", init] if init else [])
            # si el principal ya había activado el brazo de pases, la recuperación sigue con la misma recompensa
            arm = next((e["brazo_pases_activado"] for e in self.evals(main) if e.get("brazo_pases_activado")), None)
            if arm:
                extra += ["--pass-bonus", arm["pass_bonus"]]
            rec = "rl_recuperacion"
            self.note("principal cortado por deriva: recuperación", init=str(init), brazo_pases=bool(arm))
            code = self.run(rec, self.rl_cmd(rec, extra))
            chosen.append(rec)
            if (self.root / rec / "stopped.json").exists():
                self.note("la recuperación también se cortó: la cola termina en REVISAR (sin más cómputo)")
                self.state["veredicto"] = "revisar"
                self.certify(chosen)
                return 2
            self.check_finished(rec, rec, a.updates, code)
        elif finished:
            # 4. extensión única si no hay aprobado y el índice sigue subiendo (no si la corrida se cayó: se repetiría)
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
                    code = self.run(last + "_ext", self.rl_cmd(last, ["--shaping-kind", pick], updates=a.extend_to))
                    self.check_finished(last + "_ext", last, a.extend_to, code)
        return self.certify(chosen)

    def certify_one(self, name, ckpt, extra=()):
        """Una certificación (learn/x4_certify.py): (código, reporte). 0 aprueba, CERT_NOT_APPROVED no aprueba; otro
        código es una falla de ejecución (se reintenta una vez)."""
        a = self.a
        out = self.root / f"{name}.json"
        code = self.run(name, ["learn.x4_certify", "--ckpt", ckpt, "--bc", a.bc, "--device", a.device, "--out", out,
                               *a.cert_extra.split(), *extra], retries=2, ok_codes=(0, CERT_NOT_APPROVED))
        rep = json.loads(out.read_text()) if out.exists() else {}
        return code, rep

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
            code, rep = self.certify_one(name, c)
            if code == 0:
                # confirmación con otras semillas: aprueba sólo si pasa las dos (con varios candidatos, una sola
                # certificación le daba a un agente algo por debajo del promedio humano varias chances de pasar por ruido)
                code2, rep2 = self.certify_one(name + "_confirmacion", c, ["--seed", a.confirm_seed])
                rep = dict(rep, confirmacion=dict(codigo=code2, aprobado=rep2.get("aprobado"),
                                                  indice_cadena=(rep2.get("gate_sanguchito") or {}).get("indice_cadena")))
                if code2 != 0:
                    self.note(f"{name}: aprobó pero no se confirmó con otras semillas", codigo=code2)
                    code = code2
            results.append((c, code, rep))
            if code == 0:
                break
        approved = [(c, rep) for c, code, rep in results if code == 0]
        errors = [str(c) for c, code, _ in results if code not in (0, CERT_NOT_APPROVED)]
        if errors:
            self.state["certificacion_con_error"] = errors
        if approved:
            c, rep = approved[0]
            self.state["veredicto"] = "competitivo_en_pases"
        elif results:
            c, rep = max(results, key=lambda x: (x[2].get("gate_sanguchito") or {}).get("indice_cadena") or -1)[::2]
            # con una falla técnica o una certificación caída no se llegó a medir lo planeado: "revisar", no "no competitivo"
            technical = bool(self.state.get("fallo_tecnico")) or bool(errors)
            self.state["veredicto"] = self.state.get("veredicto") or ("revisar" if technical else "no_competitivo")
        else:
            self.note("no hay candidatos para certificar")
            self.state["veredicto"] = self.state.get("veredicto") or "revisar"
            self.save()
            return 2
        self.state["elegido"] = str(c)
        self.save()
        self.run("export", ["export.to_onnx_x4", "--ckpt", c, "--out", a.export_to], retries=1)
        self.note("fin de la cola", veredicto=self.state["veredicto"], elegido=str(c))
        return 0 if approved else 2


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(ROOT / "runs" / "x4_cola"))
    ap.add_argument("--bc", default=str(BC))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--envs", type=int, default=1024)
    ap.add_argument("--rollout", type=int, default=64)
    ap.add_argument("--updates", type=int, default=6000,
                    help="actualizaciones del RL principal (6000 desde el 2026-10-06, pedido del usuario: antes 3000)")
    ap.add_argument("--extend-to", type=int, default=9000)
    ap.add_argument("--ab-updates", type=int, default=300, help="actualizaciones del A/B temprano del shaping")
    ap.add_argument("--export-to", default=str(ROOT / "deploy" / "rs4z" / "x4_rl.onnx"))
    ap.add_argument("--confirm-seed", type=int, default=20261007,
                    help="semilla de la certificación de confirmación (la primera usa la de learn/x4_certify.py)")
    ap.add_argument("--rl-extra", default="", help="argumentos extra para learn.x4_ppo (sólo para pruebas en seco)")
    ap.add_argument("--cert-extra", default="", help="argumentos extra para learn.x4_certify (sólo para pruebas)")
    ap.add_argument("--preflight-extra", default="", help="argumentos extra para el preflight (sólo para pruebas)")
    a = ap.parse_args()
    sys.exit(Queue(a).main())


if __name__ == "__main__":
    main()
