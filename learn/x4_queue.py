"""Cola pre-registrada del pod: preflight → RL principal → (recuperación si se corta) → extensión → certificación.

Corre sola durante horas y no deja decisiones para después del lanzamiento: cada paso tiene su condición y su
acción escritas acá (docs/PRELANZAMIENTO.md §Plan de corrida). Es idempotente: guarda el estado en
`<root>/queue_state.json` y, si se relanza (corte de luz, reinicio del pod), saltea lo terminado y reanuda la corrida en
curso con `--resume`. Cada proceso tiene su bucle de reintentos acotado.

Pasos:
1. preflight (learn/x4_preflight.py). Si falla, la cola se detiene: no se gasta GPU con un sistema roto.
2. `principal`: RL con shaping de valor (EPV), λ 0,2 → 0,05, `--updates` (3000). El brazo de pases se decide adentro,
   en la actualización 1000 (`--auto-pass-arm-update`).
3. Si `principal` se cortó por deriva (stopped.json): `recuperacion` desde su best_pase.pt / best.pt (o la BC) con
   λ 0,4 y lr 1e-4, `--updates` (3000). Si también se corta, la cola termina en "revisar" sin más cómputo.
4. Si la corrida terminó sin corte y ningún checkpoint aprobó el gate de pases pero el índice de la cadena sigue
   subiendo (pendiente positiva en las últimas 8 evaluaciones), una sola extensión hasta `--extend-to` (6000).
5. Certificación (learn/x4_certify.py) de los candidatos: los `pase_aprobado_*.pt` (el último), best_pase.pt y best.pt.
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
            if name.startswith("rl_") and (self.root / name / "stopped.json").exists():
                break
            self.note(f"{name} terminó con código {code}; reintento en 60 s")
            time.sleep(60)
        self.state["done"][name] = code
        self.save()
        return code

    def rl_cmd(self, name, extra):
        a = self.a
        return ["learn.x4_ppo", "--bc", a.bc, "--out", self.root / name, "--device", a.device, "--envs", a.envs,
                "--rollout", a.rollout, "--updates", a.updates, "--lambda-dist", "0.2", "--lambda-decay", "0.9995",
                "--lambda-min", "0.05", "--critic-warmup", "20", "--human-starts", "0.4", "--pool-frac", "0.2",
                "--eval-every", "50", "--resume", *extra]

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
        # 2. principal
        main = "rl_principal"
        self.run(main, self.rl_cmd(main, []))
        chosen = [main]
        if (self.root / main / "stopped.json").exists():
            # 3. recuperación pre-registrada
            init = next((p for p in (self.root / main / "best_pase.pt", self.root / main / "best.pt") if p.exists()), None)
            extra = ["--lambda-dist", "0.4", "--lr", "1e-4"] + (["--init", init] if init else [])
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
                    self.state["done"].pop(last, None)
                    cmd = self.rl_cmd(last, [])
                    cmd[cmd.index("--updates") + 1] = a.extend_to
                    self.run(last + "_ext", cmd)
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
    a = ap.parse_args()
    sys.exit(Queue(a).main())


if __name__ == "__main__":
    main()
