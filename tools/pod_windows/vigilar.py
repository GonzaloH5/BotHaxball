"""Vigilante (corre en la PC de Gonzalo): mantiene vivo un entrenamiento en la PC prestada y trae los resultados.

Cada `--cada` minutos:
* si la PC prestada no responde (apagada, sin red), espera;
* baja `model.pt`, `ultimo.pt`, `historia.json` y el log a `runs/hx11_l0/<nombre>/` (nada depende solo de ese disco);
* si el trabajo no está corriendo y su `ultimo.pt` no dice `terminado`, lo relanza con `--reanudar` (sigue desde el
  último checkpoint; con la PC apagada se pierden como mucho unos minutos);
* si terminó, baja todo por última vez y sale.

  python -m tools.pod_windows.vigilar --nombre l0_sinhist_full -- <argumentos de learn.hx11_bc sin --out>
  python -m tools.pod_windows.vigilar --nombre rl_a3_p1 --modulo learn.hx11_rl -- <argumentos sin --out>

El módulo tiene que aceptar `--out` y `--reanudar` y guardar `ultimo.pt` con `terminado` y `paso`.
"""
from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

HOST = "hxpod@100.99.31.43"
PY = r"C:\hxpod\venv\Scripts\python.exe"
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", HOST]


def remote(cmd, timeout=120):
    try:
        r = subprocess.run(SSH + [cmd], capture_output=True, text=True, timeout=timeout, encoding="utf-8",
                           errors="replace")
        return r.returncode, r.stdout + r.stderr
    except subprocess.TimeoutExpired:
        return 255, "timeout"


def fetch(nombre, local):
    local.mkdir(parents=True, exist_ok=True)
    for f in ("model.pt", "ultimo.pt", "historia.json"):
        subprocess.run(["scp", "-q", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
                        f"{HOST}:C:/hxpod/runs/{nombre}/{f}", str(local)], capture_output=True, timeout=300)
    subprocess.run(["scp", "-q", "-o", "BatchMode=yes", f"{HOST}:C:/hxpod/runs/{nombre}.log", str(local / "pc.log")],
                   capture_output=True, timeout=120)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nombre", required=True)
    ap.add_argument("--cada", type=float, default=10.0, help="minutos entre controles")
    ap.add_argument("--ram-gb", type=float, default=8.0)
    ap.add_argument("--hilos", default="8,9,10,11")
    ap.add_argument("--cpu-pct", type=float, default=33.0)
    ap.add_argument("--modulo", default="learn.hx11_bc", help="módulo de Python que se corre (learn.hx11_rl: refuerzo)")
    ap.add_argument("bc_args", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    bc = [x for x in a.bc_args if x != "--"]
    local = Path("runs/hx11_l0") / a.nombre
    launch = (f'cmd /c "call C:\\hxpod\\entorno.bat && {PY} C:\\hxpod\\lanzar.py --nombre {a.nombre} '
              f'--hilos {a.hilos} --cpu-pct {a.cpu_pct} --ram-gb {a.ram_gb} -- {PY} -u -m {a.modulo} '
              f'{" ".join(bc)} --out C:\\hxpod\\runs\\{a.nombre} --reanudar"')
    while True:
        rc, out = remote(f"{PY} C:\\hxpod\\estado.py --lineas 1")
        stamp = time.strftime("%H:%M")
        if rc != 0:
            print(f"{stamp} PC no responde; espero", flush=True)
        else:
            line = next((l for l in out.splitlines() if l.startswith(a.nombre + ":")), "")
            ram = next((l for l in out.splitlines() if l.startswith("PC:")), "")
            fetch(a.nombre, local)
            if "corriendo" in line:
                print(f"{stamp} {line.strip()} | {ram.strip()}", flush=True)
            else:
                rc2, fin = remote(f'{PY} -c "import torch; c=torch.load(r\'C:\\hxpod\\runs\\{a.nombre}\\ultimo.pt\', '
                                  f'weights_only=False); print(\'TERMINADO\' if c.get(\'terminado\') else \'SEGUIR\', '
                                  f'c[\'paso\'])"')
                if "TERMINADO" in fin:
                    print(f"{stamp} terminado ({fin.strip()}); resultados en {local}", flush=True)
                    return
                rc3, msg = remote(launch, timeout=300)
                print(f"{stamp} relanzado ({fin.strip() or 'sin checkpoint'}): {msg.strip()[-200:]}", flush=True)
        time.sleep(a.cada * 60)


if __name__ == "__main__":
    main()
