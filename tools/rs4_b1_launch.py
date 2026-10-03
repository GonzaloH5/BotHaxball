"""Lanzar entrenamientos del bloque RS4-b1 con el código congelado (PLAN_RS4.md 2.1 y 3).

  python -m tools.rs4_freeze --label rs4_b1                    # una vez: congelar el código
  python -m tools.rs4_b1_launch --freeze <id> --queue A:1 B:1 A:2 B:2
  python -m tools.rs4_b1_launch --freeze <id> --queue A:1 --dry-run
  python -m tools.rs4_b1_launch --freeze <id> --queue A:1 A:2 A:3 --tag plazo   # experimento nuevo, runs *_plazo_s*

Antes de entrenar se niega si:
* el código difiere de la congelación (tools/rs4_freeze.verify);
* el checkpoint inicial no tiene el SHA-256 de `rs4_b1.init_sha256` en la config;
* falta una entrada (liga, banco de estados, contrato) o el run ya existe.

Cada corrida escribe runs/<run>/b1_manifest.json: congelación, config y semilla, hashes de
todas las entradas, comando y, al terminar, cómputo consumido (segundos de pared, GPU),
muestras de aprendizaje y hash del último checkpoint. Las corridas de la cola son secuenciales
(una GPU): candidatos distintos no compiten por CPU ni por el techo de tiempo.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.rs4_freeze import FREEZE_DIR, content_hash, verify  # noqa: E402
from train.runtime import load_config  # noqa: E402

CANDIDATES = {"A": "train/config_rs4_b1.yaml", "B": "train/config_rs4_b1_b.yaml", "C": "train/config_rs4_b1_c.yaml",
              "FINAL": "train/config_rs4_b1_final.yaml", "ESTILO": "train/config_rs4_b1_estilo.yaml"}
CONTRACT = "reports/rs4_b1/contract.json"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _path(raw):
    path = Path(str(raw).replace("\\", "/"))
    return path if path.is_absolute() else ROOT / path


def _relative(path):
    try:
        return str(Path(path).relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def freeze_manifest(identifier):
    path = Path(identifier)
    if not path.suffix:
        path = FREEZE_DIR / f"{identifier}.json"
    elif not path.is_absolute():
        path = ROOT / path
    if not path.is_file():
        raise SystemExit(f"No existe la congelación {path}; crearla con python -m tools.rs4_freeze --label rs4_b1")
    return path


def plan(candidate, seed, freeze_path, tag=""):
    """Validar una corrida sin lanzar nada; devuelve (comando, manifiesto).

    `tag` distingue experimentos posteriores con la misma config (p. ej. "plazo": rs4_b1_a_plazo_s1)."""
    if candidate not in CANDIDATES:
        raise SystemExit(f"Candidato desconocido {candidate}; usar {sorted(CANDIDATES)}")
    config_path = ROOT / CANDIDATES[candidate]
    if not config_path.is_file():
        raise SystemExit(f"Falta {config_path} (C requiere aprobar su compuerta en CPU antes de existir)")
    cfg = load_config(config_path)
    block = cfg.get("rs4_b1") or {}
    run = f"{cfg['run_name']}{'_' + tag if tag else ''}_s{seed}"
    if (ROOT / "runs" / run / "latest.pt").exists():
        raise SystemExit(f"El run {run} ya existe; un experimento no se relanza encima de otro")
    init = _path(block.get("init_from", ""))
    if not init.is_file():
        raise SystemExit(f"Falta el checkpoint inicial {init}")
    init_hash = sha256(init)
    if init_hash != block.get("init_sha256"):
        raise SystemExit(f"El checkpoint inicial {init} tiene SHA-256 {init_hash}, no el esperado {block.get('init_sha256')}")
    inputs = {"init": {"path": _relative(init), "sha256": init_hash}}
    for raw in cfg.get("rs4_v5", {}).get("league_seed", []):
        path = _path(raw)
        if not path.is_file():
            raise SystemExit(f"Falta el miembro de liga {path}")
        inputs[f"league:{_relative(path)}"] = {"path": _relative(path), "sha256": sha256(path)}
    bank = _path(block["recorded"]["path"])
    if not bank.is_file():
        raise SystemExit(f"Falta el banco de estados {bank} (python -m tools.rs4_states)")
    inputs["states"] = {"path": _relative(bank), "sha256": sha256(bank)}
    inputs["contract"] = {"path": CONTRACT, "sha256_lf": content_hash(ROOT / CONTRACT)}  # igual en Windows y Linux
    differences = verify(freeze_path)
    if differences:
        raise SystemExit("El código difiere de la congelación; no se entrena:\n  " + "\n  ".join(differences[:40]))
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    command = [sys.executable, "-u", "-m", "train.multitask", "--config", _relative(config_path), "--run", run,
               "--init-from", _relative(init), "--override", f"seed={seed}"]
    manifest = dict(block="RS4-b1", candidate=candidate, seed=seed, tag=tag, run=run, config=_relative(config_path),
                    config_sha256=sha256(config_path), freeze_id=freeze["id"], freeze_sha256=freeze["sha256"],
                    inputs=inputs, budget_samples=int(cfg["rs4_program"]["total_budget_steps"]),
                    wall_ceiling_seconds=float(cfg.get("runtime", {}).get("max_wall_seconds", 0) or 0),
                    command=command, host=platform.node(), status="planned")
    return command, manifest


def _gpu():
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True,
                              text=True, check=True).stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def _write(manifest):
    path = ROOT / "runs" / manifest["run"] / "b1_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    return path


def launch(command, manifest):
    manifest.update(status="running", gpu=_gpu(), started=time.strftime("%Y-%m-%d %H:%M:%S"))
    path = _write(manifest)
    log = path.parent / "train.log"
    print(f"RS4-b1 {manifest['run']}: {' '.join(command)}\n  manifiesto {path}\n  log {log}", flush=True)
    started = time.time()
    with log.open("a", encoding="utf-8") as handle:
        code = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT).returncode
    manifest.update(returncode=code, finished=time.strftime("%Y-%m-%d %H:%M:%S"),
                    wall_seconds=round(time.time() - started, 1), status="finished" if code == 0 else "failed")
    latest = ROOT / "runs" / manifest["run"] / "latest.pt"
    if latest.is_file():
        import torch
        ck = torch.load(latest, map_location="cpu", weights_only=False)
        manifest.update(learning_samples=int(ck.get("steps", 0)), latest_sha256=sha256(latest))
    _write(manifest)
    print(f"RS4-b1 {manifest['run']}: {manifest['status']} en {manifest['wall_seconds'] / 3600:.2f} h", flush=True)
    return code


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--freeze", required=True, help="id o ruta de reports/rs4_b1/freeze/<id>.json")
    ap.add_argument("--queue", nargs="+", required=True, help="candidato:semilla, p. ej. A:1 B:1 A:2 B:2")
    ap.add_argument("--tag", default="", help="sufijo del run para un experimento nuevo con la misma config")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.tag and not a.tag.replace("_", "").isalnum():
        raise SystemExit("--tag sólo admite letras, números y _")
    freeze_path = freeze_manifest(a.freeze)
    jobs = []
    for item in a.queue:
        candidate, _, seed = item.partition(":")
        if not seed.isdigit():
            raise SystemExit(f"Formato candidato:semilla inválido: {item}")
        jobs.append((candidate.upper(), int(seed)))
    if len(set(jobs)) != len(jobs):
        raise SystemExit("La cola repite una corrida")
    plans = [plan(candidate, seed, freeze_path, a.tag) for candidate, seed in jobs]  # validar todo antes de empezar
    for command, manifest in plans:
        if a.dry_run:
            print(json.dumps(manifest, indent=1, ensure_ascii=False))
            continue
        # La congelación se vuelve a verificar justo antes de cada corrida de la cola.
        command, manifest = plan(manifest["candidate"], manifest["seed"], freeze_path, a.tag)
        if launch(command, manifest) != 0:
            raise SystemExit(f"La corrida {manifest['run']} falló; la cola se detiene")


if __name__ == "__main__":
    main()
