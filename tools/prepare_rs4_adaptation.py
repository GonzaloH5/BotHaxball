"""Copiar un RS4 detenido para probar un calendario LR propio, sin tocar la fuente.

python -m tools.prepare_rs4_adaptation --source runs/rs4/latest.pt --run rs4_adapt
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import shutil
import tempfile

import torch
import yaml

from .prepare_rs4 import ROOT, TASK, file_hash
from train.runtime import learning_rate, load_config
from train.rs4_specialization import validate


def prepare(source, run="rs4_adapt", additional_steps=200_000_000,
            lr_initial=1e-4, lr_final=None):
    if not run or run in (".", "..") or any(char in run for char in '/\\:'):
        raise ValueError("run debe ser un nombre simple, sin rutas")
    if isinstance(additional_steps, bool) or not isinstance(additional_steps, int) or additional_steps <= 0:
        raise ValueError("additional-steps debe ser un entero positivo")
    source = Path(source).resolve()
    destination = ROOT / "runs" / run
    if destination.exists():
        raise FileExistsError(f"{destination} ya existe; reanudarlo, no repetir la preparación")
    if not source.is_file() or not source.with_name("config.yaml").is_file():
        raise ValueError("Se necesitan latest.pt y el config.yaml guardado de RS4")
    # Leer/validar antes de crear cualquier destino. La fuente debe estar detenida.
    cfg = load_config(source.with_name("config.yaml"))
    if len(cfg["stages"]) != 1 or set(cfg["stages"][0]["tasks"]) != {TASK}:
        raise ValueError("Se requiere un run ya especializado exclusivamente en RS4")
    if cfg["ppo"].get("lr_schedule") is not None:
        raise ValueError("La fuente ya tiene fase LR propia; reanudarla, no reiniciar su calendario")
    if cfg.get("bc_reference"):
        reference = Path(str(cfg["bc_reference"]).replace("\\", "/"))
        reference = reference if reference.is_absolute() else ROOT / reference
        if not reference.is_file():
            raise FileNotFoundError(f"Falta la referencia BC actual: {reference}")
    validate(cfg)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="_prepare_rs4_adapt_", dir=destination.parent) as directory:
        temporary = Path(directory)
        frozen = temporary / "parent.pt"
        shutil.copy2(source, frozen)
        shutil.copy2(source.with_name("config.yaml"), temporary / "parent_config.yaml")
        # Trabajar con la copia congelada; nunca resalvar/modificar el checkpoint fuente.
        checkpoint = torch.load(frozen, map_location="cpu", weights_only=False)
        required = ("model", "opt", "league", "steps", "iteration", "stage", "stage_steps", "task_state")
        if (any(key not in checkpoint for key in required)
                or checkpoint.get("model_config", {}).get("type") != "set"
                or checkpoint.get("env", {}).get("obs_layout") != "universal"
                or TASK not in checkpoint["task_state"] or checkpoint["stage"] != 0):
            raise ValueError("Se requiere un checkpoint PPO RS4 completo, universal y sin memoria")
        if isinstance(checkpoint["steps"], bool) or int(checkpoint["steps"]) != checkpoint["steps"] or checkpoint["steps"] < 0:
            raise ValueError("El checkpoint tiene pasos inválidos")
        # Detectar una fuente que cambió mientras se copiaba (no es un live-fork).
        digest = file_hash(frozen)
        if digest != file_hash(source) or source.with_name("config.yaml").read_bytes() != (temporary / "parent_config.yaml").read_bytes():
            raise ValueError("La fuente cambió durante la copia; detenerla guardando y repetir")
        frozen_cfg = load_config(temporary / "parent_config.yaml")
        if frozen_cfg != cfg:
            raise ValueError("El config fuente cambió; detener el entrenamiento antes de copiar")
        start = int(checkpoint["steps"])
        previous_lr = learning_rate(start, cfg["ppo"])
        final = previous_lr if lr_final is None else lr_final
        if (not math.isfinite(lr_initial) or not math.isfinite(final)
                or not 0 < final <= lr_initial):
            raise ValueError("La fase exige tasas finitas y 0 < lr-final <= lr-initial")
        cfg = copy.deepcopy(cfg)
        cfg["run_name"] = run
        # El calendario global sigue controlando entropía y BC, no el nuevo LR.
        cfg["ppo"].setdefault("schedule_steps", cfg["ppo"]["total_steps"])
        cfg["ppo"]["total_steps"] = start + additional_steps
        cfg["ppo"]["lr_schedule"] = dict(start_steps=start, duration_steps=additional_steps,
                                          initial=lr_initial, final=final)
        learning_rate(start, cfg["ppo"])
        shutil.copy2(frozen, temporary / "latest.pt")  # pesos/Adam/liga/contadores idénticos
        (temporary / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        manifest = dict(kind="rs4_lr_adaptation", task=TASK, source=str(source),
                        source_sha256=digest, source_steps=start,
                        source_iteration=checkpoint["iteration"], source_stage=checkpoint["stage"],
                        source_stage_steps=checkpoint["stage_steps"], additional_steps=additional_steps,
                        previous_lr=previous_lr, lr_schedule=cfg["ppo"]["lr_schedule"],
                        reference_label="RS4 anterior", bc_reference=cfg.get("bc_reference"))
        (temporary / "specialization.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        destination.mkdir()  # exclusivo: nunca sobrescribir otro experimento
        for name in ("parent.pt", "parent_config.yaml", "latest.pt", "config.yaml", "specialization.json"):
            (temporary / name).rename(destination / name)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="runs/rs4/latest.pt")
    parser.add_argument("--run", default="rs4_adapt")
    parser.add_argument("--additional-steps", type=int, default=200_000_000)
    parser.add_argument("--lr-initial", type=float, default=1e-4)
    parser.add_argument("--lr-final", type=float, default=None, help="Default: LR efectiva de la fuente")
    args = parser.parse_args()
    try:
        target = prepare(args.source, args.run, args.additional_steps, args.lr_initial, args.lr_final)
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        parser.error(str(error))
    cfg = load_config(target / "config.yaml")
    phase = cfg["ppo"]["lr_schedule"]
    print(f"Experimento preparado: {target}; fuente y referencia congelada intactas")
    print(f"LR {phase['initial']:.2e} → {phase['final']:.2e} en {phase['duration_steps']:,} pasos "
          f"desde {phase['start_steps']:,}; límite {cfg['ppo']['total_steps']:,}")
    print("No se inició entrenamiento. Entropía, BC y guía mantienen sus calendarios.")
    print(f"python -u -m train.multitask --config runs/{args.run}/config.yaml --run {args.run} --resume")


if __name__ == "__main__":
    main()
