"""Crear un run RS4 independiente sin entrenar ni modificar el generalista.

python -m tools.prepare_rs4 --source runs/multi/latest.pt --run rs4 --additional-steps 500000000
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import torch
import yaml

from train.checkpoints import atomic_torch_save
from train.runtime import load_config

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "train/config_rs4_specialist.yaml"
TASK = "rs4_4v4"


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(source, run="rs4", additional_steps=500_000_000):
    # Un nombre de run, nunca una ruta absoluta/parent que pueda apuntar a la fuente.
    if not run or run in (".", "..") or any(char in run for char in '/\\:'):
        raise ValueError("run debe ser un nombre simple, sin rutas")
    if additional_steps <= 0:
        raise ValueError("additional-steps debe ser positivo")
    source = Path(source).resolve()
    destination = ROOT / "runs" / run
    if destination.exists():
        raise FileExistsError(f"{destination} ya existe; no se sobrescribe ni se reinicia")
    if not source.is_file() or not source.with_name("config.yaml").is_file():
        raise ValueError("Se necesitan el checkpoint fuente y su config.yaml guardado")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="_prepare_rs4_", dir=destination.parent) as directory:
        temporary = Path(directory)
        frozen = temporary / "parent.pt"
        shutil.copy2(source, frozen)
        shutil.copy2(source.with_name("config.yaml"), temporary / "parent_config.yaml")
        checkpoint = torch.load(frozen, map_location="cpu", weights_only=False)
        cfg = load_config(temporary / "parent_config.yaml")
        if checkpoint.get("model_config", {}).get("type") != "set":
            raise ValueError("Esta infraestructura requiere un SetActorCritic universal sin memoria")
        if checkpoint.get("env", {}).get("obs_layout") != "universal" or TASK not in checkpoint.get("task_state", {}):
            raise ValueError("El checkpoint debe tener observaciones universales y estado entrenado de RS4")
        if "opt" not in checkpoint or "league" not in checkpoint:
            raise ValueError("Se requiere un checkpoint PPO completo, con optimizador y liga")
        profile = load_config(PROFILE)
        cfg["stages"] = copy.deepcopy(profile["stages"])
        cfg["run_name"] = run
        for key in ("checkpoint_interval_seconds", "checkpoint_history_interval_seconds", "checkpoint_keep"):
            cfg["log"][key] = profile["log"][key]
        cfg.setdefault("runtime", {})["preserve_bc_reference"] = True
        # Cambiar el límite de ejecución nunca debe reiniciar el annealing anterior.
        cfg["ppo"].setdefault("schedule_steps", cfg["ppo"]["total_steps"])
        cfg["ppo"]["total_steps"] = int(checkpoint["steps"]) + additional_steps
        origin = dict(source=str(source), source_sha256=file_hash(frozen),
                      source_steps=checkpoint["steps"], source_iteration=checkpoint["iteration"],
                      source_stage=checkpoint["stage"], source_stage_steps=checkpoint["stage_steps"],
                      task=TASK, additional_steps=additional_steps)
        checkpoint["stage"], checkpoint["stage_steps"] = 0, 0
        # Pesos, Adam, pasos globales, liga y estado de tareas permanecen intactos.
        # build_envs creará únicamente RS4; no reinicia su dificultad ni sus ventanas.
        checkpoint["env"] = {**checkpoint["env"], "tasks": [TASK]}
        atomic_torch_save(checkpoint, temporary / "latest.pt")
        (temporary / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        (temporary / "specialization.json").write_text(json.dumps(origin, indent=2), encoding="utf-8")
        # mkdir exclusivo: incluso si otro proceso preparó el run, nunca sobrescribirlo.
        destination.mkdir()
        for name in ("parent.pt", "parent_config.yaml", "config.yaml", "specialization.json", "latest.pt"):
            (temporary / name).rename(destination / name)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="runs/multi/latest.pt")
    parser.add_argument("--run", default="rs4")
    parser.add_argument("--additional-steps", type=int, default=500_000_000)
    parser.add_argument("--tactical-rewards", action="store_true", help="Activar la guía RS4 acotada, con decay propio")
    args = parser.parse_args()
    try:
        path = prepare(args.source, args.run, args.additional_steps)
        if args.tactical_rewards:
            from .configure_rs4_rewards import configure
            configure(args.run)
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))
    print(f"Run RS4 preparado: {path}; el checkpoint fuente no se modificó.")
    print("No se inició entrenamiento. Reanudar desde la raíz del repositorio:")
    print(f"python -m train.multitask --config runs/{args.run}/config.yaml --run {args.run} --resume")


if __name__ == "__main__":
    main()
