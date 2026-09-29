"""Validate a periodic PPO checkpoint and restore latest, backing up its old bytes.

python -m tools.restore_checkpoint --run multi --checkpoint runs/multi/ckpt_002975.pt
Stop all training processes using that run before restoring.
"""
from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

import torch

from train.checkpoints import atomic_torch_save

ROOT = Path(__file__).resolve().parent.parent


def restore_checkpoint(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination:
        raise ValueError("El origen no puede ser el latest.pt que se intenta recuperar")
    # Fully deserialize before backing up or changing anything. A valid ZIP alone
    # does not guarantee that every tensor record referenced by the pickle exists.
    state = torch.load(source, map_location="cpu", weights_only=False)
    required = {"model", "model_config", "opt", "steps", "iteration", "stage", "stage_steps",
                "task_state", "league", "learner_elo", "scripted_elo"}
    missing = required - state.keys()
    if missing:
        raise ValueError(f"No es un checkpoint PPO completo: faltan {sorted(missing)}")
    backup = None
    if destination.exists():
        fd, name = tempfile.mkstemp(prefix="latest_before_restore_", suffix=".pt", dir=destination.parent)
        backup = Path(name)
        with os.fdopen(fd, "wb") as output, destination.open("rb") as previous:
            shutil.copyfileobj(previous, output)
            output.flush()
            os.fsync(output.fileno())
        print(f"Copia del archivo anterior (puede estar dañado): {backup}", flush=True)
    atomic_torch_save(state, destination)
    print(f"Restaurado {destination}: iter {state['iteration']}, pasos {state['steps']:,}, etapa {state['stage']}", flush=True)
    return backup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="multi")
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    if args.run in (".", "..") or Path(args.run).name != args.run:
        parser.error("--run debe ser el nombre de una corrida, no una ruta")
    destination = ROOT / "runs" / args.run / "latest.pt"
    if not destination.parent.is_dir():
        parser.error(f"No existe la corrida: {destination.parent}")
    restore_checkpoint(args.checkpoint, destination)


if __name__ == "__main__":
    main()
