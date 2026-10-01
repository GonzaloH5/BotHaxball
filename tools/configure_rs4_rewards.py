"""Activar guía táctica únicamente en un run RS4 preparado y detenido.

python -m tools.configure_rs4_rewards --run rs4
No modifica checkpoints, PPO, liga ni el generalista; guarda backup de config.
"""
import argparse
import copy
import os
import shutil
import tempfile
from pathlib import Path

import torch
import yaml

from train.runtime import load_config
from train.rs4_specialization import validate
from .prepare_rs4 import ROOT, TASK

PROFILE = Path(__file__).resolve().parents[1] / "train/config_rs4_tactical.yaml"


def configure(run="rs4"):
    if not run or run in (".", "..") or any(char in run for char in '/\\:'):
        raise ValueError("run debe ser un nombre simple")
    directory = ROOT / "runs" / run
    if not (directory / "specialization.json").is_file():
        raise ValueError("Primero preparar la rama con tools.prepare_rs4; no modificar el generalista")
    path = directory / "config.yaml"
    cfg = load_config(path)
    if cfg.get("rs4_tactics"):
        raise ValueError("La guía ya está configurada; reanudar sin volver a activar ni reiniciar el decay")
    if {name for stage in cfg["stages"] for name in stage["tasks"]} != {TASK}:
        raise ValueError("El run debe entrenar sólo rs4_4v4")
    checkpoint = torch.load(directory / "latest.pt", map_location="cpu", weights_only=False)
    profile = load_config(PROFILE)
    cfg["rs4_tactics"] = copy.deepcopy(profile["rs4_tactics"])
    cfg["rs4_tactics"]["start_steps"] = int(checkpoint["steps"])
    cfg.setdefault("task_metric_versions", {})["rs4_4v4"] = profile["task_metric_versions"]["rs4_4v4"]
    for key in ("w_near_ball", "kick_to_goal", "w_spread", "team_spread_floor", "rs4_defensive_out_scale",
                "rs4_restart_approach", "rs4_restart_stall"):
        cfg["reward"][key] = profile["reward"][key]
    validate(cfg)
    # Backup único y publicación atómica; jamás cambiar latest.pt/parent.pt.
    fd, backup = tempfile.mkstemp(prefix="config_before_tactics_", suffix=".yaml", dir=directory)
    os.close(fd)
    shutil.copy2(path, backup)
    fd, temporary = tempfile.mkstemp(prefix=".config_tactics_", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            yaml.safe_dump(cfg, stream, sort_keys=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return Path(backup)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="rs4")
    args = parser.parse_args()
    try:
        backup = configure(args.run)
    except ValueError as error:
        parser.error(str(error))
    print(f"Guía RS4 activada, config anterior: {backup}")
    print("Reanudar con runs/<run>/config.yaml. No se modificó ningún checkpoint ni se inició entrenamiento.")


if __name__ == "__main__":
    main()
