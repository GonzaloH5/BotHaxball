"""Actualizar un run RS4 detenido, sin modificar pesos/Adam/PPO ni el generalista."""
import argparse
import copy
import os
from pathlib import Path
import shutil
import tempfile

import torch
import yaml

from train.runtime import load_config
from train.rs4_specialization import validate


def upgrade(path, output=None, renew_guide=False):
    path = Path(path).resolve()
    cfg = load_config(path)
    if {t for s in cfg["stages"] for t in s["tasks"]} != {"rs4_4v4"} or not cfg.get("rs4_tactics"):
        raise ValueError("Se requiere una rama exclusiva RS4 con guía ya configurada")
    if renew_guide and cfg["rs4_tactics"].get("formation_version") == 2:
        raise ValueError("La guía v2 ya está aplicada; no reiniciar su decay")
    cfg = copy.deepcopy(cfg)
    cfg["rs4_tactics"]["formation_version"] = 2
    cfg.setdefault("task_metric_versions", {})["rs4_4v4"] = 2
    cfg["reward"].update(rs4_defensive_out_scale=.2, rs4_restart_approach=.05, rs4_restart_stall=.25)
    cfg["log"].update(checkpoint_interval_seconds=300, checkpoint_history_interval_seconds=1800, checkpoint_keep=6)
    if renew_guide:
        checkpoint = torch.load(path.with_name("latest.pt"), map_location="cpu", weights_only=False)
        cfg["rs4_tactics"].update(coef=.08, coef_final=0., decay_steps=200_000_000,
                                 start_steps=int(checkpoint["steps"]))
        remaining = cfg["ppo"].get("total_steps", 0) - int(checkpoint["steps"])
        if remaining < 200_000_000:
            print(f"Aviso: sólo quedan {max(0, remaining):,} pasos del presupuesto; "
                  "no se amplía automáticamente. La fase completa de guía requiere 200M.")
    validate(cfg)
    target = Path(output).resolve() if output else path
    if output and (target == path or target.exists()):
        raise ValueError("--output debe ser un archivo nuevo; no sobrescribir el original")
    target.parent.mkdir(parents=True, exist_ok=True)
    if output is None:
        fd, backup = tempfile.mkstemp(prefix="config_before_rs4_v2_", suffix=".yaml", dir=path.parent)
        os.close(fd)
        shutil.copy2(path, backup)
        print(f"Config anterior conservada: {backup}")
    fd, temporary = tempfile.mkstemp(prefix=".rs4_v2_", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            yaml.safe_dump(cfg, stream, sort_keys=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="runs/rs4/config.yaml")
    parser.add_argument("--output", default=None)
    parser.add_argument("--renew-guide", action="store_true",
                        help="nueva fase v2: coef .08 durante 200M desde latest; no reinicia el annealing PPO")
    args = parser.parse_args()
    try:
        target = upgrade(args.config, args.output, args.renew_guide)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    print(f"Config actualizada: {target}; checkpoints y referencia BC intactos")
    print("Reanudar sin --init-from. La referencia RS4 nueva debe evaluarse antes de activarla.")


if __name__ == "__main__":
    main()
