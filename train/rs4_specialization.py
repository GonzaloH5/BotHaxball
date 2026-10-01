"""Configuración opt-in y annealing de guía RS4, independiente del PPO global."""
import math


def validate(cfg):
    setting = cfg.get("rs4_tactics")
    if not setting:
        return
    tasks = {name for stage in cfg["stages"] for name in stage["tasks"]}
    if tasks != {"rs4_4v4"}:
        raise ValueError("rs4_tactics sólo se permite en la rama exclusiva rs4_4v4")
    if setting.get("formation_version", 1) not in (1, 2):
        raise ValueError("formation_version debe ser 1 o 2")
    for key in ("coef", "coef_final"):
        value = setting[key]
        if not math.isfinite(value) or not 0 <= value <= 0.25:
            raise ValueError(f"rs4_tactics.{key} debe estar entre 0 y 0.25")
    if setting["coef_final"] > setting["coef"]:
        raise ValueError("La guía RS4 debe decaer, no aumentar")
    if (not math.isfinite(setting["decay_steps"]) or not math.isfinite(setting["start_steps"])
            or setting["decay_steps"] <= 0 or setting["start_steps"] < 0):
        raise ValueError("Horizonte RS4 inválido")


def coefficient(cfg, steps):
    setting = cfg.get("rs4_tactics")
    if not setting:
        return 0.0
    fraction = min(1.0, max(0.0, (steps - setting["start_steps"]) / setting["decay_steps"]))
    return setting["coef"] + fraction * (setting["coef_final"] - setting["coef"])
