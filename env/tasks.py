"""Catálogo de tareas (train/tasks.yaml) y fábrica de entornos con la obs universal."""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import yaml

from .haxball_env import HaxballEnv
from .rewards import RewardConfig

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CATALOG = ROOT / "train" / "tasks.yaml"


@dataclass(frozen=True)
class Task:
    name: str
    stadium: str
    n_per_team: int
    powershot: bool
    out_of_bounds: bool
    rules: str | None = None   # script completo de la sala ("pegeche", ver env/pegeche.py)
    corner_reset_prob: float = 0.0
    corner_execute_reward: float = 0.0

    @property
    def n_entities(self) -> int:
        return 2 * self.n_per_team - 1


def load_catalog(path: str | Path = DEFAULT_CATALOG) -> dict[str, Task]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    rules = data.get("rules", {})
    out = {}
    for name, spec in data["tasks"].items():
        r = dict(rules.get(spec.get("rules", "plain"), {}))
        r.update({k: v for k, v in spec.items() if k in ("powershot", "out_of_bounds", "script")})
        corner_reset_prob = float(spec.get("corner_reset_prob", 0.0))
        corner_execute_reward = float(spec.get("corner_execute_reward", 0.0))

        if not 0.0 <= corner_reset_prob <= 1.0:
            raise ValueError(
                f"{name}.corner_reset_prob debe estar entre 0 y 1"
            )

        if corner_execute_reward < 0.0:
            raise ValueError(
                f"{name}.corner_execute_reward no puede ser negativo"
            )

        out[name] = Task(
            name,
            spec["stadium"],
            int(spec["n_per_team"]),
            bool(r.get("powershot", False)),
            bool(r.get("out_of_bounds", False)),
            r.get("script"),
            corner_reset_prob,
            corner_execute_reward,
        )
    return out


def make_env(
    task: Task,
    n_envs: int,
    max_entities: int,
    reward: RewardConfig | None = None,
    seed: int | None = None,
    frame_skip: int = 3,
    max_ticks: int = 7200,
    random_reset_prob: float = 0.3,
    kickoff_timeout: int = 180,
    action_delay_max: int = 0,
    optimize_rollout: bool = True,
    corner_curriculum: bool = True,
) -> HaxballEnv:
    """Crear un entorno; las evaluaciones deben usar corner_curriculum=False."""

    rcfg = replace(
        reward or RewardConfig(),
        corner_execute=task.corner_execute_reward,
    )

    return HaxballEnv(
        n_envs,
        task.n_per_team,
        task.stadium,
        frame_skip,
        max_ticks,
        random_reset_prob,
        rcfg,
        seed=seed,
        action_delay_max=action_delay_max,
        kickoff_timeout=kickoff_timeout,
        powershot=task.powershot,
        out_of_bounds=task.out_of_bounds,
        obs_layout="universal",
        max_entities=max_entities,
        rules=task.rules,
        optimize_rollout=optimize_rollout,
        corner_reset_prob=task.corner_reset_prob if corner_curriculum else 0.0,
    )
