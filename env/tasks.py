"""Catálogo de tareas (train/tasks.yaml) y fábrica de entornos con la obs universal."""
from __future__ import annotations

from dataclasses import dataclass
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
        out[name] = Task(name, spec["stadium"], int(spec["n_per_team"]),
                         bool(r.get("powershot", False)), bool(r.get("out_of_bounds", False)),
                         r.get("script"))
    return out


def make_env(task: Task, n_envs: int, max_entities: int, reward: RewardConfig | None = None,
             seed: int | None = None, frame_skip: int = 3, max_ticks: int = 7200,
             random_reset_prob: float = 0.3, kickoff_timeout: int = 180, action_delay_max: int = 0) -> HaxballEnv:
    return HaxballEnv(n_envs, task.n_per_team, task.stadium, frame_skip, max_ticks, random_reset_prob,
                      reward, seed=seed, action_delay_max=action_delay_max, kickoff_timeout=kickoff_timeout,
                      powershot=task.powershot, out_of_bounds=task.out_of_bounds,
                      obs_layout="universal", max_entities=max_entities, rules=task.rules)
