"""Calibra p25 de cooperación usando sólo replays reservados para validación.

La selección por hash es exactamente la de ``train.bc``. Los partidos usados
para producir este JSON nunca se incorporan al rollout PPO.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from env.tasks import load_catalog
from tools.analyze_match import load

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "bc"
RAW = ROOT / "replays_real" / "stadiums"
TARGETS = {"big_3v3", "futsal_3v3", "futsal_af_3v3", "aha_3v3", "rs4_4v4"}


def is_validation(name: str, fraction=.12) -> bool:
    return int(hashlib.md5(name.encode()).hexdigest(), 16) % 1000 < fraction * 1000


def task_for(stadium: str, team_size: int):
    matches = [task for task in load_catalog().values()
               if task.stadium == stadium and task.n_per_team == team_size and task.name in TARGETS]
    return matches[0] if len(matches) == 1 else None


def replay_rates(path: Path, task):
    ticks, _, _ = load(path)
    state = {team: {"last": -1, "last_pos": None, "sender": -1, "receiver": -1,
                    "pending": None, "passes": 0, "progressive": 0, "chains": 0,
                    "turnovers": 0} for team in (1, 2)}
    playing = 0
    for tick in ticks:
        if tick.get("state") != 1 or not tick.get("discs") or tick["discs"][0].get("x") is None:
            continue
        players = [p for p in tick["players"] if p.get("team") in (1, 2)
                   and tick["discs"][p["disc"]].get("x") is not None]
        if sum(p["team"] == 1 for p in players) != task.n_per_team or sum(p["team"] == 2 for p in players) != task.n_per_team:
            continue
        playing += 1
        ball = np.array([tick["discs"][0]["x"], tick["discs"][0]["y"]], dtype=float)
        positions = {p["id"]: np.array([tick["discs"][p["disc"]]["x"],
                                         tick["discs"][p["disc"]]["y"]], dtype=float) for p in players}
        teams = {p["id"]: p["team"] for p in players}
        for row in state.values():
            if row["last"] >= 0 and row["last"] not in positions:
                row["last"], row["last_pos"], row["pending"] = -1, None, None
                row["sender"] = row["receiver"] = -1
        # Confirmar sólo si el receptor siguió siendo el último contacto 12 ticks.
        for team, row in state.items():
            pending = row["pending"]
            if pending and tick["frame"] - pending["frame"] >= 12 and row["last"] == pending["receiver"]:
                row["passes"] += 1
                row["progressive"] += pending["useful"] > .10
                chain = (row["receiver"] == pending["sender"] and row["sender"] >= 0
                         and row["sender"] != pending["receiver"] and pending["useful"] > .10)
                row["chains"] += chain
                row["sender"], row["receiver"] = pending["sender"], pending["receiver"]
                row["pending"] = None
        distances = [(float(np.linalg.norm(pos - ball)), pid) for pid, pos in positions.items()]
        contacts = [pid for distance, pid in distances if distance <= 30.0]
        if len(contacts) != 1:
            continue
        pid = contacts[0]
        team = teams[pid]
        row = state[team]
        other = state[3 - team]
        if other["last"] >= 0:
            other["turnovers"] += 1
            other["pending"] = None
            other["sender"] = other["receiver"] = -1
        previous, previous_pos = row["last"], row["last_pos"]
        if previous >= 0 and previous != pid and previous_pos is not None:
            travel = float(np.linalg.norm(ball - previous_pos))
            if travel >= .04 * task_width(task):
                sign = 1.0 if team == 1 else -1.0
                progress = float(np.clip(sign * (ball[0] - previous_pos[0]) / (.25 * task_width(task)), 0, 1))
                opponents = [positions[p] for p, t in teams.items() if t != team]
                receiver_space = min(np.linalg.norm(x - positions[pid]) for x in opponents)
                sender_space = min(np.linalg.norm(x - positions[previous]) for x in opponents)
                space = float(np.clip((receiver_space - sender_space) / (.15 * task_width(task)), 0, 1))
                row["pending"] = {"sender": previous, "receiver": pid, "frame": tick["frame"],
                                  "useful": .65 * progress + .35 * space}
        row["last"], row["last_pos"] = pid, ball
        other["last"] = -1
    minutes = playing / 3600
    if minutes <= 0:
        return []
    return [{"progressive_passes_per_minute": row["progressive"] / minutes,
             "pass_chains_per_minute": row["chains"] / minutes,
             "turnovers_per_minute": row["turnovers"] / minutes} for row in state.values()]


_WIDTH_CACHE = {}


def task_width(task):
    if task.name not in _WIDTH_CACHE:
        from env.tasks import make_env
        _WIDTH_CACHE[task.name] = make_env(task, 1, task.n_entities, random_reset_prob=0).field_w
    return _WIDTH_CACHE[task.name]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ROOT / "reports" / "human_coordination_validation.json"))
    args = ap.parse_args()
    values = defaultdict(list)
    files = defaultdict(set)
    for shard in sorted(DATA.glob("*/*.npz")):
        if not is_validation(shard.name):
            continue
        with np.load(shard, allow_pickle=False) as data:
            sizes = np.unique(data["T"])
            tasks = [task_for(str(data["stadium"]), int(size)) for size in sizes]
        replay = RAW / shard.parent.name / (shard.stem + ".hbr2")
        if not replay.exists():
            continue
        for task in (task for task in tasks if task is not None):
            values[task.name].extend(replay_rates(replay, task))
            files[task.name].add(str(replay.relative_to(ROOT)).replace("\\", "/"))
            print(f"{task.name}: {replay.name}", flush=True)
    tasks = {}
    for name, rows in values.items():
        tasks[name] = {"validation_replays": len(files[name]), "team_matches": len(rows),
                       "progressive_passes_per_minute_p25": float(np.percentile(
                           [r["progressive_passes_per_minute"] for r in rows], 25)),
                       "pass_chains_per_minute_p25": float(np.percentile(
                           [r["pass_chains_per_minute"] for r in rows], 25)),
                       "turnovers_per_minute_p75": float(np.percentile(
                           [r["turnovers_per_minute"] for r in rows], 75))}
    report = {"validation_fraction": .12, "split": "whole-replay md5, identical to train.bc",
              "training_use": False, "tasks": tasks}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(out.resolve())


if __name__ == "__main__":
    main()

