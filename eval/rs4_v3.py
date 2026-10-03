"""Independent RS4 v3 football, functional and opportunity-cohort evaluation.

No training guide is paid in evaluations. Functional setup metadata is used by
the scorer only; actors receive the same ordinary observation as in a room.
"""
from __future__ import annotations

from collections import Counter
import hashlib
from pathlib import Path
import numpy as np
import torch

from env.haxball_env import HaxballEnv
from env.rewards import RewardConfig
from env.rs4_v3 import RS4ScenarioEnv, RestartCohorts, SCENARIOS
from .agents import ModelAgent, ScriptedAgent, make_agent, reset_agents
from .rs4_objective import CONTRACT


def _file_fingerprint(path):
    if not path or not Path(path).is_file():
        return None
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def evaluation_source_fingerprint():
    """Read-only identity for runner caches and independent evaluator reports."""
    root = Path(__file__).resolve().parents[1]
    files = ("env/haxball_env.py", "env/rs4_v3.py", "env/rs4_tactics.py", "env/rewards.py",
             "bots/scripted.py", "eval/rs4_v3.py", "eval/agents.py", "sim/physics.py",
             "sim/stadium.py", "stadiums/rs_one.hbs", "env/public_signals.py", "env/public_joints.py", "train/model.py",
             "env/observation.py", "env/rollout_callbacks.py", "eval/rs4_objective.py",
             "train/rs4_program.py", "tools/evaluate_rs4_v3.py")
    hashes = {path: _file_fingerprint(root / path) for path in files}
    combined = hashlib.sha256("\n".join(f"{path}:{hashes[path]}" for path in files).encode()).hexdigest()
    return {"sha256": combined, "files": hashes}


def environment(games, seed, *, max_ticks=7200, max_entities=7):
    return HaxballEnv(games, 4, "rs_one", frame_skip=3, max_ticks=max_ticks,
                     random_reset_prob=0, seed=seed, obs_layout="universal", max_entities=max_entities,
                     out_of_bounds=True, corner_reset_prob=0, kickoff_timeout=180,
                     reward=RewardConfig(shaping_coef=0, team_spread_floor=0,
                                         kickoff_approach=0, w_defense_support=0,
                                         rs4_tactical_coef=0, corner_execute=0))


class MixedTeamAgent:
    """A fixed controller per player for the whole match; human proxies only."""
    def __init__(self, learner, teammate, learner_slots=(0, 2)):
        self.learner, self.teammate = learner, teammate
        self.learner_slots = tuple(learner_slots)

    def reset(self, env, done=None):
        reset_agents((self.learner, self.teammate), env, done)

    def record_executed(self, env, actions):
        record_executed((self.learner, self.teammate), env, actions)

    def __call__(self, env, obs, players):
        out = np.empty((env.N, len(players)), dtype=np.int64)
        own = np.asarray([i in self.learner_slots for i in range(len(players))])
        for mask, controller in ((own, self.learner), (~own, self.teammate)):
            if mask.any():
                out[:, mask] = controller(env, obs, players[mask])
        return out


def record_executed(agents, env, actions):
    for agent in agents:
        if hasattr(agent, "record_executed"):
            agent.record_executed(env, actions)


def full_game(agent, opponent, games=16, minutes=2, seed=51, color=0):
    ticks = max(3, int(minutes * 3600))
    env = environment(games, seed, max_ticks=ticks)
    obs = env.reset()
    env._reset_envs(np.arange(games), kickoff_team=0)
    # Distinct seeded initial states also for deterministic deployment policies.
    env.sim.player_pos[:] += env.rng.uniform(-8, 8, env.sim.player_pos.shape)
    env._phi = env._potentials()
    obs = env.observe()
    red, blue = np.arange(4), np.arange(4, 8)
    controllers = (agent, opponent) if color == 0 else (opponent, agent)
    reset_agents(controllers, env)
    cohorts = RestartCohorts(env)
    behavior = Counter()
    recovering = np.zeros(games, dtype=bool)
    recovery_age = np.zeros(games, dtype=np.int64)
    for _ in range((ticks + 2) // 3):
        cohorts.begin()
        actions = np.empty((games, 8), dtype=np.int64)
        actions[:, red] = controllers[0](env, obs, red)
        actions[:, blue] = controllers[1](env, obs, blue)
        obs, _, done, info = env.step(actions)
        record_executed(controllers, env, info["executed_actions"])
        cohorts.update(info, done)
        for key in ("passes", "progressive_passes", "turnovers"):
            behavior[key] += int(info["events"][key][:, color].sum())
        sign = 1 if color == 0 else -1
        ball = info["final_obs"][:, 0, 4:6] * [env.field_w, env.field_h]
        loss = info["events"]["turnovers"][:, color] > 0
        behavior["dangerous_losses"] += int((loss & (sign * ball[:, 0] < -.35 * env.field_w)).sum())
        recovering[loss] = True
        recovery_age[loss] = 0
        recovery_age[recovering] += env.frame_skip
        own = env.sim.player_pos[:, env.sim.player_team == color]
        behind = sign * own[..., 0] < sign * ball[:, None, 0] - .10 * env.field_w
        recovered = recovering & (behind.sum(axis=1) >= 2) & ~done
        behavior["coverage_recoveries"] += int(recovered.sum())
        behavior["coverage_recovery_ticks"] += int(recovery_age[recovered].sum())
        censored = recovering & ~recovered & (done | (recovery_age >= 600))
        behavior["coverage_recovery_censored"] += int(censored.sum())
        recovering[recovered | censored] = False
        if done.any():
            reset_agents(controllers, env, done)
    ours, theirs = env.score[:, color], env.score[:, 1 - color]
    wins, draws, losses = int((ours > theirs).sum()), int((ours == theirs).sum()), int((ours < theirs).sum())
    return {"games": games, "color": color, "seed": seed, "wins": wins, "draws": draws,
            "losses": losses, "points": (wins + .5 * draws) / games,
            "goals_for": int(ours.sum()), "goals_against": int(theirs.sum()),
            "behavior": dict(behavior), "restarts": cohorts.report()}


def functional_trial(agent, scenario, games=16, seed=51, color=0, teammate=None, learner_slots=(0, 2)):
    base = environment(games, seed)
    env = RS4ScenarioEnv(base, {"drill_fraction": .4, "restart_bonus": 0, "guide_coef": 0,
                             "restart_potential_coef": 0, "pass_participant_credit": 0,
                             "scenario_difficulty": 1.,
                             "scenario_weights": {scenario: 1}})
    env.reset()
    env.assign(np.arange(games), scenario=scenario, team=color)
    obs = env.observe()
    ours = MixedTeamAgent(agent, teammate, learner_slots) if teammate is not None else agent
    other = ScriptedAgent(seed=seed, policy="r3", style=-1)
    controllers = (ours, other) if color == 0 else (other, ours)
    reset_agents(controllers, env)
    alive = np.ones(games, dtype=bool)
    results = []
    opportunities = Counter()
    for _ in range(800):
        actions = np.empty((games, 8), dtype=np.int64)
        actions[:, :4] = controllers[0](env, obs, np.arange(4))
        actions[:, 4:] = controllers[1](env, obs, np.arange(4, 8))
        obs, _, done, info = env.step(actions)
        record_executed(controllers, env, info["executed_actions"])
        for outcome in info["restart_outcomes"]:
            if alive[outcome["row"]] and outcome["team"] == color:
                opportunities["resolved"] += 1
                opportunities["successes"] += int(outcome["success"])
                opportunities["attempts"] += int(outcome["attempted"])
                opportunities["timeouts"] += int(outcome["timeout"])
                opportunities["resolution_ticks"] += outcome["ticks"]
        for result in info["scenario_result"]:
            row = result["row"]
            if alive[row]:
                results.append(result)
                alive[row] = False
        if done.any():
            reset_agents(controllers, env, done)
        if not alive.any():
            break
    if alive.any():
        raise RuntimeError("Functional trials failed to resolve within the evaluation limit")
    return {"scenario": scenario, "color": color, "seed": seed, "games": games,
            "success": sum(row["success"] for row in results) / games,
            "conceded": sum(row["conceded"] for row in results) / games,
            "mean_seconds": np.mean([row["ticks"] for row in results]) / 60,
            "restarts": dict(opportunities), "mixed_teammates": teammate is not None,
            "outcomes": results,
            "metrics": {key: sum(r["metrics"][key] for r in results) / games
                        for key in env.metrics},
            "danger_fraction": sum(r["metrics"]["danger_ticks"] for r in results) / max(1, sum(r["ticks"] for r in results)),
            "goal_rate": sum(r["metrics"]["goals"] > 0 for r in results) / games}


def _cell_key(row):
    return "|".join(str(row[key]) for key in ("learner_count", "opponent", "teammate", "color"))


def match_summary(rows):
    cells = {}
    for row in rows:
        key = _cell_key(row)
        cell = cells.setdefault(key, {k: row[k] for k in ("learner_count", "opponent", "teammate", "color")})
        for k in ("games", "wins", "draws", "losses", "goals_for", "goals_against"):
            cell[k] = cell.get(k, 0) + row[k]
    for cell in cells.values():
        n = cell["games"]
        p = cell["points"] = (cell["wins"] + .5 * cell["draws"]) / n
        # Descriptive standard error of a 0/.5/1 result, not binomial winrate.
        se = np.sqrt(max(0., (cell["wins"] + .25 * cell["draws"]) / n - p*p) / n)
        cell["points_interval_95"] = [max(0., p - 1.96 * se), min(1., p + 1.96 * se)]
    means = {str(count): float(np.mean([r["points"] for r in cells.values() if r["learner_count"] == count]))
             for count in (1, 2, 3, 4)}
    return dict(cells=cells, by_learner_count=means,
                balanced_points=float(np.mean(list(means.values()))),
                single_instance_points=means["1"],
                mean_points=float(np.mean([r["points"] for r in rows])),
                all_learner_mean_points=means["4"],
                mixed_mean_points=float(np.mean([means[str(i)] for i in (1, 2, 3)])), rows=rows)


def functional_summary(rows):
    cells = {}
    for row in rows:
        count = len(row.get("learner_slots", (0, 1, 2, 3)))
        key = f"{row['scenario']}|{count}|{row['color']}"
        cell = cells.setdefault(key, dict(scenario=row["scenario"], learner_count=count,
                                         color=row["color"], games=0, successes=0))
        cell["games"] += row["games"]
        cell["successes"] += int(round(row["success"] * row["games"]))
    for cell in cells.values():
        n, k = cell["games"], cell["successes"]
        p = cell["success"] = k / n
        z = 1.96
        center = (p + z*z/(2*n)) / (1+z*z/n)
        width = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n)) / (1+z*z/n)
        cell["success_interval_95"] = [max(0., center-width), min(1., center+width)]
    return cells


def evaluate(checkpoint, *, references=(), teammate_reference=None, seeds=(51, 73, 91),
             games=16, functional_games=16, minutes=2, action_mode="greedy", holdout=False, device="cpu"):
    if action_mode not in ("greedy", "sampled"):
        raise ValueError("action_mode must be greedy or sampled")
    greedy = action_mode == "greedy"
    agent = make_agent(str(checkpoint), greedy=greedy, device=device)
    opponents = [f"scripted:r3:{style}" for style in range(3)] + [str(p) for p in references]
    full_rows, functional_rows = [], []
    full_by_seed, functional_by_seed = {}, {}
    for seed_index, seed in enumerate(seeds):
        seed_full, seed_functional = [], []
        for opponent in opponents:
            for color in (0, 1):
                torch.manual_seed(seed * 100 + color)
                row = full_game(agent, make_agent(opponent, greedy=greedy, device=device), games, minutes, seed, color)
                row["opponent"] = opponent
                row["team_mode"] = "all_learner"
                row.update(learner_count=4, teammate="none", learner_slots=list(range(4)))
                full_rows.append(row)
                seed_full.append(row)
        # Factorial cells: every seed sees every composition, color, rival and
        # proxy. A strong 3-bot cell can never substitute for the single bot.
        proxies = ["scripted:r3:0", "scripted:r3:2"]
        if teammate_reference:
            proxies.append(str(teammate_reference))
        for count in (1, 2, 3):
            for style in range(3):
                for proxy_spec in proxies:
                    for color in (0, 1):
                        slots = tuple((seed_index + color + i) % 4 for i in range(count))
                        proxy = make_agent(proxy_spec, greedy=greedy, device=device)
                        team = MixedTeamAgent(agent, proxy, slots)
                        opponent = f"scripted:r3:{style}"
                        torch.manual_seed(seed * 100 + color)
                        row = full_game(team, make_agent(opponent), games, minutes, seed, color)
                        row.update(opponent=opponent, team_mode="mixed", learner_slots=list(slots),
                                   learner_count=count, teammate=proxy_spec)
                        full_rows.append(row)
                        seed_full.append(row)
        for scenario in SCENARIOS:
            if scenario == "transition":
                continue  # explicit directions only in the sporting contract
            for color in (0, 1):
                torch.manual_seed(seed * 100 + color)
                row = functional_trial(agent, scenario, functional_games, seed, color)
                functional_rows.append(row)
                seed_functional.append(row)
        # Frozen imitator when provided; scripted approximation otherwise.
        proxy = make_agent(str(teammate_reference), greedy=greedy, device=device) if teammate_reference else ScriptedAgent(policy="r3", seed=seed)
        for count in (1, 2, 3):
            for color in (0, 1):
                slots = tuple((seed_index + color + count + i) % 4 for i in range(count))
                torch.manual_seed(seed * 100 + color)
                row = functional_trial(agent, "attack", functional_games, seed, color, teammate=proxy, learner_slots=slots)
                row["scenario"], row["learner_slots"] = "teammate", list(slots)
                functional_rows.append(row)
                seed_functional.append(row)
        full_by_seed[str(seed)] = {"mean_points": match_summary(seed_full)["balanced_points"]}
        functional_by_seed[str(seed)] = {"mean_success": float(np.mean([r["success"] for r in seed_functional]))}
    mean = lambda key, selected: float(np.mean([r[key] for r in selected])) if selected else 0.0
    scenario_rows = lambda *names: [row for row in functional_rows if row["scenario"] in names]
    restarts = scenario_rows("corner", "lateral", "goal_kick")
    resolved = sum(r["restarts"].get("resolved", 0) for r in restarts)
    skills = {"restart_success": sum(r["restarts"].get("successes", 0) for r in restarts) / resolved if resolved else 0,
              "corner_success_red": mean("success", [r for r in restarts if r["scenario"] == "corner" and r["color"] == 0]),
              "corner_success_blue": mean("success", [r for r in restarts if r["scenario"] == "corner" and r["color"] == 1]),
              "defense_conceded": mean("conceded", scenario_rows("defense", "defensive_transition")),
              "defense_recovery": mean("success", scenario_rows("defense", "defensive_transition")),
              "defense_danger_fraction": mean("danger_fraction", scenario_rows("defense", "defensive_transition")),
              "exit_success": mean("success", scenario_rows("exit")),
              "attack_success": mean("success", scenario_rows("attack", "offensive_transition")),
              "attack_goal_rate": mean("goal_rate", scenario_rows("attack", "offensive_transition")),
              "integrated_success": min(mean("success", scenario_rows("exit")),
                                          mean("success", scenario_rows("attack", "offensive_transition")),
                                          mean("success", scenario_rows("defense", "defensive_transition"))),
              "teammate_success": mean("success", scenario_rows("teammate"))}
    behavior = Counter()
    for row in full_rows:
        behavior.update(row["behavior"])
    return {"version": 3, "checkpoint": str(checkpoint), "seeds": list(seeds),
            "source_fingerprint": evaluation_source_fingerprint(),
            "suite": {"games": games, "functional_games": functional_games, "minutes": minutes,
                      "opponents": opponents, "opponent_sha256": {p: _file_fingerprint(p) for p in opponents if Path(p).is_file()},
                      "action_mode": action_mode, "holdout": holdout, "scenario_difficulty": 1., "inference_device": str(device),
                      "teammate_proxies": proxies,
                      "teammate_reference": str(teammate_reference) if teammate_reference else None,
                      "teammate_sha256": _file_fingerprint(teammate_reference)},
            "evaluation_contract": CONTRACT, "full_games": {**match_summary(full_rows),
            "by_seed": full_by_seed}, "functional": {"mean_success": mean("success", functional_rows),
            "cells": functional_summary(functional_rows),
            "by_seed": functional_by_seed, "skills": skills, "rows": functional_rows},
            "skills": skills, "behavior": dict(behavior),
            "behavior_definitions": {"dangerous_losses": "confirmed opponent transfer with ball in own defensive 35% zone",
              "coverage_recoveries": "after a transfer loss, at least two own players recover 10% field width behind the ball",
              "coverage_recovery_ticks": "sum of elapsed 60-Hz ticks for resolved recoveries; censored cases separate",
              "restarts": "per opportunity, one resolved outcome; open/censored opportunities excluded from resolved success rate",
              "exit_success": "goal or confirmed pass followed by one second of controlled exit with support",
              "attack_success": "goal or progressive pass, half-second of controlled central creation and subsequent goal-directed kick",
              "defense_recovery": "one second of controlled clearance beyond own defensive third with support",
              "integrated_success": "minimum of exit, creation/finishing and defensive recovery; match cells gated separately",
              "shots": "observable goal-directed kick trajectory proxy, not an expected-goals model"},
            "limitations": ["Frozen imitators and scripted teammates are proxies, not validation with real humans.",
                             "Functional success is an engineering proxy; inspect selected full-match replays."]}
