"""Reproducible closed-loop Futsal x3 evaluation on the validated stadium physics.

  python -m learn.x3futsal_eval --checkpoint runs/x3futsal_bc/changed_v1/best.pt
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import numba
import torch
from torch import nn

from learn.x3futsal_data import DATA_VERSION, FEATURE_CONTRACT, MOVE_DX, MOVE_DY, MOVE_BY_DELTA, featurize
from learn.x3futsal_policy import POLICY_VERSION, X3SetPolicy
from sim.physics import BatchSim
from sim.stadium import load_stadium

ROOT = Path(__file__).resolve().parent.parent
STADIUM = ROOT / "stadiums" / "futsal3bybazinga.hbs"
FRAME_SKIP = 3
KICKOFF_SAFETY_TICKS = 300
LEGACY_DATA_VERSION = "x3-futsal-bc-v1"
PREVIOUS_DATA_VERSION = "x3-futsal-bc-v2"


def action_to_sim(action: int, team: int) -> int:
    """Convert an attack-oriented model action to native HaxBall movement for red/blue."""
    action = int(action)
    if not 0 <= action < 18 or team not in (0, 1):
        raise ValueError(f"acción/equipo inválido: action={action}, team={team}")
    direction = action % 9
    dx, dy = MOVE_DX[direction], MOVE_DY[direction]
    if team == 1:
        dx = -dx
    return MOVE_BY_DELTA[(dx, dy)] + (9 if action >= 9 else 0)


class LegacyX3SetPolicy(nn.Module):
    """The first, stateless X3 checkpoint architecture, retained as a fixed pool opponent."""

    def __init__(self, hidden: int = 128, entity_hidden: int = 64):
        super().__init__()
        self.self_net = nn.Sequential(nn.Linear(12, hidden), nn.ReLU(),
                                      nn.Linear(hidden, hidden), nn.ReLU())
        self.mate_net = nn.Sequential(nn.Linear(6, entity_hidden), nn.ReLU(),
                                      nn.Linear(entity_hidden, entity_hidden), nn.ReLU())
        self.rival_net = nn.Sequential(nn.Linear(6, entity_hidden), nn.ReLU(),
                                       nn.Linear(entity_hidden, entity_hidden), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(hidden + 4 * entity_hidden, hidden), nn.ReLU(),
                                  nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 18))

    @staticmethod
    def _pool(entities: torch.Tensor) -> torch.Tensor:
        return torch.cat((entities.mean(dim=1), entities.max(dim=1).values), dim=-1)

    def forward(self, own: torch.Tensor, mates: torch.Tensor, rivals: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat((self.self_net(own), self._pool(self.mate_net(mates)),
                                    self._pool(self.rival_net(rivals))), dim=-1))


class Policy:
    name: str

    def actions(self, sim: BatchSim, team: int, scores: np.ndarray, elapsed: float,
                previous: np.ndarray, has_previous: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class ModelPolicy(Policy):
    def __init__(self, checkpoint_path: Path, device: torch.device):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.stadium_identity = checkpoint.get("datasetSignature", {}).get("targetStadiumIdentity")
        policy_version = checkpoint.get("policyVersion")
        obs_version = checkpoint.get("obs_version")
        if policy_version not in (None, POLICY_VERSION):
            raise ValueError(f"versión de política X3 no soportada en {checkpoint_path}: {policy_version}")
        hidden = int(checkpoint.get("hidden", 128))
        if policy_version == POLICY_VERSION:
            if obs_version not in (DATA_VERSION, PREVIOUS_DATA_VERSION, LEGACY_DATA_VERSION):
                raise ValueError(f"versión de observación incompatible en {checkpoint_path}")
            if checkpoint.get("featureContract") != FEATURE_CONTRACT:
                raise ValueError(f"contrato de features incompatible en {checkpoint_path}")
            self.model = X3SetPolicy(hidden=hidden)
            self.history = True
        elif obs_version == LEGACY_DATA_VERSION and checkpoint.get("obs_dims") == [12, 6, 6]:
            self.model = LegacyX3SetPolicy(hidden=hidden)
            self.history = False
        else:
            raise ValueError(f"checkpoint X3 no reconocido: {checkpoint_path}")
        self.model.load_state_dict(checkpoint["model"])
        self.model.to(device).eval()
        self.device = device
        self.name = f"{checkpoint_path.parent.name}/{checkpoint_path.stem}"
        stadium = load_stadium(STADIUM)
        self.half_w = stadium.field_half_w
        self.half_h = stadium.field_half_h

    @torch.no_grad()
    def actions(self, sim: BatchSim, team: int, scores: np.ndarray, elapsed: float,
                previous: np.ndarray, has_previous: np.ndarray) -> np.ndarray:
        if np.any(sim.kickoff):
            # The replay contract has a valid kickoff state, but argmax BC policies
            # can still collapse to "stay" before the first active transition.
            kickoff_actions = _scripted_actions(sim, team, self.half_w, self.half_h)
        else:
            kickoff_actions = None
        slots = np.arange(3 * team, 3 * team + 3, dtype=np.int64)
        states = _player_states(sim)
        count = sim.N * 3
        repeated_states = np.repeat(states[:, None, :, :], 3, axis=1).reshape(count, 7, 4)
        repeated_scores = np.repeat(scores, 3)
        times = np.full(count, elapsed, dtype=np.float32)
        actor_slots = np.tile(slots, sim.N)
        observation = featurize(repeated_states, actor_slots, repeated_scores, times,
                                sim.st.field_half_w, sim.st.field_half_h)
        tensors = [torch.from_numpy(value).to(self.device) for value in observation]
        if self.history:
            actor_previous = previous[:, slots].reshape(-1)
            actor_has_previous = has_previous[:, slots].reshape(-1)
            logits = self.model(*tensors, torch.from_numpy(actor_previous).to(self.device),
                                torch.from_numpy(actor_has_previous).to(self.device))
        else:
            logits = self.model(*tensors)
        actions = logits.argmax(-1).cpu().numpy().reshape(sim.N, 3)
        if kickoff_actions is not None:
            actions[sim.kickoff] = kickoff_actions[sim.kickoff]
        return actions


def _scripted_actions(sim: BatchSim, team: int, half_w: float, half_h: float) -> np.ndarray:
    orient = 1.0 if team == 0 else -1.0
    ball = sim.ball_pos * np.asarray([orient, 1.0])
    players = sim.player_pos[:, 3 * team:3 * team + 3] * np.asarray([orient, 1.0])
    actions = np.zeros((sim.N, 3), dtype=np.int64)
    distances = np.linalg.norm(players - ball[:, None, :], axis=-1)
    striker = distances[:, 1:].argmin(axis=1) + 1
    for row in range(sim.N):
        for slot in range(3):
            if slot == 0:
                target = np.asarray([-half_w + 55.0, np.clip(ball[row, 1] * 0.35,
                                                               -0.55 * half_h,
                                                               0.55 * half_h)])
            elif slot == striker[row]:
                target = ball[row] + np.asarray([-16.0, 0.0])
            else:
                lane = -1.0 if slot == 1 else 1.0
                target = ball[row] + np.asarray([-110.0, lane * 65.0])
            delta = target - players[row, slot]
            direction = _quantize_direction(delta)
            kick = (slot == striker[row] and distances[row, slot] <= 25.0
                    and players[row, slot, 0] < ball[row, 0] - 3.0)
            actions[row, slot] = direction + (9 if kick else 0)
    return actions


class ScriptedPolicy(Policy):
    """A fixed keeper/support/nearest-player chase policy used as the simple play baseline."""

    def __init__(self, stadium):
        self.name = "scripted"
        self.half_w = stadium.field_half_w
        self.half_h = stadium.field_half_h

    def actions(self, sim: BatchSim, team: int, scores: np.ndarray, elapsed: float,
                previous: np.ndarray, has_previous: np.ndarray) -> np.ndarray:
        return _scripted_actions(sim, team, self.half_w, self.half_h)


class RepeatPolicy(Policy):
    def __init__(self):
        self.name = "repeat"

    def actions(self, sim: BatchSim, team: int, scores: np.ndarray, elapsed: float,
                previous: np.ndarray, has_previous: np.ndarray) -> np.ndarray:
        slots = slice(3 * team, 3 * team + 3)
        return previous[:, slots].copy()


def _quantize_direction(delta: np.ndarray) -> int:
    if np.linalg.norm(delta) < 8.0:
        return 0
    best, best_dot = 0, -float("inf")
    length = float(np.linalg.norm(delta))
    for direction, (dx, dy) in enumerate(zip(MOVE_DX, MOVE_DY)):
        if direction == 0:
            continue
        norm = np.hypot(dx, dy)
        dot = (dx * delta[0] + dy * delta[1]) / (norm * length)
        if dot > best_dot:
            best, best_dot = direction, dot
    return best


def _player_states(sim: BatchSim) -> np.ndarray:
    states = np.empty((sim.N, 7, 4), dtype=np.float32)
    states[:, 0, :2] = sim.ball_pos
    states[:, 0, 2:] = sim.ball_vel
    states[:, 1:4, :2] = sim.player_pos[:, :3]
    states[:, 1:4, 2:] = sim.player_vel[:, :3]
    states[:, 4:7, :2] = sim.player_pos[:, 3:]
    states[:, 4:7, 2:] = sim.player_vel[:, 3:]
    return states


def play_batch(red: Policy, blue: Policy, *, matches: int, minutes: float, seed: int) -> list[dict]:
    if matches < 1 or minutes <= 0:
        raise ValueError("matches/minutes inválidos")
    stadium = load_stadium(STADIUM)
    sim = BatchSim(matches, 3, 3, stadium=stadium, seed=seed)
    rng = np.random.default_rng(seed)
    kickoff_team = rng.integers(0, 2, size=matches)
    sim.reset_kickoff(np.arange(matches), kickoff_team=kickoff_team)
    previous = np.zeros((matches, 6), dtype=np.int64)
    has_previous = np.zeros((matches, 6), dtype=bool)
    owner = np.full(matches, -1, dtype=np.int64)
    pending_passer = np.full(matches, -1, dtype=np.int64)
    kickoff_ticks = np.zeros(matches, dtype=np.int64)
    goals = np.zeros((matches, 2), dtype=np.int64)
    team_metrics = np.zeros((matches, 2, 8), dtype=np.float64)
    # kick, shots, completed passes, turnovers, touches, movement decisions, speed, kickoff safety
    tick_limit = int(round(minutes * 60 * 60))
    ticks_played = np.zeros(matches, dtype=np.int64)
    spread_sum = np.zeros((matches, 2), dtype=np.float64)
    spread_samples = np.zeros(matches, dtype=np.int64)
    decision = 0

    while np.any(ticks_played < tick_limit):
        live = ticks_played < tick_limit
        scores = goals[:, 0] - goals[:, 1]
        seconds = decision * FRAME_SKIP / 60.0
        intents = np.zeros((matches, 6), dtype=np.int64)
        intents[:, :3] = red.actions(sim, 0, scores, seconds, previous, has_previous)
        intents[:, 3:] = blue.actions(sim, 1, scores, seconds, previous, has_previous)
        applied = intents
        previous[live] = applied[live]
        has_previous[live] = True
        native_applied = applied.copy()
        for team in range(2):
            slots = slice(3 * team, 3 * team + 3)
            for match in range(matches):
                native_applied[match, slots] = [action_to_sim(action, team)
                                                 for action in applied[match, slots]]
        team_metrics[live, 0, 5] += np.count_nonzero(applied[live, :3] % 9, axis=1)
        team_metrics[live, 1, 5] += np.count_nonzero(applied[live, 3:] % 9, axis=1)
        speed = np.linalg.norm(sim.player_vel, axis=-1)
        team_metrics[live, 0, 6] += speed[live, :3].mean(axis=1)
        team_metrics[live, 1, 6] += speed[live, 3:].mean(axis=1)
        if decision % 10 == 0:
            positions = sim.player_pos
            for team in range(2):
                side = positions[:, 3 * team:3 * team + 3]
                d01 = np.linalg.norm(side[:, 0] - side[:, 1], axis=1)
                d02 = np.linalg.norm(side[:, 0] - side[:, 2], axis=1)
                d12 = np.linalg.norm(side[:, 1] - side[:, 2], axis=1)
                spread_sum[live, team] += ((d01 + d02 + d12) / 3.0)[live]
            spread_samples[live] += 1

        for _ in range(FRAME_SKIP):
            active = ticks_played < tick_limit
            if not active.any():
                break
            ball_before = sim.ball_pos.copy()
            sim.step(native_applied)
            for match in np.flatnonzero(active):
                kickers = np.flatnonzero(sim.kicked[match])
                for player in kickers:
                    team = int(player // 3)
                    team_metrics[match, team, 0] += 1
                    orient = 1.0 if team == 0 else -1.0
                    if (ball_before[match, 0] * orient > 0.35 * stadium.goal_x
                            and sim.ball_vel[match, 0] * orient > 0.5):
                        team_metrics[match, team, 1] += 1

                touched = np.flatnonzero(sim.touch[match])
                if len(touched):
                    player = int(touched[-1])
                    team = player // 3
                    team_metrics[match, team, 4] += 1
                    if owner[match] >= 0 and owner[match] // 3 != team:
                        team_metrics[match, team, 3] += 1
                        pending_passer[match] = -1
                    elif (pending_passer[match] >= 0 and pending_passer[match] // 3 == team
                          and pending_passer[match] != player):
                        team_metrics[match, team, 2] += 1
                        pending_passer[match] = -1
                    owner[match] = player
                if len(kickers):
                    pending_passer[match] = int(kickers[-1])

                goal = int(sim.goal[match])
                ticks_played[match] += 1
                if goal:
                    scoring_team = 0 if goal > 0 else 1
                    conceding_team = 1 - scoring_team
                    goals[match, scoring_team] += 1
                    sim.reset_kickoff([match], kickoff_team=conceding_team)
                    kickoff_ticks[match] = 0
                    owner[match] = -1
                    pending_passer[match] = -1
                    previous[match] = 0
                    has_previous[match] = False
                elif sim.kickoff[match]:
                    kickoff_ticks[match] += 1
                    if kickoff_ticks[match] >= KICKOFF_SAFETY_TICKS:
                        sim.kickoff[match] = False
                        sim.mask[match, sim.first_player:] = sim.base_mask[sim.first_player:]
                        team_metrics[match, int(sim.kickoff_team[match]), 7] += 1
                        kickoff_ticks[match] = 0
                else:
                    kickoff_ticks[match] = 0
        decision += 1

    result = []
    for match in range(matches):
        result.append({
            "goals": goals[match].tolist(),
            "ticks": int(ticks_played[match]),
            "safetyKickoffs": int(team_metrics[match, :, 7].sum()),
            "teams": [
                {
                    "kicks": int(team_metrics[match, team, 0]),
                    "shots": int(team_metrics[match, team, 1]),
                    "completedPasses": int(team_metrics[match, team, 2]),
                    "turnovers": int(team_metrics[match, team, 3]),
                    "touches": int(team_metrics[match, team, 4]),
                    "movementDecisions": int(team_metrics[match, team, 5]),
                    "meanPlayerSpeed": float(team_metrics[match, team, 6] / max(1, decision)),
                    "safetyKickoffs": int(team_metrics[match, team, 7]),
                    "meanTeamSpread": float(spread_sum[match, team] / max(1, spread_samples[match])),
                }
                for team in range(2)
            ],
        })
    return result


def _aggregate(games: list[dict], policy_role: int) -> dict:
    wins = draws = losses = goals_for = goals_against = 0
    stat_names = ("kicks", "shots", "completedPasses", "turnovers", "touches",
                  "movementDecisions", "meanPlayerSpeed", "meanTeamSpread", "safetyKickoffs")
    stats = {name: 0.0 for name in stat_names}
    for game in games:
        own_team = game["firstPolicyTeam"] if policy_role == 0 else 1 - game["firstPolicyTeam"]
        other_team = 1 - own_team
        own, other = game["goals"][own_team], game["goals"][other_team]
        goals_for += own
        goals_against += other
        wins += own > other
        losses += own < other
        draws += own == other
        for name in stat_names:
            stats[name] += game["teams"][own_team][name]
    count = max(1, len(games))
    return {
        "matches": len(games), "wins": int(wins), "draws": int(draws), "losses": int(losses),
        "winRate": float(wins / count), "goalFor": int(goals_for), "goalAgainst": int(goals_against),
        "goalDifferencePerMatch": float((goals_for - goals_against) / count),
        **{f"{name}PerMatch": float(value / count) for name, value in stats.items()},
    }


def evaluate(checkpoint: Path, pool: list[Path], *, matches: int, minutes: float,
             seed: int, device_name: str, threads: int) -> dict:
    if matches < 1 or minutes <= 0 or threads < 1:
        raise ValueError("parámetros de evaluación inválidos")
    torch.set_num_threads(threads)
    numba.set_num_threads(threads)
    torch.manual_seed(seed)
    device = torch.device(device_name)
    stadium = load_stadium(STADIUM)
    policies: list[Policy] = [ModelPolicy(checkpoint, device)]
    policies.extend(ModelPolicy(path, device) for path in pool)
    policies.extend((ScriptedPolicy(stadium), RepeatPolicy()))
    if len({policy.name for policy in policies}) != len(policies):
        raise ValueError("hay nombres de políticas duplicados en el pool")

    results = {}
    for first, second in itertools.combinations(policies, 2):
        pair_seed = seed + policies.index(first) * 100_003 + policies.index(second) * 10_007
        games = []
        for red, blue, policy_index in ((first, second, 0), (second, first, 1)):
            side_games = play_batch(red, blue, matches=matches, minutes=minutes, seed=pair_seed)
            games.extend({**game, "firstPolicyTeam": policy_index} for game in side_games)
        results[f"{first.name} vs {second.name}"] = {
            "framesPerDecision": FRAME_SKIP,
            "networkLatencyModeled": False,
            "firstPolicy": _aggregate(games, 0),
            "secondPolicy": _aggregate(games, 1),
            "firstPolicyBySide": {
                "red": _aggregate([g for g in games if g["firstPolicyTeam"] == 0], 0),
                "blue": _aggregate([g for g in games if g["firstPolicyTeam"] == 1], 0),
            },
            "secondPolicyBySide": {
                "red": _aggregate([g for g in games if g["firstPolicyTeam"] == 1], 1),
                "blue": _aggregate([g for g in games if g["firstPolicyTeam"] == 0], 1),
            },
            "games": games,
        }

    return {
        "version": "x3-closed-loop-eval-3",
        "stadium": stadium.name,
        "stadiumIdentity": policies[0].stadium_identity,
        "seed": seed,
        "minutesPerMatch": minutes,
        "matchesPerSide": matches,
        "decisionFrames": FRAME_SKIP,
        "controlMode": "host-local; no HaxBall network ping modeled",
        "kickoffSafetyTicks": KICKOFF_SAFETY_TICKS,
        "policies": [policy.name for policy in policies],
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pool", default="", help="checkpoints de políticas previas, separados por coma")
    parser.add_argument("--matches", type=int, default=4, help="partidos por política y lado")
    parser.add_argument("--minutes", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    pool = [Path(value) for value in args.pool.split(",") if value]
    report = evaluate(args.checkpoint, pool, matches=args.matches, minutes=args.minutes,
                      seed=args.seed, device_name=args.device, threads=args.threads)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "policies": report["policies"],
                      "pairs": len(report["results"])}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
