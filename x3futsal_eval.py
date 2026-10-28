"""Reproducible closed-loop Futsal x3 evaluation on the validated stadium physics.

  python -m learn.x3futsal_eval --checkpoint tactical --out reports/x3futsal_eval_tactical_20261028.json
  python -m learn.x3futsal_eval --checkpoint runs/x3futsal_bc/tactical_v3/best_competitive.pt --out reports/x3futsal_eval.json
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

try:
    import numba
except ImportError:  # pragma: no cover
    numba = None

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None

from learn.x3futsal_data import (
    DATA_VERSION,
    EFFECTIVE_KICK_DIST,
    FEATURE_CONTRACT,
    MOVE_BY_DELTA,
    MOVE_DX,
    MOVE_DY,
    TACTICAL_DATA_VERSION,
    TACTICAL_FEATURE_CONTRACT,
    featurize,
    featurize_tactical,
)
from learn.x3futsal_policy import (
    POLICY_VERSION,
    TACTICAL_POLICY_VERSION,
    X3SetPolicy,
    X3TacticalPolicy,
    decode_actions_np,
)
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


if nn is not None:
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
    def __init__(self, checkpoint_path: Path, device=None, *,
                 assist_mode: str = "hybrid", greedy: bool = True,
                 temperature: float = 0.85, kick_threshold: float = 0.28):
        if torch is None:
            raise RuntimeError("torch es necesario para cargar checkpoints de ModelPolicy")
        if assist_mode not in ("none", "kickoff", "hybrid"):
            raise ValueError(f"modo de asistencia inválido: {assist_mode}")
        self.device = torch.device("cpu") if device is None else device
        self.assist_mode = assist_mode
        self.greedy = bool(greedy)
        self.temperature = float(temperature)
        self.kick_threshold = float(kick_threshold)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.stadium_identity = checkpoint.get("datasetSignature", {}).get("targetStadiumIdentity")
        policy_version = checkpoint.get("policyVersion")
        obs_version = checkpoint.get("obs_version")
        if policy_version not in (None, POLICY_VERSION, TACTICAL_POLICY_VERSION):
            raise ValueError(f"versión de política X3 no soportada en {checkpoint_path}: {policy_version}")
        hidden = int(checkpoint.get("hidden", 128))
        entity_hidden = int(checkpoint.get("entityHidden", 64 if policy_version != TACTICAL_POLICY_VERSION else 96))
        self.tactical_obs = False
        if policy_version == TACTICAL_POLICY_VERSION:
            if obs_version not in (TACTICAL_DATA_VERSION, DATA_VERSION):
                raise ValueError(f"versión de observación incompatible en {checkpoint_path}")
            if checkpoint.get("featureContract") not in (TACTICAL_FEATURE_CONTRACT, FEATURE_CONTRACT):
                raise ValueError(f"contrato de features incompatible en {checkpoint_path}")
            use_history = bool(checkpoint.get("useHistory", False))
            self.model = X3TacticalPolicy(hidden=hidden, entity_hidden=entity_hidden, use_history=use_history)
            self.history = use_history
            self.tactical_obs = True
        elif policy_version == POLICY_VERSION:
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
        self.model.to(self.device).eval()
        self.name = f"{checkpoint_path.parent.name}/{checkpoint_path.stem}"
        stadium = load_stadium(STADIUM)
        self.half_w = stadium.field_half_w
        self.half_h = stadium.field_half_h
        self.goal_x = stadium.goal_x
        self.goal_hh = stadium.goal_half_height

    def actions(self, sim: BatchSim, team: int, scores: np.ndarray, elapsed: float,
                previous: np.ndarray, has_previous: np.ndarray) -> np.ndarray:
        slots = np.arange(3 * team, 3 * team + 3, dtype=np.int64)
        states = _player_states(sim)
        count = sim.N * 3
        repeated_states = np.repeat(states[:, None, :, :], 3, axis=1).reshape(count, 7, 4)
        actor_slots = np.tile(slots, sim.N)
        kick_cancel = getattr(sim, "kick_cancel", np.zeros((sim.N, 6), dtype=bool))[:, slots].reshape(-1)

        if self.tactical_obs:
            observation = featurize_tactical(
                repeated_states, actor_slots, self.half_w, self.half_h, self.goal_x, kick_cancel=kick_cancel
            )
        else:
            repeated_scores = np.repeat(scores, 3)
            times = np.full(count, elapsed, dtype=np.float32)
            observation = featurize(
                repeated_states, actor_slots, repeated_scores, times, self.half_w, self.half_h
            )

        with torch.no_grad():
            tensors = [torch.from_numpy(value).to(self.device) for value in observation]
            if self.history:
                actor_previous = previous[:, slots].reshape(-1)
                actor_has_previous = has_previous[:, slots].reshape(-1)
                logits = self.model(*tensors, torch.from_numpy(actor_previous).to(self.device),
                                    torch.from_numpy(actor_has_previous).to(self.device))
            else:
                logits = self.model(*tensors)
            logits_np = logits.cpu().numpy()

        orient = 1.0 if team == 0 else -1.0
        ball = sim.ball_pos * np.asarray([orient, 1.0])
        players = sim.player_pos[:, slots] * np.asarray([orient, 1.0])
        ball_delta = ball[:, None, :] - players
        ball_dist = np.linalg.norm(ball_delta, axis=-1).reshape(-1)
        goal_vec = np.asarray([self.goal_x, 0.0]) - ball
        goal_norm = np.linalg.norm(goal_vec, axis=-1, keepdims=True) + 1e-6
        shot_align = (
            (ball_delta[:, :, 0] * goal_vec[:, None, 0] + ball_delta[:, :, 1] * goal_vec[:, None, 1])
            / ((np.linalg.norm(ball_delta, axis=-1) + 1e-6) * goal_norm)
        ).reshape(-1)

        actions = decode_actions_np(
            logits_np,
            ball_dist_px=ball_dist,
            kick_cancel=kick_cancel,
            shot_alignment=shot_align,
            temperature=self.temperature,
            greedy=self.greedy,
            kick_threshold=self.kick_threshold,
        ).reshape(sim.N, 3)

        if self.assist_mode == "hybrid":
            tactical_acts = _tactical_actions(sim, team, self.half_w, self.half_h, self.goal_x, self.goal_hh)
            if np.any(sim.kickoff):
                actions[sim.kickoff] = tactical_acts[sim.kickoff]
            dist_matrix = ball_dist.reshape(sim.N, 3)
            for row in range(sim.N):
                if sim.kickoff[row]:
                    continue
                anchor = 0
                striker = int(dist_matrix[row, 1:].argmin() + 1)
                support = 3 - striker
                actions[row, anchor] = tactical_acts[row, anchor]
                if dist_matrix[row, striker] <= 195.0 or (actions[row, striker] % 9) == 0:
                    actions[row, striker] = tactical_acts[row, striker]
                if (actions[row, support] % 9) == 0 or abs(players[row, support, 0]) > self.half_w - 25.0:
                    actions[row, support] = tactical_acts[row, support]
        elif self.assist_mode == "kickoff" and np.any(sim.kickoff):
            kickoff_actions = _scripted_actions(sim, team, self.half_w, self.half_h)
            actions[sim.kickoff] = kickoff_actions[sim.kickoff]
        return actions


def _tactical_actions(sim: BatchSim, team: int, half_w: float, half_h: float,
                      goal_x: float = 557.5, goal_hh: float = 80.0) -> np.ndarray:
    """Role-coordinated 3v3 futsal policy (Dedicated Goalkeeper slot 0, Presser/Striker, Wing/Support).

    Combines deep goal-line goalkeeper protection (-half_w + 18 px), angled corner finishing,
    bouncy-wall corner unsticking, and kick_cancel management.
    """
    orient = 1.0 if team == 0 else -1.0
    vec_orient = np.asarray([orient, 1.0])
    ball_pos = sim.ball_pos * vec_orient
    ball_vel = getattr(sim, "ball_vel", np.zeros_like(ball_pos)) * vec_orient
    own_slots = slice(3 * team, 3 * team + 3)
    opp_slots = slice(3 * (1 - team), 3 * (1 - team) + 3)
    players = sim.player_pos[:, own_slots] * vec_orient
    player_vel = getattr(sim, "player_vel", np.zeros_like(sim.player_pos))[:, own_slots] * vec_orient
    rivals = sim.player_pos[:, opp_slots] * vec_orient
    kick_cancel = getattr(sim, "kick_cancel", np.zeros((sim.N, 6), dtype=bool))[:, own_slots]
    kickoff = getattr(sim, "kickoff", np.zeros(sim.N, dtype=bool))
    kickoff_team = getattr(sim, "kickoff_team", np.zeros(sim.N, dtype=np.int64))

    actions = np.zeros((sim.N, 3), dtype=np.int64)
    distances = np.linalg.norm(players - ball_pos[:, None, :], axis=-1)

    for row in range(sim.N):
        bx, by = float(ball_pos[row, 0]), float(ball_pos[row, 1])
        bvx, bvy = float(ball_vel[row, 0]), float(ball_vel[row, 1])
        pred_bx = float(np.clip(bx + 5.0 * bvx, -half_w + 12.0, half_w - 12.0))
        pred_by = float(np.clip(by + 5.0 * bvy, -half_h + 12.0, half_h - 12.0))

        # Dedicated goalkeeper on slot 0 (never abandons goal mouth for corner balls);
        # field slots 1 and 2 dynamically split striker vs support by intercept distance.
        anchor = 0
        d1 = float(np.hypot((players[row, 1, 0] + 4.0 * player_vel[row, 1, 0]) - pred_bx,
                            (players[row, 1, 1] + 4.0 * player_vel[row, 1, 1]) - pred_by))
        d2 = float(np.hypot((players[row, 2, 0] + 4.0 * player_vel[row, 2, 0]) - pred_bx,
                            (players[row, 2, 1] + 4.0 * player_vel[row, 2, 1]) - pred_by))
        if d1 <= d2:
            striker, support = 1, 2
        else:
            striker, support = 2, 1

        opp_gk_idx = int(np.argmax(rivals[row, :, 0]))
        opp_gk_y = float(rivals[row, opp_gk_idx, 1])
        aim_y = -46.0 if opp_gk_y >= 0.0 else 46.0
        if abs(by) > 135.0:
            aim_y = float(np.sign(by) * 32.0)

        for slot in range(3):
            px, py = float(players[row, slot, 0]), float(players[row, slot, 1])
            pvx, pvy = float(player_vel[row, slot, 0]), float(player_vel[row, slot, 1])
            dist = float(distances[row, slot])
            is_cancelled = bool(kick_cancel[row, slot])

            if kickoff[row] and int(kickoff_team[row]) != team and slot != anchor:
                lane = -48.0 if slot == striker else 48.0
                target = np.asarray([-95.0, lane])
                delta = target - (players[row, slot] + 2.0 * player_vel[row, slot])
                actions[row, slot] = _quantize_direction(delta, dead=5.0)
                continue

            if slot == anchor:
                # Deep goal-line keeper: stays at x = -half_w + 18 (-532 px) when ball is in own half
                # so no corner bounce at x = -534 can ever get behind the keeper.
                in_goal_mouth = bx < -half_w + 115.0 and abs(by) <= goal_hh + 28.0 and dist < 95.0
                if in_goal_mouth and px <= bx + 2.0:
                    clear_y = -half_h * 0.75 if by < 0.0 else half_h * 0.75
                    ax, ay = 200.0 - bx, clear_y - by
                    norm = float(np.hypot(ax, ay)) + 1e-6
                    target = np.asarray([bx + (ax / norm) * 10.0, by + (ay / norm) * 10.0])
                    delta = target - players[row, slot]
                    direction = _quantize_direction(delta, dead=2.5)
                    can_clear = dist <= EFFECTIVE_KICK_DIST and px <= bx + 1.0 and not is_cancelled
                    actions[row, slot] = direction + (9 if can_clear else 0)
                else:
                    guard_x = -half_w + 18.0 if bx < -80.0 else -half_w + 34.0
                    if bvx < -0.25 and bx > guard_x:
                        ticks_to_line = min(45.0, max(0.0, (bx - guard_x) / max(-bvx, 0.3)))
                        proj_y = by + bvy * ticks_to_line
                    else:
                        proj_y = by * 0.50
                    guard_y = float(np.clip(proj_y, -goal_hh + 15.0, goal_hh - 15.0))
                    target = np.asarray([guard_x, guard_y])
                    delta = target - (players[row, slot] + 2.5 * player_vel[row, slot])
                    direction = _quantize_direction(delta, dead=3.5)
                    can_kick = dist <= EFFECTIVE_KICK_DIST and px <= bx + 1.0 and not is_cancelled
                    actions[row, slot] = direction + (9 if can_kick else 0)

            elif slot == striker:
                pinned_corner = bx > half_w - 65.0 and abs(by) > goal_hh + 25.0
                pinned_sidewall = abs(by) > half_h - 28.0
                if pinned_corner or pinned_sidewall:
                    target = np.asarray([bx - 16.0, float(np.clip(by, -half_h + 18.0, half_h - 18.0))])
                    delta = target - players[row, slot]
                    direction = _quantize_direction(delta, dead=2.0)
                    kick_ok = dist <= EFFECTIVE_KICK_DIST and px < bx - 2.0 and not is_cancelled
                    actions[row, slot] = direction + (9 if kick_ok else 0)
                    continue

                sup_x, sup_y = float(players[row, support, 0]), float(players[row, support, 1])
                min_riv_to_sup = float(np.min(np.hypot(rivals[row, :, 0] - sup_x, rivals[row, :, 1] - sup_y)))
                if (sup_x > bx + 85.0 and abs(sup_y) < half_h - 30.0
                        and min_riv_to_sup > 95.0 and bx < half_w - 220.0):
                    tx_aim, ty_aim = sup_x + 25.0, sup_y
                elif bx < -half_w + 160.0:
                    tx_aim, ty_aim = 150.0, (-half_h * 0.7 if by < 0.0 else half_h * 0.7)
                else:
                    tx_aim, ty_aim = goal_x, aim_y

                ax, ay = tx_aim - pred_bx, ty_aim - pred_by
                anorm = float(np.hypot(ax, ay)) + 1e-6
                ux, uy = ax / anorm, ay / anorm

                vx, vy = bx - px, by - py
                vnorm = float(np.hypot(vx, vy)) + 1e-6
                cos_align = (vx * ux + vy * uy) / vnorm

                if px > bx + 4.0:
                    bypass_y = 32.0 if py >= by else -32.0
                    target = np.asarray([pred_bx - 26.0, float(np.clip(pred_by + bypass_y, -half_h + 18.0, half_h - 18.0))])
                    dead = 2.0
                elif cos_align < 0.58 and dist > 26.0:
                    lead = min(16.0, max(8.0, dist * 0.22))
                    target = np.asarray([
                        pred_bx - ux * lead,
                        float(np.clip(pred_by - uy * lead, -half_h + 18.0, half_h - 18.0)),
                    ])
                    dead = 2.0
                else:
                    target = np.asarray([bx + ux * 18.0, float(np.clip(by + uy * 18.0, -half_h + 15.0, half_h - 15.0))])
                    dead = 1.0

                delta = target - (players[row, slot] + 1.5 * np.asarray([pvx, pvy]))
                direction = _quantize_direction(delta, dead=dead)
                kick_ok = (
                    dist <= EFFECTIVE_KICK_DIST
                    and px < bx + 1.0
                    and cos_align >= 0.42
                    and not is_cancelled
                )
                actions[row, slot] = direction + (9 if kick_ok else 0)

            else:
                if bx > -60.0:
                    wing_sign = -1.0 if by > 20.0 else (1.0 if by < -20.0 else (-1.0 if slot == 1 else 1.0))
                    sup_tx = float(np.clip(pred_bx - 45.0, -half_w + 160.0, half_w - 95.0))
                    sup_ty = float(np.clip(wing_sign * 105.0, -half_h + 35.0, half_h - 35.0))
                else:
                    wing_sign = -1.0 if by > 0.0 else 1.0
                    sup_tx = float(np.clip(pred_bx * 0.55 - 95.0, -half_w + 130.0, 60.0))
                    sup_ty = float(np.clip(wing_sign * 58.0, -half_h + 40.0, half_h - 40.0))
                target = np.asarray([sup_tx, sup_ty])
                delta = target - (players[row, slot] + 2.0 * player_vel[row, slot])
                direction = _quantize_direction(delta, dead=6.0)
                kick_ok = (
                    dist <= EFFECTIVE_KICK_DIST
                    and px < bx - 2.0
                    and not is_cancelled
                )
                actions[row, slot] = direction + (9 if kick_ok else 0)

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


class TacticalPolicy(Policy):
    """Role-coordinated 3v3 futsal policy with kick-cancel management and corner finishing."""

    def __init__(self, stadium):
        self.name = "tactical"
        self.half_w = stadium.field_half_w
        self.half_h = stadium.field_half_h
        self.goal_x = stadium.goal_x
        self.goal_hh = stadium.goal_half_height
        self.stadium_identity = None

    def actions(self, sim: BatchSim, team: int, scores: np.ndarray, elapsed: float,
                previous: np.ndarray, has_previous: np.ndarray) -> np.ndarray:
        return _tactical_actions(sim, team, self.half_w, self.half_h, self.goal_x, self.goal_hh)


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


def _quantize_direction(delta: np.ndarray, dead: float = 8.0) -> int:
    if np.linalg.norm(delta) < dead:
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

        # Track kick_cancel at the start of the 3-frame window: if a player wants to kick (action >= 9)
        # while kick_cancel is True, pulse movement-only on sub-frame 0 so kick_cancel resets for sub-frame 1.
        rearm_mask = (native_applied >= 9) & sim.kick_cancel
        step_actions = native_applied.copy()
        for sub_frame in range(FRAME_SKIP):
            active = ticks_played < tick_limit
            if not active.any():
                break
            if sub_frame == 0 and np.any(rearm_mask):
                step_actions = np.where(rearm_mask, native_applied % 9, native_applied)
            elif sub_frame == 1 and np.any(rearm_mask):
                step_actions = native_applied.copy()
            ball_before = sim.ball_pos.copy()
            sim.step(step_actions)
            if np.any(sim.kicked):
                step_actions = np.where(sim.kicked, step_actions % 9, step_actions)
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


def _load_policy_spec(spec: str | Path, stadium, device, *, assist_mode: str = "hybrid") -> Policy:
    text = str(spec)
    if text == "tactical":
        return TacticalPolicy(stadium)
    if text == "scripted":
        return ScriptedPolicy(stadium)
    if text == "repeat":
        return RepeatPolicy()
    return ModelPolicy(Path(spec), device, assist_mode=assist_mode)


def evaluate(checkpoint: Path | str, pool: list[Path | str], *, matches: int, minutes: float,
             seed: int, device_name: str, threads: int, assist_mode: str = "hybrid") -> dict:
    if matches < 1 or minutes <= 0 or threads < 1:
        raise ValueError("parámetros de evaluación inválidos")
    if torch is not None:
        torch.set_num_threads(threads)
        torch.manual_seed(seed)
        device = torch.device(device_name)
    else:
        device = None
    if numba is not None:
        numba.set_num_threads(threads)
    stadium = load_stadium(STADIUM)
    policies: list[Policy] = [_load_policy_spec(checkpoint, stadium, device, assist_mode=assist_mode)]
    policies.extend(_load_policy_spec(path, stadium, device, assist_mode=assist_mode) for path in pool)
    for builtin in (TacticalPolicy(stadium), ScriptedPolicy(stadium), RepeatPolicy()):
        if all(p.name != builtin.name for p in policies):
            policies.append(builtin)
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

    from tools.x3futsal_audit import stadium_identity
    identity = getattr(policies[0], "stadium_identity", None) or stadium_identity(STADIUM)
    return {
        "version": "x3-closed-loop-eval-4",
        "stadium": stadium.name,
        "stadiumIdentity": identity,
        "seed": seed,
        "minutesPerMatch": minutes,
        "matchesPerSide": matches,
        "decisionFrames": FRAME_SKIP,
        "assistMode": assist_mode,
        "controlMode": "host-local; no HaxBall network ping modeled",
        "kickoffSafetyTicks": KICKOFF_SAFETY_TICKS,
        "policies": [policy.name for policy in policies],
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="tactical",
                        help="ruta a checkpoint o nombre de política ('tactical', 'scripted')")
    parser.add_argument("--pool", default="", help="checkpoints de políticas previas, separados por coma")
    parser.add_argument("--matches", type=int, default=4, help="partidos por política y lado")
    parser.add_argument("--minutes", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--assist", choices=("none", "kickoff", "hybrid"), default="hybrid",
                        help="modo de arbitraje táctico en lazo cerrado para checkpoints BC")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    pool = [value for value in args.pool.split(",") if value]
    report = evaluate(args.checkpoint, pool, matches=args.matches, minutes=args.minutes,
                      seed=args.seed, device_name=args.device, threads=args.threads,
                      assist_mode=args.assist)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "policies": report["policies"],
                      "pairs": len(report["results"])}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()