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
    MIRROR_X_ACTION,
    MOVE_BY_DELTA,
    MOVE_DX,
    MOVE_DY,
    TACTICAL_DATA_VERSION,
    TACTICAL_FEATURE_CONTRACT,
    featurize,
    featurize_tactical,
    featurize_tactical_team,
)
from learn.x3futsal_policy import (
    POLICY_VERSION,
    TACTICAL_POLICY_VERSION,
    X3SetPolicy,
    X3TacticalPolicy,
    decode_actions_np,
)
from sim.physics import MOVE_UNIT, BatchSim, set_physics_threads
from sim.stadium import load_stadium

ROOT = Path(__file__).resolve().parent.parent
STADIUM = ROOT / "stadiums" / "futsal3bybazinga.hbs"
FRAME_SKIP = 3
KICKOFF_SAFETY_TICKS = 300
LEGACY_DATA_VERSION = "x3-futsal-bc-v1"
PREVIOUS_DATA_VERSION = "x3-futsal-bc-v2"

_MOVE_DIRS_8 = np.ascontiguousarray(MOVE_UNIT[1:], dtype=np.float64)


def action_to_sim(action: int, team: int) -> int:
    """Convert an attack-oriented model action to native HaxBall movement for red/blue."""
    action = int(action)
    if not 0 <= action < 18 or team not in (0, 1):
        raise ValueError(f"acción/equipo inválido: action={action}, team={team}")
    if team == 0:
        return action
    return int(MIRROR_X_ACTION[action])


def action_to_sim_batch(actions: np.ndarray, team: int) -> np.ndarray:
    """Vectorized conversion of attack-oriented actions `(...,)` to native HaxBall simulation actions."""
    actions = np.asarray(actions, dtype=np.int64)
    if team == 0:
        return actions.copy()
    if team == 1:
        return MIRROR_X_ACTION[actions]
    raise ValueError(f"equipo inválido: {team}")


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
        kick_cancel = getattr(sim, "kick_cancel", np.zeros((sim.N, 6), dtype=bool))[:, slots].reshape(-1)

        if self.tactical_obs:
            observation = featurize_tactical_team(sim, team, self.half_w, self.half_h, self.goal_x)
        else:
            states = _player_states(sim)
            count = sim.N * 3
            repeated_states = np.repeat(states[:, None, :, :], 3, axis=1).reshape(count, 7, 4)
            actor_slots = np.tile(slots, sim.N)
            repeated_scores = np.repeat(scores, 3)
            times = np.full(count, elapsed, dtype=np.float32)
            observation = featurize(
                repeated_states, actor_slots, repeated_scores, times, self.half_w, self.half_h
            )

        ctx_mgr = torch.inference_mode if hasattr(torch, "inference_mode") else torch.no_grad
        with ctx_mgr():
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
        ball_dist = np.hypot(ball_delta[:, :, 0], ball_delta[:, :, 1]).reshape(-1)
        goal_vec = np.asarray([self.goal_x, 0.0]) - ball
        goal_norm = np.hypot(goal_vec[:, 0], goal_vec[:, 1])[:, None] + 1e-6
        shot_align = (
            (ball_delta[:, :, 0] * goal_vec[:, None, 0] + ball_delta[:, :, 1] * goal_vec[:, None, 1])
            / ((ball_dist.reshape(sim.N, 3) + 1e-6) * goal_norm)
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
            # El modo híbrido ya NO ancla a ningún jugador (slot 0) como arquero fijo en el área:
            # 1. Coordina los saques iniciales (kickoff) y desbloquea jugadores pasivos.
            # 2. Si un jugador del modelo intenta quedarse acampando dentro del área propia mientras
            #    la pelota está adelantada, lo impulsa a salir y rotar con el equipo.
            tactical_acts = _tactical_actions(sim, team, self.half_w, self.half_h, self.goal_x, self.goal_hh)
            if np.any(sim.kickoff):
                actions[sim.kickoff] = tactical_acts[sim.kickoff]
            open_play = (~sim.kickoff)[:, None]
            dist_matrix = ball_dist.reshape(sim.N, 3)
            px = players[:, :, 0]
            py = players[:, :, 1]
            bx = ball[:, 0:1]
            move_dir = actions % 9
            # Evitar que cualquier jugador quede anclado en su área propia cuando la pelota está lejos
            box_camping = (
                open_play
                & (px < -self.half_w + 115.0)
                & (bx > -self.half_w + 210.0)
                & ((move_dir == 0) | (move_dir == 6) | (move_dir == 7) | (move_dir == 8))
            )
            passive_or_pinned = (
                open_play
                & (
                    (move_dir == 0)
                    | (dist_matrix <= 165.0)
                    | (np.abs(px) > self.half_w - 22.0)
                    | (np.abs(py) > self.half_h - 20.0)
                )
            )
            override_mask = box_camping | passive_or_pinned
            if np.any(override_mask):
                actions = np.where(override_mask, tactical_acts, actions)
        elif self.assist_mode == "kickoff" and np.any(sim.kickoff):
            kickoff_actions = _tactical_actions(sim, team, self.half_w, self.half_h, self.goal_x, self.goal_hh)
            actions[sim.kickoff] = kickoff_actions[sim.kickoff]
        return actions


def _quantize_direction(delta: np.ndarray, dead: float = 8.0) -> int:
    """Scalar direction quantizer (retained for unit test compatibility)."""
    length = float(np.hypot(delta[0], delta[1]))
    if length < dead:
        return 0
    dots = delta[0] * _MOVE_DIRS_8[:, 0] + delta[1] * _MOVE_DIRS_8[:, 1]
    return int(np.argmax(dots) + 1)


def _quantize_direction_batch(delta: np.ndarray, dead: float | np.ndarray = 8.0) -> np.ndarray:
    """Vectorized direction quantizer over arbitrary batch shapes `(..., 2)`."""
    norm = np.hypot(delta[..., 0], delta[..., 1])
    active = norm >= dead
    dots = delta[..., 0:1] * _MOVE_DIRS_8[:, 0] + delta[..., 1:2] * _MOVE_DIRS_8[:, 1]
    best_dir = np.argmax(dots, axis=-1) + 1
    return np.where(active, best_dir, 0).astype(np.int64)


def _tactical_actions(sim: BatchSim, team: int, half_w: float, half_h: float,
                      goal_x: float = 557.5, goal_hh: float = 80.0,
                      rng: np.random.Generator | None = None) -> np.ndarray:
    """Vectorized 3v3 rotational futsal policy (1st Man Presser/Striker, 2nd Man Wing/Support, 3rd Man Dynamic Cierre).

    Eliminates the permanent goal-box goalkeeper:
    - All 3 players (`slots 0, 1, 2`) dynamically rotate roles based on real-time intercept geometry and pitch depth.
    - When a presser is overtaken (`px > pred_bx + 16`), a teammate behind the ball steps in to press while the
      overtaken player rotates into wing support or third-man cover.
    - The 3rd man (`cierre`) advances outside the penalty box up to midfield/attack (`x in [-240, +95]`) during
      build-up and possession to participate in ball circulation and high interception, only dropping near the
      goal mouth when the ball is deep in the defensive danger zone (`bx < -350`).
    """
    N = sim.N
    orient = 1.0 if team == 0 else -1.0
    vec_orient = np.asarray([orient, 1.0], dtype=np.float64)
    ball_pos = sim.ball_pos * vec_orient
    ball_vel = getattr(sim, "ball_vel", np.zeros_like(ball_pos)) * vec_orient
    own_slots = slice(3 * team, 3 * team + 3)
    opp_slots = slice(3 * (1 - team), 3 * (1 - team) + 3)
    players = sim.player_pos[:, own_slots] * vec_orient
    player_vel = getattr(sim, "player_vel", np.zeros_like(sim.player_pos))[:, own_slots] * vec_orient
    rivals = sim.player_pos[:, opp_slots] * vec_orient
    kick_cancel = getattr(sim, "kick_cancel", np.zeros((N, 6), dtype=bool))[:, own_slots]
    kickoff = getattr(sim, "kickoff", np.zeros(N, dtype=bool))
    kickoff_team = getattr(sim, "kickoff_team", np.zeros(N, dtype=np.int64))

    bx = ball_pos[:, 0]
    by = ball_pos[:, 1]
    bvx = ball_vel[:, 0]
    bvy = ball_vel[:, 1]
    pred_bx = np.clip(bx + 5.0 * bvx, -half_w + 12.0, half_w - 12.0)
    pred_by = np.clip(by + 5.0 * bvy, -half_h + 12.0, half_h - 12.0)

    px = players[:, :, 0]
    py = players[:, :, 1]
    pvx = player_vel[:, :, 0]
    pvy = player_vel[:, :, 1]
    distances = np.hypot(px - bx[:, None], py - by[:, None])

    # Dynamic role rotation across all 3 slots {0, 1, 2}
    pred_px = px + 4.0 * pvx
    pred_py = py + 4.0 * pvy
    intercept_dist = np.hypot(pred_px - pred_bx[:, None], pred_py - pred_by[:, None])
    ahead_penalty = np.where(px > pred_bx[:, None] + 16.0, 65.0, 0.0)
    in_mouth_guard = (
        (bx[:, None] < -half_w + 140.0)
        & (px <= bx[:, None] + 2.0)
        & (px == px.min(axis=1, keepdims=True))
    )
    press_cost = intercept_dist + ahead_penalty + np.where(in_mouth_guard, 120.0, 0.0)
    striker = np.argmin(press_cost, axis=1)

    rem_a = (striker + 1) % 3
    rem_b = (striker + 2) % 3
    rows = np.arange(N)
    xa, ya = pred_px[rows, rem_a], pred_py[rows, rem_a]
    xb, yb = pred_px[rows, rem_b], pred_py[rows, rem_b]
    # En defensa profunda (bx < -220) el más retrasado cubre como cierre; en medio/ataque (bx >= -220)
    # rotan dinámicamente cierre <-> ala según quién esté mejor perfilado hacia el carril débil de pase.
    open_wing_y = np.where(by >= 0.0, -105.0, 105.0)
    cover_x = np.clip(bx - 195.0, -260.0, 95.0)
    cover_y = np.clip(by * 0.65, -110.0, 110.0)
    sup_cand_x = np.clip(pred_bx + 15.0, -half_w + 180.0, half_w - 85.0)
    cost_a_cierre = np.hypot(xa - cover_x, ya - cover_y) + np.hypot(xb - sup_cand_x, yb - open_wing_y)
    cost_b_cierre = np.hypot(xb - cover_x, yb - cover_y) + np.hypot(xa - sup_cand_x, ya - open_wing_y)
    a_is_cierre = np.where(bx < -220.0, xa <= xb, cost_a_cierre <= cost_b_cierre)
    cierre = np.where(a_is_cierre, rem_a, rem_b)
    support = np.where(a_is_cierre, rem_b, rem_a)

    actions = np.zeros((N, 3), dtype=np.int64)

    # --- Role 1: Dynamic Cierre (3rd Man — Sweeper / Build-up Pivot, steps out of the box!) ---
    c_px = px[rows, cierre]
    c_py = py[rows, cierre]
    c_pvx = pvx[rows, cierre]
    c_pvy = pvy[rows, cierre]
    c_dist = distances[rows, cierre]
    c_cancel = kick_cancel[rows, cierre]

    in_goal_mouth = (bx < -half_w + 115.0) & (np.abs(by) <= goal_hh + 28.0) & (c_dist < 95.0) & (c_px <= bx + 2.0)
    clear_y = np.where(by < 0.0, -half_h * 0.75, half_h * 0.75)
    c_ax = 200.0 - bx
    c_ay = clear_y - by
    c_norm = np.hypot(c_ax, c_ay) + 1e-6
    clear_tx = bx + (c_ax / c_norm) * 10.0
    clear_ty = by + (c_ay / c_norm) * 10.0

    guard_x = np.where(
        bx < -half_w + 200.0,
        np.clip(bx - 55.0, -half_w + 32.0, -half_w + 105.0),
        np.where(
            bx < -80.0,
            np.clip(bx - 140.0, -395.0, -230.0),
            np.clip(bx - 210.0, -240.0, 95.0),
        ),
    )
    fast_shot_back = (bvx < -0.25) & (bx > guard_x)
    ticks_to_line = np.clip((bx - guard_x) / np.maximum(-bvx, 0.3), 0.0, 45.0)
    proj_y = np.where(fast_shot_back, by + bvy * ticks_to_line, by * np.where(bx < 80.0, 0.65, 0.80))
    y_limit = np.where(bx < -half_w + 200.0, goal_hh - 14.0, half_h - 38.0)
    guard_y = np.clip(proj_y, -y_limit, y_limit)

    c_tx = np.where(in_goal_mouth, clear_tx, guard_x)
    c_ty = np.where(in_goal_mouth, clear_ty, guard_y)
    c_lead = np.where(in_goal_mouth, 0.0, 2.5)
    c_delta = np.stack((c_tx - (c_px + c_lead * c_pvx), c_ty - (c_py + c_lead * c_pvy)), axis=-1)
    c_dead = np.where(in_goal_mouth, 2.5, 3.5)
    c_dir = _quantize_direction_batch(c_delta, c_dead)
    c_kick = (c_dist <= EFFECTIVE_KICK_DIST) & (c_px <= bx + 1.0) & (~c_cancel)
    actions[rows, cierre] = c_dir + np.where(c_kick, 9, 0)

    # --- Role 2: Dynamic Support (2nd Man — Wing Passing Lane & Secondary Press) ---
    u_px = px[rows, support]
    u_py = py[rows, support]
    u_pvx = pvx[rows, support]
    u_pvy = pvy[rows, support]
    u_dist = distances[rows, support]
    u_cancel = kick_cancel[rows, support]

    wing_sign = np.where(by > 20.0, -1.0, np.where(by < -20.0, 1.0, np.where(support == 1, -1.0, 1.0)))
    sup_tx = np.where(
        bx > -60.0,
        np.clip(pred_bx + 15.0, -half_w + 180.0, half_w - 85.0),
        np.clip(pred_bx * 0.65 - 65.0, -half_w + 175.0, 80.0),
    )
    sup_ty = np.where(
        bx > -60.0,
        np.clip(wing_sign * 105.0, -half_h + 35.0, half_h - 35.0),
        np.clip(wing_sign * 65.0, -half_h + 40.0, half_h - 40.0),
    )
    u_delta = np.stack((sup_tx - (u_px + 2.0 * u_pvx), sup_ty - (u_py + 2.0 * u_pvy)), axis=-1)
    u_dir = _quantize_direction_batch(u_delta, 5.5)
    u_kick = (u_dist <= EFFECTIVE_KICK_DIST) & (u_px < bx - 2.0) & (~u_cancel)
    actions[rows, support] = u_dir + np.where(u_kick, 9, 0)

    # --- Role 3: Dynamic Striker (1st Man — Presser / Ball Carrier / Passer / Finisher) ---
    s_px = px[rows, striker]
    s_py = py[rows, striker]
    s_pvx = pvx[rows, striker]
    s_pvy = pvy[rows, striker]
    s_dist = distances[rows, striker]
    s_cancel = kick_cancel[rows, striker]

    sup_px = u_px
    sup_py = u_py
    sup_to_riv = np.hypot(rivals[:, :, 0] - sup_px[:, None], rivals[:, :, 1] - sup_py[:, None]).min(axis=1)

    opp_gk_idx = np.argmax(rivals[:, :, 0], axis=1)
    opp_gk_y = rivals[rows, opp_gk_idx, 1]
    aim_y = np.where(opp_gk_y >= 0.0, -46.0, 46.0)
    aim_y = np.where(np.abs(by) > 135.0, np.sign(by) * 32.0, aim_y)
    if rng is not None and np.any(kickoff):
        # Diversidad de ángulo en el saque durante entrenamiento RL para impedir memorización de rutinas fijas
        ko_jitter = rng.uniform(-38.0, 38.0, size=N)
        aim_y = np.where(kickoff, aim_y + ko_jitter, aim_y)

    pass_to_sup = (
        (sup_px > bx + 65.0)
        & (np.abs(sup_py) < half_h - 28.0)
        & (sup_to_riv > 88.0)
        & (bx < half_w - 190.0)
    )
    deep_clear = (~pass_to_sup) & (bx < -half_w + 155.0)
    tx_aim = np.where(pass_to_sup, sup_px + 25.0, np.where(deep_clear, 160.0, goal_x))
    ty_aim = np.where(
        pass_to_sup,
        sup_py,
        np.where(deep_clear, np.where(by < 0.0, -half_h * 0.7, half_h * 0.7), aim_y),
    )

    ax = tx_aim - pred_bx
    ay = ty_aim - pred_by
    anorm = np.hypot(ax, ay) + 1e-6
    ux = ax / anorm
    uy = ay / anorm

    vx = bx - s_px
    vy = by - s_py
    vnorm = np.hypot(vx, vy) + 1e-6
    cos_align = (vx * ux + vy * uy) / vnorm

    riv_to_ball = np.hypot(rivals[:, :, 0] - bx[:, None], rivals[:, :, 1] - by[:, None]).min(axis=1)
    bspeed = np.hypot(bvx, bvy)
    pinned = ((riv_to_ball < 85.0) | (bspeed < 0.6)) & (((bx > half_w - 65.0) & (np.abs(by) > goal_hh + 25.0)) | (np.abs(by) > half_h - 28.0))
    ahead = (~pinned) & (s_px > bx + 4.0)
    approach = (~pinned) & (~ahead) & (cos_align < 0.58) & (s_dist > 26.0)

    bypass_y = np.where(s_py >= by, 32.0, -32.0)
    lead = np.clip(s_dist * 0.22, 8.0, 16.0)

    s_tx = np.where(
        pinned, bx - 16.0,
        np.where(ahead, pred_bx - 26.0,
                 np.where(approach, pred_bx - ux * lead, bx + ux * 18.0)),
    )
    s_ty = np.where(
        pinned, np.clip(by, -half_h + 18.0, half_h - 18.0),
        np.where(
            ahead, np.clip(pred_by + bypass_y, -half_h + 18.0, half_h - 18.0),
            np.where(
                approach, np.clip(pred_by - uy * lead, -half_h + 18.0, half_h - 18.0),
                np.clip(by + uy * 18.0, -half_h + 15.0, half_h - 15.0),
            ),
        ),
    )
    s_vel_lead = np.where(pinned, 0.0, 1.5)
    s_delta = np.stack((s_tx - (s_px + s_vel_lead * s_pvx), s_ty - (s_py + s_vel_lead * s_pvy)), axis=-1)
    s_dead = np.where(pinned | ahead | approach, 2.0, 1.0)
    s_dir = _quantize_direction_batch(s_delta, s_dead)
    s_kick = np.where(
        pinned,
        (s_dist <= EFFECTIVE_KICK_DIST) & (s_px < bx - 2.0) & (~s_cancel),
        (s_dist <= EFFECTIVE_KICK_DIST) & (s_px < bx + 1.0) & (cos_align >= 0.42) & (~s_cancel),
    )
    actions[rows, striker] = s_dir + np.where(s_kick, 9, 0)

    def_ko = kickoff & (kickoff_team != team)
    if np.any(def_ko):
        ko_rows = np.flatnonzero(def_ko)
        for role_arr, lane_y in ((striker, -48.0), (support, 48.0)):
            sl = role_arr[ko_rows]
            d_ko = np.stack((-95.0 - (px[ko_rows, sl] + 2.0 * pvx[ko_rows, sl]),
                             lane_y - (py[ko_rows, sl] + 2.0 * pvy[ko_rows, sl])), axis=-1)
            actions[ko_rows, sl] = _quantize_direction_batch(d_ko, 5.0)

    return actions


def _scripted_actions(sim: BatchSim, team: int, half_w: float, half_h: float,
                      rng: np.random.Generator | None = None) -> np.ndarray:
    """Vectorized baseline chase policy (100% parity with original `_scripted_actions`, ~25x faster)."""
    orient = 1.0 if team == 0 else -1.0
    vec_orient = np.asarray([orient, 1.0], dtype=np.float64)
    ball = sim.ball_pos * vec_orient
    players = sim.player_pos[:, 3 * team:3 * team + 3] * vec_orient
    distances = np.hypot(players[:, :, 0] - ball[:, None, 0], players[:, :, 1] - ball[:, None, 1])
    striker = distances[:, 1:].argmin(axis=1) + 1

    targets = np.empty_like(players)
    targets[:, 0, 0] = -half_w + 55.0
    targets[:, 0, 1] = np.clip(ball[:, 1] * 0.35, -0.55 * half_h, 0.55 * half_h)
    targets[:, 1, 0] = ball[:, 0] - 110.0
    targets[:, 1, 1] = ball[:, 1] - 65.0
    targets[:, 2, 0] = ball[:, 0] - 110.0
    targets[:, 2, 1] = ball[:, 1] + 65.0

    rows = np.arange(sim.N)
    targets[rows, striker, 0] = ball[:, 0] - 16.0
    targets[rows, striker, 1] = ball[:, 1]
    if rng is not None and np.any(getattr(sim, "kickoff", False)):
        ko = sim.kickoff
        targets[ko, striker[ko], 1] += rng.uniform(-14.0, 14.0, size= int(ko.sum()))

    delta = targets - players
    directions = _quantize_direction_batch(delta, dead=8.0)
    is_striker = (np.arange(3)[None, :] == striker[:, None])
    kick = is_striker & (distances <= 25.0) & (players[:, :, 0] < ball[:, None, 0] - 3.0)
    return directions + np.where(kick, 9, 0)


class TacticalPolicy(Policy):
    """Dynamic 3-man rotational futsal policy with high line, wing passing lanes, and kick_cancel management."""

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


def _player_states(sim: BatchSim) -> np.ndarray:
    states = np.empty((sim.N, 7, 4), dtype=np.float32)
    states[:, 0, :2] = sim.ball_pos
    states[:, 0, 2:] = sim.ball_vel
    states[:, 1:4, :2] = sim.player_pos[:, :3]
    states[:, 1:4, 2:] = sim.player_vel[:, :3]
    states[:, 4:7, :2] = sim.player_pos[:, 3:]
    states[:, 4:7, 2:] = sim.player_vel[:, 3:]
    return states


def play_batch(red: Policy, blue: Policy, *, matches: int, minutes: float, seed: int,
               capture_trajectory: bool = False) -> list[dict]:
    if matches < 1 or minutes <= 0:
        raise ValueError("matches/minutes inválidos")
    stadium = load_stadium(STADIUM)
    half_w = stadium.field_half_w
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
    team_metrics = np.zeros((matches, 2, 11), dtype=np.float64)
    # 0:kicks, 1:shots, 2:completedPasses, 3:turnovers, 4:touches,
    # 5:movementDecisions, 6:speed, 7:kickoffSafety,
    # 8:boxCampSamples, 9:ballAdvancedSamples, 10:roleRotations
    prev_last_man = np.full((matches, 2), -1, dtype=np.int64)
    tick_limit = int(round(minutes * 60 * 60))
    ticks_played = np.zeros(matches, dtype=np.int64)
    spread_sum = np.zeros((matches, 2), dtype=np.float64)
    spread_samples = np.zeros(matches, dtype=np.int64)
    decision = 0
    trajectory = [[] for _ in range(matches)] if capture_trajectory else None

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
        native_applied = np.empty_like(applied)
        native_applied[:, :3] = applied[:, :3]
        native_applied[:, 3:] = MIRROR_X_ACTION[applied[:, 3:]]

        team_metrics[live, 0, 5] += np.count_nonzero(applied[live, :3] % 9, axis=1)
        team_metrics[live, 1, 5] += np.count_nonzero(applied[live, 3:] % 9, axis=1)
        speed = np.hypot(sim.player_vel[:, :, 0], sim.player_vel[:, :, 1])
        team_metrics[live, 0, 6] += speed[live, :3].mean(axis=1)
        team_metrics[live, 1, 6] += speed[live, 3:].mean(axis=1)

        # Tactical metrics: box camping when ball is advanced & dynamic last-man rotations
        p_pos = sim.player_pos
        b_pos = sim.ball_pos
        for team_idx in range(2):
            orient = 1.0 if team_idx == 0 else -1.0
            team_x = p_pos[:, 3 * team_idx:3 * team_idx + 3, 0] * orient
            ball_x = b_pos[:, 0] * orient
            cur_last = np.argmin(team_x, axis=1)
            rotated = live & (~sim.kickoff) & (prev_last_man[:, team_idx] >= 0) & (cur_last != prev_last_man[:, team_idx])
            team_metrics[rotated, team_idx, 10] += 1.0
            prev_last_man[live, team_idx] = cur_last[live]
            adv = live & (~sim.kickoff) & (ball_x > -half_w + 220.0)
            if np.any(adv):
                team_metrics[adv, team_idx, 9] += 1.0
                camp = adv & (team_x.min(axis=1) < -half_w + 115.0)
                team_metrics[camp, team_idx, 8] += 1.0

        if decision % 10 == 0:
            for team_idx in range(2):
                side = p_pos[:, 3 * team_idx:3 * team_idx + 3]
                d01 = np.hypot(side[:, 0, 0] - side[:, 1, 0], side[:, 0, 1] - side[:, 1, 1])
                d02 = np.hypot(side[:, 0, 0] - side[:, 2, 0], side[:, 0, 1] - side[:, 2, 1])
                d12 = np.hypot(side[:, 1, 0] - side[:, 2, 0], side[:, 1, 1] - side[:, 2, 1])
                spread_sum[live, team_idx] += ((d01 + d02 + d12) / 3.0)[live]
            spread_samples[live] += 1

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
            ball_before_x = sim.ball_pos[:, 0].copy()
            sim.step(step_actions)
            if np.any(sim.kicked):
                step_actions = np.where(sim.kicked, step_actions % 9, step_actions)

            ticks_played[active] += 1

            # Process kicks & shots only on matches that registered a kick
            kicked_matches = np.flatnonzero(active & sim.kicked.any(axis=1))
            for match in kicked_matches:
                kickers = np.flatnonzero(sim.kicked[match])
                for player in kickers:
                    team_idx = int(player // 3)
                    team_metrics[match, team_idx, 0] += 1
                    orient = 1.0 if team_idx == 0 else -1.0
                    if (ball_before_x[match] * orient > 0.35 * stadium.goal_x
                            and sim.ball_vel[match, 0] * orient > 0.5):
                        team_metrics[match, team_idx, 1] += 1
                pending_passer[match] = int(kickers[-1])

            # Process touches, completed passes & turnovers only on matches with a touch
            touched_matches = np.flatnonzero(active & sim.touch.any(axis=1))
            for match in touched_matches:
                touched = np.flatnonzero(sim.touch[match])
                player = int(touched[-1])
                team_idx = player // 3
                team_metrics[match, team_idx, 4] += 1
                if owner[match] >= 0 and owner[match] // 3 != team_idx:
                    team_metrics[match, team_idx, 3] += 1
                    pending_passer[match] = -1
                elif (pending_passer[match] >= 0 and pending_passer[match] // 3 == team_idx
                      and pending_passer[match] != player):
                    team_metrics[match, team_idx, 2] += 1
                    pending_passer[match] = -1
                owner[match] = player

            # Vectorized goal and kickoff timeout handling
            scored_matches = np.flatnonzero(active & (sim.goal != 0))
            if len(scored_matches):
                for match in scored_matches:
                    g = int(sim.goal[match])
                    scoring_team = 0 if g > 0 else 1
                    conceding_team = 1 - scoring_team
                    goals[match, scoring_team] += 1
                    sim.reset_kickoff([match], kickoff_team=conceding_team)
                    kickoff_ticks[match] = 0
                    owner[match] = -1
                    pending_passer[match] = -1
                    previous[match] = 0
                    has_previous[match] = False
                    prev_last_man[match] = -1

            no_goal_active = active & (sim.goal == 0)
            ko_active = no_goal_active & sim.kickoff
            kickoff_ticks[ko_active] += 1
            kickoff_ticks[no_goal_active & (~sim.kickoff)] = 0
            expired_ko = np.flatnonzero(ko_active & (kickoff_ticks >= KICKOFF_SAFETY_TICKS))
            if len(expired_ko):
                for match in expired_ko:
                    sim.kickoff[match] = False
                    sim.mask[match, sim.first_player:] = sim.base_mask[sim.first_player:]
                    team_metrics[match, int(sim.kickoff_team[match]), 7] += 1
                    kickoff_ticks[match] = 0

            if trajectory is not None:
                for match in np.flatnonzero(active):
                    trajectory[match].append({
                        "ball": sim.ball_pos[match].tolist(),
                        "players": sim.player_pos[match].tolist(),
                        "goals": goals[match].tolist(),
                        "goal": int(sim.goal[match]),
                    })
        decision += 1

    result = []
    for match in range(matches):
        game = {
            "goals": goals[match].tolist(),
            "ticks": int(ticks_played[match]),
            "safetyKickoffs": int(team_metrics[match, :, 7].sum()),
            "teams": [
                {
                    "kicks": int(team_metrics[match, team_idx, 0]),
                    "shots": int(team_metrics[match, team_idx, 1]),
                    "completedPasses": int(team_metrics[match, team_idx, 2]),
                    "turnovers": int(team_metrics[match, team_idx, 3]),
                    "touches": int(team_metrics[match, team_idx, 4]),
                    "movementDecisions": int(team_metrics[match, team_idx, 5]),
                    "meanPlayerSpeed": float(team_metrics[match, team_idx, 6] / max(1, decision)),
                    "safetyKickoffs": int(team_metrics[match, team_idx, 7]),
                    "meanTeamSpread": float(spread_sum[match, team_idx] / max(1, spread_samples[match])),
                    "ownBoxCampingFrac": float(
                        team_metrics[match, team_idx, 8] / max(1.0, team_metrics[match, team_idx, 9])
                    ),
                    "roleRotations": int(team_metrics[match, team_idx, 10]),
                }
                for team_idx in range(2)
            ],
        }
        if trajectory is not None:
            game["trajectory"] = trajectory[match]
        result.append(game)
    return result


def _aggregate(games: list[dict], policy_role: int) -> dict:
    wins = draws = losses = goals_for = goals_against = 0
    stat_names = ("kicks", "shots", "completedPasses", "turnovers", "touches",
                  "movementDecisions", "meanPlayerSpeed", "meanTeamSpread", "safetyKickoffs")
    optional_names = ("ownBoxCampingFrac", "roleRotations")
    stats = {name: 0.0 for name in stat_names}
    opt_stats = {name: 0.0 for name in optional_names}
    has_opt = {name: False for name in optional_names}
    for game in games:
        own_team = game["firstPolicyTeam"] if policy_role == 0 else 1 - game["firstPolicyTeam"]
        other_team = 1 - own_team
        own, other = game["goals"][own_team], game["goals"][other_team]
        goals_for += own
        goals_against += other
        wins += own > other
        losses += own < other
        draws += own == other
        team_dict = game["teams"][own_team]
        for name in stat_names:
            stats[name] += team_dict[name]
        for name in optional_names:
            if name in team_dict:
                opt_stats[name] += team_dict[name]
                has_opt[name] = True
    count = max(1, len(games))
    out = {
        "matches": len(games), "wins": int(wins), "draws": int(draws), "losses": int(losses),
        "winRate": float(wins / count), "goalFor": int(goals_for), "goalAgainst": int(goals_against),
        "goalDifferencePerMatch": float((goals_for - goals_against) / count),
        **{f"{name}PerMatch": float(value / count) for name, value in stats.items()},
    }
    for name in optional_names:
        if has_opt[name]:
            out[f"{name}PerMatch"] = float(opt_stats[name] / count)
    return out


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
    set_physics_threads(threads)
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
        "version": "x3-closed-loop-eval-5",
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