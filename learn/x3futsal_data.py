"""Prepare replay-grounded 3v3 Futsal x3 behavior-cloning data and fast vectorized tactical observations.

  python -m learn.x3futsal_data
  python -m learn.x3futsal_data --cache data/x3futsal_jsonl --out data/x3futsal_bc_active
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from sim.stadium import load_stadium
from tools.x3futsal_audit import DEFAULT_OUT as DEFAULT_CACHE, ROOT, STADIUM, _span_for, make_splits
from tools.x3futsal_physics import _read_replay, transition_is_valid

MOVE_DX = (0, 0, 1, 1, 1, 0, -1, -1, -1)
MOVE_DY = (0, -1, -1, 0, 1, 1, 1, 0, -1)
MOVE_BY_DELTA = dict(zip(zip(MOVE_DX, MOVE_DY), range(9)))
Y_FLIP_MOVE = np.asarray([0, 5, 4, 3, 2, 1, 8, 7, 6], dtype=np.int64)
X_FLIP_MOVE = np.asarray([0, 1, 8, 7, 6, 5, 4, 3, 2], dtype=np.int64)
MIRROR_X_ACTION = np.concatenate((X_FLIP_MOVE, X_FLIP_MOVE + 9)).astype(np.int64)

N_PLAYERS = 6
N_ACTIONS = 18
DECISION_STRIDE = 3
PLAYER_RADIUS = 15.0
BALL_RADIUS = 6.25
KICK_REACH = 4.0
CONTACT_DIST = PLAYER_RADIUS + BALL_RADIUS
EFFECTIVE_KICK_DIST = CONTACT_DIST + KICK_REACH
NEAR_BALL_DIST_PX = 140.0
MAX_HELD_KICK_OUTSIDE_REACH = 9

DATA_VERSION = "x3-futsal-bc-v3-active"
TACTICAL_DATA_VERSION = "x3-futsal-bc-v4-tactical"
DEFAULT_OUT = ROOT / "data" / "x3futsal_bc_active"
AFK_OUTSIDE_MARGIN = 25.0
AFK_STATIONARY_FRAMES = 600
AFK_SPEED_THRESHOLD = 0.08
# Umbral para detectar jugadores anclados pasivamente dentro del área propia mientras la pelota está lejos
BOX_CAMP_STATIONARY_FRAMES = 90
BOX_CAMP_DEPTH_MARGIN = 115.0
BOX_CAMP_BALL_ADVANCE_MARGIN = 210.0

TACTICAL_SELF_DIM = 28
TACTICAL_ENTITY_DIM = 10

_MATE_PAIRS = np.asarray([[1, 2], [2, 0], [0, 1]], dtype=np.int64)
_MATE_A = np.asarray([1, 2, 0], dtype=np.int64)
_MATE_B = np.asarray([2, 0, 1], dtype=np.int64)

FILTER_CONTRACT = {
    "targetStadiumSpans": True,
    "state": "0 (kickoff/restart) or 1 (active gameplay)",
    "roster": "stable 3v3 across consecutive frames",
    "physics": "standard player properties; property-event frames excluded",
    "degeneratePositions": "coincident players and neighboring ticks excluded",
    "transition": "consecutive frame t -> t+1; label is input at t (with 3-frame kick window and stride-3 history support)",
    "kickLabel": "kick key bit (16) at t, filtered against stale held-kick states outside reach",
    "afk": "transition excluded when any rostered player is outside the field envelope or stationary with no directional input for 600 consecutive frames",
}
FEATURE_CONTRACT = {
    "version": "x3-futsal-observation-v2",
    "attackDirection": "team 1 and team 2 are mirrored so the controlled team attacks +x",
    "observation": {
        "self": ["x", "y", "vx", "vy", "ball_x", "ball_y", "ball_vx", "ball_vy",
                 "ball_relative_x", "ball_relative_y", "score_difference", "elapsed_seconds"],
        "entities": ["relative_x", "relative_y", "vx", "vy", "ball_relative_x", "ball_relative_y"],
        "entitySets": {"teammates": 2, "opponents": 3},
        "normalization": {
            "position": ["field_half_width", "field_half_height"],
            "velocity": 6.0,
            "score_difference": 5.0,
            "elapsed_seconds": 300.0,
            "elapsed_clipped": [0.0, 1.0],
        },
    },
    "history": {
        "previousAction": "one-hot 18-class action at t-1 (or t-3 decision stride) for the controlled player",
        "validityMask": "present only when the previous sample is contiguous in the same segment",
    },
    "action": {
        "classes": 18,
        "directions": ["stay", "up", "up_right", "right", "down_right", "down",
                       "down_left", "left", "up_left"],
        "kick_offset": 9,
        "coordinate_convention": "x right, y down; team 2 direction is mirrored horizontally",
    },
}
TACTICAL_FEATURE_CONTRACT = {
    "version": "x3-futsal-observation-v3",
    "attackDirection": "team 1 and team 2 are mirrored so the controlled team attacks +x",
    "selfDim": TACTICAL_SELF_DIM,
    "entityDim": TACTICAL_ENTITY_DIM,
    "selfFeatures": [
        "x", "y", "vx", "vy",
        "ball_x", "ball_y", "ball_vx", "ball_vy",
        "ball_dx", "ball_dy", "ball_dist", "ball_gap", "ball_in_reach",
        "ball_rvx", "ball_rvy", "ball_pred6_dx", "ball_pred6_dy",
        "opp_goal_dx", "opp_goal_dy", "own_goal_dx", "own_goal_dy",
        "ball_opp_goal_dx", "ball_opp_goal_dy", "shot_alignment",
        "is_closest_to_ball", "is_last_man", "race_margin_rival", "kick_armed",
    ],
    "entityFeatures": [
        "relative_x", "relative_y", "vx", "vy",
        "ball_relative_x", "ball_relative_y", "ball_dist", "ball_gap",
        "ball_in_reach", "pass_alignment",
    ],
}


def action_from_input(value: int, team: int) -> int:
    """Map HaxBall key bits to 9 directions x kick, mirrored to attack toward +x."""
    value = int(value)
    if not 0 <= value < 32:
        raise ValueError(f"input de HaxBall fuera de rango: {value}")
    dx = ((value >> 3) & 1) - ((value >> 2) & 1)
    dy = ((value >> 1) & 1) - (value & 1)
    if team == 2:
        dx = -dx
    elif team != 1:
        raise ValueError(f"equipo de jugador inválido: {team}")
    return MOVE_BY_DELTA[(dx, dy)] + (9 if value & 16 else 0)


def flip_action_y(actions: np.ndarray) -> np.ndarray:
    """Reflect action indices across the horizontal axis (y -> -y) while preserving kick bit."""
    actions = np.asarray(actions, dtype=np.int64)
    return Y_FLIP_MOVE[actions % 9] + 9 * (actions // 9)


def mirror_states_y(states: np.ndarray) -> np.ndarray:
    """Reflect raw (B, 7, 4) states across y -> -y, vy -> -vy."""
    mirrored = np.asarray(states, dtype=np.float32).copy()
    mirrored[:, :, 1] *= -1.0
    mirrored[:, :, 3] *= -1.0
    return mirrored


def featurize(states: np.ndarray, slots: np.ndarray, score_diff: np.ndarray, elapsed: np.ndarray,
              field_half_w: float, field_half_h: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    states = np.asarray(states, dtype=np.float32).copy()
    slots = np.asarray(slots, dtype=np.int64)
    orient = np.where(slots < 3, 1.0, -1.0).astype(np.float32)
    states[:, :, 0] *= orient[:, None]
    states[:, :, 2] *= orient[:, None]
    scale = np.asarray([max(field_half_w, 1.0), max(field_half_h, 1.0), 6.0, 6.0], dtype=np.float32)
    states /= scale

    batch = np.arange(len(states))
    players = states[:, 1:, :]
    own = players[batch, slots]
    ball = states[:, 0, :]
    local_slot = slots % 3
    mates_idx = (slots // 3)[:, None] * 3 + (local_slot[:, None] + np.asarray([1, 2])) % 3
    rival_slots = np.arange(3, dtype=np.int64)[None, :]
    rivals_idx = np.where((slots < 3)[:, None], rival_slots + 3, rival_slots)
    mates = players[batch[:, None], mates_idx]
    rivals = players[batch[:, None], rivals_idx]

    score = np.asarray(score_diff, dtype=np.float32) * orient / 5.0
    clock = np.clip(np.asarray(elapsed, dtype=np.float32) / 300.0, 0.0, 1.0)
    own_features = np.concatenate((own, ball, ball[:, :2] - own[:, :2],
                                   score[:, None], clock[:, None]), axis=1)
    mate_features = np.concatenate((mates[:, :, :2] - own[:, None, :2], mates[:, :, 2:],
                                    mates[:, :, :2] - ball[:, None, :2]), axis=2)
    rival_features = np.concatenate((rivals[:, :, :2] - own[:, None, :2], rivals[:, :, 2:],
                                     rivals[:, :, :2] - ball[:, None, :2]), axis=2)
    return own_features, mate_features, rival_features


def featurize_tactical(
    states: np.ndarray,
    slots: np.ndarray,
    field_half_w: float = 550.0,
    field_half_h: float = 240.0,
    goal_x: float = 557.5,
    kick_cancel: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build contact-aware, goal-oriented, and role-aware 3v3 observations (28 self, 10 entity).

    Strictly mirror-invariant across teams (both attack +x) and equivariant under y -> -y.
    """
    raw = np.asarray(states, dtype=np.float32).copy()
    slots = np.asarray(slots, dtype=np.int64)
    batch_size = len(raw)
    orient = np.where(slots < 3, 1.0, -1.0).astype(np.float32)
    raw[:, :, 0] *= orient[:, None]
    raw[:, :, 2] *= orient[:, None]

    w = max(float(field_half_w), 1.0)
    h = max(float(field_half_h), 1.0)
    gx = max(float(goal_x), w)
    sr = 600.0
    inv_w = np.float32(1.0 / w)
    inv_h = np.float32(1.0 / h)
    inv_v = np.float32(1.0 / 6.0)
    inv_sr = np.float32(1.0 / sr)

    batch = np.arange(batch_size)
    players_px = raw[:, 1:, :]
    ball_px = raw[:, 0, :]
    own_px = players_px[batch, slots]

    local_slot = slots % 3
    mates_idx = (slots // 3)[:, None] * 3 + (local_slot[:, None] + np.asarray([1, 2])) % 3
    rival_slots = np.arange(3, dtype=np.int64)[None, :]
    rivals_idx = np.where((slots < 3)[:, None], rival_slots + 3, rival_slots)
    mates_px = players_px[batch[:, None], mates_idx]
    rivals_px = players_px[batch[:, None], rivals_idx]

    ball_delta_px = ball_px[:, :2] - own_px[:, :2]
    ball_dist_px = np.hypot(ball_delta_px[:, 0], ball_delta_px[:, 1])
    ball_gap = np.clip((ball_dist_px - CONTACT_DIST) * np.float32(0.125), np.float32(-1.0), np.float32(4.0))
    ball_in_reach = (ball_dist_px <= EFFECTIVE_KICK_DIST).astype(np.float32)

    mate_ball_delta_px = ball_px[:, None, :2] - mates_px[:, :, :2]
    mate_ball_dist_px = np.hypot(mate_ball_delta_px[:, :, 0], mate_ball_delta_px[:, :, 1])
    rival_ball_delta_px = ball_px[:, None, :2] - rivals_px[:, :, :2]
    rival_ball_dist_px = np.hypot(rival_ball_delta_px[:, :, 0], rival_ball_delta_px[:, :, 1])

    pred_ball_px = ball_px[:, :2] + np.float32(6.0) * ball_px[:, 2:]
    pred_own_px = own_px[:, :2] + np.float32(6.0) * own_px[:, 2:]
    pred_delta_px = pred_ball_px - pred_own_px
    pred_own_dist = np.hypot(pred_delta_px[:, 0], pred_delta_px[:, 1])
    pred_mates_px = mates_px[:, :, :2] + np.float32(6.0) * mates_px[:, :, 2:]
    pred_mate_dist = np.hypot(pred_ball_px[:, None, 0] - pred_mates_px[:, :, 0],
                              pred_ball_px[:, None, 1] - pred_mates_px[:, :, 1])

    is_closest_to_ball = (pred_own_dist <= pred_mate_dist.min(axis=1) + np.float32(1e-3)).astype(np.float32)
    is_last_man = (own_px[:, 0] <= mates_px[:, :, 0].min(axis=1) + np.float32(1e-3)).astype(np.float32)
    min_rival_dist = rival_ball_dist_px.min(axis=1)
    race_margin_rival = np.clip((min_rival_dist - ball_dist_px) * np.float32(1.0 / 200.0), np.float32(-1.0), np.float32(1.0))

    opp_goal_dx = (np.float32(gx) - own_px[:, 0]) * inv_w
    opp_goal_dy = -own_px[:, 1] * inv_h
    own_goal_dx = (np.float32(-gx) - own_px[:, 0]) * inv_w
    own_goal_dy = -own_px[:, 1] * inv_h
    ball_opp_goal_x = np.float32(gx) - ball_px[:, 0]
    ball_opp_goal_y = -ball_px[:, 1]
    ball_opp_goal_norm = np.hypot(ball_opp_goal_x, ball_opp_goal_y) + np.float32(1e-6)
    shot_alignment = (
        (ball_delta_px[:, 0] * ball_opp_goal_x + ball_delta_px[:, 1] * ball_opp_goal_y)
        / ((ball_dist_px + np.float32(1e-6)) * ball_opp_goal_norm)
    )

    if kick_cancel is None:
        kick_armed = np.ones(batch_size, dtype=np.float32)
    else:
        kick_armed = (~np.asarray(kick_cancel, dtype=bool)).astype(np.float32)

    own_features = np.empty((batch_size, TACTICAL_SELF_DIM), dtype=np.float32)
    own_features[:, 0] = own_px[:, 0] * inv_w
    own_features[:, 1] = own_px[:, 1] * inv_h
    own_features[:, 2] = own_px[:, 2] * inv_v
    own_features[:, 3] = own_px[:, 3] * inv_v
    own_features[:, 4] = ball_px[:, 0] * inv_w
    own_features[:, 5] = ball_px[:, 1] * inv_h
    own_features[:, 6] = ball_px[:, 2] * inv_v
    own_features[:, 7] = ball_px[:, 3] * inv_v
    own_features[:, 8] = ball_delta_px[:, 0] * inv_w
    own_features[:, 9] = ball_delta_px[:, 1] * inv_h
    own_features[:, 10] = ball_dist_px * inv_sr
    own_features[:, 11] = ball_gap
    own_features[:, 12] = ball_in_reach
    own_features[:, 13] = (ball_px[:, 2] - own_px[:, 2]) * inv_v
    own_features[:, 14] = (ball_px[:, 3] - own_px[:, 3]) * inv_v
    own_features[:, 15] = pred_delta_px[:, 0] * inv_sr
    own_features[:, 16] = pred_delta_px[:, 1] * inv_sr
    own_features[:, 17] = opp_goal_dx
    own_features[:, 18] = opp_goal_dy
    own_features[:, 19] = own_goal_dx
    own_features[:, 20] = own_goal_dy
    own_features[:, 21] = ball_opp_goal_x * inv_w
    own_features[:, 22] = ball_opp_goal_y * inv_h
    own_features[:, 23] = shot_alignment
    own_features[:, 24] = is_closest_to_ball
    own_features[:, 25] = is_last_man
    own_features[:, 26] = race_margin_rival
    own_features[:, 27] = kick_armed

    def _entity_block(ent_px: np.ndarray, ent_ball_dist: np.ndarray) -> np.ndarray:
        out = np.empty((batch_size, ent_px.shape[1], TACTICAL_ENTITY_DIM), dtype=np.float32)
        ball_to_ent_x = ent_px[:, :, 0] - ball_px[:, None, 0]
        ball_to_ent_y = ent_px[:, :, 1] - ball_px[:, None, 1]
        ball_to_ent_norm = np.hypot(ball_to_ent_x, ball_to_ent_y) + np.float32(1e-6)
        out[:, :, 0] = (ent_px[:, :, 0] - own_px[:, None, 0]) * inv_w
        out[:, :, 1] = (ent_px[:, :, 1] - own_px[:, None, 1]) * inv_h
        out[:, :, 2] = ent_px[:, :, 2] * inv_v
        out[:, :, 3] = ent_px[:, :, 3] * inv_v
        out[:, :, 4] = ball_to_ent_x * inv_w
        out[:, :, 5] = ball_to_ent_y * inv_h
        out[:, :, 6] = ent_ball_dist * inv_sr
        out[:, :, 7] = np.clip((ent_ball_dist - CONTACT_DIST) * np.float32(0.125), np.float32(-1.0), np.float32(4.0))
        out[:, :, 8] = (ent_ball_dist <= EFFECTIVE_KICK_DIST).astype(np.float32)
        out[:, :, 9] = (
            (ball_delta_px[:, None, 0] * ball_to_ent_x + ball_delta_px[:, None, 1] * ball_to_ent_y)
            / ((ball_dist_px[:, None] + np.float32(1e-6)) * ball_to_ent_norm)
        )
        return out

    mate_features = _entity_block(mates_px, mate_ball_dist_px)
    rival_features = _entity_block(rivals_px, rival_ball_dist_px)
    return own_features, mate_features, rival_features


def featurize_tactical_team(
    sim,
    team: int,
    field_half_w: float = 550.0,
    field_half_h: float = 240.0,
    goal_x: float = 557.5,
    out_own: np.ndarray | None = None,
    out_mates: np.ndarray | None = None,
    out_rivals: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute tactical v3 observations for all 3 players of `team` across `sim.N` matches in one pass.

    Avoids constructing `(N, 7, 4)`, repeating it 3x, and recomputing distances 3x redundantly.
    Guaranteed bit-compatible (`atol <= 1e-6`) with `featurize_tactical`.
    """
    N = sim.N
    B = N * 3
    w = max(float(field_half_w), 1.0)
    h = max(float(field_half_h), 1.0)
    gx = max(float(goal_x), w)
    sr = 600.0
    inv_w = np.float32(1.0 / w)
    inv_h = np.float32(1.0 / h)
    inv_v = np.float32(1.0 / 6.0)
    inv_sr = np.float32(1.0 / sr)
    orient = np.float32(1.0 if team == 0 else -1.0)

    fp = sim.first_player
    own_slice = slice(fp + 3 * team, fp + 3 * team + 3)
    riv_slice = slice(fp + 3 * (1 - team), fp + 3 * (1 - team) + 3)

    b_x = (sim.pos[:, 0, 0] * orient).astype(np.float32)
    b_y = sim.pos[:, 0, 1].astype(np.float32)
    b_vx = (sim.vel[:, 0, 0] * orient).astype(np.float32)
    b_vy = sim.vel[:, 0, 1].astype(np.float32)

    t_x = (sim.pos[:, own_slice, 0] * orient).astype(np.float32)
    t_y = sim.pos[:, own_slice, 1].astype(np.float32)
    t_vx = (sim.vel[:, own_slice, 0] * orient).astype(np.float32)
    t_vy = sim.vel[:, own_slice, 1].astype(np.float32)

    r_x = (sim.pos[:, riv_slice, 0] * orient).astype(np.float32)
    r_y = sim.pos[:, riv_slice, 1].astype(np.float32)
    r_vx = (sim.vel[:, riv_slice, 0] * orient).astype(np.float32)
    r_vy = sim.vel[:, riv_slice, 1].astype(np.float32)

    t_bdx = b_x[:, None] - t_x
    t_bdy = b_y[:, None] - t_y
    t_bdist = np.hypot(t_bdx, t_bdy)
    t_bgap = np.clip((t_bdist - CONTACT_DIST) * np.float32(0.125), np.float32(-1.0), np.float32(4.0))
    t_breach = (t_bdist <= EFFECTIVE_KICK_DIST).astype(np.float32)

    r_bdx = b_x[:, None] - r_x
    r_bdy = b_y[:, None] - r_y
    r_bdist = np.hypot(r_bdx, r_bdy)
    r_bgap = np.clip((r_bdist - CONTACT_DIST) * np.float32(0.125), np.float32(-1.0), np.float32(4.0))
    r_breach = (r_bdist <= EFFECTIVE_KICK_DIST).astype(np.float32)
    min_r_bdist = r_bdist.min(axis=1)

    pred_bx = b_x + np.float32(6.0) * b_vx
    pred_by = b_y + np.float32(6.0) * b_vy
    pred_tx = t_x + np.float32(6.0) * t_vx
    pred_ty = t_y + np.float32(6.0) * t_vy
    pred_dx = pred_bx[:, None] - pred_tx
    pred_dy = pred_by[:, None] - pred_ty
    pred_dist = np.hypot(pred_dx, pred_dy)

    min_mate_pred = np.minimum(pred_dist[:, _MATE_A], pred_dist[:, _MATE_B])
    is_closest = (pred_dist <= min_mate_pred + np.float32(1e-3)).astype(np.float32)
    min_mate_x = np.minimum(t_x[:, _MATE_A], t_x[:, _MATE_B])
    is_last_man = (t_x <= min_mate_x + np.float32(1e-3)).astype(np.float32)
    race_margin = np.clip((min_r_bdist[:, None] - t_bdist) * np.float32(1.0 / 200.0), np.float32(-1.0), np.float32(1.0))

    ball_opp_gx = np.float32(gx) - b_x
    ball_opp_gy = -b_y
    ball_opp_gnorm = np.hypot(ball_opp_gx, ball_opp_gy) + np.float32(1e-6)
    shot_align = (
        (t_bdx * ball_opp_gx[:, None] + t_bdy * ball_opp_gy[:, None])
        / ((t_bdist + np.float32(1e-6)) * ball_opp_gnorm[:, None])
    )
    kc = getattr(sim, "kick_cancel", None)
    if kc is None:
        kick_armed = np.ones((N, 3), dtype=np.float32)
    else:
        kick_armed = (~kc[:, 3 * team:3 * team + 3]).astype(np.float32)

    if out_own is None:
        out_own = np.empty((B, TACTICAL_SELF_DIM), dtype=np.float32)
    own_3d = out_own.reshape(N, 3, TACTICAL_SELF_DIM)
    own_3d[:, :, 0] = t_x * inv_w
    own_3d[:, :, 1] = t_y * inv_h
    own_3d[:, :, 2] = t_vx * inv_v
    own_3d[:, :, 3] = t_vy * inv_v
    own_3d[:, :, 4] = (b_x * inv_w)[:, None]
    own_3d[:, :, 5] = (b_y * inv_h)[:, None]
    own_3d[:, :, 6] = (b_vx * inv_v)[:, None]
    own_3d[:, :, 7] = (b_vy * inv_v)[:, None]
    own_3d[:, :, 8] = t_bdx * inv_w
    own_3d[:, :, 9] = t_bdy * inv_h
    own_3d[:, :, 10] = t_bdist * inv_sr
    own_3d[:, :, 11] = t_bgap
    own_3d[:, :, 12] = t_breach
    own_3d[:, :, 13] = (b_vx[:, None] - t_vx) * inv_v
    own_3d[:, :, 14] = (b_vy[:, None] - t_vy) * inv_v
    own_3d[:, :, 15] = pred_dx * inv_sr
    own_3d[:, :, 16] = pred_dy * inv_sr
    own_3d[:, :, 17] = (np.float32(gx) - t_x) * inv_w
    own_3d[:, :, 18] = -t_y * inv_h
    own_3d[:, :, 19] = (np.float32(-gx) - t_x) * inv_w
    own_3d[:, :, 20] = -t_y * inv_h
    own_3d[:, :, 21] = (ball_opp_gx * inv_w)[:, None]
    own_3d[:, :, 22] = (ball_opp_gy * inv_h)[:, None]
    own_3d[:, :, 23] = shot_align
    own_3d[:, :, 24] = is_closest
    own_3d[:, :, 25] = is_last_man
    own_3d[:, :, 26] = race_margin
    own_3d[:, :, 27] = kick_armed

    if out_mates is None:
        out_mates = np.empty((B, 2, TACTICAL_ENTITY_DIM), dtype=np.float32)
    mates_4d = out_mates.reshape(N, 3, 2, TACTICAL_ENTITY_DIM)
    mx = t_x[:, _MATE_PAIRS]
    my = t_y[:, _MATE_PAIRS]
    mvx = t_vx[:, _MATE_PAIRS]
    mvy = t_vy[:, _MATE_PAIRS]
    mbdist = t_bdist[:, _MATE_PAIRS]
    mbgap = t_bgap[:, _MATE_PAIRS]
    mbreach = t_breach[:, _MATE_PAIRS]
    m_rel_bx = mx - b_x[:, None, None]
    m_rel_by = my - b_y[:, None, None]
    m_bnorm = np.hypot(m_rel_bx, m_rel_by) + np.float32(1e-6)
    m_pass_align = (
        (t_bdx[:, :, None] * m_rel_bx + t_bdy[:, :, None] * m_rel_by)
        / ((t_bdist[:, :, None] + np.float32(1e-6)) * m_bnorm)
    )
    mates_4d[:, :, :, 0] = (mx - t_x[:, :, None]) * inv_w
    mates_4d[:, :, :, 1] = (my - t_y[:, :, None]) * inv_h
    mates_4d[:, :, :, 2] = mvx * inv_v
    mates_4d[:, :, :, 3] = mvy * inv_v
    mates_4d[:, :, :, 4] = m_rel_bx * inv_w
    mates_4d[:, :, :, 5] = m_rel_by * inv_h
    mates_4d[:, :, :, 6] = mbdist * inv_sr
    mates_4d[:, :, :, 7] = mbgap
    mates_4d[:, :, :, 8] = mbreach
    mates_4d[:, :, :, 9] = m_pass_align

    if out_rivals is None:
        out_rivals = np.empty((B, 3, TACTICAL_ENTITY_DIM), dtype=np.float32)
    rivals_4d = out_rivals.reshape(N, 3, 3, TACTICAL_ENTITY_DIM)
    r_rel_bx = -r_bdx
    r_rel_by = -r_bdy
    r_bnorm = np.hypot(r_rel_bx, r_rel_by) + np.float32(1e-6)
    r_pass_align = (
        (t_bdx[:, :, None] * r_rel_bx[:, None, :] + t_bdy[:, :, None] * r_rel_by[:, None, :])
        / ((t_bdist[:, :, None] + np.float32(1e-6)) * r_bnorm[:, None, :])
    )
    rivals_4d[:, :, :, 0] = (r_x[:, None, :] - t_x[:, :, None]) * inv_w
    rivals_4d[:, :, :, 1] = (r_y[:, None, :] - t_y[:, :, None]) * inv_h
    rivals_4d[:, :, :, 2] = (r_vx * inv_v)[:, None, :]
    rivals_4d[:, :, :, 3] = (r_vy * inv_v)[:, None, :]
    rivals_4d[:, :, :, 4] = (r_rel_bx * inv_w)[:, None, :]
    rivals_4d[:, :, :, 5] = (r_rel_by * inv_h)[:, None, :]
    rivals_4d[:, :, :, 6] = (r_bdist * inv_sr)[:, None, :]
    rivals_4d[:, :, :, 7] = r_bgap[:, None, :]
    rivals_4d[:, :, :, 8] = r_breach[:, None, :]
    rivals_4d[:, :, :, 9] = r_pass_align

    return out_own, out_mates, out_rivals


class X3Data:
    """Split shards in memory; supports uniform, near-ball, and anti-box-camping rotational sampling plus Y-mirroring."""

    def __init__(self, dataset: Path, split: str, *, tactical: bool = False):
        manifest_path = dataset / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("dataVersion") not in (DATA_VERSION, TACTICAL_DATA_VERSION):
            raise ValueError(f"versión de dataset incompatible: {manifest.get('dataVersion')}")
        if manifest.get("buildStatus") != "complete":
            raise ValueError(f"el dataset aún no está completo: {manifest_path}")
        self.tactical = bool(tactical)
        self.shards = []
        self.near_ball_pairs = []
        self.active_rotational_pairs = []
        stadium = load_stadium(STADIUM)
        self.field_half_w = float(stadium.field_half_w)
        self.field_half_h = float(stadium.field_half_h)
        self.goal_x = float(stadium.goal_x)
        offset = 0
        orient_vec = np.asarray([1.0, 1.0, 1.0, -1.0, -1.0, -1.0], dtype=np.float32)
        for row in manifest["shards"]:
            if row["split"] != split:
                continue
            with np.load(dataset / row["file"], allow_pickle=False) as archive:
                shard = {key: archive[key] for key in archive.files}
            if len(shard["states"]) != row["ticks"] or shard["states"].shape[1:] != (7, 4):
                raise ValueError(f"shard corrupto o incompatible: {row['file']}")
            if str(shard.get("dataVersion", "")) not in (DATA_VERSION, TACTICAL_DATA_VERSION) \
                    or str(shard.get("sourceSha256", "")) != row["sourceSha256"]:
                raise ValueError(f"metadatos de shard incompatibles: {row['file']}")
            self.shards.append(shard)
            states_f32 = shard["states"].astype(np.float32)
            dx = states_f32[:, 1:, 0] - states_f32[:, 0:1, 0]
            dy = states_f32[:, 1:, 1] - states_f32[:, 0:1, 1]
            near_rows, near_slots = np.nonzero((dx * dx + dy * dy) <= NEAR_BALL_DIST_PX ** 2)
            if len(near_rows):
                self.near_ball_pairs.append(
                    np.column_stack((near_rows + offset, near_slots)).astype(np.int64)
                )
            # Exclude passive box-camping player-ticks (player parked in own box with action==0 while ball is advanced)
            px_oriented = states_f32[:, 1:, 0] * orient_vec[None, :]
            bx_oriented = states_f32[:, 0:1, 0] * orient_vec[None, :]
            acts = shard["actions"]
            passive_box_camp = (
                (px_oriented < -self.field_half_w + BOX_CAMP_DEPTH_MARGIN)
                & (bx_oriented > -self.field_half_w + BOX_CAMP_BALL_ADVANCE_MARGIN)
                & ((acts % 9) == 0)
            )
            active_rows, active_slots = np.nonzero(~passive_box_camp)
            if len(active_rows):
                self.active_rotational_pairs.append(
                    np.column_stack((active_rows + offset, active_slots)).astype(np.int64)
                )
            offset += len(shard["states"])
        if not self.shards:
            raise ValueError(f"el split {split!r} no tiene transiciones válidas en {dataset}")
        self.ends = np.cumsum([len(shard["states"]) for shard in self.shards], dtype=np.int64)
        self.ticks = int(self.ends[-1])
        self.near_ball_index = (
            np.concatenate(self.near_ball_pairs, axis=0)
            if self.near_ball_pairs else np.empty((0, 2), dtype=np.int64)
        )
        self.active_rotational_index = (
            np.concatenate(self.active_rotational_pairs, axis=0)
            if self.active_rotational_pairs else np.empty((0, 2), dtype=np.int64)
        )

    def sample_indices(self, count: int, rng: np.random.Generator,
                       near_ball_frac: float = 0.0,
                       filter_box_camping: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """Sample (tick_indices, player_slots) with near-ball oversampling and anti-box-camping filtering."""
        near_count = int(round(count * float(np.clip(near_ball_frac, 0.0, 1.0))))
        if near_count > 0 and len(self.near_ball_index) > 0:
            chosen = self.near_ball_index[rng.integers(0, len(self.near_ball_index), size=near_count)]
            rem = count - near_count
            if rem > 0:
                if filter_box_camping and self.tactical and len(self.active_rotational_index) > 0:
                    act_chosen = self.active_rotational_index[
                        rng.integers(0, len(self.active_rotational_index), size=rem)
                    ]
                    uni_idx, uni_slots = act_chosen[:, 0], act_chosen[:, 1]
                else:
                    uni_idx = rng.integers(0, self.ticks, size=rem)
                    uni_slots = rng.integers(0, N_PLAYERS, size=rem)
                indices = np.concatenate((chosen[:, 0], uni_idx))
                slots = np.concatenate((chosen[:, 1], uni_slots))
                perm = rng.permutation(count)
                return indices[perm], slots[perm]
            return chosen[:, 0], chosen[:, 1]
        if filter_box_camping and self.tactical and len(self.active_rotational_index) > 0:
            act_chosen = self.active_rotational_index[
                rng.integers(0, len(self.active_rotational_index), size=count)
            ]
            return act_chosen[:, 0], act_chosen[:, 1]
        return rng.integers(0, self.ticks, size=count), rng.integers(0, N_PLAYERS, size=count)

    def batch(self, indices: np.ndarray, slots: np.ndarray, *,
              tactical: bool | None = None, mirror_mask: np.ndarray | None = None,
              use_stride_prior: bool = True) -> tuple:
        indices = np.asarray(indices, dtype=np.int64)
        slots = np.asarray(slots, dtype=np.int64)
        if indices.shape != slots.shape or np.any(indices < 0) or np.any(indices >= self.ticks):
            raise ValueError("índices de muestras fuera de rango")
        if np.any((slots < 0) | (slots >= N_PLAYERS)):
            raise ValueError("slot de jugador fuera de rango")
        use_tactical = self.tactical if tactical is None else bool(tactical)
        shard_ids = np.searchsorted(self.ends, indices, side="right")
        starts = np.concatenate(([0], self.ends[:-1]))
        local = indices - starts[shard_ids]
        states = np.empty((len(indices), 7, 4), dtype=np.float32)
        actions = np.empty(len(indices), dtype=np.int64)
        prior = np.empty(len(indices), dtype=np.int64)
        has_prior = np.empty(len(indices), dtype=bool)
        changed = np.empty(len(indices), dtype=bool)
        score_diff = np.empty(len(indices), dtype=np.float32)
        elapsed = np.empty(len(indices), dtype=np.float32)
        for shard_id in np.unique(shard_ids):
            mask = shard_ids == shard_id
            shard = self.shards[int(shard_id)]
            row = local[mask]
            actor = slots[mask]
            states[mask] = shard["states"][row]
            actions[mask] = shard["actions"][row, actor]
            if use_stride_prior and "stride3_prior_actions" in shard:
                prior[mask] = shard["stride3_prior_actions"][row, actor]
                has_prior[mask] = shard["stride3_has_prior"][row]
                changed[mask] = shard["stride3_changed"][row, actor]
            else:
                prior[mask] = shard["prior_actions"][row, actor]
                has_prior[mask] = shard["has_prior"][row]
                changed[mask] = shard["changed"][row, actor]
            score_diff[mask] = shard["score_diff"][row]
            elapsed[mask] = shard["time"][row]
        if mirror_mask is not None:
            mirror_mask = np.asarray(mirror_mask, dtype=bool)
            if mirror_mask.any():
                states[mirror_mask] = mirror_states_y(states[mirror_mask])
                actions[mirror_mask] = flip_action_y(actions[mirror_mask])
                prior[mirror_mask] = flip_action_y(prior[mirror_mask])
        if use_tactical:
            observation = featurize_tactical(
                states, slots, self.field_half_w, self.field_half_h, self.goal_x
            )
        else:
            observation = featurize(
                states, slots, score_diff, elapsed, self.field_half_w, self.field_half_h
            )
        return observation, actions, prior, has_prior, changed

    def fixed_indices(self, count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
        if count < 1:
            raise ValueError("el conjunto fijo de evaluación debe tener al menos una muestra")
        rng = np.random.default_rng(seed)
        return rng.integers(0, self.ticks, size=count), rng.integers(0, N_PLAYERS, size=count)

    def label_counts(self) -> np.ndarray:
        counts = np.zeros(N_ACTIONS, dtype=np.int64)
        for shard in self.shards:
            counts += np.bincount(shard["actions"].ravel(), minlength=N_ACTIONS)
        return counts


def _ordered_players(tick: dict) -> list[dict] | None:
    players = tick.get("players", [])
    counts = Counter(player.get("team") for player in players)
    if counts != {1: 3, 2: 3}:
        return None
    if any(player.get("id") is None or player.get("disc") is None for player in players):
        return None
    ordered = sorted(players, key=lambda player: (player["team"], player["id"]))
    disc_indices = [player["disc"] for player in ordered]
    if len(set(disc_indices)) != N_PLAYERS or any(not 0 < index < len(tick.get("discs", []))
                                                   for index in disc_indices):
        return None
    return ordered


def _encode_tick(tick: dict, ordered: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    discs = tick["discs"]
    indices = [0] + [player["disc"] for player in ordered]
    state = np.asarray([[discs[index][key] for key in ("x", "y", "vx", "vy")] for index in indices],
                       dtype=np.float32)
    if not np.isfinite(state).all():
        raise ValueError(f"estado no finito en frame {tick['frame']}")
    actions = np.asarray([action_from_input(player.get("input", 0), player["team"]) for player in ordered],
                         dtype=np.uint8)
    return state, actions


def _tick_is_encodable(tick: dict, ordered: list[dict]) -> bool:
    indices = [0] + [player["disc"] for player in ordered]
    return all(
        all(disc.get(key) is not None and np.isfinite(disc[key]) for key in ("x", "y", "vx", "vy"))
        for disc in (tick["discs"][index] for index in indices)
    )


def extract_recording(name: str, audit_row: dict, index_row: dict, cache: Path) -> dict:
    """Extract valid transition starts; labels are inputs at t for the verified t -> t+1 transition."""
    if not audit_row.get("complete"):
        raise ValueError(f"replay incompleto no elegible para BC: {name}")
    spans = audit_row.get("targetSpans", [])
    if not spans:
        return _empty_shard()
    ticks, event_frames, _, property_summary, _, _ = _read_replay(
        cache / index_row["jsonl"], spans, default_player_properties=load_stadium(
            ROOT / "stadiums" / "futsal3bybazinga.hbs").player)
    target_stadium = load_stadium(STADIUM)

    states, actions, prior_actions, has_prior, changed, frames, score_diff, elapsed = (
        [] for _ in range(8)
    )
    stride3_prior_actions, stride3_has_prior, stride3_changed = [], [], []
    previous_frame = None
    previous_actions = None
    last_observed_frame = None
    stationary_frames = np.zeros(N_PLAYERS, dtype=np.int64)
    held_kick_outside_frames = np.zeros(N_PLAYERS, dtype=np.int64)
    recent_by_frame: dict[int, np.ndarray] = {}

    for previous, current in zip(ticks, ticks[1:]):
        span = _span_for(int(previous["frame"]), spans)
        if span is None or not transition_is_valid(
                previous, current, span, event_frames, check_coincident=False, allowed_states=(0, 1)):
            continue
        ordered = _ordered_players(previous)
        next_ordered = _ordered_players(current)
        if ordered is None or next_ordered is None \
                or not _tick_is_encodable(previous, ordered) or not _tick_is_encodable(current, next_ordered):
            continue
        roster = tuple((player["id"], player["team"]) for player in ordered)
        if roster != tuple((player["id"], player["team"]) for player in next_ordered):
            continue
        state, label = _encode_tick(previous, ordered)
        frame_no = int(previous["frame"])
        if last_observed_frame != frame_no - 1:
            stationary_frames.fill(0)
            held_kick_outside_frames.fill(0)
            recent_by_frame.clear()

        ball_xy = state[0, :2]
        cleaned_label = label.copy()
        for player_index in range(N_PLAYERS):
            disc = state[player_index + 1]
            no_direction = int(label[player_index] % 9) == 0
            slow = float(np.hypot(disc[2], disc[3])) < AFK_SPEED_THRESHOLD
            stationary_frames[player_index] = (
                stationary_frames[player_index] + 1 if no_direction and slow else 0
            )
            dist_ball = float(np.hypot(disc[0] - ball_xy[0], disc[1] - ball_xy[1]))
            if int(label[player_index]) >= 9 and dist_ball > EFFECTIVE_KICK_DIST + 12.0:
                held_kick_outside_frames[player_index] += 1
                if held_kick_outside_frames[player_index] > MAX_HELD_KICK_OUTSIDE_REACH:
                    cleaned_label[player_index] = int(label[player_index] % 9)
            else:
                held_kick_outside_frames[player_index] = 0

        outside = np.any(np.abs(state[1:, 0]) > target_stadium.field_half_w + AFK_OUTSIDE_MARGIN) \
            or np.any(np.abs(state[1:, 1]) > target_stadium.field_half_h + AFK_OUTSIDE_MARGIN)
        if outside or np.any(stationary_frames >= AFK_STATIONARY_FRAMES):
            last_observed_frame = frame_no
            continue
        contiguous = previous_frame == frame_no - 1
        stride3_prev = recent_by_frame.get(frame_no - DECISION_STRIDE)
        has_stride3 = stride3_prev is not None

        states.append(state)
        actions.append(cleaned_label)
        prior_actions.append(previous_actions if contiguous else np.zeros(N_PLAYERS, dtype=np.uint8))
        has_prior.append(contiguous)
        changed.append((cleaned_label != previous_actions) if contiguous else np.zeros(N_PLAYERS, dtype=bool))
        stride3_prior_actions.append(stride3_prev if has_stride3 else np.zeros(N_PLAYERS, dtype=np.uint8))
        stride3_has_prior.append(has_stride3)
        stride3_changed.append((cleaned_label != stride3_prev) if has_stride3 else np.zeros(N_PLAYERS, dtype=bool))
        frames.append(frame_no)
        scores = previous.get("score")
        if scores is None or len(scores) != 2 or scores[0] is None or scores[1] is None:
            raise ValueError(f"marcador inválido en {name}, frame {previous['frame']}")
        score_diff.append(int(scores[0]) - int(scores[1]))
        if previous.get("time") is None:
            raise ValueError(f"reloj faltante en {name}, frame {previous['frame']}")
        elapsed.append(float(previous["time"]))
        previous_frame = frame_no
        previous_actions = cleaned_label
        last_observed_frame = frame_no
        recent_by_frame[frame_no] = cleaned_label
        if len(recent_by_frame) > 8:
            recent_by_frame.pop(min(recent_by_frame), None)

    if not states:
        return _empty_shard()
    return {
        "states": np.asarray(states, dtype=np.float16),
        "actions": np.asarray(actions, dtype=np.uint8),
        "prior_actions": np.asarray(prior_actions, dtype=np.uint8),
        "has_prior": np.asarray(has_prior, dtype=bool),
        "changed": np.asarray(changed, dtype=bool),
        "stride3_prior_actions": np.asarray(stride3_prior_actions, dtype=np.uint8),
        "stride3_has_prior": np.asarray(stride3_has_prior, dtype=bool),
        "stride3_changed": np.asarray(stride3_changed, dtype=bool),
        "frame": np.asarray(frames, dtype=np.int32),
        "score_diff": np.asarray(score_diff, dtype=np.int16),
        "time": np.asarray(elapsed, dtype=np.float32),
        "playerPhysicsOverrideTicks": property_summary["playerPhysicsOverrideTicks"],
    }


def _empty_shard() -> dict:
    return {
        "states": np.empty((0, 7, 4), dtype=np.float16),
        "actions": np.empty((0, N_PLAYERS), dtype=np.uint8),
        "prior_actions": np.empty((0, N_PLAYERS), dtype=np.uint8),
        "has_prior": np.empty(0, dtype=bool),
        "changed": np.empty((0, N_PLAYERS), dtype=bool),
        "stride3_prior_actions": np.empty((0, N_PLAYERS), dtype=np.uint8),
        "stride3_has_prior": np.empty(0, dtype=bool),
        "stride3_changed": np.empty((0, N_PLAYERS), dtype=bool),
        "frame": np.empty(0, dtype=np.int32),
        "score_diff": np.empty(0, dtype=np.int16),
        "time": np.empty(0, dtype=np.float32),
        "playerPhysicsOverrideTicks": 0,
    }


def _atomic_save_npz(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    try:
        with temp.open("wb") as handle:
            np.savez_compressed(handle, **{key: value for key, value in data.items()
                                           if key != "playerPhysicsOverrideTicks"})
        temp.replace(path)
    finally:
        if temp.exists():
            temp.unlink()


def _load_existing_shard(path: Path, expected_sha256: str) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    required = {"states", "actions", "prior_actions", "has_prior", "changed",
                "frame", "score_diff", "time"}
    if not required.issubset(data):
        raise ValueError(f"shard existente incompleto: {path}")
    count = len(data["states"])
    if data["states"].shape != (count, 7, 4) or data["actions"].shape != (count, N_PLAYERS) \
            or data["prior_actions"].shape != (count, N_PLAYERS) or data["changed"].shape != (count, N_PLAYERS) \
            or data["has_prior"].shape != (count,) or data["frame"].shape != (count,) \
            or data["score_diff"].shape != (count,) or data["time"].shape != (count,):
        raise ValueError(f"shard existente con dimensiones incompatibles: {path}")
    version = str(data.get("dataVersion", ""))
    source_sha = str(data.get("sourceSha256", ""))
    if version and version not in (DATA_VERSION, TACTICAL_DATA_VERSION):
        raise ValueError(f"shard existente de otra versión: {path}")
    if source_sha and source_sha != expected_sha256:
        raise ValueError(f"shard existente de otro replay: {path}")
    if not version or not source_sha:
        data["dataVersion"] = np.asarray(DATA_VERSION)
        data["sourceSha256"] = np.asarray(expected_sha256)
        _atomic_save_npz(path, data)
    return data


def _build_shard(job: tuple) -> dict:
    name, split, audit_row, index_row, cache, out = job
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:12]
    filename = f"{Path(name).stem}.{digest}.npz"
    relative = Path("shards") / split / filename
    path = out / relative
    if path.is_file():
        data = _load_existing_shard(path, index_row["sha256"])
        count = len(data["states"])
        return {"recording": name, "sourceSha256": index_row["sha256"], "room": audit_row.get("room"),
                "split": split, "file": str(relative).replace("\\", "/"), "ticks": count,
                "empty": count == 0, "reused": True}

    data = extract_recording(name, audit_row, index_row, cache)
    count = len(data["states"])
    if count:
        data["dataVersion"] = np.asarray(DATA_VERSION)
        data["sourceSha256"] = np.asarray(index_row["sha256"])
        _atomic_save_npz(path, data)
    return {"recording": name, "sourceSha256": index_row["sha256"], "room": audit_row.get("room"),
            "split": split, "file": str(relative).replace("\\", "/"), "ticks": count,
            "empty": count == 0, "reused": False}


def build_dataset(cache: Path, out: Path, seed: int, workers: int) -> dict:
    if workers < 1:
        raise ValueError("workers debe ser al menos 1")
    audit = json.loads((cache / "audit.json").read_text(encoding="utf-8"))["recordings"]
    index = json.loads((cache / "index.json").read_text(encoding="utf-8"))["recordings"]
    splits = make_splits(audit, seed)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    existing_rows = {}
    processed = set()
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old.get("dataVersion") not in (DATA_VERSION, TACTICAL_DATA_VERSION) or old.get("seed") != seed:
            raise ValueError("el directorio ya contiene un dataset de otra versión/semilla; usa otro --out")
        if old.get("split", {}).get("assignments") != splits["assignments"]:
            raise ValueError("el split cambió; usa otro --out para no mezclar shards")
        old_rows = {row["recording"]: row for row in old.get("shards", [])}
        processed = set(old.get("emptyRecordings", []))
        for name, row in old_rows.items():
            if name not in splits["assignments"] or row["split"] != splits["assignments"][name]:
                raise ValueError(f"shard pertenece a otro split: {name}")
            path = out / row["file"]
            if not path.is_file():
                continue
            data = _load_existing_shard(path, index[name]["sha256"])
            if len(data["states"]) != row["ticks"]:
                raise ValueError(f"cantidad de ticks cambió en shard existente: {name}")
            row["sourceSha256"] = index[name]["sha256"]
            existing_rows[name] = row
            processed.add(name)

    manifest = {
        "dataVersion": DATA_VERSION,
        "targetStadiumIdentity": json.loads((cache / "audit.json").read_text(encoding="utf-8"))[
            "summary"]["targetStadiumIdentity"],
        "cache": str(cache),
        "seed": seed,
        "split": splits,
        "filter": FILTER_CONTRACT,
        "featureContract": FEATURE_CONTRACT,
        "tacticalFeatureContract": TACTICAL_FEATURE_CONTRACT,
        "buildStatus": "building",
        "processedRecordings": sorted(processed),
        "emptyRecordings": sorted(set(processed) - set(existing_rows)),
        "shardCount": len(existing_rows),
        "tickCountBySplit": {
            split: sum(row["ticks"] for row in existing_rows.values() if row["split"] == split)
            for split in ("train", "dev", "test")
        },
        "shards": sorted(existing_rows.values(), key=lambda row: row["recording"]),
    }
    _write_json(manifest_path, manifest)
    _write_json(out / "splits.json", splits)

    jobs = [(name, split, audit[name], index[name], cache, out)
            for name, split in splits["assignments"].items() if name not in processed]
    if workers == 1:
        results = map(_build_shard, jobs)
        for result in results:
            _record_result(manifest, result, manifest_path)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for result in pool.map(_build_shard, jobs, chunksize=1):
                _record_result(manifest, result, manifest_path)

    manifest["buildStatus"] = "complete"
    manifest["processedRecordings"] = sorted(splits["assignments"])
    _write_json(manifest_path, manifest)
    return manifest


def _record_result(manifest: dict, result: dict, manifest_path: Path) -> None:
    name = result["recording"]
    manifest["processedRecordings"] = sorted(set(manifest["processedRecordings"]) | {name})
    empty = set(manifest.get("emptyRecordings", []))
    rows = {row["recording"]: row for row in manifest["shards"]}
    if result["ticks"]:
        rows[name] = {key: result[key] for key in
                      ("recording", "sourceSha256", "room", "split", "file", "ticks")}
        empty.discard(name)
    else:
        rows.pop(name, None)
        empty.add(name)
    manifest["emptyRecordings"] = sorted(empty)
    manifest["shards"] = sorted(rows.values(), key=lambda row: row["recording"])
    manifest["shardCount"] = len(manifest["shards"])
    manifest["tickCountBySplit"] = {
        split: sum(row["ticks"] for row in manifest["shards"] if row["split"] == split)
        for split in ("train", "dev", "test")
    }
    _write_json(manifest_path, manifest)
    print(f"{result['split']:5s} {result['ticks']:7d} {name}"
          f"{' (reutilizado)' if result['reused'] else ''}", flush=True)


def refresh_manifest(cache: Path, out: Path, seed: int) -> dict:
    """Refresh split and feature metadata after an interrupted or earlier shard build."""
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    audit_doc = json.loads((cache / "audit.json").read_text(encoding="utf-8"))
    index = json.loads((cache / "index.json").read_text(encoding="utf-8"))["recordings"]
    splits = make_splits(audit_doc["recordings"], seed)
    for row in manifest["shards"]:
        recording = row["recording"]
        split = splits["assignments"].get(recording)
        if split is None:
            raise ValueError(f"shard no elegible para el split actual: {recording}")
        expected = Path("shards") / split / Path(row["file"]).name
        if Path(row["file"]) != expected or not (out / expected).is_file():
            raise ValueError(f"shard falta o pertenece a otro split: {row['file']}")
        row["split"] = split
        row["file"] = str(expected).replace("\\", "/")
        row["sourceSha256"] = index[recording]["sha256"]
        shard = _load_existing_shard(out / expected, index[recording]["sha256"])
        if len(shard["states"]) != row["ticks"]:
            raise ValueError(f"cantidad de ticks cambió en shard existente: {recording}")
    manifest["dataVersion"] = DATA_VERSION
    manifest["targetStadiumIdentity"] = audit_doc["summary"]["targetStadiumIdentity"]
    manifest["seed"] = seed
    manifest["split"] = splits
    manifest["filter"] = FILTER_CONTRACT
    manifest["featureContract"] = FEATURE_CONTRACT
    manifest["tacticalFeatureContract"] = TACTICAL_FEATURE_CONTRACT
    manifest["shardCount"] = len(manifest["shards"])
    manifest["buildStatus"] = "complete" if set(splits["assignments"]) <= set(
        manifest.get("processedRecordings", [])) else "building"
    manifest["tickCountBySplit"] = {
        split: sum(row["ticks"] for row in manifest["shards"] if row["split"] == split)
        for split in ("train", "dev", "test")
    }
    _write_json(manifest_path, manifest)
    _write_json(out / "splits.json", splits)
    return manifest


def _write_json(path: Path, value: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=20261027)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--refresh-manifest", action="store_true",
                        help="actualizar split y metadatos sin reextraer shards existentes")
    args = parser.parse_args()
    if args.refresh_manifest:
        manifest = refresh_manifest(args.cache, args.out, args.seed)
    else:
        manifest = build_dataset(args.cache, args.out, args.seed, args.workers)
    print(json.dumps({
        "dataVersion": manifest["dataVersion"],
        "shardCount": manifest["shardCount"],
        "tickCountBySplit": manifest["tickCountBySplit"],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()