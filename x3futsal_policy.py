"""Permutation-invariant Futsal x3 behavior-cloning and tactical set policies."""
from __future__ import annotations

import numpy as np

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError:  # pragma: no cover - permite importar utilidades numpy sin torch
    torch = None
    nn = None
    F = None

from learn.x3futsal_data import (
    EFFECTIVE_KICK_DIST,
    TACTICAL_ENTITY_DIM,
    TACTICAL_SELF_DIM,
)

SELF_DIM = 12
ENTITY_DIM = 6
TACTICAL_CTX_DIM = 17
N_MATES = 2
N_RIVALS = 3
N_ACTIONS = 18
POLICY_VERSION = "x3-set-history-v2"
TACTICAL_POLICY_VERSION = "x3-tactical-set-v3"


def decode_actions_np(
    logits: np.ndarray,
    *,
    ball_dist_px: np.ndarray | None = None,
    kick_cancel: np.ndarray | None = None,
    shot_alignment: np.ndarray | None = None,
    temperature: float = 1.0,
    greedy: bool = True,
    kick_threshold: float = 0.28,
    stay_escape_threshold: float = 0.36,
    max_kick_dist_px: float = EFFECTIVE_KICK_DIST + 3.5,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Decode 18-class logits into actions by decoupling movement and kick decisions.

    Fully vectorized across the batch (both greedy and stochastic inverse-CDF modes).
    """
    logits = np.asarray(logits, dtype=np.float64)
    temp = max(float(temperature), 1e-3)
    scaled = logits / temp
    scaled -= scaled.max(axis=-1, keepdims=True)
    probs = np.exp(scaled)
    probs /= np.maximum(probs.sum(axis=-1, keepdims=True), 1e-12)

    move_probs = probs[:, :9] + probs[:, 9:]
    kick_probs = probs[:, 9:].sum(axis=-1)

    if greedy or rng is None:
        directions = move_probs.argmax(axis=-1)
        active_mass = 1.0 - move_probs[:, 0]
        escape_mask = (directions == 0) & (active_mass >= stay_escape_threshold)
        if np.any(escape_mask):
            directions[escape_mask] = move_probs[escape_mask, 1:].argmax(axis=-1) + 1
        kick = kick_probs >= kick_threshold
    else:
        p = move_probs.copy()
        escape_mask = (1.0 - p[:, 0]) >= stay_escape_threshold
        if np.any(escape_mask):
            p[escape_mask, 0] *= 0.35
            p[escape_mask] /= np.maximum(p[escape_mask].sum(axis=-1, keepdims=True), 1e-12)
        cdf = np.cumsum(p, axis=-1)
        cdf[:, -1] = 1.0
        u = rng.random((len(probs), 1))
        directions = (cdf >= u).argmax(axis=-1).astype(np.int64)
        kick = rng.random(len(probs)) < np.clip(kick_probs / max(kick_threshold * 2.0, 1e-6), 0.0, 1.0)

    if ball_dist_px is not None:
        ball_dist_px = np.asarray(ball_dist_px, dtype=np.float64).reshape(-1)
        kick &= ball_dist_px <= max_kick_dist_px
    if shot_alignment is not None:
        shot_alignment = np.asarray(shot_alignment, dtype=np.float64).reshape(-1)
        kick &= shot_alignment >= -0.35
    if kick_cancel is not None:
        kick &= ~np.asarray(kick_cancel, dtype=bool).reshape(-1)

    return directions.astype(np.int64) + np.where(kick, 9, 0)


if nn is not None:
    class X3SetPolicy(nn.Module):
        """Shared policy over self/ball context and pooled teammate/opponent sets (v2 contract)."""

        def __init__(self, hidden: int = 128, entity_hidden: int = 64, history_dropout: float = 0.0):
            super().__init__()
            self.history_dropout = float(history_dropout)
            self.self_net = nn.Sequential(nn.Linear(SELF_DIM, hidden), nn.ReLU(),
                                          nn.Linear(hidden, hidden), nn.ReLU())
            self.mate_net = nn.Sequential(nn.Linear(ENTITY_DIM, entity_hidden), nn.ReLU(),
                                          nn.Linear(entity_hidden, entity_hidden), nn.ReLU())
            self.rival_net = nn.Sequential(nn.Linear(ENTITY_DIM, entity_hidden), nn.ReLU(),
                                           nn.Linear(entity_hidden, entity_hidden), nn.ReLU())
            self.head = nn.Sequential(nn.Linear(hidden + 4 * entity_hidden + N_ACTIONS + 1, hidden), nn.ReLU(),
                                      nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, N_ACTIONS))

        @staticmethod
        def _pool(entities: torch.Tensor) -> torch.Tensor:
            return torch.cat((entities.mean(dim=1), entities.max(dim=1).values), dim=-1)

        def forward(self, own: torch.Tensor, mates: torch.Tensor, rivals: torch.Tensor,
                    previous_action: torch.Tensor, has_previous: torch.Tensor) -> torch.Tensor:
            if previous_action.shape != own.shape[:-1] or has_previous.shape != own.shape[:-1]:
                raise ValueError("la historia de acción debe tener una entrada por observación")
            context = self.self_net(own)
            mate_set = self._pool(self.mate_net(mates))
            rival_set = self._pool(self.rival_net(rivals))
            effective_has_prev = has_previous.to(dtype=own.dtype)
            if self.training and self.history_dropout > 0.0:
                keep = (torch.rand_like(effective_has_prev) >= self.history_dropout).to(dtype=own.dtype)
                effective_has_prev = effective_has_prev * keep
            history_mask = effective_has_prev.unsqueeze(-1)
            history_action = F.one_hot(previous_action.to(dtype=torch.long), num_classes=N_ACTIONS).to(dtype=own.dtype)
            history = torch.cat((history_action * history_mask, history_mask), dim=-1)
            return self.head(torch.cat((context, mate_set, rival_set, history), dim=-1))


    class X3TacticalPolicy(nn.Module):
        """Context-conditioned Deep Sets policy over v3 tactical/contact observations (28 self, 10 entity)."""

        def __init__(self, hidden: int = 192, entity_hidden: int = 96,
                     use_history: bool = False, history_dropout: float = 0.5):
            super().__init__()
            self.hidden = int(hidden)
            self.entity_hidden = int(entity_hidden)
            self.use_history = bool(use_history)
            self.history_dropout = float(history_dropout)
            ent_in = TACTICAL_ENTITY_DIM + TACTICAL_CTX_DIM
            hist_dim = (N_ACTIONS + 1) if self.use_history else 0
            self.self_net = nn.Sequential(
                nn.Linear(TACTICAL_SELF_DIM, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
            )
            self.mate_net = nn.Sequential(
                nn.Linear(ent_in, entity_hidden),
                nn.ReLU(),
                nn.Linear(entity_hidden, entity_hidden),
                nn.ReLU(),
            )
            self.rival_net = nn.Sequential(
                nn.Linear(ent_in, entity_hidden),
                nn.ReLU(),
                nn.Linear(entity_hidden, entity_hidden),
                nn.ReLU(),
            )
            self.trunk = nn.Sequential(
                nn.Linear(hidden + 4 * entity_hidden + hist_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
            )
            self.logits = nn.Linear(hidden, N_ACTIONS)

        @staticmethod
        def _pool(entities: torch.Tensor) -> torch.Tensor:
            return torch.cat((entities.mean(dim=1), entities.max(dim=1).values), dim=-1)

        @staticmethod
        def _encode_entities(net: nn.Sequential, entities: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
            """Zero-allocation linear decomposition: avoids expanding and concatenating `ctx` across entities."""
            first: nn.Linear = net[0]
            w = first.weight
            b = first.bias
            ent_proj = F.linear(entities, w[:, :TACTICAL_ENTITY_DIM])
            ctx_proj = F.linear(ctx, w[:, TACTICAL_ENTITY_DIM:], b).unsqueeze(1)
            return net[1:](ent_proj + ctx_proj)

        def features(self, own: torch.Tensor, mates: torch.Tensor, rivals: torch.Tensor,
                     previous_action: torch.Tensor | None = None,
                     has_previous: torch.Tensor | None = None) -> torch.Tensor:
            ctx = own[:, :TACTICAL_CTX_DIM]
            parts = [
                self.self_net(own),
                self._pool(self._encode_entities(self.mate_net, mates, ctx)),
                self._pool(self._encode_entities(self.rival_net, rivals, ctx)),
            ]
            if self.use_history:
                if previous_action is None or has_previous is None:
                    previous_action = torch.zeros(own.shape[:-1], dtype=torch.long, device=own.device)
                    has_previous = torch.zeros(own.shape[:-1], dtype=torch.bool, device=own.device)
                effective_has_prev = has_previous.to(dtype=own.dtype)
                if self.training and self.history_dropout > 0.0:
                    keep = (torch.rand_like(effective_has_prev) >= self.history_dropout).to(dtype=own.dtype)
                    effective_has_prev = effective_has_prev * keep
                history_mask = effective_has_prev.unsqueeze(-1)
                history_action = F.one_hot(previous_action.to(dtype=torch.long), num_classes=N_ACTIONS).to(dtype=own.dtype)
                parts.append(torch.cat((history_action * history_mask, history_mask), dim=-1))
            return self.trunk(torch.cat(parts, dim=-1))

        def forward(self, own: torch.Tensor, mates: torch.Tensor, rivals: torch.Tensor,
                    previous_action: torch.Tensor | None = None,
                    has_previous: torch.Tensor | None = None) -> torch.Tensor:
            return self.logits(self.features(own, mates, rivals, previous_action, has_previous))
else:  # pragma: no cover
    X3SetPolicy = None
    X3TacticalPolicy = None