"""Permutation-invariant Futsal x3 behavior-cloning policy."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

SELF_DIM = 12
ENTITY_DIM = 6
N_MATES = 2
N_RIVALS = 3
N_ACTIONS = 18
POLICY_VERSION = "x3-set-history-v2"


class X3SetPolicy(nn.Module):
    """Shared policy over self/ball context and pooled teammate/opponent sets."""

    def __init__(self, hidden: int = 128, entity_hidden: int = 64):
        super().__init__()
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
        history_mask = has_previous.to(dtype=own.dtype).unsqueeze(-1)
        history_action = F.one_hot(previous_action.to(dtype=torch.long), num_classes=N_ACTIONS).to(dtype=own.dtype)
        history = torch.cat((history_action * history_mask, history_mask), dim=-1)
        return self.head(torch.cat((context, mate_set, rival_set, history), dim=-1))
