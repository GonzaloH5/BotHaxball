"""Memoria GRU residual sobre el modelo universal; permite aprovechar un BC existente."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .model import SetActorCritic


class RecurrentSetActorCritic(SetActorCritic):
    is_recurrent = True

    def __init__(self, *args, memory_size=64, **kwargs):
        super().__init__(*args, **kwargs)
        if memory_size < 1:
            raise ValueError("memory_size debe ser positivo")
        self.memory_size = memory_size
        joint = self.self_dim + 4 * self.ent_hidden
        if self.pooling == "attention":
            joint += self.ent_hidden
        self.gru = nn.GRUCell(joint + self.n_actions + 1, memory_size)
        self.memory_pi = nn.Linear(memory_size, self.n_actions, bias=False)
        self.memory_v = nn.Linear(memory_size, 1, bias=False)
        # Al importar BC, la política inicial sigue siendo exactamente la de BC
        # sobre las mismas observaciones. PPO aprende luego la contribución temporal.
        nn.init.zeros_(self.memory_pi.weight)
        nn.init.zeros_(self.memory_v.weight)

    def config(self):
        return {**super().config(), "type": "recurrent_set", "memory_size": self.memory_size}

    def initial_state(self, batch):
        return self.pi.weight.new_zeros(batch, self.memory_size)

    def _advance(self, features, memory, previous_action, episode_start=None):
        if episode_start is not None:
            memory = torch.where(episode_start[:, None], torch.zeros_like(memory), memory)
            previous_action = torch.where(episode_start, self.n_actions, previous_action)
        previous = F.one_hot(previous_action.long(), self.n_actions + 1).to(features.dtype)
        return self.gru(torch.cat([features, previous], dim=-1), memory)

    def _heads(self, features, memory):
        logits = self.pi(self.pi_body(features)) + self.memory_pi(memory)
        value = self.v(self.v_body(features)) + self.memory_v(memory)
        return logits, value.squeeze(-1)

    def step(self, obs, memory, previous_action, episode_start=None):
        features = self._features(obs)
        memory = self._advance(features, memory, previous_action, episode_start)
        logits, value = self._heads(features, memory)
        return logits, value, memory

    def sequence(self, obs, memory, previous_action, episode_start):
        """(tiempo, secuencias, obs). No mezclar ticks al formar minibatches."""
        length, batch, width = obs.shape
        features = self._features(obs.reshape(length * batch, width)).reshape(length, batch, -1)
        history = []
        for t in range(length):
            memory = self._advance(features[t], memory, previous_action[t], episode_start[t])
            history.append(memory)
        states = torch.stack(history)
        logits, values = self._heads(features.reshape(length * batch, -1),
                                     states.reshape(length * batch, self.memory_size))
        return logits.reshape(length, batch, -1), values.reshape(length, batch), memory

    def initialize_from(self, checkpoint):
        kind = checkpoint["model_config"].get("type", "mlp")
        if kind == "recurrent_set":
            self.load_state_dict(checkpoint["model"])
            return
        if kind != "set":
            raise ValueError("La memoria requiere un checkpoint universal de tipo set")
        missing, unexpected = self.load_state_dict(checkpoint["model"], strict=False)
        allowed_prefixes = ("gru.", "memory_pi.", "memory_v.", "q.", "attn.", "residual_attn.",
                            "mate_attn_proj.", "opp_attn_proj.", "attn_gate")
        allowed = {name for name in self.state_dict() if name.startswith(allowed_prefixes)}
        if set(missing) != allowed or unexpected:
            raise ValueError(f"Checkpoint BC incompatible: faltan {missing}, sobran {unexpected}")
        nn.init.zeros_(self.memory_pi.weight)
        nn.init.zeros_(self.memory_v.weight)

    def forward(self, obs):
        raise RuntimeError("Modelo recurrente: usar step(obs, memory, previous_action) o sequence")

    def logits(self, obs):
        raise RuntimeError("La política recurrente requiere memoria; usar step")


class RecurrentPolicyOnly(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, obs, memory, previous_action):
        logits, _, updated = self.model.step(obs, memory, previous_action)
        return logits, updated
