"""Liga de oponentes para self-play: snapshots congelados del agente + Elo + PFSP.

PFSP (prioritized fictitious self-play, AlphaStar): se juega más contra los rivales a los que
peor les ganamos: peso = (1 - winrate)^p. Así no se olvida cómo ganarle a estilos viejos.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np

from .model import ActorCritic


@dataclass
class Member:
    name: str
    model: ActorCritic
    elo: float
    wins: float = 1.0     # prior (1 victoria, 1 derrota)
    games: float = 2.0

    @property
    def winrate(self) -> float:  # winrate del *aprendiz* contra este miembro
        return self.wins / self.games


@dataclass
class League:
    max_size: int = 50
    pfsp_power: float = 2.0
    k_elo: float = 16.0
    learner_elo: float = 1000.0
    members: list[Member] = field(default_factory=list)
    scripted_elo: float = 1000.0

    def add_snapshot(self, model: ActorCritic, name: str) -> None:
        m = copy.deepcopy(model).eval()
        for p in m.parameters():
            p.requires_grad_(False)
        self.members.append(Member(name, m, self.learner_elo))
        if len(self.members) > self.max_size:
            # descartar el más fácil (mayor winrate), nunca el más reciente
            i = int(np.argmax([x.winrate for x in self.members[:-1]]))
            self.members.pop(i)

    def sample(self, k: int, rng: np.random.Generator) -> list[int]:
        if not self.members:
            return []
        w = np.array([(1.0 - m.winrate) ** self.pfsp_power + 1e-3 for m in self.members])
        # siempre algo de peso extra al último snapshot
        w[-1] += w.sum() * 0.25
        return list(rng.choice(len(self.members), size=k, p=w / w.sum()))

    @staticmethod
    def _expected(a: float, b: float) -> float:
        return 1.0 / (1.0 + 10 ** ((b - a) / 400))

    def record(self, idx: int | None, learner_goals: int, opp_goals: int) -> None:
        """Cada gol cuenta como una 'partida' para Elo y winrate. idx=None -> bot scripteado."""
        for won, n in ((1.0, learner_goals), (0.0, opp_goals)):
            for _ in range(int(n)):
                opp_elo = self.scripted_elo if idx is None else self.members[idx].elo
                e = self._expected(self.learner_elo, opp_elo)
                self.learner_elo += self.k_elo * (won - e)
                if idx is None:
                    self.scripted_elo -= self.k_elo * (won - e)
                else:
                    m = self.members[idx]
                    m.wins += won
                    m.games += 1
