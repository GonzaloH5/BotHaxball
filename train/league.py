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
    match_points: float = 1.0
    matches: float = 2.0
    protected: bool = False
    snapshot_steps: int = 0

    @property
    def winrate(self) -> float:  # winrate del *aprendiz* contra este miembro
        return self.wins / self.games

    @property
    def match_rate(self) -> float:
        """Puntos por partido completo: victoria 1, empate 0.5, derrota 0."""
        return self.match_points / self.matches


@dataclass
class League:
    max_size: int = 50
    pfsp_power: float = 2.0
    k_elo: float = 16.0
    learner_elo: float = 1000.0
    members: list[Member] = field(default_factory=list)
    scripted_elo: float = 1000.0
    recent_weight: float = 0.25
    match_pfsp: bool = False
    mixture: tuple[float, float, float, float] = (0.2, 0.4, 0.2, 0.2)

    def add_snapshot(self, model: ActorCritic, name: str, *, protected=False, steps=0) -> None:
        if self.match_pfsp and len(self.members) >= self.max_size and all(m.protected for m in self.members):
            raise ValueError("La liga está llena de referencias protegidas; no se eliminan automáticamente")
        m = copy.deepcopy(model).eval()
        for p in m.parameters():
            p.requires_grad_(False)
        self.members.append(Member(name, m, self.learner_elo, protected=bool(protected), snapshot_steps=int(steps)))
        if len(self.members) > self.max_size:
            # descartar el más fácil (mayor winrate), nunca el más reciente
            if self.match_pfsp:
                candidates = [i for i, x in enumerate(self.members[:-1]) if not x.protected]
                if not candidates:
                    candidates = [len(self.members) - 1] if not protected else []
                if not candidates:
                    self.members.pop()
                    raise ValueError("No hay snapshots no protegidos para liberar espacio")
                i = max(candidates, key=lambda index: self.members[index].match_rate)
            else:
                i = int(np.argmax([x.winrate for x in self.members[:-1]]))
            self.members.pop(i)

    def sample(self, k: int, rng: np.random.Generator) -> list[int]:
        if not self.members:
            return []
        if self.match_pfsp:
            return list(rng.choice(len(self.members), size=k, p=self.sampling_probabilities()))
        w = np.array([(1.0 - m.winrate) ** self.pfsp_power + 1e-3 for m in self.members])
        # siempre algo de peso extra al último snapshot
        w[-1] += w.sum() * self.recent_weight
        return list(rng.choice(len(self.members), size=k, p=w / w.sum()))

    def sampling_probabilities(self) -> np.ndarray:
        """Mezcla opt-in de anclas/intermedios/recientes/uniforme.

        Si un grupo no tiene miembros su masa vuelve al grupo uniforme, no a
        un rival arbitrario. La ruta legacy no consulta puntos por partido.
        """
        if not self.members:
            return np.zeros(0, dtype=np.float64)
        mix = np.asarray(self.mixture, dtype=np.float64)
        if mix.shape != (4,) or not np.all(np.isfinite(mix)) or np.any(mix < 0) or not np.isclose(mix.sum(), 1):
            raise ValueError("league.mixture debe sumar 1: referencias/intermedios/recientes/uniforme")
        size = len(self.members)
        uniform = np.full(size, 1 / size, dtype=np.float64)
        anchors = np.array([m.protected for m in self.members], dtype=bool)
        intermediate = np.array([max(0., 1 - 2 * abs(m.match_rate - .5)) ** self.pfsp_power + 1e-3
                                 if not m.protected else 0. for m in self.members])
        recent = np.zeros(size)
        unprotected = [i for i, m in enumerate(self.members) if not m.protected]
        count = max(1, int(np.ceil(len(unprotected) * .2)))
        for index in sorted(unprotected, key=lambda i: (self.members[i].snapshot_steps, i))[-count:]:
            recent[index] = 1
        groups = (anchors.astype(float), intermediate, recent, uniform)
        probabilities = sum(weight * (values / values.sum() if values.sum() > 0 else uniform)
                            for weight, values in zip(mix, groups))
        return probabilities / probabilities.sum()

    def record_match(self, idx: int | None, score: float) -> None:
        """Sólo partidos completos, nunca goles ni ejercicios del currículo."""
        if score not in (0., .5, 1.):
            raise ValueError("Un resultado de partido debe ser 0, 0.5 o 1")
        if not self.match_pfsp or idx is None:
            return
        member = self.members[idx]
        member.match_points += float(score)
        member.matches += 1

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
