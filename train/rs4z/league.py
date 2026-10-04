"""Liga RS4-Z: instantáneas del aprendiz como rivales congelados, muestreo PFSP y exploiters.

PFSP (AlphaStar): peso (1 − p)² sobre la tasa de victoria p del aprendiz contra cada miembro, así que
se juega más contra los que todavía le ganan. Los exploiters (entrenados aparte contra el principal
congelado) se marcan y se muestrean en su propia fracción.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .model import ActorCritic

MAX_LOADED = 4


class League:
    def __init__(self, directory, device="cpu", max_members=48):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.max_members = max_members
        self.members = []          # dict(path, samples, wins, games, exploiter)
        self._loaded = {}          # índice → modelo en el dispositivo

    def __len__(self):
        return len(self.members)

    def add(self, model, samples, exploiter=False):
        path = self.dir / f"member_{len(self.members):03d}_{int(samples)}.pt"
        torch.save(dict(model=model.state_dict(), model_config=model.config, samples=samples), path)
        self.members.append(dict(path=str(path), samples=int(samples), wins=1.0, games=2.0, exploiter=exploiter))
        if len(self.members) > self.max_members:
            # retirar el más fácil (mayor tasa de victoria del aprendiz) que no sea exploiter
            rates = [m["wins"] / m["games"] if not m["exploiter"] else -1 for m in self.members]
            drop = int(np.argmax(rates))
            self.members.pop(drop)
            self._loaded = {}

    def sample(self, rng, exploiter=False):
        idx = [i for i, m in enumerate(self.members) if m["exploiter"] == exploiter]
        if not idx:
            idx = [i for i, m in enumerate(self.members) if not m["exploiter"]]
            if not idx:
                return None
        p = np.array([self.members[i]["wins"] / self.members[i]["games"] for i in idx])
        w = (1.0 - p) ** 2 + 1e-3
        return int(idx[rng.choice(len(idx), p=w / w.sum())])

    def record(self, member, learner_points):
        if 0 <= member < len(self.members):
            self.members[member]["wins"] += float(learner_points)
            self.members[member]["games"] += 1.0

    def _model(self, i):
        if i not in self._loaded:
            if len(self._loaded) >= MAX_LOADED:
                self._loaded.pop(next(iter(self._loaded)))
            s = torch.load(self.members[i]["path"], map_location=self.device, weights_only=False)
            cfg = s.get("model_config", {})
            m = ActorCritic(hidden=cfg.get("hidden", 256), enc_hidden=cfg.get("enc_hidden", 128),
                            ent_hidden=cfg.get("ent_hidden", 64)).to(self.device)
            m.load_state_dict(s["model"])
            m.eval()
            self._loaded[i] = m
        return self._loaded[i]

    @torch.no_grad()
    def act(self, env, obs, mask, member_of_row, out):
        for i in np.unique(member_of_row[mask.any(axis=1)]):
            if i < 0:
                continue
            sel = mask & (member_of_row == i)[:, None]
            rows = np.nonzero(sel)
            x = torch.from_numpy(obs[rows]).to(self.device)
            logits = self._model(int(i)).logits(x)
            a = torch.distributions.Categorical(logits=logits).sample()
            out[rows] = a.cpu().numpy()
        return out

    def state(self):
        return dict(members=self.members)

    def load_state(self, state):
        self.members = list(state.get("members", []))
        self._loaded = {}
