"""Liga RS4-Z: instantáneas del aprendiz como rivales congelados, muestreo PFSP y exploiters.

PFSP (AlphaStar): peso (1 − p)² sobre la tasa de victoria p del aprendiz contra cada miembro, así que
se juega más contra los que todavía le ganan. Los exploiters (entrenados aparte contra el principal
congelado) se marcan y se muestrean en su propia fracción.

Costo por paso: cada rival congelado distinto en juego es una pasada de red por decisión. Con más de
`active_size` miembros, los partidos nuevos eligen dentro de un grupo activo (sorteado por PFSP sin
reposición) que se renueva cada `refresh_every` partidos; todas las redes quedan cargadas en el dispositivo
(0,4M parámetros cada una), sin releer del disco.

Los índices de miembro son estables: los partidos en curso guardan el índice de su rival, así que un
miembro nunca se borra de la lista; al superar `max_members` el más fácil se marca retirado y deja de
muestrearse.
"""
from __future__ import annotations

from pathlib import Path

import copy

import numpy as np
import torch
from torch import nn
from torch.func import functional_call, stack_module_state, vmap

from .model import ActorCritic

MAX_LOADED = 128


class _Actor(nn.Module):
    """Sólo el actor de un miembro (obs → logits), para apilar los parámetros de varios miembros."""

    def __init__(self, m):
        super().__init__()
        self.actor_enc, self.pi = m.actor_enc, m.pi

    def forward(self, x):
        return self.pi(self.actor_enc(x))


class League:
    def __init__(self, directory, device="cpu", max_members=48, active_size=8, refresh_every=1000):
        self.dir = Path(directory).resolve()
        self.dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.max_members = max_members
        self.active_size = active_size
        self.refresh_every = refresh_every
        self._active = {False: [], True: []}
        self._draws = {False: 0, True: 0}
        self.members = []          # dict(path, samples, wins, games, exploiter[, retired])
        self._loaded = {}          # índice → modelo en el dispositivo
        self._stack = None         # parámetros del actor apilados (miembro, ...) para una sola llamada vmap
        self._stack_pos = {}
        self._base = None

    def __len__(self):
        return len(self.members)

    def add(self, model, samples, exploiter=False):
        path = self.dir / f"member_{len(self.members):03d}_{int(samples)}.pt"
        torch.save(dict(model=model.state_dict(), model_config=model.config, samples=samples), path)
        self.members.append(dict(path=str(path), samples=int(samples), wins=1.0, games=2.0, exploiter=exploiter))
        self._retire()

    def _retire(self):
        live = [i for i, m in enumerate(self.members) if not m.get("retired")]
        if len(live) > self.max_members:
            # retirar el más fácil (mayor tasa de victoria del aprendiz) que no sea exploiter
            rates = [self.members[i]["wins"] / self.members[i]["games"] if not self.members[i]["exploiter"] else -1
                     for i in live]
            self.members[live[int(np.argmax(rates))]]["retired"] = True

    def _pfsp(self, idx):
        p = np.array([self.members[i]["wins"] / self.members[i]["games"] for i in idx])
        w = (1.0 - p) ** 2 + 1e-3
        return w / w.sum()

    def sample(self, rng, exploiter=False):
        live = [(i, m) for i, m in enumerate(self.members) if not m.get("retired")]
        idx = [i for i, m in live if m["exploiter"] == exploiter]
        flag = exploiter
        if not idx:
            idx = [i for i, m in live if not m["exploiter"]]
            flag = False
            if not idx:
                return None
        if len(idx) <= self.active_size:
            return int(idx[rng.choice(len(idx), p=self._pfsp(idx))])
        pool = [i for i in self._active[flag] if i in idx]
        if not pool or self._draws[flag] % self.refresh_every == 0:
            pool = [int(i) for i in rng.choice(idx, size=self.active_size, replace=False, p=self._pfsp(idx))]
            self._active[flag] = pool
        self._draws[flag] += 1
        return int(pool[rng.integers(len(pool))])

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
    def _stacked(self, ids):
        """Parámetros del actor de los miembros `ids`, apilados en ese orden (se reconstruye sólo si
        aparece un miembro nuevo)."""
        if any(i not in self._stack_pos for i in ids):
            all_ids = sorted(set(self._stack_pos) | set(int(i) for i in ids))
            models = [_Actor(self._model(i)) for i in all_ids]
            params, _ = stack_module_state(models)
            self._stack = params
            self._stack_pos = {i: k for k, i in enumerate(all_ids)}
            self._base = copy.deepcopy(models[0]).to("meta")
        pos = torch.tensor([self._stack_pos[int(i)] for i in ids], device=self.device)
        return {k: v.index_select(0, pos) for k, v in self._stack.items()}

    @torch.no_grad()
    def act(self, env, obs, mask, member_of_row, out):
        """Acciones de los miembros congelados en `mask`: todos los miembros en una sola llamada (vmap sobre
        los parámetros apilados), una transferencia y una sincronización por paso. Antes era una pasada de
        red por miembro: ~0,7 s por iteración con 8 rivales activos."""
        n_idx, p_idx = np.nonzero(mask)
        mem = member_of_row[n_idx]
        keep = mem >= 0
        n_idx, p_idx, mem = n_idx[keep], p_idx[keep], mem[keep]
        if len(mem) == 0:
            return out
        order = np.argsort(mem, kind="stable")
        ms = mem[order]
        uniq, start, counts = np.unique(ms, return_index=True, return_counts=True)
        grp = np.repeat(np.arange(len(uniq)), counts)
        slot = np.arange(len(ms)) - start[grp]
        x = torch.from_numpy(obs[n_idx[order], p_idx[order]]).to(self.device)
        g = torch.from_numpy(grp).to(self.device)
        sl = torch.from_numpy(slot).to(self.device)
        X = torch.zeros(len(uniq), int(counts.max()), x.shape[-1], device=self.device, dtype=x.dtype)
        X[g, sl] = x
        params = self._stacked(uniq)
        logits = vmap(lambda prm, xb: functional_call(self._base, prm, (xb,)))(params, X)[g, sl].float()
        # muestreo Gumbel-max: misma distribución que Categorical, sin validaciones que sincronizan
        u = torch.rand_like(logits).clamp_(1e-10, 1.0)
        a = (logits - torch.log(-torch.log(u))).argmax(dim=-1).cpu().numpy()
        out[n_idx[order], p_idx[order]] = a
        return out

    def merge_exploiters(self, path):
        """Sumar los exploiters entrenados aparte (escritos en exploiters.json por su corrida)."""
        import json
        path = Path(path)
        if not path.exists():
            return 0
        known = {m["path"] for m in self.members}
        added = 0
        for m in json.loads(path.read_text(encoding="utf-8")).get("members", []):
            if m.get("exploiter") and m["path"] not in known and Path(m["path"]).exists():
                self.members.append(dict(m, wins=1.0, games=2.0))
                added += 1
        if added:
            self._retire()
        return added

    def state(self):
        return dict(members=self.members)

    def load_state(self, state):
        self.members = list(state.get("members", []))
        self._loaded = {}
        self._stack, self._stack_pos, self._base = None, {}, None
