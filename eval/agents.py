"""Agentes intercambiables para evaluar: modelos entrenados, bot scripteado o azar."""
from __future__ import annotations

import numpy as np
import torch
import weakref

from bots.scripted import scripted_actions
from train.model import load_model


class ModelAgent:
    def __init__(self, path: str, greedy: bool = False, device="cpu"):
        self.device = torch.device(device)
        self.model = load_model(path).to(self.device)
        self.public_config = (torch.load(path, map_location="cpu", weights_only=False).get("env", {}).get("public_signal_config", {})
                              if getattr(self.model, "public_signals_version", 0) else {})
        self.greedy = greedy
        self.name = path
        self._states = weakref.WeakKeyDictionary()

    def reset(self, env, done=None):
        if getattr(self.model, "public_signals_version", 0):
            env.configure_public_signals(self.public_config)
        if not getattr(self.model, "is_recurrent", False):
            return
        if done is None or env not in self._states:
            self._states[env] = (self.model.initial_state(env.N * env.P).reshape(env.N, env.P, -1),
                                torch.full((env.N, env.P), self.model.n_actions, dtype=torch.long, device=self.device))
        else:
            memory, previous = self._states[env]
            memory[done] = 0
            previous[done] = self.model.n_actions

    @torch.no_grad()
    def __call__(self, env, obs: np.ndarray, players: np.ndarray) -> np.ndarray:
        selected = obs[:, players].copy()
        if getattr(self.model, "public_signals_version", 0):
            if env.enable_public_signals():
                env.configure_public_signals(self.public_config)
            selected[..., 56:71] = env.public_features()[:, players]
        elif getattr(env, "_public_signals", None) is not None:
            selected[..., 56:71] = 0  # old full/masked references keep their input contract
        o = torch.from_numpy(selected.reshape(-1, obs.shape[-1])).to(self.device)
        if getattr(self.model, "is_recurrent", False):
            if env not in self._states:
                self.reset(env)
            memory, previous = self._states[env]
            logits, _, updated = self.model.step(o, memory[:, players].reshape(-1, self.model.memory_size),
                                                previous[:, players].reshape(-1))
            memory[:, players] = updated.reshape(env.N, len(players), -1)
        else:
            logits = self.model.logits(o)
        a = logits.argmax(-1) if self.greedy else torch.distributions.Categorical(logits=logits).sample()
        if getattr(self.model, "is_recurrent", False):
            previous[:, players] = a.reshape(env.N, len(players))
        return a.cpu().numpy().reshape(obs.shape[0], len(players))

    def record_executed(self, env, actions):
        if getattr(self.model, "is_recurrent", False) and env in self._states:
            _, previous = self._states[env]
            previous.copy_(torch.as_tensor(actions, dtype=torch.long, device=previous.device))


class ScriptedAgent:
    def __init__(self, eps: float = 0.0, seed: int = 0, policy: str = "r2", style: int = -1):
        self.eps = eps
        self.policy = policy
        self.style = style
        self.rng = np.random.default_rng(seed)
        self.name = f"scripted(policy={policy},style={style},eps={eps})"

    def __call__(self, env, obs, players):
        return scripted_actions(env, players, self.eps, self.rng, policy=self.policy, style=self.style)


def reset_agents(agents, env, done=None):
    """Reset por entorno; un mismo agente puede controlar ambos colores sin mezclar estados."""
    seen = set()
    for agent in agents:
        if id(agent) not in seen and hasattr(agent, "reset"):
            agent.reset(env, done)
        seen.add(id(agent))


class RandomAgent:
    name = "random"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def __call__(self, env, obs, players):
        return self.rng.integers(0, 18, (obs.shape[0], len(players)))


def env_config(specs, stadium=None, n_per_team=None, task=None) -> dict:
    """Config del entorno (estadio, tamaño de equipo, powershot...) tomada del primer checkpoint
    de la lista; `stadium` / `n_per_team` explícitos la pisan. `task` (nombre de train/tasks.yaml)
    fija estadio, formato y reglas de una vez (modelos multi-tarea, obs "universal")."""
    cfg = {"stadium": "classic", "n_per_team": 1, "frame_skip": 3, "powershot": False, "out_of_bounds": False}
    for sp in specs:
        if sp.endswith(".pt"):
            ck = torch.load(sp, map_location="cpu", weights_only=False)
            cfg.update(ck.get("env", {}))
            break
    if cfg.get("obs_layout") == "universal" and task is None and stadium is None:
        task = (cfg.get("tasks") or ["classic_1v1"])[0]
    if task is not None:
        from env.tasks import load_catalog
        t = load_catalog()[task]
        cfg.update(stadium=t.stadium, n_per_team=t.n_per_team, powershot=t.powershot, out_of_bounds=t.out_of_bounds,
                   rules=t.rules)
        return cfg
    if stadium is not None:
        if stadium != cfg["stadium"] and stadium.startswith("x6"):
            cfg.update(powershot=True, out_of_bounds=True)  # mecánicas de la sala Pegeche
        cfg["stadium"] = stadium
    if n_per_team is not None:
        cfg["n_per_team"] = n_per_team
    return cfg


def env_kwargs(cfg: dict) -> dict:
    return {"powershot": cfg.get("powershot", False), "out_of_bounds": cfg.get("out_of_bounds", False),
            "obs_layout": cfg.get("obs_layout", "flat"),  # universal: max_entities = 2T-1 (el modelo acepta cualquiera)
            "rules": cfg.get("rules") if cfg.get("obs_layout") == "universal" else None,
            "kickoff_timeout": cfg.get("kickoff_timeout", 180)}


def make_agent(spec: str, greedy: bool = False, device="cpu"):
    if spec == "scripted":
        return ScriptedAgent()
    if spec.startswith("scripted:"):
        parts = spec.split(":")
        if parts[1] in ("r2", "r3"):
            style = int(parts[2]) if len(parts) > 2 else -1
            return ScriptedAgent(policy=parts[1], style=style)
        return ScriptedAgent(float(parts[1]))
    if spec == "random":
        return RandomAgent()
    return ModelAgent(spec, greedy, device=device)
