"""Controlador de red RS4-Z (obs v2) para el ejecutor de partidos, baterías y la liga."""
from __future__ import annotations

import numpy as np
import torch

from env.rs4z.obs_v2 import OBS_DIM, observe


class NetController:
    def __init__(self, model, device="cpu", greedy=True, temperature=1.0, seed=0):
        self.model = model.to(device).eval()
        self.device = device
        self.greedy = greedy
        self.temperature = temperature
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(seed)
        self._obs = None

    def sync(self, env, rows):
        pass

    def push(self, env):
        pass

    @torch.no_grad()
    def act(self, env, ctrl, out):
        if self._obs is None or self._obs.shape[0] != env.N:
            self._obs = np.empty((env.N, 8, OBS_DIM), dtype=np.float32)
        observe(env, self._obs)
        ctrl = np.asarray(ctrl, dtype=bool)
        rows = np.nonzero(ctrl)
        if len(rows[0]) == 0:
            return out
        x = torch.from_numpy(self._obs[rows]).to(self.device)
        logits = self.model.logits(x)
        if self.greedy:
            a = logits.argmax(dim=-1)
        else:
            probs = torch.softmax(logits / self.temperature, dim=-1)
            a = torch.multinomial(probs, 1, generator=self.gen).squeeze(-1)
        out[rows] = a.cpu().numpy()
        return out


def load_model(path, device="cpu"):
    from train.rs4z.model import ActorCritic
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt.get("model_config", {})
    model = ActorCritic(hidden=cfg.get("hidden", 256), enc_hidden=cfg.get("enc_hidden", 128),
                        ent_hidden=cfg.get("ent_hidden", 64))
    model.load_state_dict(ckpt["model"])
    return model
