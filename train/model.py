"""Red actor-crítico. La normalización de observaciones vive dentro del modelo (buffers),
así el ONNX exportado recibe observaciones crudas."""
from __future__ import annotations

import torch
from torch import nn


class RunningNorm(nn.Module):
    def __init__(self, dim: int, clip: float = 10.0):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(1e-4))
        self.clip = clip

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        bm, bv, bc = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        delta = bm - self.mean
        tot = self.count + bc
        self.mean += delta * bc / tot
        self.var = (self.var * self.count + bv * bc + delta.pow(2) * self.count * bc / tot) / tot
        self.count = tot

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ((x - self.mean) / torch.sqrt(self.var + 1e-8)).clamp(-self.clip, self.clip)


def _mlp(i, h, layers):
    mods, d = [], i
    for _ in range(layers):
        mods += [nn.Linear(d, h), nn.ReLU()]
        d = h
    return nn.Sequential(*mods)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int = 18, hidden: int = 256, layers: int = 3):
        super().__init__()
        self.obs_dim, self.n_actions, self.hidden, self.layers = obs_dim, n_actions, hidden, layers
        self.norm = RunningNorm(obs_dim)
        self.pi_body = _mlp(obs_dim, hidden, layers)
        self.v_body = _mlp(obs_dim, hidden, layers)
        self.pi = nn.Linear(hidden, n_actions)
        self.v = nn.Linear(hidden, 1)
        nn.init.orthogonal_(self.pi.weight, 0.01)
        nn.init.zeros_(self.pi.bias)
        nn.init.orthogonal_(self.v.weight, 1.0)

    def config(self) -> dict:
        return dict(obs_dim=self.obs_dim, n_actions=self.n_actions, hidden=self.hidden, layers=self.layers)

    @torch.no_grad()
    def update_norm(self, obs: torch.Tensor) -> None:
        self.norm.update(obs)

    def forward(self, obs: torch.Tensor):
        x = self.norm(obs)
        return self.pi(self.pi_body(x)), self.v(self.v_body(x)).squeeze(-1)

    def logits(self, obs: torch.Tensor) -> torch.Tensor:
        return self.pi(self.pi_body(self.norm(obs)))


class PolicyOnly(nn.Module):
    """Para exportar a ONNX: obs crudas -> logits."""

    def __init__(self, ac: ActorCritic):
        super().__init__()
        self.ac = ac

    def forward(self, obs):
        return self.ac.logits(obs)


def load_model(path: str, map_location="cpu") -> ActorCritic:
    ck = torch.load(path, map_location=map_location, weights_only=False)
    m = build_model(ck["model_config"])
    m.load_state_dict(ck["model"])
    m.eval()
    return m


class EntityActorCritic(nn.Module):
    """Actor-crítico que sirve para CUALQUIER tamaño de equipo (currículo 1v1 -> 2v2 -> 3v3 -> 6v6).

    La obs (layout "entities" de env/haxball_env.py) es [propio (self_dim) | entidades x ent_dim],
    con compañeros primero y rivales después (T-1 y T). Cada entidad pasa por la MISMA red chica;
    compañeros y rivales se resumen por separado con promedio y máximo, así el tamaño de la
    representación no depende de cuántos jugadores haya y los pesos se transfieren entre etapas.
    """

    def __init__(self, self_dim: int, ent_dim: int = 7, n_actions: int = 18, hidden: int = 256,
                 layers: int = 2, ent_hidden: int = 64):
        super().__init__()
        self.self_dim, self.ent_dim, self.n_actions = self_dim, ent_dim, n_actions
        self.hidden, self.layers, self.ent_hidden = hidden, layers, ent_hidden
        self.self_norm = RunningNorm(self_dim)
        self.ent_norm = RunningNorm(ent_dim)
        self.ent_enc = _mlp(ent_dim, ent_hidden, 2)
        joint = self_dim + 4 * ent_hidden  # (media, máximo) x (compañeros, rivales)
        self.pi_body = _mlp(joint, hidden, layers)
        self.v_body = _mlp(joint, hidden, layers)
        self.pi = nn.Linear(hidden, n_actions)
        self.v = nn.Linear(hidden, 1)
        nn.init.orthogonal_(self.pi.weight, 0.01)
        nn.init.zeros_(self.pi.bias)
        nn.init.orthogonal_(self.v.weight, 1.0)

    def config(self) -> dict:
        return dict(type="entity", self_dim=self.self_dim, ent_dim=self.ent_dim, n_actions=self.n_actions,
                    hidden=self.hidden, layers=self.layers, ent_hidden=self.ent_hidden)

    def _split(self, obs):
        s = obs[:, : self.self_dim]
        e = obs[:, self.self_dim:].reshape(obs.shape[0], -1, self.ent_dim)
        return s, e

    @torch.no_grad()
    def update_norm(self, obs: torch.Tensor) -> None:
        s, e = self._split(obs)
        self.self_norm.update(s)
        self.ent_norm.update(e.reshape(-1, self.ent_dim))

    def _features(self, obs):
        s, e = self._split(obs)
        n_ent = e.shape[1]
        T = (n_ent + 1) // 2
        h = self.ent_enc(self.ent_norm(e))  # (B, n_ent, H)
        B, H = obs.shape[0], self.ent_hidden
        if T > 1:
            m = h[:, : T - 1]
            mates = torch.cat([m.mean(1), m.amax(1)], dim=-1)
        else:
            mates = obs.new_zeros(B, 2 * H)
        o = h[:, T - 1:]
        opps = torch.cat([o.mean(1), o.amax(1)], dim=-1)
        return torch.cat([self.self_norm(s), mates, opps], dim=-1)

    def forward(self, obs: torch.Tensor):
        x = self._features(obs)
        return self.pi(self.pi_body(x)), self.v(self.v_body(x)).squeeze(-1)

    def logits(self, obs: torch.Tensor) -> torch.Tensor:
        return self.pi(self.pi_body(self._features(obs)))


class SetActorCritic(nn.Module):
    """Actor-crítico para la obs "universal" (env/haxball_env.py): cualquier mapa y cualquier formato.

    obs = [propio (self_dim) | E entidades x ent_dim], con entidad = [presente, es_rival, ...].
    Cada entidad pasa por la misma red chica; compañeros y rivales se resumen por separado con promedio
    y máximo SOBRE LAS PRESENTES (máscara), así el relleno no cambia la salida y un mismo lote puede
    mezclar 1v1 con 11v11. Con pooling="attention" se suma una capa de atención (el bloque propio consulta
    a las entidades), útil para equipos grandes.
    """

    def __init__(self, self_dim: int, ent_dim: int = 8, n_actions: int = 18, hidden: int = 256,
                 layers: int = 2, ent_hidden: int = 64, pooling: str = "meanmax", ent_layers: int = 2,
                 rule_observation: str = "full"):
        super().__init__()
        self.self_dim, self.ent_dim, self.n_actions = self_dim, ent_dim, n_actions
        self.hidden, self.layers, self.ent_hidden, self.pooling = hidden, layers, ent_hidden, pooling
        self.ent_layers = ent_layers
        if rule_observation not in ("full", "masked"):
            raise ValueError(f"rule_observation desconocida: {rule_observation}")
        # full preserva checkpoints y callers antiguos (incluido BC). Los nuevos runs
        # de RL usan masked: el cliente no puede observar el estado privado del script.
        self.rule_observation = rule_observation
        if rule_observation == "masked":
            from env.haxball_env import U_SELF_DIM
            if self_dim != U_SELF_DIM:
                raise ValueError("masked requiere el layout universal actual")
        self.self_norm = RunningNorm(self_dim)
        self.ent_norm = RunningNorm(ent_dim - 2)  # no se normalizan las banderas presente/es_rival
        # el encoder de entidades corre una vez por jugador visible: es lo que más cuesta en CPU
        self.ent_enc = _mlp(ent_dim, ent_hidden, ent_layers)
        joint = self_dim + 4 * ent_hidden
        if pooling == "attention":
            self.q = nn.Linear(self_dim, ent_hidden)
            self.attn = nn.MultiheadAttention(ent_hidden, num_heads=4, batch_first=True)
            joint += ent_hidden
        self.pi_body = _mlp(joint, hidden, layers)
        self.v_body = _mlp(joint, hidden, layers)
        self.pi = nn.Linear(hidden, n_actions)
        self.v = nn.Linear(hidden, 1)
        nn.init.orthogonal_(self.pi.weight, 0.01)
        nn.init.zeros_(self.pi.bias)
        nn.init.orthogonal_(self.v.weight, 1.0)

    def config(self) -> dict:
        return dict(type="set", self_dim=self.self_dim, ent_dim=self.ent_dim, n_actions=self.n_actions,
                    hidden=self.hidden, layers=self.layers, ent_hidden=self.ent_hidden, pooling=self.pooling,
                    ent_layers=self.ent_layers, rule_observation=self.rule_observation)

    def _split(self, obs):
        s = obs[:, : self.self_dim]
        e = obs[:, self.self_dim:].reshape(obs.shape[0], -1, self.ent_dim)
        if self.rule_observation == "masked":
            # Dentro del modelo: idéntico en PPO, evaluación y ONNX. No se cambian
            # el ancho de la obs, los datasets ni la física/reglas del entorno.
            from env.pegeche import N_RULE_FEATS
            s = torch.cat([s[:, :-N_RULE_FEATS], torch.zeros_like(s[:, -N_RULE_FEATS:])], dim=-1)
        return s, e

    @torch.no_grad()
    def update_norm(self, obs: torch.Tensor) -> None:
        s, e = self._split(obs)
        self.self_norm.update(s)
        present = e[..., 0] > 0.5
        if present.any():
            self.ent_norm.update(e[present][:, 2:])

    @staticmethod
    def _pool(h, m):
        """promedio y máximo de h (B,E,H) sobre las entidades con m (B,E) = True; ceros si no hay ninguna."""
        mf = m.unsqueeze(-1).to(h.dtype)
        cnt = mf.sum(1)
        mean = (h * mf).sum(1) / cnt.clamp(min=1.0)
        mx = h.masked_fill(~m.unsqueeze(-1), -1e9).amax(1)
        mx = torch.where(cnt > 0, mx, torch.zeros_like(mx))
        return torch.cat([mean, mx], dim=-1)

    def _features(self, obs):
        s, e = self._split(obs)
        present = e[..., 0] > 0.5
        rival = e[..., 1] > 0.5
        e_in = torch.cat([e[..., :2], self.ent_norm(e[..., 2:])], dim=-1) * present.unsqueeze(-1)
        h = self.ent_enc(e_in)
        sn = self.self_norm(s)
        feats = [sn, self._pool(h, present & ~rival), self._pool(h, present & rival)]
        if self.pooling == "attention":
            q = self.q(sn).unsqueeze(1)
            # una fila sin entidades presentes daría NaN: se le deja pasar al menos la primera (relleno)
            key_pad = ~present
            key_pad[:, 0] = key_pad[:, 0] & present.any(1)
            a, _ = self.attn(q, h, h, key_padding_mask=key_pad)
            feats.append(a.squeeze(1) * present.any(1, keepdim=True))
        return torch.cat(feats, dim=-1)

    def forward(self, obs: torch.Tensor):
        x = self._features(obs)
        return self.pi(self.pi_body(x)), self.v(self.v_body(x)).squeeze(-1)

    def logits(self, obs: torch.Tensor) -> torch.Tensor:
        return self.pi(self.pi_body(self._features(obs)))


def build_model(config: dict) -> nn.Module:
    cfg = dict(config)
    kind = cfg.pop("type", "mlp")
    if kind == "recurrent_set":
        from .recurrent_model import RecurrentSetActorCritic
        return RecurrentSetActorCritic(**cfg)
    if kind == "entity":
        return EntityActorCritic(**cfg)
    if kind == "set":
        return SetActorCritic(**cfg)
    return ActorCritic(**cfg)
