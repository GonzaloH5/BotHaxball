"""PPO por secuencias, con memoria independiente por jugador y por política rival."""
from __future__ import annotations

import time
import numpy as np
import torch
import yaml

from bots.scripted import scripted_actions
from .multitask import MultiTrainer, SELF, POOL, SCRIPTED
from .ppo_selfplay import lerp


def pack_sequences(buffer, learner, length):
    """Preserva cada trayectoria. Sólo se baraja el eje de secuencias, nunca los ticks."""
    steps, n, players = buffer["act"].shape
    if length < 1 or steps % length:
        raise ValueError("rollout_len debe ser múltiplo de sequence_length")
    mask = learner.reshape(-1)
    count = int(mask.sum())
    chunks = steps // length
    out = {}
    for key in ("obs", "act", "previous_action", "episode_start", "logp", "adv", "ret"):
        tail = buffer[key].shape[3:]
        a = buffer[key].reshape((steps, n * players) + tail)[:, mask]
        out[key] = a.reshape((chunks, length, count) + tail).swapaxes(0, 1).reshape((length, chunks * count) + tail)
    states = buffer["memory"].reshape(steps, n * players, -1)[:, mask]
    out["initial_memory"] = states[::length].reshape(chunks * count, -1)
    return out


class RecurrentTrainer(MultiTrainer):
    def __init__(self, cfg, run, resume, init_from=None):
        length = cfg["ppo"].get("sequence_length", 32)
        if length < 1 or cfg["ppo"]["rollout_len"] % length:
            raise ValueError("rollout_len debe ser múltiplo de sequence_length")
        if cfg["model"].get("type") != "recurrent_set":
            raise ValueError("RecurrentTrainer requiere model.type=recurrent_set")
        # El CLI suele reanudar con el YAML base: recuperar la referencia de BC
        # guardada evita desactivar silenciosamente la regularización al reanudar.
        if resume and not init_from and "bc_reference" not in cfg:
            from . import multitask
            saved_config = multitask.ROOT / "runs" / run / "config.yaml"
            if saved_config.exists():
                reference = yaml.safe_load(saved_config.read_text(encoding="utf-8")).get("bc_reference")
                if reference:
                    cfg = {**cfg, "bc_reference": reference}
        super().__init__(cfg, run, resume, init_from)

    def build_envs(self):
        super().build_envs()
        for s in self.slots:
            s.memory = np.zeros((s.N, s.P, self.model.memory_size), np.float32)
            s.opponent_memory = np.zeros_like(s.memory)
            s.previous_action = np.full((s.N, s.P), self.model.n_actions, np.int64)
            s.episode_start = np.ones((s.N, s.P), bool)
            s.opponent_keys = None

    def assign_modes(self, s):
        super().assign_modes(s)
        # Comparar nombres, no índices: la liga puede expulsar un snapshot y moverlos.
        keys = [self.league.members[s.opp_id[e]].name if s.modes[e] == POOL else
                ("learner" if s.modes[e] == SELF else f"scripted:{s.scripted_eps}") for e in range(s.N)]
        changed = np.ones(s.N, bool) if s.opponent_keys is None else np.array([a != b for a, b in zip(keys, s.opponent_keys)])
        s.memory[changed, s.T:] = 0
        s.opponent_memory[changed, s.T:] = 0
        s.previous_action[changed, s.T:] = self.model.n_actions
        s.episode_start[changed, s.T:] = True
        s.opponent_keys = keys

    @torch.no_grad()
    def act(self, obs_list):
        out = []
        for s, obs in zip(self.slots, obs_list):
            def tensor(a):
                return torch.from_numpy(a).to(self.device)
            logits, value, memory = self.model.step(tensor(obs.reshape(-1, self.obs_dim)),
                tensor(s.memory.reshape(-1, self.model.memory_size)), tensor(s.previous_action.reshape(-1)),
                tensor(s.episode_start.reshape(-1)))
            dist = torch.distributions.Categorical(logits=logits)
            actions = dist.sample()
            logp = dist.log_prob(actions).cpu().numpy().reshape(s.N, s.P)
            acts = actions.cpu().numpy().reshape(s.N, s.P)
            s.memory = memory.cpu().numpy().reshape(s.N, s.P, -1)
            blue = slice(s.T, s.P)
            for oid in np.unique(s.opp_id[s.opp_id >= 0]):
                rows = np.where(s.opp_id == oid)[0]
                opponent = self.league.members[oid].model
                lg, _, hm = opponent.step(tensor(obs[rows, blue].reshape(-1, self.obs_dim)),
                    tensor(s.opponent_memory[rows, blue].reshape(-1, opponent.memory_size)),
                    tensor(s.previous_action[rows, blue].reshape(-1)),
                    tensor(s.episode_start[rows, blue].reshape(-1)))
                acts[rows, blue] = torch.distributions.Categorical(logits=lg).sample().cpu().numpy().reshape(len(rows), s.T)
                s.opponent_memory[rows, blue] = hm.cpu().numpy().reshape(len(rows), s.T, -1)
            scripted = np.where(s.modes == SCRIPTED)[0]
            if len(scripted):
                acts[scripted, blue] = scripted_actions(s.env, np.arange(s.T, s.P), s.scripted_eps, self.rng)[scripted]
            out.append((acts, logp, value.cpu().numpy().reshape(s.N, s.P)))
        return out

    @torch.no_grad()
    def state_values(self, obs, memory, previous_action):
        # Consulta sin avanzar la memoria viva: usada para bootstrap, incluidos timeouts.
        _, value, _ = self.model.step(torch.from_numpy(obs.reshape(-1, self.obs_dim)).to(self.device),
            torch.from_numpy(memory.reshape(-1, self.model.memory_size)).to(self.device),
            torch.from_numpy(previous_action.reshape(-1)).to(self.device))
        return value.cpu().numpy().reshape(obs.shape[:2])

    def iterate(self):
        cfg, p = self.cfg, self.cfg["ppo"]
        length, width = p["rollout_len"], self.obs_dim
        if not hasattr(self, "_obs"):
            self._obs = [s.env.reset() for s in self.slots]
        fraction = self.steps / p["total_steps"]
        lr = lerp(p["lr"], p["lr_final"], fraction)
        for group in self.opt.param_groups:
            group["lr"] = lr
        entropy_coef = lerp(p["ent_coef"], p["ent_coef_final"], fraction)
        self.rcfg.shaping_coef = max(0.0, 1.0 - self.steps / cfg["reward"]["shaping_decay_steps"])
        for s in self.slots:
            self.assign_modes(s)
            s.buf = {key: np.zeros((length, s.N, s.P) + shape, dtype) for key, shape, dtype in
                     (("obs", (width,), np.float32), ("act", (), np.int64), ("previous_action", (), np.int64),
                      ("episode_start", (), bool), ("memory", (self.model.memory_size,), np.float32),
                      ("logp", (), np.float32), ("val", (), np.float32), ("rew", (), np.float32))}
            s.buf["done"] = np.zeros((length, s.N), np.float32)
            s.pool_goals = {}
        started = time.time()
        obs = self._obs
        for t in range(length):
            for s, o in zip(self.slots, obs):
                s.buf["obs"][t] = o
                s.buf["memory"][t] = s.memory
                s.buf["previous_action"][t] = s.previous_action
                s.buf["episode_start"][t] = s.episode_start
            decisions = self.act(obs)
            next_obs = []
            for s, (acts, logp, values) in zip(self.slots, decisions):
                b = s.buf
                b["act"][t], b["logp"][t], b["val"][t] = acts, logp, values
                o2, reward, done, info = s.env.step(acts)
                # Memoria posterior a obs[t], acción realmente enviada. Antes de resetear.
                truncated = info["truncated"]
                if truncated.any():
                    reward[truncated] += p["gamma"] * self.state_values(info["final_obs"][truncated],
                        s.memory[truncated], acts[truncated])
                b["rew"][t], b["done"][t] = reward, done
                s.previous_action[:] = acts
                s.previous_action[done] = self.model.n_actions
                s.memory[done] = 0
                s.opponent_memory[done] = 0
                s.episode_start[:] = done[:, None]
                goal = info["goal"]
                for mode in (SELF, POOL, SCRIPTED):
                    selected = s.modes == mode
                    s.goals[mode][0] += int((goal[selected] == 1).sum())
                    s.goals[mode][1] += int((goal[selected] == -1).sum())
                for e in np.where((goal != 0) & (s.modes == POOL))[0]:
                    s.pool_goals.setdefault(s.opp_id[e], [0, 0])[0 if goal[e] == 1 else 1] += 1
                scripted = s.modes == SCRIPTED
                self.league.record(None, int((goal[scripted] == 1).sum()), int((goal[scripted] == -1).sum()))
                next_obs.append(o2)
            obs = next_obs
        self._obs = obs
        rollout_time = time.time() - started
        parts = []
        for s, o in zip(self.slots, obs):
            for oid, (won, lost) in s.pool_goals.items():
                self.league.record(int(oid), won, lost)
            b = s.buf
            last_value = self.state_values(o, s.memory, s.previous_action)
            advantage = np.zeros_like(b["rew"])
            gae = np.zeros((s.N, s.P), np.float32)
            for t in reversed(range(length)):
                next_value = last_value if t == length - 1 else b["val"][t + 1]
                alive = 1 - b["done"][t][:, None]
                delta = b["rew"][t] + p["gamma"] * next_value * alive - b["val"][t]
                gae = delta + p["gamma"] * p["gae_lambda"] * alive * gae
                advantage[t] = gae
            b["adv"], b["ret"] = advantage, advantage + b["val"]
            parts.append(pack_sequences(b, s.learner, p.get("sequence_length", 32)))
            s.steps += length * int(s.learner.sum())
            s.buf = None
        batch = {key: torch.from_numpy(np.concatenate([part[key] for part in parts],
                    axis=0 if key == "initial_memory" else 1)).to(self.device) for key in parts[0]}
        # Normalización congelada (BC o escalas del entorno). Cambiarla entre rollout
        # y update alteraría tanto el ratio PPO como el significado del estado oculto.
        samples = batch["act"].numel()
        self.steps += samples
        self.stage_steps += samples
        self.bc_coef = lerp(p.get("bc_kl_coef", 0), p.get("bc_kl_final", 0), fraction) if self.bc_model is not None else 0
        started = time.time()
        stats = self.update(batch, entropy_coef)
        update_time = time.time() - started
        self.iteration += 1
        self.opponent_curriculum()
        if self.iteration % cfg["league"]["snapshot_every"] == 0 and any(s.opp_stage >= 1 for s in self.slots):
            self.league.add_snapshot(self.model, f"it{self.iteration}")
        if self.iteration % cfg["log"]["checkpoint_every"] == 0:
            self.save(self.run_dir / "latest.pt")
            self.save(self.run_dir / f"ckpt_{self.iteration:06d}.pt")
        self.log(stats, lr, samples, rollout_time, update_time)
        self.maybe_rebalance_or_advance()

    def update(self, b, ent_coef):
        p = self.cfg["ppo"]
        length, sequences = b["act"].shape
        advantage = (b["adv"] - b["adv"].mean()) / (b["adv"].std(unbiased=False) + 1e-8)
        per_batch = max(1, p["minibatch"] // length)
        stats = {key: 0.0 for key in ("pg_loss", "v_loss", "entropy", "approx_kl", "clipfrac", "bc_kl")}
        total = 0
        for _ in range(p["epochs"]):
            order = torch.randperm(sequences, device=self.device)
            for start in range(0, sequences, per_batch):
                selected = order[start:start + per_batch]
                logits, value, _ = self.model.sequence(b["obs"][:, selected], b["initial_memory"][selected],
                    b["previous_action"][:, selected], b["episode_start"][:, selected])
                dist = torch.distributions.Categorical(logits=logits)
                log_ratio = dist.log_prob(b["act"][:, selected]) - b["logp"][:, selected]
                ratio = log_ratio.exp()
                adv = advantage[:, selected]
                pg = -torch.min(ratio * adv, ratio.clamp(1 - p["clip"], 1 + p["clip"]) * adv).mean()
                vl = 0.5 * (value - b["ret"][:, selected]).square().mean()
                entropy = dist.entropy().mean()
                loss = pg + p["vf_coef"] * vl - ent_coef * entropy
                kl = logits.new_zeros(())
                if self.bc_coef > 0:
                    with torch.no_grad():
                        teacher = self.bc_model.logits(b["obs"][:, selected].reshape(-1, self.obs_dim))
                        teacher = teacher.log_softmax(-1).reshape_as(logits)
                    kl = (teacher.exp() * (teacher - logits.log_softmax(-1))).sum(-1).mean()
                    loss = loss + self.bc_coef * kl
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), p["max_grad_norm"])
                self.opt.step()
                count = b["act"][:, selected].numel()
                with torch.no_grad():
                    values = (pg, vl, entropy, (ratio - 1 - log_ratio).mean(),
                              ((ratio - 1).abs() > p["clip"]).float().mean(), kl)
                    for key, value in zip(stats, values):
                        stats[key] += value.item() * count
                total += count
        return {key: value / max(total, 1) for key, value in stats.items()}
