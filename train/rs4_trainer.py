"""RS4 v3: useful-row PPO, match-stable controllers and device-resident memory.

The legacy trainers remain unchanged. Scenario labels and referee state never
enter observations; routing is a training mask, not a feature for the policy.
"""
from __future__ import annotations

import copy
import shutil
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from bots.scripted import scripted_actions
from env.haxball_env import RESTART_EVENT_NAMES
from . import multitask
from .checkpoints import atomic_torch_save, maybe_save_checkpoints
from .multitask import MultiTrainer, SELF, POOL, SCRIPTED, ROOT
from .rs4_program_v5 import program_from_config
from .runtime import batch_to_device


def useful_mask(mask, remaining):
    """Exact budget cap in time-major order; padding/frozen rows never count."""
    result = np.asarray(mask, dtype=bool).copy()
    indices = np.flatnonzero(result.reshape(-1))
    if len(indices) > remaining:
        result.reshape(-1)[indices[int(remaining):]] = False
    return result


def generalized_advantage(reward, values, done, last_value, gamma, lam, learner):
    """Stop credit at terminals and controller changes, not at rollout edges."""
    advantage = np.zeros_like(reward)
    gae = np.zeros_like(last_value)
    for t in reversed(range(len(reward))):
        next_value = last_value if t == len(reward) - 1 else values[t + 1]
        next_learner = learner[t] if t == len(reward) - 1 else learner[t + 1]
        alive = (~done[t, :, None]) & learner[t] & next_learner
        delta = reward[t] + gamma * next_value * alive - values[t]
        gae = delta + gamma * lam * alive * gae
        advantage[t] = gae
    return advantage


def sequence_arrays(buffer, length):
    """Pack dynamic learner masks without splicing players or trajectories."""
    steps, n, players = buffer["act"].shape
    if steps % length:
        raise ValueError("rollout_len debe ser múltiplo de sequence_length")
    chunks, rows = steps // length, n * players
    out = {}
    for key in ("obs", "act", "logp", "adv", "ret", "valid", "previous_action", "episode_start", "critic"):
        if key not in buffer:
            continue
        tail = buffer[key].shape[3:]
        out[key] = buffer[key].reshape(chunks, length, rows, *tail).swapaxes(0, 1).reshape(length, chunks * rows, *tail)
    selected = out["valid"].any(axis=0)
    return {key: value[:, selected] for key, value in out.items()}, selected


def block_settings(settings, block):
    """Calendario del programa más las opciones fijas del bloque RS4-b1 (cfg["rs4_b1"], PLAN_RS4 3).

    `recorded` = {path, ticks, mix}: banco de estados reales para el escenario "recorded"; la ruta
    relativa se resuelve desde la raíz del repo. `pass_participant_credit` reemplaza el crédito
    por participar en pases (0 en el bloque: la señal es sólo el gol). `restart_deadline_ticks` fija
    el plazo de entrenamiento para ejecutar saques del árbitro real (multa rs4_restart_stall).
    """
    settings = dict(settings)
    if not block:
        return settings
    recorded = block.get("recorded")
    if recorded:
        path = Path(str(recorded["path"]).replace("\\", "/"))
        settings["recorded"] = {**recorded, "path": str(path if path.is_absolute() else ROOT / path)}
    if "pass_participant_credit" in block:
        settings["pass_participant_credit"] = float(block["pass_participant_credit"])
    if "restart_deadline_ticks" in block:
        settings["restart_deadline_ticks"] = int(block["restart_deadline_ticks"])
    if "support_coef" in block:
        settings["support_coef"] = float(block["support_coef"])  # guía de apoyo (env/rs4_v3.support_potential)
    return settings


class SituationWindow:
    """Goles en situaciones reales (escenario "recorded") entre dos logs, por tipo de situación."""
    def __init__(self):
        self.counts = Counter()

    def add(self, results):
        for result in results:
            if result.get("scenario") != "recorded":
                continue
            pool, scorer = result["pool"], result["goal_team"]
            self.counts[(pool, "situations")] += 1
            self.counts[(pool, "goals")] += int(scorer >= 0)
            self.counts[(pool, "attacker_goals")] += int(scorer >= 0 and scorer == result["attacker"])

    def report(self):
        """{tipo: (situaciones, goles/100, goles del atacante/100)} y reinicio de la ventana."""
        out = {}
        for pool in ("attack", "open", "restart"):
            n = self.counts[(pool, "situations")]
            if n:
                out[pool] = (n, 100 * self.counts[(pool, "goals")] / n, 100 * self.counts[(pool, "attacker_goals")] / n)
        self.counts.clear()
        return out


class RS4V3Trainer(MultiTrainer):
    privileged_critic = 0  # build_envs lo fija desde model.critic_features

    def __init__(self, cfg, run, resume, init_from=None):
        self.program = program_from_config(cfg)
        self.v5 = self.program.config["version"] == 5
        fresh_v5 = self.v5 and not resume and init_from
        if not fresh_v5 and (not resume or init_from):
            raise ValueError("RS4 v3 exige un checkpoint preparado y --resume; no usar --init-from")
        if fresh_v5 and (ROOT / "runs" / run / "latest.pt").exists():
            raise ValueError("El run v5 ya existe; reanudar con --resume en lugar de --init-from")
        if self.v5 and cfg["ppo"].get("gamma") != self.program.config["reward"]["gamma"]:
            raise ValueError("ppo.gamma y rs4_program.reward.gamma deben coincidir")
        self._saved_extra = {}
        super().__init__(cfg, run, resume, init_from)
        if fresh_v5:
            self._seed_v5_league(cfg.get("rs4_v5", {}).get("league_seed", []))

        self.model.to(self.device)

        if self.steps - self.program.config["start_steps"] != self.program.relative_steps:
            raise ValueError("El contador del checkpoint no coincide con el presupuesto RS4 v3")

        self.league.match_pfsp = True
        self.league.mixture = tuple(cfg.get("league", {}).get("mixture", (.2, .4, .2, .2)))

        for name in ("self_norm", "ent_norm", "norm"):
            normalizer = getattr(self.model, name, None)
            if normalizer is not None:
                normalizer.eval()

        from .rs4_inference import RoutedPolicyInference
        self.inference = RoutedPolicyInference(self.device, self.cuda_decisions, max_policies=4)
        self.total_rows = sum(s.N * s.P for s in self.slots)
        self.inference.register("learner", self.model, self.total_rows)

        self._controller_models = {"learner": self.model}
        if self.bc_model is not None:
            self._controller_models["teacher"] = self.bc_model
            self.inference.register("teacher", self.bc_model, self.total_rows)
        self._pool_keys = []
        self._retiring_keys = set()
        self._in_rollout = False
        for slot in self.slots:
            self.assign_modes(slot)
        self._refresh_routes()
        if cfg.get("public_signals", {}).get("auto_joints"):
            print("RS4 señales públicas v1 | pelota RS ONE | barreras laterales por joints (sin estados privados)")

    def load(self, path):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        if "rs4_program_state" not in ck:
            raise ValueError("Checkpoint sin programa v3; usar tools.prepare_rs4_v3")
        self.program = program_from_config(self.cfg, ck["rs4_program_state"])
        self._saved_extra = {key: copy.deepcopy(ck[key]) for key in
                             ("migration", "rs4_migration", "public_signal_migration", "source_reference", "program_manifest", "evaluation_version") if key in ck}
        super().load(path)
        by_name = {saved["name"]: saved for saved in ck.get("league", []) if isinstance(saved, dict)}
        for member in self.league.members:
            saved = by_name.get(member.name)
            if isinstance(saved, dict):
                for key in ("match_points", "matches", "protected", "snapshot_steps"):
                    if key in saved:
                        setattr(member, key, saved[key])
        rng = ck.get("rng_state")
        if rng:
            self.rng.bit_generator.state = rng["numpy"]
            torch.set_rng_state(rng["torch"])
            if self.device.type == "cuda" and rng.get("cuda") is not None:
                torch.cuda.set_rng_state_all(rng["cuda"])

    def _seed_v5_league(self, paths):
        """Rivales protegidos de referencia: el modelo inicial y checkpoints externos."""
        from .model import build_model
        # protect_initial=false: el punto de partida queda como snapshot común y
        # las anclas (mezcla de liga) son sólo los rivales externos, p. ej. el campeón.
        protect = bool(self.cfg.get("rs4_v5", {}).get("protect_initial", True))
        if self.cfg.get("rs4_v5", {}).get("add_initial", True):
            self.league.add_snapshot(self.model, "v5_inicial", protected=protect)
        for raw in paths:
            path = Path(str(raw).replace("\\", "/"))
            path = path if path.is_absolute() else ROOT / path
            ck = torch.load(path, map_location="cpu", weights_only=False)
            model = build_model(ck["model_config"])
            model.load_state_dict(ck["model"])
            self.league.add_snapshot(model.to(self.device), "ref_" + path.parent.name + "_" + path.stem, protected=True)
        print("RS4 v5 liga inicial: " + ", ".join(m.name for m in self.league.members), flush=True)

    def _settings(self):
        return block_settings(self.program.settings(), self.cfg.get("rs4_b1"))

    def build_envs(self):
        if self.cfg["env"].get("action_delay_max", 0):
            raise ValueError("RS4 v3 requiere action_delay_max=0 para registrar la acción realmente ejecutada")
        super().build_envs()
        from env.rs4_v3 import RS4ScenarioEnv
        self.privileged_critic = int(getattr(self.model, "critic_features", 0))
        for s in self.slots:
            if s.P != 8 or s.T != 4 or s.task.name != "rs4_4v4":
                raise ValueError("RS4 v3 conserva exclusivamente RS4 4v4")
            if self.cfg.get("model", {}).get("public_signals_version"):
                cues = self.cfg.get("public_signals", {})
                s.env.configure_public_signals(cues)
            s.env.emit_critic_features = bool(self.privileged_critic)
            s.env = RS4ScenarioEnv(s.env)
            s.env.configure(self._settings())
            s.controller = np.full((s.N, s.P), "learner", dtype=object)
            s.learner_color = np.zeros(s.N, dtype=np.int8)
            s.env.learner_team = s.learner_color
            s.opponent_key = np.full(s.N, "scripted", dtype=object)
            s.modes = np.full(s.N, SCRIPTED, dtype=np.int64)
            s.opp_id = np.full(s.N, -1, dtype=np.int64)
            s.learner = np.ones((s.N, s.P), dtype=bool)
            s.match_finished[:] = True
            s.scripted_eps, s.scripted_policy, s.scripted_style = 0., "r3", -1

    def _refresh_pool(self):
        if self._in_rollout:
            return  # At most four distinct neural policies over the whole rollout.
        active = {str(key) for s in self.slots for key in s.controller.reshape(-1)
                  if str(key).startswith("pool:")}
        for key in list(self._pool_keys):
            if key not in active:
                self.inference.unregister(key)
                self._controller_models.pop(key, None)
                self._pool_keys.remove(key)
                self._retiring_keys.discard(key)
        available = 4 - len(self._controller_models)
        if available and self.league.members:
            for index in self.league.sample(max(available * 3, 2), self.rng):
                member = self.league.members[int(index)]
                key = "pool:" + member.name
                if key not in self._controller_models:
                    self._controller_models[key] = member.model
                    self.inference.register(key, member.model, self.total_rows)
                    self._pool_keys.append(key)
                    available -= 1
                if not available:
                    break

    def assign_modes(self, s):
        if not hasattr(self, "inference"):
            return
        rows = np.flatnonzero(s.match_finished)
        if not len(rows):
            return
        self._refresh_pool()
        settings = self._settings()
        mix = settings["opponent_mix"]
        usable_pool = [key for key in self._pool_keys if key not in self._retiring_keys]
        probabilities = np.array([mix["selfplay"], mix["pool"] if usable_pool else 0., mix["scripted"]])
        probabilities /= probabilities.sum()
        offset = sum(x.N * x.P for x in self.slots[:self.slots.index(s)])
        before = s.controller[rows].copy()
        learner_weights = self.cfg.get("rs4_b1", {}).get("learner_weights")
        for e in rows:
            mode = int(self.rng.choice(3, p=probabilities))
            color = int(self.rng.integers(2))
            ours = np.arange(color * s.T, (color + 1) * s.T)
            theirs = np.arange((1 - color) * s.T, (2 - color) * s.T)
            opponent = "learner" if mode == SELF else "scripted" if mode == SCRIPTED else str(self.rng.choice(usable_pool))
            s.controller[e] = "learner"
            s.controller[e, theirs] = opponent
            if learner_weights is not None:
                # PLAN_RS4: 70% un aprendiz + tres compañeros congelados, 20% dos o tres, 10% cuatro.
                count = int(self.rng.choice(4, p=np.asarray(learner_weights) / np.sum(learner_weights))) + 1
                frozen = self.rng.permutation(ours)[count:]
                if len(frozen):
                    companions = ["scripted", *usable_pool]
                    s.controller[e, frozen] = str(self.rng.choice(companions))
                s.modes[e], s.learner_color[e] = mode, color
                s.opponent_key[e] = opponent
                name = opponent.removeprefix("pool:")
                s.opp_id[e] = next((i for i, member in enumerate(self.league.members) if member.name == name), -1)
                continue
            companion_fraction = self.cfg.get("runtime", {}).get("benchmark_companion_fraction", settings["frozen_teammates_fraction"])
            if self.rng.random() < companion_fraction:
                learners = (int(self.rng.integers(1, 4)) if self.program.objective_contract is None else
                            int(self.rng.choice((1, 2, 3), p=settings["teammate_learner_weights"])))
                frozen = self.rng.permutation(ours)[learners:]
                companions = ["scripted", *usable_pool]
                if self.bc_model is not None:
                    companions.append("teacher")
                s.controller[e, frozen] = str(self.rng.choice(companions))
            s.modes[e], s.learner_color[e] = mode, color
            s.opponent_key[e] = opponent
            name = opponent.removeprefix("pool:")
            s.opp_id[e] = next((i for i, member in enumerate(self.league.members) if member.name == name), -1)
        changed = before != s.controller[rows]
        indices = (offset + rows[:, None] * s.P + np.arange(s.P))[changed]
        if len(indices):
            self.inference.reset_rows(indices)
        s.learner = s.controller == "learner"
        s.match_finished[rows] = False
        mismatched = np.flatnonzero(s.env.is_drill & (s.env.drill_team != s.learner_color))
        if len(mismatched):
            s.env.retarget(mismatched, s.learner_color[mismatched])
            s.observation_dirty = True

    def _refresh_routes(self):
        flat = np.concatenate([s.controller.reshape(-1) for s in self.slots])
        self._routes = {str(key): np.flatnonzero(flat == key) for key in np.unique(flat) if key != "scripted"}
        self._route_device = {key: torch.as_tensor(indices, device=self.device) for key, indices in self._routes.items()}

    @torch.no_grad()
    def act(self, obs_list):
        arrays = [o.reshape(-1, self.obs_dim) for o in obs_list]
        obs = (self._rollout_transfer.upload_many(arrays) if self._rollout_transfer else
               torch.from_numpy(np.concatenate(arrays)).to(self.device))
        actions = torch.zeros(self.total_rows, dtype=torch.long, device=self.device)
        logp = torch.zeros(self.total_rows, device=self.device)
        values = torch.zeros(self.total_rows, device=self.device)
        for key, indices in self._route_device.items():
            selected_obs = obs[indices]
            model = self._controller_models[key]
            if (self.model.public_signals_version and not getattr(model, "public_signals_version", 0)
                    and getattr(model, "rule_observation", "masked") == "full"):
                selected_obs[:, 56:71] = 0  # frozen legacy full references expect no RS4 private rule fields
            logits, value, _ = self.inference.infer(key, selected_obs, indices)
            dist = torch.distributions.Categorical(logits=logits, validate_args=False)
            sampled = dist.sample()  # RNG intentionally outside graph.
            actions[indices], logp[indices], values[indices] = sampled, dist.log_prob(sampled), value
            self.inference.record_actions(key, indices, sampled)
        if self._rollout_transfer:
            self._rollout_transfer.enqueue_output(actions, logp, values)
        # Scripted work overlaps the pending D2H decision transfer.
        scripted = []
        for s in self.slots:
            overrides = np.zeros((s.N, s.P), dtype=np.int64)
            mask = s.controller == "scripted"
            for team in (0, 1):
                players = np.arange(team * s.T, (team + 1) * s.T)
                rows = np.flatnonzero(mask[:, players].any(axis=1))
                if len(rows):
                    overrides[np.ix_(rows, players)] = scripted_actions(s.env, players, 0., self.rng,
                        env_indices=rows, policy="r3", style=int(self.cfg.get("rs4_b1", {}).get("scripted_style", -1)))
            scripted.append((mask, overrides))
        if self._rollout_transfer:
            act, lp, val = self._rollout_transfer.wait_output()
        else:
            packed = torch.stack((actions.float(), logp, values)).cpu().numpy()
            act, lp, val = packed[0].astype(np.int64), packed[1], packed[2]
        out, offset = [], 0
        for s, (mask, overrides) in zip(self.slots, scripted):
            size = s.N * s.P
            a = act[offset:offset + size].reshape(s.N, s.P)
            a[mask] = overrides[mask]
            out.append((a, lp[offset:offset + size].reshape(s.N, s.P), val[offset:offset + size].reshape(s.N, s.P)))
            offset += size
        return out

    @torch.no_grad()
    def values_many(self, obs_list):
        outputs, offset = [], 0
        for s, obs in zip(self.slots, obs_list):
            indices = torch.arange(offset, offset + s.N * s.P, device=self.device)
            _, value, _ = self.inference.infer("learner", torch.from_numpy(obs.reshape(-1, self.obs_dim)).to(self.device),
                                             indices, commit=False)
            outputs.append(value.cpu().numpy().reshape(s.N, s.P))
            offset += s.N * s.P
        return outputs

    def _record(self, s, info):
        full = np.asarray(info.get("match_done", np.zeros(s.N, bool)), bool).copy()
        full &= ~np.asarray(info.get("is_drill", np.zeros(s.N, bool)), bool)
        goal = np.asarray(info["goal"]) * np.where(s.learner_color == 0, 1, -1)
        for mode in (SELF, POOL, SCRIPTED):
            mask = (s.modes == mode) & ~np.asarray(info.get("is_drill", np.zeros(s.N, bool)), bool)
            s.goals[mode][0] += int((goal[mask] == 1).sum())
            s.goals[mode][1] += int((goal[mask] == -1).sum())
        for e in np.flatnonzero(full):
            score = info["final_score"][e]
            ours, theirs = int(score[s.learner_color[e]]), int(score[1 - s.learner_color[e]])
            result = float(ours > theirs) + .5 * float(ours == theirs)
            bucket = (int(ours > theirs), int(ours == theirs), int(ours < theirs), ours, theirs, int(ours + theirs == 0))
            s.match_results[int(s.modes[e])] += bucket
            if s.modes[e] == SCRIPTED:
                s.match_window.append(bucket)
                s.match_window = s.match_window[-400:]
                # Shared console/TensorBoard logging reads winrate() from this
                # legacy window. Use the same complete, non-drill matches and
                # learner-relative scores as the RS4 points metric, including
                # history restored from checkpoints.
                s.wr_window = [(row[3], row[4]) for row in s.match_window]
            # Indices can move when history is evicted. Resolve stable names at
            # result time; never attribute a held controller to another member.
            name = str(s.opponent_key[e]).removeprefix("pool:")
            member_id = next((i for i, member in enumerate(self.league.members) if member.name == name), None)
            if s.modes[e] == POOL and member_id is not None:
                self.league.record_match(member_id, result)
            if s.modes[e] != SELF:
                self.league.record(member_id if s.modes[e] == POOL else None, ours, theirs)
        # Drills also allow new controllers, but never create a league result.
        finished_drill = np.asarray(info.get("is_drill", np.zeros(s.N, bool)), bool) & np.asarray(info["done"])
        s.match_finished |= full | finished_drill

    def iterate(self):
        iteration_started = time.perf_counter()
        p, length = self.cfg["ppo"], self.cfg["ppo"]["rollout_len"]
        if self.remaining_steps <= 0:
            raise RuntimeError("Presupuesto RS4 v3 agotado")
        recurrent = getattr(self.model, "is_recurrent", False)
        seq_length = p.get("sequence_length", 32)
        if recurrent and length % seq_length:
            raise ValueError("rollout_len debe ser múltiplo de sequence_length")
        settings = self._settings()
        self._in_rollout = False
        self._refresh_pool()
        if self.iteration > 0 and self.iteration % 25 == 0 and self._pool_keys:
            # Retired controllers finish their existing matches; replacement is
            # registered only after their final holder has finished.
            self._retiring_keys.add(self._pool_keys[self.iteration // 25 % len(self._pool_keys)])
        for group in self.opt.param_groups:
            group["lr"] = self.program.lr
        self.bc_coef = settings["bc_coef"] if self.bc_model is not None else 0.
        for s in self.slots:
            s.env.configure(settings)
            self.assign_modes(s)
            shapes = (("obs", (self.obs_dim,), np.float32), ("act", (), np.int64), ("logp", (), np.float32),
                      ("val", (), np.float32), ("rew", (), np.float32), ("valid", (), bool))
            if not hasattr(s, "v3_buffers"):
                s.v3_buffers = {k: np.empty((length, s.N, s.P) + shape, dtype) for k, shape, dtype in shapes}
                s.v3_buffers["done"] = np.empty((length, s.N), bool)
                if self.privileged_critic:
                    s.v3_buffers["critic"] = np.empty((length, s.N, s.P, self.privileged_critic), np.float32)
            s.buf = s.v3_buffers
        self._refresh_routes()
        self._in_rollout = True
        if not hasattr(self, "_obs"):
            self._obs = [s.env.reset() for s in self.slots]
        for i, s in enumerate(self.slots):
            if getattr(s, "observation_dirty", False):
                self._obs[i] = s.env.observe()
                s.observation_dirty = False
        started = time.perf_counter()
        setup_seconds = started - iteration_started
        memory_chunks, obs = [], self._obs
        previous_buffer = torch.empty((length, self.total_rows), dtype=torch.long, device=self.device) if recurrent else None
        start_buffer = torch.empty((length, self.total_rows), dtype=torch.bool, device=self.device) if recurrent else None
        event_counts = np.zeros((len(RESTART_EVENT_NAMES), 2), np.int64)
        tactical_sum = tactical_abs = tactical_count = 0
        # Ventana entre logs: los partidos espejo terminan sincronizados cada
        # ~19 iteraciones, así que una sola iteración no representa los términos.
        term_sums = self.__dict__.setdefault("_term_window", {})
        situations = self.__dict__.setdefault("_situation_window", SituationWindow())
        drill_samples = 0
        for t in range(length):
            if recurrent:
                hidden, previous, start = self.inference.get_state("learner", clone=t % seq_length == 0)
                if t % seq_length == 0:
                    memory_chunks.append(hidden)
                previous_buffer[t].copy_(previous)
                start_buffer[t].copy_(start)
            offset = 0
            for s, o in zip(self.slots, obs):
                size = s.N * s.P
                s.buf["obs"][t], s.buf["valid"][t] = o, s.learner
                if self.privileged_critic:
                    s.buf["critic"][t] = s.env.critic_features()
                offset += size
            decisions = self.act(obs)
            next_obs, offset = [], 0
            reroute = False
            for s, (acts, logp, values) in zip(self.slots, decisions):
                b = s.buf
                b["act"][t], b["logp"][t], b["val"][t] = acts, logp, values
                o2, reward, done, info = s.env.step(acts)
                executed = np.asarray(info.get("executed_actions", acts))
                changed = executed != acts
                if changed.any():
                    for key in np.unique(s.controller[changed]):
                        if key == "scripted":
                            continue
                        local = np.flatnonzero((changed & (s.controller == key)).reshape(-1))
                        self.inference.record_actions(str(key), offset + local, executed.reshape(-1)[local])
                if info["truncated"].any():
                    rows = np.flatnonzero(info["truncated"])
                    indices = (offset + rows[:, None] * s.P + np.arange(s.P)).reshape(-1)
                    if self.privileged_critic:
                        value = self._critic_values(info["final_obs"][rows], info["final_critic"][rows])
                    else:
                        _, value, _ = self.inference.infer("learner",
                            torch.from_numpy(info["final_obs"][rows].reshape(-1, self.obs_dim)).to(self.device),
                            torch.as_tensor(indices, device=self.device), commit=False)
                    reward[rows] += p["gamma"] * value.cpu().numpy().reshape(len(rows), s.P)
                b["rew"][t], b["done"][t] = reward, done
                indices = (offset + np.flatnonzero(done)[:, None] * s.P + np.arange(s.P)).reshape(-1)
                if len(indices):
                    self.inference.reset_rows(indices)
                for k, name in enumerate(RESTART_EVENT_NAMES):
                    event_counts[k] += info.get("events", {}).get(name, np.zeros((s.N, 2), int)).sum(axis=0)
                if "reward_terms" in info:
                    learner_rows = s.learner
                    term_sums["_rows"] = term_sums.get("_rows", 0) + int(learner_rows.sum())
                    for name, values in info["reward_terms"].items():
                        term_sums[name] = term_sums.get(name, 0.) + float(np.asarray(values)[learner_rows].sum())
                if "rs4_tactical_reward" in info:
                    r = info["rs4_tactical_reward"]
                    tactical_sum += float(r.sum()); tactical_abs += float(np.abs(r).sum()); tactical_count += r.size
                info["done"] = done
                situations.add(info.get("scenario_result", ()))
                drill_samples += int((b["valid"][t] & info.get("is_drill", np.zeros(s.N, bool))[:, None]).sum())
                self._record(s, info)
                if s.match_finished.any():
                    self.assign_modes(s)
                    reroute = True
                    if getattr(s, "observation_dirty", False):
                        o2 = s.env.observe()
                        s.observation_dirty = False
                next_obs.append(o2)
                offset += s.N * s.P
            if reroute:
                self._refresh_routes()
            obs = next_obs
        self._obs = obs
        self._in_rollout = False
        rollout_seconds = time.perf_counter() - started
        started = time.perf_counter()
        last_values = self.values_many(obs)
        if self.privileged_critic and recurrent:
            # Valor recurrente con estado privilegiado: misma memoria que la inferencia, recalculada
            # desde el inicio del rollout (acciones previas y reinicios grabados).
            hidden_now, previous_now, start_now = self.inference.get_state("learner", clone=True)
            last_values, offset = [], 0
            for s, o in zip(self.slots, obs):
                b, width = s.buf, s.N * s.P
                rows = slice(offset, offset + width)
                b["val"] = self._critic_values_recurrent(b["obs"], b["critic"], memory_chunks[0][rows],
                                                         previous_buffer[:, rows], start_buffer[:, rows]
                                                         ).cpu().numpy().reshape(b["val"].shape)
                last_values.append(self._critic_step_value(o, s.env.critic_features(), hidden_now[rows],
                                                           previous_now[rows], start_now[rows]
                                                           ).cpu().numpy().reshape(s.N, s.P))
                offset += width
        elif self.privileged_critic:
            for s in self.slots:
                b = s.buf
                b["val"] = self._critic_values(b["obs"], b["critic"]).cpu().numpy().reshape(b["val"].shape)
            last_values = [self._critic_values(o, s.env.critic_features()).cpu().numpy().reshape(s.N, s.P)
                           for s, o in zip(self.slots, obs)]
        count_remaining, parts, selection = self.remaining_steps, {}, []
        offsets, offset = [], 0
        for s, last in zip(self.slots, last_values):
            b = s.buf
            b["adv"] = generalized_advantage(b["rew"], b["val"], b["done"], last,
                                             p["gamma"], p["gae_lambda"], b["valid"])
            b["ret"] = b["adv"] + b["val"]
            b["valid"][:] = useful_mask(b["valid"], count_remaining)
            count = int(b["valid"].sum())
            count_remaining -= count
            s.steps += count
            if recurrent:
                arrays, selected = sequence_arrays(b, seq_length)
                selection.append(selected)
                offsets.append((offset, offset + s.N * s.P))
                for key, value in arrays.items():
                    if key in ("previous_action", "episode_start"):
                        continue
                    # BatchTransfer's first axis is sequences.
                    parts.setdefault(key, []).append(np.swapaxes(value, 0, 1))
            else:
                for key in ("obs", "act", "logp", "adv", "ret") + (("critic",) if self.privileged_critic else ()):
                    parts.setdefault(key, []).append(b[key][b["valid"]])
            offset += s.N * s.P
        batch = batch_to_device(parts, self.device, self._batch_transfer)
        if recurrent:
            batch = {key: tensor.transpose(0, 1) for key, tensor in batch.items()}
            all_memories = torch.stack(memory_chunks)
            packed = [all_memories[:, left:right].reshape(-1, self.model.memory_size)[torch.as_tensor(mask, device=self.device)]
                      for (left, right), mask in zip(offsets, selection)]
            batch["initial_memory"] = torch.cat(packed)
            for key, tensor in (("previous_action", previous_buffer), ("episode_start", start_buffer)):
                pieces = []
                for (left, right), mask in zip(offsets, selection):
                    width = right - left
                    piece = tensor[:, left:right].reshape(length // seq_length, seq_length, width).transpose(0, 1).reshape(seq_length, -1)
                    pieces.append(piece[:, torch.as_tensor(mask, device=self.device)])
                batch[key] = torch.cat(pieces, dim=1)
            samples = int(batch["valid"].sum().item())
        else:
            samples = len(batch["act"])
        prepare_seconds = time.perf_counter() - started
        started = time.perf_counter()
        stats = self.update(batch, settings["entropy_coef"])
        update_seconds = time.perf_counter() - started
        self.steps += samples
        self.stage_steps += samples
        self.iteration += 1
        self.program.advance_steps(samples)
        self.program.update_lr(stats["endpoint_kl"], self.iteration)
        graph_metrics = dict(capture_seconds=self.inference.capture_seconds, captures=self.inference.captures,
                             replays=self.inference.replays, input_rows=self.inference.graph_input_rows,
                             padding_rows=self.inference.graph_padding_rows)
        self.inference.invalidate()
        started = time.perf_counter()
        benchmark = self.cfg.get("runtime", {}).get("benchmark", False)
        if self.program.snapshot_due and not benchmark:
            prefix = "v5" if self.v5 else "v3"
            self.league.add_snapshot(self.model, f"{prefix}_{self.program.relative_steps}", steps=self.program.relative_steps)
            self.program.mark_snapshot()
        if not benchmark:
            maybe_save_checkpoints(self)
        component_stats = {}
        if self.iteration % self.cfg["log"]["every"] == 0:
            from env.rs4_tactics import components
            public = [components(s.env.sim.player_pos, s.env.sim.player_team, s.env.sim.ball_pos,
                                 s.env.goal_x, s.env.field_h, s.env.sim.st.goal_half_height,
                                 s.env.rs4_formation_version) for s in self.slots]
            means = np.concatenate(public).mean(axis=(0, 1))
            component_stats = {f"rs4/component_{name}": float(value) for name, value in
                               zip(("structure", "threat", "danger"), means)}
        maintenance_seconds = time.perf_counter() - started
        stats.update({"timing/prepare_seconds": prepare_seconds, "timing/maintenance_seconds": maintenance_seconds,
                      "timing/setup_seconds": setup_seconds, "rollout/drill_learner_samples": drill_samples,
                      "rollout/neural_controllers": len(self._routes),
                      "cuda_graph/capture_seconds": graph_metrics["capture_seconds"],
                      "cuda_graph/captures": graph_metrics["captures"], "cuda_graph/replays": graph_metrics["replays"],
                      "cuda_graph/input_rows": graph_metrics["input_rows"], "cuda_graph/padding_rows": graph_metrics["padding_rows"],
                      "program/relative_steps": self.program.relative_steps, "program/phase": self.program.phase_index,
                      "program/budget_remaining": self.program.remaining_steps,
                      "program/segment_remaining": self.remaining_steps, "program/next_lr": self.program.lr,
                      "rollout/simulated_samples": length * self.total_rows,
                      "rollout/learner_samples": samples,
                      "rollout/learner_fraction": samples / (length * self.total_rows),
                      "rs4/formation_version": self.slots[0].env.rs4_formation_version, "rs4/restart_potential_bound": settings["restart_potential_coef"],
                      "rs4/tactical_potential_bound": settings["guide_coef"],
                      "rs4/pass_possession_cap": self.slots[0].env.rcfg.team_pass_possession_cap,
                      "rs4/tactical_coef": settings["guide_coef"], "rs4/tactical_reward_mean": tactical_sum / max(tactical_count, 1),
                      "rs4/frozen_teammates_fraction": settings["frozen_teammates_fraction"],
                      "rs4/single_learner_assignment_fraction": settings["frozen_teammates_fraction"] * settings["teammate_learner_weights"][0],
                      "rs4/scenario_difficulty": settings["scenario_difficulty"],
                      "rs4/tactical_reward_abs_mean": tactical_abs / max(tactical_count, 1)})
        stats.update(component_stats)
        # Por 1000 decisiones de aprendiz, acumulado desde el último log.
        if self.iteration % self.cfg["log"]["every"] == 0 and term_sums:
            rows = term_sums.pop("_rows", 0)
            for name, total in term_sums.items():
                stats[f"reward_terms/{name}_per_1k"] = 1000 * total / max(rows, 1)
            term_sums.clear()
        if self.iteration % self.cfg["log"]["every"] == 0:
            for pool, (count, goals, attacker_goals) in situations.report().items():
                stats[f"rs4_b1/{pool}_situations"] = count
                stats[f"rs4_b1/{pool}_goals_per_100"] = goals
                stats[f"rs4_b1/{pool}_attacker_goals_per_100"] = attacker_goals
        for i, name in enumerate(RESTART_EVENT_NAMES):
            for color, label in enumerate(("red", "blue")):
                stats[f"rs4/window_{name}_{label}"] = int(event_counts[i, color])
        logging_started = time.perf_counter()
        self.log(stats, settings["lr"], samples, rollout_seconds, update_seconds)
        if self.iteration % self.cfg["log"]["every"] == 0 and self.v5:
            self._log_v5(stats, settings, samples, length)
        elif self.iteration % self.cfg["log"]["every"] == 0:
            print(f"      RS4 v3 fase {settings['phase_id']} | útiles {self.program.relative_steps / 1e6:.1f}M "
                  f"| hasta pausa {self.remaining_steps / 1e6:.1f}M "
                  f"| presupuesto propio restante {self.program.remaining_steps / 1e6:.1f}M "
                  f"| KL final {stats['endpoint_kl']:.5f} "
                  f"| épocas {stats['epochs_completed']} | repaso {settings['skill_debts']} "
                  f"| bonus saque {settings['restart_execute_bonus']:.4f}", flush=True)
            print("      potenciales estructura/amenaza/peligro "
                  + "/".join(f"{component_stats['rs4/component_' + name]:.3f}" for name in ("structure", "threat", "danger"))
                  + f" | límites guía {settings['guide_coef']:.3f}, saque {settings['restart_potential_coef']:.3f}, "
                    f"pase/posesión {self.slots[0].env.rcfg.team_pass_possession_cap:.4f}", flush=True)
            print(f"      carga rollout | filas aprendices {samples:,}/{length * self.total_rows:,} "
                  f"({100 * stats['rollout/learner_fraction']:.1f}%) | políticas neuronales {len(self._routes)} "
                  f"| setup {setup_seconds:.3f}s mantenimiento {maintenance_seconds:.3f}s", flush=True)
            if self.program.objective_contract:
                print(f"      objetivo {self.program.objective_contract} | compañeros congelados "
                      f"{settings['frozen_teammates_fraction']:.0%} | asignaciones de una instancia "
                      f"{stats['rs4/single_learner_assignment_fraction']:.1%} | dificultad "
                      f"{settings['scenario_difficulty']:.2f}", flush=True)
        self._last_iteration_timings = dict(setup=setup_seconds, maintenance=maintenance_seconds,
                                            logging=time.perf_counter() - logging_started, schedule=0.)

    def _log_v5(self, stats, settings, samples, length):
        terms = " ".join(f"{name} {stats[f'reward_terms/{name}_per_1k']:+.3f}"
                         for name in ("goal", "shaping", "pass", "possession", "no_goal", "pressure")
                         if f"reward_terms/{name}_per_1k" in stats)
        print(f"      RS4 v5 | útiles {self.program.relative_steps / 1e6:.1f}M | restante {self.program.remaining_steps / 1e6:.1f}M "
              f"| KL final {stats['endpoint_kl']:.4f} épocas {stats['epochs_completed']} | lr {self.program.lr:.2e} "
              f"| BC-KL x{settings['bc_coef']:.3f} ({stats.get('bc_kl', 0.):.3f}) | ent x{settings['entropy_coef']:.4f}"
              + (" | CALENTAMIENTO CRÍTICO (política congelada)" if getattr(self.program, "critic_warmup", False) else ""),
              flush=True)
        if terms:  # sólo hay desglose con términos v5 activos (multas, posesión)
            print(f"      recompensa por 1000 decisiones | {terms}", flush=True)
        results = []
        previous = getattr(self, "_v5_logged_results", {})
        for mode, label in ((SELF, "espejo"), (POOL, "liga"), (SCRIPTED, "R3")):
            rows = [s.match_results[mode] for s in self.slots if mode in getattr(s, "match_results", {})]
            if rows:
                cumulative = np.sum(rows, axis=0)
                total = cumulative - previous.get(mode, 0)
                previous[mode] = cumulative
                games = int(total[0] + total[1] + total[2])
                if games:
                    results.append(f"{label} {games} partidos, goles/partido {(total[3] + total[4]) / games:.2f}, "
                                   f"0-0 {total[5] / games:.0%}, puntos {(total[0] + .5 * total[1]) / games:.2f}")
        self._v5_logged_results = previous
        if results:
            print("      partidos desde el último log | " + " | ".join(results), flush=True)
        labels = {"attack": "ataque", "open": "juego abierto", "restart": "saque"}
        situations = [f"{labels[pool]} {stats[f'rs4_b1/{pool}_situations']}: goles {stats[f'rs4_b1/{pool}_goals_per_100']:.1f}/100, "
                      f"del atacante {stats[f'rs4_b1/{pool}_attacker_goals_per_100']:.1f}/100"
                      for pool in labels if f"rs4_b1/{pool}_situations" in stats]
        if situations:
            print("      situaciones reales desde el último log | " + " | ".join(situations), flush=True)
        print(f"      filas aprendices {samples:,}/{length * self.total_rows:,} ({100 * stats['rollout/learner_fraction']:.1f}%) "
              f"| políticas neuronales {len(self._routes)} | liga {len(self.league.members)}", flush=True)

    @property
    def remaining_steps(self):
        segment = int(self.cfg["ppo"]["total_steps"]) - self.steps
        return max(0, min(segment, self.program.remaining_steps))

    @torch.no_grad()
    def _critic_values(self, obs, critic, chunk=65536):
        """V(obs, estado privilegiado) del aprendiz; filas en cualquier forma (..., D)."""
        flat_obs = torch.as_tensor(np.ascontiguousarray(obs).reshape(-1, self.obs_dim), device=self.device)
        flat_critic = torch.as_tensor(np.ascontiguousarray(critic).reshape(-1, self.privileged_critic), device=self.device)
        return torch.cat([self.model.value(flat_obs[i:i + chunk], flat_critic[i:i + chunk])
                          for i in range(0, len(flat_obs), chunk)])

    @torch.no_grad()
    def _critic_values_recurrent(self, obs, critic, memory, previous, start, chunk=8192):
        """V(obs, privilegiado | memoria) para (tiempo, filas) reproduciendo la secuencia del rollout."""
        length = obs.shape[0]
        flat_obs = torch.as_tensor(np.ascontiguousarray(obs).reshape(length, -1, self.obs_dim), device=self.device)
        flat_critic = torch.as_tensor(np.ascontiguousarray(critic).reshape(length, -1, self.privileged_critic), device=self.device)
        pieces = []
        for i in range(0, flat_obs.shape[1], chunk):
            part = slice(i, i + chunk)
            _, values, _ = self.model.sequence(flat_obs[:, part], memory[part], previous[:, part], start[:, part],
                                               flat_critic[:, part])
            pieces.append(values)
        return torch.cat(pieces, dim=1)

    @torch.no_grad()
    def _critic_step_value(self, obs, critic, memory, previous, start):
        x = torch.as_tensor(np.ascontiguousarray(obs).reshape(-1, self.obs_dim), device=self.device)
        c = torch.as_tensor(np.ascontiguousarray(critic).reshape(-1, self.privileged_critic), device=self.device)
        _, value, _ = self.model.step(x, memory, previous, start, c)
        return value

    def _forward_batch(self, batch, indices):
        if getattr(self.model, "is_recurrent", False):
            critic = batch["critic"][:, indices] if self.privileged_critic else None
            return self.model.sequence(batch["obs"][:, indices], batch["initial_memory"][indices],
                                       batch["previous_action"][:, indices], batch["episode_start"][:, indices],
                                       critic)[:2]
        if self.privileged_critic:
            return self.model(batch["obs"][indices], batch["critic"][indices])
        return self.model(batch["obs"][indices])

    @torch.no_grad()
    def endpoint_kl(self, batch, indices):
        logits, _ = self._forward_batch(batch, indices)
        recurrent = getattr(self.model, "is_recurrent", False)
        actions = batch["act"][:, indices] if recurrent else batch["act"][indices]
        old = batch["logp"][:, indices] if recurrent else batch["logp"][indices]
        delta = torch.distributions.Categorical(logits=logits, validate_args=False).log_prob(actions) - old
        values = delta.exp() - 1 - delta
        return values[batch["valid"][:, indices]].mean().item() if recurrent else values.mean().item()

    def update(self, batch, ent_coef):
        p = self.cfg["ppo"]
        recurrent = getattr(self.model, "is_recurrent", False)
        axis = 1 if recurrent else 0
        count = batch["act"].shape[axis]
        seq = batch["act"].shape[0] if recurrent else 1
        mb = max(1, min(count, min(8192, p["minibatch"]) // seq))
        valid = batch["valid"] if recurrent else torch.ones_like(batch["act"], dtype=torch.bool)
        advantage = batch["adv"][valid]
        norm_adv = (batch["adv"] - advantage.mean()) / (advantage.std(unbiased=False) + 1e-8)
        keys = ("pg_loss", "v_loss", "entropy", "approx_kl", "clipfrac", "bc_kl")
        aggregate = torch.zeros(len(keys), device=self.device)
        samples_seen = 0
        estimate_indices = torch.randperm(count, device=self.device)[:min(count, max(1, 8192 // seq))]
        teacher_cache = None
        if self.bc_coef:
            with torch.no_grad():
                flat = batch["obs"].reshape(-1, self.obs_dim)
                teacher_cache = torch.cat([self.bc_model.logits(flat[i:i + 8192]).log_softmax(-1)
                                          for i in range(0, len(flat), 8192)]).reshape(*batch["act"].shape, -1)
        endpoint, epochs = 0., 0
        critic_warmup = bool(getattr(self.program, "critic_warmup", False))
        for epoch in range(p["epochs"]):
            order = torch.randperm(count, device=self.device)
            for start in range(0, count, mb):
                indices = order[start:start + mb]
                def take(key):
                    return batch[key][:, indices] if recurrent else batch[key][indices]
                mask = take("valid") if recurrent else torch.ones_like(take("act"), dtype=torch.bool)
                logits, value = self._forward_batch(batch, indices)
                dist = torch.distributions.Categorical(logits=logits, validate_args=False)
                log_ratio = dist.log_prob(take("act")) - take("logp")
                ratio = log_ratio.exp()
                a = norm_adv[:, indices] if recurrent else norm_adv[indices]
                pg = -torch.minimum(ratio * a, ratio.clamp(1 - p["clip"], 1 + p["clip"]) * a)[mask].mean()
                vl = .5 * (value - take("ret")).square()[mask].mean()
                entropy = dist.entropy()[mask].mean()
                bc = logits.new_zeros(())
                if teacher_cache is not None:
                    teacher = teacher_cache[:, indices] if recurrent else teacher_cache[indices]
                    bc = (teacher.exp() * (teacher - logits.log_softmax(-1))).sum(-1)[mask].mean()
                if critic_warmup:
                    # Política congelada: sólo la cabeza de valor aprende la recompensa nueva.
                    loss = p["vf_coef"] * vl
                else:
                    loss = pg + p["vf_coef"] * vl - ent_coef * entropy + self.bc_coef * bc
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                if critic_warmup:
                    for name, parameter in self.model.named_parameters():
                        if not name.startswith(("v_body.", "v.", "critic_proj.", "memory_v.")):
                            parameter.grad = None  # el encoder compartido tampoco cambia
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), p["max_grad_norm"])
                self.opt.step()
                with torch.no_grad():
                    n = mask.sum()
                    aggregate += torch.stack((pg, vl, entropy, (ratio - 1 - log_ratio)[mask].mean(),
                                              ((ratio - 1).abs() > p["clip"])[mask].float().mean(), bc)) * n
                    samples_seen += n
            epochs += 1
            endpoint = self.endpoint_kl(batch, estimate_indices)
            if endpoint > self.program.config["lr"]["epoch_stop_kl"]:
                break
        result = dict(zip(keys, (aggregate / samples_seen).cpu().tolist()))
        result.update(endpoint_kl=endpoint, epochs_completed=epochs)
        return result

    def save(self, path: Path):
        self.sync_task_state()
        payload = dict(checkpoint_version=3, model=self.model.state_dict(), model_config=self.model.config(),
                       opt=self.opt.state_dict(), optimizer_param_names=[[name for name, _ in self.model.named_parameters()]],
                       steps=self.steps, iteration=self.iteration, stage=self.stage, stage_steps=self.stage_steps,
                       task_state=self.task_state, learner_elo=self.league.learner_elo, scripted_elo=self.league.scripted_elo,
                       rs4_program_state=self.program.state_dict(), frozen_normalizers=True,
                       league=[dict(name=m.name, model=m.model.state_dict(), model_config=m.model.config(), elo=m.elo,
                                    wins=m.wins, games=m.games, match_points=m.match_points, matches=m.matches,
                                    protected=m.protected, snapshot_steps=m.snapshot_steps) for m in self.league.members],
                       env=dict(obs_layout="universal", tasks=[s.task.name for s in self.slots],
                                max_entities=self.max_entities, frame_skip=self.cfg["env"]["frame_skip"],
                                public_signal_config=self.cfg.get("public_signals", {})),
                       rng_state=dict(numpy=self.rng.bit_generator.state, torch=torch.get_rng_state(),
                                      cuda=torch.cuda.get_rng_state_all() if self.device.type == "cuda" else None),
                       **self._saved_extra)
        # Atomic replacement temporarily requires both old and new archives.
        bytes_needed = sum(t.numel() * t.element_size() for t in self.model.state_dict().values()) * 5
        bytes_needed += sum(sum(t.numel() * t.element_size() for t in m.model.state_dict().values()) for m in self.league.members)
        if shutil.disk_usage(Path(path).parent).free < max(64 * 1024**2, bytes_needed * 2):
            raise OSError("Espacio insuficiente para guardar atómicamente; latest anterior intacto")
        atomic_torch_save(payload, path)

    def train(self):
        limit = float(self.cfg.get("runtime", {}).get("max_wall_seconds", 0) or 0)
        started = time.time()
        try:
            while self.remaining_steps:
                if limit and time.time() - started > limit:
                    print(f"RS4: tope de tiempo alcanzado ({limit / 3600:.2f} h); se guarda y termina", flush=True)
                    break
                self.iterate()
                if (self.program.evaluation_due and not self.cfg.get("runtime", {}).get("benchmark", False)
                        and self.cfg.get("rs4_v3", {}).get("pause_for_evaluation", True)):
                    print("RS4 v3: pausa para evaluación independiente; continuar con tools.run_rs4_v3", flush=True)
                    break
        except KeyboardInterrupt:
            print("Interrumpido: guardando RS4 v3 sin reiniciar calendarios", flush=True)
        self.save(self.run_dir / "latest.pt")
        self.writer.close()
        print(f"guardado en {self.run_dir / 'latest.pt'}", flush=True)
