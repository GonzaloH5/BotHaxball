"""PPO multi-tarea: un solo modelo para muchos mapas y formatos a la vez (que no se encasille).

python -m train.multitask --config train/config_multi.yaml [--run nombre] [--resume] [--override clave=valor ...]

Cómo funciona:
  * Cada iteración juega partidos de TODAS las tareas activas (mapa × formato × reglas, ver train/tasks.yaml)
    y hace un único update con todas las muestras mezcladas. La obs "universal" (env/haxball_env.py) describe
    el mapa y rellena entidades, y SetActorCritic (train/model.py) acepta cualquier cantidad de jugadores.
  * Currículo por AMPLITUD con repaso: las etapas (`stages`) van sumando tareas; una tarea que ya entró nunca
    baja de `min_share` de las muestras, así no se olvida. Se pasa de etapa cuando todas las tareas activas
    llegaron a la liga del currículo de rivales y se cumplió `min_steps` (o al llegar a `max_steps`).
  * Currículo de rivales POR TAREA (scripteado fácil -> difícil -> liga), igual que train/ppo_selfplay.py.
  * Una sola liga de snapshots para todos los formatos (el modelo juega cualquiera); no se vacía entre etapas.
  * Control de regresiones: si el winrate contra el bot de una tarea cae `regression_drop` respecto de su mejor
    marca, se sube su peso (x`regression_boost`) hasta que se recupere.
"""
from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from bots.scripted import scripted_actions
from env.rewards import RewardConfig
from env.tasks import load_catalog, make_env

from .league import League
from .model import SetActorCritic, build_model
from .ppo_selfplay import lerp, resolve_device

ROOT = Path(__file__).resolve().parent.parent
SELF, POOL, SCRIPTED = 0, 1, 2
ENT_DIM = 8


class TaskSlot:
    """Estado de una tarea dentro de la iteración: entorno, rivales, buffers y estadísticas."""

    def __init__(self, task, env):
        self.task, self.env = task, env
        self.N, self.P, self.T = env.N, env.P, env.T
        self.opp_stage = 0
        self.wr_window: list[tuple[int, int]] = []
        self.best_wr = 0.0
        self.boost = 1.0
        self.goals = {SELF: [0, 0], POOL: [0, 0], SCRIPTED: [0, 0]}
        self.steps = 0
        self.intro_step = 0   # paso global en que la tarea entró al entrenamiento (decaimiento del shaping)
        self.regression_context = None

    def set_regression_context(self, scripted_eps):
        """Una marca sólo es comparable contra la misma dificultad de rival."""
        context = (self.opp_stage, float(scripted_eps))
        if self.regression_context != context:
            self.best_wr, self.boost = 0.0, 1.0
            self.wr_window = []
            self.regression_context = context

    def winrate(self) -> tuple[float, int]:
        w = sum(a for a, _ in self.wr_window)
        l_ = sum(b for _, b in self.wr_window)
        return w / max(w + l_, 1), w + l_


class MultiTrainer:
    def __init__(self, cfg: dict, run: str, resume: bool, init_from: str | None = None):
        self.cfg = cfg
        self.run_dir = ROOT / "runs" / run
        self.run_dir.mkdir(parents=True, exist_ok=True)
        p, r = cfg["ppo"], cfg["reward"]
        torch.set_num_threads(p["torch_threads"])
        torch.manual_seed(cfg["seed"])
        self.rng = np.random.default_rng(cfg["seed"])
        self.rcfg = RewardConfig(goal=r["goal"], w_ball_progress=r["w_ball_progress"], w_near_ball=r["w_near_ball"],
                                 kick_to_goal=r["kick_to_goal"], kickoff_stall=r.get("kickoff_stall", 1.0),
                                 kickoff_approach=r.get("kickoff_approach", 0.0),
                                 out_penalty=r.get("out_penalty", 0.1), w_spread=r.get("w_spread", 0.0),
                                 gamma=p["gamma"])
        self.catalog = load_catalog(ROOT / cfg.get("tasks_file", "train/tasks.yaml"))
        self.stages = cfg["stages"]
        for st in self.stages:
            for name in st["tasks"]:
                assert name in self.catalog, f"tarea desconocida: {name}"
        self.stage = 0
        self.stage_steps = 0
        self.steps = 0
        self.iteration = 0
        self.task_state: dict[str, dict] = {}  # estado persistente por tarea (sobrevive a reconstruir entornos)
        self.device = resolve_device(p.get("device"))
        from env.haxball_env import U_SELF_DIM
        mc = cfg["model"]
        model_cfg = dict(type=mc.get("type", "set"), self_dim=U_SELF_DIM, ent_dim=ENT_DIM,
                         hidden=mc["hidden"], layers=mc["layers"], ent_hidden=mc.get("ent_hidden", 64),
                         pooling=mc.get("pooling", "meanmax"), ent_layers=mc.get("ent_layers", 2),
                         rule_observation=mc.get("rule_observation", "masked"))
        if model_cfg["type"] == "recurrent_set":
            model_cfg["memory_size"] = mc.get("memory_size", 64)
        self.model = build_model(model_cfg)
        self.model.to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=p["lr"], eps=1e-5)
        lg = cfg["league"]
        self.league = League(max_size=lg["max_size"], pfsp_power=lg["pfsp_power"])
        # Imitación (train/bc.py): arrancar desde el modelo que imita a jugadores reales y/o regularizar
        # con KL(pi_bc || pi) para no alejarse del estilo humano (posiciones, esperar el saque...).
        if init_from:
            cfg["bc_reference"] = str(init_from)
        self.bc_model = None
        ref = cfg.get("bc_reference")
        if ref:
            ck = torch.load(ref, map_location="cpu", weights_only=False)
            if p.get("bc_kl_coef", 0.0) > 0:
                self.bc_model = build_model(ck["model_config"])
                if getattr(self.bc_model, "is_recurrent", False):
                    raise ValueError("La referencia KL de BC debe ser sin memoria; para inicializar desde un "
                                     "recurrente, usar ppo.bc_kl_coef=0")
                self.bc_model.load_state_dict(ck["model"])
                self.bc_model.rule_observation = self.model.rule_observation
                self.bc_model.to(self.device).eval()
                for prm in self.bc_model.parameters():
                    prm.requires_grad_(False)
        if resume and (self.run_dir / "latest.pt").exists():
            self.load(self.run_dir / "latest.pt")
        elif init_from:
            if getattr(self.model, "is_recurrent", False):
                self.model.initialize_from(ck)
            else:
                self.model.load_state_dict(ck["model"])
            print(f"modelo inicial: {init_from} (imitación)" +
                  (f", regularizado con KL x{p['bc_kl_coef']}" if self.bc_model is not None else ""))
        (self.run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        self.build_envs()
        self.writer = SummaryWriter(str(self.run_dir / "tb"))

    # ------------------------------------------------------------ tareas y entornos
    def active_tasks(self) -> list[str]:
        """Tareas de la etapa actual y de todas las anteriores (las viejas quedan como repaso)."""
        names: list[str] = []
        for st in self.stages[: self.stage + 1]:
            for n in st["tasks"]:
                if n not in names:
                    names.append(n)
        return names

    def task_shares(self) -> dict[str, float]:
        """Fracción de MUESTRAS de agente por tarea: pesos de la etapa (1 por defecto), las tareas de etapas
        anteriores con su peso de repaso, x boost por regresión, y nunca menos de min_share."""
        st = self.stages[self.stage]
        w = {}
        for n in self.active_tasks():
            base = st.get("weights", {}).get(n, 1.0 if n in st["tasks"] else self.cfg["schedule"].get("rehearsal_weight", 0.5))
            w[n] = base * self.task_state.get(n, {}).get("boost", 1.0)
        tot = sum(w.values())
        share = {n: v / tot for n, v in w.items()}
        floor = self.cfg["schedule"].get("min_share", 0.05)
        for _ in range(3):  # piso de repaso
            low = {n for n, s in share.items() if s < floor}
            if not low:
                break
            rest = 1.0 - floor * len(low)
            hi = sum(s for n, s in share.items() if n not in low)
            share = {n: (floor if n in low else s * rest / hi) for n, s in share.items()}
        return share

    def build_envs(self) -> None:
        e = self.cfg["env"]
        names = self.active_tasks()
        # Migrar marcas antiguas sin procedencia antes de calcular los pesos.
        # Tampoco reutilizar una marca si se cambió scripted_eps en el config.
        for n in names:
            state = self.task_state.get(n)
            if state:
                expected = (state["opp_stage"], float(self.cfg["curriculum"][state["opp_stage"]]["scripted_eps"]))
                if tuple(state.get("regression_context") or ()) != expected:
                    state.update(best_wr=0.0, boost=1.0, wr_window=[], regression_context=expected)
        self.max_entities = max(self.catalog[n].n_entities for n in names)
        shares = self.task_shares()
        budget = e["agents"]
        self.slots: list[TaskSlot] = []
        for i, n in enumerate(names):
            t = self.catalog[n]
            n_envs = max(1, int(round(shares[n] * budget / (2 * t.n_per_team))))
            # cada tarea tiene su propia config de recompensa: el shaping decae según cuánto lleva ESA tarea
            # en el entrenamiento (ver shaping_for), así un mapa que entra tarde arranca con guía completa
            rcfg = copy.copy(self.rcfg)
            env = make_env(t, n_envs, self.max_entities, rcfg, seed=self.cfg["seed"] + 1000 * self.stage + i,
                           frame_skip=e["frame_skip"], max_ticks=e["max_ticks"],
                           random_reset_prob=e["random_reset_prob"], kickoff_timeout=e.get("kickoff_timeout", 180),
                           action_delay_max=e.get("action_delay_max", 0))
            slot = TaskSlot(t, env)
            # intro_step: paso global en que la tarea entró. Tareas nuevas = ahora; estados guardados antes de
            # existir este campo = 0 (estaban desde el principio o se comportaban así hasta ahora)
            s = self.task_state.setdefault(n, {"opp_stage": 0, "best_wr": 0.0, "boost": 1.0, "steps": 0,
                                               "intro_step": getattr(self, "steps", 0)})
            s.setdefault("intro_step", 0)
            slot.opp_stage, slot.best_wr, slot.boost, slot.steps = s["opp_stage"], s["best_wr"], s["boost"], s["steps"]
            slot.intro_step = s["intro_step"]
            context = s.get("regression_context")
            slot.regression_context = tuple(context) if context is not None else None
            slot.wr_window = [tuple(pair) for pair in s.get("wr_window", [])][-400:]  # ventana por goles
            slot.set_regression_context(self.cfg["curriculum"][slot.opp_stage]["scripted_eps"])
            self.slots.append(slot)
        self.obs_dim = self.slots[0].env.obs_dim
        desc = ", ".join(f"{s.task.name}:{s.N}x{s.P}" for s in self.slots)
        print(f"etapa {self.stage} ({self.stages[self.stage].get('name', '')}): {desc} | entidades {self.max_entities}",
              flush=True)

    def shaping_for(self, s) -> float:
        """Guía (shaping) de una tarea: baja de 1 a 0 en `shaping_decay_steps` pasos GLOBALES contados desde que
        la tarea entró al entrenamiento (no desde el paso 0), así las tareas de las etapas B/C —mapas grandes con
        pocos goles— reciben la misma ayuda que tuvieron las de la etapa A."""
        return max(0.0, 1.0 - (self.steps - s.intro_step) / self.cfg["reward"]["shaping_decay_steps"])

    def sync_task_state(self) -> None:
        for s in self.slots:
            self.task_state[s.task.name] = {"opp_stage": s.opp_stage, "best_wr": s.best_wr, "boost": s.boost,
                                            "steps": s.steps, "intro_step": s.intro_step,
                                            "regression_context": s.regression_context,
                                            "wr_window": list(s.wr_window)}

    # ------------------------------------------------------------ checkpoints
    def save(self, path: Path) -> None:
        self.sync_task_state()
        torch.save({
            "model": self.model.state_dict(), "model_config": self.model.config(), "opt": self.opt.state_dict(),
            "steps": self.steps, "iteration": self.iteration, "stage": self.stage, "stage_steps": self.stage_steps,
            "task_state": self.task_state, "learner_elo": self.league.learner_elo,
            "scripted_elo": self.league.scripted_elo,
            "league": [(m.name, m.model.state_dict(), m.elo, m.wins, m.games) for m in self.league.members],
            "env": {"obs_layout": "universal", "tasks": self.active_tasks(), "max_entities": self.max_entities,
                    "frame_skip": self.cfg["env"]["frame_skip"]},
        }, path)

    def load(self, path: Path) -> None:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        saved_mode = ck["model_config"].get("rule_observation", "full")
        if saved_mode != self.model.rule_observation:
            raise ValueError("El checkpoint usa rule_observation=" + saved_mode +
                             ". Para reanudar sin cambiar la política, configurar model.rule_observation=" +
                             saved_mode + "; para adaptar a masked, usar --init-from en un run nuevo.")
        self.model.load_state_dict(ck["model"])
        self.opt.load_state_dict(ck["opt"])
        self.steps, self.iteration, self.stage = ck["steps"], ck["iteration"], ck["stage"]
        self.stage_steps, self.task_state = ck["stage_steps"], ck["task_state"]
        self.league.learner_elo, self.league.scripted_elo = ck["learner_elo"], ck["scripted_elo"]
        for name, sd, elo, wins, games in ck["league"]:
            m = build_model(ck["model_config"])
            m.load_state_dict(sd)
            self.league.add_snapshot(m.to(self.device), name)
            mem = self.league.members[-1]
            mem.elo, mem.wins, mem.games = elo, wins, games
        print(f"reanudado desde {path} (paso {self.steps:,}, iter {self.iteration}, etapa {self.stage})")

    # ------------------------------------------------------------ rivales
    def assign_modes(self, s: TaskSlot) -> None:
        st = self.cfg["curriculum"][s.opp_stage]
        fr = np.array([st["selfplay"], st["pool"] if self.league.members else 0.0, st["scripted"]])
        fr = fr / fr.sum()
        counts = np.floor(fr * s.N).astype(int)
        counts[0] += s.N - counts.sum()
        modes = np.concatenate([np.full(c, m) for m, c in zip((SELF, POOL, SCRIPTED), counts)])
        self.rng.shuffle(modes)
        opp_id = np.full(s.N, -1)
        pool_envs = np.where(modes == POOL)[0]
        if len(pool_envs):
            ids = self.league.sample(self.cfg["league"]["opponents_per_iter"], self.rng)
            opp_id[pool_envs] = np.array(ids)[np.arange(len(pool_envs)) % len(ids)]
        learner = np.zeros((s.N, s.P), dtype=bool)
        learner[:, : s.T] = True
        learner[modes == SELF, s.T:] = True
        s.modes, s.opp_id, s.learner, s.scripted_eps = modes, opp_id, learner, st["scripted_eps"]

    # ------------------------------------------------------------ actuar
    @torch.no_grad()
    def act(self, obs_list):
        """Una sola pasada de la red para los agentes de todas las tareas."""
        dev = self.device
        flat = np.concatenate([o.reshape(-1, self.obs_dim) for o in obs_list])
        logits, value = self.model(torch.from_numpy(flat).to(dev))
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        logp, a, value = dist.log_prob(a).cpu().numpy(), a.cpu().numpy(), value.cpu().numpy()
        out, k = [], 0
        for s, o in zip(self.slots, obs_list):
            n = s.N * s.P
            acts = a[k:k + n].reshape(s.N, s.P)
            blue = slice(s.T, s.P)
            for oid in np.unique(s.opp_id[s.opp_id >= 0]):
                envs = np.where(s.opp_id == oid)[0]
                ob = torch.from_numpy(o[envs, blue].reshape(-1, self.obs_dim)).to(dev)
                lg = self.league.members[oid].model.logits(ob)
                acts[envs, blue] = torch.distributions.Categorical(logits=lg).sample().cpu().numpy().reshape(len(envs), s.T)
            sc = np.where(s.modes == SCRIPTED)[0]
            if len(sc):
                acts[sc, blue] = scripted_actions(s.env, np.arange(s.T, s.P), s.scripted_eps, self.rng)[sc]
            out.append((acts, logp[k:k + n].reshape(s.N, s.P), value[k:k + n].reshape(s.N, s.P)))
            k += n
        return out

    @torch.no_grad()
    def values(self, obs: np.ndarray) -> np.ndarray:
        v = self.model(torch.from_numpy(obs.reshape(-1, self.obs_dim)).to(self.device))[1]
        return v.cpu().numpy().reshape(obs.shape[:2])

    # ------------------------------------------------------------ loop
    def train(self) -> None:
        try:
            while self.steps < self.cfg["ppo"]["total_steps"]:
                self.iterate()
        except KeyboardInterrupt:
            print("interrumpido: guardando...")
        self.save(self.run_dir / "latest.pt")
        print(f"guardado en {self.run_dir / 'latest.pt'}")

    def iterate(self) -> None:
        if getattr(self.model, "is_recurrent", False):
            raise ValueError("Usar RecurrentTrainer para entrenar con memoria")
        cfg, p = self.cfg, self.cfg["ppo"]
        L, D = p["rollout_len"], self.obs_dim
        if not hasattr(self, "_obs"):
            self._obs = [s.env.reset() for s in self.slots]
        frac = self.steps / p["total_steps"]
        lr = lerp(p["lr"], p["lr_final"], frac)
        for g in self.opt.param_groups:
            g["lr"] = lr
        ent_coef = lerp(p["ent_coef"], p["ent_coef_final"], frac)
        self.rcfg.shaping_coef = max(0.0, 1.0 - self.steps / cfg["reward"]["shaping_decay_steps"])  # sólo para el log
        for s in self.slots:
            s.env.rcfg.shaping_coef = self.shaping_for(s)
        for s in self.slots:
            self.assign_modes(s)
            s.buf = {k: np.zeros((L, s.N, s.P) + sh, dt) for k, sh, dt in
                     (("obs", (D,), np.float32), ("act", (), np.int64), ("logp", (), np.float32),
                      ("val", (), np.float32), ("rew", (), np.float32))}
            s.buf["done"] = np.zeros((L, s.N), np.float32)
            s.pool_goals = {}
        t0 = time.time()
        obs = self._obs
        for t in range(L):
            decisions = self.act(obs)
            new_obs = []
            for s, o, (acts, logp, val) in zip(self.slots, obs, decisions):
                b = s.buf
                b["obs"][t], b["act"][t], b["logp"][t], b["val"][t] = o, acts, logp, val
                o2, rew, done, info = s.env.step(acts)
                if info["truncated"].any():
                    tr = np.where(info["truncated"])[0]
                    rew[tr] += p["gamma"] * self.values(info["final_obs"][tr])
                b["rew"][t], b["done"][t] = rew, done
                g = info["goal"]
                for m in (SELF, POOL, SCRIPTED):
                    sel = s.modes == m
                    s.goals[m][0] += int((g[sel] == 1).sum())
                    s.goals[m][1] += int((g[sel] == -1).sum())
                for e in np.where((g != 0) & (s.modes == POOL))[0]:
                    s.pool_goals.setdefault(s.opp_id[e], [0, 0])[0 if g[e] == 1 else 1] += 1
                sc = (g != 0) & (s.modes == SCRIPTED)
                if sc.any():
                    self.league.record(None, int((g[sc] == 1).sum()), int((g[sc] == -1).sum()))
                new_obs.append(o2)
            obs = new_obs
        self._obs = obs
        t_roll = time.time() - t0

        # GAE por tarea y muestras de los agentes que aprenden
        parts = {k: [] for k in ("obs", "act", "logp", "adv", "ret")}
        for s, o in zip(self.slots, obs):
            for k_, (a_, b_) in s.pool_goals.items():
                self.league.record(int(k_), a_, b_)
            b = s.buf
            last_v = self.values(o)
            adv = np.zeros_like(b["rew"])
            gae = np.zeros((s.N, s.P), np.float32)
            for t in reversed(range(L)):
                nv = last_v if t == L - 1 else b["val"][t + 1]
                nd = 1.0 - b["done"][t][:, None]
                delta = b["rew"][t] + p["gamma"] * nv * nd - b["val"][t]
                gae = delta + p["gamma"] * p["gae_lambda"] * nd * gae
                adv[t] = gae
            m = np.broadcast_to(s.learner[None], (L, s.N, s.P)).reshape(-1)
            parts["obs"].append(b["obs"].reshape(-1, D)[m])
            parts["act"].append(b["act"].reshape(-1)[m])
            parts["logp"].append(b["logp"].reshape(-1)[m])
            parts["adv"].append(adv.reshape(-1)[m])
            parts["ret"].append((adv + b["val"]).reshape(-1)[m])
            s.steps += int(m.sum())
            s.buf = None
        dev = self.device
        batch = {k: torch.from_numpy(np.concatenate(v)).to(dev) for k, v in parts.items()}
        self.model.update_norm(batch["obs"])
        n = len(batch["act"])
        self.steps += n
        self.stage_steps += n
        t1 = time.time()
        self.bc_coef = lerp(p.get("bc_kl_coef", 0.0), p.get("bc_kl_final", 0.0), frac) if self.bc_model is not None else 0.0
        stats = self.update(batch, ent_coef)
        t_upd = time.time() - t1
        self.iteration += 1

        self.opponent_curriculum()
        lg = cfg["league"]
        if self.iteration % lg["snapshot_every"] == 0 and any(s.opp_stage >= 1 for s in self.slots):
            self.league.add_snapshot(self.model, f"it{self.iteration}")
        if self.iteration % cfg["log"]["checkpoint_every"] == 0:
            self.save(self.run_dir / "latest.pt")
            self.save(self.run_dir / f"ckpt_{self.iteration:06d}.pt")
            if cfg["log"].get("replay_every") and self.iteration % cfg["log"]["replay_every"] == 0:
                self.spawn_replay(self.run_dir / f"ckpt_{self.iteration:06d}.pt")
        self.log(stats, lr, n, t_roll, t_upd)
        self.maybe_rebalance_or_advance()

    def spawn_replay(self, ckpt: Path) -> None:
        """Graba un partido del checkpoint en un proceso aparte (no frena el entrenamiento):
        runs/<run>/replays/it<iter>_<tarea>_vs_<rival>.html, rotando tareas y alternando rival (bot / sí mismo)."""
        import subprocess
        import sys
        names = self.active_tasks()
        k = self.iteration // self.cfg["log"]["replay_every"]
        task = names[k % len(names)]
        rival = "scripted" if k % 2 == 0 else str(ckpt)
        out_dir = self.run_dir / "replays"
        out_dir.mkdir(exist_ok=True)
        out = out_dir / f"it{self.iteration:06d}_{task}_vs_{'bot' if rival == 'scripted' else 'si_mismo'}.html"
        # Un solo hilo y prioridad normal: con prioridad baja, el entrenamiento (que ocupa toda la CPU) no le
        # dejaba tiempo y las grabaciones quedaban trabadas horas. En un hilo es liviano y termina en ~1-2 min.
        import os
        env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMBA_NUM_THREADS="1")
        log = open(out_dir / "render.log", "a", encoding="utf-8")
        subprocess.Popen([sys.executable, "-m", "eval.render", str(ckpt), rival, "--task", task,
                          "--minutes", str(self.cfg["log"].get("replay_minutes", 2.0)), "--out", str(out)],
                         cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env)
        print(f"      grabando replay: {out.name}", flush=True)

    def opponent_curriculum(self) -> None:
        cur = self.cfg["curriculum"]
        for s in self.slots:
            s.set_regression_context(cur[s.opp_stage]["scripted_eps"])
            gs = s.goals[SCRIPTED]
            # Ventana medida en GOLES, no en iteraciones: se guardan las iteraciones más recientes necesarias
            # para juntar `min_goals`. Con muchas tareas repartiéndose el tiempo, las de pocos goles (Big 3v3,
            # AHA...) nunca llegaban a 200 goles en 20 iteraciones y no podían avanzar de rival.
            need = self.cfg.get("schedule", {}).get("min_goals", 200)
            s.wr_window = (s.wr_window + [(gs[0], gs[1])])[-400:]
            while len(s.wr_window) > 1 and sum(a + b for a, b in s.wr_window[1:]) >= need:
                s.wr_window.pop(0)
            s.goals[SCRIPTED] = [0, 0]
            wr, n = s.winrate()
            if n >= 200:
                s.best_wr = max(s.best_wr, wr)
            if n >= 200 and wr >= cur[s.opp_stage]["advance_winrate"] and s.opp_stage < len(cur) - 1:
                s.opp_stage += 1
                s.set_regression_context(cur[s.opp_stage]["scripted_eps"])
                print(f"*** {s.task.name}: rivales -> {cur[s.opp_stage]['name']} ***", flush=True)
                if not self.league.members:
                    self.league.add_snapshot(self.model, f"it{self.iteration}")

    def maybe_rebalance_or_advance(self) -> None:
        sch = self.cfg["schedule"]
        if self.iteration % sch.get("rebalance_every", 25):
            return
        changed = False
        # regresiones: una tarea que cayó respecto de su mejor marca recibe más peso hasta recuperarse
        for s in self.slots:
            wr, n = s.winrate()
            if n < 200:
                continue
            dropped = wr < s.best_wr - sch.get("regression_drop", 0.15)
            new_boost = min(s.boost * sch.get("regression_boost", 1.5), 4.0) if dropped else max(1.0, s.boost / 1.5)
            if abs(new_boost - s.boost) > 1e-6:
                if dropped:
                    print(f"!!! regresión en {s.task.name}: {wr:.2f} vs mejor {s.best_wr:.2f} -> peso x{new_boost:.2f}",
                          flush=True)
                s.boost = new_boost
                changed = True
        # avance de etapa
        st = self.stages[self.stage]
        last_opp = len(self.cfg["curriculum"]) - 1
        ready = all(s.opp_stage >= last_opp for s in self.slots)
        if self.stage < len(self.stages) - 1 and (
                (ready and self.stage_steps >= st.get("min_steps", 0)) or self.stage_steps >= st.get("max_steps", float("inf"))):
            self.save(self.run_dir / f"fin_etapa{self.stage}.pt")
            self.stage += 1
            self.stage_steps = 0
            print(f"*** etapa {self.stage}: {self.stages[self.stage].get('name', '')} ***", flush=True)
            changed = True
        if changed:
            self.sync_task_state()
            self.build_envs()
            del self._obs

    def log(self, stats, lr, n, t_roll, t_upd) -> None:
        sps = n / max(t_roll + t_upd, 1e-9)
        w = self.writer
        for k, v in stats.items():
            w.add_scalar(k, v, self.steps)
        w.add_scalar("stage", self.stage, self.steps)
        w.add_scalar("training/goal_elo", self.league.learner_elo, self.steps)
        w.add_scalar("sps", sps, self.steps)
        w.add_scalar("shaping_coef", self.rcfg.shaping_coef, self.steps)
        for s in self.slots:
            wr, goals = s.winrate()
            w.add_scalar(f"task/{s.task.name}/goal_share_vs_scripted", wr, self.steps)
            w.add_scalar(f"task/{s.task.name}/goals_in_window", goals, self.steps)
            w.add_scalar(f"task/{s.task.name}/opp_stage", s.opp_stage, self.steps)
            w.add_scalar(f"task/{s.task.name}/shaping", s.env.rcfg.shaping_coef, self.steps)
        if self.iteration % self.cfg["log"]["every"] == 0:
            print(f"it {self.iteration:5d} | etapa {self.stage} | pasos {self.steps / 1e6:7.1f}M | {sps:6.0f}/s "
                  f"(rollout {t_roll:.1f}s upd {t_upd:.1f}s) | elo por goles {self.league.learner_elo:5.0f} | "
                  f"ent {stats['entropy']:.2f} | shaping {self.rcfg.shaping_coef:.2f}", flush=True)
            cells = []
            for s in self.slots:
                wr, ng = s.winrate()
                cells.append(f"{s.task.name} r{s.opp_stage} {wr:.2f}({ng})")
            print("      proporción de goles vs bot: " + " | ".join(cells), flush=True)

    def update(self, b, ent_coef):
        p = self.cfg["ppo"]
        n = len(b["act"])
        adv = (b["adv"] - b["adv"].mean()) / (b["adv"].std() + 1e-8)
        mb = min(p["minibatch"], n)
        agg = {"pg_loss": 0.0, "v_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0, "clipfrac": 0.0, "bc_kl": 0.0}
        cnt = 0
        bc_coef = getattr(self, "bc_coef", 0.0)
        for _ in range(p["epochs"]):
            perm = torch.randperm(n, device=b["obs"].device)
            for s in range(0, n - mb + 1, mb):
                i = perm[s:s + mb]
                logits, v = self.model(b["obs"][i])
                dist = torch.distributions.Categorical(logits=logits)
                ratio = torch.exp(dist.log_prob(b["act"][i]) - b["logp"][i])
                a = adv[i]
                pg = -torch.min(ratio * a, ratio.clamp(1 - p["clip"], 1 + p["clip"]) * a).mean()
                vl = 0.5 * (v - b["ret"][i]).pow(2).mean()
                ent = dist.entropy().mean()
                loss = pg + p["vf_coef"] * vl - ent_coef * ent
                if bc_coef > 0:
                    with torch.no_grad():
                        lb = torch.log_softmax(self.bc_model.logits(b["obs"][i]), -1)
                    bc_kl = (lb.exp() * (lb - torch.log_softmax(logits, -1))).sum(-1).mean()
                    loss = loss + bc_coef * bc_kl
                    agg["bc_kl"] += bc_kl.item()
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), p["max_grad_norm"])
                self.opt.step()
                with torch.no_grad():
                    agg["pg_loss"] += pg.item()
                    agg["v_loss"] += vl.item()
                    agg["entropy"] += ent.item()
                    agg["approx_kl"] += ((ratio - 1) - torch.log(ratio)).mean().item()
                    agg["clipfrac"] += ((ratio - 1).abs() > p["clip"]).float().mean().item()
                cnt += 1
        return {k: v / max(cnt, 1) for k, v in agg.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "train" / "config_multi.yaml"))
    ap.add_argument("--run", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--init-from", default=None, help="checkpoint de imitación (train/bc.py) para arrancar")
    ap.add_argument("--override", nargs="*", default=[], help="clave.sub=valor, ej. env.agents=256")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    for ov in args.override:
        k, v = ov.split("=", 1)
        d = cfg
        *path, last = k.split(".")
        for part in path:
            d = d[int(part)] if isinstance(d, list) else d[part]
        if isinstance(d, list):
            d[int(last)] = yaml.safe_load(v)
        else:
            d[last] = yaml.safe_load(v)
    trainer_class = MultiTrainer
    if cfg["model"].get("type") == "recurrent_set":
        from .recurrent_ppo import RecurrentTrainer
        trainer_class = RecurrentTrainer
    trainer_class(cfg, args.run or cfg["run_name"], args.resume, args.init_from).train()


if __name__ == "__main__":
    main()
