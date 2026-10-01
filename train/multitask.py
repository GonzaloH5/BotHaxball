"""PPO multi-tarea: un solo modelo para muchos mapas y formatos a la vez (que no se encasille).

python -m train.multitask --config train/config_multi.yaml [--run nombre] [--resume] [--override clave=valor ...]

Cómo funciona:
  * Cada iteración juega partidos de TODAS las tareas activas (mapa × formato × reglas, ver train/tasks.yaml)
    y hace un único update con todas las muestras mezcladas. La obs "universal" (env/haxball_env.py) describe
    el mapa y rellena entidades, y SetActorCritic (train/model.py) acepta cualquier cantidad de jugadores.
  * Currículo por AMPLITUD con repaso: las etapas (`stages`) van sumando tareas; una tarea que ya entró nunca
    baja de `min_share` de las muestras, así no se olvida. Se pasa de etapa cuando todas las tareas activas
    llegaron a la liga del currículo de rivales y se cumplió `min_steps` (o al llegar a `max_steps`).
  * Currículo de rivales POR TAREA, promovido por puntos de partidos completos y no por proporción de goles.
  * Una sola liga de snapshots para todos los formatos (el modelo juega cualquiera); no se vacía entre etapas.
  * Control de regresiones: si el winrate contra el bot de una tarea cae `regression_drop` respecto de su mejor
    marca, se sube su peso (x`regression_boost`) hasta que se recupere.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from bots.scripted import scripted_actions
from env.rewards import RewardConfig
from env.tasks import load_catalog, make_env

from .league import League
from .checkpoints import atomic_torch_save
from .cuda_decisions import DECISION_BACKENDS, CudaDecisionGraph, sample_decisions
from .model import SetActorCritic, build_model
from .ppo_selfplay import lerp, resolve_device
from .runtime import CudaRolloutTransfer, PpoBatchTransfer, batch_to_device, cpu_budget, load_config

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
        self.match_window: list[tuple[int, int, int, int, int, int]] = []
        self.match_results = {SELF: np.zeros(6, dtype=np.int64),
                              POOL: np.zeros(6, dtype=np.int64),
                              SCRIPTED: np.zeros(6, dtype=np.int64)}
        self.promotion_streak = 0
        self.best_wr = 0.0
        self.boost = 1.0
        self.goals = {SELF: [0, 0], POOL: [0, 0], SCRIPTED: [0, 0]}
        self.steps = 0
        self.intro_step = 0   # paso global en que la tarea entró al entrenamiento (decaimiento del shaping)
        self.regression_context = None
        self.mode_context = None
        self.mode_carry = np.zeros(3, dtype=np.float64)
        self.match_finished = np.ones(self.N, dtype=bool)
        self.metric_version = 0

    def set_regression_context(self, scripted_eps, scripted_policy="r2", scripted_style=-1):
        """Una marca sólo es comparable contra la misma dificultad/política de rival."""
        context = (self.opp_stage, float(scripted_eps), str(scripted_policy), int(scripted_style))
        if self.regression_context != context:
            self.best_wr, self.boost = 0.0, 1.0
            self.wr_window = []
            self.match_window = []
            self.promotion_streak = 0
            self.regression_context = context

    def winrate(self) -> tuple[float, int]:
        w = sum(a for a, _ in self.wr_window)
        l_ = sum(b for _, b in self.wr_window)
        return w / max(w + l_, 1), w + l_

    def match_performance(self) -> tuple[float, int, tuple[int, int, int], float]:
        totals = np.array(self.match_window, dtype=np.int64).sum(axis=0) if self.match_window else np.zeros(6, int)
        wins, draws, losses, _, _, scoreless = totals
        games = int(wins + draws + losses)
        return ((wins + 0.5 * draws) / max(games, 1), games,
                (int(wins), int(draws), int(losses)), scoreless / max(games, 1))


class MultiTrainer:
    def __init__(self, cfg: dict, run: str, resume: bool, init_from: str | None = None):
        self.cfg = cfg
        self._validate_scripted_readiness()
        self.optimize_rollout = cfg.get("runtime", {}).get("optimize_rollout", True)
        self.optimize_cpu = cfg.get("runtime", {}).get("optimize_cpu", False)
        self.cuda_decisions = cfg.get("runtime", {}).get("cuda_decisions", "legacy")
        if self.cuda_decisions not in DECISION_BACKENDS:
            raise ValueError(f"runtime.cuda_decisions debe ser uno de {DECISION_BACKENDS}")
        self._decision_profile = None
        self.run_dir = ROOT / "runs" / run
        self.run_dir.mkdir(parents=True, exist_ok=True)
        p, r = cfg["ppo"], cfg["reward"]
        budget = cpu_budget()
        threads = p.get("torch_threads", 8)
        auto_threads = 2 if resolve_device(p.get("device")).type == "cuda" else 8
        torch.set_num_threads(min(budget, auto_threads) if threads == "auto" else int(threads))
        from numba import config as numba_config, set_num_threads, get_num_threads
        physics_threads = p.get("numba_threads")
        if physics_threads is not None:
            set_num_threads(min(budget, numba_config.NUMBA_NUM_THREADS) if physics_threads == "auto"
                            else int(physics_threads))
        torch.manual_seed(cfg["seed"])
        self.rng = np.random.default_rng(cfg["seed"])
        reward_fields = {field.name for field in fields(RewardConfig)}
        reward_args = {key: value for key, value in r.items() if key in reward_fields}
        reward_args["gamma"] = p["gamma"]
        self.rcfg = RewardConfig(**reward_args)
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
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("ppo.device=cuda requiere PyTorch con CUDA y una GPU NVIDIA disponible")
        print(f"runtime: {self.device} | torch {torch.get_num_threads()} hilos | física {get_num_threads()} hilos",
              flush=True)
        self._rollout_transfer = (CudaRolloutTransfer(self.device, reuse_device=self.optimize_rollout)
                                  if self.device.type == "cuda" else None)
        self._batch_transfer = (PpoBatchTransfer(self.device)
                                if self.device.type == "cuda" and cfg.get("runtime", {}).get("reuse_ppo_batch", True)
                                else None)
        self._decision_graph = (CudaDecisionGraph(self.device, self.cuda_decisions)
                                if self.device.type == "cuda" and self.cuda_decisions in ("auto", "graph") else None)
        if self.device.type == "cuda":
            print(f"decisiones CUDA: {self.cuda_decisions}", flush=True)
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
            # Los YAML de Windows también deben poder reanudarse en Linux.
            ref_path = Path(str(ref).replace("\\", "/"))
            ref = str(ref_path if ref_path.is_absolute() else ROOT / ref_path)
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
            if hasattr(self.model, "initialize_from"):
                self.model.initialize_from(ck)
            else:
                self.model.load_state_dict(ck["model"])
            if cfg.get("seed_league_from_init", False):
                self._restore_league(ck, include_learner=True)
            print(f"modelo inicial: {init_from} (migración compatible)" +
                  (f", regularizado con KL x{p['bc_kl_coef']}" if self.bc_model is not None else ""))
        (self.run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        self.build_envs()
        self.writer = SummaryWriter(str(self.run_dir / "tb"))

    def _validate_scripted_readiness(self) -> None:
        """Impide usar R3 con un informe ausente, fallido o de código anterior."""
        uses_r3 = any(str(stage.get("scripted_policy", "r2")).lower() == "r3"
                      and float(stage.get("scripted", 0)) > 0 for stage in self.cfg.get("curriculum", []))
        readiness = self.cfg.get("scripted_readiness", {})
        if not uses_r3 or not readiness.get("required", False):
            return
        raw = Path(str(readiness.get("report", "reports/scripted_r3_gate.json")).replace("\\", "/"))
        path = raw if raw.is_absolute() else ROOT / raw
        if not path.exists():
            raise RuntimeError(f"R3 requiere un gate aprobado: ejecutar python -m eval.scripted_gate --out {path}")
        report = json.loads(path.read_text(encoding="utf-8"))
        if not report.get("gate", {}).get("passed", False):
            raise RuntimeError(f"R3 no está aprobado por {path}: {report.get('gate', {}).get('reasons', [])}")
        recorded = report.get("source_sha256", {})
        for relative in ("bots/scripted.py", "env/haxball_env.py", "env/pegeche.py"):
            current = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            if recorded.get(relative) != current:
                raise RuntimeError(f"El gate R3 quedó obsoleto porque cambió {relative}; volver a evaluarlo")

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
        anteriores con su peso de repaso, x boost por regresión, y nunca menos de min_share.
        fixed_weights ignora boosts, también los guardados en un checkpoint anterior.
        El presupuesto incluye agentes rivales: la fracción final de PPO depende del modo de rival.
        """
        st = self.stages[self.stage]
        w = {}
        for n in self.active_tasks():
            base = st.get("weights", {}).get(n, 1.0 if n in st["tasks"] else self.cfg["schedule"].get("rehearsal_weight", 0.5))
            boost = 1.0 if st.get("fixed_weights", False) else self.task_state.get(n, {}).get("boost", 1.0)
            w[n] = base * boost
        tot = sum(w.values())
        if tot <= 0:
            raise ValueError("La etapa debe declarar al menos una tarea con peso positivo")
        share = {n: v / tot for n, v in w.items()}
        # Un perfil fijo es un contrato exacto: en particular, peso 0 significa
        # que la tarea queda fuera de esta fase. El piso sólo protege el repaso
        # de las etapas adaptativas.
        if st.get("fixed_weights", False):
            return share
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
        self._reset_decision_graph()
        e = self.cfg["env"]
        names = self.active_tasks()
        shares = self.task_shares()
        # En perfiles de reparto fijo, un peso cero es una exclusión real: no
        # se crea un entorno residual ni se consumen muestras para esa tarea.
        names = [n for n in names if shares[n] > 0]
        # Migrar marcas antiguas sin procedencia antes de calcular los pesos.
        # Tampoco reutilizar una marca si se cambió scripted_eps en el config.
        for n in names:
            state = self.task_state.get(n)
            if state:
                version = self.cfg.get("task_metric_versions", {}).get(n, 0)
                if state.get("metric_version", 0) != version:
                    state.update(best_wr=0.0, boost=1.0, wr_window=[], metric_version=version)
                cst = self.cfg["curriculum"][state["opp_stage"]]
                expected = (state["opp_stage"], float(cst["scripted_eps"]),
                            str(cst.get("scripted_policy", "r2")), int(cst.get("scripted_style", -1)))
                previous = tuple(state.get("regression_context") or ())
                legacy = (state["opp_stage"], float(cst["scripted_eps"]))
                # Checkpoints anteriores guardaban sólo (stage, eps). Si siguen usando
                # R2 mixed, es exactamente el mismo rival: actualizar metadata sin perder
                # la ventana ya medida. Cualquier cambio real de política sí la invalida.
                if previous == legacy and expected[2:] == ("r2", -1):
                    state["regression_context"] = expected
                elif previous != expected:
                    state.update(best_wr=0.0, boost=1.0, wr_window=[], regression_context=expected)
        self.max_entities = max(self.catalog[n].n_entities for n in names)
        budget = e["agents"]
        self.slots: list[TaskSlot] = []
        for i, n in enumerate(names):
            t = self.catalog[n]
            n_envs = max(1, int(round(shares[n] * budget / (2 * t.n_per_team))))
            # cada tarea tiene su propia config de recompensa: el shaping decae según cuánto lleva ESA tarea
            # en el entrenamiento (ver shaping_for), así un mapa que entra tarde arranca con guía completa
            rcfg = copy.copy(self.rcfg)
            overrides = self.cfg.get("task_reward_overrides", {}).get(n, {})
            for key, value in overrides.items():
                if key not in ("w_defense_support", "defense_shaping_floor"):
                    raise ValueError(f"Recompensa por tarea desconocida: {n}.{key}")
                if not np.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError(f"{n}.{key} debe estar entre 0 y 1")
                setattr(rcfg, key, float(value))
            env = make_env(t, n_envs, self.max_entities, rcfg, seed=self.cfg["seed"] + 1000 * self.stage + i,
                           frame_skip=e["frame_skip"], max_ticks=e["max_ticks"],
                           random_reset_prob=e["random_reset_prob"], kickoff_timeout=e.get("kickoff_timeout", 180),
                           action_delay_max=e.get("action_delay_max", 0),
                           optimize_rollout=getattr(self, "optimize_rollout", True))
            slot = TaskSlot(t, env)
            # intro_step: paso global en que la tarea entró. Tareas nuevas = ahora; estados guardados antes de
            # existir este campo = 0 (estaban desde el principio o se comportaban así hasta ahora)
            s = self.task_state.setdefault(n, {"opp_stage": 0, "best_wr": 0.0, "boost": 1.0, "steps": 0,
                                               "intro_step": getattr(self, "steps", 0)})
            s.setdefault("intro_step", 0)
            slot.opp_stage, slot.best_wr, slot.boost, slot.steps = s["opp_stage"], s["best_wr"], s["boost"], s["steps"]
            slot.intro_step = s["intro_step"]
            slot.metric_version = self.cfg.get("task_metric_versions", {}).get(n, 0)
            context = s.get("regression_context")
            slot.regression_context = tuple(context) if context is not None else None
            slot.wr_window = [tuple(pair) for pair in s.get("wr_window", [])][-400:]  # ventana por goles
            slot.match_window = [tuple(row) for row in s.get("match_window", [])][-400:]
            slot.promotion_streak = int(s.get("promotion_streak", 0))
            cst = self.cfg["curriculum"][slot.opp_stage]
            slot.set_regression_context(cst["scripted_eps"], cst.get("scripted_policy", "r2"),
                                        cst.get("scripted_style", -1))
            context = s.get("mode_context")
            slot.mode_context = tuple(context) if context is not None else None
            slot.mode_carry = np.array(s.get("mode_carry", [0.0, 0.0, 0.0]), dtype=np.float64)
            self.slots.append(slot)
        self.obs_dim = self.slots[0].env.obs_dim
        self._policy_groups = None
        desc = ", ".join(f"{s.task.name}:{s.N}x{s.P}" for s in self.slots)
        print(f"etapa {self.stage} ({self.stages[self.stage].get('name', '')}): {desc} | entidades {self.max_entities}",
              flush=True)
        if self.stages[self.stage].get("fixed_weights", False):
            total = sum(s.N * s.P for s in self.slots)
            actual = " | ".join(f"{s.task.name} {100 * s.N * s.P / total:.1f}%" for s in self.slots)
            print(f"reparto fijo de agentes: {actual} (PPO usa sólo las filas que aprenden)", flush=True)

    def shaping_for(self, s) -> float:
        """Shaping por dificultad del rival.

        R0: guía ligera para tareas todavía inmaduras.
        R1: guía mínima.
        R2/R3: sin shaping auxiliar.
        """
        by_stage = self.cfg["reward"].get("shaping_by_opp_stage")

        if by_stage is not None:
            idx = min(s.opp_stage, len(by_stage) - 1)
            return float(by_stage[idx])

        # Compatibilidad con configs antiguas.
        return max(
            0.0,
            1.0 - (self.steps - s.intro_step)
            / self.cfg["reward"]["shaping_decay_steps"],
        )

    def sync_task_state(self) -> None:
        for s in self.slots:
            self.task_state[s.task.name] = {"opp_stage": s.opp_stage, "best_wr": s.best_wr, "boost": s.boost,
                                            "steps": s.steps, "intro_step": s.intro_step,
                                            "regression_context": s.regression_context,
                                            "wr_window": list(s.wr_window),
                                            "match_window": list(s.match_window),
                                            "promotion_streak": s.promotion_streak,
                                            "mode_context": s.mode_context,
                                            "mode_carry": s.mode_carry.tolist(),
                                            "metric_version": s.metric_version}

    # ------------------------------------------------------------ checkpoints
    def save(self, path: Path) -> None:
        self.sync_task_state()
        atomic_torch_save({
            "checkpoint_version": 2,
            "model": self.model.state_dict(), "model_config": self.model.config(), "opt": self.opt.state_dict(),
            "steps": self.steps, "iteration": self.iteration, "stage": self.stage, "stage_steps": self.stage_steps,
            "task_state": self.task_state, "learner_elo": self.league.learner_elo,
            "scripted_elo": self.league.scripted_elo,
            "league": [{"name": m.name, "model": m.model.state_dict(), "model_config": m.model.config(),
                        "elo": m.elo, "wins": m.wins, "games": m.games} for m in self.league.members],
            "env": {"obs_layout": "universal", "tasks": [s.task.name for s in self.slots],
                    "max_entities": self.max_entities,
                    "frame_skip": self.cfg["env"]["frame_skip"]},
        }, path)

    def load(self, path: Path) -> None:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        saved_mode = ck["model_config"].get("rule_observation", "full")
        if saved_mode != self.model.rule_observation:
            raise ValueError("El checkpoint usa rule_observation=" + saved_mode +
                             ". Para reanudar sin cambiar la política, configurar model.rule_observation=" +
                             saved_mode + "; para adaptar a masked, usar --init-from en un run nuevo.")
        if not 0 <= ck["stage"] < len(self.stages):
            raise ValueError(f"El checkpoint está en etapa {ck['stage']}, pero este perfil sólo define "
                             f"{len(self.stages)} etapas. Usar una configuración compatible; "
                             "no se reinicia ni cambia de etapa automáticamente.")
        self.model.load_state_dict(ck["model"])
        self.opt.load_state_dict(ck["opt"])
        self.steps, self.iteration, self.stage = ck["steps"], ck["iteration"], ck["stage"]
        self.stage_steps, self.task_state = ck["stage_steps"], ck["task_state"]
        self.league.learner_elo, self.league.scripted_elo = ck["learner_elo"], ck["scripted_elo"]
        self._restore_league(ck)
        print(f"reanudado desde {path} (paso {self.steps:,}, iter {self.iteration}, etapa {self.stage})")

    def _restore_league(self, ck, include_learner=False) -> None:
        """Carga ligas v1/v2; v2 permite mezclar arquitecturas históricas."""
        if include_learner:
            self.league.learner_elo = ck.get("learner_elo", self.league.learner_elo)
            self.league.scripted_elo = ck.get("scripted_elo", self.league.scripted_elo)
        for saved in ck["league"]:
            if isinstance(saved, dict):
                name, sd = saved["name"], saved["model"]
                elo, wins, games = saved["elo"], saved["wins"], saved["games"]
                member_config = saved.get("model_config", ck["model_config"])
            else:
                name, sd, elo, wins, games = saved
                member_config = ck["model_config"]
            m = build_model(member_config)
            m.load_state_dict(sd)
            self.league.add_snapshot(m.to(self.device), name)
            mem = self.league.members[-1]
            mem.elo, mem.wins, mem.games = elo, wins, games
        if include_learner:
            source = build_model(ck["model_config"])
            source.load_state_dict(ck["model"])
            self.league.add_snapshot(source.to(self.device), "baseline_inicial")

    # ------------------------------------------------------------ rivales
    def assign_modes(self, s: TaskSlot) -> None:
        st = self.cfg["curriculum"][s.opp_stage]
        fr = np.array([st["selfplay"], st["pool"] if self.league.members else 0.0, st["scripted"]])
        fr = fr / fr.sum()
        # Reparto compensado: las fracciones pequeñas reciben entornos a lo
        # largo de los rollouts, sin imponer un mínimo que distorsione el 5%.
        # Se conserva el saldo por tarea al reconstruir entornos y reanudar.
        context = tuple(fr)
        if s.mode_context != context:
            s.mode_carry = np.zeros(3, dtype=np.float64)
            s.mode_context = context
        quota = fr * s.N
        counts = np.floor(quota).astype(int)
        debt = s.mode_carry + quota - counts
        for _ in range(s.N - int(counts.sum())):
            mode = int(np.argmax(np.where(fr > 0, debt, -np.inf)))
            counts[mode] += 1
            debt[mode] -= 1
        s.mode_carry = debt
        proposed = np.concatenate([np.full(c, m) for m, c in zip((SELF, POOL, SCRIPTED), counts)])
        self.rng.shuffle(proposed)
        hold_for_match = bool(self.cfg.get("schedule", {}).get("hold_opponent_for_match", False))
        if hold_for_match and hasattr(s, "modes"):
            # Un rival no cambia a mitad de un partido de max_ticks. Sólo se
            # reasignan filas cuyo partido terminó durante el rollout anterior.
            modes = s.modes.copy()
            available = np.flatnonzero(s.match_finished)
            locked = ~s.match_finished
            locked_counts = np.bincount(modes[locked], minlength=3)
            wanted = np.maximum(counts - locked_counts, 0)
            assignments = np.concatenate([np.full(c, m) for m, c in enumerate(wanted)])
            if len(assignments) < len(available):
                extra = self.rng.choice(3, len(available) - len(assignments), p=fr)
                assignments = np.concatenate((assignments, extra))
            self.rng.shuffle(assignments)
            modes[available] = assignments[:len(available)]
            opp_id = s.opp_id.copy()
            opp_id[available] = -1
            new_rows = available
        else:
            modes = proposed
            opp_id = np.full(s.N, -1)
            new_rows = np.arange(s.N)
        pool_envs = new_rows[modes[new_rows] == POOL]
        if len(pool_envs):
            ids = self.league.sample(self.cfg["league"]["opponents_per_iter"], self.rng)
            opp_id[pool_envs] = np.array(ids)[np.arange(len(pool_envs)) % len(ids)]
        learner = np.zeros((s.N, s.P), dtype=bool)
        learner[:, : s.T] = True
        learner[modes == SELF, s.T:] = True
        s.modes, s.opp_id, s.learner = modes, opp_id, learner
        if hold_for_match:
            s.match_finished[:] = False
        s.scripted_eps = st["scripted_eps"]
        s.scripted_policy = st.get("scripted_policy", "r2")
        s.scripted_style = st.get("scripted_style", -1)

    def record_match_results(self, s: TaskSlot, info: dict) -> None:
        """Acumula partidos completos, separados del reset PPO que ocurre tras cada gol."""
        finished = info.get("match_done")
        scores = info.get("final_score")
        if finished is None or scores is None or not np.asarray(finished).any():
            return
        finished = np.asarray(finished, dtype=bool)
        for mode in (SELF, POOL, SCRIPTED):
            rows = finished & (s.modes == mode)
            if not rows.any():
                continue
            red, blue = scores[rows, 0], scores[rows, 1]
            bucket = s.match_results[mode]
            bucket[0] += int((red > blue).sum())
            bucket[1] += int((red == blue).sum())
            bucket[2] += int((red < blue).sum())
            bucket[3] += int(red.sum())
            bucket[4] += int(blue.sum())
            bucket[5] += int(((red + blue) == 0).sum())
        if self.cfg.get("schedule", {}).get("hold_opponent_for_match", False):
            s.match_finished |= finished

    # ------------------------------------------------------------ actuar
    def _reset_decision_graph(self):
        graph = getattr(self, "_decision_graph", None)
        if graph is not None:
            graph.reset()

    def _prepare_policy_groups(self):
        """Los modos/rivales no cambian dentro del rollout: índices una vez, no 128 veces."""
        grouped, offset, sizes = {}, 0, {}
        self._cpu_policy_routes = []
        for slot in self.slots:
            routes = []
            slot.scripted_rows = np.flatnonzero(slot.modes == SCRIPTED) if hasattr(slot, "modes") else None
            for oid in np.unique(slot.opp_id[slot.opp_id >= 0]):
                rows = np.where(slot.opp_id == oid)[0]
                indices = (offset + rows[:, None] * slot.P + np.arange(slot.T, slot.P)).reshape(-1)
                grouped.setdefault(int(oid), []).append(indices)
                start = sizes.get(int(oid), 0)
                sizes[int(oid)] = start + len(indices)
                routes.append((int(oid), rows, slice(start, start + len(indices))))
            self._cpu_policy_routes.append(routes)
            offset += slot.N * slot.P
        self._policy_groups = [(oid, torch.as_tensor(np.concatenate(chunks), device=self.device))
                               for oid, chunks in grouped.items()]
        return self._policy_groups

    @torch.no_grad()
    def act(self, obs_list):
        """Una sola pasada de la red para los agentes de todas las tareas."""
        if self.device.type == "cuda":
            return self._act_cuda(obs_list)
        if getattr(self, "optimize_cpu", False):
            return self._act_cpu_grouped(obs_list)
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
                acts[sc, blue] = scripted_actions(
                    s.env, np.arange(s.T, s.P), s.scripted_eps, self.rng, env_indices=sc,
                    policy=s.scripted_policy, style=s.scripted_style,
                )
            out.append((acts, logp[k:k + n].reshape(s.N, s.P), value[k:k + n].reshape(s.N, s.P)))
            k += n
        return out

    @torch.no_grad()
    def _act_cpu_grouped(self, obs_list):
        """Un forward por snapshot; muestreos por tarea en el orden original."""
        if getattr(self, "_policy_groups", None) is None:
            self._prepare_policy_groups()
        shape = (sum(o.shape[0] * o.shape[1] for o in obs_list), self.obs_dim)
        flat = getattr(self, "_cpu_obs", None)
        if flat is None or flat.shape != shape:
            flat = self._cpu_obs = np.empty(shape, dtype=np.float32)
        offset = 0
        for obs in obs_list:
            rows = obs.reshape(-1, self.obs_dim)
            np.copyto(flat[offset:offset + len(rows)], rows)
            offset += len(rows)
        tensor = torch.from_numpy(flat)
        logits, value = self.model(tensor)
        dist = torch.distributions.Categorical(logits=logits, validate_args=False)
        actions = dist.sample()
        logp = dist.log_prob(actions).numpy()
        actions, value = actions.numpy(), value.numpy()
        opponent_logits = {
            oid: self.league.members[oid].model.logits(tensor.index_select(0, indices))
            for oid, indices in self._policy_groups
        }
        out, offset = [], 0
        for slot, routes in zip(self.slots, self._cpu_policy_routes):
            count = slot.N * slot.P
            acts = actions[offset:offset + count].reshape(slot.N, slot.P)
            for oid, rows, group_slice in routes:
                dist = torch.distributions.Categorical(logits=opponent_logits[oid][group_slice],
                                                       validate_args=False)
                acts[rows, slot.T:] = dist.sample().numpy().reshape(len(rows), slot.T)
            rows = slot.scripted_rows
            if len(rows):
                acts[rows, slot.T:] = scripted_actions(
                    slot.env, np.arange(slot.T, slot.P), slot.scripted_eps, self.rng,
                    env_indices=rows, policy=slot.scripted_policy, style=slot.scripted_style)
            out.append((acts, logp[offset:offset + count].reshape(slot.N, slot.P),
                        value[offset:offset + count].reshape(slot.N, slot.P)))
            offset += count
        return out

    @torch.no_grad()
    def _policy_decisions(self, flat):
        """Agrupa por snapshot los rivales de todas las tareas en una pasada por rival."""
        logits, values = self.model(flat)
        dist = torch.distributions.Categorical(logits=logits, validate_args=False)
        actions = dist.sample()
        logp = dist.log_prob(actions)
        groups = getattr(self, "_policy_groups", None)
        if groups is None or not getattr(self, "optimize_rollout", True):
            groups = self._prepare_policy_groups()
        for oid, indices in groups:
            logits = self.league.members[oid].model.logits(flat[indices])
            actions[indices] = torch.distributions.Categorical(logits=logits, validate_args=False).sample()
        return actions, logp, values

    @torch.no_grad()
    def _mixed_policy_logits(self, flat):
        """Un forward por modelo, sin RNG. Los valores siempre pertenecen al aprendiz."""
        logits, values = self.model(flat)
        groups = getattr(self, "_policy_groups", None)
        if groups is None:
            groups = self._prepare_policy_groups()
        for oid, indices in groups:
            logits[indices] = self.league.members[oid].model.logits(flat[indices])
        return logits, values

    def _cuda_inference(self, flat):
        graph = self._decision_graph
        if graph is None:
            return self._mixed_policy_logits(flat)
        before = graph.capture_seconds, graph.captures, graph.replays
        result = graph.run(self._mixed_policy_logits, flat)
        profile = self._decision_profile
        if profile is not None:
            profile.capture_seconds += graph.capture_seconds - before[0]
            profile.captures += graph.captures - before[1]
            profile.replays += graph.replays - before[2]
        return result

    @staticmethod
    def _sample_cuda(logits, values):
        return sample_decisions(logits, values)

    def _act_cuda(self, obs_list):
        transfer = self._rollout_transfer
        profile = self._decision_profile
        host_start = time.perf_counter() if profile is not None else 0.0
        if profile is not None:
            profile.mark(0)
        arrays = [o.reshape(-1, self.obs_dim) for o in obs_list]
        flat = (transfer.upload_many(arrays) if self.optimize_rollout
                else transfer.upload(np.concatenate(arrays)))
        if profile is not None:
            profile.mark(1)
        if self.cuda_decisions == "legacy":
            decisions = self._policy_decisions(flat)
            if profile is not None:
                profile.mark(2)  # legacy incluye el muestreo de cada rival dentro de esta fase
        else:
            logits, values = self._cuda_inference(flat)
            if profile is not None:
                profile.mark(2)
            decisions = self._sample_cuda(logits, values)
        if profile is not None:
            profile.mark(3)
        transfer.enqueue_output(*decisions)
        if profile is not None:
            profile.mark(4)
            transfer.record_ready()  # incluir el último evento sin otra sincronización
        host_submit = time.perf_counter() - host_start if profile is not None else 0.0
        # La copia GPU->CPU está en vuelo: calcular bots sobre los estados actuales, sin avanzar física.
        scripted = []
        for slot in self.slots:
            rows = slot.scripted_rows if self.optimize_rollout else np.where(slot.modes == SCRIPTED)[0]
            acts = (scripted_actions(
                        slot.env, np.arange(slot.T, slot.P), slot.scripted_eps, self.rng,
                        env_indices=rows, policy=slot.scripted_policy, style=slot.scripted_style,
                    ) if len(rows) else None)
            scripted.append((rows, acts))
        wait_start = time.perf_counter() if profile is not None else 0.0
        actions, logp, values = transfer.wait_output()
        if profile is not None:
            profile.finish(host_submit, time.perf_counter() - wait_start)
        out, offset = [], 0
        for slot, (rows, bot_actions) in zip(self.slots, scripted):
            size = slot.N * slot.P
            acts = actions[offset:offset + size].reshape(slot.N, slot.P)
            if len(rows):
                acts[rows, slot.T:] = bot_actions
            out.append((acts, logp[offset:offset + size].reshape(slot.N, slot.P),
                        values[offset:offset + size].reshape(slot.N, slot.P)))
            offset += size
        return out

    @torch.no_grad()
    def values(self, obs: np.ndarray) -> np.ndarray:
        tensor = torch.from_numpy(obs.reshape(-1, self.obs_dim)).to(self.device)
        if (self.device.type == "cpu" and getattr(self, "optimize_cpu", False)
                and hasattr(self.model, "value_only")):
            v = self.model.value_only(tensor)
        else:
            v = self.model(tensor)[1]
        return v.cpu().numpy().reshape(obs.shape[:2])

    def values_many(self, observations):
        """Un bootstrap para todas las tareas/timeouts, con una sola sincronización CUDA."""
        if not observations:
            return []
        grouped_cpu = self.device.type == "cpu" and getattr(self, "optimize_cpu", False)
        if not grouped_cpu and (not self.optimize_rollout or self.device.type != "cuda"):
            return [self.values(o) for o in observations]
        flat = np.concatenate([o.reshape(-1, self.obs_dim) for o in observations])
        values = self.values(flat[:, None, :]).reshape(-1)
        result, offset = [], 0
        for o in observations:
            size = o.shape[0] * o.shape[1]
            result.append(values[offset:offset + size].reshape(o.shape[:2]))
            offset += size
        return result

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
        iteration_start = time.perf_counter()
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
        for s in self.slots:
            s.env.rcfg.shaping_coef = self.shaping_for(s)

        # Sólo para mostrar un valor resumen en consola.
        self.rcfg.shaping_coef = float(
            np.mean([s.env.rcfg.shaping_coef for s in self.slots])
        )

        for s in self.slots:
            self.assign_modes(s)
            cache = getattr(s, "_buffer_cache", None)
            if not self.optimize_rollout or cache is None or cache["obs"].shape != (L, s.N, s.P, D):
                allocate = np.empty if self.optimize_rollout else np.zeros
                cache = {k: allocate((L, s.N, s.P) + sh, dt) for k, sh, dt in
                         (("obs", (D,), np.float32), ("act", (), np.int64), ("logp", (), np.float32),
                          ("val", (), np.float32), ("rew", (), np.float32))}
                cache["done"] = allocate((L, s.N), np.float32)
                if self.optimize_rollout:
                    s._buffer_cache = cache
            s.buf = cache
            s.pool_goals = {}
        self._policy_groups = None
        self._reset_decision_graph()
        if self.optimize_rollout:
            self._prepare_policy_groups()
        t0 = time.perf_counter()
        obs = self._obs
        for t in range(L):
            decisions = self.act(obs)
            new_obs = []
            timeouts = []
            for s, o, (acts, logp, val) in zip(self.slots, obs, decisions):
                b = s.buf
                b["obs"][t], b["act"][t], b["logp"][t], b["val"][t] = o, acts, logp, val
                o2, rew, done, info = s.env.step(acts)
                if info["truncated"].any():
                    tr = np.where(info["truncated"])[0]
                    timeouts.append((s, tr, info["final_obs"][tr]))
                b["rew"][t], b["done"][t] = rew, done
                g = info["goal"]
                for m in ((SELF, POOL, SCRIPTED) if g.any() else ()):
                    sel = s.modes == m
                    s.goals[m][0] += int((g[sel] == 1).sum())
                    s.goals[m][1] += int((g[sel] == -1).sum())
                for e in np.where((g != 0) & (s.modes == POOL))[0]:
                    s.pool_goals.setdefault(s.opp_id[e], [0, 0])[0 if g[e] == 1 else 1] += 1
                sc = (g != 0) & (s.modes == SCRIPTED)
                if sc.any():
                    self.league.record(None, int((g[sc] == 1).sum()), int((g[sc] == -1).sum()))
                self.record_match_results(s, info)
                new_obs.append(o2)
            for (slot, rows, _), bootstrap in zip(timeouts, self.values_many([x[2] for x in timeouts])):
                slot.buf["rew"][t, rows] += p["gamma"] * bootstrap
            obs = new_obs
        self._obs = obs
        graph = self._decision_graph
        capture_seconds = graph.capture_seconds if graph is not None else 0.0
        captures = graph.captures if graph is not None else 0
        replays = graph.replays if graph is not None else 0
        # No permitir replay de buffers de RunningNorm anteriores después del update.
        self._reset_decision_graph()
        t_roll = time.perf_counter() - t0

        t_prepare = time.perf_counter()
        # GAE por tarea y muestras de los agentes que aprenden
        parts = {k: [] for k in ("obs", "act", "logp", "adv", "ret")}
        for s, last_v in zip(self.slots, self.values_many(obs)):
            for k_, (a_, b_) in s.pool_goals.items():
                self.league.record(int(k_), a_, b_)
            b = s.buf
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
        batch = batch_to_device(parts, dev, self._batch_transfer)
        self.model.update_norm(batch["obs"])
        n = len(batch["act"])
        self.steps += n
        self.stage_steps += n
        t1 = time.perf_counter()
        self.bc_coef = lerp(p.get("bc_kl_coef", 0.0), p.get("bc_kl_final", 0.0), frac) if self.bc_model is not None else 0.0
        stats = self.update(batch, ent_coef)
        if self._batch_transfer is not None:
            self._batch_transfer.mark_consumed()  # proteger también cualquier lector PPO pendiente
        update_end = time.perf_counter()
        t_upd = update_end - t1
        stats["timing/setup_seconds"] = t0 - iteration_start
        stats["timing/prepare_seconds"] = t1 - t_prepare
        stats["timing/rollout_seconds"] = t_roll
        stats["timing/update_seconds"] = t_upd
        stats["timing/cuda_capture_seconds"] = capture_seconds
        stats["runtime/cuda_graph_captures"] = captures
        stats["runtime/cuda_graph_replays"] = replays
        if self._batch_transfer is not None:
            for phase, seconds in self._batch_transfer.last_timings.items():
                stats[f"timing/ppo_batch_{phase}_seconds"] = seconds
            stats["runtime/ppo_batch_allocations"] = self._batch_transfer.last_allocations
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
        log_start = time.perf_counter()
        stats["timing/maintenance_seconds"] = log_start - update_end
        self.log(stats, lr, n, t_roll, t_upd)
        log_end = time.perf_counter()
        self.maybe_rebalance_or_advance()
        end = time.perf_counter()
        self._last_iteration_timings = dict(
            setup=t0 - iteration_start, maintenance=log_start - update_end,
            logging=log_end - log_start, schedule=end - log_end, total=end - iteration_start)

    def spawn_replay(self, ckpt: Path) -> None:
        """Graba un partido del checkpoint en un proceso aparte (no frena el entrenamiento):
        runs/<run>/replays/it<iter>_<tarea>_vs_<rival>.html, rotando tareas y alternando rival (bot / sí mismo)."""
        import subprocess
        import sys
        names = [s.task.name for s in self.slots]
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
            cst = cur[s.opp_stage]
            s.set_regression_context(cst["scripted_eps"], cst.get("scripted_policy", "r2"),
                                     cst.get("scripted_style", -1))
            gs = s.goals[SCRIPTED]
            if "advance_points" in cst:
                row = tuple(int(v) for v in s.match_results[SCRIPTED])
                has_new_evaluation = bool(sum(row[:3]))
                if has_new_evaluation:
                    s.match_window = (s.match_window + [row])[-400:]
                s.match_results[SCRIPTED][:] = 0
                need_games = int(self.cfg.get("schedule", {}).get("min_games", 128))
                while len(s.match_window) > 1 and sum(sum(x[:3]) for x in s.match_window[1:]) >= need_games:
                    s.match_window.pop(0)
                points, games, _, _ = s.match_performance()
                if games >= need_games and has_new_evaluation:
                    s.best_wr = max(s.best_wr, points)
                    s.promotion_streak = s.promotion_streak + 1 if points >= cst["advance_points"] else 0
                if (games >= need_games and s.promotion_streak >= 2
                        and s.opp_stage < len(cur) - 1):
                    s.opp_stage += 1
                    next_stage = cur[s.opp_stage]
                    s.set_regression_context(next_stage["scripted_eps"],
                                             next_stage.get("scripted_policy", "r2"),
                                             next_stage.get("scripted_style", -1))
                    print(f"*** {s.task.name}: rivales -> {next_stage['name']} por puntos ***", flush=True)
                    if not self.league.members:
                        self.league.add_snapshot(self.model, f"it{self.iteration}")
                s.goals[SCRIPTED] = [0, 0]
                continue
            # Ventana medida en GOLES, no en iteraciones: se guardan las iteraciones más recientes necesarias
            # para juntar `min_goals`. Con muchas tareas repartiéndose el tiempo, las de pocos goles (Big 3v3,
            # AHA...) nunca llegaban a 200 goles en 20 iteraciones y no podían avanzar de rival.
            task_min_goals = self.cfg.get("schedule", {}).get("min_goals_by_task", {})
            need = task_min_goals.get(
                s.task.name,
                self.cfg.get("schedule", {}).get("min_goals", 200)
            )
            s.wr_window = (s.wr_window + [(gs[0], gs[1])])[-400:]
            while len(s.wr_window) > 1 and sum(a + b for a, b in s.wr_window[1:]) >= need:
                s.wr_window.pop(0)
            s.goals[SCRIPTED] = [0, 0]
            wr, n = s.winrate()
            if n >= need:
                s.best_wr = max(s.best_wr, wr)

            if n >= need and wr >= cur[s.opp_stage]["advance_winrate"] and s.opp_stage < len(cur) - 1:
                s.opp_stage += 1
                cst = cur[s.opp_stage]
                s.set_regression_context(cst["scripted_eps"], cst.get("scripted_policy", "r2"),
                                         cst.get("scripted_style", -1))
                print(f"*** {s.task.name}: rivales -> {cst['name']} ({cst.get('scripted_policy', 'r2')}) ***",
                      flush=True)
                if not self.league.members:
                    self.league.add_snapshot(self.model, f"it{self.iteration}")

    def maybe_rebalance_or_advance(self) -> None:
        sch = self.cfg["schedule"]
        if self.iteration % sch.get("rebalance_every", 25):
            return
        changed = False
        st = self.stages[self.stage]
        # regresiones: una tarea que cayó respecto de su mejor marca recibe más peso hasta recuperarse
        # Un perfil fijo conserva las métricas/boosts previos, pero no cambia el reparto ni reinicia entornos.
        rebalance_slots = [] if st.get("fixed_weights", False) else self.slots
        for s in rebalance_slots:
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
        sps = n / max(t_roll + t_upd + stats.get("timing/prepare_seconds", 0.0), 1e-9)
        w = self.writer
        for k, v in stats.items():
            w.add_scalar(k, v, self.steps)
        w.add_scalar("stage", self.stage, self.steps)
        w.add_scalar("training/goal_elo", self.league.learner_elo, self.steps)
        w.add_scalar("sps", sps, self.steps)
        w.add_scalar("shaping_coef", self.rcfg.shaping_coef, self.steps)
        for s in self.slots:
            wr, goals = s.winrate()
            points, games, record, scoreless = s.match_performance()
            w.add_scalar(f"task/{s.task.name}/goal_share_vs_scripted", wr if goals else float("nan"), self.steps)
            w.add_scalar(f"task/{s.task.name}/points_vs_scripted", points if games else float("nan"), self.steps)
            w.add_scalar(f"task/{s.task.name}/scoreless_vs_scripted", scoreless if games else float("nan"), self.steps)
            w.add_scalar(f"task/{s.task.name}/goals_in_window", goals, self.steps)
            w.add_scalar(f"task/{s.task.name}/opp_stage", s.opp_stage, self.steps)
            w.add_scalar(f"task/{s.task.name}/scripted_r3",
                         1.0 if getattr(s, "scripted_policy", "r2") == "r3" else 0.0, self.steps)
            w.add_scalar(f"task/{s.task.name}/shaping", s.env.rcfg.shaping_coef, self.steps)
        if self.iteration % self.cfg["log"]["every"] == 0:
            print(f"it {self.iteration:5d} | etapa {self.stage} | pasos {self.steps / 1e6:7.1f}M | {sps:6.0f}/s "
                  f"(rollout {t_roll:.1f}s prep {stats.get('timing/prepare_seconds', 0.0):.2f}s upd {t_upd:.1f}s) | elo por goles {self.league.learner_elo:5.0f} | "
                  f"ent {stats['entropy']:.2f} | shaping {self.rcfg.shaping_coef:.2f}", flush=True)
            cells = []
            for s in self.slots:
                wr, ng = s.winrate()
                points, games, record, _ = s.match_performance()
                match_result = f"pts {points:.2f} {record[0]}-{record[1]}-{record[2]}" if games else "pts sin datos"
                goal_result = f"goles {wr:.2f}({ng})" if ng else "goles sin datos"
                cells.append(f"{s.task.name} r{s.opp_stage} {match_result} {goal_result}")
            print("      rendimiento vs bot: " + " | ".join(cells), flush=True)

    def update(self, b, ent_coef):
        p = self.cfg["ppo"]
        n = len(b["act"])
        adv = (b["adv"] - b["adv"].mean()) / (b["adv"].std() + 1e-8)
        mb = min(p["minibatch"], n)
        keys = ("pg_loss", "v_loss", "entropy", "approx_kl", "clipfrac", "bc_kl")
        agg = torch.zeros(len(keys), device=b["obs"].device)
        cnt = 0
        bc_coef = getattr(self, "bc_coef", 0.0)
        cpu_fast = b["obs"].device.type == "cpu" and getattr(self, "optimize_cpu", False)
        obs_buffer = b["obs"].new_empty((mb, b["obs"].shape[1])) if cpu_fast else None
        profile = self.cfg.get("runtime", {}).get("profile_update", False)
        timings = dict(bc_cache=0.0, gather=0.0, forward_loss=0.0, backward=0.0, optimizer=0.0)
        started = time.perf_counter() if profile else 0.0
        # La referencia BC está en eval y congelada: obs y normalizadores de
        # ese modelo no cambian durante PPO. Cache local, nunca en checkpoint.
        bc_logp = None
        if (bc_coef > 0 and b["obs"].device.type == "cpu" and p["epochs"] > 1
                and self.cfg.get("runtime", {}).get("cache_bc_logits_cpu", False)):
            with torch.no_grad():
                for start in range(0, n, mb):
                    chunk = torch.log_softmax(self.bc_model.logits(b["obs"][start:start + mb]), -1)
                    if bc_logp is None:
                        bc_logp = chunk.new_empty((n, chunk.shape[-1]))
                    bc_logp[start:start + len(chunk)].copy_(chunk)
        if profile:
            timings["bc_cache"] = time.perf_counter() - started
        for _ in range(p["epochs"]):
            perm = torch.randperm(n, device=b["obs"].device)
            for s in range(0, n - mb + 1, mb):
                i = perm[s:s + mb]
                started = time.perf_counter() if profile else 0.0
                obs = (torch.index_select(b["obs"], 0, i, out=obs_buffer) if cpu_fast else b["obs"][i])
                if profile:
                    now = time.perf_counter()
                    timings["gather"] += now - started
                    started = now
                logits, v = self.model(obs)
                dist = torch.distributions.Categorical(logits=logits, validate_args=False)
                ratio = torch.exp(dist.log_prob(b["act"][i]) - b["logp"][i])
                a = adv[i]
                pg = -torch.min(ratio * a, ratio.clamp(1 - p["clip"], 1 + p["clip"]) * a).mean()
                vl = 0.5 * (v - b["ret"][i]).pow(2).mean()
                ent = dist.entropy().mean()
                loss = pg + p["vf_coef"] * vl - ent_coef * ent
                bc_kl = loss.new_zeros(())
                if bc_coef > 0:
                    with torch.no_grad():
                        lb = (bc_logp[i] if bc_logp is not None else
                              torch.log_softmax(self.bc_model.logits(obs), -1))
                    bc_kl = (lb.exp() * (lb - torch.log_softmax(logits, -1))).sum(-1).mean()
                    loss = loss + bc_coef * bc_kl
                self.opt.zero_grad(set_to_none=True)
                if profile:
                    now = time.perf_counter()
                    timings["forward_loss"] += now - started
                    started = now
                loss.backward()
                if profile:
                    now = time.perf_counter()
                    timings["backward"] += now - started
                    started = now
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), p["max_grad_norm"])
                self.opt.step()
                if profile:
                    timings["optimizer"] += time.perf_counter() - started
                with torch.no_grad():
                    agg += torch.stack((pg, vl, ent, ((ratio - 1) - torch.log(ratio)).mean(),
                                        ((ratio - 1).abs() > p["clip"]).float().mean(), bc_kl))
                cnt += 1
        # Una sola sincronización para estadísticas al terminar todos los minibatches.
        result = dict(zip(keys, (agg / max(cnt, 1)).cpu().tolist()))
        if profile:
            result.update({f"timing/update_{key}_seconds": value for key, value in timings.items()})
        return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "train" / "config_multi.yaml"))
    ap.add_argument("--run", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--init-from", default=None, help="checkpoint de imitación (train/bc.py) para arrancar")
    ap.add_argument("--override", nargs="*", default=[], help="clave.sub=valor, ej. env.agents=256")
    args = ap.parse_args()
    cfg = load_config(args.config)
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
