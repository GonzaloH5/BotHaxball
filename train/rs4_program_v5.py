"""Programa RS4 v5: una sola fase, arranque desde imitación humana.

Diagnóstico que lo motiva (reports/rs4_v5_diagnostico_20261003.md): en v3 el
91% de los partidos entre redes terminaba 0-0, la recompensa era casi sólo el
gol y el 45% de los compañeros eran controladores congelados heterogéneos. La
política dejó de cambiar (KL ~0,001) y jugaba en bloque detrás del balón.

v5 mantiene la misma interfaz que ``ProgramState`` (v3) para reutilizar
``RS4V3Trainer``, pero con calendarios simples:

* sin fases, contratos ni deudas de habilidad;
* compañeros casi siempre aprendices (convenciones de equipo propias);
* ancla KL al BC humano fuerte y con piso;
* LR adaptativo que sí sube cuando la KL es baja, con techo decreciente.
"""
from __future__ import annotations

import copy
import math


VERSION = 5


def default_program_v5(start_steps=0):
    return dict(
        version=VERSION, start_steps=start_steps,
        total_budget_steps=3_000_000_000,
        evaluation_every_steps=200_000_000,
        critic_warmup_steps=0,
        snapshot_every_steps=25_000_000,
        drills=.10,
        drill_weights=dict(corner=.4, lateral=.3, goal_kick=.3),
        opponents=dict(selfplay=.50, pool=.40, scripted=.10),
        frozen_teammates=.10,
        reward=dict(ball_progress=.10, near_ball=.02, spread=.05, pass_possession_cap=.05,
                    possession_change=.01, no_goal_penalty=.30, no_goal_step_penalty=0., restart_potential=.011,
                    restart_bonus=.02, gamma=.995),
        lr=dict(initial=3e-4, minimum=5e-5, ceiling_initial=4e-4, ceiling_final=1.5e-4,
                low_kl=.003, high_kl=.008, epoch_stop_kl=.015,
                every_iterations=10, increase=1.15, decrease=.75),
        entropy=dict(initial=.005, final=.003),
        bc=dict(initial=.05, final=.01),
    )


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} debe ser finito")
    return float(value)


def _integer(value, name, *, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} debe ser entero >= {minimum}")
    return value


class ProgramStateV5:
    phase_index = 0
    objective_contract = None
    skill_debts = ()

    def __init__(self, config, saved=None):
        merged = default_program_v5()
        for key, value in copy.deepcopy(config).items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key].update(value)
            else:
                merged[key] = value
        self.config = merged
        self._validate()
        self.relative_steps = 0
        self.lr = float(self.config["lr"]["initial"])
        self.last_lr_iteration = -1
        self.last_endpoint_iteration = -1
        self.lr_kl_window = []
        self.lr_events = []
        self.last_evaluation_steps = 0
        self.last_snapshot_steps = 0
        self.evaluations = []
        if saved is not None:
            self._restore(saved)

    @classmethod
    def from_config(cls, cfg, checkpoint_state=None):
        raw = cfg.get("rs4_program")
        if raw is None:
            raise ValueError("Falta rs4_program")
        return cls(raw, checkpoint_state)

    def _validate(self):
        cfg = self.config
        if cfg.get("version") != VERSION:
            raise ValueError("ProgramStateV5 requiere rs4_program.version=5")
        _integer(cfg["start_steps"], "start_steps")
        for field in ("total_budget_steps", "evaluation_every_steps", "snapshot_every_steps"):
            _integer(cfg[field], field, minimum=1)
        _integer(cfg["critic_warmup_steps"], "critic_warmup_steps")
        if not 0 <= _finite(cfg["drills"], "drills") <= .4:
            raise ValueError("drills debe estar en [0, 0.4]: al menos 60% partidos completos")
        if not 0 <= _finite(cfg["frozen_teammates"], "frozen_teammates") <= 1:
            raise ValueError("frozen_teammates fuera de rango")
        mix = cfg["opponents"]
        if set(mix) != {"selfplay", "pool", "scripted"} or not math.isclose(sum(mix.values()), 1) \
                or any(_finite(v, "opponents") < 0 for v in mix.values()):
            raise ValueError("opponents: selfplay/pool/scripted no negativos que sumen 1")
        weights = cfg["drill_weights"]
        if not weights or any(_finite(v, "drill_weight") < 0 for v in weights.values()) or not sum(weights.values()) > 0:
            raise ValueError("drill_weights inválidos")
        for key, value in cfg["reward"].items():
            if _finite(value, f"reward.{key}") < 0:
                raise ValueError(f"reward.{key} no puede ser negativo")
        if not 0 < cfg["reward"]["gamma"] < 1:
            raise ValueError("reward.gamma debe estar en (0, 1)")
        if cfg["reward"]["restart_bonus"] > .02:
            raise ValueError("restart_bonus admite como máximo 0.02")
        if cfg["reward"]["no_goal_step_penalty"] / (1 - cfg["reward"]["gamma"]) >= .25:
            raise ValueError("no_goal_step_penalty/(1-γ) debe ser < 0.25 (multa por trabar un saque)")
        lr = cfg["lr"]
        for key in ("initial", "minimum", "ceiling_initial", "ceiling_final", "low_kl", "high_kl",
                    "epoch_stop_kl", "increase", "decrease"):
            _finite(lr[key], f"lr.{key}")
        if not 0 < lr["minimum"] <= lr["initial"] <= lr["ceiling_initial"] or lr["ceiling_final"] < lr["minimum"]:
            raise ValueError("Rango LR inválido")
        if not 0 <= lr["low_kl"] < lr["high_kl"] < lr["epoch_stop_kl"] or not lr["increase"] > 1 or not 0 < lr["decrease"] < 1:
            raise ValueError("Control KL inválido")
        _integer(lr["every_iterations"], "lr.every_iterations", minimum=1)
        for name in ("entropy", "bc"):
            if not 0 <= _finite(cfg[name]["final"], name) <= _finite(cfg[name]["initial"], name):
                raise ValueError(f"Calendario {name} inválido")

    # ---------------------------------------------------------------- presupuesto
    @property
    def effective_branch_limit(self):
        return self.config["total_budget_steps"]

    @property
    def remaining_steps(self):
        return max(0, self.effective_branch_limit - self.relative_steps)

    @property
    def complete(self):
        return self.remaining_steps == 0

    @property
    def phase(self):
        return dict(id="V5", name="Juego_completo")

    def _fraction(self):
        return min(1., self.relative_steps / self.effective_branch_limit)

    def _linear(self, schedule):
        return schedule["initial"] + self._fraction() * (schedule["final"] - schedule["initial"])

    @property
    def critic_warmup(self):
        """Al arrancar desde un PPO con otra recompensa, recalibrar V antes de mover π."""
        return self.relative_steps < self.config["critic_warmup_steps"]

    @property
    def entropy_coef(self):
        return self._linear(self.config["entropy"])

    @property
    def bc_coef(self):
        return self._linear(self.config["bc"])

    @property
    def lr_ceiling(self):
        lr = self.config["lr"]
        return lr["ceiling_initial"] + self._fraction() * (lr["ceiling_final"] - lr["ceiling_initial"])

    def advance_steps(self, n):
        _integer(n, "useful_steps")
        self.relative_steps += n
        self.lr = min(self.lr, self.lr_ceiling)

    @property
    def evaluation_due(self):
        return self.relative_steps - self.last_evaluation_steps >= self.config["evaluation_every_steps"]

    @property
    def snapshot_due(self):
        return self.relative_steps - self.last_snapshot_steps >= self.config["snapshot_every_steps"]

    def mark_snapshot(self):
        self.last_snapshot_steps = self.relative_steps

    def update_lr(self, endpoint_kl, iteration):
        _integer(iteration, "iteration")
        if self.critic_warmup:
            return self.lr  # la KL ~0 del calentamiento no debe inflar el LR
        kl = _finite(endpoint_kl, "endpoint_kl")
        if kl < -1e-6:
            raise ValueError("La KL de final de update no puede ser negativa")
        kl = max(kl, 0.)  # redondeo: con la política congelada la estimación ronda -0
        cfg = self.config["lr"]
        if iteration > self.last_endpoint_iteration:
            self.last_endpoint_iteration = iteration
            self.lr_kl_window = (self.lr_kl_window + [kl])[-cfg["every_iterations"]:]
        if iteration == self.last_lr_iteration or iteration % cfg["every_iterations"]:
            return self.lr
        self.last_lr_iteration = iteration
        previous, decision = self.lr, "hold"
        mean = sum(self.lr_kl_window) / len(self.lr_kl_window)
        # A diferencia de v3 se usa la media, no el máximo: un pico aislado no
        # bloquea la subida cuando la política casi no se mueve.
        if mean < cfg["low_kl"]:
            self.lr, decision = self.lr * cfg["increase"], "increase"
        elif mean > cfg["high_kl"]:
            self.lr, decision = self.lr * cfg["decrease"], "decrease"
        self.lr = min(self.lr_ceiling, max(cfg["minimum"], self.lr))
        self.lr_events = (self.lr_events + [dict(iteration=iteration, steps=self.relative_steps, window_mean=mean,
                                                 previous_lr=previous, lr=self.lr, decision=decision)])[-64:]
        return self.lr

    def settings(self):
        cfg, reward = self.config, self.config["reward"]
        weights = {key: float(value) for key, value in cfg["drill_weights"].items()}
        total = sum(weights.values())
        weights = {key: value / total for key, value in weights.items()}
        return dict(phase_id="V5", phase_name="Juego_completo", reward_version=5,
                    guide_coef=0., exercise_fraction=cfg["drills"], exercise_weights=weights,
                    opponent_mix=dict(cfg["opponents"]), frozen_teammates_fraction=cfg["frozen_teammates"],
                    teammate_learner_weights=(.2, .3, .5), scenario_difficulty=.5,
                    restart_execute_bonus=reward["restart_bonus"], restart_bonus=reward["restart_bonus"],
                    restart_potential_coef=reward["restart_potential"], restart_stall_penalty=.25,
                    ball_progress=reward["ball_progress"], near_ball=reward["near_ball"], spread=reward["spread"],
                    pass_possession_cap=reward["pass_possession_cap"],
                    possession_change=reward["possession_change"], no_goal_penalty=reward["no_goal_penalty"],
                    no_goal_step_penalty=reward["no_goal_step_penalty"],
                    gamma=reward["gamma"], recovery=False, skill_debts=[],
                    entropy_coef=self.entropy_coef, bc_coef=self.bc_coef, lr=self.lr, freeze_normalizers=True)

    def record_evaluation(self, report):
        self.last_evaluation_steps = self.relative_steps
        self.evaluations = (self.evaluations + [dict(steps=self.relative_steps, report=copy.deepcopy(report))])[-16:]
        return dict(advanced=False)

    # ---------------------------------------------------------------- persistencia
    _FIELDS = ("relative_steps", "lr", "last_lr_iteration", "last_endpoint_iteration", "lr_kl_window",
               "lr_events", "last_evaluation_steps", "last_snapshot_steps", "evaluations")

    def state_dict(self):
        return dict(version=VERSION, config=copy.deepcopy(self.config),
                    **{name: copy.deepcopy(getattr(self, name)) for name in self._FIELDS})

    def _restore(self, saved):
        if saved.get("version") != VERSION:
            raise ValueError("El checkpoint no pertenece a un programa RS4 v5")
        stored = copy.deepcopy(saved.get("config") or {})
        for key, value in default_program_v5().items():
            # Campos agregados después (critic_warmup_steps, reward.no_goal_step_penalty...).
            stored[key] = {**value, **stored.get(key, {})} if isinstance(value, dict) else stored.get(key, value)
        if stored != self.config:
            raise ValueError("El programa guardado no coincide con el config: no cambiar calendarios al reanudar")
        for name in self._FIELDS:
            if name not in saved:
                raise ValueError(f"Estado RS4 v5 incompleto: {name}")
            setattr(self, name, copy.deepcopy(saved[name]))
        _integer(self.relative_steps, "relative_steps")


def program_from_config(cfg, checkpoint_state=None):
    """Elegir v3 o v5 según rs4_program.version, sin tocar la ruta v3."""
    raw = cfg.get("rs4_program")
    if raw is None:
        raise ValueError("Falta rs4_program")
    if raw.get("version") == VERSION:
        return ProgramStateV5.from_config(cfg, checkpoint_state)
    from .rs4_program import ProgramState
    return ProgramState.from_config(cfg, checkpoint_state)
