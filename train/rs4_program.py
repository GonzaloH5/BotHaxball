"""Estado persistente del programa RS4 v3, independiente de los pasos heredados.

No crea archivos, modelos ni entornos. Las muestras contabilizadas son únicamente
las filas que participan en PPO. Evaluación y selección usan informes externos.
"""
from __future__ import annotations

import copy
import math


VERSION = 3
TOTAL_BUDGET = 6_000_000_000
PILOT_BUDGET = 200_000_000
BRANCH_BUDGET = TOTAL_BUDGET - PILOT_BUDGET
SKILLS = ("restarts", "defense", "attack", "integrated", "teammates")
DEFAULT_PHASES = [
    dict(id="A", name="Fundamentos_y_saques", steps=800_000_000, exercises=.40,
         guide=.08, scripted=.50, selfplay=.20, pool=.30, teammates=.10,
         exercise_weights=dict(corner=.45, throw_in=.25, goal_kick=.30)),
    dict(id="B", name="Defensa_y_transiciones", steps=1_000_000_000, exercises=.30,
         guide=.06, scripted=.35, selfplay=.25, pool=.40, teammates=.15,
         exercise_weights=dict(defense=.55, defensive_transition=.30, corner=.15)),
    dict(id="C", name="Ataque_colectivo", steps=1_200_000_000, exercises=.30,
         guide=.05, scripted=.25, selfplay=.30, pool=.45, teammates=.20,
         exercise_weights=dict(build_up=.40, attack=.40, offensive_transition=.20)),
    dict(id="D", name="Juego_integrado", steps=1_200_000_000, exercises=.20,
         guide=.035, scripted=.15, selfplay=.30, pool=.55, teammates=.25,
         exercise_weights=dict(corner=.20, defense=.25, attack=.30, build_up=.25)),
    dict(id="E", name="Companeros_heterogeneos", steps=1_000_000_000, exercises=.20,
         guide=.02, scripted=.15, selfplay=.25, pool=.60, teammates=.35,
         exercise_weights=dict(build_up=.35, attack=.35, defense=.30)),
    dict(id="F", name="Consolidacion", steps=600_000_000, exercises=.05,
         guide=0., scripted=.10, selfplay=.25, pool=.65, teammates=.35,
         exercise_weights=dict(corner=.20, defense=.25, attack=.30, build_up=.25)),
]


def default_program(start_steps=0):
    return dict(version=VERSION, start_steps=start_steps,
                total_budget_steps=TOTAL_BUDGET, branch_budget_steps=BRANCH_BUDGET,
                pilot_steps=PILOT_BUDGET, evaluation_every_steps=100_000_000,
                snapshot_every_steps=50_000_000, recovery_steps=400_000_000,
                recovery_block_steps=100_000_000, final_unassisted_steps=200_000_000,
                transition_steps=10_000_000, phases=copy.deepcopy(DEFAULT_PHASES),
                lr=dict(initial=1e-4, minimum=5e-5, maximum=2e-4,
                        low_kl=.0008, high_kl=.006, epoch_stop_kl=.012,
                        every_iterations=25, increase=1.1, decrease=.8),
                entropy=dict(initial=.003, final=.001), bc=dict(initial=.005, final=.001))


def _integer(value, name, *, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} debe ser entero >= {minimum}")
    return value


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} debe ser finito")
    return float(value)


class ProgramState:
    def __init__(self, config, saved=None):
        self.config = copy.deepcopy(config)
        self._validate_config()
        self.phase_index = 0
        self.relative_steps = 0
        self.diagnostic_steps = 0
        self.hardware_diagnostic_steps = 0
        self.phase_start_steps = 0
        self.pass_streak = 0
        self.lr = float(self.config["lr"]["initial"])
        self.last_lr_iteration = -1
        self.last_endpoint_iteration = -1
        self.lr_kl_window = []
        self.lr_events = []
        self.last_evaluation_steps = 0
        self.last_snapshot_steps = 0
        self.skill_debts = []
        self.evaluations = []
        self.transitions = []
        self.recovery_consumed = 0
        self.recovery_block_skill = None
        self.recovery_block_start = None
        self.objective_signature = None
        self.objective_contract = None
        if saved is not None:
            self._restore(saved)

    @classmethod
    def from_config(cls, cfg, checkpoint_state=None):
        raw = cfg.get("rs4_program")
        if raw is None:
            raise ValueError("Falta rs4_program; usar tools.prepare_rs4_v3")
        return cls(raw, checkpoint_state)

    def _validate_config(self):
        cfg = self.config
        if cfg.get("version") != VERSION:
            raise ValueError("Este entrenador requiere rs4_program.version=3")
        for field in ("start_steps", "recovery_steps", "transition_steps"):
            _integer(cfg[field], field)
        for field in ("branch_budget_steps", "total_budget_steps", "pilot_steps",
                      "evaluation_every_steps", "snapshot_every_steps", "recovery_block_steps",
                      "final_unassisted_steps"):
            _integer(cfg[field], field, minimum=1)
        if cfg["branch_budget_steps"] + cfg["pilot_steps"] != cfg["total_budget_steps"]:
            raise ValueError("El presupuesto total incluye la rama elegida más el piloto descartado")
        phases = cfg["phases"]
        if len(phases) != 6 or [p["id"] for p in phases] != list("ABCDEF"):
            raise ValueError("El programa necesita las seis fases A–F en orden")
        if sum(_integer(p["steps"], "phase.steps", minimum=1) for p in phases) != cfg["branch_budget_steps"]:
            raise ValueError("Las seis fases deben sumar exactamente el presupuesto de la rama elegida")
        if cfg["pilot_steps"] > phases[0]["steps"]:
            raise ValueError("El piloto debe caber en la fase A")
        if cfg["recovery_steps"] + cfg["final_unassisted_steps"] > phases[-1]["steps"]:
            raise ValueError("Consolidación debe reservar recuperación y el tramo final sin guía")
        for phase in phases:
            for field in ("exercises", "teammates", "guide", "scripted", "selfplay", "pool"):
                if not 0 <= _finite(phase[field], field) <= 1:
                    raise ValueError(f"{field} fuera de rango")
            if phase["exercises"] > .4 or not math.isclose(sum(phase[k] for k in ("scripted", "selfplay", "pool")), 1):
                raise ValueError("Al menos 60% partidos; la mezcla de rivales debe sumar 1")
            weights = phase["exercise_weights"]
            if not weights or any(_finite(v, "exercise_weight") < 0 for v in weights.values()) or not math.isclose(sum(weights.values()), 1):
                raise ValueError("Los pesos de ejercicios deben ser no negativos y sumar 1")
        lr = cfg["lr"]
        for key in ("initial", "minimum", "maximum", "low_kl", "high_kl", "epoch_stop_kl", "increase", "decrease"):
            _finite(lr[key], f"lr.{key}")
        if not 0 < lr["minimum"] <= lr["initial"] <= lr["maximum"]:
            raise ValueError("Rango LR inválido")
        if not 0 <= lr["low_kl"] < lr["high_kl"] < lr["epoch_stop_kl"] or not lr["increase"] > 1 or not 0 < lr["decrease"] < 1:
            raise ValueError("Control KL inválido")
        _integer(lr["every_iterations"], "lr.every_iterations", minimum=1)
        for name in ("entropy", "bc"):
            if not 0 <= _finite(cfg[name]["final"], name) <= _finite(cfg[name]["initial"], name):
                raise ValueError(f"Calendario {name} inválido")

    @property
    def phase(self):
        return copy.deepcopy(self.config["phases"][self.phase_index])

    @property
    def complete(self):
        return self.relative_steps >= self.effective_branch_limit

    @property
    def effective_branch_limit(self):
        return self.config["branch_budget_steps"] - self.diagnostic_steps - self.hardware_diagnostic_steps

    def charge_hardware_diagnostics(self, n):
        """Monotone runtime benchmark cost, separate from immutable pilot cost."""
        _integer(n, "hardware_diagnostic_steps")
        if n == self.hardware_diagnostic_steps:
            return
        if n < self.hardware_diagnostic_steps:
            raise ValueError("El coste hardware no puede disminuir")
        total = self.diagnostic_steps + n
        if (total > self.config["phases"][-1]["steps"] - self.config["final_unassisted_steps"]
                or self.relative_steps + self.config["final_unassisted_steps"] > self.config["branch_budget_steps"] - total):
            raise ValueError("El diagnóstico hardware consume el tramo final reservado")
        self.hardware_diagnostic_steps = n

    def charge_diagnostics(self, n):
        """Cobrar benchmarks descartados una sola vez, sin mover el ancla inicial.

        Se acorta consolidación, preservando por lo menos 200M finales sin guía.
        La comparación de arquitectura también forma parte del presupuesto 6B.
        """
        _integer(n, "diagnostic_steps")
        if n == self.diagnostic_steps:
            return
        if self.diagnostic_steps:
            raise ValueError("El coste diagnóstico ya está fijado; no recalcularlo al reanudar")
        maximum = self.config["phases"][-1]["steps"] - self.config["final_unassisted_steps"]
        if n > maximum or self.relative_steps + self.config["final_unassisted_steps"] > self.config["branch_budget_steps"] - n:
            raise ValueError("El diagnóstico no puede consumir los 200M finales sin guía")
        self.diagnostic_steps = n

    @property
    def remaining_steps(self):
        return max(0, self.effective_branch_limit - self.relative_steps)

    @property
    def entropy_coef(self):
        return self._linear("entropy")

    @property
    def bc_coef(self):
        return self._linear("bc")

    def _linear(self, name):
        fraction = min(1., self.relative_steps / self.effective_branch_limit)
        schedule = self.config[name]
        return schedule["initial"] + fraction * (schedule["final"] - schedule["initial"])

    def _change_phase(self, reason):
        if self.phase_index >= 5:
            return False
        self.transitions.append(dict(from_phase=self.phase["id"], steps=self.relative_steps, reason=reason))
        self.phase_index += 1
        self.phase_start_steps = self.relative_steps
        self.pass_streak = 0
        return True

    def advance_steps(self, n):
        """Registrar TODAS las muestras útiles; nunca esconder un exceso de presupuesto."""
        _integer(n, "useful_steps")
        remaining = n
        while remaining:
            if self.phase_index == 5:
                self.relative_steps += remaining
                remaining = 0
            else:
                boundary = self.phase_start_steps + self.phase["steps"]
                count = min(remaining, max(0, boundary - self.relative_steps))
                self.relative_steps += count
                remaining -= count
                if self.relative_steps >= boundary:
                    skill = SKILLS[self.phase_index]
                    if self.pass_streak < 2 and skill not in self.skill_debts:
                        self.skill_debts.append(skill)
                    self._change_phase("nominal_budget")
        self._update_recovery()

    def _update_recovery(self):
        if self.phase_index != 5:
            return
        cutoff = self.effective_branch_limit - self.config["final_unassisted_steps"]
        if self.recovery_block_start is not None:
            elapsed = self.relative_steps - self.recovery_block_start
            block = self.config["recovery_block_steps"]
            if elapsed >= block or self.relative_steps >= cutoff or self.recovery_block_skill not in self.skill_debts:
                self.recovery_consumed += max(0, min(elapsed, block, cutoff - self.recovery_block_start))
                self.recovery_block_start = self.recovery_block_skill = None
        if (self.recovery_block_start is None and self.skill_debts
                and self.recovery_consumed + self.config["recovery_block_steps"] <= self.config["recovery_steps"]
                and self.relative_steps + self.config["recovery_block_steps"] <= cutoff):
            # Rotación determinista; la evaluación retira habilidades que aprueban.
            index = self.recovery_consumed // self.config["recovery_block_steps"] % len(self.skill_debts)
            self.recovery_block_skill = self.skill_debts[index]
            self.recovery_block_start = self.relative_steps

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
        kl = _finite(endpoint_kl, "endpoint_kl")
        if kl < 0:
            raise ValueError("La KL de final de update no puede ser negativa")
        cfg = self.config["lr"]
        if iteration > self.last_endpoint_iteration:
            self.last_endpoint_iteration = iteration
            self.lr_kl_window.append(kl)
            self.lr_kl_window = self.lr_kl_window[-cfg["every_iterations"]:]
        if iteration == self.last_lr_iteration or iteration % cfg["every_iterations"]:
            return self.lr
        self.last_lr_iteration = iteration
        previous = self.lr
        decision = "hold"
        if max(self.lr_kl_window) < cfg["low_kl"]:
            self.lr = min(cfg["maximum"], self.lr * cfg["increase"])
            decision = "increase"
        elif sum(self.lr_kl_window) / len(self.lr_kl_window) > cfg["high_kl"]:
            self.lr = max(cfg["minimum"], self.lr * cfg["decrease"])
            decision = "decrease"
        self.lr_events.append(dict(iteration=iteration, steps=self.relative_steps, endpoint_kl=kl,
                                   window_mean=sum(self.lr_kl_window) / len(self.lr_kl_window),
                                   window_max=max(self.lr_kl_window), previous_lr=previous,
                                   lr=self.lr, decision=decision))
        self.lr_events = self.lr_events[-64:]
        return self.lr

    def settings(self):
        phase = self.phase
        # La mezcla de ejercicios sólo cambia al reiniciar el episodio.
        guide = phase["guide"]
        if self.phase_index > 0:
            elapsed = self.relative_steps - self.phase_start_steps
            width = max(1, self.config["transition_steps"])
            prior = self.config["phases"][self.phase_index - 1]["guide"]
            guide = prior + min(1., elapsed / width) * (guide - prior)
        weights = copy.deepcopy(phase["exercise_weights"])
        if self.skill_debts:
            debt_scenarios = dict(restarts="corner", defense="defense", attack="attack",
                                  integrated="build_up", teammates="build_up")
            for skill in self.skill_debts:
                key = debt_scenarios[skill]
                weights[key] = weights.get(key, 0) + .5 / len(self.skill_debts)
            total = sum(weights.values())
            weights = {key: value / total for key, value in weights.items()}
        aliases = dict(throw_in="lateral", build_up="exit")
        canonical = {}
        for key, value in weights.items():
            key = aliases.get(key, key)
            canonical[key] = canonical.get(key, 0.) + value
        weights = canonical
        recovery = self.phase_index == 5 and self.recovery_block_skill is not None
        if recovery:
            guide = .02
            phase["exercises"] = .4
            focused = {"restarts": dict(corner=.5, lateral=.25, goal_kick=.25),
                       "defense": dict(defense=.7, defensive_transition=.3),
                       "attack": dict(attack=.6, exit=.4),
                       "integrated": dict(exit=.4, attack=.3, defense=.3),
                       "teammates": dict(exit=.5, attack=.5)}[self.recovery_block_skill]
            keys = set(weights) | set(focused)
            weights = {key: .25 * weights.get(key, 0.) + .75 * focused.get(key, 0.) for key in keys}
        if self.remaining_steps <= self.config["final_unassisted_steps"]:
            guide = 0.
            recovery = False
            phase["exercises"] = .05
        bonus = .02
        if self.phase_index == 5:
            # El tramo final elimina también premios de práctica, no los goles.
            bonus *= min(1., max(0., (self.remaining_steps - self.config["final_unassisted_steps"])
                                 / max(1, self.effective_branch_limit - self.phase_start_steps - self.config["final_unassisted_steps"])))
        from eval.rs4_objective import CONTRACT
        corrected = self.objective_contract == CONTRACT
        teammates = phase["teammates"]
        if corrected and self.phase_index >= 2:
            teammates = max(teammates, (.35, .45, .50, .50)[self.phase_index - 2])
        return dict(phase_id=phase["id"], phase_name=phase["name"], guide_coef=guide,
                    exercise_fraction=phase["exercises"], exercise_weights=weights,
                    opponent_mix={name: phase[name] for name in ("scripted", "selfplay", "pool")},
                    frozen_teammates_fraction=teammates,
                    teammate_learner_weights=(.5, .3, .2) if corrected else (1/3, 1/3, 1/3),
                    scenario_difficulty=min(1., .2 * self.phase_index) if corrected else 0.,
                    restart_execute_bonus=bonus,
                    restart_bonus=bonus,
                    restart_potential_coef=.025 * min(1., guide / .08),
                    restart_stall_penalty=.25, recovery=recovery,
                    skill_debts=list(self.skill_debts), entropy_coef=self.entropy_coef,
                    bc_coef=self.bc_coef, lr=self.lr, freeze_normalizers=True)

    @staticmethod
    def evaluation_gates(report):
        """Contrato normalizado: skills/baseline/matches con métricas escalares.

        Ausencia de evidencia NO equivale a aprobar. El evaluador mantiene sus
        conteos y semillas originales junto a este resumen.
        """
        from eval.rs4_objective import CONTRACT, gates
        if report.get("evaluation_contract") == CONTRACT:
            return gates(report)
        skills, baseline = report.get("skills", {}), report.get("baseline", {})
        matches = report.get("matches", {})
        def number(mapping, key):
            value = mapping.get(key)
            return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None
        restart = [number(skills, key) for key in ("restart_success", "corner_success_red", "corner_success_blue")]
        conceded, base_conceded = number(skills, "defense_conceded"), number(baseline, "defense_conceded")
        attack, base_attack = number(skills, "attack_success"), number(baseline, "attack_success")
        points, base_points = number(matches, "points"), number(matches, "baseline_points")
        safe = points is not None and base_points is not None and points >= base_points - .05
        integrated, base_integrated = number(skills, "integrated_success"), number(baseline, "integrated_success")
        teammates, base_teammates = number(skills, "teammate_success"), number(baseline, "teammate_success")
        return dict(restarts=all(x is not None for x in restart) and restart[0] >= .8 and min(restart[1:]) >= .7,
                    defense=conceded is not None and base_conceded is not None and conceded <= .8 * base_conceded,
                    attack=attack is not None and base_attack is not None and attack >= base_attack + .1,
                    integrated=safe and integrated is not None and base_integrated is not None and integrated > base_integrated,
                    teammates=safe and teammates is not None and base_teammates is not None and teammates > base_teammates)

    def reconcile_objective(self, report):
        """Reassess debts at the same weights/steps; preserve original evidence.

        This is never a second approval and cannot advance a phase. Settings
        adaptations are enabled only after evaluating the new objective.
        """
        from eval.rs4_objective import CONTRACT, signature, curriculum_signature
        if report.get("suite", {}).get("holdout"):
            raise ValueError("La suite reservada no modifica el currículo")
        if report.get("evaluation_contract") != CONTRACT or not report.get("reference"):
            raise ValueError("La reconciliación requiere referencia evaluada con el contrato actual")
        if signature(report) != signature(report["reference"]):
            raise ValueError("Referencia y candidato deben usar la misma suite")
        current = curriculum_signature(report)
        if self.objective_signature == current:
            return
        self.objective_signature, self.objective_contract = current, CONTRACT
        gates = self.evaluation_gates(report)
        taught = SKILLS[:min(5, self.phase_index + 1)]
        self.skill_debts = [skill for skill in taught if not gates[skill]]
        self.pass_streak = 0
        self.last_evaluation_steps = self.relative_steps
        self.evaluations.append(dict(steps=self.relative_steps, phase=self.phase["id"],
                                     kind="objective_reconciliation", gates=gates, report=copy.deepcopy(report)))
        self._update_recovery()

    def record_evaluation(self, report):
        if report.get("suite", {}).get("holdout"):
            raise ValueError("La suite reservada no modifica el currículo")
        if self.evaluations and self.relative_steps <= self.last_evaluation_steps:
            raise ValueError("Una evaluación repetida del mismo punto no cuenta como segunda aprobación")
        gates = self.evaluation_gates(report)
        taught = SKILLS[:min(5, self.phase_index + 1)]
        for skill in taught:
            if gates[skill] and skill in self.skill_debts:
                self.skill_debts.remove(skill)
            elif not gates[skill] and skill not in self.skill_debts:
                self.skill_debts.append(skill)
        self.last_evaluation_steps = self.relative_steps
        self.evaluations.append(dict(steps=self.relative_steps, phase=self.phase["id"],
                                     gates=gates, report=copy.deepcopy(report)))
        self.evaluations = self.evaluations[-64:]  # informes completos viven fuera del checkpoint
        passed = gates[SKILLS[self.phase_index]] if self.phase_index < 5 else all(gates.values())
        self.pass_streak = self.pass_streak + 1 if passed else 0
        changed = False
        if (self.phase_index < 5 and self.pass_streak >= 2
                and self.relative_steps - self.phase_start_steps >= self.phase["steps"] / 2):
            changed = self._change_phase("two_evaluations_passed")
        self._update_recovery()
        return dict(gates=gates, advanced=changed, skill_debts=list(self.skill_debts))

    def state_dict(self):
        fields = ("phase_index", "relative_steps", "diagnostic_steps", "hardware_diagnostic_steps", "phase_start_steps", "pass_streak", "lr",
                  "last_lr_iteration", "last_endpoint_iteration", "lr_kl_window", "lr_events", "last_evaluation_steps", "last_snapshot_steps", "skill_debts",
                  "evaluations", "transitions", "recovery_consumed", "recovery_block_skill", "recovery_block_start",
                  "objective_signature", "objective_contract")
        return dict(version=VERSION, config=copy.deepcopy(self.config),
                    **{name: copy.deepcopy(getattr(self, name)) for name in fields})

    def _restore(self, saved):
        if saved.get("version") != VERSION or saved.get("config") != self.config:
            raise ValueError("El programa guardado no coincide: no cambiar anclas/calendarios al reanudar")
        fresh = self.state_dict()
        for name in fresh:
            if name not in ("version", "config"):
                if name not in saved:
                    if name in ("objective_signature", "objective_contract", "hardware_diagnostic_steps"):
                        continue  # old v3 checkpoints acquire this only through reevaluation
                    raise ValueError(f"Estado RS4 incompleto: {name}")
                setattr(self, name, copy.deepcopy(saved[name]))
        for key in ("phase_index", "relative_steps", "diagnostic_steps", "phase_start_steps", "pass_streak", "last_evaluation_steps",
                    "last_snapshot_steps", "recovery_consumed"):
            _integer(getattr(self, key), key)
        if not 0 <= self.phase_index < 6 or self.phase_start_steps > self.relative_steps:
            raise ValueError("Fase guardada inválida")
        if not self.config["lr"]["minimum"] <= _finite(self.lr, "lr") <= self.config["lr"]["maximum"]:
            raise ValueError("LR guardada fuera de rango")
        if any(skill not in SKILLS for skill in self.skill_debts):
            raise ValueError("Deuda de habilidad desconocida")
        if self.diagnostic_steps > self.config["phases"][-1]["steps"] - self.config["final_unassisted_steps"]:
            raise ValueError("Diagnóstico guardado consume el tramo final reservado")
        _integer(self.hardware_diagnostic_steps, "hardware_diagnostic_steps")
        if self.diagnostic_steps + self.hardware_diagnostic_steps > self.config["phases"][-1]["steps"] - self.config["final_unassisted_steps"]:
            raise ValueError("Diagnóstico guardado consume el tramo final reservado")
