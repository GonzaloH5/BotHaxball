"""Curriculum RS4-Z: etapas, mezcla de tareas, rewards, latencia, rivales, presupuesto y compuertas.

Una etapa avanza SÓLO si pasa sus compuertas (baterías y partidos de evaluación, `eval/rs4z/gates.py`),
nunca por la recompensa media. Si agota su presupuesto sin pasar, el entrenamiento se detiene para
diagnóstico (no se avanza "a la fuerza"). Las etapas anteriores quedan como repaso (≥ `retention`).

Presupuestos en muestras de aprendizaje (filas que entran a PPO). Los umbrales de las compuertas se
expresan relativos a RS-Pro L5 en la misma batería y se congelan en `reports/rs4z/gates.yaml` antes
de entrenar (`tools/rs4z_calibrate_gates.py`).
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Latencia (ticks) por etapa: (probabilidad host d=0, rango cliente). El scripted también la sufre.
LATENCY_NONE = dict(host=1.0, client=(0, 0))
LATENCY_LIGHT = dict(host=0.5, client=(1, 3))
LATENCY_MIXED = dict(host=0.4, client=(6, 11))
LATENCY_CLIENT = dict(host=0.3, client=(6, 11))


@dataclass
class Stage:
    name: str
    budget: float                      # muestras de aprendizaje (techo)
    tasks: dict                        # tarea → peso
    coefs: dict                        # coeficientes de rewards al inicio de la etapa
    coefs_end: dict = field(default_factory=dict)  # objetivo al final (retiro lineal si se indica)
    latency: dict = field(default_factory=lambda: LATENCY_NONE)
    variants: float = 0.0              # probabilidad de variante de mapa (kickStrength 5,75 / radio 8)
    gamma: float = 0.997
    retention: float = 0.10            # fracción mínima de tareas de etapas anteriores
    opponents: dict = field(default_factory=lambda: dict(rspro=1.0))
    frozen_mates: float = 0.0          # fracción de partidos con compañeros congelados (ad-hoc)
    gates: tuple = ()
    notes: str = ""


STAGES = [
    Stage("S1", 0.3e9,
          tasks=dict(touch=0.25, empty_goal=0.35, carry_goal=0.2, receive_shot=0.2),
          coefs=dict(goal=1.0, drill=1.0, ball=0.5, threat=0.25),
          coefs_end=dict(ball=0.0),
          gates=("control_battery",),
          notes="1v0: pelota, conducción y definición a arco vacío"),
    Stage("S2", 0.5e9,
          tasks=dict(shot_vs_last=0.25, attack_1v1=0.2, defend_1v1=0.2, loose_ball=0.15, match_1v1=0.2),
          coefs=dict(goal=1.0, drill=1.0, threat=0.25, access=0.05),
          latency=LATENCY_LIGHT,
          gates=("duel_battery", "match_1v1_vs_l5"),
          notes="duelos, tiro contra el último hombre, pelotas divididas, 1v1 a cancha completa"),
    Stage("S3", 0.7e9,
          tasks=dict(attack_2v1=0.25, one_two=0.2, through_ball=0.2, redirect=0.15, attack_2v2=0.2),
          coefs=dict(goal=1.0, drill=1.0, threat=0.25, access=0.05),
          latency=LATENCY_LIGHT,
          gates=("passing_battery",),
          notes="pase, pared, profundidad y desvíos con 2 aprendices de parámetros compartidos"),
    Stage("S4", 0.8e9,
          tasks=dict(defend_2v2=0.2, defend_4v4=0.2, defend_restart=0.2, restart_take=0.15, match_2v2=0.25),
          coefs=dict(goal=1.0, drill=1.0, threat=0.2, access=0.05),
          coefs_end=dict(access=0.0),
          latency=LATENCY_LIGHT,
          gates=("defense_battery", "detectors_small"),
          notes="defensa y posicionamiento, saques propios; Φ_access se retira al terminar"),
    Stage("S5", 2.5e9,
          tasks=dict(match_4v4=0.45, match_4v3=0.1, match_3v4=0.1, match_4v2=0.05, situation_open=0.1,
                     situation_attack=0.08, restart_attack=0.06, transition=0.06),
          coefs=dict(goal=1.0, drill=1.0, result=0.3, threat=0.2),
          coefs_end=dict(threat=0.0),
          latency=LATENCY_MIXED, variants=0.2,
          opponents=dict(rspro=0.7, mirror=0.3),
          gates=("match_4v4_vs_l5", "reserved_style", "restart_battery", "detectors_4v4"),
          notes="4v4 contra la escalera RS-Pro, superioridades, transiciones y estados humanos; Φ_threat se retira"),
    Stage("S6", 7.5e9,
          tasks=dict(league_4v4=0.9, match_4v3=0.03, match_3v4=0.03, situation_open=0.02, restart_attack=0.02),
          coefs=dict(goal=1.0, drill=1.0, result=0.3),
          latency=LATENCY_MIXED, variants=0.2, retention=0.05,
          opponents=dict(mirror=0.45, pfsp=0.25, rspro=0.2, exploiter=0.1),
          frozen_mates=0.15,
          gates=("league_milestone",),
          notes="liga: espejo, PFSP sobre históricos, RS-Pro L3–L5, exploiters; compañeros ad-hoc"),
    Stage("S7", 1.0e9,
          tasks=dict(league_4v4=0.95, situation_open=0.05),
          coefs=dict(goal=1.0, drill=1.0, result=0.3),
          latency=LATENCY_CLIENT, variants=0.25, retention=0.05,
          opponents=dict(mirror=0.4, pfsp=0.3, rspro=0.2, exploiter=0.1),
          frozen_mates=0.2,
          gates=("robustness",),
          notes="robustez: latencia de cliente, compañeros ad-hoc, sondas con estilos reservados"),
]

STAGE_INDEX = {s.name: i for i, s in enumerate(STAGES)}


def total_budget():
    return sum(s.budget for s in STAGES)
