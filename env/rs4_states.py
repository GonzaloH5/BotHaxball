"""Cargar estados reales (tools/rs4_states.py) en un entorno RS ONE 4v4 con referee=rs_one_v1.

Un estado fija pelota, jugadores, teclas de patada mantenidas y último toque; si es una
colocación de saque, el árbitro ejecuta las mismas acciones de inicio que el script.
`mirror=True` intercambia colores reflejando x: la política compartida ve la situación
desde el otro equipo (aumenta los datos sin inventar estados).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

KIND_OPEN, KIND_LATERAL, KIND_CORNER, KIND_GOAL_KICK, KIND_KICKOFF = 0, 1, 2, 3, 4
POOLS = ("attack", "open", "restart")
DEFAULT_MIX = {"attack": 0.4, "open": 0.3, "restart": 0.3}
GOAL_X = 1150.0
# Ataque = pelota a <= 500 px del centro del arco que ataca el último toque. Medido con A (v3 2568M)
# contra sí misma: con el último 35% del campo convertía 5,5 cada 100 (umbral del plan: 10) y sólo
# definía como un humano cerca del arco; con 500 px convierte 15 cada 100 y quedan 2610 estados de
# entrenamiento (reports/rs4_b1/situations_v3_2568M.json). El resto del juego abierto queda en "open".
ATTACK_RADIUS = 500.0


class StateBank:
    def __init__(self, path):
        data = np.load(Path(path), allow_pickle=False)
        self.arrays = {key: data[key] for key in data.files}
        self.size = len(self.arrays["kind"])
        self._pools = None

    def __len__(self):
        return self.size

    def select(self, kinds=None):
        mask = np.ones(self.size, dtype=bool) if kinds is None else np.isin(self.arrays["kind"], list(kinds))
        return np.flatnonzero(mask)

    def attack_mask(self):
        """Juego abierto con la pelota a <= ATTACK_RADIUS del arco que ataca el equipo del último toque."""
        a = self.arrays
        sign = np.where(a["last_touch"] == 0, 1.0, -1.0)
        distance = np.hypot(GOAL_X - sign * a["ball_pos"][:, 0], a["ball_pos"][:, 1])
        return (a["kind"] == KIND_OPEN) & (a["last_touch"] >= 0) & (distance <= ATTACK_RADIUS)

    def pools(self):
        """Índices por tipo de situación (POOLS): ataque, otro juego abierto y saques."""
        if self._pools is None:
            attack = np.flatnonzero(self.attack_mask())
            self._pools = {"attack": attack,
                           "open": np.setdiff1d(self.select([KIND_OPEN]), attack),
                           "restart": self.select([KIND_LATERAL, KIND_CORNER, KIND_GOAL_KICK])}
        return self._pools


def apply_states(env, rows, bank, indices, mirror=None):
    """Colocar estados `indices` del banco en las filas `rows` del entorno."""
    rows = np.asarray(rows, dtype=np.int64)
    indices = np.asarray(indices, dtype=np.int64)
    mirror = np.zeros(len(rows), dtype=bool) if mirror is None else np.asarray(mirror, dtype=bool)
    if getattr(env, "_rs1", None) is None:
        raise ValueError("Los estados reales requieren referee=rs_one_v1")
    sim, a = env.sim, bank.arrays
    fp = sim.first_player
    kinds = a["kind"][indices]
    kickoff_rows = rows[kinds == KIND_KICKOFF]
    if len(kickoff_rows):
        takers = a["taker"][indices[kinds == KIND_KICKOFF]]
        flips = mirror[kinds == KIND_KICKOFF]
        env._reset_envs(kickoff_rows, kickoff_team=np.where(flips, 1 - takers, takers))
    other = kinds != KIND_KICKOFF
    if other.any():
        env._reset_envs(rows[other])
    for row, i, flip in zip(rows, indices, mirror):
        if a["kind"][i] == KIND_KICKOFF:
            continue
        bp, bv = a["ball_pos"][i].astype(np.float64), a["ball_vel"][i].astype(np.float64)
        pp, pv = a["player_pos"][i].astype(np.float64), a["player_vel"][i].astype(np.float64)
        held = a["kick_held"][i].astype(bool)
        last, taker, spot = int(a["last_touch"][i]), int(a["taker"][i]), a["spot"][i].astype(np.float64)
        if flip:
            reflect = np.array([-1.0, 1.0])
            bp, bv, spot = bp * reflect, bv * reflect, spot * reflect
            pp = np.concatenate([pp[4:], pp[:4]]) * reflect
            pv = np.concatenate([pv[4:], pv[:4]]) * reflect
            held = np.concatenate([held[4:], held[:4]])
            last = 1 - last if last >= 0 else -1
            taker = 1 - taker if taker >= 0 else -1
        sim.kickoff[row] = False
        sim.mask[row] = sim.base_mask
        sim.pos[row, 0], sim.vel[row, 0] = bp, bv
        sim.pos[row, fp:], sim.vel[row, fp:] = pp, pv
        sim.kick_cancel[row] = held
        env.last_touch[row] = last
        env.setpiece_team[row] = -1
        env.setpiece_kind[row] = 0
        env._rs1.reset_rows([row])
        if a["kind"][i] in (KIND_LATERAL, KIND_CORNER, KIND_GOAL_KICK):
            env._rs1.start(row, int(a["kind"][i]), taker, (float(spot[0]), float(spot[1])))
    env._rs1.protect()
    env._phi = env._potentials()
    if getattr(env, "_public_signals", None) is not None:
        env._public_signals.reset(rows)


def mix_weights(bank, mix):
    """Probabilidad de cada tipo de POOLS; un tipo sin estados en el banco queda en 0."""
    unknown = set(mix) - set(POOLS)
    if unknown:
        raise ValueError(f"Tipos de situación desconocidos: {sorted(unknown)}")
    members = bank.pools()
    weights = np.array([float(mix.get(name, 0.0)) if len(members[name]) else 0.0 for name in POOLS])
    if (weights < 0).any() or not np.isfinite(weights).all() or weights.sum() <= 0:
        raise ValueError("La mezcla de situaciones no tiene estados disponibles")
    return weights / weights.sum()


def place_recorded(scenario_env, rows):
    """Situación técnica: un estado real de la partición de entrenamiento (PLAN_RS4 3).

    Mezcla por defecto: 40% ataques (pelota en el último 35% del equipo con el último toque),
    30% otro juego abierto y 30% saques. Espejo al azar. Termina en gol, salida nueva o
    `ticks` (12 s por defecto; ver RS4ScenarioEnv.step).
    """
    bank, config, rng = scenario_env._bank, scenario_env.recorded, scenario_env.rng
    rows = np.asarray(rows, dtype=np.int64)
    members = bank.pools()
    choice = rng.choice(len(POOLS), size=len(rows), p=mix_weights(bank, dict(config.get("mix", DEFAULT_MIX))))
    ids = np.array([rng.choice(members[POOLS[c]]) for c in choice], dtype=np.int64)
    mirror = rng.random(len(rows)) < 0.5
    env = scenario_env.base
    apply_states(env, rows, bank, ids, mirror)
    scenario_env.drill_limit[rows] = int(config.get("ticks", 720))
    scenario_env.recorded_from_restart[rows] = env.setpiece_team[rows] >= 0
    scenario_env.recorded_released[rows] = False
    scenario_env.recorded_pool[rows] = choice
    scenario_env.recorded_state[rows] = ids
    # Atacante: quien saca en un saque; si no, el último toque (ya espejado).
    scenario_env.recorded_attacker[rows] = np.where(env.setpiece_team[rows] >= 0, env.setpiece_team[rows],
                                                    env.last_touch[rows])
