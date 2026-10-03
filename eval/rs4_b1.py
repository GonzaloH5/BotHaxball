"""Protocolo fijo de evaluación RS4-b1 (PLAN_RS4.md sección 4).

* Juego real: referee=rs_one_v1 (contrato reports/rs4_b1/contract.json), partidos de 3 minutos.
* Comparación principal: UN candidato con tres compañeros externos reservados contra rivales
  congelados, ambos colores y el candidato rotando por los cuatro puestos.
* Mismos estados iniciales (estados reales de la partición de desarrollo, perturbados) y mismas
  semillas para candidato y referencia: comparación pareada por estado.
* Greedy (como deploy/bot.js --temp 0); el muestreo se informa aparte si se pide.
* Por celda (compañero, rival, color, puesto): puntos, diferencia de gol, remates, pases/pérdidas,
  detectores de atajos y unicidad de trayectorias.
* Batería técnica: saques reales (laterales, córners, saques de arco, saques iniciales) donde saca
  el equipo del candidato (controla a los cuatro), para la condición de saques.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from bots.scripted import scripted_actions
from env.haxball_env import HaxballEnv
from env.rewards import RewardConfig
from env.rs4_states import StateBank, apply_states, KIND_CORNER, KIND_GOAL_KICK, KIND_KICKOFF, KIND_LATERAL, KIND_OPEN

ROOT = Path(__file__).resolve().parent.parent
PROTOCOL = dict(version="RS4-b1-eval-1", minutes=3.0, frame_skip=3, referee="rs_one_v1", greedy=True,
                perturb_px=8.0, restart_success_ticks=600, kickoff_success_ticks=300)
PUBLIC_SIGNALS = dict(version=1, barrier_discs=[], barrier_segments=[], auto_joints=True,
                      joint_profile="rs4_paired_lateral_v1",
                      red_colors=[16711680, 15035990, 15496280, 16727860], blue_colors=[255, 5671397, 4767481, 1031417])


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def environment(n, seed, minutes=PROTOCOL["minutes"]):
    reward = RewardConfig(shaping_coef=0, team_spread_floor=0, kickoff_approach=0, w_defense_support=0,
                          rs4_tactical_coef=0, corner_execute=0, out_penalty=0)
    env = HaxballEnv(n, 4, "rs_one", frame_skip=PROTOCOL["frame_skip"], max_ticks=int(minutes * 3600),
                     random_reset_prob=0, seed=seed, obs_layout="universal", max_entities=7, out_of_bounds=True,
                     corner_reset_prob=0, kickoff_timeout=180, reward=reward, referee=PROTOCOL["referee"])
    env.configure_public_signals(PUBLIC_SIGNALS)
    return env


class Policy:
    """Controlador por fila y jugador: checkpoint (feedforward o recurrente) o scripted:r3[:estilo]."""

    def __init__(self, spec, device="cpu"):
        self.spec, self.device = spec, torch.device(device)
        if spec.startswith("scripted"):
            parts = spec.split(":")
            self.kind, self.model = "scripted", None
            self.policy = parts[1] if len(parts) > 1 else "r3"
            self.style = int(parts[2]) if len(parts) > 2 else -1
            self.identity = spec
        else:
            from train.model import load_model
            self.kind = "model"
            self.model = load_model(spec).to(self.device).eval()
            self.identity = sha256(spec)
        self.recurrent = bool(getattr(self.model, "is_recurrent", False))
        self.public = bool(getattr(self.model, "public_signals_version", 0))
        self.memory = self.previous = None

    def reset(self, n, p, rows=None):
        if not self.recurrent:
            return
        if self.memory is None or rows is None:
            self.memory = self.model.initial_state(n * p).reshape(n, p, -1)
            self.previous = torch.full((n, p), self.model.n_actions, dtype=torch.long, device=self.device)
        else:
            self.memory[rows] = 0
            self.previous[rows] = self.model.n_actions

    @torch.no_grad()
    def act(self, env, obs, mask, rng, greedy=True):
        out = np.zeros(mask.shape, dtype=np.int64)
        if not mask.any():
            return out
        if self.kind == "scripted":
            actions = scripted_actions(env, np.arange(env.P), 0.0, rng, policy=self.policy, style=self.style)
            out[mask] = actions[mask]
            return out
        rows, players = np.nonzero(mask)
        selected = obs[rows, players].copy()
        if not self.public:
            selected[:, 56:71] = 0  # referencias sin señales públicas: mismo contrato que en entrenamiento
        x = torch.as_tensor(selected, dtype=torch.float32, device=self.device)
        if self.recurrent:
            memory = self.memory[rows, players]
            previous = self.previous[rows, players]
            logits, _, updated = self.model.step(x, memory, previous)
            self.memory[rows, players] = updated
        else:
            logits = self.model.logits(x)
        if greedy:
            chosen = logits.argmax(-1)
        else:
            chosen = torch.distributions.Categorical(logits=logits).sample()
        if self.recurrent:
            self.previous[rows, players] = chosen
        out[rows, players] = chosen.cpu().numpy()
        return out


@dataclass
class CellSpec:
    teammate: str
    rival: str


def _layout(games):
    """Filas: juego g, color c (0 rojo, 1 azul), puesto s del candidato (0-3 dentro de su equipo)."""
    rows = np.arange(games * 8)
    return rows // 8, rows % 2, (rows // 2) % 4


def _controllers(n, colors, slots):
    """Matrices (n, 8): candidato, compañeros y rivales."""
    candidate = np.zeros((n, 8), dtype=bool)
    teammate = np.zeros((n, 8), dtype=bool)
    rival = np.zeros((n, 8), dtype=bool)
    for row in range(n):
        own = np.arange(4) + 4 * colors[row]
        candidate[row, own[slots[row]]] = True
        teammate[row, own] = True
        teammate[row, own[slots[row]]] = False
        rival[row, np.arange(4) + 4 * (1 - colors[row])] = True
    return candidate, teammate, rival


def play_cell(candidate, teammate, rival, bank, state_ids, seed, greedy=True, minutes=PROTOCOL["minutes"]):
    """Juega len(state_ids) juegos × 2 colores × 4 puestos; devuelve métricas por fila."""
    games = len(state_ids)
    n = games * 8
    game, colors, slots = _layout(games)
    env = environment(n, seed, minutes)
    env.reset()
    rng = np.random.default_rng(seed)
    apply_states(env, np.arange(n), bank, np.asarray(state_ids)[game], mirror=colors == 1)
    # perturbación pequeña y determinista por juego: diversidad sin cambiar la situación
    jitter = np.random.default_rng(seed + 7).uniform(-PROTOCOL["perturb_px"], PROTOCOL["perturb_px"], (games, 8, 2))
    sim = env.sim
    for row in range(n):
        delta = jitter[game[row]].copy()
        if colors[row] == 1:
            delta = np.concatenate([delta[4:], delta[:4]]) * [-1, 1]
        sim.player_pos[row] += delta
    env._rs1.protect()
    env.match_ticks[:] = 0
    env.match_score[:] = 0
    env.score[:] = 0
    obs = env.observe()
    is_candidate, is_teammate, is_rival = _controllers(n, colors, slots)
    policies = [(candidate, is_candidate), (teammate, is_teammate), (rival, is_rival)]
    for policy, _ in policies:
        policy.reset(n, 8)
    decisions = int(minutes * 3600) // PROTOCOL["frame_skip"]
    team = colors
    stats = {k: np.zeros(n) for k in ("passes", "turnovers", "shots", "own_goals", "parked", "idle", "keepers",
                                       "kickoff_stalls", "candidate_touches")}
    trace = [[] for _ in range(n)]
    finished = np.zeros(n, dtype=bool)
    final = np.zeros((n, 2), dtype=np.int64)
    candidate_index = np.argmax(is_candidate, axis=1)
    goal_x = env.goal_x
    for step in range(decisions + 2):
        actions = np.zeros((n, 8), dtype=np.int64)
        for policy, mask in policies:
            chosen = policy.act(env, obs, mask, rng, greedy)
            actions[mask] = chosen[mask]
        last_before = env.last_touch.copy()
        obs, _, done, info = env.step(actions)
        live = ~finished
        ev = info["events"]
        rows = np.arange(n)
        stats["passes"] += ev["passes"][rows, team] * live
        stats["turnovers"] += ev["turnovers"][rows, team] * live
        # remate: patada del equipo del candidato con la pelota yendo hacia el arco rival
        kicked_team = (info["kicked"] & (sim.player_team[None, :] == team[:, None])).any(axis=1)
        sign = np.where(team == 0, 1.0, -1.0)
        bv = sim.ball_vel
        towards = sign * bv[:, 0] > 2.0
        if towards.any():
            t = (sign * goal_x - sim.ball_pos[:, 0]) / np.where(np.abs(bv[:, 0]) > 1e-6, bv[:, 0], 1e-6)
            y_at = sim.ball_pos[:, 1] + bv[:, 1] * np.maximum(t, 0)
            on_target = towards & (np.abs(y_at) < sim.st.goal_half_height + 30)
        else:
            on_target = np.zeros(n, dtype=bool)
        stats["shots"] += (kicked_team & on_target) * live
        goal = info["goal"]
        conceded = np.where(team == 0, goal == -1, goal == 1)
        stats["own_goals"] += (conceded & (last_before == team)) * live
        stall = info["stall"] & (env.sim.kickoff_team == team)
        stats["kickoff_stalls"] += stall * live
        cpos = sim.player_pos[rows, candidate_index]
        dist = np.hypot(*(cpos - sim.ball_pos).T)
        stats["parked"] += (dist > 800) * live
        stats["idle"] += (np.hypot(*sim.player_vel[rows, candidate_index].T) < 0.1) * live
        own = sim.player_team[None, :] == team[:, None]
        near_goal = np.hypot(sim.player_pos[..., 0] + sign[:, None] * goal_x, sim.player_pos[..., 1]) < 250
        stats["keepers"] += ((near_goal & own).sum(axis=1) >= 2) * live
        stats["candidate_touches"] += env._rs1.contact[rows, candidate_index] * live
        if step % 30 == 0:
            for row in np.flatnonzero(live):
                trace[row].append((round(float(sim.ball_pos[row, 0]) / 5), round(float(sim.ball_pos[row, 1]) / 5)))
        match_end = info["match_done"] & live
        final[match_end] = info["final_score"][match_end]
        finished |= match_end
        if done.any():
            for policy, _ in policies:
                policy.reset(n, 8, np.flatnonzero(done))
        if finished.all():
            break
    ours = final[np.arange(n), team]
    theirs = final[np.arange(n), 1 - team]
    digest = [hashlib.sha256(json.dumps(t).encode()).hexdigest()[:16] for t in trace]
    return dict(game=game, color=colors, slot=slots, goals_for=ours, goals_against=theirs,
                points=(ours > theirs) + 0.5 * (ours == theirs), trace=digest,
                decisions=float(step + 1), **stats)


def restart_battery(candidate, rival, bank, seed, limit_per_kind=None):
    """Saques reales con el equipo del candidato como ejecutor (controla a los cuatro)."""
    kinds = {KIND_LATERAL: "lateral", KIND_CORNER: "corner", KIND_GOAL_KICK: "goal_kick", KIND_KICKOFF: "kickoff"}
    ids = []
    rng = np.random.default_rng(seed)
    for code in kinds:
        chosen = bank.select([code])
        if limit_per_kind and len(chosen) > limit_per_kind:
            chosen = np.sort(rng.choice(chosen, limit_per_kind, replace=False))
        ids += list(chosen)
    ids = np.asarray(ids, dtype=np.int64)
    n = len(ids)
    env = environment(n, seed, minutes=10)
    env.reset()
    taker = bank.arrays["taker"][ids]
    # el candidato juega de rojo: espejar los saques del azul
    apply_states(env, np.arange(n), bank, ids, mirror=taker == 1)
    obs = env.observe()
    own = np.zeros((n, 8), dtype=bool)
    own[:, :4] = True
    candidate.reset(n, 8)
    rival.reset(n, 8)
    kind = bank.arrays["kind"][ids]
    limit = np.where(kind == KIND_KICKOFF, PROTOCOL["kickoff_success_ticks"], PROTOCOL["restart_success_ticks"])
    success = np.zeros(n, dtype=bool)
    resolved = np.zeros(n, dtype=bool)
    ticks = np.zeros(n, dtype=np.int64)
    rng_actions = np.random.default_rng(seed + 1)
    for _ in range(int(limit.max()) // PROTOCOL["frame_skip"] + 2):
        actions = candidate.act(env, obs, own, rng_actions) + rival.act(env, obs, ~own, rng_actions)
        active_before = env.setpiece_team >= 0
        kickoff_before = env.sim.kickoff.copy()
        obs, _, done, info = env.step(actions)
        ticks += PROTOCOL["frame_skip"]
        open_ = ~resolved
        set_piece_done = open_ & (kind != KIND_KICKOFF) & active_before & (env.setpiece_team < 0)
        kickoff_done = open_ & (kind == KIND_KICKOFF) & kickoff_before & ~env.sim.kickoff
        legal = (set_piece_done & (env.setpiece_ticks < env._rs1.c["safety_ticks"])) | kickoff_done
        success |= legal & (ticks <= limit)
        resolved |= set_piece_done | kickoff_done | (ticks > limit) | done
        if resolved.all():
            break
    out = {}
    for code, name in kinds.items():
        mask = kind == code
        if mask.any():
            out[name] = dict(n=int(mask.sum()), success=float(success[mask].mean()))
    out["all"] = dict(n=int(n), success=float(success.mean()))
    return out


def bootstrap_mean(values, clusters, rng, n=2000):
    values, clusters = np.asarray(values, dtype=np.float64), np.asarray(clusters)
    unique = np.unique(clusters)
    means = np.array([values[clusters == c].mean() for c in unique])
    draws = rng.integers(0, len(unique), (n, len(unique)))
    boots = means[draws].mean(axis=1)
    return float(means.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)), boots
