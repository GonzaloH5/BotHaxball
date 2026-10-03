"""Compuerta previa del candidato C en CPU (PLAN_RS4.md 3). Decide si C gasta GPU.

C entra a la comparación sólo si, jugando en el simulador con el árbitro real (greedy, como la
comparación principal del protocolo; el muestreo se informa aparte y no decide):
1. ejecuta >= 80% de sus saques: batería técnica de eval/rs4_b1.restart_battery (la misma de las compuertas);
2. su equipo patea al arco >= 1 vez por minuto (remate = definición de eval/rs4_b1.play_cell);
3. pasa con una relación pases/pérdidas >= 50% de la humana, medida con la misma herramienta
   (tools/rs4_b1_demos.touch_metrics, a la cadencia de decisión) que las grabaciones de desarrollo.

Partidos: C controla a los cuatro jugadores de su equipo contra el scripted R3 equilibrado (estilo 0,
control básico de la familia de entrenamiento), la mitad de rojo y la mitad de azul, desde estados
reales de juego abierto de la partición de desarrollo.

  python -m tools.rs4_b1_c_gate --candidate runs/rs4_b1/c_bc/bc.pt --out reports/rs4_b1/c_gate.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
THRESHOLDS = dict(restarts=0.80, shots_per_minute=1.0, pass_turnover_fraction_of_human=0.50)


def play(candidate, rival, bank, state_ids, seed, greedy, minutes):
    """C (4 jugadores) contra el rival; devuelve remates, pases y pérdidas del equipo de C y goles."""
    from env.rs4_states import apply_states
    from eval.rs4_b1 import PROTOCOL, environment
    from tools.rs4_b1_demos import touch_metrics
    games = len(state_ids)
    n = 2 * games
    colors = np.arange(n) % 2  # color del equipo de C
    env = environment(n, seed, minutes)
    env.reset()
    apply_states(env, np.arange(n), bank, np.repeat(state_ids, 2), mirror=colors == 1)
    env.match_ticks[:] = 0
    env.match_score[:] = 0
    obs = env.observe()
    sim = env.sim
    own = sim.player_team[None, :] == colors[:, None]
    candidate.reset(n, 8)
    rival.reset(n, 8)
    rng = np.random.default_rng(seed)
    decisions = int(minutes * 3600) // PROTOCOL["frame_skip"]
    shots = np.zeros(n)
    touches, balls = [], []
    finished = np.zeros(n, dtype=bool)
    final = np.zeros((n, 2), dtype=np.int64)
    sign = np.where(colors == 0, 1.0, -1.0)
    for _ in range(decisions + 2):
        actions = candidate.act(env, obs, own, rng, greedy) + rival.act(env, obs, ~own, rng, True)
        obs, _, done, info = env.step(actions)
        live = ~finished
        kicked_team = (info["kicked"] & own).any(axis=1)
        bv = sim.ball_vel
        towards = sign * bv[:, 0] > 2.0
        t = (sign * env.goal_x - sim.ball_pos[:, 0]) / np.where(np.abs(bv[:, 0]) > 1e-6, bv[:, 0], 1e-6)
        on_target = towards & (np.abs(sim.ball_pos[:, 1] + bv[:, 1] * np.maximum(t, 0)) < sim.st.goal_half_height + 30)
        shots += kicked_team & on_target & live
        open_play = ~sim.kickoff & (env.setpiece_team < 0)
        touches.append((env._rs1.contact | info["kicked"]) & (live & open_play)[:, None])
        balls.append(sim.ball_pos.copy())
        ended = info["match_done"] & live
        final[ended] = info["final_score"][ended]
        finished |= ended
        if done.any():
            for policy in (candidate, rival):
                policy.reset(n, 8, np.flatnonzero(done))
        if finished.all():
            break
    touch, ball = np.stack(touches, axis=1), np.stack(balls, axis=1)
    passes = turnovers = 0
    for row in range(n):
        p, t = touch_metrics(touch[row], sim.player_team, ball[row])
        passes += p[colors[row]]
        turnovers += t[colors[row]]
    goals_for = final[np.arange(n), colors].sum()
    goals_against = final[np.arange(n), 1 - colors].sum()
    return dict(matches=int(n), minutes=float(n * minutes), shots=float(shots.sum()), passes=int(passes),
                turnovers=int(turnovers), goals_for=int(goals_for), goals_against=int(goals_against))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--rival", default="scripted:r3:0")
    ap.add_argument("--states", default=str(ROOT / "data" / "rs4_states" / "desarrollo.npz"))
    ap.add_argument("--demos", default=str(ROOT / "data" / "rs4_b1_demos" / "manifest.json"))
    ap.add_argument("--games", type=int, default=32, help="estados iniciales (x2 colores)")
    ap.add_argument("--minutes", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import torch
    torch.set_num_threads(4)
    from env.rs4_states import KIND_OPEN, StateBank
    from eval.rs4_b1 import Policy, restart_battery
    from tools.evaluate_rs4_b1 import EVAL_RIVALS, _resolve
    started = time.time()
    bank = StateBank(a.states)
    state_ids = np.sort(np.random.default_rng(a.seed).choice(bank.select([KIND_OPEN]), a.games, replace=False))
    human = json.loads(Path(a.demos).read_text(encoding="utf-8"))["desarrollo"]["pass_turnover_ratio"]
    report = dict(version="RS4-b1-c-gate-1", candidate=a.candidate, rival=a.rival, thresholds=THRESHOLDS,
                  human_pass_turnover_ratio=human, games=a.games, minutes=a.minutes, seed=a.seed, modes={})
    for mode, greedy in (("greedy", True), ("muestreo", False)):
        candidate = Policy(a.candidate)
        if not greedy:
            act = candidate.act
            candidate.act = lambda env, obs, mask, rng, greedy=True, _act=act: _act(env, obs, mask, rng, False)
        battery = restart_battery(candidate, Policy(_resolve(EVAL_RIVALS[0])), bank, a.seed)
        games = play(candidate, Policy(a.rival), bank, state_ids, a.seed, greedy, a.minutes)
        shots_per_minute = games["shots"] / games["minutes"]
        ratio = games["passes"] / max(games["turnovers"], 1)
        checks = dict(restarts=battery["all"]["success"] >= THRESHOLDS["restarts"],
                      shots_per_minute=shots_per_minute >= THRESHOLDS["shots_per_minute"],
                      pass_turnover=ratio >= THRESHOLDS["pass_turnover_fraction_of_human"] * human)
        report["modes"][mode] = dict(restart_battery=battery, games=games, shots_per_minute=shots_per_minute,
                                     pass_turnover_ratio=ratio, pass_turnover_fraction_of_human=ratio / human,
                                     checks=checks, passed=all(checks.values()))
        print(f"{mode}: saques {100 * battery['all']['success']:.1f}% | remates/min {shots_per_minute:.2f} | "
              f"pases/pérdidas {ratio:.2f} ({100 * ratio / human:.0f}% de la humana {human:.2f}) | "
              f"goles {games['goals_for']}-{games['goals_against']} | {'PASA' if all(checks.values()) else 'NO PASA'}",
              flush=True)
    report["passed"] = report["modes"]["greedy"]["passed"]
    report["seconds"] = round(time.time() - started, 1)
    Path(a.out).write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"compuerta de C: {'APROBADA' if report['passed'] else 'NO APROBADA'} (decide greedy) | {report['seconds']} s")


if __name__ == "__main__":
    main()
