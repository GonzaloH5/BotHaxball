"""Conducción X4 según quién defiende: ¿el self-play castiga conducir como lo castiga un humano?

Hipótesis (2026-10-07): la política no necesita pasar porque en self-play conducir no se castiga. Contra humanos
("en real soccer no se le gana a un humano 1v1") sí. Para probarla se juegan los mismos estados con distintas
combinaciones de atacante y defensor.

Estados: de las grabaciones humanas de test de Sanguchito, momentos de juego abierto 4v4 en que un jugador tiene la
pelota (a ≤ 12 px del borde) y nadie más está a menos de 20 px de ella. Se juega `--horizon` decisiones (5 s) con la latencia
de sala; el equipo con la pelota lo maneja el atacante y el otro el defensor. Se registra qué pasa primero con el
portador:
  pierde: la toca un rival (contacto o patada) antes que un compañero;
  pasa: la toca primero un compañero;
  gol / gol_en_contra / sale (lateral, córner o saque de arco);
  conserva: nada de eso en el horizonte.
Además: segundos con la pelota hasta el evento y avance de la pelota hacia el arco rival (px) mientras la tiene.
Los resultados se desglosan por presión inicial (distancia del rival más cercano a la pelota).

Referencia humana: lo que pasó en la grabación desde los mismos estados (humanos contra humanos), con las mismas reglas.

  python -m tools.x4_dribble_probe --agent runs/x4_estilo/rl_control/best.pt --n 512 --out reports/x4/dribble_probe.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from env.rs4z import obs_v3
from env.rs4z.core import RS4ZEnv
from learn import x4_data as XD

ROOT = Path(__file__).resolve().parent.parent
TEAM = np.array([0, 0, 0, 0, 1, 1, 1, 1])
PLAYER_R = 15.0
REACH = 4.0                     # alcance de patada: el kernel cuenta "toque" con el borde a menos de esto
PRESSURE_BINS = ((0, 90), (90, 160), (160, 10 ** 6))   # px del rival más cercano a la pelota (centro)


def pick_states(data, n, rng, hold_gap=12.0, free_gap=20.0, stride=3, horizon=100):
    """(tick, portador, presión) de estados 4v4 de juego abierto con un portador claro y grabación suficiente."""
    v = data.valid
    v = v[(data.state[v] == 1) & (data.rkind[v] == 0)]
    v = v[v + stride * horizon < data.ticks]
    rng.shuffle(v)
    out = []
    for t in v:
        if data.rec_of_tick[t + stride * horizon] != data.rec_of_tick[t]:
            continue
        b = data.ball[t, :2]
        gap = np.hypot(data.pos[t, :, 0] - b[0], data.pos[t, :, 1] - b[1]) - PLAYER_R - data.ball_r[t]
        c = int(np.argmin(gap))
        if gap[c] > hold_gap or np.sort(gap)[1] < free_gap:
            continue
        rivals = TEAM != TEAM[c]
        pressure = float(np.hypot(*(data.pos[t, rivals] - b).T).min())
        out.append((int(t), c, pressure))
        if len(out) >= n:
            break
    return out


class Outcome:
    """Primer evento del portador, con el mismo criterio en simulación y en la grabación."""

    def __init__(self, states):
        n = len(states)
        self.carrier = np.array([s[1] for s in states])
        self.team = TEAM[self.carrier]
        self.result = np.full(n, "conserva", dtype=object)
        self.done = np.zeros(n, bool)
        self.steps = np.zeros(n, int)
        self.x0 = None
        self.adv = np.zeros(n)

    def start(self, ball_x):
        self.x0 = np.where(self.team == 0, ball_x, -ball_x)

    def step(self, i, s, ball_x, touch, goal, out):
        """touch (8,) bool en este paso; goal: +1 rojo, −1 azul, 0; out: la pelota salió o empezó un saque."""
        if self.done[i]:
            return
        c, team = self.carrier[i], self.team[i]
        self.steps[i] = s + 1
        self.adv[i] = max(self.adv[i], (ball_x if team == 0 else -ball_x) - self.x0[i])
        if goal != 0:
            self.result[i] = "gol" if (goal > 0) == (team == 0) else "gol_en_contra"
        elif out:
            self.result[i] = "sale"
        elif touch[TEAM != team].any():
            self.result[i] = "pierde"
        elif touch[(TEAM == team) & (np.arange(8) != c)].any():
            self.result[i] = "pasa"
        else:
            return
        self.done[i] = True

    def summary(self, pressure):
        res = self.result.astype(str)

        def block(m):
            n = int(m.sum())
            if not n:
                return dict(n=0)
            vals, cnt = np.unique(res[m], return_counts=True)
            return dict(n=n, **{v: round(c / n, 3) for v, c in zip(vals, cnt)},
                        segundos_con_pelota_p50=round(float(np.median(self.steps[m])) * 3 / 60, 2),
                        avance_px_p50=round(float(np.median(self.adv[m])), 1))
        out = block(np.ones(len(res), bool))
        out["por_presion"] = {f"{lo}-{hi if hi < 10 ** 6 else 'inf'}": block((pressure >= lo) & (pressure < hi))
                              for lo, hi in PRESSURE_BINS}
        return out


@torch.no_grad()
def simulate(attacker, defender, data, states, horizon, delays, seed):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    N = len(states)
    env = RS4ZEnv(N, map="sanguchito_rs_x4", frame_skip=3, max_delay=15, deadline=0, kickoff_deadline=0, seed=seed)
    env.start_match(np.arange(N), delay=rng.choice(delays, size=(N, 8)), match_ticks=10 ** 9)
    for i, (t, c, _) in enumerate(states):
        held = ((data.inp[t] & 16) != 0) & ~data.kicking[t]
        env.radius[i, 0] = data.ball_r[t]
        env.place(i, ball_pos=data.ball[t, :2], ball_vel=data.ball[t, 2:4], player_pos=data.pos[t],
                  player_vel=data.vel[t], kick_held=held, last_touch=int(TEAM[c]),
                  mass_phase=0 if abs(data.mass[t] - 0.5) < 1e-6 else 1)
    oc = Outcome(states)
    oc.start(env.ball_pos[:, 0].copy())
    attacking = TEAM[None, :] == oc.team[:, None]                     # (N, 8) lugares del equipo con la pelota
    for s in range(horizon):
        flat = obs_v3.observe(env).reshape(N * 8, -1)
        a_att = attacker(flat, rng).reshape(N, 8)
        a_def = defender(flat, rng).reshape(N, 8) if defender is not attacker else a_att
        ev = env.step(np.where(attacking, a_att, a_def))
        touch = ev["touched"] | ev["kicked"]
        out = ev["out"] | (ev["restart_start"] != 0)
        bx = env.ball_pos[:, 0]
        for i in np.flatnonzero(~oc.done):
            oc.step(i, s, bx[i], touch[i], int(ev["goal"][i]), bool(out[i]))
        if oc.done.all():
            break
    return oc


def human(data, states, horizon, stride=3):
    """Lo que pasó en la grabación desde los mismos estados (humanos contra humanos)."""
    oc = Outcome(states)
    oc.start(np.array([data.ball[t, 0] for t, _, _ in states]))
    for i, (t, _, _) in enumerate(states):
        for s in range(horizon):
            u = t + stride * (s + 1)
            if data.rec_of_tick[u] != data.rec_of_tick[t] or np.any(data.pid[u] != data.pid[t]):
                break
            goal = 0
            if data.state[u] == 2:
                goal = 1 if data.ball[u, 0] > 0 else -1
            out = data.state[u] != 1 or data.rkind[u] > 0
            b = data.ball[u, :2]
            gap = np.hypot(data.pos[u, :, 0] - b[0], data.pos[u, :, 1] - b[1]) - PLAYER_R - data.ball_r[u]
            touch = (gap < REACH) | data.kick_ev[u - stride + 1:u + 1].any(0)
            oc.step(i, s, data.ball[u, 0], touch, goal, bool(out and goal == 0))
            if oc.done[i]:
                break
    return oc


def main(argv=None):
    from learn.x4_eval import Policy
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agent", required=True, help="checkpoint del RL")
    ap.add_argument("--bc", default="runs/x4_bc/final_sangu_rsone/best.pt")
    ap.add_argument("--split", default="test")
    ap.add_argument("--recordings", type=int, default=80)
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--horizon", type=int, default=100, help="decisiones (5 s)")
    ap.add_argument("--delays", default="8,9,10,11")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    names = XD.split_names(ROOT / "reports" / "x4" / "splits.json", a.split, None, ("sanguchito_rs_x4",))
    data = XD.load(names[:a.recordings], maps=("sanguchito_rs_x4",))
    states = pick_states(data, a.n, np.random.default_rng(a.seed), horizon=a.horizon)
    pressure = np.array([s[2] for s in states])
    delays = tuple(int(x) for x in a.delays.split(","))
    pols = dict(agente=Policy(a.agent, device=a.device), bc=Policy(a.bc, device=a.device))
    rep = dict(agente=a.agent, bc=a.bc, estados=len(states), horizonte_s=a.horizon * 3 / 60)
    rep["humanos_vs_humanos"] = human(data, states, a.horizon).summary(pressure)
    print("humanos_vs_humanos", json.dumps(rep["humanos_vs_humanos"], ensure_ascii=False), flush=True)
    for att in ("agente", "bc"):
        for de in ("agente", "bc"):
            key = f"{att}_ataca_vs_{de}_defiende"
            oc = simulate(pols[att], pols[de], data, states, a.horizon, delays, a.seed + 1)
            rep[key] = oc.summary(pressure)
            print(key, json.dumps(rep[key], ensure_ascii=False), flush=True)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
