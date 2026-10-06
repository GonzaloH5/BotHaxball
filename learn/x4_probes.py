"""Sondas de coordinación 2v1 y 3v2 desde estados humanos (E0; Liu 2019, arXiv 1902.07151).

De las grabaciones de prueba se toman estados de juego abierto en los que un equipo tiene la pelota (un
jugador a ≤ 30 px) en campo rival. Se deja al portador, sus compañeros más cercanos y los rivales más cercanos
a la pelota (2v1 o 3v2; el resto se desactiva) y la política juega `--horizon` decisiones con latencia de sala.
Se mide qué pasa primero: pase completado (toca un compañero después del portador), pérdida (toca un rival o
la pelota sale), gol o nada. Es una medida relativa entre políticas (BC contra snapshots del RL), no un
número humano: en Liu 2019 la tasa de pase en la sonda 2v1 crece con el entrenamiento. Como referencia aproximada se
informa además qué hicieron los humanos desde esos mismos estados en la grabación (`human_baseline`, con los 8 jugadores).

  python -m learn.x4_probes --policy runs/x4_bc/final/best.pt --n 256 --out reports/x4/probes_bc.json
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
CONTACT = 15.0 + 8.325 + 2.0


def pick_states(data, n_att, n_def, n, rng, min_x=150.0):
    """(tick, equipo atacante, lugares activos) de estados con posesión en campo rival."""
    v = data.valid
    v = v[(data.state[v] == 1) & (data.rkind[v] == 0)]
    rng.shuffle(v)
    out = []
    for t in v:
        b = data.ball[t, :2]
        d = np.hypot(data.pos[t, :, 0] - b[0], data.pos[t, :, 1] - b[1])
        c = int(np.argmin(d))
        if d[c] > 30.0:
            continue
        team = TEAM[c]
        s = 1.0 if team == 0 else -1.0
        if b[0] * s < min_x:
            continue
        mates = [q for q in np.argsort(d) if TEAM[q] == team and q != c][:n_att - 1]
        rivals = [q for q in np.argsort(d) if TEAM[q] != team][:n_def]
        active = np.zeros(8, bool)
        active[[c] + list(mates) + list(rivals)] = True
        out.append((int(t), int(team), int(c), active))
        if len(out) >= n:
            break
    return out


def human_baseline(data, states, horizon=100, stride=3):
    """Lo que pasó en la grabación desde los mismos estados (referencia humana aproximada: en la grabación juegan
    los 8 jugadores, no 2v1 ni 3v2). Mismas reglas que `run`: toque = contacto o patada en la ventana del muestreo;
    primero que ocurra entre pase completado, pérdida, salida (saque del script) o gol."""
    out = {}
    passes_any = 0
    goals = 0
    T = data.ticks
    for t, team, c, _ in states:
        last, passes, res = c, 0, "nada"
        rec = data.rec_of_tick[t]
        for j in range(1, horizon + 1):
            u = t + stride * j
            if u >= T or data.rec_of_tick[u] != rec or np.any(data.pid[u] != data.pid[t]):
                res = "corte"
                break
            if data.state[u] == 2:          # gol (animación); el arco por el lado de la pelota
                bx = data.ball[u, 0]
                res = "gol" if (bx > 0) == (team == 0) else "gol_en_contra"
                break
            if data.state[u] != 1 or data.rkind[u] > 0:
                res = "salida"
                break
            b = data.ball[u, :2]
            d = np.hypot(data.pos[u, :, 0] - b[0], data.pos[u, :, 1] - b[1])
            touch = d <= CONTACT
            touch |= data.kick_ev[max(u - stride + 1, 0):u + 1].any(0)
            who = np.flatnonzero(touch)
            if len(who) == 0:
                continue
            if any(TEAM[q] != team for q in who):
                res = "pase_y_perdida" if passes else "perdida"
                break
            q = who[np.argmin(d[who])]
            if q != last:
                passes += 1
                last = q
        if res == "nada" and passes:
            res = "pase"
        out[res] = out.get(res, 0) + 1
        passes_any += passes > 0
        goals += res == "gol"
    n = len(states)
    out.update(n=n, con_pase=int(passes_any), tasa_pase=round(passes_any / max(1, n), 3),
               tasa_gol=round(goals / max(1, n), 3))
    return out


@torch.no_grad()
def run(policy, data, states, *, map_name="sanguchito_rs_x4", horizon=100, delays=(8, 9, 10, 11), seed=0):
    rng = np.random.default_rng(seed)
    N = len(states)
    env = RS4ZEnv(N, map=map_name, frame_skip=3, max_delay=15, deadline=0, kickoff_deadline=0, seed=seed)
    act = np.stack([s[3] for s in states])
    env.start_match(np.arange(N), active=act, delay=rng.choice(delays, size=(N, 8)), match_ticks=10 ** 9)
    carrier = np.array([s[2] for s in states])
    team = np.array([s[1] for s in states])
    for i, (t, _, _, a) in enumerate(states):
        held = ((data.inp[t] & 16) != 0) & ~data.kicking[t]
        env.radius[i, 0] = data.ball_r[t]
        env.place(i, ball_pos=data.ball[t, :2], ball_vel=data.ball[t, 2:4], player_pos=data.pos[t],
                  player_vel=data.vel[t], kick_held=held, last_touch=int(states[i][1]),
                  mass_phase=0 if abs(data.mass[t] - 0.5) < 1e-6 else 1)
    outcome = np.full(N, "nada", dtype=object)
    last = carrier.copy()
    passes = np.zeros(N, int)
    done = np.zeros(N, bool)
    for _ in range(horizon):
        obs = obs_v3.observe(env)
        flat = obs.reshape(N * 8, -1)
        logits = policy(torch.from_numpy(flat))
        a = torch.distributions.Categorical(logits=logits).sample().numpy().reshape(N, 8)
        ev = env.step(a)
        bp = env.ball_pos
        pp = env.player_pos
        d = np.hypot(pp[..., 0] - bp[:, None, 0], pp[..., 1] - bp[:, None, 1])
        touch = (d <= CONTACT) & env.active | ev["kicked"]
        for i in np.flatnonzero(~done):
            if ev["goal"][i] != 0:
                outcome[i] = "gol" if (ev["goal"][i] > 0) == (team[i] == 0) else "gol_en_contra"
                done[i] = True
                continue
            if ev["out"][i] or ev["restart_start"][i] != 0:
                outcome[i] = "salida"
                done[i] = True
                continue
            who = np.flatnonzero(touch[i])
            if len(who) == 0:
                continue
            if any(TEAM[q] != team[i] for q in who):
                outcome[i] = "pase_y_perdida" if passes[i] else "perdida"
                done[i] = True
                continue
            q = who[np.argmin(d[i, who])]
            if q != last[i]:
                passes[i] += 1
                last[i] = q
        if done.all():
            break
    for i in np.flatnonzero(~done):
        if passes[i]:
            outcome[i] = "pase"
    vals, cnt = np.unique(outcome.astype(str), return_counts=True)
    res = {v: int(c) for v, c in zip(vals, cnt)}
    res["con_pase"] = int((passes > 0).sum())
    res["n"] = N
    res["tasa_pase"] = round(float((passes > 0).mean()), 3)
    res["tasa_gol"] = round(float((outcome == "gol").mean()), 3)
    return res


def main():
    from learn.x4_eval import Policy
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--horizon", type=int, default=100, help="decisiones (5 s)")
    ap.add_argument("--recordings", type=int, default=40)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    names = XD.split_names(ROOT / "reports" / "x4" / "splits.json", a.split, None, ("sanguchito_rs_x4",))
    data = XD.load(names[:a.recordings], maps=("sanguchito_rs_x4",))
    pol = Policy(a.policy)
    model = pol.model
    rng = np.random.default_rng(0)
    report = dict(policy=a.policy, split=a.split, horizon=a.horizon)
    for name, (na, nd) in (("2v1", (2, 1)), ("3v2", (3, 2))):
        states = pick_states(data, na, nd, a.n, rng)
        report[name] = run(model, data, states, horizon=a.horizon, seed=1 + na)
        report[name + "_humanos_4v4"] = human_baseline(data, states, horizon=a.horizon)
        print(name, json.dumps(report[name]), flush=True)
        print(name, "humanos (4v4, misma situación)", json.dumps(report[name + "_humanos_4v4"]), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
