"""Prueba del controlador de pase (learn/x4_pass_macro.py) en simulación, antes de meterlo en el RL.

Arranca de momentos de partidos humanos de Sanguchito (split de entrenamiento) en que un jugador tiene la pelota
controlada, con la latencia de la sala (D de 8 a 11 ticks). Desde los mismos estados compara:
  A0. el que tiene la pelota usa el controlador reactivo (predice su posición sin las teclas en cola) para pasarle al
      compañero más libre; el resto, el agente;
  A. igual con el controlador predictivo (teclas en cola y búsqueda sobre las 18 acciones);
  B. como A, pero el receptor va a buscar la pelota después de la patada;
  C. el agente solo (sin controlador): qué hace hoy en esas situaciones.
Cada modo también se desglosa por el giro necesario: el ángulo entre jugador → pelota y pelota → receptor.
El pase se cuenta como en tools/x4_pass_chain: el primer toque después de la patada es de un compañero.

  python -m tools.x4_pass_probe --agent runs/x4_estilo/rl_control/best.pt --n 512 --out reports/x4/pass_probe.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from env.rs4z import contract as C
from env.rs4z import kernel as K
from env.rs4z import obs_v3
from env.rs4z.core import RS4ZEnv
from learn.x4_pass_macro import PLAYER_R, TEAM, PassMacro, move_toward, own_action

ROOT = Path(__file__).resolve().parent.parent


def open_play(env):
    return (env.ri[:, K.RI_KO] == 0) & (env.ri[:, K.RI_TEAM] < 0) & (env.ri[:, K.RI_PEND] == 0)


def pick_receiver(ball, pos, carrier):
    """El compañero más libre a 120–550 px: el que tiene más lejos al rival más cercano a la línea de pase."""
    best, best_open = -1, -1.0
    team = TEAM[carrier]
    rivals = [q for q in range(8) if TEAM[q] != team]
    for m in range(8):
        if m == carrier or TEAM[m] != team:
            continue
        seg = pos[m] - ball
        L = float(np.linalg.norm(seg))
        if not 120.0 <= L <= 550.0:
            continue
        u = seg / L
        openness = min(float(np.linalg.norm((pos[q] - ball) - np.clip((pos[q] - ball) @ u, 0, L) * u)) for q in rivals)
        if openness > best_open:
            best, best_open = m, openness
    return best, best_open


def setup(n, seed, names_limit, delays):
    """Partidos colocados en momentos humanos con la pelota controlada; devuelve el entorno, quién la tiene y a quién
    pasar."""
    from learn import x4_data as XD
    from learn.x4_ppo import StateBank
    names = XD.split_names(ROOT / "reports" / "x4" / "splits.json", "train", None, ("sanguchito_rs_x4",))[:names_limit]
    bank = StateBank(XD.load(names, maps=("sanguchito_rs_x4",)), seed=seed)
    rng = np.random.default_rng(seed)
    env = RS4ZEnv(n, map="sanguchito_rs_x4", frame_skip=3, max_delay=15, deadline=0, kickoff_deadline=C.KICKOFF_SAFETY,
                  seed=seed)
    env.start_match(np.arange(n), delay=rng.choice(delays, size=(n, 8)), match_ticks=3 * 3600)
    carrier = np.full(n, -1)
    receiver = np.full(n, -1)
    for row in range(n):
        for _ in range(60):
            bank.apply(env, row, restart_frac=0.0)
            b = env.pos[row, 0]
            d = np.hypot(*(env.player_pos[row] - b).T)
            p = int(np.argmin(d))
            if d[p] > PLAYER_R + env.radius[row, 0] + 12.0 or not open_play(env)[row]:
                continue
            r, _ = pick_receiver(b, env.player_pos[row], p)
            if r >= 0:
                carrier[row], receiver[row] = p, r
                break
    return env, carrier, receiver


def run(env, snap, carrier, receiver, agent, mode, decisions, seed):
    """Juega desde el estado guardado. mode: "A0" controlador reactivo, "A" predictivo, "B" predictivo + receptor que
    va a la pelota, "C" agente."""
    env.restore(snap)
    rng = np.random.default_rng(seed)
    n = env.N
    rows = np.flatnonzero(carrier >= 0)
    macros = {r: PassMacro(receiver[r], ball_r=float(env.radius[r, 0])) for r in rows} if mode != "C" else {}
    out = {r: dict(result=None, kick_step=None, aim_err=None) for r in rows}
    kick_at = {}
    for s in range(decisions):
        obs = obs_v3.observe(env)
        acts = agent(obs.reshape(n * 8, -1), rng).reshape(n, 8)
        for r in rows:
            if out[r]["result"] is not None:
                continue
            p = carrier[r]
            lag = int(env.delay[r, p]) + 2
            if mode != "C":
                mac = macros[r]
                if not mac.kicked:
                    q = dict(hist=env.act_hist[r, p], delay=int(env.delay[r, p])) if mode != "A0" else {}
                    acts[r, p] = mac.act(p, env.pos[r, 0], env.vel[r, 0], env.player_pos[r, p], env.player_vel[r, p],
                                         env.player_pos[r], env.player_vel[r], lag, **q)
                elif mode == "B":
                    m = receiver[r]
                    tgt = env.pos[r, 0] + env.vel[r, 0] * 8.0
                    acts[r, m] = own_action(move_toward(*(tgt - env.player_pos[r, m]), dead=0.0), False, int(TEAM[m]))
        ev = env.step(acts)
        touched = ev["touched"] | ev["kicked"]
        live = open_play(env)
        for r in rows:
            o = out[r]
            if o["result"] is not None:
                continue
            p, team = carrier[r], TEAM[carrier[r]]
            if mode != "C":
                mac = macros[r]
                if not mac.kicked:
                    mac.saw_kick(bool(ev["kicked"][r, p]))
                    if mac.kicked:
                        kick_at[r] = s
                        o["kick_step"] = s
                        v = env.vel[r, 0]
                        aim = mac.aim - env.pos[r, 0]
                        o["aim_err"] = math.degrees(math.acos(max(-1.0, min(1.0, float(v @ aim) / (
                            np.linalg.norm(v) * np.linalg.norm(aim) + 1e-9)))))
                        continue
                    if touched[r, TEAM != team].any():
                        o["result"] = "pierde_antes_de_patear"
                    elif not live[r]:
                        o["result"] = "saque_antes_de_patear"
                    elif mac.done:
                        o["result"] = "no_pateo"
                    continue
            # después de la patada (o en C, desde el principio): primer toque de otro jugador
            others = touched[r].copy()
            if mode != "C" and s - kick_at.get(r, -99) <= 2:
                others[p] = False                            # el que pateó sigue en contacto un instante
            if not live[r] or ev["goal"][r] != 0:
                o["result"] = "gol" if ev["goal"][r] != 0 else "saque"
            elif others[TEAM != team].any():
                o["result"] = "rival"
            elif mode == "C" and others[(TEAM == team) & (np.arange(8) != p)].any():
                o["result"] = "pase"
            elif mode != "C" and others[receiver[r]]:
                o["result"] = "receptor"
            elif mode != "C" and others[(TEAM == team) & (np.arange(8) != p)].any():
                o["result"] = "otro_compañero"
            elif mode != "C" and others[p]:
                o["result"] = "vuelve_al_que_pateo"
        if all(out[r]["result"] is not None for r in rows):
            break
    for r in rows:
        if out[r]["result"] is None:
            out[r]["result"] = "nadie" if (mode == "C" or macros[r].kicked) else "no_pateo"
    return out


def turn_angles(env, carrier, receiver):
    """Giro necesario al empezar (grados): entre jugador → pelota y pelota → receptor."""
    out = {}
    for r in np.flatnonzero(carrier >= 0):
        b = env.pos[r, 0]
        u = b - env.player_pos[r, carrier[r]]
        w = env.player_pos[r, receiver[r]] - b
        c = float(u @ w) / (np.linalg.norm(u) * np.linalg.norm(w) + 1e-9)
        out[r] = math.degrees(math.acos(max(-1.0, min(1.0, c))))
    return out


def by_turn(out, turns, bins=((0, 45), (45, 90), (90, 181))):
    """Pase completado (a un compañero) por giro necesario."""
    res = {}
    for lo, hi in bins:
        rows = [r for r in out if lo <= turns[r] < hi]
        if rows:
            ok = sum(out[r]["result"] in ("receptor", "otro_compañero", "pase") for r in rows)
            res[f"{lo}-{min(hi, 180)}"] = dict(n=len(rows), pase=round(ok / len(rows), 3))
    return res


def summarize(out):
    res = [o["result"] for o in out.values()]
    n = len(res)
    count = {k: res.count(k) for k in sorted(set(res))}
    kicked = [o for o in out.values() if o["kick_step"] is not None]
    s = dict(n=n, resultados={k: round(v / n, 3) for k, v in count.items()})
    if kicked:
        s["pateo"] = round(len(kicked) / n, 3)
        s["decisiones_hasta_patear_p50"] = float(np.median([o["kick_step"] for o in kicked]))
        s["error_de_direccion_p50_grados"] = round(float(np.median([o["aim_err"] for o in kicked])), 1)
        ok = sum(o["result"] in ("receptor", "otro_compañero") for o in kicked)
        s["completados_sobre_pateados"] = round(ok / len(kicked), 3)
        s["al_receptor_sobre_pateados"] = round(sum(o["result"] == "receptor" for o in kicked) / len(kicked), 3)
    return s


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agent", required=True, help="checkpoint que juega con el resto de los jugadores")
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--recordings", type=int, default=150)
    ap.add_argument("--decisions", type=int, default=100, help="5 s")
    ap.add_argument("--delays", default="8,9,10,11")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--modes", default="A0,A,C")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    import torch
    from learn.x4_eval import Policy
    torch.manual_seed(a.seed)
    agent = Policy(a.agent, device=a.device)
    env, carrier, receiver = setup(a.n, a.seed, a.recordings, tuple(int(x) for x in a.delays.split(",")))
    snap = env.snapshot()
    rep = dict(agente=a.agent, partidos=int((carrier >= 0).sum()), delays=a.delays)
    turns = turn_angles(env, carrier, receiver)
    names = dict(A0="controlador_reactivo", A="controlador", B="controlador_receptor_busca", C="agente_solo")
    for mode in a.modes.split(","):
        out = run(env, snap, carrier, receiver, agent, mode, a.decisions, a.seed + 1)
        rep[names[mode]] = summarize(out)
        rep[names[mode]]["por_giro"] = by_turn(out, turns)
        print(names[mode], json.dumps(rep[names[mode]], ensure_ascii=False), flush=True)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
