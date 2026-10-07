"""Laterales con el controlador de pase (learn/x4_pass_macro) contra el agente solo, desde laterales humanos.

Motivo (docs/PASE_MACRO.md §7): en self-play el agente conserva su propio lateral mucho menos que un humano (primer
toque propio 0,33 contra 0,75; conserva 3 s 0,15 contra 0,34), así que sacar la pelota por la línea le cuesta poco
contra sí mismo. En un lateral la pelota está quieta y hay tiempo: el mejor caso del controlador.

Estados: inicios de lateral de las grabaciones de entrenamiento (Sanguchito), con la latencia de sala (8–11). Saca el
jugador del equipo del saque más cercano a la pelota; el receptor es el compañero más libre a 80–550 px.
  C. el agente juega con los 8;
  M. el que saca usa el controlador predictivo hacia el receptor; el resto, el agente.
Resultado (mismo criterio que tools/x4_mechanics): cede (lateral mal ejecutado o por tiempo), primer toque después de
sacar (propio o rival, sin contar al que sacó en los 2 primeros pasos), conserva 3 s (ningún toque rival ni saque en
contra), sale (otro saque), gol.

  python -m tools.x4_restart_probe --agent runs/x4_estilo/rl_control/best.pt --n 256 --out reports/x4/restart_probe.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from env.rs4z import contract as C
from env.rs4z import kernel as K
from env.rs4z import obs_v3
from env.rs4z.core import RS4ZEnv
from learn.x4_pass_macro import TEAM, PassMacro
from tools.x4_pass_probe import open_play, openness

ROOT = Path(__file__).resolve().parent.parent
SAMPLES_PER_S = 20


def setup(n, seed, recordings, delays):
    from learn import x4_data as XD
    from learn.x4_ppo import restart_spot
    names = XD.split_names(ROOT / "reports" / "x4" / "splits.json", "train", None, ("sanguchito_rs_x4",))[:recordings]
    d = XD.load(names, maps=("sanguchito_rs_x4",))
    rng = np.random.default_rng(seed)
    v = d.valid
    lat = v[(d.state[v] == 1) & (d.rkind[v] == K.LATERAL) & (d.rage[v] == 0) & (d.rteam[v] >= 0)]
    rng.shuffle(lat)
    env = RS4ZEnv(n, map="sanguchito_rs_x4", frame_skip=3, max_delay=15, deadline=0, kickoff_deadline=C.KICKOFF_SAFETY,
                  seed=seed)
    env.start_match(np.arange(n), delay=rng.choice(delays, size=(n, 8)), match_ticks=3 * 3600)
    taker = np.full(n, -1)
    receiver = np.full(n, -1)
    team = np.full(n, -1)
    k = 0
    for row in range(n):
        while k < len(lat):
            t = int(lat[k])
            k += 1
            held = ((d.inp[t] & 16) != 0) & ~d.kicking[t]
            env.radius[row, 0] = d.ball_r[t]
            env.rf[row, K.RF_KSTR] = d.kstr[t]
            env.place(row, ball_pos=d.ball[t, :2], ball_vel=d.ball[t, 2:4], player_pos=d.pos[t], player_vel=d.vel[t],
                      kick_held=held, last_touch=-1, mass_phase=0 if abs(d.mass[t] - 0.5) < 1e-6 else 1)
            tm = int(d.rteam[t])
            env.start_restart(row, K.LATERAL, tm, restart_spot(env, K.LATERAL, d.ball[t, :2]))
            b = env.pos[row, 0]
            own = np.flatnonzero(TEAM == tm)
            p = int(own[np.argmin(np.hypot(*(env.player_pos[row, own] - b).T))])
            mates = [q for q in own if q != p and 80.0 <= np.hypot(*(env.player_pos[row, q] - b)) <= 550.0]
            if not mates:
                continue
            taker[row], team[row] = p, tm
            receiver[row] = max(mates, key=lambda q: openness(env.player_pos[row], b, tm, q))
            break
    return env, taker, receiver, team


def run(env, snap, taker, receiver, team, agent, mode, decisions, seed):
    env.restore(snap)
    rng = np.random.default_rng(seed)
    n = env.N
    rows = np.flatnonzero(taker >= 0)
    macros = {r: PassMacro(receiver[r], ball_r=float(env.radius[r, 0]), timeout=10 ** 6, no_push=True)
              for r in rows} if mode == "M" else {}
    out = {r: dict(result=None, first=None, released_at=None) for r in rows}
    for s in range(decisions):
        obs = obs_v3.observe(env)
        acts = agent(obs.reshape(n * 8, -1), rng).reshape(n, 8)
        for r in rows:
            o = out[r]
            if mode == "M" and o["released_at"] is None and o["result"] is None and not macros[r].kicked:
                p = taker[r]
                acts[r, p] = macros[r].act(p, env.pos[r, 0], env.vel[r, 0], env.player_pos[r, p], env.player_vel[r, p],
                                           env.player_pos[r], env.player_vel[r], 0, hist=env.act_hist[r, p],
                                           delay=int(env.delay[r, p]))
        ev = env.step(acts)
        touched = ev["touched"] | ev["kicked"]
        live = open_play(env)
        for r in rows:
            o = out[r]
            if o["result"] is not None:
                continue
            p, tm = taker[r], team[r]
            if ev["forfeit"][r] >= 0:
                o["why"] = int(ev["forfeit_why"][r])
            if mode == "M" and not macros[r].kicked:
                macros[r].saw_kick(bool(ev["kicked"][r, p]))
            if o["released_at"] is None:
                if ev["forfeit"][r] >= 0:
                    o["result"] = "cede"
                elif ev["goal"][r] != 0:
                    o["result"] = "gol" if (ev["goal"][r] > 0) == (tm == 0) else "gol_en_contra"
                elif live[r]:
                    o["released_at"] = s
                continue
            others = touched[r].copy()
            if s - o["released_at"] <= 2:
                others[p] = False
            if ev["goal"][r] != 0:
                o["result"] = "gol" if (ev["goal"][r] > 0) == (tm == 0) else "gol_en_contra"
            elif not live[r]:
                rt = int(env.ri[r, K.RI_TEAM])
                o["result"] = "sale_a_favor" if rt == tm else "sale_en_contra"
            elif others[TEAM != tm].any():
                if o["first"] is None:
                    o["first"] = "rival"
                o["result"] = "rival"
            elif others[TEAM == tm].any() and o["first"] is None:
                o["first"] = "propio"
            if o["result"] is None and s - o["released_at"] >= 3 * SAMPLES_PER_S:
                o["result"] = "conserva_3s"
        if all(out[r]["result"] is not None for r in rows):
            break
    for r in rows:
        if out[r]["result"] is None:
            out[r]["result"] = "no_saca" if out[r]["released_at"] is None else "conserva_3s"
    return out


def summarize(out):
    res = [o["result"] for o in out.values()]
    n = len(res)
    rel = [o for o in out.values() if o["released_at"] is not None]
    why = [o.get("why") for o in out.values() if o["result"] == "cede"]
    return dict(n=n, resultados={k: round(res.count(k) / n, 3) for k in sorted(set(res))},
                cede_por={str(k): why.count(k) for k in sorted(set(why), key=str)},
                cede=round(res.count("cede") / n, 3),
                primer_toque_propio=round(sum(o["first"] == "propio" for o in out.values()) / n, 3),
                conserva_3s=round(res.count("conserva_3s") / n, 3),
                segundos_hasta_sacar_p50=round(float(np.median([o["released_at"] for o in rel])) / SAMPLES_PER_S, 2)
                if rel else None)


def main(argv=None):
    import torch
    from learn.x4_eval import Policy
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agent", required=True)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--recordings", type=int, default=150)
    ap.add_argument("--decisions", type=int, default=240, help="12 s: hasta 9 s para sacar más 3 s después")
    ap.add_argument("--delays", default="8,9,10,11")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    torch.manual_seed(a.seed)
    agent = Policy(a.agent, device=a.device)
    env, taker, receiver, team = setup(a.n, a.seed, a.recordings, tuple(int(x) for x in a.delays.split(",")))
    snap = env.snapshot()
    rep = dict(agente=a.agent, laterales=int((taker >= 0).sum()))
    for mode, name in (("C", "agente_solo"), ("M", "controlador")):
        rep[name] = summarize(run(env, snap, taker, receiver, team, agent, mode, a.decisions, a.seed + 1))
        print(name, json.dumps(rep[name], ensure_ascii=False), flush=True)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
