"""Mecánicas de HaxBall que usan los humanos, en grabaciones y en self-play (definiciones del usuario, 2026-10-07).

Las tres existen en la física del simulador (env/rs4z/kernel.physics_tick): dos patadas en la misma ventana suman sus
impulsos, la pelota rebota en los jugadores (y la patada de quien la recibe suma el suyo) y los jugadores chocan entre
sí. La pregunta es si el agente las usa y con qué frecuencia y resultado lo hacen los humanos.

* **Rifle**: dos jugadores del mismo equipo patean la pelota en la misma ventana de 2 muestreos (6 ticks) llegando
  desde lados opuestos en y. Las componentes verticales se cancelan y sale horizontal y rápida. Se mide la velocidad,
  cuán horizontal sale, si va hacia el arco rival y si hay gol en los 3 s siguientes.
* **Usar**: una patada con un rival a < 80 px en la dirección en que sale la pelota (coseno > 0,8) y que ese rival
  toca en ≤ 0,5 s. Resultado en los 1,5 s siguientes: gol, saque a favor, rebote propio (la vuelve a tocar el equipo que
  pateó), saque en contra, el rival se queda con ella. Además, si el rival pateó al recibirla y la pelota salió hacia
  su propio arco. No se puede saber la intención: son todas las patadas contra un rival cercano y su resultado.
* **Cuerpear**: el que tiene la pelota (a ≤ 8 px del borde) está en contacto con un rival (centros a ≤ 33 px) que está
  más lejos de la pelota y detrás suyo (el portador entre el rival y la pelota). Tramos de muestreos seguidos; resultado
  en 1,5 s desde que termina: la toca un compañero (apoyo), el mismo portador sigue y patea, o la toca un rival.

Todo se cuenta por minuto de juego abierto, en el formato `Episode` de tools/x4_metrics (muestreo cada 3 ticks), igual
para grabaciones (`episodes_from_ticks`) y simulación (`EpisodeRecorder` vía learn.x4_eval.play).

  python -m tools.x4_mechanics --agent runs/x4_estilo/rl_control/best.pt --out reports/x4/mechanics.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tools import x4_metrics as XM

ROOT = Path(__file__).resolve().parent.parent
TEAMS = np.array([0, 0, 0, 0, 1, 1, 1, 1])
PLAYER_R = 15.0
SAMPLES_PER_S = 20                       # muestreo cada 3 ticks a 60 ticks/s


def sign(team):
    return 1.0 if team == 0 else -1.0


def detect(ep):
    """Eventos de las tres mecánicas y patadas simples de referencia en un tramo."""
    T = len(ep.ball)
    b, bv = ep.ball[:, :2], ep.ball[:, 2:4]
    d = np.hypot(ep.pos[..., 0] - b[:, None, 0], ep.pos[..., 1] - b[:, None, 1])
    gap = d - PLAYER_R - ep.ball_r
    owner, _ = XM._touch_owner(ep)
    open_ = ep.open_play
    goal = ep.goal_ev if ep.goal_ev is not None else np.zeros(T, np.int8)
    rteam = ep.restart_team if ep.restart_team is not None else np.full(T, -1)
    rifles, usos, cuerpeos, simples = [], [], [], []
    used = np.zeros(T, bool)
    # ---- rifle
    for i in range(T - 3):
        if not open_[i] or used[i] or not ep.kicked[i].any():
            continue
        win = ep.kicked[i] | ep.kicked[i + 1]
        for team in (0, 1):
            ks = np.flatnonzero(win & (TEAMS == team))
            if len(ks) < 2:
                continue
            ys = ep.pos[i, ks, 1] - b[i, 1]
            if not (ys.max() > 2.0 and ys.min() < -2.0):
                continue
            v = bv[i + 2]
            sp = float(np.hypot(*v))
            g = (goal[i:i + 3 * SAMPLES_PER_S] == (1 if team == 0 else -1)).any()
            rifles.append(dict(speed=sp, horizontal=abs(float(v[0])) / (sp + 1e-9),
                               forward=bool(v[0] * sign(team) > 0), goal=bool(g)))
            used[i:i + 2] = True
    # ---- patadas simples (referencia) y usar
    for i in range(T - 2):
        if not open_[i] or used[i]:
            continue
        ks = np.flatnonzero(ep.kicked[i])
        if len(ks) != 1:
            continue
        p = int(ks[0])
        team = TEAMS[p]
        v = bv[i + 1]
        sp = float(np.hypot(*v))
        simples.append(sp)
        if sp < 1.0:
            continue
        rivals = np.flatnonzero(TEAMS != team)
        rel = ep.pos[i, rivals] - b[i]
        dist = np.hypot(rel[:, 0], rel[:, 1])
        cos = (rel @ v) / (dist * sp + 1e-9)
        cand = rivals[(dist < 80.0) & (cos > 0.8)]
        if not len(cand):
            continue
        hit, j = -1, -1
        for k in range(i + 1, min(T, i + 1 + SAMPLES_PER_S // 2)):
            t = [q for q in cand if gap[k, q] < 4.0 or ep.kicked[k, q]]
            if t:
                hit, j = t[0], k
                break
        if hit < 0:
            continue
        rival_kick_favor = bool(ep.kicked[j, hit] and j + 1 < T and bv[j + 1, 0] * sign(team) > 0)
        res = "nada"
        for k in range(j + 1, min(T, j + 1 + 3 * SAMPLES_PER_S // 2)):
            if goal[k] == (1 if team == 0 else -1):
                res = "gol"
            elif goal[k] != 0:
                res = "gol_en_contra"
            elif not open_[k] and rteam[k] >= 0:
                res = "saque_a_favor" if rteam[k] == team else "saque_en_contra"
            elif owner[k] >= 0 and TEAMS[owner[k]] == team:
                res = "rebote_propio"
            elif owner[k] >= 0 and (owner[k] != hit or k > j + 2):
                res = "rival_controla"
            else:
                continue
            break
        usos.append(dict(result=res, rival_kick_favor=rival_kick_favor))
    # ---- cuerpear
    i = 0
    while i < T:
        s = int(np.argmin(gap[i]))
        shield = False
        if open_[i] and gap[i, s] <= 8.0:
            for r in np.flatnonzero(TEAMS != TEAMS[s]):
                cc = ep.pos[i, s] - ep.pos[i, r]
                if np.hypot(*cc) > 2 * PLAYER_R + 3.0 or d[i, r] <= d[i, s]:
                    continue
                sb = b[i] - ep.pos[i, s]
                if float(cc @ sb) / (np.hypot(*cc) * np.hypot(*sb) + 1e-9) > 0.3:
                    shield = True
                    break
        if not shield:
            i += 1
            continue
        start = i
        while i + 1 < T and open_[i + 1] and int(np.argmin(gap[i + 1])) == s and gap[i + 1, s] <= 8.0 and \
                np.any(np.hypot(*(ep.pos[i + 1, s] - ep.pos[i + 1, TEAMS != TEAMS[s]]).T) <= 2 * PLAYER_R + 3.0):
            i += 1
        res = "nada"
        for k in range(i + 1, min(T, i + 1 + 3 * SAMPLES_PER_S // 2)):
            o = owner[k]
            if not open_[k]:
                res = "saque_a_favor" if rteam[k] == TEAMS[s] else "saque_en_contra"
            elif o >= 0 and TEAMS[o] != TEAMS[s]:
                res = "rival"
            elif o >= 0 and o != s:
                res = "apoyo"
            elif ep.kicked[k, s]:
                res = "patea"
            else:
                continue
            break
        cuerpeos.append(dict(samples=i - start + 1, result=res))
        i += 1
    return dict(minutes=float(open_.sum()) / SAMPLES_PER_S / 60.0, rifles=rifles, usos=usos, cuerpeos=cuerpeos,
                simples=simples)


def summarize(dets):
    mins = sum(x["minutes"] for x in dets)
    rif = [r for x in dets for r in x["rifles"]]
    uso = [u for x in dets for u in x["usos"]]
    cue = [c for x in dets for c in x["cuerpeos"]]
    simp = [s for x in dets for s in x["simples"]]

    def frac(rows, key, val):
        return round(sum(r[key] == val for r in rows) / len(rows), 3) if rows else None

    out = dict(minutos_juego_abierto=round(mins, 1),
               velocidad_patada_simple_p50=round(float(np.median(simp)), 2) if simp else None)
    out["rifle"] = dict(por_min=round(len(rif) / max(mins, 1e-9), 3), n=len(rif))
    if rif:
        out["rifle"].update(velocidad_p50=round(float(np.median([r["speed"] for r in rif])), 2),
                            horizontal_p50=round(float(np.median([r["horizontal"] for r in rif])), 3),
                            hacia_arco_rival=frac(rif, "forward", True), gol_3s=frac(rif, "goal", True))
    out["usar"] = dict(por_min=round(len(uso) / max(mins, 1e-9), 3), n=len(uso))
    if uso:
        res = [u["result"] for u in uso]
        out["usar"]["resultados"] = {k: round(res.count(k) / len(res), 3) for k in sorted(set(res))}
        out["usar"]["favorable"] = round(sum(r in ("gol", "saque_a_favor", "rebote_propio") for r in res) / len(res), 3)
        out["usar"]["rival_patea_a_favor"] = frac(uso, "rival_kick_favor", True)
    out["cuerpear"] = dict(por_min=round(len(cue) / max(mins, 1e-9), 3), n=len(cue))
    if cue:
        res = [c["result"] for c in cue]
        out["cuerpear"]["duracion_s_p50"] = round(float(np.median([c["samples"] for c in cue])) / SAMPLES_PER_S, 2)
        out["cuerpear"]["resultados"] = {k: round(res.count(k) / len(res), 3) for k in sorted(set(res))}
        out["cuerpear"]["conserva"] = round(sum(r in ("apoyo", "patea", "saque_a_favor") for r in res) / len(res), 3)
    return out


def human_episodes(split="test", limit=80, map_name="sanguchito_rs_x4"):
    from tools.x4_ticks import MAP_IDS
    rows = json.loads((ROOT / "reports" / "x4" / "splits.json").read_text(encoding="utf-8"))["recordings"]
    eps = []
    n = 0
    for name, r in sorted(rows.items()):
        if r["split"] != split or map_name not in (r.get("maps") or {}):
            continue
        path = ROOT / "data" / "x4_ticks" / f"{Path(name).stem}.npz"
        if not path.exists():
            continue
        eps += XM.episodes_from_ticks(str(path), map_id=MAP_IDS[map_name])
        n += 1
        if n >= limit:
            break
    return eps


def selfplay_episodes(spec, matches, minutes, device, seed):
    from learn.x4_eval import Policy, play
    import torch
    torch.manual_seed(seed)
    pol = Policy(spec, device=device)
    return play(pol, pol, matches=matches, minutes=minutes, seed=seed, record=matches)["episodes"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agent", action="append", default=[], help="checkpoint (repetible) para self-play")
    ap.add_argument("--bc", default="runs/x4_bc/final_sangu_rsone/best.pt")
    ap.add_argument("--recordings", type=int, default=80)
    ap.add_argument("--matches", type=int, default=48)
    ap.add_argument("--minutes", type=float, default=3.0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    rep = {}
    rep["humanos"] = summarize([detect(ep) for ep in human_episodes(limit=a.recordings)])
    print("humanos", json.dumps(rep["humanos"], ensure_ascii=False), flush=True)
    for i, spec in enumerate([a.bc] + a.agent):
        name = "bc" if i == 0 else Path(spec).parent.name + "/" + Path(spec).stem
        rep[name] = summarize([detect(ep) for ep in selfplay_episodes(spec, a.matches, a.minutes, a.device, 11 + i)])
        rep[name]["spec"] = spec
        print(name, json.dumps(rep[name], ensure_ascii=False), flush=True)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
