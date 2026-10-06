"""Conformidad de los saques del árbitro RS4-Z contra las grabaciones, con los jugadores forzados.

Para cada saque grabado (el script colorea la pelota y la coloca) se carga el estado real unos ticks antes
de la salida y se simula tick a tick con las entradas grabadas. Antes de cada tick los jugadores se
fuerzan a su estado grabado, así que el error medido es sólo del árbitro y de la pelota: tipo, punto y
ejecutor del saque, empujes de rivales, tick de liberación, lateral cedido al rival y trayectoria de la
pelota después de la patada (velocidad y curva del script). Además compara las posiciones del saque
inicial del mapa con las grabadas después de cada reposicionamiento.

Alineación: el estado tras simular el tick f equivale al tick grabado f + 1. Con esa convención, el inicio
de cada saque y la liberación de los laterales salen con desfase 0. Las liberaciones de córner y saque de
arco salen con +1, porque el evento de patada se graba un frame antes. Lo que se juzga es la trayectoria
de la pelota (`ball_err_after_release`).

  python -m tools.rs4z_restart_conformance --map sanguchito_rs_x4 --pattern 'SanguREC*' --out restart_sangu.json
"""
from __future__ import annotations

import argparse
import collections
import functools
import glob
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from tools.rs4z_conformance import _input_to_action, _read, in_spans, mass_phases, stadium_frames

ROOT = Path(__file__).resolve().parent.parent
PRE = 5          # ticks de juego antes de la salida
AFTER = 60       # ticks comparados después de la liberación
WHITE = 0xFFFFFF
KIND_NAMES = {1: "lateral", 2: "corner", 3: "goal_kick"}
MIRROR = np.array([0, 1, 8, 7, 6, 5, 4, 3, 2] + [9, 10, 17, 16, 15, 14, 13, 12, 11])


def real_restarts(ticks, events):
    """Saques del script: inicio (pelota coloreada y colocada), liberación (pelota blanca), reasignación."""
    out, cur = [], None
    for fr in sorted(events):
        for e in events[fr]:
            if e["name"] in ("positions_reset", "game_stop") and cur is not None:
                cur["end"] = ("reset", fr)
                out.append(cur)
                cur = None
            if e["name"] != "disc_props" or e.get("kind") or e.get("id") != 0:
                continue
            d1 = e.get("data1") or [None] * 10
            col = (e.get("data2") or [None])[0]
            if d1[0] is not None and col is not None and col != WHITE:
                if cur is not None:
                    cur["end"] = ("reassigned", fr)
                    out.append(cur)
                cur = dict(start=fr, spot=(d1[0], d1[1]), taker=0 if col == 0xFF0000 else 1, end=None)
            elif col == WHITE and cur is not None:
                cur["end"] = ("released", fr)
                out.append(cur)
                cur = None
    return out


def _order(tick):
    return [p[0] for p in sorted(tick["players"], key=lambda p: (p[1], p[0]))]


def _force(env, tick, order):
    fp = env.fp
    by_id = {p[0]: p for p in tick["players"]}
    for s, pid in enumerate(order):
        p = by_id[pid]
        env.pos[0, fp + s] = (p[4], p[5])
        env.vel[0, fp + s] = (p[6], p[7])
        env.kick_cancel[0, s] = bool(p[2] & 16) and not p[3]


def _actions(tick, order):
    by_id = {p[0]: p for p in tick["players"]}
    acts = np.array([[_input_to_action(by_id[pid][2]) for pid in order]], dtype=np.int64)
    acts[0, 4:] = MIRROR[acts[0, 4:]]
    return acts


def replay_restart(env, ticks_by_frame, r, phase, order, next_start=None):
    """Simula un saque con los jugadores forzados. Devuelve métricas o None si la ventana no sirve."""
    from env.rs4z import kernel as K
    end_kind, end_fr = r["end"]
    t0 = r["start"] - PRE
    t_end = end_fr + (AFTER if end_kind == "released" else 2)
    if next_start is not None:
        t_end = min(t_end, next_start - 1)   # el saque siguiente teletransporta la pelota
    frames = range(t0, t_end + 1)
    if any(f not in ticks_by_frame for f in frames):
        return None
    if any(len(ticks_by_frame[f]["players"]) != 8 or _order(ticks_by_frame[f]) != order for f in frames):
        return None
    a = ticks_by_frame[t0]
    if a["state"] != 1 or phase.get(t0) is None:
        return None
    bx, by, bvx, bvy, br, im, gx, gy = a["ball"]
    env.radius[0, 0] = br
    by_id = {p[0]: p for p in a["players"]}
    pl = [by_id[pid] for pid in order]
    env.place(0, ball_pos=(bx, by), ball_vel=(bvx, bvy), player_pos=np.array([[p[4], p[5]] for p in pl]),
              player_vel=np.array([[p[6], p[7]] for p in pl]), kick_held=[bool(p[2] & 16) and not p[3] for p in pl],
              last_touch=1 - r["taker"], mass_phase=0 if phase[t0] == 0.5 else 1)
    env.inv[0, 0] = im
    env.grav[0] = (gx, gy)
    sim_start = sim_end = None
    sim_kind = sim_taker = None
    sim_spot = None
    sim_end_kind = None
    push_err = None
    ball_err = {}
    just_started = False
    for f in range(t0, t_end):
        # el tick grabado f no incluye los empujes que el script aplica a continuación: tras iniciar un
        # saque, los jugadores siguen con el estado simulado (empujados) un tick antes de volver a forzarlos
        if not just_started:
            _force(env, ticks_by_frame[f], order)
        just_started = False
        team_before = env.ri[0, K.RI_TEAM]
        ev = env.step(_actions(ticks_by_frame[f], order))
        # el estado tras el tick f equivale al tick grabado f + 1
        if ev["restart_start"][0] != 0:
            just_started = True
        if sim_start is None and ev["restart_start"][0] != 0:
            sim_start = f + 1
            sim_kind = int(env.ri[0, K.RI_KIND])
            sim_taker = int(env.ri[0, K.RI_TEAM])
            sim_spot = (float(env.rf[0, K.RF_SPOT_X]), float(env.rf[0, K.RF_SPOT_Y]))
            # empujes del script: contra el tick grabado siguiente (o el otro, según la alineación)
            errs = []
            for g in (f + 1, f + 2):
                rp = {p[0]: p for p in ticks_by_frame[g]["players"]}
                errs.append(max(float(np.hypot(env.player_pos[0, s, 0] - rp[pid][4], env.player_pos[0, s, 1] - rp[pid][5]))
                                for s, pid in enumerate(order)))
            push_err = min(errs)
        elif sim_start is not None and sim_end is None:
            if ev["forfeit"][0] >= 0:
                sim_end, sim_end_kind = f + 1, "reassigned"
            elif team_before >= 0 and env.ri[0, K.RI_TEAM] < 0:
                sim_end, sim_end_kind = f + 1, "released"
        if sim_end is not None and end_kind == "released" and f + 1 > end_fr:
            k = f + 1 - end_fr
            if k in (1, 5, 10, 30, 59):
                rb = ticks_by_frame[f + 1]["ball"]
                ball_err[k] = float(np.hypot(env.ball_pos[0, 0] - rb[0], env.ball_pos[0, 1] - rb[1]))
        if sim_end is not None and end_kind != "released":
            break
    return dict(start=r["start"], real_kind=None, real_end=end_kind, real_dur=end_fr - r["start"],
                sim_start_off=None if sim_start is None else sim_start - r["start"],
                sim_kind=sim_kind, taker_ok=sim_taker == r["taker"],
                spot_err=None if sim_spot is None else float(np.hypot(sim_spot[0] - r["spot"][0], sim_spot[1] - r["spot"][1])),
                push_err=push_err, sim_end=sim_end_kind,
                end_off=None if sim_end is None else sim_end - end_fr, ball_err=ball_err)


def formation_check(ticks_by_frame, events, env, spans):
    """Posiciones tras cada reposicionamiento grabado contra las del saque inicial del simulador."""
    from env.rs4z import kernel as K
    errs = []
    env.start_match([0])
    sim = {0: env.player_pos[0, :4].copy(), 1: env.player_pos[0, 4:].copy()}
    for fr, evs in events.items():
        if not any(e["name"] in ("positions_reset", "game_start") for e in evs) or not in_spans(fr + 1, spans):
            continue
        t = ticks_by_frame.get(fr + 1)
        if t is None:
            continue
        for team in (0, 1):
            real = np.array([[p[4], p[5]] for p in t["players"] if p[1] == team + 1])
            if len(real) != 4:
                continue
            d = np.hypot(real[:, None, 0] - sim[team][None, :, 0], real[:, None, 1] - sim[team][None, :, 1])
            errs.append(float(d.min(axis=1).max()))
    return errs


def audit_file(path, map_name):
    from env.rs4z.core import RS4ZEnv
    ticks, events = _read(path)
    spans = stadium_frames(path, map_name)
    ticks = [t for t in ticks if in_spans(t["frame"], spans)]
    by_frame = {t["frame"]: t for t in ticks}
    phase = mass_phases(ticks, events)
    env = RS4ZEnv(1, contract="v2", frame_skip=1, deadline=0, kickoff_deadline=0, max_delay=0, map=map_name, seed=0)
    rows = []
    restarts = real_restarts(ticks, events)
    reassigned = {r["end"][1] for r in restarts if r["end"] and r["end"][0] == "reassigned"}
    starts = sorted(r["start"] for r in restarts)
    for r in restarts:
        if r["end"] is None or r["start"] in reassigned:
            continue
        t0 = by_frame.get(r["start"] - PRE)
        if t0 is None or len(t0["players"]) != 8:
            continue
        nxt = next((s for s in starts if s > r["start"]), None)
        res = replay_restart(env, by_frame, r, phase, _order(t0), nxt)
        if res is not None:
            res["real_kind"] = (1 if abs(abs(r["spot"][1]) - env.prm[env_pi("lateral_ball_y")]) < 1.0 else
                                3 if abs(abs(r["spot"][0]) - env.prm[env_pi("goal_kick_x")]) < 1.0 else 2)
            res["file"] = Path(path).name
            rows.append(res)
    return dict(rows=rows, formation=formation_check(by_frame, events, env, spans))


def env_pi(name):
    from env.rs4z import contract as C
    return C.PI[name]


def _pct(values, qs=(50, 90, 100)):
    if not values:
        return None
    return {f"p{q}": float(np.percentile(values, q)) for q in qs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default="rs_one")
    ap.add_argument("--pattern", default="*")
    ap.add_argument("--out", default="reports/rs4z/restart_conformance.json")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    files = sorted(glob.glob(str(ROOT / "data/rs4_jsonl" / args.pattern / "*.jsonl.gz")))
    rows, formation = [], []
    with Pool(args.workers) as pool:
        for r in pool.imap_unordered(functools.partial(audit_file, map_name=args.map), files):
            rows += r["rows"]
            formation += r["formation"]
    by_kind = collections.defaultdict(list)
    for r in rows:
        by_kind[KIND_NAMES[r["real_kind"]]].append(r)
    summary = {}
    for kind, rs in sorted(by_kind.items()):
        started = [r for r in rs if r["sim_start_off"] is not None]
        ended = [r for r in started if r["end_off"] is not None]
        summary[kind] = dict(
            n=len(rs), detected=len(started),
            start_offset=dict(collections.Counter(r["sim_start_off"] for r in started)),
            kind_ok=sum(r["sim_kind"] == r["real_kind"] for r in started),
            taker_ok=sum(r["taker_ok"] for r in started),
            spot_err=_pct([r["spot_err"] for r in started]),
            push_err=_pct([r["push_err"] for r in started]),
            end_match=sum(r["sim_end"] == r["real_end"] for r in started),
            end_offset=dict(collections.Counter(r["end_off"] for r in ended)),
            ball_err_after_release={k: _pct([r["ball_err"][k] for r in ended if k in r["ball_err"]])
                                    for k in (1, 5, 10, 30, 59)},
        )
    report = dict(version="RS4-Z-restart-conformance-1", map=args.map, pattern=args.pattern, files=len(files),
                  restarts=summary, kickoff_formation_err=_pct(formation),
                  worst=sorted(rows, key=lambda r: -(r["ball_err"].get(30, 0.0) if r["ball_err"] else 0.0))[:10])
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "worst"}, indent=1, default=str))


if __name__ == "__main__":
    main()
