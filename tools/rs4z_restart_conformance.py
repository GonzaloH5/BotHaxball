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
AFTER = 60       # ticks comparados después de la liberación
WHITE = 0xFFFFFF
KIND_NAMES = {1: "lateral", 2: "corner", 3: "goal_kick"}
MIRROR = np.array([0, 1, 8, 7, 6, 5, 4, 3, 2] + [9, 10, 17, 16, 15, 14, 13, 12, 11])


def real_restarts(ticks, events):
    """Saques del script: inicio (pelota coloreada y colocada), liberación (pelota blanca), reasignación.

    El script puede mandar posición y color de la pelota en eventos separados del mismo frame: se unen.
    El color identifica al ejecutor sólo en Sanguchito (rojo/azul); RS ONE y 2K23 usan un color fijo
    (ver `infer_taker`).
    """
    out, cur = [], None
    for fr in sorted(events):
        ball = {}
        for e in events[fr]:
            if e["name"] in ("positions_reset", "game_stop") and cur is not None:
                cur["end"] = ("reset", fr)
                out.append(cur)
                cur = None
            if e["name"] == "disc_props" and not e.get("kind") and e.get("id") == 0:
                d1 = e.get("data1") or [None] * 10
                col = (e.get("data2") or [None])[0]
                if d1[0] is not None:
                    ball["x"], ball["y"] = d1[0], d1[1]
                if col is not None:
                    ball["col"] = col
        col = ball.get("col")
        if "x" in ball and col is not None and col != WHITE:
            if cur is not None:
                cur["end"] = ("reassigned", fr)
                out.append(cur)
            cur = dict(start=fr, spot=(ball["x"], ball["y"]), col=col, end=None)
        elif col == WHITE and cur is not None:
            cur["end"] = ("released", fr)
            out.append(cur)
            cur = None
    return out


def infer_taker(r, kind, events, team_of):
    """Equipo ejecutor (0 rojo, 1 azul): color en Sanguchito; si no, geometría (córner y saque de arco) o
    rivales con la barrera c1 del lateral (C1 en su cGroup)."""
    if r["col"] in (0xFF0000, 0x0000FF):
        return 0 if r["col"] == 0xFF0000 else 1
    left = r["spot"][0] < 0          # arco del rojo
    if kind == 2:
        return 1 if left else 0      # córner: ataca el dueño del otro arco
    if kind == 3:
        return 0 if left else 1      # saque de arco: el que defiende
    for fr in range(r["start"], r["start"] + 3):
        for e in events.get(fr, []):
            if e["name"] == "disc_props" and e.get("kind"):
                cg = (e.get("data2") or [None, None, None])[2]
                if cg is not None and cg & (1 << 29) and e["id"] in team_of:
                    return 1 - (team_of[e["id"]] - 1)
    return None


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


def _load(env, tick, order, phase_mass, last_touch):
    bx, by, bvx, bvy, br, im, gx, gy = tick["ball"]
    env.radius[0, 0] = br
    by_id = {p[0]: p for p in tick["players"]}
    pl = [by_id[pid] for pid in order]
    env.place(0, ball_pos=(bx, by), ball_vel=(bvx, bvy), player_pos=np.array([[p[4], p[5]] for p in pl]),
              player_vel=np.array([[p[6], p[7]] for p in pl]), kick_held=[bool(p[2] & 16) and not p[3] for p in pl],
              last_touch=last_touch, mass_phase=0 if phase_mass == 0.5 else 1)
    env.inv[0, 0] = im
    env.grav[0] = (gx, gy)


def _window_ok(ticks_by_frame, frames, order):
    return all(f in ticks_by_frame and len(ticks_by_frame[f]["players"]) == 8 and _order(ticks_by_frame[f]) == order
               for f in frames)


def out_tick(ticks_by_frame, start, env):
    """Primer tick de la salida que terminó en el saque `start` (la sala puede tardar en cobrarlo)."""
    rb = env.radius[0, 0]
    lw, lh = env.line_w, env.prm[env_pi("line_half_h")]
    f = start
    while f - 1 in ticks_by_frame:
        x, y = ticks_by_frame[f - 1]["ball"][:2]
        if abs(x) <= lw + rb and abs(y) <= lh + rb:
            break
        f -= 1
    return f


def detect_restart(env, ticks_by_frame, r, phase, order, out):
    """Desde el juego abierto antes de la salida: ¿el simulador cobra el mismo saque?"""
    from env.rs4z import kernel as K
    t0 = out - 3
    horizon = out + 3 + int(max(env.prm[env_pi(k)] for k in ("pend_lat_max", "pend_corner_max", "pend_gk_max")))
    horizon = min(horizon, r["start"] + 3) if horizon > r["start"] else horizon
    frames = range(t0, horizon + 1)
    if not _window_ok(ticks_by_frame, frames, order) or ticks_by_frame[t0]["state"] != 1 or phase.get(t0) is None:
        return None
    _load(env, ticks_by_frame[t0], order, phase[t0], 1 - r["taker"])
    for f in range(t0, horizon):
        _force(env, ticks_by_frame[f], order)
        ev = env.step(_actions(ticks_by_frame[f], order))
        if ev["restart_start"][0] != 0:
            spot = (float(env.rf[0, K.RF_SPOT_X]), float(env.rf[0, K.RF_SPOT_Y]))
            return dict(sim_start_off=f + 1 - out, sim_kind=int(env.ri[0, K.RI_KIND]),  # = demora simulada
                        taker_ok=int(env.ri[0, K.RI_TEAM]) == r["taker"],
                        spot_err=float(np.hypot(spot[0] - r["spot"][0], spot[1] - r["spot"][1])))
    return dict(sim_start_off=None, sim_kind=None, taker_ok=False, spot_err=None)


def execute_restart(env, ticks_by_frame, r, phase, order, kind, next_start=None):
    """Desde la colocación real de la pelota: empujes, liberación y trayectoria de la pelota."""
    from env.rs4z import kernel as K
    s = r["start"]
    end_kind, end_fr = r["end"]
    t_end = end_fr + (AFTER if end_kind == "released" else 2)
    if next_start is not None:
        t_end = min(t_end, next_start - 1)   # el saque siguiente teletransporta la pelota
    if not _window_ok(ticks_by_frame, range(s, t_end + 1), order):
        return None
    # en córner y saque de arco la masa del script puede ser la de pieza (None); start_restart la fija
    _load(env, ticks_by_frame[s], order, phase.get(s) or 0.3, 1 - r["taker"])
    env.start_restart(0, kind, r["taker"], r["spot"])
    # la sala aplica los empujes en el tick de la colocación o en el siguiente: se toma el más cercano
    push_err = min(max(float(np.hypot(env.player_pos[0, k, 0] - rp[pid][4], env.player_pos[0, k, 1] - rp[pid][5]))
                       for k, pid in enumerate(order))
                   for rp in ({p[0]: p for p in ticks_by_frame[g]["players"]} for g in (s + 1, s + 2)))
    sim_end = sim_end_kind = None
    ball_err = {}
    for f in range(s + 1, t_end):
        _force(env, ticks_by_frame[f], order)
        team_before = env.ri[0, K.RI_TEAM]
        ev = env.step(_actions(ticks_by_frame[f], order))
        if sim_end is None:
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
    return dict(push_err=push_err, sim_end=sim_end_kind, end_off=None if sim_end is None else sim_end - end_fr,
                ball_err=ball_err)


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
        t_s = by_frame.get(r["start"])
        if t_s is None or len(t_s["players"]) != 8:
            continue
        order = _order(t_s)
        kind = (1 if abs(abs(r["spot"][1]) - env.prm[env_pi("lateral_ball_y")]) < 1.0 else
                3 if abs(abs(r["spot"][0]) - env.prm[env_pi("goal_kick_x")]) < 1.0 else 2)
        r["taker"] = infer_taker(r, kind, events, {p[0]: p[1] for p in t_s["players"]})
        if r["taker"] is None:
            continue
        out = out_tick(by_frame, r["start"], env)
        row = dict(start=r["start"], real_kind=kind, real_end=r["end"][0], real_dur=r["end"][1] - r["start"],
                   real_delay=r["start"] - out, file=Path(path).name)
        det = detect_restart(env, by_frame, r, phase, order, out)
        nxt = next((s for s in starts if s > r["start"]), None)
        exe = execute_restart(env, by_frame, r, phase, order, kind, nxt)
        if det is None and exe is None:
            continue
        row.update(det or dict(sim_start_off=None, sim_kind=None, taker_ok=None, spot_err=None, skipped_detect=True))
        row.update(exe or dict(push_err=None, sim_end=None, end_off=None, ball_err={}, skipped_execute=True))
        rows.append(row)
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
    ap.add_argument("--data", default="data/rs4_jsonl", help="caché JSONL (data/haxarg_jsonl para 2K23)")
    args = ap.parse_args()
    files = sorted(glob.glob(str(ROOT / args.data / args.pattern / "*.jsonl.gz")))
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
        executed = [r for r in rs if not r.get("skipped_execute")]
        ended = [r for r in executed if r["end_off"] is not None]
        summary[kind] = dict(
            n=len(rs), detected=len(started), executed=len(executed),
            room_delay=dict(collections.Counter(r["real_delay"] for r in rs)),
            sim_delay=dict(collections.Counter(r["sim_start_off"] for r in started)),
            kind_ok=sum(r["sim_kind"] == r["real_kind"] for r in started),
            taker_ok=sum(r["taker_ok"] for r in started),
            spot_err=_pct([r["spot_err"] for r in started]),
            push_err=_pct([r["push_err"] for r in executed]),
            end_match=sum(r["sim_end"] == r["real_end"] for r in executed),
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
