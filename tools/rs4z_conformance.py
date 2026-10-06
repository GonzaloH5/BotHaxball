"""Conformidad del contrato RS4-Z-2 contra las grabaciones reales (motor original).

Amplía la auditoría RS4-b1 (`tools/rs4_rules_audit.py`), que excluía saques iniciales, goles y
tramos tocados por el script y medía sólo 1500 transiciones por grabación:

1. Física de 1 tick y de 60 ticks con el kernel RS4-Z en TODAS las transiciones en juego con
   plantel 4v4 estable y sin intervención del script en ese tick, separadas por fase de masa
   (F1: 0,5 tras cada reposicionamiento hasta la patada de un saque; 0,3 después). Compara la
   hipótesis vieja (0,3 siempre) contra la de fases.
2. Córners: distancia de los defensores al centro del disco de exclusión del mapa (RS ONE: (±1150, ±740),
   radio 445) mientras el córner está activo (F22: debe ser ≥ radio + 15).
3. Reloj congelado durante la espera del saque inicial (F2).

  python -m tools.rs4z_conformance --out reports/rs4z/conformance.json [--files 72] [--workers 12]
  python -m tools.rs4z_conformance --map sanguchito_rs_x4 --pattern 'SanguREC*' --out conformance_sangu.json

El punto 2 (córners de RS ONE) no aplica a otros mapas: sus saques se validan con `tools.rs4z_restart_conformance`.
"""
from __future__ import annotations

import argparse
import collections
import functools
import glob
import gzip
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
HORIZON = 60
MAX_ROLLOUTS_PER_FILE = 120
MAX_ONE_TICK_PER_PHASE = 15000


def _input_to_action(inp):
    from bridge.compare_sim import input_to_action
    return input_to_action(inp)


def _compact(item):
    """Tick grabado → registro chico (la grabación completa como dicts ocupa ~1,3 GB)."""
    discs = item["discs"]
    b = discs[0]
    ball = item.get("ball") or {}
    players = tuple((p["id"], p["team"], p["input"], p["kicking"], discs[p["disc"]]["x"], discs[p["disc"]]["y"],
                     discs[p["disc"]]["vx"], discs[p["disc"]]["vy"]) for p in item["players"])
    return dict(frame=item["frame"], state=item["state"], time=item["time"],
                ball=(b["x"], b["y"], b["vx"], b["vy"], b["r"], ball.get("im", 1.05), ball.get("gx", 0.0),
                      ball.get("gy", 0.0)), players=players)


def _read(path):
    ticks, events = [], collections.defaultdict(list)
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            item = json.loads(line)
            kind = item.get("type")
            if kind == "tick":
                ticks.append(_compact(item))
            elif kind == "event":
                events[item["frame"]].append(item)
    return ticks, events


def restart_frames(ticks, events):
    """Frames con un saque activo según el script: desde que colorea/coloca la pelota hasta que la
    vuelve a blanco. Esos tramos se auditan aparte (reglas), no como física libre."""
    active, out = False, set()
    ordered = sorted(events)
    j = 0
    for tick in ticks:
        fr = tick["frame"]
        while j < len(ordered) and ordered[j] <= fr:
            for e in events[ordered[j]]:
                if e["name"] == "disc_props" and not e.get("kind") and e.get("id") == 0:
                    d2 = e.get("data2") or [None] * 3
                    if d2[0] is not None:
                        active = d2[0] != 0xFFFFFF
                elif e["name"] in ("positions_reset", "game_start"):
                    active = False
            j += 1
        if active:
            out.add(fr)
    return out


def mass_phases(ticks, events):
    """Masa de jugadores por frame según el script, con el reposicionamiento del motor (F1).

    Devuelve dict frame -> 0.5 | 0.3 | None (None: saque con masa de pieza o mezcla).
    """
    current = 0.5
    piece = False
    out = {}
    ordered = sorted(events)
    j = 0
    for tick in ticks:
        fr = tick["frame"]
        while j < len(ordered) and ordered[j] <= fr:
            for e in events[ordered[j]]:
                if e["name"] in ("positions_reset", "game_start"):
                    current, piece = 0.5, False
                elif e["name"] == "disc_props" and e.get("kind"):
                    d1 = e.get("data1") or [None] * 10
                    if d1[8] is not None:
                        if d1[8] >= 1000:
                            piece = True
                        else:
                            current, piece = float(d1[8]), False
            j += 1
        out[fr] = None if piece else current
    return out


class Replayer:
    """Un partido RS4-Z de un solo entorno cargado desde ticks reales (registros compactos)."""

    def __init__(self, tick0, map_name="rs_one"):
        from env.rs4z.core import RS4ZEnv
        self.env = RS4ZEnv(1, contract="v2", frame_skip=1, deadline=0, kickoff_deadline=0, max_delay=0, map=map_name,
                           seed=0)  # sorteos (último toque desconocido) reproducibles
        self.order = [p[0] for p in sorted(tick0["players"], key=lambda p: (p[1], p[0]))]

    def by_slot(self, t):
        by_id = {p[0]: p for p in t["players"]}
        return [by_id[i] for i in self.order]

    def load(self, t, mass):
        env = self.env
        players = self.by_slot(t)
        pp = np.array([[p[4], p[5]] for p in players])
        pv = np.array([[p[6], p[7]] for p in players])
        bx, by, bvx, bvy, br, im, gx, gy = t["ball"]
        env.radius[0, 0] = br
        env.place(0, ball_pos=(bx, by), ball_vel=(bvx, bvy), player_pos=pp, player_vel=pv,
                  kick_held=[bool(p[2] & 16) and not p[3] for p in players])
        env.inv[0, env.fp:] = mass
        env.inv[0, 0] = im
        env.grav[0] = (gx, gy)
        env.ri[0, 8] = 0  # en juego (state 1)
        env.ri[0, 5] = 0  # sin cobrar salidas: los saques se auditan aparte

    def actions(self, t):
        acts = np.array([[_input_to_action(p[2]) for p in self.by_slot(t)]], dtype=np.int64)
        acts[0, 4:] = np.array([0, 1, 8, 7, 6, 5, 4, 3, 2] + [9, 10, 17, 16, 15, 14, 13, 12, 11])[acts[0, 4:]]
        return acts

    def real(self, t):
        players = self.by_slot(t)
        return np.array(t["ball"][:2]), np.array([[p[4], p[5]] for p in players])


def _stable(a, b):
    roster = lambda t: tuple(sorted((p[0], p[1]) for p in t["players"]))
    return (len(a["players"]) == 8 and b["frame"] == a["frame"] + 1 and roster(a) == roster(b)
            and a["state"] == 1 and b["state"] == 1)


def stadium_frames(path, map_name):
    """Tramos [desde, hasta) de la grabación jugados en el estadio del mapa.

    Las grabaciones mezclan variantes: Classic, entrenamiento o la tanda de penales al final de
    Sanguchito. Se reconocen por la cantidad de segmentos del .hbs exportado.
    """
    from pathlib import PureWindowsPath
    from env.rs4z import contract as C
    ref = len(json.loads((ROOT / "stadiums" / f"{C.MAPS[map_name]['stadium']}.hbs").read_text(encoding="utf-8"))["segments"])
    folder = Path(path).parent
    index = json.loads((folder.parent / "index.json").read_text(encoding="utf-8"))["recordings"]
    entry = next((v for v in index.values() if v.get("jsonl") and PureWindowsPath(v["jsonl"]).parts[0] == folder.name), None)
    if entry is None:
        return [(0, float("inf"))]
    stadiums = entry.get("stadiums") or []
    spans = []
    for i, st in enumerate(stadiums):
        end = stadiums[i + 1]["frame"] if i + 1 < len(stadiums) else float("inf")
        segs = len(json.loads((folder / st["file"]).read_text(encoding="utf-8"))["segments"])
        if segs == ref:
            spans.append((st["frame"], end))
    return spans


def in_spans(frame, spans):
    return any(a <= frame < b for a, b in spans)


def audit_file(path, map_name="rs_one"):
    from env.rs4z import contract as C
    prm = C.params(map_name)
    ticks, events = _read(path)
    spans = stadium_frames(path, map_name)
    ticks = [t for t in ticks if in_spans(t["frame"], spans)]
    phase = mass_phases(ticks, events)
    script = {f for f, evs in events.items() if any(e["name"] == "disc_props" for e in evs)}
    resets = {f for f, evs in events.items() if any(e["name"] in ("positions_reset", "goal") for e in evs)}
    script |= restart_frames(ticks, events)
    one = collections.defaultdict(list)       # (phase, hypothesis) -> [(ball_err, player_err)]
    roll = collections.defaultdict(list)
    rep = None
    rollouts = 0
    for i in range(len(ticks) - 1):
        a, b = ticks[i], ticks[i + 1]
        m = phase.get(a["frame"])
        if (not _stable(a, b) or m is None or phase.get(b["frame"]) != m or a["frame"] in script
                or b["frame"] in script or b["frame"] in resets):
            continue
        if rep is None or rep.order != [p[0] for p in sorted(a["players"], key=lambda p: (p[1], p[0]))]:
            rep = Replayer(a, map_name)
        hyps = (("fases", m), ("fija_0.3", 0.3)) if m == 0.5 else (("fases", m),)
        if len(one[(m, "fases")]) >= MAX_ONE_TICK_PER_PHASE:
            hyps_one = ()
        else:
            hyps_one = hyps
        for name, mass in hyps_one:
            rep.load(a, mass)
            rep.env.step(rep.actions(a))
            ball, players = rep.real(b)
            env = rep.env
            one[(m, name)].append((float(np.hypot(*(env.ball_pos[0] - ball))),
                                   float(np.hypot(*(env.player_pos[0] - players).T).max())))
        if rollouts < MAX_ROLLOUTS_PER_FILE and i % 53 == 0 and i + HORIZON < len(ticks):
            seq = ticks[i:i + HORIZON + 1]
            ok = all(_stable(seq[k], seq[k + 1]) and phase.get(seq[k]["frame"]) == m and seq[k + 1]["frame"] not in script
                     and seq[k + 1]["frame"] not in resets for k in range(HORIZON))
            if ok:
                for name, mass in hyps:
                    rep.load(a, mass)
                    for k in range(HORIZON):
                        rep.env.step(rep.actions(seq[k]))
                    ball, players = rep.real(seq[-1])
                    roll[(m, name)].append((float(np.hypot(*(rep.env.ball_pos[0] - ball))),
                                            float(np.hypot(*(rep.env.player_pos[0] - players).T).max())))
                rollouts += 1
    # F22: defensores durante córners activos (de la colocación a la patada)
    corner = []
    active = None
    for t in ticks:
        fr = t["frame"]
        for e in events.get(fr, []):
            if e["name"] == "disc_props" and not e.get("kind") and e.get("id") == 0:
                d1 = e.get("data1") or [None] * 10
                if (d1[0] is not None and d1[1] is not None and abs(abs(d1[0]) - prm[C.PI["corner_x"]]) < 1
                        and abs(abs(d1[1]) - prm[C.PI["corner_y"]]) < 1):
                    active = (np.sign(d1[0]), np.sign(d1[1]), fr)
            if e["name"] in ("kick", "positions_reset"):
                active = None
        if active is not None and fr > active[2] + 2:
            sx, sy, _ = active
            defender = 1 if sx < 0 else 2  # equipo HaxBall del arco de ese lado (1 rojo, 2 azul)
            c = np.array([sx * prm[C.PI["corner_disc_x"]], sy * prm[C.PI["corner_disc_y"]]])
            ds = [np.hypot(p[4] - c[0], p[5] - c[1]) for p in t["players"] if p[1] == defender]
            if ds:
                corner.append(min(ds))
    # F2: reloj durante la espera del saque inicial
    frozen = moving = 0
    for a, b in zip(ticks, ticks[1:]):
        if a["state"] == 0 and b["state"] == 0 and b["frame"] == a["frame"] + 1:
            if abs(a["time"] - b["time"]) < 1e-9:
                frozen += 1
            else:
                moving += 1
    return dict(file=Path(path).name, one={f"{k[0]}|{k[1]}": v for k, v in one.items()},
                roll={f"{k[0]}|{k[1]}": v for k, v in roll.items()}, corner=corner, clock=(frozen, moving))


def _summary(values):
    if not values:
        return dict(n=0)
    arr = np.asarray(values)
    return dict(n=len(arr), ball_p50=float(np.percentile(arr[:, 0], 50)), ball_p90=float(np.percentile(arr[:, 0], 90)),
                ball_p99=float(np.percentile(arr[:, 0], 99)), players_p50=float(np.percentile(arr[:, 1], 50)),
                players_p90=float(np.percentile(arr[:, 1], 90)), players_p99=float(np.percentile(arr[:, 1], 99)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/rs4z/conformance.json")
    ap.add_argument("--files", type=int, default=0, help="0 = todas")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--map", default="rs_one", help="mapa del contrato (env/rs4z/contract.MAPS)")
    ap.add_argument("--pattern", default="*", help="carpetas del caché, p. ej. 'SanguREC*'")
    ap.add_argument("--data", default="data/rs4_jsonl", help="caché JSONL (data/haxarg_jsonl para 2K23)")
    args = ap.parse_args()
    from env.rs4z import contract as C
    files = sorted(glob.glob(str(ROOT / args.data / args.pattern / "*.jsonl.gz")))
    if args.files:
        files = files[:args.files]
    one, roll, corner = collections.defaultdict(list), collections.defaultdict(list), []
    frozen = moving = 0
    with Pool(args.workers) as pool:
        for r in pool.imap_unordered(functools.partial(audit_file, map_name=args.map), files):
            for k, v in r["one"].items():
                one[k] += v
            for k, v in r["roll"].items():
                roll[k] += v
            corner += r["corner"]
            frozen += r["clock"][0]
            moving += r["clock"][1]
    corner = np.asarray(corner)
    report = dict(
        version="RS4-Z-2-conformance-1", map=args.map, pattern=args.pattern, files=len(files),
        one_tick={k: _summary(v) for k, v in sorted(one.items())},
        sixty_ticks={k: _summary(v) for k, v in sorted(roll.items())},
        corner_defender_min_distance=dict(n=int(len(corner)), p0=float(corner.min()) if len(corner) else None,
                                          p1=float(np.percentile(corner, 1)) if len(corner) else None,
                                          disc_radius=float(C.params(args.map)[C.PI["corner_disc_radius"]]),
                                          share_inside=float((corner < C.params(args.map)[C.PI["corner_disc_radius"]] + 14.0).mean())
                                          if len(corner) else None,
                                          share_below_v1_rule=None),
        kickoff_clock=dict(frozen=frozen, moving=moving),
    )
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
