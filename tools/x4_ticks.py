"""Caché compacto por grabación: arrays numpy tick a tick de los tramos jugados en un mapa conocido.

El JSONL de `tools.rs4_jsonl_cache` guarda todos los discos de cada tick (~4 MB comprimido por partido de
Sanguchito, ~25 MB uno largo de RS ONE) y parsearlo cuesta segundos. Este caché guarda sólo lo que usan
las métricas de parecido humano, el dataset de imitación y los arranques desde estados humanos:

* por tick: frame, state (0 saque inicial, 1 en juego), equipo y edad del saque inicial, reloj, marcador, pelota (x, y, vx, vy, r, invMass,
  gx, gy), mapa del tramo, masa de los jugadores según el script (0,5 / 0,3 / -1 pieza), saque activo
  (tipo, equipo ejecutor, edad) y cobro pendiente (salida ya ocurrida, pelota todavía no colocada);
* por tick y lugar (0..3 rojos, 4..7 azules, ordenados por id): id del jugador (-1 vacío), posición,
  velocidad, entrada (bits de HaxBall) y `kicking`; `n_red`, `n_blue` cuentan a todos (pueden pasar de 4);
* eventos: patadas (frame, id), goles (frame, equipo), reposicionamientos y saques del script
  (inicio, fin, tipo, ejecutor, punto, demora de colocación).

Mapas reconocidos por nombre del estadio y cantidad de segmentos del .hbs exportado (las variantes como la
tanda de penales de Sanguchito o Classic quedan fuera): ver `MAP_NAMES`.

  python -m tools.x4_ticks --cache data/rs4_jsonl --out data/x4_ticks --workers 4
"""
from __future__ import annotations

import argparse
import gzip
import json
import time
from pathlib import Path, PureWindowsPath

import numpy as np

try:
    import orjson
    _loads = orjson.loads
except ImportError:  # pragma: no cover
    _loads = json.loads

ROOT = Path(__file__).resolve().parent.parent
VERSION = "x4-ticks-1"
MAP_NAMES = {"SANGUCHITO RS X4": "sanguchito_rs_x4", "Real Soccer ONE": "rs_one", "HAXARG 2K23": "haxarg_2k23"}
MAP_IDS = {"rs_one": 0, "sanguchito_rs_x4": 1, "haxarg_2k23": 2}
WHITE = 0xFFFFFF
LATERAL, CORNER, GOAL_KICK = 1, 2, 3


def _ref_segments():
    from env.rs4z import contract as C
    out = {}
    for name in MAP_IDS:
        hbs = ROOT / "stadiums" / f"{C.MAPS[name]['stadium']}.hbs"
        out[name] = len(json.loads(hbs.read_text(encoding="utf-8"))["segments"])
    return out


def map_spans(folder, stadiums):
    """[(desde, hasta, mapa)] de los tramos jugados en un mapa conocido (frames de la grabación)."""
    ref = _ref_segments()
    spans = []
    for i, st in enumerate(stadiums or []):
        end = stadiums[i + 1]["frame"] if i + 1 < len(stadiums) else 1 << 40
        name = MAP_NAMES.get(st.get("name"))
        if name is None or not st.get("file"):
            continue
        try:
            segs = len(json.loads((folder / st["file"]).read_text(encoding="utf-8"))["segments"])
        except Exception:
            continue
        if segs == ref[name]:
            spans.append((int(st["frame"]), int(end), name))
    return spans


def _kick_strength(folder, stadiums):
    out = {}
    for st in stadiums or []:
        try:
            data = json.loads((folder / st["file"]).read_text(encoding="utf-8"))
            out[int(st["frame"])] = float((data.get("playerPhysics") or {}).get("kickStrength"))
        except Exception:
            pass
    return out


def _restarts(events, frames_set, prm):
    """Saques del script a partir de los eventos de la pelota (color y posición), como
    `tools.rs4z_restart_conformance.real_restarts`, más tipo y ejecutor."""
    from env.rs4z import contract as C
    out, cur = [], None
    for fr in sorted(events):
        ball = {}
        for e in events[fr]:
            if e["name"] in ("positions_reset", "game_stop") and cur is not None:
                cur["end"], cur["end_kind"] = fr, "reset"
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
                cur["end"], cur["end_kind"] = fr, "reassigned"
                out.append(cur)
            cur = dict(start=fr, x=float(ball["x"]), y=float(ball["y"]), col=int(col), end=None, end_kind=None)
        elif col == WHITE and cur is not None:
            cur["end"], cur["end_kind"] = fr, "released"
            out.append(cur)
            cur = None
    lat_y = prm[C.PI["lateral_ball_y"]]
    gk_x = prm[C.PI["goal_kick_x"]]
    for r in out:
        r["kind"] = (LATERAL if abs(abs(r["y"]) - lat_y) < 1.0 else GOAL_KICK if abs(abs(r["x"]) - gk_x) < 1.0
                     else CORNER)
    return out


def _taker(r, events, team_of):
    """0 rojo, 1 azul, -1 desconocido (`rs4z_restart_conformance.infer_taker`)."""
    if r["col"] in (0xFF0000, 0x0000FF):
        return 0 if r["col"] == 0xFF0000 else 1
    left = r["x"] < 0
    if r["kind"] == CORNER:
        return 1 if left else 0
    if r["kind"] == GOAL_KICK:
        return 0 if left else 1
    for fr in range(r["start"], r["start"] + 3):
        for e in events.get(fr, []):
            if e["name"] == "disc_props" and e.get("kind"):
                cg = (e.get("data2") or [None, None, None])[2]
                if cg is not None and cg & (1 << 29) and e["id"] in team_of:
                    return 1 - (team_of[e["id"]] - 1)
    return -1


def build(jsonl_path, stadiums):
    """Arrays del caché para una grabación (None si no tiene tramos en mapas conocidos)."""
    from env.rs4z import contract as C
    folder = Path(jsonl_path).parent
    spans = map_spans(folder, stadiums)
    if not spans:
        return None
    kstr = _kick_strength(folder, stadiums)

    def span_of(f):
        for a, b, name in spans:
            if a <= f < b:
                return name
        return None

    rows = []
    events = {}
    with gzip.open(jsonl_path, "rb") as fh:
        for line in fh:
            if line.startswith(b'{"type":"tick"'):
                f = int(line[23:line.index(b",", 23)]) if line[15:23] == b'"frame":' else None
                if f is not None and span_of(f) is None:
                    continue
                item = _loads(line)
                if span_of(item["frame"]) is None:
                    continue
                rows.append(item)
            else:
                item = _loads(line)
                if item.get("type") == "event":
                    events.setdefault(item["frame"], []).append(item)
    if not rows:
        return None
    T = len(rows)
    frame = np.empty(T, np.int32)
    state = np.empty(T, np.int8)
    clock = np.empty(T, np.float32)
    score = np.empty((T, 2), np.int16)
    ball = np.empty((T, 8), np.float32)
    map_id = np.empty(T, np.int8)
    pid = np.full((T, 8), -1, np.int16)
    pos = np.zeros((T, 8, 2), np.float32)
    vel = np.zeros((T, 8, 2), np.float32)
    inp = np.zeros((T, 8), np.uint8)
    kicking = np.zeros((T, 8), np.bool_)
    n_red = np.zeros(T, np.int8)
    n_blue = np.zeros(T, np.int8)
    ks = np.zeros(T, np.float32)
    ko_team = np.full(T, -1, np.int8)
    ko_age = np.zeros(T, np.int32)
    ks_frames = sorted(kstr)
    for i, t in enumerate(rows):
        f = t["frame"]
        frame[i] = f
        state[i] = t["state"]
        clock[i] = t["time"]
        score[i] = t["score"]
        if t["state"] == 0:
            ko_team[i] = (t.get("ko") or 1) - 1
            ko_age[i] = ko_age[i - 1] + 1 if i > 0 and state[i - 1] == 0 and frame[i - 1] == f - 1 else 0
        d = t["discs"]
        b = d[0]
        bp = t.get("ball") or {}
        ball[i] = (b["x"], b["y"], b["vx"], b["vy"], b["r"], bp.get("im", 1.0), bp.get("gx", 0.0), bp.get("gy", 0.0))
        map_id[i] = MAP_IDS[span_of(f)]
        k = None
        for kf in ks_frames:
            if kf <= f:
                k = kstr[kf]
        ks[i] = k if k is not None else np.nan
        reds = sorted((p for p in t["players"] if p["team"] == 1), key=lambda p: p["id"])
        blues = sorted((p for p in t["players"] if p["team"] == 2), key=lambda p: p["id"])
        n_red[i], n_blue[i] = len(reds), len(blues)
        for base, team in ((0, reds), (4, blues)):
            for s, p in enumerate(team[:4]):
                q = d[p["disc"]]
                pid[i, base + s] = p["id"]
                pos[i, base + s] = (q["x"], q["y"])
                vel[i, base + s] = (q["vx"], q["vy"])
                inp[i, base + s] = p["input"]
                kicking[i, base + s] = p["kicking"]
    # eventos
    kicks, goals, resets = [], [], []
    for f, evs in events.items():
        for e in evs:
            if e["name"] == "kick":
                kicks.append((f, e.get("playerId", -1)))
            elif e["name"] == "goal":
                goals.append((f, e.get("team", 0)))
            elif e["name"] in ("positions_reset", "game_start"):
                resets.append(f)
    kicks.sort()
    goals.sort()
    resets.sort()
    # masa del script por tick (0,5 tras reposicionar; lo que fije el script; -1 masa de pieza)
    mass = np.empty(T, np.float32)
    cur, piece = 0.5, False
    ordered = sorted(events)
    j = 0
    for i in range(T):
        f = int(frame[i])
        while j < len(ordered) and ordered[j] <= f:
            for e in events[ordered[j]]:
                if e["name"] in ("positions_reset", "game_start"):
                    cur, piece = 0.5, False
                elif e["name"] == "disc_props" and e.get("kind"):
                    d1 = e.get("data1") or [None] * 10
                    if d1[8] is not None:
                        if d1[8] >= 1000:
                            piece = True
                        else:
                            cur, piece = float(d1[8]), False
            j += 1
        mass[i] = -1.0 if piece else cur
    # saques por tramo de mapa
    idx = {int(f): i for i, f in enumerate(frame)}
    r_kind = np.zeros(T, np.int8)
    r_team = np.full(T, -1, np.int8)
    r_age = np.zeros(T, np.int32)
    pend = np.zeros(T, np.int8)
    restarts = []
    for a, b, name in spans:
        prm = C.params(name)
        lw, lh = C.LINE_W, prm[C.PI["line_half_h"]]
        evs = {f: v for f, v in events.items() if a <= f < b}
        for r in _restarts(evs, None, prm):
            i0 = idx.get(r["start"])
            if i0 is None:
                continue
            team_of = {}
            for s in range(8):
                if pid[i0, s] >= 0:
                    team_of[int(pid[i0, s])] = 1 if s < 4 else 2
            r["taker"] = _taker(r, events, team_of)
            # salida: primer tick con la pelota fuera de la cancha antes de la colocación
            out = r["start"]
            while out - 1 in idx:
                bx, by, br = ball[idx[out - 1], 0], ball[idx[out - 1], 1], ball[idx[out - 1], 4]
                if abs(bx) <= lw + br and abs(by) <= lh + br:
                    break
                out -= 1
            r["out"] = out
            r["map"] = name
            end = r["end"] if r["end"] is not None else r["start"] + 1
            for f in range(out, r["start"]):
                if f in idx:
                    pend[idx[f]] = r["kind"]
            for f in range(r["start"], end):
                if f in idx:
                    i = idx[f]
                    r_kind[i] = r["kind"]
                    r_team[i] = r["taker"]
                    r_age[i] = f - r["start"]
            restarts.append(r)
    rs = np.array([(r["start"], r["end"] if r["end"] is not None else -1, r["kind"], r["taker"], r["out"],
                    {"released": 1, "reassigned": 2, "reset": 3}.get(r["end_kind"], 0)) for r in restarts],
                  dtype=np.int32).reshape(-1, 6)
    rs_xy = np.array([(r["x"], r["y"]) for r in restarts], dtype=np.float32).reshape(-1, 2)
    return dict(version=np.array(VERSION), frame=frame, state=state, clock=clock, score=score, ball=ball,
                map_id=map_id, kick_strength=ks, ko_team=ko_team, ko_age=ko_age, pid=pid, pos=pos, vel=vel, inp=inp, kicking=kicking,
                n_red=n_red, n_blue=n_blue, mass=mass, restart_kind=r_kind, restart_team=r_team,
                restart_age=r_age, pending=pend,
                kicks=np.array(kicks, dtype=np.int32).reshape(-1, 2),
                goals=np.array(goals, dtype=np.int32).reshape(-1, 2),
                resets=np.array(resets, dtype=np.int32),
                restarts=rs, restarts_xy=rs_xy)


def _job(args):
    name, jsonl, stadiums, out = args
    t0 = time.time()
    try:
        res = build(jsonl, stadiums)
    except Exception as error:  # una grabación rota no tumba el lote
        import traceback
        return name, None, f"{type(error).__name__}: {error} {traceback.format_exc()[-300:]}"
    if res is None:
        return name, None, "sin tramos en mapas conocidos"
    np.savez_compressed(out, **res)
    maps = {k: int((res["map_id"] == v).sum()) for k, v in MAP_IDS.items() if (res["map_id"] == v).any()}
    return name, dict(ticks=int(len(res["frame"])), maps=maps, restarts=int(len(res["restarts"])),
                      kicks=int(len(res["kicks"])), goals=int(len(res["goals"])),
                      seconds=round(time.time() - t0, 1)), None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default=str(ROOT / "data" / "rs4_jsonl"))
    ap.add_argument("--out", default=str(ROOT / "data" / "x4_ticks"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--pattern", default="")
    a = ap.parse_args()
    cache, out = Path(a.cache), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    index = json.loads((cache / "index.json").read_text(encoding="utf-8"))["recordings"]
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    jobs = []
    for name, row in sorted(index.items()):
        if row.get("status") != "ok" or a.pattern not in name:
            continue
        stem = Path(name).stem
        target = out / f"{stem}.npz"
        if name in manifest and target.exists():
            continue
        stadiums = row.get("stadiums") or []
        jobs.append((name, str(cache / PureWindowsPath(row["jsonl"]).as_posix()), stadiums, str(target)))
    if a.limit:
        jobs = jobs[:a.limit]
    from multiprocessing import get_context
    with get_context("fork").Pool(max(1, a.workers)) as pool:
        for name, info, err in pool.imap_unordered(_job, jobs):
            manifest[name] = dict(status="ok" if info else "skip", error=err, **(info or {}))
            print(f"{'ok' if info else 'skip':4s} {name[:60]:60s} {json.dumps(info) if info else err}", flush=True)
            manifest_path.write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
