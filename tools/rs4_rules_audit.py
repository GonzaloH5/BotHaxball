"""Auditoría de física y reglas de Real Soccer ONE 4v4 contra las grabaciones (PLAN_RS4.md 2.2).

Lee el caché `data/rs4_jsonl` (motor original de HaxBall, `tools/rs4_jsonl_cache.py`) y mide:

1. Variantes de mapa: parámetros físicos y geometría de cada estadio grabado frente al catálogo.
2. Física: error de 1 tick y de 60 ticks del simulador con el mismo estado y las mismas teclas,
   sólo en transiciones sin intervención del script y con plantel 4v4 estable. Con el mapa grabado
   y con el del catálogo.
3. Reglas del script, reconstruidas desde sus propias acciones (`disc_props` con valores):
   laterales, córners y saques de arco (colocación, equipo, duración, liberación, primer toque,
   restricciones de rivales), saque inicial, masa/grupos de jugadores y comba de la pelota.

No modifica el simulador ni las grabaciones. Lo que no puede inferirse se informa como pendiente.

  python -m tools.rs4_rules_audit --out reports/rs4_b1/rules_audit.json
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path

import numpy as np

from tools.rs4_jsonl_cache import open_ticks

ROOT = Path(__file__).resolve().parent.parent
WHITE = 0xFFFFFF
LINE_Y, LINE_X = 670.0, 1150.0  # medidas reales de RS ONE (bg del .hbs)


def load(path):
    header, ticks, events = None, [], collections.defaultdict(list)
    with open_ticks(path) as handle:
        for line in handle:
            item = json.loads(line)
            kind = item.get("type")
            if kind == "tick":
                ticks.append(item)
            elif kind == "header":
                header = item
            elif kind == "event":
                events[item["frame"]].append(item)
    return header, ticks, events


# ----------------------------------------------------------------------------------- mapas
def map_variant(hbs_path):
    data = json.loads(Path(hbs_path).read_text(encoding="utf-8"))
    ball = data["discs"][0] if data.get("ballPhysics") == "disc0" else data.get("ballPhysics", {})
    player = data.get("playerPhysics", {})
    geometry = json.dumps({k: data.get(k) for k in ("vertexes", "segments", "planes", "goals", "joints")}, sort_keys=True)
    return dict(name=data.get("name"), bg=data.get("bg"), width=data.get("width"), height=data.get("height"),
                spawn_distance=data.get("spawnDistance"),
                ball={k: ball.get(k) for k in ("radius", "invMass", "bCoef", "damping")},
                player={k: player.get(k) for k in ("radius", "invMass", "bCoef", "damping", "acceleration",
                                                    "kickingAcceleration", "kickingDamping", "kickStrength", "kickback")},
                discs=len(data.get("discs", [])), geometry_sha=hashlib.sha256(geometry.encode()).hexdigest()[:16])


# ----------------------------------------------------------------------------------- física
def with_player_inv_mass(stadium, value):
    import dataclasses
    return dataclasses.replace(stadium, player={**stadium.player, "invMass": float(value)})


def mass_regime(ticks, events):
    """Masa inversa de jugadores vigente por frame según las acciones del script.

    0,5 (mapa) hasta la primera asignación; luego la última asignada. Si algún jugador
    está en 100000 (córner/saque de arco) el frame se marca como None y no se compara.
    """
    current = {}
    changes = collections.defaultdict(list)
    for frame, evs in events.items():
        for e in evs:
            if e["name"] == "disc_props" and e.get("kind"):
                d1 = e.get("data1") or [None] * 10
                if d1[8] is not None:
                    changes[frame].append((e["id"], d1[8]))
    regime = {}
    for tick in ticks:
        for pid, value in changes.get(tick["frame"], []):
            current[pid] = value
        values = {current.get(p["id"], 0.5) for p in tick["players"]}
        regime[tick["frame"]] = values.pop() if len(values) == 1 and 100000 not in values else (
            None if any(v >= 1000 for v in values) else "mixto")
    return regime


def physics(ticks, events, hbs_path, catalog, max_one_tick=1500, max_rollouts=150, horizon=60):
    """Tres simuladores: el actual (masa del mapa, 0,5, siempre) y el contrato con la masa que
    fija el script (0,5 hasta el primer saque, luego 0,3), con el mapa grabado y con el catálogo."""
    from bridge.compare_sim import Replayer, real_stadium
    script = {f for f, evs in events.items() if any(e["name"] == "disc_props" for e in evs)}
    resets = {f for f, evs in events.items() if any(e["name"] in ("positions_reset", "goal") for e in evs)}
    roster = lambda t: tuple(sorted((p["id"], p["team"]) for p in t["players"]))
    regime = mass_regime(ticks, events)
    recorded = real_stadium(hbs_path)
    configs = {"actual_masa_0.5": (recorded, None),
               "grabado_masa_script": (recorded, "script")}
    if catalog is not None:
        configs["catalogo_masa_script"] = (catalog, "script")
    out = {}
    for label, (base_stadium, mode) in configs.items():
        variants = {0.5: with_player_inv_mass(base_stadium, 0.5), 0.3: with_player_inv_mass(base_stadium, 0.3)}
        one = collections.defaultdict(list)
        roll = []
        replay, replay_mass = None, None
        for i in range(len(ticks) - 1):
            a, b = ticks[i], ticks[i + 1]
            mass = 0.5 if mode is None else regime.get(a["frame"])
            if (len(a["players"]) != 8 or b["frame"] != a["frame"] + 1 or roster(a) != roster(b)
                    or a["state"] != 1 or b["state"] != 1 or a["frame"] in script or b["frame"] in script
                    or b["frame"] in resets or mass not in (0.5, 0.3)
                    or (mode is not None and regime.get(b["frame"]) != mass)):
                replay = None
                continue
            try:
                if replay is None or replay_mass != mass:
                    replay, replay_mass = Replayer(variants[mass], [a]), mass
                if sum(len(v) for v in one.values()) < max_one_tick:
                    replay.load(a)
                    replay.step(a)
                    actual, velocity = replay.real(b)
                    fp = replay.sim.first_player
                    ball_err = float(np.linalg.norm(replay.sim.pos[0, 0] - actual[0]))
                    player_err = float(np.linalg.norm(replay.sim.pos[0, fp:] - actual[fp:], axis=1).max())
                    kicked = any(e["name"] == "kick" for e in events.get(a["frame"], []) + events.get(b["frame"], []))
                    near = float(np.linalg.norm(actual[fp:] - actual[0], axis=1).min()) < 30
                    category = "patada" if kicked else ("contacto" if near else "libre")
                    one[category].append((ball_err, player_err))
                if len(roll) < max_rollouts and i % 97 == 0 and i + horizon < len(ticks):
                    window = ticks[i:i + horizon + 1]
                    if all(window[j + 1]["frame"] == window[j]["frame"] + 1 and window[j]["frame"] not in script
                           and window[j]["state"] == 1 and roster(window[j]) == roster(a)
                           and (mode is None or regime.get(window[j]["frame"]) == mass) for j in range(horizon)):
                        replay.load(window[0])
                        for j in range(horizon):
                            replay.step(window[j])
                        actual, _ = replay.real(window[-1])
                        fp = replay.sim.first_player
                        roll.append((float(np.linalg.norm(replay.sim.pos[0, 0] - actual[0])),
                                     float(np.linalg.norm(replay.sim.pos[0, fp:] - actual[fp:], axis=1).max())))
            except (SystemExit, ValueError, IndexError, KeyError):
                replay = None
        summary = {}
        for category, rows in one.items():
            array = np.asarray(rows)
            summary[category] = dict(n=len(rows), pelota_p90=float(np.percentile(array[:, 0], 90)),
                                     pelota_p99=float(np.percentile(array[:, 0], 99)),
                                     jugadores_p90=float(np.percentile(array[:, 1], 90)),
                                     jugadores_p99=float(np.percentile(array[:, 1], 99)))
        if roll:
            array = np.asarray(roll)
            summary[f"secuencia_{horizon}"] = dict(n=len(roll), pelota_p50=float(np.median(array[:, 0])),
                                                   pelota_p90=float(np.percentile(array[:, 0], 90)),
                                                   jugadores_p50=float(np.median(array[:, 1])),
                                                   jugadores_p90=float(np.percentile(array[:, 1], 90)))
        out[label] = summary
    return out


# ----------------------------------------------------------------------------------- reglas
def _ball_actions(events):
    """Acciones del script sobre la pelota por frame: colocación, color y máscara."""
    rows = collections.defaultdict(dict)
    for frame, evs in events.items():
        for e in evs:
            if e["name"] != "disc_props" or e.get("kind") or e.get("id") != 0:
                continue
            d1 = e.get("data1") or [None] * 10
            d2 = e.get("data2") or [None] * 3
            if d1[0] is not None and d1[1] is not None:
                rows[frame]["place"] = (d1[0], d1[1], d1[2] or 0.0, d1[3] or 0.0)
            if d2[0] is not None:
                rows[frame]["color"] = d2[0]
            if d2[1] is not None:
                rows[frame]["cmask"] = d2[1]
            if d1[4] is not None:
                rows[frame]["gravity"] = (d1[4], d1[5])
    return rows


def classify(place):
    x, y = abs(place[0]), abs(place[1])
    if abs(y - 688) < 6:
        return "lateral"
    if abs(x - 1140) < 6 and abs(y - 660) < 6:
        return "corner"
    if abs(x - 1030) < 6 and abs(y - 180) < 6:
        return "goal_kick"
    return "otro"


def _players(tick):
    return {p["id"]: (p["team"], tick["discs"][p["disc"]]) for p in tick["players"]}


def rules(ticks, events):
    by_frame = {t["frame"]: t for t in ticks}
    actions = _ball_actions(events)
    team_colors = {}  # color -> equipo inferido por quién toca primero
    restarts = []
    kicks_by_frame = {f: [e["playerId"] for e in evs if e["name"] == "kick"] for f, evs in events.items()}
    frames = sorted(actions)
    for frame in frames:
        action = actions[frame]
        if "place" not in action or abs(action["place"][2]) > 1e-6 or abs(action["place"][3]) > 1e-6:
            continue
        color = action.get("color")
        if color is None or color == WHITE:
            continue
        kind = classify(action["place"])
        before = by_frame.get(frame - 1)
        out_point = (before["discs"][0]["x"], before["discs"][0]["y"]) if before else None
        # liberación: siguiente frame en que la pelota vuelve a blanco o se coloca otra vez
        release, end_reason = None, None
        for later in frames:
            if later <= frame:
                continue
            nxt = actions[later]
            if nxt.get("color") == WHITE:
                release, end_reason = later, "liberado"
                break
            if "place" in nxt or ("color" in nxt and nxt["color"] != color):
                release, end_reason = later, "reemplazado"
                break
        if release is None:
            continue
        first_touch, kicks = None, 0
        opponents_min, opponents_beyond = [], 0
        samples = 0
        for f in range(frame, release + 1):
            tick = by_frame.get(f)
            if not tick:
                continue
            ball = tick["discs"][0]
            players = _players(tick)
            for pid in kicks_by_frame.get(f, []):
                kicks += 1
                if first_touch is None and pid in players:
                    first_touch = (players[pid][0], f - frame, "patada")
            if first_touch is None:
                for pid, (team, disc) in players.items():
                    if np.hypot(disc["x"] - ball["x"], disc["y"] - ball["y"]) <= disc["r"] + ball["r"] + 0.5:
                        first_touch = (team, f - frame, "contacto")
                        break
            samples += 1
        if first_touch is not None and end_reason == "liberado":
            team_colors.setdefault(color, collections.Counter())[first_touch[0]] += 1
        rel_tick = by_frame.get(release)
        restarts.append(dict(frame=frame, kind=kind, color=color, place=action["place"][:2], out_point=out_point,
                             duration=release - frame, end=end_reason,
                             release_ball=(rel_tick["discs"][0]["x"], rel_tick["discs"][0]["y"]) if rel_tick else None,
                             first_touch_team=first_touch[0] if first_touch else None,
                             first_touch_after=first_touch[1] if first_touch else None,
                             first_touch_kind=first_touch[2] if first_touch else None, kicks=kicks,
                             cmask=action.get("cmask")))
    color_team = {c: counts.most_common(1)[0][0] for c, counts in team_colors.items()}
    for row in restarts:
        row["team"] = color_team.get(row["color"])
        row["taken_by_owner"] = (row["first_touch_team"] == row["team"]) if row["team"] and row["first_touch_team"] else None
    # Restricciones de rivales durante laterales: distancia mínima a la pelota y |y| máximo.
    for row in restarts:
        if row["team"] is None:
            continue
        mins, ys = [], []
        for f in range(row["frame"] + 1, row["frame"] + row["duration"]):
            tick = by_frame.get(f)
            if not tick:
                continue
            ball = tick["discs"][0]
            for pid, (team, disc) in _players(tick).items():
                if team != row["team"]:
                    mins.append(np.hypot(disc["x"] - ball["x"], disc["y"] - ball["y"]))
                    ys.append(abs(disc["y"]))
        row["opponent_min_distance"] = float(min(mins)) if mins else None
        row["opponent_max_abs_y"] = float(max(ys)) if ys else None
    return restarts, {str(k): v for k, v in color_team.items()}


def player_props(ticks, events):
    """Masa y grupos de colisión que el script asigna a jugadores, con contexto."""
    by_frame = {t["frame"]: t for t in ticks}
    rows = collections.Counter()
    for frame, evs in events.items():
        for e in evs:
            if e["name"] != "disc_props" or not e.get("kind"):
                continue
            d1 = e.get("data1") or [None] * 10
            d2 = e.get("data2") or [None] * 3
            tick = by_frame.get(frame)
            state = tick["state"] if tick else None
            if d1[8] is not None:
                rows[("invMass", d1[8], state)] += 1
            if d2[2] is not None:
                rows[("cGroup", d2[2], state)] += 1
            if d1[1] is not None:
                rows[("y", round(d1[1]), state)] += 1
            if d1[0] is not None:
                rows[("x", round(d1[0]), state)] += 1
    return [dict(prop=k[0], value=k[1], state=k[2], count=v) for k, v in rows.most_common()]


def kickoffs(ticks, events):
    rows, start = [], None
    for tick in ticks:
        if tick["state"] == 0 and start is None:
            start = tick
        elif tick["state"] != 0 and start is not None:
            rows.append(dict(frame=start["frame"], duration=tick["frame"] - start["frame"], team=start.get("ko")))
            start = None
    return rows


def curve(events):
    actions = _ball_actions(events)
    frames = sorted(f for f in actions if "gravity" in actions[f])
    starts, last = 0, (0.0, 0.0)
    for f in frames:
        g = actions[f]["gravity"]
        if (g[0] or g[1]) and not (last[0] or last[1]):
            starts += 1
        last = g
    kicks = sum(1 for evs in events.values() for e in evs if e["name"] == "kick")
    return dict(curve_kicks=starts, kicks=kicks, fraction=starts / max(kicks, 1))


def audit_recording(path, hbs_path, catalog, with_physics=True):
    header, ticks, events = load(path)
    restarts, color_team = rules(ticks, events)
    out = dict(source=str(path), room=(header or {}).get("room"), ticks=len(ticks),
               map=map_variant(hbs_path), restarts=restarts, color_team=color_team,
               player_props=player_props(ticks, events)[:40], kickoffs=kickoffs(ticks, events), curve=curve(events),
               out_of_line_ball=dict(max_abs_y=float(max((abs(t["discs"][0]["y"]) for t in ticks if t["state"] == 1), default=0)),
                                     max_abs_x=float(max((abs(t["discs"][0]["x"]) for t in ticks if t["state"] == 1), default=0))))
    if with_physics:
        out["physics"] = physics(ticks, events, hbs_path, catalog)
    return out


def summarize(reports):
    variants = collections.Counter()
    for r in reports:
        m = r["map"]
        variants[(m["name"], m["geometry_sha"], json.dumps(m["player"], sort_keys=True), json.dumps(m["ball"], sort_keys=True),
                  json.dumps(m["bg"], sort_keys=True))] += 1
    restarts = [row for r in reports for row in r["restarts"]]
    by_kind = collections.defaultdict(list)
    for row in restarts:
        by_kind[row["kind"]].append(row)
    kinds = {}
    for kind, rows in by_kind.items():
        durations = [r["duration"] for r in rows]
        owners = [r["taken_by_owner"] for r in rows if r["taken_by_owner"] is not None]
        opp = [r["opponent_min_distance"] for r in rows if r.get("opponent_min_distance") is not None]
        kinds[kind] = dict(n=len(rows), duration_p10=float(np.percentile(durations, 10)),
                           duration_p50=float(np.median(durations)), duration_p90=float(np.percentile(durations, 90)),
                           duration_max=int(max(durations)), replaced=sum(r["end"] == "reemplazado" for r in rows),
                           taken_by_owner=float(np.mean(owners)) if owners else None,
                           first_touch_contact=float(np.mean([r["first_touch_kind"] == "contacto" for r in rows if r["first_touch_kind"]])) if any(r["first_touch_kind"] for r in rows) else None,
                           opponent_min_distance_p10=float(np.percentile(opp, 10)) if opp else None,
                           placements=collections.Counter(f"({abs(round(r['place'][0]))},{abs(round(r['place'][1]))})"
                                                          for r in rows if kind != "lateral").most_common(4))
    ko = [k["duration"] for r in reports for k in r["kickoffs"]]
    curves = [r["curve"] for r in reports]
    physics_rows = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in reports:
        for stadium, cats in (r.get("physics") or {}).items():
            for cat, values in cats.items():
                physics_rows[stadium][cat].append(values)
    phys = {s: {c: {k: float(np.median([v[k] for v in vals])) for k in vals[0] if k != "n"} | {"n": int(sum(v["n"] for v in vals))}
                for c, vals in cats.items()} for s, cats in physics_rows.items()}
    return dict(recordings=len(reports),
                map_variants=[dict(count=c, name=k[0], geometry_sha=k[1], player=json.loads(k[2]), ball=json.loads(k[3]),
                                   bg=json.loads(k[4])) for k, c in variants.most_common()],
                restarts=kinds,
                kickoff=dict(n=len(ko), duration_p50=float(np.median(ko)) if ko else None,
                             duration_p90=float(np.percentile(ko, 90)) if ko else None, duration_max=int(max(ko)) if ko else None),
                curve=dict(curve_kicks=int(sum(c["curve_kicks"] for c in curves)), kicks=int(sum(c["kicks"] for c in curves))),
                ball_out_of_line=dict(max_abs_y=float(max(r["out_of_line_ball"]["max_abs_y"] for r in reports))),
                physics_median_of_recordings=phys)


def _audit_job(job):
    name, digest, path, hbs, with_physics = job
    from sim.stadium import load_stadium
    try:
        report = audit_recording(Path(path), Path(hbs), load_stadium("rs_one"), with_physics)
    except Exception as error:
        print(f"error {name}: {error}", flush=True)
        return None
    report["recording"], report["sha256"] = name, digest
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default=str(ROOT / "data" / "rs4_jsonl"))
    ap.add_argument("--out", default=str(ROOT / "reports" / "rs4_b1" / "rules_audit.json"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-physics", action="store_true")
    ap.add_argument("--stadium-filter", default="Real Soccer ONE")
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    cache = Path(a.cache)
    index = json.loads((cache / "index.json").read_text(encoding="utf-8"))
    jobs = []
    for name, row in sorted(index["recordings"].items()):
        if row.get("status") != "ok":
            continue
        stadiums = row.get("stadiums") or []
        if not stadiums or not any(a.stadium_filter.lower() in (s.get("name") or "").lower() for s in stadiums):
            continue
        if len(stadiums) != 1:
            continue  # cambio de mapa dentro de la grabación: se audita aparte
        path = cache / row["jsonl"].replace("\\", "/")
        jobs.append((name, row["sha256"], str(path), str(path.parent / stadiums[0]["file"]), not a.no_physics))
    if a.limit:
        jobs = jobs[:a.limit]
    reports = []
    if a.workers > 1:
        from multiprocessing import get_context
        with get_context("spawn").Pool(a.workers) as pool:
            for report in pool.imap_unordered(_audit_job, jobs):
                if report is not None:
                    reports.append(report)
                    print(f"{report['recording']}: {len(report['restarts'])} saques, {report['ticks']} ticks", flush=True)
    else:
        for job in jobs:
            report = _audit_job(job)
            if report is not None:
                reports.append(report)
                print(f"{report['recording']}: {len(report['restarts'])} saques, {report['ticks']} ticks", flush=True)
    reports.sort(key=lambda r: r["recording"])
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    result = dict(version="RS4-b1-rules-audit-1", summary=summarize(reports), recordings=reports)
    out.write_text(json.dumps(result, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(result["summary"], indent=1, ensure_ascii=False, default=str)[:6000])


if __name__ == "__main__":
    main()
