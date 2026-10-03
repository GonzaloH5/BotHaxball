"""Conformidad del árbitro rs_one_v1 contra saques reales (PLAN_RS4.md 2.2, condición para entrenar).

Dos pruebas por saque grabado (lateral, córner, saque de arco), tick a tick con las teclas reales:

A. Cobro: arranca `lead` ticks antes de la colocación, con el último toque real (escaneado
   hacia atrás en la grabación). Compara si se cobra, tipo, equipo, colocación y demora.
   La sala cobra con una demora variable de 1 a ~8 ticks; el simulador cobra al instante.
B. Ejecución: arranca en el tick de la colocación real, con el saque activo y el estado
   posterior a las acciones del script. Compara la liberación y la velocidad tras el impulso.

Las pruebas son a lazo abierto: una divergencia física acumulada invalida la comparación,
por eso las ventanas son cortas y se informan los descartes.

  python -m tools.rs4_referee_conformance --workers 10 --out reports/rs4_b1/referee_conformance.json
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np

from tools.rs4_rules_audit import load, rules

ROOT = Path(__file__).resolve().parent.parent
KIND = {"lateral": 1, "corner": 2, "goal_kick": 3}


def _env():
    from env.haxball_env import HaxballEnv
    from env.rewards import RewardConfig
    reward = RewardConfig(shaping_coef=0, team_spread_floor=0, kickoff_approach=0, w_defense_support=0,
                          rs4_tactical_coef=0, corner_execute=0, out_penalty=0)
    return HaxballEnv(1, 4, "rs_one", frame_skip=1, max_ticks=10 ** 9, random_reset_prob=0, seed=0,
                      obs_layout="universal", max_entities=7, out_of_bounds=True, corner_reset_prob=0,
                      kickoff_timeout=0, reward=reward, referee="rs_one_v1", optimize_rollout=False)


def last_toucher(by_frame, events, frame, lookback=900):
    """Equipo (0 rojo, 1 azul) del último contacto o patada antes de `frame`; None si ambiguo."""
    for f in range(frame - 1, frame - lookback, -1):
        tick = by_frame.get(f)
        if tick is None:
            return None
        ball = tick["discs"][0]
        teams = set()
        for e in events.get(f, []):
            if e["name"] == "kick":
                player = next((p for p in tick["players"] if p["id"] == e["playerId"]), None)
                if player is not None:
                    teams.add(player["team"] - 1)
        for p in tick["players"]:
            d = tick["discs"][p["disc"]]
            if np.hypot(d["x"] - ball["x"], d["y"] - ball["y"]) <= d["r"] + ball["r"] + 0.5:
                teams.add(p["team"] - 1)
        if len(teams) == 1:
            return teams.pop()
        if len(teams) > 1:
            return None
    return None


def _actions(env, order, tick):
    from bridge.compare_sim import input_to_action
    from env.haxball_env import MIRROR_ACTION
    by_id = {p["id"]: p for p in tick["players"]}
    if any(p["id"] not in by_id for p in order):
        return None
    world = np.array([[input_to_action(by_id[p["id"]]["input"]) for p in order]], dtype=np.int64)
    blue = env.sim.player_team == 1
    actions = world.copy()
    actions[:, blue] = MIRROR_ACTION[world[:, blue]]
    return actions


def _load(env, replay_cls, tick):
    replay = replay_cls(env.sim.st, [tick])
    replay.sim = env.sim
    env.reset()
    env.sim.kickoff[:] = False
    replay.load(tick)
    env.setpiece_team[:] = -1
    env.setpiece_kind[:] = 0
    env._rs1.reset_rows(np.array([0]))
    return replay


def test_call(env, replay_cls, by_frame, events, restart, previous_release, lead=30):
    place = restart["frame"]
    start = max(place - lead, (previous_release or 0) + 20)
    if place - start < 6:
        return dict(status="ventana_corta")
    first = by_frame.get(start)
    if first is None or len(first["players"]) != 8 or first["state"] != 1:
        return dict(status="no_4v4_o_no_en_juego")
    ball = first["discs"][0]
    if abs(ball["y"]) > 670 or abs(ball["x"]) > 1150:
        return dict(status="pelota_fuera_al_inicio")
    team_last = last_toucher(by_frame, events, start)
    replay = _load(env, replay_cls, first)
    env._rs1.armed[:] = True
    env.last_touch[:] = -1 if team_last is None else team_last
    detected = kind = team = spot = None
    for f in range(start, place + 12):
        tick = by_frame.get(f)
        if tick is None:
            break
        actions = _actions(env, replay.players, tick)
        if actions is None:
            break
        env.step(actions)
        if env.setpiece_team[0] >= 0:
            detected, kind, team = f + 1, int(env.setpiece_kind[0]), int(env.setpiece_team[0])
            spot = env.sim.ball_pos[0].tolist()
            break
    team_real = None if restart["team"] is None else restart["team"] - 1
    return dict(status="ok", last_touch_known=team_last is not None, detected=detected,
                detect_delta=None if detected is None else detected - place,
                kind_sim=kind, team_sim=team, team_real=team_real, spot_sim=spot)


def test_execution(env, replay_cls, by_frame, events, restart, follow=900):
    place = restart["frame"]
    tick = by_frame.get(place)
    if tick is None or len(tick["players"]) != 8:
        return dict(status="sin_ticks")
    team_real = restart["team"]
    if team_real is None:
        return dict(status="equipo_desconocido")
    replay = _load(env, replay_cls, tick)
    kind = KIND[restart["kind"]]
    sim = env.sim
    spot = np.array(restart["place"], dtype=np.float64)
    env._rs1.start(0, kind, team_real - 1, spot)  # mismas acciones de inicio que el script
    release = boosted = None
    for f in range(place, place + follow):
        current = by_frame.get(f)
        if current is None:
            break
        actions = _actions(env, replay.players, current)
        if actions is None:
            break
        env.step(actions)
        if release is None and env.setpiece_team[0] < 0:
            release = f + 1
        if release is not None and f + 1 >= release + 3:
            boosted = float(np.hypot(*sim.ball_vel[0]))
            break
    real_release = place + restart["duration"]
    after = by_frame.get(real_release + 3)
    real_boost = float(np.hypot(after["discs"][0]["vx"], after["discs"][0]["vy"])) if after else None
    return dict(status="ok", release_real=real_release, release_sim=release,
                release_delta=None if release is None else release - real_release,
                boost_real=real_boost, boost_sim=boosted)


def conformance(path):
    from bridge.compare_sim import Replayer
    header, ticks, events = load(path)
    by_frame = {t["frame"]: t for t in ticks}
    restarts, _ = rules(ticks, events)
    env = _env()
    rows, previous_release = [], None
    for restart in sorted(restarts, key=lambda r: r["frame"]):
        if restart["end"] == "liberado" and restart["kind"] in KIND:
            row = dict(recording=Path(path).parent.name, kind_real=restart["kind"], place_real=restart["place"])
            for label, test in (("cobro", lambda: test_call(env, Replayer, by_frame, events, restart, previous_release)),
                                ("ejecucion", lambda: test_execution(env, Replayer, by_frame, events, restart))):
                try:
                    row[label] = test()
                except (SystemExit, KeyError, IndexError, ValueError) as error:
                    row[label] = dict(status=f"error: {error}")
            rows.append(row)
        previous_release = restart["frame"] + restart["duration"]
    return rows


def _job(path):
    try:
        return conformance(path)
    except Exception as error:
        print(f"error {path}: {error}", flush=True)
        return []


def summarize(rows):
    out = {}
    for kind, code in KIND.items():
        sub = [r for r in rows if r["kind_real"] == kind]
        calls = [r["cobro"] for r in sub if r["cobro"]["status"] == "ok"]
        detected = [c for c in calls if c["detected"] is not None]
        known = [c for c in detected if c["last_touch_known"] and c["team_real"] is not None]
        execs = [r for r in sub if r["ejecucion"]["status"] == "ok"]
        released = [r["ejecucion"] for r in execs if r["ejecucion"]["release_sim"] is not None]
        release_err = [abs(e["release_delta"]) for e in released]
        boost = [(e["boost_sim"], e["boost_real"]) for e in released if e["boost_sim"] and e["boost_real"] and e["boost_real"] > 1]
        spot_err = [float(np.hypot(c["spot_sim"][0] - r["place_real"][0], c["spot_sim"][1] - r["place_real"][1]))
                    for r in sub for c in [r["cobro"]] if c["status"] == "ok" and c["detected"] is not None and c["kind_sim"] == code]
        out[kind] = dict(
            n=len(sub),
            cobro=dict(evaluados=len(calls), descartes=dict(collections.Counter(r["cobro"]["status"] for r in sub if r["cobro"]["status"] != "ok")),
                       detectado=len(detected) / max(len(calls), 1),
                       demora_ticks=dict(collections.Counter(int(np.clip(c["detect_delta"], -9, 9)) for c in detected).most_common(10)),
                       dentro_de_8_ticks=float(np.mean([abs(c["detect_delta"]) <= 8 for c in detected])) if detected else None,
                       ultimo_toque_conocido=len(known),
                       tipo_coincide=float(np.mean([c["kind_sim"] == code for c in known])) if known else None,
                       equipo_coincide=float(np.mean([c["team_sim"] == c["team_real"] for c in known])) if known else None,
                       error_colocacion_p50=float(np.median(spot_err)) if spot_err else None,
                       error_colocacion_p90=float(np.percentile(spot_err, 90)) if spot_err else None),
            ejecucion=dict(evaluados=len(execs), liberado=len(released) / max(len(execs), 1),
                           error_liberacion_ticks_p50=float(np.median(release_err)) if release_err else None,
                           error_liberacion_ticks_p90=float(np.percentile(release_err, 90)) if release_err else None,
                           liberacion_dentro_de_3_ticks=float(np.mean([x <= 3 for x in release_err])) if release_err else None,
                           impulso_sim_sobre_real_p10=float(np.percentile([s / r for s, r in boost], 10)) if boost else None,
                           impulso_sim_sobre_real_p50=float(np.median([s / r for s, r in boost])) if boost else None,
                           impulso_sim_sobre_real_p90=float(np.percentile([s / r for s, r in boost], 90)) if boost else None))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default=str(ROOT / "data" / "rs4_jsonl"))
    ap.add_argument("--out", default=str(ROOT / "reports" / "rs4_b1" / "referee_conformance.json"))
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--limit-recordings", type=int, default=0)
    a = ap.parse_args()
    cache = Path(a.cache)
    index = json.loads((cache / "index.json").read_text(encoding="utf-8"))
    paths = [str(cache / row["jsonl"]) for row in index["recordings"].values() if row.get("status") == "ok"
             and [s.get("name") for s in row.get("stadiums") or []] == ["Real Soccer ONE"]]
    if a.limit_recordings:
        paths = paths[:a.limit_recordings]
    rows = []
    if a.workers > 1:
        from multiprocessing import get_context
        with get_context("spawn").Pool(a.workers) as pool:
            for part in pool.imap_unordered(_job, paths):
                rows += part
    else:
        for path in paths:
            rows += _job(path)
    result = dict(version="RS4-b1-referee-conformance-2", summary=summarize(rows), episodes=rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    print(json.dumps(result["summary"], indent=1, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
