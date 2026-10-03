"""Extraer estados reales de partidos RS ONE 4v4 para situaciones técnicas y evaluación (PLAN_RS4.md 3-4).

Por partición (reports/rs4_b1/splits.json) guarda en data/rs4_states/<partición>.npz:

* juego abierto cada `--stride` ticks (plantel 4v4 estable, pelota dentro, sin saque activo);
* cada colocación de lateral, córner y saque de arco (tipo, equipo, punto);
* inicio de cada saque inicial.

Orden de jugadores como el simulador: rojos por id (0-3) y azules por id (4-7). Se guarda
también el último toque (contacto exacto o patada) y si hubo gol en los 12 s siguientes y
de quién, para medir cuánta señal de gol dan las situaciones.

  python -m tools.rs4_states --workers 10
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tools.rs4_rules_audit import load, rules

ROOT = Path(__file__).resolve().parent.parent
KIND = {"open": 0, "lateral": 1, "corner": 2, "goal_kick": 3, "kickoff": 4}
HORIZON = 720  # 12 s


def _order(tick):
    return sorted(tick["players"], key=lambda p: (p["team"], p["id"]))


def _state(tick, order):
    by_id = {p["id"]: p for p in tick["players"]}
    discs = tick["discs"]
    pp = np.array([[discs[by_id[p["id"]]["disc"]]["x"], discs[by_id[p["id"]]["disc"]]["y"]] for p in order])
    pv = np.array([[discs[by_id[p["id"]]["disc"]]["vx"], discs[by_id[p["id"]]["disc"]]["vy"]] for p in order])
    held = np.array([bool(by_id[p["id"]]["input"] & 16) and not by_id[p["id"]]["kicking"] for p in order])
    ball = discs[0]
    return (np.array([ball["x"], ball["y"]]), np.array([ball["vx"], ball["vy"]]), pp, pv, held)


def _last_touch(ticks_by_frame, events, frame, lookback=900):
    for f in range(frame, frame - lookback, -1):
        tick = ticks_by_frame.get(f)
        if tick is None:
            return -1
        ball = tick["discs"][0]
        teams = {next((p["team"] - 1 for p in tick["players"] if p["id"] == e["playerId"]), None)
                 for e in events.get(f, []) if e["name"] == "kick"} - {None}
        for p in tick["players"]:
            d = tick["discs"][p["disc"]]
            if np.hypot(d["x"] - ball["x"], d["y"] - ball["y"]) <= d["r"] + ball["r"] + 0.01:
                teams.add(p["team"] - 1)
        if len(teams) == 1:
            return teams.pop()
        if len(teams) > 1:
            return -1
    return -1


def extract(path, recording, split, stride=60):
    header, ticks, events = load(path)
    by = {t["frame"]: t for t in ticks}
    restarts, _ = rules(ticks, events)
    busy = np.zeros(max(by) + 2, dtype=bool)
    for r in restarts:
        busy[max(0, r["frame"] - 30):r["frame"] + r["duration"] + 30] = True
    goals = sorted((f, e.get("team")) for f, evs in events.items() for e in evs if e["name"] == "goal")
    rows = []

    def add(frame, kind, taker=-1, spot=(0.0, 0.0)):
        tick = by.get(frame)
        if tick is None or len(tick["players"]) != 8 or sum(p["team"] == 1 for p in tick["players"]) != 4:
            return
        order = _order(tick)
        # plantel estable durante el horizonte corto (no se mezclan cambios de jugadores)
        later = by.get(frame + 60)
        if later is None or {p["id"] for p in later["players"]} != {p["id"] for p in order}:
            return
        bp, bv, pp, pv, held = _state(tick, order)
        goal = next(((f, team) for f, team in goals if frame < f <= frame + HORIZON), None)
        rows.append(dict(ball_pos=bp, ball_vel=bv, player_pos=pp, player_vel=pv, kick_held=held,
                         last_touch=_last_touch(by, events, frame), kind=KIND[kind], taker=taker,
                         spot=np.asarray(spot, dtype=np.float64),
                         goal_team=-1 if goal is None else int(goal[1]) - 1,
                         goal_ticks=-1 if goal is None else int(goal[0] - frame),
                         score=np.asarray(tick.get("score") or [0, 0]), time=float(tick.get("time") or 0),
                         frame=frame))
    for tick in ticks[::stride]:
        f = tick["frame"]
        ball = tick["discs"][0]
        if tick["state"] == 1 and not busy[f] and abs(ball["y"]) <= 670 and abs(ball["x"]) <= 1150:
            add(f, "open")
    for r in restarts:
        if r["end"] == "liberado" and r["team"] is not None and r["kind"] in KIND:
            add(r["frame"], r["kind"], r["team"] - 1, r["place"])
    previous = None
    for tick in ticks:
        if tick["state"] == 0 and (previous is None or previous["state"] != 0):
            add(tick["frame"], "kickoff", (tick.get("ko") or 1) - 1)
        previous = tick
    for row in rows:
        row.update(recording=recording, split=split)
    return rows


def _job(args):
    path, recording, split, stride = args
    try:
        return extract(path, recording, split, stride)
    except Exception as error:
        print(f"error {recording}: {error}", flush=True)
        return []


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default=str(ROOT / "data" / "rs4_jsonl"))
    ap.add_argument("--splits", default=str(ROOT / "reports" / "rs4_b1" / "splits.json"))
    ap.add_argument("--out", default=str(ROOT / "data" / "rs4_states"))
    ap.add_argument("--stride", type=int, default=60)
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    splits = json.loads(Path(a.splits).read_text(encoding="utf-8"))
    jobs = [(str(Path(a.cache) / row["jsonl"].replace("\\", "/")), name, row["split"], a.stride)
            for name, row in splits["recordings"].items() if row["only_rs_one"]]
    rows = []
    if a.workers > 1:
        from multiprocessing import get_context
        with get_context("spawn").Pool(a.workers) as pool:
            for part in pool.imap_unordered(_job, jobs):
                rows += part
    else:
        for job in jobs:
            rows += _job(job)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for split in ("entrenamiento", "desarrollo", "prueba"):
        sub = sorted((r for r in rows if r["split"] == split), key=lambda r: (r["recording"], r["frame"]))
        if not sub:
            continue
        arrays = {key: np.stack([np.asarray(r[key]) for r in sub]) for key in
                  ("ball_pos", "ball_vel", "player_pos", "player_vel", "kick_held", "last_touch", "kind", "taker",
                   "spot", "goal_team", "goal_ticks", "score", "time", "frame")}
        recordings = sorted({r["recording"] for r in sub})
        arrays["recording"] = np.array([recordings.index(r["recording"]) for r in sub], dtype=np.int64)
        np.savez_compressed(out / f"{split}.npz", recordings=np.array(recordings), **arrays)
        kinds = {name: int((arrays["kind"] == code).sum()) for name, code in KIND.items()}
        summary[split] = dict(states=len(sub), recordings=len(recordings), kinds=kinds,
                              goal_within_12s=float((arrays["goal_team"] >= 0).mean()))
    (out / "manifest.json").write_text(json.dumps(dict(version="RS4-b1-states-1", stride=a.stride, horizon_ticks=HORIZON,
                                                        summary=summary), indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
