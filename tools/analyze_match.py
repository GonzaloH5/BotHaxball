"""Diagnóstico de un partido (p. ej. una persona contra el bot): en qué se diferencian los jugadores.

python -m tools.analyze_match <replay.hbr2 | grabación.jsonl>

Por jugador: goles, distancia media a la pelota, cuánto está más cerca de la pelota que el rival, patadas
reales, cuánto aprieta patear, cuánto está quieto, velocidad, cuánto está "del lado de su arco" (entre la
pelota y su arco: la base de defender) y, en cada gol recibido, dónde estaba el que defendía.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def load(path: Path):
    if path.suffix == ".jsonl":
        jsonl, names = path, {}
    else:
        tmp = Path(tempfile.mkdtemp(prefix="haxballrl_match_"))
        subprocess.run(["node", "bridge/replay_to_jsonl.js", str(path), "--out", str(tmp), "--max-minutes", "120"],
                       cwd=ROOT, check=True, capture_output=True)
        jsonl = next(tmp.glob("*.jsonl"))
        info = json.loads(subprocess.run(["node", "bridge/replay_players.js", str(path)], cwd=ROOT, check=True,
                                         capture_output=True, text=True, encoding="utf-8").stdout)
        names = {p["id"]: p["name"] for p in info["players"]}
    ticks, events = [], []
    for line in jsonl.open(encoding="utf-8"):
        o = json.loads(line)
        (ticks if o["type"] == "tick" else events).append(o)
    return ticks, events, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    a = ap.parse_args()
    ticks, events, names = load(Path(a.path))
    st = defaultdict(lambda: defaultdict(float))
    team_of = {}
    n_play = 0
    for t in ticks:
        if t["state"] != 1 or t["discs"][0]["x"] is None:
            continue
        n_play += 1
        b = t["discs"][0]
        bx, by = b["x"], b["y"]
        dists = {}
        for p in t["players"]:
            d = t["discs"][p["disc"]]
            if d["x"] is None:
                continue
            pid = p["id"]
            team_of[pid] = p["team"]
            s = st[pid]
            dist = float(np.hypot(d["x"] - bx, d["y"] - by))
            dists[pid] = dist
            s["n"] += 1
            s["dist"] += dist
            s["speed"] += float(np.hypot(d["vx"], d["vy"]))
            s["kick_held"] += bool(p["input"] & 16)
            s["idle"] += p["input"] == 0 and np.hypot(d["vx"], d["vy"]) < 0.3
            # rojo defiende el arco de x<0: está "del lado de su arco" si está a la izquierda de la pelota
            goal_side = d["x"] < bx if p["team"] == 1 else d["x"] > bx
            s["goal_side"] += goal_side
        if dists:
            best = min(dists, key=dists.get)
            st[best]["closest"] += 1
    kicks = defaultdict(int)
    for e in events:
        if e.get("name") == "kick":
            kicks[e["playerId"]] += 1
    goals = defaultdict(int)
    conceded_pos = defaultdict(list)
    by_frame = {t["frame"]: t for t in ticks}
    for e in events:
        if e.get("name") != "goal":
            continue
        goals[e["team"]] += 1
        t = by_frame.get(e["frame"] - 1) or by_frame.get(e["frame"])
        if not t:
            continue
        b = t["discs"][0]
        for p in t["players"]:
            if p["team"] != e["team"]:  # el que recibió el gol
                d = t["discs"][p["disc"]]
                if d["x"] is not None and b["x"] is not None:
                    goal_side = d["x"] < b["x"] if p["team"] == 1 else d["x"] > b["x"]
                    conceded_pos[p["id"]].append((float(np.hypot(d["x"] - b["x"], d["y"] - b["y"])), goal_side))
    minutes = n_play / 3600
    print(f"{minutes:.1f} min en juego | goles: rojo {goals[1]} - azul {goals[2]}\n")
    head = ["jugador", "equipo", "dist. pelota", "más cerca", "patadas/min", "aprieta patear", "quieto",
            "velocidad", "del lado de su arco"]
    print(" | ".join(head))
    for pid, s in sorted(st.items()):
        n = max(s["n"], 1)
        print(f"{names.get(pid, pid)} | {'rojo' if team_of[pid] == 1 else 'azul'} | {s['dist'] / n:6.0f} | "
              f"{s['closest'] / max(n_play, 1):6.0%} | {kicks[pid] / max(minutes, 1e-9):5.1f} | {s['kick_held'] / n:6.0%} | "
              f"{s['idle'] / n:5.0%} | {s['speed'] / n:5.2f} | {s['goal_side'] / n:6.0%}")
    print("\ncuando recibió un gol (justo antes):")
    for pid, lst in conceded_pos.items():
        if lst:
            d = np.array([x[0] for x in lst])
            gs = np.mean([x[1] for x in lst])
            print(f"  {names.get(pid, pid)}: {len(lst)} goles | distancia a la pelota mediana {np.median(d):.0f} | "
                  f"estaba del lado de su arco en {gs:.0%}")


if __name__ == "__main__":
    main()
