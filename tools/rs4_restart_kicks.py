"""Reconstruir el impulso y el efecto de córners y saques de arco de RS ONE (PLAN_RS4.md 2.2).

En las grabaciones, la patada que libera un córner o saque de arco sale a ~6 px/tick
(patada normal); 1-3 ticks después la velocidad salta a ~12 (córner) o ~16 (saque de
arco), y el script fija una gravedad que decae ×0,97 por tick (efecto). Este análisis
mide, por saque: velocidad de la patada, salto (magnitud, demora y dirección), gravedad
inicial y su componente a lo largo/perpendicular, tecla del pateador y lado de la cancha.

  python -m tools.rs4_restart_kicks --out reports/rs4_b1/restart_kicks.json
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np

from tools.rs4_rules_audit import load, rules

ROOT = Path(__file__).resolve().parent.parent


def input_direction(value):
    """Teclas HaxBall: 1 arriba, 2 abajo, 4 izquierda, 8 derecha (y crece hacia abajo)."""
    dx = ((value >> 3) & 1) - ((value >> 2) & 1)
    dy = ((value >> 1) & 1) - (value & 1)
    return dx, dy


def kicks_of(path):
    header, ticks, events = load(path)
    by = {t["frame"]: t for t in ticks}
    restarts, color_team = rules(ticks, events)
    out = []
    for r in restarts:
        if r["kind"] not in ("corner", "goal_kick") or r["end"] != "liberado":
            continue
        k = r["frame"] + r["duration"]
        kick = next((e for e in events.get(k, []) if e["name"] == "kick"), None)
        if kick is None:
            continue
        g0 = None
        for e in events.get(k, []):
            if e["name"] == "disc_props" and not e.get("kind") and e.get("id") == 0:
                d1 = e.get("data1") or [None] * 10
                if d1[4] is not None and (d1[4] or d1[5]):
                    g0 = (d1[4], d1[5])
        t0, t1 = by.get(k), by.get(k + 1)
        if not t0 or not t1:
            continue
        v1 = np.array([t1["discs"][0]["vx"], t1["discs"][0]["vy"]])
        jump, delay, v2 = None, None, None
        for j in range(2, 8):
            t = by.get(k + j)
            if not t:
                break
            v = np.array([t["discs"][0]["vx"], t["discs"][0]["vy"]])
            if np.linalg.norm(v) > 1.4 * np.linalg.norm(v1):
                v2, delay = v, j
                break
        player = next((p for p in t0["players"] if p["id"] == kick["playerId"]), None)
        if player is None:
            continue
        pd = t0["discs"][player["disc"]]
        ball = t0["discs"][0]
        team = player["team"]
        row = dict(kind=r["kind"], team=team, place=r["place"], v1=v1.tolist(), v1_speed=float(np.linalg.norm(v1)),
                   v2=v2.tolist() if v2 is not None else None, v2_speed=float(np.linalg.norm(v2)) if v2 is not None else None,
                   delay=delay, g0=g0, input=player["input"], input_dir=input_direction(player["input"]),
                   player_rel=(pd["x"] - ball["x"], pd["y"] - ball["y"]), player_vel=(pd["vx"], pd["vy"]))
        if v2 is not None:
            u1, u2 = v1 / (np.linalg.norm(v1) + 1e-9), v2 / np.linalg.norm(v2)
            row["v2_angle_deg"] = float(np.degrees(np.arctan2(u1[0] * u2[1] - u1[1] * u2[0], u1 @ u2)))
            row["v2_minus_v1"] = (v2 - v1).tolist()
        if g0 is not None:
            g = np.array(g0)
            ref = v2 if v2 is not None else v1
            u = ref / (np.linalg.norm(ref) + 1e-9)
            n = np.array([-u[1], u[0]])
            row["g_along"], row["g_perp"], row["g_mag"] = float(g @ u), float(g @ n), float(np.linalg.norm(g))
        out.append(row)
    return out


def _job(path):
    try:
        return kicks_of(path)
    except Exception as error:
        print(f"error {path}: {error}", flush=True)
        return []


def summarize(rows):
    result = {}
    for kind in ("corner", "goal_kick"):
        sub = [r for r in rows if r["kind"] == kind]
        boosted = [r for r in sub if r["v2_speed"] is not None]
        with_g = [r for r in sub if r.get("g_mag") is not None]
        stats = lambda values: dict(p10=float(np.percentile(values, 10)), p50=float(np.median(values)),
                                    p90=float(np.percentile(values, 90))) if len(values) else None
        result[kind] = dict(
            n=len(sub), boosted=len(boosted),
            v1_speed=stats([r["v1_speed"] for r in sub]),
            v2_speed=stats([r["v2_speed"] for r in boosted]),
            ratio=stats([r["v2_speed"] / max(r["v1_speed"], 1e-6) for r in boosted]),
            delay=collections.Counter(r["delay"] for r in boosted).most_common(5),
            angle_v2_v1_deg=stats([r["v2_angle_deg"] for r in boosted]),
            g_mag=stats([r["g_mag"] for r in with_g]),
            g_along=stats([r["g_along"] for r in with_g]),
            g_perp=stats([r["g_perp"] for r in with_g]),
            with_gravity=len(with_g))
        # ¿El signo del efecto depende de la tecla del pateador o del lado de la cancha?
        signs = collections.Counter()
        for r in with_g:
            side = (int(np.sign(r["place"][0])), int(np.sign(r["place"][1])))
            signs[(str(r["input_dir"]), str(side), "perp+" if r["g_perp"] > 0 else "perp-")] += 1
        result[kind]["perp_sign_by_input_and_side"] = [dict(input=k[0], side=k[1], sign=k[2], n=v)
                                                       for k, v in signs.most_common(24)]
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default=str(ROOT / "data" / "rs4_jsonl"))
    ap.add_argument("--out", default=str(ROOT / "reports" / "rs4_b1" / "restart_kicks.json"))
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    cache = Path(a.cache)
    index = json.loads((cache / "index.json").read_text(encoding="utf-8"))
    paths = [str(cache / row["jsonl"]) for row in index["recordings"].values() if row.get("status") == "ok"
             and [s.get("name") for s in row.get("stadiums") or []] == ["Real Soccer ONE"]]
    rows = []
    if a.workers > 1:
        from multiprocessing import get_context
        with get_context("spawn").Pool(a.workers) as pool:
            for part in pool.imap_unordered(_job, paths):
                rows += part
    else:
        for path in paths:
            rows += _job(path)
    result = dict(summary=summarize(rows), kicks=rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    print(json.dumps(result["summary"], indent=1, default=str))


if __name__ == "__main__":
    main()
