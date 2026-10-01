"""Elige entre los dos pilotos con desempate conservador hacia feedforward."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def summary(report):
    rows = list(report["candidate"]["r3_aggregate"].values())
    games = sum(row["games"] for row in rows)
    wins = sum(row["wins"] for row in rows)
    draws = sum(row["draws"] for row in rows)
    points = (wins + .5 * draws) / max(games, 1)
    second = (wins + .25 * draws) / max(games, 1)
    points_ci = 1.96 * math.sqrt(max(0.0, second - points ** 2) / max(games, 1))
    progressive = sum(row["events"]["progressive_passes"] for row in rows) / max(
        sum(row["games"] for row in rows) * report.get("minutes", 3), 1)
    # Al estar ambos informes bajo el mismo protocolo, comparar tasas medias es estable.
    return {"points": points, "points_ci95": points_ci, "progressive_passes_per_minute": progressive,
            "corner_useful_fraction": report["restarts"]["corner_useful_fraction"]}


def choose(feedforward, recurrent):
    a, b = summary(feedforward), summary(recurrent)
    if abs(a["points"] - b["points"]) > a["points_ci95"] + b["points_ci95"]:
        return ("feedforward" if a["points"] > b["points"] else "recurrent"), "puntos contra R3", a, b
    # Las tasas conductuales no son Bernoulli independientes; exigimos 10% de
    # separación para no convertir ruido de pocos eventos en una elección.
    pa, pb = a["progressive_passes_per_minute"], b["progressive_passes_per_minute"]
    if abs(pa - pb) > .10 * max(pa, pb, 1e-9):
        return ("feedforward" if pa > pb else "recurrent"), "pases progresivos", a, b
    ca, cb = a["corner_useful_fraction"], b["corner_useful_fraction"]
    if abs(ca - cb) > .05:
        return ("feedforward" if ca > cb else "recurrent"), "córners útiles", a, b
    return "feedforward", "empate estadístico: arquitectura simple", a, b


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("feedforward_report")
    ap.add_argument("recurrent_report")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ff = json.loads(Path(args.feedforward_report).read_text(encoding="utf-8"))
    rec = json.loads(Path(args.recurrent_report).read_text(encoding="utf-8"))
    if ff["protocol_id"] != rec["protocol_id"]:
        raise ValueError("Los pilotos no fueron evaluados bajo el mismo protocolo")
    selected, reason, a, b = choose(ff, rec)
    result = {"selected": selected, "reason": reason, "feedforward": a, "recurrent": b,
              "full_config": f"train/config_coordination_{selected}_full.yaml"}
    Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()

