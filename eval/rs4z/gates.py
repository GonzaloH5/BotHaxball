"""Compuertas de etapa RS4-Z: deciden si el curriculum avanza. Nunca usan la recompensa media.

Los umbrales relativos a RS-Pro L5 salen de `reports/rs4z/gates.json`, generado ANTES de entrenar por
`tools/rs4z_calibrate_gates.py` (pre-registro). Si el archivo falta, la evaluación se niega a correr.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from bots.rspro.controller import RSProController
from bots.rspro.policy import sample_style

from .batteries import run_battery
from .net_controller import NetController
from .runner import play, points

ROOT = Path(__file__).resolve().parent.parent.parent
GATES_PATH = ROOT / "reports" / "rs4z" / "gates.json"

# Reglas (pre-registradas): fracción de la referencia L5 o valor absoluto.
RULES = {
    "control_battery": dict(mean_frac=0.90, task_frac=0.75),
    "duel_battery": dict(mean_frac=0.85, task_frac=0.65, conceded_slack=1.3),
    "passing_battery": dict(mean_frac=0.80, task_frac=0.60),
    "defense_battery": dict(mean_frac=0.80, task_frac=0.60, conceded_slack=1.1),
    "match_1v1_vs_l3": dict(points=0.55, level=3, active=(1, 1)),
    "detectors_small": dict(level=3, active=(2, 2), parked=0.05, goal_hug=0.05, clump=0.12),
    "match_4v4_vs_l5": dict(points=0.60, level=5),
    "reserved_style": dict(points=0.50, level=5),
    "restart_battery": dict(max_forfeit_rate=0.02),
    "detectors_4v4": dict(parked=0.05, goal_hug=0.05, clump=0.10),
    "league_milestone": dict(points_l5=0.75, points_reserved=0.65),
    "robustness": dict(max_drop=0.10, latency=8),
}
STAGE_GATES = {
    "S1": ("control_battery",),
    "S2": ("duel_battery", "match_1v1_vs_l3"),
    "S3": ("passing_battery",),
    "S4": ("defense_battery", "detectors_small"),
    "S5": ("match_4v4_vs_l5", "reserved_style", "restart_battery", "detectors_4v4"),
    "S6": ("league_milestone",),
    "S7": ("robustness",),
}


def _active(n_red, n_blue):
    a = np.zeros(8, dtype=bool)
    a[:n_red] = True
    a[4:4 + n_blue] = True
    return a


def _vs_rspro(cand, level, games, seed, active=None, reserved=False, latency=0):
    """Candidato contra RS-Pro (estilos de entrenamiento o reservados), ambos colores. Puntos y métricas."""
    rng = np.random.default_rng(seed)
    half = games // 2
    pts, events, metrics = [], dict(forfeits=0.0, restarts=0.0), []
    for flip in (False, True):
        levels = np.full((half, 2), level)
        styles = np.stack([[sample_style(rng, reserved=reserved) for _ in range(2)] for _ in range(half)])
        bot = RSProController(levels=levels, styles=styles, seed=seed + int(flip))
        red, blue = (bot, cand) if flip else (cand, bot)
        act = None if active is None else np.broadcast_to(active, (half, 8))
        delay = None if latency == 0 else np.full((half, 8), latency)
        r = play(red, blue, n_envs=half, match_ticks=3 * 3600, seed=seed + 11 * int(flip), active=act, delay=delay)
        p = points(r["score"])
        pts.append(1 - p if flip else p)
        events["forfeits"] += float(r["events"]["forfeits"].sum())
        metrics.append(r["metrics"].summary(team=1 if flip else 0))
    pts = np.concatenate(pts)
    rng2 = np.random.default_rng(seed)
    boots = pts[rng2.integers(0, len(pts), (2000, len(pts)))].mean(axis=1)
    m = {k: float(np.mean([x[k] for x in metrics])) for k in metrics[0]}
    return dict(points=float(pts.mean()), ci90=(float(np.percentile(boots, 5)), float(np.percentile(boots, 95))),
                games=int(len(pts)), forfeits=events["forfeits"], metrics=m)


def evaluate_gate(name, cand, calib, episodes=256, seed=0):
    rule = RULES[name]
    if name.endswith("_battery") and name != "restart_battery":
        res = run_battery(cand, name, episodes=episodes, seed=seed)
        ref = calib[name]
        # sólo cuentan las tareas que la referencia resuelve (≥ 15%): una referencia ~0 no informa
        judged = [t for t in res if ref[t]["success"] >= 0.15]
        mean = float(np.mean([res[t]["success"] for t in judged])) if judged else float("nan")
        ref_mean = float(np.mean([ref[t]["success"] for t in judged])) if judged else float("nan")
        ok = bool(judged) and mean >= rule["mean_frac"] * ref_mean
        per_task = {}
        for t, v in res.items():
            per_task[t] = (v["success"] >= rule["task_frac"] * ref[t]["success"]) if t in judged else None
            if t in judged:
                ok &= per_task[t]
            if "conceded_slack" in rule and not np.isnan(ref[t]["conceded"]):
                ok &= v["conceded"] <= rule["conceded_slack"] * ref[t]["conceded"] + 0.02
        return dict(passed=bool(ok), value=mean, reference=ref_mean, tasks=res, per_task=per_task)
    if name == "match_1v1_vs_l3":
        r = _vs_rspro(cand, rule["level"], episodes // 2, seed, active=_active(*rule["active"]))
        return dict(passed=r["points"] >= rule["points"] and r["ci90"][0] > 0.5, **r)
    if name == "detectors_small":
        r = _vs_rspro(cand, rule["level"], episodes // 2, seed, active=_active(*rule["active"]))
        m = r["metrics"]
        ok = m["parked"] <= rule["parked"] and m["goal_hug"] <= rule["goal_hug"] and m["clump"] <= rule["clump"]
        return dict(passed=bool(ok), **r)
    if name in ("match_4v4_vs_l5", "reserved_style"):
        r = _vs_rspro(cand, rule["level"], episodes // 2, seed, reserved=name == "reserved_style")
        return dict(passed=r["points"] >= rule["points"] and r["ci90"][0] > 0.5, **r)
    if name == "restart_battery":
        r = _vs_rspro(cand, 4, episodes // 4, seed)
        restarts = max(1.0, r["metrics"]["minutes"] * 6.0)   # ~6 saques por minuto de partido
        rate = r["forfeits"] / restarts
        return dict(passed=rate <= rule["max_forfeit_rate"], forfeit_rate=rate, **r)
    if name == "detectors_4v4":
        r = _vs_rspro(cand, 4, episodes // 2, seed)
        m = r["metrics"]
        ok = m["parked"] <= rule["parked"] and m["goal_hug"] <= rule["goal_hug"] and m["clump"] <= rule["clump"]
        return dict(passed=bool(ok), **r)
    if name == "league_milestone":
        a = _vs_rspro(cand, 5, episodes // 2, seed)
        b = _vs_rspro(cand, 5, episodes // 2, seed + 1, reserved=True)
        ok = a["points"] >= rule["points_l5"] and b["points"] >= rule["points_reserved"]
        return dict(passed=bool(ok), l5=a, reserved=b)
    if name == "robustness":
        a = _vs_rspro(cand, 5, episodes // 2, seed)
        b = _vs_rspro(cand, 5, episodes // 2, seed, latency=rule["latency"])
        return dict(passed=a["points"] - b["points"] <= rule["max_drop"], host=a, client=b)
    raise KeyError(name)


def evaluate_stage(stage, model, episodes=256, seed=0):
    if not GATES_PATH.exists():
        raise FileNotFoundError("falta reports/rs4z/gates.json: correr tools/rs4z_calibrate_gates.py antes de entrenar")
    calib = json.loads(GATES_PATH.read_text(encoding="utf-8"))["reference"]
    cand = NetController(model, greedy=True)
    details = {}
    passed = True
    for gate in STAGE_GATES[stage]:
        res = evaluate_gate(gate, cand, calib, episodes=episodes, seed=seed)
        details[gate] = res
        passed &= bool(res["passed"])
    summary = {g: (round(d.get("value", d.get("points", float("nan"))), 3), d["passed"]) for g, d in details.items()
               if "value" in d or "points" in d}
    summary.update({g: d["passed"] for g, d in details.items() if g not in summary})
    return dict(stage=stage, passed=bool(passed), summary=summary, details=_jsonable(details))


def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    return x
