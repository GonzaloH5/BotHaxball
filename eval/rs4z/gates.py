"""Compuertas de etapa RS4-Z: deciden si el curriculum avanza. Nunca usan la recompensa media.

Los umbrales relativos a RS-Pro L5 salen de `reports/rs4z/gates.json`, generado ANTES de entrenar por
`tools/rs4z_calibrate_gates.py` (pre-registro). Si el archivo falta, la evaluación se niega a correr.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from bots.rspro.controller import RSProController
from bots.rspro.policy import sample_style

from env.rs4z import contract as C

from .batteries import run_battery
from .net_controller import NetController
from .runner import BLUE, RED, RESTART_KINDS, play, points

ROOT = Path(__file__).resolve().parent.parent.parent
GATES_PATH = ROOT / "reports" / "rs4z" / "gates.json"
# archivos que fijan el significado de la calibración: si cambian, hay que recalibrar (experimento nuevo)
CALIBRATED_CODE = ["env/rs4z/kernel.py", "env/rs4z/core.py", "env/rs4z/drills.py", "env/rs4z/contract.py",
                   "bots/rspro/brain.py", "bots/rspro/policy.py", "bots/rspro/geom.py", "eval/rs4z/batteries.py",
                   "eval/rs4z/gates.py", "eval/rs4z/metrics.py", "eval/rs4z/runner.py"]


def code_hash():
    h = hashlib.sha256()
    for f in CALIBRATED_CODE:
        h.update((ROOT / f).read_bytes().replace(b"\r\n", b"\n"))   # igual en Windows y en el Pod
    return h.hexdigest()[:16]


# medianas humanas de duración de saques (ticks; `reports/rs4z/human_reference.json`, 69 grabaciones)
HUMAN_RESTART_P50 = dict(lateral=117.0, corner=153.0, goal_kick=249.0)

# Reglas (pre-registradas): fracción de la referencia L5 o valor absoluto.
RULES = {
    "control_battery": dict(mean_frac=0.90, task_frac=0.75),
    "duel_battery": dict(mean_frac=0.85, task_frac=0.65, conceded_slack=1.3),
    "passing_battery": dict(mean_frac=0.80, task_frac=0.60),
    "defense_battery": dict(mean_frac=0.80, task_frac=0.60, conceded_slack=1.1),
    # RS-Pro ataca mal en 1v1 (auditoría §5): contra L3 la compuerta no informa, se exige contra L5
    "match_1v1_vs_l5": dict(points=0.55, level=5, active=(1, 1)),
    "detectors_small": dict(level=3, active=(2, 2), parked=0.05, goal_hug=0.05, clump=0.12),
    "match_4v4_vs_l5": dict(points=0.60, level=5),
    "reserved_style": dict(points=0.50, level=5),
    # saques PROPIOS del candidato: ≥ 98% ejecutados antes del plazo de entrenamiento (600 ticks) y medianas
    # por tipo ≤ 2× la mediana humana (laterales 117, córners 153, saques de arco 249 ticks)
    "restart_battery": dict(max_late_rate=0.02, late_ticks=600, median_factor=2.0),
    "detectors_4v4": dict(parked=0.05, goal_hug=0.05, clump=0.10),
    "league_milestone": dict(points_l5=0.75, points_reserved=0.65, max_style_gap=0.10, parked=0.05, goal_hug=0.05,
                             clump=0.10),
    # latencia de cliente sólo en el candidato (RS-Pro sin retraso) y variante del mapa; sin piso absoluto
    # una política débil la aprobaría por no tener nada que perder
    "robustness": dict(max_drop=0.10, latency=8, floor_l5=0.70),
}
STAGE_GATES = {
    "S1": ("control_battery",),
    "S2": ("duel_battery", "match_1v1_vs_l5"),
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


def _vs_rspro(cand, level, games, seed, active=None, reserved=False, latency=0, variant=False):
    """Candidato contra RS-Pro (estilos de entrenamiento o reservados), ambos colores. Puntos y métricas.

    latency: retraso (ticks) SÓLO del candidato. variant: kickStrength 5,75 y pelota de radio 8.
    """
    rng = np.random.default_rng(seed)
    half = games // 2
    pts, events, metrics = [], dict(forfeits=0.0, restarts=0.0), []
    own_restarts = {k: [] for k in RESTART_KINDS.values()}
    for flip in (False, True):
        levels = np.full((half, 2), level)
        styles = np.stack([[sample_style(rng, reserved=reserved) for _ in range(2)] for _ in range(half)])
        bot = RSProController(levels=levels, styles=styles, seed=seed + int(flip))
        red, blue = (bot, cand) if flip else (cand, bot)
        act = None if active is None else np.broadcast_to(active, (half, 8))
        delay = None
        if latency:
            delay = np.zeros((half, 8), dtype=np.int64)
            delay[:, BLUE if flip else RED] = latency
        kw = dict(kick_strength=C.KICK_STRENGTHS[1], ball_radius=C.BALL_RADII[1]) if variant else {}
        r = play(red, blue, n_envs=half, match_ticks=3 * 3600, seed=seed + 11 * int(flip), active=act, delay=delay,
                 **kw)
        p = points(r["score"])
        pts.append(1 - p if flip else p)
        events["forfeits"] += float(r["events"]["forfeits"].sum())
        metrics.append(r["metrics"].summary(team=1 if flip else 0))
        for k, v in r["restarts"].team(1 if flip else 0).items():
            own_restarts[k] += v
    pts = np.concatenate(pts)
    rng2 = np.random.default_rng(seed)
    boots = pts[rng2.integers(0, len(pts), (2000, len(pts)))].mean(axis=1)
    m = {k: float(np.mean([x[k] for x in metrics])) for k in metrics[0]}
    return dict(points=float(pts.mean()), ci90=(float(np.percentile(boots, 5)), float(np.percentile(boots, 95))),
                games=int(len(pts)), forfeits=events["forfeits"], metrics=m, restarts=own_restarts)


def head_to_head(model_a, model_b, games=128, seed=0, greedy=False):
    """Puntos de la red A contra la red B en partidos 4v4 de 3 min, ambos colores, con IC90 por bootstrap."""
    half = max(1, games // 2)
    pts = []
    for flip in (False, True):
        a = NetController(model_a, greedy=greedy, seed=seed + 2 * int(flip))
        b = NetController(model_b, greedy=greedy, seed=seed + 2 * int(flip) + 1)
        red, blue = (b, a) if flip else (a, b)
        r = play(red, blue, n_envs=half, match_ticks=3 * 3600, seed=seed + 11 * int(flip), on_stall="end")
        p = points(r["score"])
        pts.append(1 - p if flip else p)
    pts = np.concatenate(pts)
    boots = pts[np.random.default_rng(seed).integers(0, len(pts), (2000, len(pts)))].mean(axis=1)
    return float(pts.mean()), (float(np.percentile(boots, 5)), float(np.percentile(boots, 95)))


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
    if name == "match_1v1_vs_l5":
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
        own = r.pop("restarts")
        allv = np.concatenate([np.asarray(v, dtype=np.float64) for v in own.values()] + [np.zeros(0)])
        late = float((allv >= rule["late_ticks"]).mean()) if allv.size else 1.0
        medians = {k: (float(np.median(v)) if v else float("nan")) for k, v in own.items()}
        slow = {k: medians[k] > rule["median_factor"] * h for k, h in HUMAN_RESTART_P50.items()
                if not np.isnan(medians[k])}
        ok = allv.size >= 20 and late <= rule["max_late_rate"] and not any(slow.values())
        return dict(passed=bool(ok), value=1.0 - late, late_rate=late, n_restarts=int(allv.size), medians=medians,
                    slow=slow, **r)
    if name == "detectors_4v4":
        r = _vs_rspro(cand, 4, episodes // 2, seed)
        m = r["metrics"]
        ok = m["parked"] <= rule["parked"] and m["goal_hug"] <= rule["goal_hug"] and m["clump"] <= rule["clump"]
        return dict(passed=bool(ok), **r)
    if name == "league_milestone":
        a = _vs_rspro(cand, 5, episodes // 2, seed)
        b = _vs_rspro(cand, 5, episodes // 2, seed + 1, reserved=True)
        m = a["metrics"]
        detectors = m["parked"] <= rule["parked"] and m["goal_hug"] <= rule["goal_hug"] and m["clump"] <= rule["clump"]
        ok = (a["points"] >= rule["points_l5"] and b["points"] >= rule["points_reserved"]
              and a["points"] - b["points"] <= rule["max_style_gap"] and detectors)
        return dict(passed=bool(ok), value=a["points"], l5=a, reserved=b, detectors=bool(detectors))
    if name == "robustness":
        a = _vs_rspro(cand, 5, episodes // 2, seed)
        b = _vs_rspro(cand, 5, episodes // 2, seed, latency=rule["latency"])
        c = _vs_rspro(cand, 5, episodes // 2, seed, variant=True)
        ok = (a["points"] >= rule["floor_l5"] and a["points"] - b["points"] <= rule["max_drop"]
              and a["points"] - c["points"] <= rule["max_drop"])
        return dict(passed=bool(ok), value=a["points"], host=a, client=b, variant=c)
    raise KeyError(name)


def evaluate_stage(stage, model, episodes=256, seed=0):
    if not GATES_PATH.exists():
        raise FileNotFoundError("falta reports/rs4z/gates.json: correr tools/rs4z_calibrate_gates.py antes de entrenar")
    frozen = json.loads(GATES_PATH.read_text(encoding="utf-8"))
    if frozen.get("code_hash") != code_hash():
        raise RuntimeError(f"reports/rs4z/gates.json se calibró con otro código ({frozen.get('code_hash')} ≠ "
                           f"{code_hash()}): recalibrar con tools/rs4z_calibrate_gates.py")
    calib = frozen["reference"]
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
