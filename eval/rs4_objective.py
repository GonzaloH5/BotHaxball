"""Versioned sporting criteria shared by evaluations, curriculum and selection."""
from __future__ import annotations

import hashlib
import json
import math

CONTRACT = "RS4-v4-impact-1"
TRAIN_SEEDS = (51, 73, 91)
HOLDOUT_SEEDS = (137, 211, 307)
TOLERANCE = .05


def signature(report):
    """Only identical scorer, controllers and suite may share sporting scores."""
    return hashlib.sha256(json.dumps({key: report.get(key) for key in
        ("evaluation_contract", "source_fingerprint", "suite", "seeds")},
        sort_keys=True).encode()).hexdigest()


def curriculum_signature(report):
    # More games at a phase close improve precision without changing what the
    # phase teaches or invalidating a streak from the identical control suite.
    identity = {k: report.get(k) for k in ("evaluation_contract", "source_fingerprint", "seeds")}
    identity["suite"] = {k: v for k, v in report.get("suite", {}).items()
                         if k not in ("games", "functional_games")}
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def number(mapping, key):
    value = mapping.get(key)
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def no_regression_cells(report, reference=None):
    reference = reference or report.get("reference", {})
    ours = report.get("full_games", {}).get("cells", {})
    theirs = reference.get("full_games", {}).get("cells", {})
    if (not ours or set(ours) != set(theirs)
            or {row.get("learner_count") for row in ours.values()} != {1, 2, 3, 4}):
        return {"passed": False, "reason": "missing_or_different_cells", "failed_cells": []}
    failures = []
    for key, row in ours.items():
        other = theirs[key]
        a, b = number(row, "points"), number(other, "points")
        if a is None or b is None or min(row.get("games", 0), other.get("games", 0)) < 32 or a < b - TOLERANCE:
            failures.append(key)
    return {"passed": not failures, "failed_cells": failures, "tolerance": TOLERANCE}


def gates(report):
    """Absolute competence plus improvement/non-regression; saturated baselines
    cannot make competence impossible. Missing metrics always fail.
    """
    skills, baseline = report.get("skills", {}), report.get("baseline", {})
    def at_least(key, floor, gain=0.):
        a, b = number(skills, key), number(baseline, key)
        if a is None or b is None:
            return False
        # If the baseline is near the ceiling, maintaining competence suffices.
        target = max(floor, min(1., b + gain)) if b < 1 - gain else max(floor, b - .02)
        return a >= target
    conceded, base_conceded = number(skills, "defense_conceded"), number(baseline, "defense_conceded")
    risk, base_risk = number(skills, "defense_danger_fraction"), number(baseline, "defense_danger_fraction")
    defensive = (conceded is not None and base_conceded is not None
                 and conceded <= max(.02, .8 * base_conceded)
                 and at_least("defense_recovery", .6)
                 and risk is not None and base_risk is not None and risk <= base_risk + .02)
    safe = no_regression_cells(report)["passed"]
    restarts = [number(skills, k) for k in ("restart_success", "corner_success_red", "corner_success_blue")]
    return dict(restarts=all(x is not None for x in restarts) and restarts[0] >= .8 and min(restarts[1:]) >= .7,
                defense=defensive,
                attack=at_least("exit_success", .5, .05) and at_least("attack_success", .3, .05),
                integrated=safe and at_least("integrated_success", .5),
                teammates=safe and at_least("teammate_success", .3))


def eligibility(report, reference=None):
    reference = reference or report.get("reference", {})
    if report.get("evaluation_contract") != CONTRACT or signature(report) != signature(reference):
        return dict(passed=False, reason="different_contract_or_suite")
    cells = no_regression_cells(report, reference)
    a, b = report.get("skills", {}), reference.get("skills", {})
    failures = []
    for key, direction in (("restart_success", 1), ("defense_recovery", 1),
                           ("defense_conceded", -1), ("exit_success", 1), ("attack_success", 1)):
        x, y = number(a, key), number(b, key)
        if x is None or y is None or direction * (x - y) < -TOLERANCE:
            failures.append(key)
    return dict(passed=cells["passed"] and not failures, cells=cells, failed_skills=failures)


def selection_score(report):
    """Worst single-instance cell first; global means cannot mask that use case."""
    cells = report["full_games"]["cells"]
    single = [r["points"] for r in cells.values() if r["learner_count"] == 1]
    return (min(single), report["full_games"]["balanced_points"],
            report["functional"]["mean_success"])
