"""Reproducible one-tick RS4 physics/map audit from local recordings.

Script-mutated transitions are excluded from pure-physics comparisons and
reported, not guessed. This command never changes simulation dynamics.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import numpy as np

from bridge.compare_sim import Replayer, real_stadium
from sim.physics import KICK_REACH
from sim.stadium import load_stadium
from tools.build_bc_dataset import ROOT, catalog_by_name
from tools.build_rs4_sequences import iter_ticks


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def audit_jsonl(path, max_samples=2000):
    path = Path(path)
    script_frames = set()
    event_counts = Counter()
    for item in iter_ticks(path):
        if item.get("type") == "event":
            event_counts[item.get("name", "unknown")] += 1
            if item.get("name") == "disc_props":
                script_frames.add(item["frame"])
    cat = catalog_by_name()
    catalog = load_stadium("rs_one")
    previous, replay, stadium = None, None, None
    errors, contacts = defaultdict(list), Counter()
    exclusions, maps = Counter(), []
    for item in iter_ticks(path):
        if item.get("type") == "header" or item.get("name") == "stadium_change":
            previous, replay, stadium = None, None, None
            meta = cat.get((item.get("stadium") or "").strip())
            if meta and meta["stadium"] == "rs_one" and item.get("stadiumFile"):
                hbs = path.parent / item["stadiumFile"]
                stadium = real_stadium(hbs)
                differences = {}
                for field in ("ball", "player", "goal_x", "goal_half_height", "spawn_distance"):
                    actual, expected = getattr(stadium, field), getattr(catalog, field)
                    if actual != expected:
                        differences[field] = {"replay": actual, "catalog": expected}
                maps.append({"path": str(hbs), "sha256": digest(hbs), "differences": differences})
            continue
        if item.get("type") != "tick" or stadium is None:
            continue
        roster = lambda t: tuple(sorted((p["id"], p["team"], p["disc"]) for p in t["players"]))
        players = item.get("players", [])
        if len(players) != 8 or sum(p["team"] == 1 for p in players) != 4:
            previous = None
            exclusions["not_4v4"] += 1
            continue
        if previous is None:
            previous = item
            continue
        if (item["frame"] != previous["frame"] + 1 or roster(item) != roster(previous)
                or item.get("score") != previous.get("score") or item["state"] not in (0, 1)
                or previous["state"] != item["state"]):
            previous, replay = item, None
            exclusions["boundary"] += 1
            continue
        if item["frame"] in script_frames or previous["frame"] in script_frames:
            previous = item
            exclusions["script_mutation"] += 1
            continue
        label = "kickoff" if item["state"] == 0 else "open_play"
        if len(errors[label]) >= max_samples:
            previous = item
            continue
        try:
            if replay is None:
                replay = Replayer(stadium, [previous])
            replay.load(previous)
            replay.step(previous)
            actual, actual_velocity = replay.real(item)
            if not np.isfinite(actual).all() or not np.isfinite(replay.sim.pos).all():
                raise ValueError("nonfinite discs")
            position_error = np.linalg.norm(replay.sim.pos[0] - actual, axis=-1)
            velocity_error = np.linalg.norm(replay.sim.vel[0] - actual_velocity, axis=-1)
            fp = replay.sim.first_player
            errors[label].append([float(position_error[0]), float(position_error[fp:].max()),
                                  float(velocity_error[0]), float(velocity_error[fp:].max())])
            real_touch = (np.linalg.norm(actual[fp:] - actual[0], axis=-1)
                          < stadium.player["radius"] + stadium.ball["radius"] + KICK_REACH)
            contacts["transitions_with_contact"] += int(real_touch.any())
            contacts["contact_disagreements"] += int((real_touch != replay.sim.touch[0]).any())
        except (ValueError, IndexError, KeyError, SystemExit):
            exclusions["unsupported_layout_or_missing_state"] += 1
            replay = None
        previous = item
    summary = {}
    for label, rows in errors.items():
        array = np.asarray(rows)
        summary[label] = {"samples": len(rows), "position_ball_p90": float(np.percentile(array[:, 0], 90)),
                          "position_players_p90": float(np.percentile(array[:, 1], 90)),
                          "velocity_ball_p90": float(np.percentile(array[:, 2], 90)),
                          "velocity_players_p90": float(np.percentile(array[:, 3], 90))}
    pure = summary.get("open_play", {})
    gate = pure.get("samples", 0) >= 50 and pure.get("position_ball_p90", 1) <= .05 and pure.get("position_players_p90", 1) <= .05
    return {"source": str(path), "sha256": digest(path), "maps": maps, "physics": summary,
            "physics_tested_map": "exact_recorded_stadium", "catalog_fidelity_verified": False,
            "physics_gate_passed": gate, "contacts": dict(contacts), "events": dict(event_counts),
            "excluded": dict(exclusions), "unknown": ["Exact restart ownership and protection timing cannot be inferred from keyboard inputs alone.",
            "Room-script disc mutations are excluded, not replicated; no simulator correction is made without a reproduced discrepancy.",
            "Contact comparison is geometric and may differ from tick-internal contact timing."]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default=str(ROOT / "replays_real" / "stadiums" / "rsx4"))
    ap.add_argument("--out", default=str(ROOT / "reports" / "rs4_v3_simulator_audit.json"))
    ap.add_argument("--limit", type=int, default=3)
    ap.add_argument("--max-samples", type=int, default=2000)
    args = ap.parse_args()
    files = sorted([*Path(args.source).rglob("*.hbr2"), *Path(args.source).rglob("*.jsonl")])
    seen, reports = set(), []
    for path in files:
        checksum = digest(path)
        if checksum in seen:
            continue
        seen.add(checksum)
        if path.suffix == ".jsonl":
            report = audit_jsonl(path, args.max_samples)
        else:
            with tempfile.TemporaryDirectory(prefix="rs4-audit-") as temp:
                subprocess.run(["node", str(ROOT / "bridge" / "replay_to_jsonl.js"), str(path),
                                "--out", temp, "--max-minutes", "10"], check=True, cwd=ROOT)
                report = audit_jsonl(next(Path(temp).glob("*.jsonl")), args.max_samples)
                report["source"], report["source_sha256"] = str(path), checksum
        reports.append(report)
        print(f"{path.name}: physics gate {report['physics_gate_passed']}", flush=True)
        if len(reports) >= args.limit:
            break
    if not reports:
        ap.error("No local RS4 recordings available")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"contract_version": "RS4-v3-physics-audit-1", "dynamics_changed": False,
                               "recordings": reports}, indent=2), encoding="utf-8")
    print(out.resolve())


if __name__ == "__main__":
    main()
