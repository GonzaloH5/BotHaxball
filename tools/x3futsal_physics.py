"""Compare the Futsal x3 simulator against target-stadium replay spans.

Run with:
    python -m tools.x3futsal_physics --limit 12
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from sim.stadium import load_stadium
from tools.x3futsal_audit import DEFAULT_OUT, ROOT, _span_for

REPORT = DEFAULT_OUT / "physics.json"
EXCLUDED_EVENTS = {"disc_props", "positions_reset", "game_start", "game_stop", "stadium_change", "goal", "pause",
                   "time_up"}
PLAYER_DISC_PROPERTIES = {6: "radius", 7: "bCoef", 8: "invMass", 9: "damping"}


def transition_is_valid(previous: dict, current: dict, span: dict, event_frames: set[int],
                        check_coincident: bool = True, allowed_states: tuple[int, ...] = (1,)) -> bool:
    if current["frame"] != previous["frame"] + 1:
        return False
    if not span["start"] <= previous["frame"] < span["end"] or not span["start"] <= current["frame"] < span["end"]:
        return False
    if previous.get("state") not in allowed_states or current.get("state") not in allowed_states \
            or previous.get("score") != current.get("score"):
        return False
    previous_players = previous.get("players", [])
    current_players = current.get("players", [])
    if len(previous_players) != 6 or len(current_players) != 6:
        return False
    roster = lambda tick: tuple(sorted((p["id"], p["team"]) for p in tick["players"]))
    if roster(previous) != roster(current):
        return False
    if check_coincident and (_has_coincident_players(previous) or _has_coincident_players(current)):
        return False
    if any((p["frame"] in event_frames) for p in (previous, current)):
        return False
    return all(d.get("x") is not None and d.get("y") is not None and d.get("vx") is not None and d.get("vy") is not None
               for tick in (previous, current) for d in tick.get("discs", []))


def _has_coincident_players(tick: dict) -> bool:
    discs = tick.get("discs", [])
    positions = []
    for player in tick.get("players", []):
        disc_index = player.get("disc")
        if disc_index is None or disc_index >= len(discs):
            continue
        disc = discs[disc_index]
        if disc.get("x") is not None and disc.get("y") is not None:
            positions.append((disc["x"], disc["y"]))
    positions.sort()
    for i, (x1, y1) in enumerate(positions[:-1]):
        for x2, y2 in positions[i + 1:]:
            dx = x2 - x1
            if dx * dx > 1e-12:
                break
            if (y1 - y2) ** 2 <= 1e-12 - dx * dx:
                return True
    return False


def _read_replay(jsonl_path: Path, spans: list[dict], default_player_properties: dict):
    events = Counter()
    event_frames = set()
    event_names = defaultdict(set)
    disc_property_fields = Counter()
    disc_property_updates = 0
    player_properties = {}
    player_property_override_ticks = 0
    player_property_override_player_ticks = 0
    overridden_properties = Counter()
    coincident_player_frames = set()
    ticks = []
    with gzip.open(jsonl_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            if item.get("type") == "tick":
                if _span_for(int(item["frame"]), spans):
                    ticks.append(item)
                    if _has_coincident_players(item):
                        coincident_player_frames.add(int(item["frame"]))
                    if item.get("state") == 1:
                        overridden_players = []
                        for player in item.get("players", []):
                            properties = player_properties.get(player.get("id"), {})
                            overridden = [name for name, value in properties.items()
                                          if value != default_player_properties[name]]
                            if overridden:
                                overridden_players.append(overridden)
                        if overridden_players:
                            event_frames.add(int(item["frame"]))
                            player_property_override_ticks += 1
                            player_property_override_player_ticks += len(overridden_players)
                            for properties in overridden_players:
                                overridden_properties.update(properties)
            elif item.get("type") == "event":
                frame = int(item.get("frame", -1))
                if item.get("name") == "disc_props" and item.get("kind") is True:
                    data = item.get("data1") or []
                    player_id = item.get("id")
                    if player_id is not None:
                        properties = player_properties.setdefault(player_id, {})
                        for field, name in PLAYER_DISC_PROPERTIES.items():
                            if len(data) > field and data[field] is not None:
                                properties[name] = data[field]
                if _span_for(frame, spans):
                    name = item.get("name", "unknown")
                    event_names[frame].add(name)
                    if name in EXCLUDED_EVENTS:
                        event_frames.add(frame)
                        events[name] += 1
                    if name == "disc_props":
                        disc_property_updates += 1
                        for field, value in enumerate(item.get("data1") or []):
                            if value is not None:
                                disc_property_fields[(item.get("kind"), field, value)] += 1
    property_goal_frames = sum("disc_props" in names and "goal" in names for names in event_names.values())
    for frame in coincident_player_frames:
        event_frames.update(range(frame - 2, frame + 3))
    return ticks, event_frames, events, {
        "discPropertyUpdates": disc_property_updates,
        "discPropertyFramesCoincidentWithGoal": property_goal_frames,
        "playerPhysicsOverrideTicks": player_property_override_ticks,
        "playerPhysicsOverridePlayerTicks": player_property_override_player_ticks,
        "overriddenPlayerProperties": dict(sorted(overridden_properties.items())),
        "coincidentPlayerTicks": len(coincident_player_frames),
        "discPropertyValues": [
            {"kind": kind, "fieldIndex": field, "value": value, "count": count}
            for (kind, field, value), count in sorted(disc_property_fields.items(), key=lambda row: str(row[0]))
        ],
    }, event_names, coincident_player_frames


def evaluate_recording(name: str, audit_row: dict, cache: Path) -> dict:
    from bridge.compare_sim import Replayer

    spans = audit_row.get("targetSpans", [])
    jsonl_path = cache / audit_row["jsonl"]
    stadium = load_stadium(ROOT / "stadiums" / "futsal3bybazinga.hbs")
    ticks, event_frames, excluded_events, property_summary, event_names, coincident_player_frames = _read_replay(
        jsonl_path, spans, default_player_properties=stadium.player)
    if not ticks:
        return {"source": name, "transitions": 0, "excludedEvents": dict(excluded_events),
                "propertyEvents": property_summary, "error": "no target ticks"}
    coincident_excluded_frames = {
        frame for coincident_frame in coincident_player_frames for frame in range(coincident_frame - 2,
                                                                                 coincident_frame + 3)
    }
    excluded_coincident_transitions = sum(
        previous["frame"] in coincident_excluded_frames or current["frame"] in coincident_excluded_frames
        for previous, current in zip(ticks, ticks[1:])
    )
    replayer = None
    previous_segment_key = None
    ball_errors = []
    player_errors = []
    disc_errors = []
    worst = []
    transitions = 0
    skipped_roster = Counter()
    for previous, current in zip(ticks, ticks[1:]):
        span = _span_for(int(previous["frame"]), spans)
        if span is None:
            continue
        if previous.get("state") == 1 and current.get("state") == 1:
            counts = Counter(p["team"] for p in previous.get("players", []))
            if counts != {1: 3, 2: 3}:
                skipped_roster[f"{counts.get(1, 0)}v{counts.get(2, 0)}"] += 1
                continue
        if not transition_is_valid(previous, current, span, event_frames):
            continue
        segment_key = tuple(sorted((p["id"], p["team"]) for p in previous["players"]))
        if segment_key != previous_segment_key or replayer is None:
            replayer = Replayer(stadium, [previous, current])
            previous_segment_key = segment_key
        replayer.load(previous)
        replayer.step(previous)
        sim_pos = replayer.sim.pos[0].copy()
        real_pos, _ = replayer.real(current)
        if sim_pos.shape != real_pos.shape or not np.isfinite(sim_pos).all() or not np.isfinite(real_pos).all():
            continue
        errors = np.linalg.norm(sim_pos - real_pos, axis=1)
        ball_errors.append(float(errors[0]))
        player_offset = int(np.argmax(errors[replayer.sim.first_player:]))
        player_errors.append(float(errors[replayer.sim.first_player + player_offset]))
        disc_errors.append(float(errors.max()))
        max_player = replayer.players[player_offset]
        worst.append({
            "frame": int(previous["frame"]),
            "ballErrorPx": float(errors[0]),
            "maxPlayerErrorPx": float(errors[replayer.sim.first_player + player_offset]),
            "playerId": int(max_player["id"]),
            "playerTeam": int(max_player["team"]),
            "input": int(max_player["input"]),
            "nearbyEvents": [
                {"frame": frame, "names": sorted(names)}
                for frame, names in sorted(event_names.items())
                if previous["frame"] - 2 <= frame <= current["frame"] + 2
            ],
        })
        transitions += 1

    def stats(values: list[float]) -> dict:
        if not values:
            return {"count": 0, "median": None, "p90": None, "p99": None, "max": None,
                    "over0_5Px": 0, "over1Px": 0}
        array = np.asarray(values)
        return {"count": int(array.size), "median": float(np.median(array)), "p90": float(np.percentile(array, 90)),
                "p99": float(np.percentile(array, 99)), "max": float(array.max()),
                "over0_5Px": int(np.count_nonzero(array > 0.5)), "over1Px": int(np.count_nonzero(array > 1.0))}

    return {
        "source": name,
        "targetTicks": len(ticks),
        "spans": len(spans),
        "transitions": transitions,
        "threeVThreeTransitions": transitions,
        "excludedCoincidentPlayerTransitions": excluded_coincident_transitions,
        "skippedRosters": dict(sorted(skipped_roster.items())),
        "excludedEvents": dict(sorted(excluded_events.items())),
        "propertyEvents": property_summary,
        "ballErrorPx": stats(ball_errors),
        "maxPlayerErrorPx": stats(player_errors),
        "maxAnyDiscErrorPx": stats(disc_errors),
        "worstTransitions": sorted(worst, key=lambda row: max(row["ballErrorPx"], row["maxPlayerErrorPx"]),
                                   reverse=True)[:10],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=12, help="maximum deterministic replay sample size; 0 means all")
    parser.add_argument("--seed", type=int, default=20261027)
    parser.add_argument("--eventful", action="store_true", help="prioritize recordings with disc-property events")
    parser.add_argument("--output", type=Path, default=REPORT)
    args = parser.parse_args()
    audit = json.loads((args.cache / "audit.json").read_text(encoding="utf-8"))
    index = json.loads((args.cache / "index.json").read_text(encoding="utf-8"))["recordings"]
    candidates = [(name, {**row, **index[name]}) for name, row in audit["recordings"].items()
                  if row.get("status") == "ok" and row.get("targetTicks", 0) > 0]
    if args.eventful:
        candidates = [candidate for candidate in candidates if candidate[1].get("events", {}).get("disc_props", 0)]
        candidates.sort(key=lambda candidate: (-candidate[1]["events"]["disc_props"], candidate[0]))
    else:
        random.Random(args.seed).shuffle(candidates)
    if args.limit:
        candidates = candidates[:args.limit]
    results = [evaluate_recording(name, row, args.cache) for name, row in candidates]
    transitions = sum(row.get("transitions", 0) for row in results)
    ball = [row["ballErrorPx"]["p90"] for row in results if row.get("transitions")]
    players = [row["maxPlayerErrorPx"]["p90"] for row in results if row.get("transitions")]
    report = {
        "targetStadium": str(ROOT / "stadiums" / "futsal3bybazinga.hbs"),
        "targetStadiumIdentity": audit["summary"]["targetStadiumIdentity"],
        "seed": args.seed,
        "sampleSize": len(candidates),
        "sampledRecordings": [row["source"] for row in results],
        "transitions": transitions,
        "aggregate": {
            "ballErrorPx": {"recordingP90Median": float(np.median(ball)) if ball else None,
                            "recordingP90Max": float(np.max(ball)) if ball else None},
            "maxPlayerErrorPx": {"recordingP90Median": float(np.median(players)) if players else None,
                                 "recordingP90Max": float(np.max(players)) if players else None},
        },
        "recordings": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "recordings"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
