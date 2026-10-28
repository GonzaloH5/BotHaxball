"""Convert and audit Futsal x3 replays, filter target-stadium spans, and create replay-level splits.

Run with:
    python -m tools.x3futsal_audit --workers 2
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np

from sim.stadium import load_stadium
from tools.rs4_jsonl_cache import convert, sha256

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "replays_real" / "stadiums" / "x3futsal"
STADIUM = ROOT / "stadiums" / "futsal3bybazinga.hbs"
DEFAULT_OUT = ROOT / "data" / "x3futsal_jsonl"
SPLIT_SEED = 20261027


def _canonical_value(value):
    if isinstance(value, np.ndarray):
        return _canonical_value(value.tolist())
    if isinstance(value, np.generic):
        return _canonical_value(value.item())
    if isinstance(value, dict):
        return {key: _canonical_value(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        rounded = round(float(value), 9)
        return 0.0 if rounded == 0 else rounded
    return value


def stadium_identity(path: Path) -> str:
    """Hash normalized stadium physics, resolving traits and ignoring display name."""
    stadium = load_stadium(path)
    content = {key: _canonical_value(value) for key, value in vars(stadium).items() if key != "name"}
    canonical = json.dumps(content, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def target_spans(stadiums: list[dict], target_id: str, max_frame: int | None,
                 last_frame: int | None) -> list[dict]:
    spans = []
    for i, stadium in enumerate(stadiums):
        start = stadium.get("frame")
        if start is None:
            continue
        end = next((item.get("frame") for item in stadiums[i + 1:] if item.get("frame") is not None), None)
        if end is None:
            final_frame = max(value for value in (max_frame, last_frame) if value is not None) \
                if max_frame is not None or last_frame is not None else None
            end = final_frame + 1 if final_frame is not None else None
        if end is None or end <= start:
            continue
        if stadium.get("identity") == target_id:
            spans.append({"start": int(start), "end": int(end), "name": stadium.get("name"),
                          "file": stadium.get("file"), "identity": stadium["identity"]})
    return spans


def _span_for(frame: int, spans: list[dict]) -> dict | None:
    return next((span for span in spans if span["start"] <= frame < span["end"]), None)


def is_complete_recording(log: str, last_frame: int | None, max_frame: int | None) -> bool:
    if "límite de minutos" in log or "timeout de seguridad" in log:
        return False
    if "fin del replay" in log:
        return True
    return last_frame is not None and max_frame is not None and last_frame >= int(max_frame) - 2


def audit_recording(name: str, row: dict, out: Path, target_id: str) -> dict:
    jsonl_path = out / row["jsonl"]
    stadiums = row.get("stadiums") or []
    last_frame = row.get("lastFrame")
    folder = jsonl_path.parent
    for stadium in stadiums:
        filename = stadium.get("file")
        if not filename:
            stadium["identity"] = None
            stadium["identityError"] = "missing exported stadium file"
            continue
        try:
            stadium["identity"] = stadium_identity(folder / filename)
        except (OSError, ValueError, KeyError, TypeError) as error:
            stadium["identity"] = None
            stadium["identityError"] = f"{type(error).__name__}: {error}"
    spans = target_spans(stadiums, target_id, row.get("maxFrameNo"), last_frame)

    ticks = 0
    selected_ticks = 0
    team_counts = Counter()
    state_counts = Counter()
    actions = Counter()
    player_ids = {1: set(), 2: set()}
    events = Counter()
    goals_by_team = Counter()
    span_ticks = Counter()
    with gzip.open(jsonl_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            frame = item.get("frame")
            if item.get("type") == "tick":
                last_frame = int(frame)
                ticks += 1
                span = _span_for(int(frame), spans)
                if span is None:
                    continue
                selected_ticks += 1
                span_ticks[(span["start"], span["end"])] += 1
                state_counts[str(item.get("state"))] += 1
                counts = {1: 0, 2: 0}
                for player in item.get("players", []):
                    team = player.get("team")
                    if team not in counts:
                        continue
                    counts[team] += 1
                    player_ids[team].add(player.get("id"))
                    actions[str(player.get("input", 0))] += 1
                team_counts[f"{counts[1]}v{counts[2]}"] += 1
            elif item.get("type") == "event" and frame is not None and _span_for(int(frame), spans):
                events[item.get("name", "unknown")] += 1
                if item.get("name") == "goal":
                    goals_by_team[str(item.get("team"))] += 1

    max_frame = row.get("maxFrameNo")
    log = row.get("log", "")
    return {
        "status": "ok",
        "sha256": row["sha256"],
        "source": row.get("source"),
        "bytes": row.get("bytes"),
        "room": row.get("room"),
        "log": log,
        "maxFrameNo": max_frame,
        "lastFrame": last_frame,
        "complete": is_complete_recording(log, last_frame, max_frame),
        "ticks": ticks,
        "targetTicks": selected_ticks,
        "excludedTicks": ticks - selected_ticks,
        "targetSpans": [{**span, "ticks": span_ticks[(span["start"], span["end"])]} for span in spans],
        "stadiums": stadiums,
        "teamCounts": dict(sorted(team_counts.items())),
        "stateCounts": dict(sorted(state_counts.items())),
        "playerIds": {str(team): sorted(pid for pid in ids if pid is not None) for team, ids in player_ids.items()},
        "inputCounts": dict(sorted(actions.items(), key=lambda pair: int(pair[0]))),
        "events": dict(sorted(events.items())),
        "goalsByTeam": dict(sorted(goals_by_team.items())),
    }


def make_splits(recordings: dict[str, dict], seed: int = SPLIT_SEED) -> dict:
    eligible = sorted(name for name, row in recordings.items()
                      if row.get("status") == "ok" and row.get("complete") and row.get("targetTicks", 0) > 0
                      and int(row.get("teamCounts", {}).get("3v3", row.get("targetTicks", 0))) > 0)
    groups = {}
    for name in eligible:
        room = recordings[name].get("room")
        key = f"room:{unicodedata.normalize('NFKC', room).strip().casefold()}" if room and room.strip() \
            else f"recording:{name}"
        groups.setdefault(key, []).append(name)

    weights = {
        key: sum(int(recordings[name].get("teamCounts", {}).get("3v3", recordings[name]["targetTicks"]))
                 for name in names)
        for key, names in groups.items()
    }
    ratios = {"train": 0.8, "dev": 0.1, "test": 0.1}
    split_names = tuple(ratios)
    group_names = sorted(groups)
    if len(group_names) < 3:
        raise ValueError("se necesitan al menos tres grupos de sala para crear train/dev/test sin fuga")
    total_ticks = sum(weights.values())
    counts = {
        "train": min(len(group_names) - 2, max(1, int(len(group_names) * ratios["train"]))),
        "dev": min(len(group_names) - 1, max(1, round(len(group_names) * ratios["dev"]))),
    }
    counts["test"] = len(group_names) - counts["train"] - counts["dev"]
    if counts["test"] < 1:
        counts["train"] -= 1
        counts["test"] = 1
    slots = [split for split in split_names for _ in range(counts[split])]
    rng = random.Random(seed)

    def split_cost(sums: dict[str, int]) -> float:
        return sum((sums[split] / total_ticks - ratios[split]) ** 2 for split in split_names)

    best_cost = float("inf")
    best_assignment = None
    restarts = 256
    for _ in range(restarts):
        order = group_names.copy()
        rng.shuffle(order)
        assignment = dict(zip(order, slots))
        sums = {split: sum(weights[key] for key, value in assignment.items() if value == split)
                for split in split_names}
        while True:
            current_cost = split_cost(sums)
            best_swap = None
            improvement = 1e-14
            for i, first in enumerate(group_names):
                first_split = assignment[first]
                for second in group_names[i + 1:]:
                    second_split = assignment[second]
                    if first_split == second_split:
                        continue
                    candidate = sums.copy()
                    candidate[first_split] += weights[second] - weights[first]
                    candidate[second_split] += weights[first] - weights[second]
                    gain = current_cost - split_cost(candidate)
                    if gain > improvement:
                        improvement = gain
                        best_swap = first, second, candidate
            if best_swap is None:
                break
            first, second, sums = best_swap
            assignment[first], assignment[second] = assignment[second], assignment[first]
        cost = split_cost(sums)
        signature = tuple(assignment[key] for key in group_names)
        if cost < best_cost or (cost == best_cost and signature < tuple(best_assignment[key]
                                                                          for key in group_names)):
            best_cost = cost
            best_assignment = assignment

    group_assignment = best_assignment
    assignment = {name: group_assignment[key] for key, names in groups.items() for name in names}
    recording_counts = Counter(assignment.values())
    group_counts = Counter(group_assignment.values())
    tick_counts = {
        split: sum(int(recordings[name].get("teamCounts", {}).get("3v3", recordings[name]["targetTicks"]))
                   for name in eligible if assignment[name] == split)
        for split in ratios
    }
    total_ticks = sum(tick_counts.values())
    return {
        "seed": seed,
        "unit": "normalized room name (all recordings in a room stay together)",
        "groupKeyStrategy": "NFKC-normalized, stripped, casefolded room name; roomless recordings stay individual",
        "eligibleRecordings": len(eligible),
        "groupCount": len(groups),
        "groupAssignments": dict(sorted(group_assignment.items())),
        "recordingCounts": dict(sorted(recording_counts.items())),
        "groupCounts": dict(sorted(group_counts.items())),
        "threeVThreeTicks": dict(sorted(tick_counts.items())),
        "threeVThreeTickFractions": {
            split: round(count / total_ticks, 6) if total_ticks else 0.0
            for split, count in sorted(tick_counts.items())
        },
        "assignments": dict(sorted(assignment.items())),
    }


def _write_json(path: Path, data: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-minutes", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=SPLIT_SEED)
    parser.add_argument("--limit", type=int, default=0, help="convert at most this many new recordings")
    parser.add_argument("--splits-only", action="store_true", help="recalcular splits desde el audit.json existente")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers debe ser al menos 1")
    out = args.out.resolve()
    if args.splits_only:
        audit_path = out / "audit.json"
        if not audit_path.is_file():
            parser.error(f"no existe el audit requerido: {audit_path}")
        report = json.loads(audit_path.read_text(encoding="utf-8"))
        split = make_splits(report["recordings"], args.seed)
        _write_json(out / "splits.json", split)
        summary = report["summary"]
        summary["splitCounts"] = split["recordingCounts"]
        summary["splitGroupCounts"] = split["groupCounts"]
        summary["splitThreeVThreeTickFractions"] = split["threeVThreeTickFractions"]
        summary["splitSeed"] = args.seed
        _write_json(audit_path, report)
        print(json.dumps({key: summary[key] for key in
                          ("splitCounts", "splitGroupCounts", "splitThreeVThreeTickFractions", "splitSeed")},
                         indent=2, ensure_ascii=False))
        return
    if not args.source.is_dir():
        parser.error(f"no existe el directorio de grabaciones: {args.source}")
    if not STADIUM.is_file():
        parser.error(f"no existe el estadio objetivo: {STADIUM}")

    out.mkdir(parents=True, exist_ok=True)
    target_id = stadium_identity(STADIUM)
    index_path = out / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {"recordings": {}}
    index.setdefault("recordings", {})
    source_files = sorted(args.source.glob("*.hbr2"))
    hashes = {}
    pending = []
    for path in source_files:
        digest = sha256(path)
        row = {"sha256": digest, "source": str(path), "bytes": path.stat().st_size}
        if digest in hashes:
            row.update(status="duplicate", duplicate_of=hashes[digest])
            index["recordings"][path.name] = row
            continue
        hashes[digest] = path.name
        previous = index["recordings"].get(path.name, {})
        data_path = out / previous.get("jsonl", "")
        if previous.get("sha256") == digest and previous.get("status") == "ok" and data_path.is_file():
            previous.update(row)
            continue
        pending.append((path, row))
    if args.limit:
        pending = pending[:args.limit]

    from concurrent.futures import ThreadPoolExecutor

    def work(item):
        path, row = item
        try:
            jsonl, log = convert(path, out, args.max_minutes)
            from tools.rs4_jsonl_cache import header_and_stadiums
            header, stadiums = header_and_stadiums(jsonl)
            last_frame = None
            with gzip.open(jsonl, "rt", encoding="utf-8") as handle:
                for line in handle:
                    item = json.loads(line)
                    if item.get("type") == "tick":
                        last_frame = int(item["frame"])
            row.update(status="ok", jsonl=str(jsonl.relative_to(out)).replace("\\", "/"), log=log,
                       room=(header or {}).get("room"), maxFrameNo=(header or {}).get("maxFrameNo"),
                       lastFrame=last_frame, stadiums=stadiums)
        except Exception as error:
            row.update(status="error", error=f"{type(error).__name__}: {error}")
        return path.name, row

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for name, row in pool.map(work, pending):
            index["recordings"][name] = row
            _write_json(index_path, index)
            print(f"{row['status']:9s} {name}", flush=True)

    audited = {}
    for path in source_files:
        row = index["recordings"].get(path.name)
        if not row:
            row = {"status": "not_converted", "source": str(path), "bytes": path.stat().st_size}
        if row.get("status") == "ok":
            try:
                audited[path.name] = audit_recording(path.name, row, out, target_id)
                print(f"audit {path.name}: {audited[path.name]['targetTicks']} target ticks", flush=True)
            except (OSError, json.JSONDecodeError, KeyError, ValueError) as error:
                audited[path.name] = {"status": "audit_error", "sha256": row.get("sha256"),
                                      "error": f"{type(error).__name__}: {error}"}
                print(f"audit_error {path.name}: {type(error).__name__}: {error}", flush=True)
        else:
            audited[path.name] = {key: row.get(key) for key in
                                  ("status", "sha256", "source", "bytes", "duplicate_of", "error")}

    split = make_splits(audited, args.seed)
    split_path = out / "splits.json"
    _write_json(split_path, split)
    summary = {
        "source": str(args.source.resolve()),
        "targetStadium": str(STADIUM),
        "targetStadiumName": json.loads(STADIUM.read_text(encoding="utf-8")).get("name"),
        "targetStadiumIdentity": target_id,
        "sourceFiles": len(source_files),
        "statuses": dict(sorted(Counter(row.get("status", "unknown") for row in audited.values()).items())),
        "completeRecordings": sum(row.get("complete", False) for row in audited.values()),
        "targetRecordings": sum(row.get("targetTicks", 0) > 0 for row in audited.values()),
        "targetTicks": sum(row.get("targetTicks", 0) for row in audited.values()),
        "targetMinutes": round(sum(row.get("targetTicks", 0) for row in audited.values()) / 3600, 2),
        "excludedTicks": sum(row.get("excludedTicks", 0) for row in audited.values()),
        "excludedMinutes": round(sum(row.get("excludedTicks", 0) for row in audited.values()) / 3600, 2),
        "teamCounts": dict(sorted(sum_counters(row.get("teamCounts", {}) for row in audited.values()).items())),
        "stateCounts": dict(sorted(sum_counters(row.get("stateCounts", {}) for row in audited.values()).items())),
        "inputCounts": dict(sorted(sum_counters(row.get("inputCounts", {}) for row in audited.values()).items(),
                                   key=lambda pair: int(pair[0]))),
        "events": dict(sorted(sum_counters(row.get("events", {}) for row in audited.values()).items())),
        "goalsByTeam": dict(sorted(sum_counters(row.get("goalsByTeam", {}) for row in audited.values()).items())),
        "splitCounts": dict(sorted(Counter(split["assignments"].values()).items())),
        "splitSeed": args.seed,
    }
    _write_json(index_path, {"recordings": index["recordings"]})
    _write_json(out / "audit.json", {"summary": summary, "recordings": audited})
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def sum_counters(counters) -> Counter:
    total = Counter()
    for counter in counters:
        total.update(counter)
    return total


if __name__ == "__main__":
    main()
