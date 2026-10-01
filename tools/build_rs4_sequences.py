"""Stream available RSx4 recordings into replay-disjoint sequence shards.

No downloads and no concatenation of an entire dataset in RAM. Keys are the
recorded human inputs, not an oracle for possession, tactics or restart owners.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import numpy as np

from tools.build_bc_dataset import Loader, catalog_by_name, replay_stadium, ROOT


def replay_split(identity, validation_fraction=.2):
    value = int(hashlib.sha256(identity.encode()).hexdigest()[:8], 16) / 2**32
    return "validation" if value < validation_fraction else "train"


def replay_identity(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()[:20]


def iter_ticks(path):
    """Yield JSON objects one at a time (headers/events preserve boundaries)."""
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def recognizable_label(tick, env, previous_ball=None):
    """Conservative geometry labels; -1 explicitly means ambiguous."""
    if tick["state"] == 0:
        return 4  # visible kickoff
    ball = tick["discs"][0]
    if previous_ball is None or np.linalg.norm(np.array([ball["x"], ball["y"]]) - previous_ball) > .01:
        return -1
    if np.hypot(ball["vx"], ball["vy"]) > .01:
        return -1
    W, H, radius = env.field_w, env.field_h, env.sim.st.ball["radius"]
    if abs(abs(ball["y"]) - (H - radius - 1)) < 4:
        if abs(abs(ball["x"]) - (W - radius - 1)) < 4:
            return 2
        if abs(ball["x"]) < .85 * W:
            return 1
    # Goal-kick locations/script offsets vary. Do not infer them from a player
    # standing near their goal: that would silently label ordinary play.
    return -1


class SequenceShards:
    def __init__(self, out, replay_id, split, length=32, shard_sequences=512):
        self.directory = Path(out) / split
        self.directory.mkdir(parents=True, exist_ok=True)
        self.replay_id = replay_id
        self.length, self.limit = length, shard_sequences
        self.pending = []
        self.manifest = []
        self.index = 0

    def add(self, player_id, rows):
        if not rows:
            return
        n, width = len(rows), rows[0][0].shape[-1]
        obs = np.zeros((self.length, width), dtype=np.float16)
        action = np.zeros(self.length, dtype=np.uint8)
        previous = np.full(self.length, 18, dtype=np.uint8)
        valid = np.zeros(self.length, dtype=bool)
        start = np.zeros(self.length, dtype=bool)
        labels = np.full(self.length, -1, dtype=np.int8)
        frames = np.full(self.length, -1, dtype=np.int64)
        for i, (o, a, prior, frame, label, episode_start) in enumerate(rows):
            obs[i], action[i], previous[i], frames[i], labels[i] = o, a, prior, frame, label
            start[i] = episode_start
            valid[i] = True
        self.pending.append((obs, action, previous, valid, start, labels, frames, player_id))
        if len(self.pending) >= self.limit:
            self.flush()

    def flush(self):
        if not self.pending:
            return
        name = f"{self.replay_id}-{self.index:05d}.npz"
        path = self.directory / name
        columns = list(zip(*self.pending))
        np.savez_compressed(path, obs=np.stack(columns[0]), act=np.stack(columns[1]),
                            previous_action=np.stack(columns[2]), valid=np.stack(columns[3]),
                            episode_start=np.stack(columns[4]), labels=np.stack(columns[5]),
                            frames=np.stack(columns[6]), player_id=np.asarray(columns[7]),
                            replay_id=np.array(self.replay_id), cadence=np.array(3))
        self.manifest.append({"path": str(path), "sequences": len(self.pending),
                              "useful_samples": int(sum(row[3].sum() for row in self.pending))})
        self.pending = []
        self.index += 1


def process_jsonl(path, out, *, replay_id, sequence_length=32, shard_sequences=512, validation_fraction=.2):
    """Bounded-memory conversion; never join roster, map or game boundaries."""
    cat = catalog_by_name()
    split = replay_split(replay_id, validation_fraction)
    writer = SequenceShards(out, replay_id, split, sequence_length, shard_sequences)
    loader = None
    streams, prior = {}, {}
    last_frame, roster, score, previous_ball = None, None, None, None
    since = 0

    def boundary():
        nonlocal streams, prior, last_frame, roster, score, previous_ball, since
        for player_id, rows in streams.items():
            writer.add(player_id, rows)
        streams, prior = {}, {}
        last_frame, roster, score, previous_ball, since = None, None, None, None, 0

    for item in iter_ticks(path):
        if item.get("type") == "header" or item.get("name") == "stadium_change":
            boundary()
            meta = cat.get((item.get("stadium") or "").strip())
            loader = None
            if meta and meta["stadium"] == "rs_one" and not meta.get("script") and item.get("stadiumFile"):
                hbs = Path(path).parent / item["stadiumFile"]
                stadium, _ = replay_stadium(hbs, meta)
                loader = Loader(stadium, 4, False, True, batch=1)
            continue
        if item.get("type") != "tick":
            if item.get("name") in ("game_stop", "game_start", "player_join", "player_leave", "team_change"):
                boundary()
            continue
        if loader is None or item.get("state") not in (0, 1):
            boundary()
            continue
        players = sorted(item["players"], key=lambda p: (p["team"], p["id"]))
        signature = tuple((p["id"], p["team"], p["disc"]) for p in players)
        valid = len(players) == 8 and sum(p["team"] == 1 for p in players) == 4
        valid &= all(p["disc"] >= 0 and p["disc"] < len(item["discs"]) for p in players)
        discs = [item["discs"][0]] + [item["discs"][p["disc"]] for p in players] if valid else []
        valid &= all(all(d.get(k) is not None and np.isfinite(d[k]) for k in ("x", "y", "vx", "vy")) for d in discs)
        if not valid:
            boundary()
            continue
        current_score = tuple(item.get("score", (0, 0)))
        if (roster is not None and roster != signature) or (score is not None and score != current_score):
            boundary()
        roster, score = signature, current_score
        since = 0 if item["state"] == 0 else since + 1
        frame = int(item["frame"])
        if frame % 3:
            continue
        if last_frame is not None and frame != last_frame + 3:
            boundary()
            roster, score = signature, current_score
        label = recognizable_label(item, loader.env, previous_ball)
        try:
            observations, actions = loader.obs([(item, players, since)])
        except (IndexError, ValueError):
            boundary()
            continue
        if not np.isfinite(observations).all():
            boundary()
            continue
        for p, observation, action in zip(players, observations, actions):
            player_id = int(p["id"])
            rows = streams.setdefault(player_id, [])
            rows.append((observation, action, prior.get(player_id, 18), frame, label, player_id not in prior))
            prior[player_id] = action
            if len(rows) == sequence_length:
                writer.add(player_id, rows)
                streams[player_id] = []
        last_frame = frame
        previous_ball = np.array([item["discs"][0]["x"], item["discs"][0]["y"]])
    boundary()
    writer.flush()
    return {"replay_id": replay_id, "split": split, "shards": writer.manifest,
            "sequences": sum(row["sequences"] for row in writer.manifest),
            "useful_samples": sum(row["useful_samples"] for row in writer.manifest)}


def iter_shards(manifest_path, split="train"):
    """Bounded-memory reader consumed by BC tooling, one compressed shard at a time."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    for replay in manifest["replays"]:
        if replay["split"] == split:
            for shard in replay["shards"]:
                with np.load(shard["path"], allow_pickle=False) as data:
                    yield {key: data[key] for key in data.files}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default=str(ROOT / "replays_real" / "stadiums" / "rsx4"))
    ap.add_argument("--out", default=str(ROOT / "data" / "rs4_v3_sequences"))
    ap.add_argument("--sequence-length", type=int, default=32)
    ap.add_argument("--shard-sequences", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        ap.error("Destination exists; choose a new dataset directory")
    if args.sequence_length < 1 or args.shard_sequences < 1:
        ap.error("Sequence/shard sizes must be positive")
    source = Path(args.source)
    files = sorted([*source.rglob("*.hbr2"), *source.rglob("*.hbr"), *source.rglob("*.jsonl")])
    if args.limit:
        files = files[:args.limit]
    if not files:
        ap.error("No local recordings found (no download is attempted)")
    out.mkdir(parents=True)
    results, seen = [], set()
    for file in files:
        identity = replay_identity(file)
        if identity in seen:
            continue
        seen.add(identity)
        if file.suffix == ".jsonl":
            result = process_jsonl(file, out, replay_id=identity, sequence_length=args.sequence_length,
                                   shard_sequences=args.shard_sequences)
        else:
            with tempfile.TemporaryDirectory(prefix="rs4-sequences-") as temporary:
                subprocess.run(["node", str(ROOT / "bridge" / "replay_to_jsonl.js"), str(file),
                                "--out", temporary, "--max-minutes", "120"], check=True, cwd=ROOT)
                jsonl = next(Path(temporary).glob("*.jsonl"))
                result = process_jsonl(jsonl, out, replay_id=identity, sequence_length=args.sequence_length,
                                       shard_sequences=args.shard_sequences)
        results.append(result)
        print(f"{file.name}: {result['useful_samples']} samples, {result['split']}", flush=True)
    manifest = {"version": 3, "cadence_ticks": 3, "sequence_length": args.sequence_length,
                "split_contract": "SHA256-replay-level-20%-validation", "replays": results,
                "labels": {"-1": "ambiguous", "1": "visible_stationary_lateral", "2": "visible_stationary_corner", "4": "kickoff"},
                "limitations": ["Geometry labels are not exact room-script labels.",
                                 "Current frozen BC remains active until a separately trained replacement passes validation."]}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(out.resolve() / "manifest.json")


if __name__ == "__main__":
    main()
