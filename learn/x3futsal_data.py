"""Prepare replay-grounded 3v3 Futsal x3 behavior-cloning data.

  python -m learn.x3futsal_data
  python -m learn.x3futsal_data --cache data/x3futsal_jsonl --out data/x3futsal_bc
"""
from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
from pathlib import Path

import numpy as np

from sim.stadium import load_stadium
from tools.x3futsal_audit import DEFAULT_OUT as DEFAULT_CACHE, ROOT, STADIUM, _span_for, make_splits
from tools.x3futsal_physics import _read_replay, transition_is_valid

MOVE_DX = (0, 0, 1, 1, 1, 0, -1, -1, -1)
MOVE_DY = (0, -1, -1, 0, 1, 1, 1, 0, -1)
MOVE_BY_DELTA = dict(zip(zip(MOVE_DX, MOVE_DY), range(9)))
N_PLAYERS = 6
N_ACTIONS = 18
DATA_VERSION = "x3-futsal-bc-v3-active"
DEFAULT_OUT = ROOT / "data" / "x3futsal_bc_active"
AFK_OUTSIDE_MARGIN = 25.0
AFK_STATIONARY_FRAMES = 600
AFK_SPEED_THRESHOLD = 0.08
FILTER_CONTRACT = {
    "targetStadiumSpans": True,
    "state": "0 (kickoff/restart) or 1 (active gameplay)",
    "roster": "stable 3v3 across consecutive frames",
    "physics": "standard player properties; property-event frames excluded",
    "degeneratePositions": "coincident players and neighboring ticks excluded",
    "transition": "consecutive frame t -> t+1; label is input at t",
    "kickLabel": "kick key bit (16) at t",
    "afk": "transition excluded when any rostered player is outside the field envelope or stationary with no directional input for 600 consecutive frames",
}
FEATURE_CONTRACT = {
    "version": "x3-futsal-observation-v2",
    "attackDirection": "team 1 and team 2 are mirrored so the controlled team attacks +x",
    "observation": {
        "self": ["x", "y", "vx", "vy", "ball_x", "ball_y", "ball_vx", "ball_vy",
                 "ball_relative_x", "ball_relative_y", "score_difference", "elapsed_seconds"],
        "entities": ["relative_x", "relative_y", "vx", "vy", "ball_relative_x", "ball_relative_y"],
        "entitySets": {"teammates": 2, "opponents": 3},
        "normalization": {
            "position": ["field_half_width", "field_half_height"],
            "velocity": 6.0,
            "score_difference": 5.0,
            "elapsed_seconds": 300.0,
            "elapsed_clipped": [0.0, 1.0],
        },
    },
    "history": {
        "previousAction": "one-hot 18-class action at t-1 for the controlled player",
        "validityMask": "present only when the previous sample is contiguous in the same segment",
    },
    "action": {
        "classes": 18,
        "directions": ["stay", "up", "up_right", "right", "down_right", "down",
                       "down_left", "left", "up_left"],
        "kick_offset": 9,
        "coordinate_convention": "x right, y down; team 2 direction is mirrored horizontally",
    },
}


def action_from_input(value: int, team: int) -> int:
    """Map HaxBall key bits to 9 directions x kick, mirrored to attack toward +x."""
    value = int(value)
    if not 0 <= value < 32:
        raise ValueError(f"input de HaxBall fuera de rango: {value}")
    dx = ((value >> 3) & 1) - ((value >> 2) & 1)
    dy = ((value >> 1) & 1) - (value & 1)
    if team == 2:
        dx = -dx
    elif team != 1:
        raise ValueError(f"equipo de jugador inválido: {team}")
    return MOVE_BY_DELTA[(dx, dy)] + (9 if value & 16 else 0)


def featurize(states: np.ndarray, slots: np.ndarray, score_diff: np.ndarray, elapsed: np.ndarray,
              field_half_w: float, field_half_h: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    states = np.asarray(states, dtype=np.float32).copy()
    slots = np.asarray(slots, dtype=np.int64)
    orient = np.where(slots < 3, 1.0, -1.0).astype(np.float32)
    states[:, :, 0] *= orient[:, None]
    states[:, :, 2] *= orient[:, None]
    scale = np.asarray([max(field_half_w, 1.0), max(field_half_h, 1.0), 6.0, 6.0], dtype=np.float32)
    states /= scale

    batch = np.arange(len(states))
    players = states[:, 1:, :]
    own = players[batch, slots]
    ball = states[:, 0, :]
    local_slot = slots % 3
    mates_idx = (slots // 3)[:, None] * 3 + (local_slot[:, None] + np.asarray([1, 2])) % 3
    rival_slots = np.arange(3, dtype=np.int64)[None, :]
    rivals_idx = np.where((slots < 3)[:, None], rival_slots + 3, rival_slots)
    mates = players[batch[:, None], mates_idx]
    rivals = players[batch[:, None], rivals_idx]

    score = np.asarray(score_diff, dtype=np.float32) * orient / 5.0
    clock = np.clip(np.asarray(elapsed, dtype=np.float32) / 300.0, 0.0, 1.0)
    own_features = np.concatenate((own, ball, ball[:, :2] - own[:, :2],
                                   score[:, None], clock[:, None]), axis=1)
    mate_features = np.concatenate((mates[:, :, :2] - own[:, None, :2], mates[:, :, 2:],
                                    mates[:, :, :2] - ball[:, None, :2]), axis=2)
    rival_features = np.concatenate((rivals[:, :, :2] - own[:, None, :2], rivals[:, :, 2:],
                                     rivals[:, :, :2] - ball[:, None, :2]), axis=2)
    return own_features, mate_features, rival_features


class X3Data:
    """Split shards in memory; sampling is uniform over eligible player-ticks."""

    def __init__(self, dataset: Path, split: str):
        manifest_path = dataset / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("dataVersion") != DATA_VERSION:
            raise ValueError(f"versión de dataset incompatible: {manifest.get('dataVersion')}")
        if manifest.get("buildStatus") != "complete":
            raise ValueError(f"el dataset aún no está completo: {manifest_path}")
        self.shards = []
        for row in manifest["shards"]:
            if row["split"] != split:
                continue
            with np.load(dataset / row["file"], allow_pickle=False) as archive:
                shard = {key: archive[key] for key in archive.files}
            if len(shard["states"]) != row["ticks"] or shard["states"].shape[1:] != (7, 4):
                raise ValueError(f"shard corrupto o incompatible: {row['file']}")
            if str(shard.get("dataVersion", "")) != DATA_VERSION \
                    or str(shard.get("sourceSha256", "")) != row["sourceSha256"]:
                raise ValueError(f"metadatos de shard incompatibles: {row['file']}")
            self.shards.append(shard)
        if not self.shards:
            raise ValueError(f"el split {split!r} no tiene transiciones válidas en {dataset}")
        self.ends = np.cumsum([len(shard["states"]) for shard in self.shards], dtype=np.int64)
        self.ticks = int(self.ends[-1])
        stadium = load_stadium(STADIUM)
        self.field_half_w = float(stadium.field_half_w)
        self.field_half_h = float(stadium.field_half_h)

    def batch(self, indices: np.ndarray, slots: np.ndarray) -> tuple:
        indices = np.asarray(indices, dtype=np.int64)
        slots = np.asarray(slots, dtype=np.int64)
        if indices.shape != slots.shape or np.any(indices < 0) or np.any(indices >= self.ticks):
            raise ValueError("índices de muestras fuera de rango")
        if np.any((slots < 0) | (slots >= N_PLAYERS)):
            raise ValueError("slot de jugador fuera de rango")
        shard_ids = np.searchsorted(self.ends, indices, side="right")
        starts = np.concatenate(([0], self.ends[:-1]))
        local = indices - starts[shard_ids]
        states = np.empty((len(indices), 7, 4), dtype=np.float32)
        actions = np.empty(len(indices), dtype=np.int64)
        prior = np.empty(len(indices), dtype=np.int64)
        has_prior = np.empty(len(indices), dtype=bool)
        changed = np.empty(len(indices), dtype=bool)
        score_diff = np.empty(len(indices), dtype=np.float32)
        elapsed = np.empty(len(indices), dtype=np.float32)
        for shard_id in np.unique(shard_ids):
            mask = shard_ids == shard_id
            shard = self.shards[int(shard_id)]
            row = local[mask]
            actor = slots[mask]
            states[mask] = shard["states"][row]
            actions[mask] = shard["actions"][row, actor]
            prior[mask] = shard["prior_actions"][row, actor]
            has_prior[mask] = shard["has_prior"][row]
            changed[mask] = shard["changed"][row, actor]
            score_diff[mask] = shard["score_diff"][row]
            elapsed[mask] = shard["time"][row]
        observation = featurize(states, slots, score_diff, elapsed, self.field_half_w, self.field_half_h)
        return observation, actions, prior, has_prior, changed

    def fixed_indices(self, count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
        if count < 1:
            raise ValueError("el conjunto fijo de evaluación debe tener al menos una muestra")
        rng = np.random.default_rng(seed)
        return rng.integers(0, self.ticks, size=count), rng.integers(0, N_PLAYERS, size=count)

    def label_counts(self) -> np.ndarray:
        counts = np.zeros(N_ACTIONS, dtype=np.int64)
        for shard in self.shards:
            counts += np.bincount(shard["actions"].ravel(), minlength=N_ACTIONS)
        return counts


def _ordered_players(tick: dict) -> list[dict] | None:
    players = tick.get("players", [])
    counts = Counter(player.get("team") for player in players)
    if counts != {1: 3, 2: 3}:
        return None
    if any(player.get("id") is None or player.get("disc") is None for player in players):
        return None
    ordered = sorted(players, key=lambda player: (player["team"], player["id"]))
    disc_indices = [player["disc"] for player in ordered]
    if len(set(disc_indices)) != N_PLAYERS or any(not 0 < index < len(tick.get("discs", []))
                                                   for index in disc_indices):
        return None
    return ordered


def _encode_tick(tick: dict, ordered: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    discs = tick["discs"]
    indices = [0] + [player["disc"] for player in ordered]
    state = np.asarray([[discs[index][key] for key in ("x", "y", "vx", "vy")] for index in indices],
                       dtype=np.float32)
    if not np.isfinite(state).all():
        raise ValueError(f"estado no finito en frame {tick['frame']}")
    actions = np.asarray([action_from_input(player.get("input", 0), player["team"]) for player in ordered],
                         dtype=np.uint8)
    return state, actions


def _tick_is_encodable(tick: dict, ordered: list[dict]) -> bool:
    indices = [0] + [player["disc"] for player in ordered]
    return all(
        all(disc.get(key) is not None and np.isfinite(disc[key]) for key in ("x", "y", "vx", "vy"))
        for disc in (tick["discs"][index] for index in indices)
    )


def extract_recording(name: str, audit_row: dict, index_row: dict, cache: Path) -> dict:
    """Extract valid transition starts; labels are inputs at t for the verified t -> t+1 transition."""
    if not audit_row.get("complete"):
        raise ValueError(f"replay incompleto no elegible para BC: {name}")
    spans = audit_row.get("targetSpans", [])
    if not spans:
        return _empty_shard()
    ticks, event_frames, _, property_summary, _, _ = _read_replay(
        cache / index_row["jsonl"], spans, default_player_properties=load_stadium(
            ROOT / "stadiums" / "futsal3bybazinga.hbs").player)
    target_stadium = load_stadium(STADIUM)

    states, actions, prior_actions, has_prior, changed, frames, score_diff, elapsed = (
        [] for _ in range(8)
    )
    previous_frame = None
    previous_actions = None
    last_observed_frame = None
    stationary_frames = np.zeros(N_PLAYERS, dtype=np.int64)
    for previous, current in zip(ticks, ticks[1:]):
        span = _span_for(int(previous["frame"]), spans)
        if span is None or not transition_is_valid(
                previous, current, span, event_frames, check_coincident=False, allowed_states=(0, 1)):
            continue
        ordered = _ordered_players(previous)
        next_ordered = _ordered_players(current)
        if ordered is None or next_ordered is None \
                or not _tick_is_encodable(previous, ordered) or not _tick_is_encodable(current, next_ordered):
            continue
        roster = tuple((player["id"], player["team"]) for player in ordered)
        if roster != tuple((player["id"], player["team"]) for player in next_ordered):
            continue
        state, label = _encode_tick(previous, ordered)
        if last_observed_frame != int(previous["frame"]) - 1:
            stationary_frames.fill(0)
        for player_index, player in enumerate(ordered):
            disc = state[player_index + 1]
            no_direction = int(label[player_index] % 9) == 0
            slow = float(np.hypot(disc[2], disc[3])) < AFK_SPEED_THRESHOLD
            stationary_frames[player_index] = (
                stationary_frames[player_index] + 1 if no_direction and slow else 0
            )
        outside = np.any(np.abs(state[1:, 0]) > target_stadium.field_half_w + AFK_OUTSIDE_MARGIN) \
            or np.any(np.abs(state[1:, 1]) > target_stadium.field_half_h + AFK_OUTSIDE_MARGIN)
        if outside or np.any(stationary_frames >= AFK_STATIONARY_FRAMES):
            last_observed_frame = int(previous["frame"])
            continue
        contiguous = previous_frame == int(previous["frame"]) - 1
        states.append(state)
        actions.append(label)
        prior_actions.append(previous_actions if contiguous else np.zeros(N_PLAYERS, dtype=np.uint8))
        has_prior.append(contiguous)
        changed.append((label != previous_actions) if contiguous else np.zeros(N_PLAYERS, dtype=bool))
        frames.append(int(previous["frame"]))
        scores = previous.get("score")
        if scores is None or len(scores) != 2 or scores[0] is None or scores[1] is None:
            raise ValueError(f"marcador inválido en {name}, frame {previous['frame']}")
        score_diff.append(int(scores[0]) - int(scores[1]))
        if previous.get("time") is None:
            raise ValueError(f"reloj faltante en {name}, frame {previous['frame']}")
        elapsed.append(float(previous["time"]))
        previous_frame = int(previous["frame"])
        previous_actions = label
        last_observed_frame = int(previous["frame"])

    if not states:
        return _empty_shard()
    return {
        "states": np.asarray(states, dtype=np.float16),
        "actions": np.asarray(actions, dtype=np.uint8),
        "prior_actions": np.asarray(prior_actions, dtype=np.uint8),
        "has_prior": np.asarray(has_prior, dtype=bool),
        "changed": np.asarray(changed, dtype=bool),
        "frame": np.asarray(frames, dtype=np.int32),
        "score_diff": np.asarray(score_diff, dtype=np.int16),
        "time": np.asarray(elapsed, dtype=np.float32),
        "playerPhysicsOverrideTicks": property_summary["playerPhysicsOverrideTicks"],
    }


def _empty_shard() -> dict:
    return {
        "states": np.empty((0, 7, 4), dtype=np.float16),
        "actions": np.empty((0, N_PLAYERS), dtype=np.uint8),
        "prior_actions": np.empty((0, N_PLAYERS), dtype=np.uint8),
        "has_prior": np.empty(0, dtype=bool),
        "changed": np.empty((0, N_PLAYERS), dtype=bool),
        "frame": np.empty(0, dtype=np.int32),
        "score_diff": np.empty(0, dtype=np.int16),
        "time": np.empty(0, dtype=np.float32),
        "playerPhysicsOverrideTicks": 0,
    }


def _atomic_save_npz(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    try:
        with temp.open("wb") as handle:
            np.savez_compressed(handle, **{key: value for key, value in data.items()
                                           if key != "playerPhysicsOverrideTicks"})
        temp.replace(path)
    finally:
        if temp.exists():
            temp.unlink()


def _load_existing_shard(path: Path, expected_sha256: str) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    required = {"states", "actions", "prior_actions", "has_prior", "changed",
                "frame", "score_diff", "time"}
    if not required.issubset(data):
        raise ValueError(f"shard existente incompleto: {path}")
    count = len(data["states"])
    if data["states"].shape != (count, 7, 4) or data["actions"].shape != (count, N_PLAYERS) \
            or data["prior_actions"].shape != (count, N_PLAYERS) or data["changed"].shape != (count, N_PLAYERS) \
            or data["has_prior"].shape != (count,) or data["frame"].shape != (count,) \
            or data["score_diff"].shape != (count,) or data["time"].shape != (count,):
        raise ValueError(f"shard existente con dimensiones incompatibles: {path}")
    version = str(data.get("dataVersion", ""))
    source_sha = str(data.get("sourceSha256", ""))
    if version and version != DATA_VERSION:
        raise ValueError(f"shard existente de otra versión: {path}")
    if source_sha and source_sha != expected_sha256:
        raise ValueError(f"shard existente de otro replay: {path}")
    if not version or not source_sha:
        data["dataVersion"] = np.asarray(DATA_VERSION)
        data["sourceSha256"] = np.asarray(expected_sha256)
        _atomic_save_npz(path, data)
    return data


def _build_shard(job: tuple) -> dict:
    name, split, audit_row, index_row, cache, out = job
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:12]
    filename = f"{Path(name).stem}.{digest}.npz"
    relative = Path("shards") / split / filename
    path = out / relative
    if path.is_file():
        data = _load_existing_shard(path, index_row["sha256"])
        count = len(data["states"])
        return {"recording": name, "sourceSha256": index_row["sha256"], "room": audit_row.get("room"),
                "split": split, "file": str(relative).replace("\\", "/"), "ticks": count,
                "empty": count == 0, "reused": True}

    data = extract_recording(name, audit_row, index_row, cache)
    count = len(data["states"])
    if count:
        data["dataVersion"] = np.asarray(DATA_VERSION)
        data["sourceSha256"] = np.asarray(index_row["sha256"])
        _atomic_save_npz(path, data)
    return {"recording": name, "sourceSha256": index_row["sha256"], "room": audit_row.get("room"),
            "split": split, "file": str(relative).replace("\\", "/"), "ticks": count,
            "empty": count == 0, "reused": False}


def build_dataset(cache: Path, out: Path, seed: int, workers: int) -> dict:
    if workers < 1:
        raise ValueError("workers debe ser al menos 1")
    audit = json.loads((cache / "audit.json").read_text(encoding="utf-8"))["recordings"]
    index = json.loads((cache / "index.json").read_text(encoding="utf-8"))["recordings"]
    splits = make_splits(audit, seed)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    existing_rows = {}
    processed = set()
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old.get("dataVersion") != DATA_VERSION or old.get("seed") != seed:
            raise ValueError("el directorio ya contiene un dataset de otra versión/semilla; usa otro --out")
        if old.get("split", {}).get("assignments") != splits["assignments"]:
            raise ValueError("el split cambió; usa otro --out para no mezclar shards")
        old_rows = {row["recording"]: row for row in old.get("shards", [])}
        processed = set(old.get("emptyRecordings", []))
        for name, row in old_rows.items():
            if name not in splits["assignments"] or row["split"] != splits["assignments"][name]:
                raise ValueError(f"shard pertenece a otro split: {name}")
            path = out / row["file"]
            if not path.is_file():
                continue
            data = _load_existing_shard(path, index[name]["sha256"])
            if len(data["states"]) != row["ticks"]:
                raise ValueError(f"cantidad de ticks cambió en shard existente: {name}")
            row["sourceSha256"] = index[name]["sha256"]
            existing_rows[name] = row
            processed.add(name)

    manifest = {
        "dataVersion": DATA_VERSION,
        "targetStadiumIdentity": json.loads((cache / "audit.json").read_text(encoding="utf-8"))[
            "summary"]["targetStadiumIdentity"],
        "cache": str(cache),
        "seed": seed,
        "split": splits,
        "filter": FILTER_CONTRACT,
        "featureContract": FEATURE_CONTRACT,
        "buildStatus": "building",
        "processedRecordings": sorted(processed),
        "emptyRecordings": sorted(set(processed) - set(existing_rows)),
        "shardCount": len(existing_rows),
        "tickCountBySplit": {
            split: sum(row["ticks"] for row in existing_rows.values() if row["split"] == split)
            for split in ("train", "dev", "test")
        },
        "shards": sorted(existing_rows.values(), key=lambda row: row["recording"]),
    }
    _write_json(manifest_path, manifest)
    _write_json(out / "splits.json", splits)

    jobs = [(name, split, audit[name], index[name], cache, out)
            for name, split in splits["assignments"].items() if name not in processed]
    if workers == 1:
        results = map(_build_shard, jobs)
        for result in results:
            _record_result(manifest, result, manifest_path)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for result in pool.map(_build_shard, jobs, chunksize=1):
                _record_result(manifest, result, manifest_path)

    manifest["buildStatus"] = "complete"
    manifest["processedRecordings"] = sorted(splits["assignments"])
    _write_json(manifest_path, manifest)
    return manifest


def _record_result(manifest: dict, result: dict, manifest_path: Path) -> None:
    name = result["recording"]
    manifest["processedRecordings"] = sorted(set(manifest["processedRecordings"]) | {name})
    empty = set(manifest.get("emptyRecordings", []))
    rows = {row["recording"]: row for row in manifest["shards"]}
    if result["ticks"]:
        rows[name] = {key: result[key] for key in
                      ("recording", "sourceSha256", "room", "split", "file", "ticks")}
        empty.discard(name)
    else:
        rows.pop(name, None)
        empty.add(name)
    manifest["emptyRecordings"] = sorted(empty)
    manifest["shards"] = sorted(rows.values(), key=lambda row: row["recording"])
    manifest["shardCount"] = len(manifest["shards"])
    manifest["tickCountBySplit"] = {
        split: sum(row["ticks"] for row in manifest["shards"] if row["split"] == split)
        for split in ("train", "dev", "test")
    }
    _write_json(manifest_path, manifest)
    print(f"{result['split']:5s} {result['ticks']:7d} {name}"
          f"{' (reutilizado)' if result['reused'] else ''}", flush=True)


def refresh_manifest(cache: Path, out: Path, seed: int) -> dict:
    """Refresh split and feature metadata after an interrupted or earlier shard build."""
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    audit_doc = json.loads((cache / "audit.json").read_text(encoding="utf-8"))
    index = json.loads((cache / "index.json").read_text(encoding="utf-8"))["recordings"]
    splits = make_splits(audit_doc["recordings"], seed)
    for row in manifest["shards"]:
        recording = row["recording"]
        split = splits["assignments"].get(recording)
        if split is None:
            raise ValueError(f"shard no elegible para el split actual: {recording}")
        expected = Path("shards") / split / Path(row["file"]).name
        if Path(row["file"]) != expected or not (out / expected).is_file():
            raise ValueError(f"shard falta o pertenece a otro split: {row['file']}")
        row["split"] = split
        row["file"] = str(expected).replace("\\", "/")
        row["sourceSha256"] = index[recording]["sha256"]
        shard = _load_existing_shard(out / expected, index[recording]["sha256"])
        if len(shard["states"]) != row["ticks"]:
            raise ValueError(f"cantidad de ticks cambió en shard existente: {recording}")
    manifest["dataVersion"] = DATA_VERSION
    manifest["targetStadiumIdentity"] = audit_doc["summary"]["targetStadiumIdentity"]
    manifest["seed"] = seed
    manifest["split"] = splits
    manifest["filter"] = FILTER_CONTRACT
    manifest["featureContract"] = FEATURE_CONTRACT
    manifest["shardCount"] = len(manifest["shards"])
    manifest["buildStatus"] = "complete" if set(splits["assignments"]) <= set(
        manifest.get("processedRecordings", [])) else "building"
    manifest["tickCountBySplit"] = {
        split: sum(row["ticks"] for row in manifest["shards"] if row["split"] == split)
        for split in ("train", "dev", "test")
    }
    _write_json(manifest_path, manifest)
    _write_json(out / "splits.json", splits)
    return manifest


def _write_json(path: Path, value: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=20261027)
    parser.add_argument("--workers", type=int, default=2, help="replays procesados simultáneamente")
    parser.add_argument("--refresh-manifest", action="store_true",
                        help="actualizar metadata de un dataset ya generado, sin releer sus replays")
    args = parser.parse_args()
    if args.refresh_manifest:
        manifest = refresh_manifest(args.cache, args.out, args.seed)
        print(json.dumps({"shards": manifest["shardCount"], "ticks": manifest["tickCountBySplit"]},
                         indent=2, ensure_ascii=False))
        return
    manifest = build_dataset(args.cache, args.out, args.seed, args.workers)
    print(json.dumps({"shards": manifest["shardCount"], "ticks": manifest["tickCountBySplit"],
                      "recordings": manifest["split"]["recordingCounts"],
                      "groups": manifest["split"]["groupCounts"],
                      "threeVThreeTickFractions": manifest["split"]["threeVThreeTickFractions"]},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
