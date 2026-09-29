"""Identify byte-identical recordings before constructing train/validation shards."""
import hashlib
from pathlib import Path


def duplicate_recordings(recordings):
    """Map duplicate names to a deterministic canonical name (prefer shortest)."""
    seen, duplicates = {}, {}
    for record in sorted(recordings, key=lambda r: (len(r["name"]), r["name"])):
        if "error" in record or "file" not in record:
            continue
        digest = hashlib.sha256()
        with Path(record["file"]).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        key = digest.digest()
        if key in seen:
            duplicates[record["name"]] = seen[key]
        else:
            seen[key] = record["name"]
    return duplicates
