"""Identidad de las condiciones de evaluación; latest.pt no identifica un rival fijo."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def protocol_id(protocol):
    return hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()


def opponent_identity(spec):
    if spec in ("scripted", "random") or spec.startswith("scripted:"):
        return {"kind": spec}
    return {"checkpoint_sha256": file_hash(spec)}


def source_fingerprint(tasks):
    paths = set()
    for folder in ("sim", "env", "bots", "eval"):
        paths.update((ROOT / folder).glob("*.py"))
    paths.add(ROOT / "train" / "model.py")
    paths.add(ROOT / "train" / "recurrent_model.py")
    paths.update(ROOT / "stadiums" / (t.stadium + ".hbs") for t in tasks)
    return {str(p.relative_to(ROOT)).replace("\\", "/"): file_hash(p) for p in sorted(paths)}
