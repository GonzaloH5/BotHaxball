"""Atomic checkpoint publication: readers never see a partially written archive."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import torch


def atomic_torch_save(checkpoint, path):
    path = Path(path)
    # Same directory/filesystem is required for an atomic replace.
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(checkpoint, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        # Only remove our own temporary file, never a prior checkpoint.
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
