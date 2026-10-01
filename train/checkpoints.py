"""Atomic checkpoint publication: readers never see a partially written archive."""
from __future__ import annotations

import os
import re
import tempfile
import time
from pathlib import Path

import torch


_HISTORY_NAME = re.compile(r"ckpt_([0-9]{6,})\.pt\Z")


def prune_checkpoint_history(run_dir, keep):
    """Sólo archivos automáticos regulares del run; nunca enlaces ni subdirectorios."""
    if keep is None:
        return []
    if isinstance(keep, bool) or not isinstance(keep, int) or keep < 1:
        raise ValueError("log.checkpoint_keep debe ser un entero >= 1 o null")
    root = Path(run_dir).resolve()
    candidates = []
    for path in root.iterdir():
        match = _HISTORY_NAME.fullmatch(path.name)
        if match and not path.is_symlink() and path.is_file() and path.resolve().parent == root:
            candidates.append((int(match[1]), path))
    removed = []
    for _, path in sorted(candidates, reverse=True)[keep:]:
        path.unlink()
        removed.append(path)
    return removed


def maybe_save_checkpoints(trainer):
    """Latest frecuente, historia espaciada y retención tras publicación exitosa.

    Intervalos monotónicos de sesión; sin opciones nuevas conserva checkpoint_every.
    Devuelve la historia nueva para replays. Ctrl+C/fin siguen guardando latest aparte.
    """
    log = trainer.cfg["log"]
    keep = log.get("checkpoint_keep")
    if keep is not None and (isinstance(keep, bool) or not isinstance(keep, int) or keep < 1):
        raise ValueError("log.checkpoint_keep debe ser un entero >= 1 o null")
    interval = log.get("checkpoint_interval_seconds")
    history_interval = log.get("checkpoint_history_interval_seconds", interval)
    if interval is None:
        due_latest = due_history = trainer.iteration % log["checkpoint_every"] == 0
    else:
        import math
        if (not math.isfinite(interval) or interval <= 0 or history_interval is None
                or not math.isfinite(history_interval) or history_interval < interval):
            raise ValueError("Intervalos de checkpoint positivos; historia >= latest")
        now = time.monotonic()
        if not hasattr(trainer, "_checkpoint_latest_at"):
            trainer._checkpoint_latest_at = trainer._checkpoint_history_at = now
            return None
        due_latest = now - trainer._checkpoint_latest_at >= interval
        due_history = now - trainer._checkpoint_history_at >= history_interval
    if not due_latest and not due_history:
        return None
    trainer.save(trainer.run_dir / "latest.pt")
    if interval is not None:
        trainer._checkpoint_latest_at = time.monotonic()
    if not due_history:
        return None
    path = trainer.run_dir / f"ckpt_{trainer.iteration:06d}.pt"
    trainer.save(path)
    if interval is not None:
        trainer._checkpoint_history_at = time.monotonic()
    removed = prune_checkpoint_history(trainer.run_dir, keep)
    if removed:
        print(f"checkpoints: eliminadas {len(removed)} copias automáticas antiguas; "
              f"retención {keep}. No se pueden recuperar sin un backup externo.", flush=True)
    return path


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
