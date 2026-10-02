"""Enable map-discovered joint barriers in an existing, stopped public-v1 run.

No checkpoint/model/Adam/calendar/budget migration. Config and manifest only;
normal checkpoint saving persists the new sensor configuration on resume.
"""
from __future__ import annotations
import argparse
import contextlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import torch
import yaml

from env.public_joints import discover_joint_barriers
from env.public_signals import RED_COLORS, BLUE_COLORS
from tools.run_rs4_v3 import atomic_json
from train.runtime import load_config

ROOT = Path(__file__).resolve().parents[1]


@contextlib.contextmanager
def stopped(directory, *, dry_run=False):
    # Keep the inode stable, same advisory lock as the runner. Do not guess PIDs.
    lock_path = directory / ".runner.lock"
    lock = lock_path.open("r+b") if lock_path.exists() else None if dry_run else lock_path.open("a+b")
    try:
        if lock:
            try:
                if os.name == "nt":
                    import msvcrt
                    if not dry_run and lock_path.stat().st_size == 0:
                        lock.write(b"0")
                        lock.flush()
                    lock.seek(0)
                    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                raise ValueError("Stop the runner and wait for its checkpoint save before configuring joints") from error
        yield
    finally:
        if lock:
            lock.close()


def atomic_yaml(data, path):
    fd, temporary = tempfile.mkstemp(prefix=".public-joints-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            yaml.safe_dump(data, stream, sort_keys=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def configure(run="rs4_v3_public", *, dry_run=False):
    if not run or run in (".", "..") or any(c in run for c in "/\\:"):
        raise ValueError("Run must be a simple name")
    directory = ROOT / "runs" / run
    if not directory.is_dir():
        raise ValueError("Missing public run; migrate to public-v1 first")
    with stopped(directory, dry_run=dry_run):
        manifest_path = directory / "specialization.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        ledger = json.loads((directory / "ledger.json").read_text(encoding="utf-8"))
        branch = ledger.get("selected")
        if manifest.get("kind") != "rs4_v3_program" or manifest.get("version") != 3 or branch not in ("control", "memory"):
            raise ValueError("Requires a selected RS4 v3 program")
        path = directory / branch / "config.yaml"
        cfg = load_config(path)
        ck = torch.load(directory / branch / "latest.pt", map_location="cpu", weights_only=False)
        if ck["model_config"].get("public_signals_version") != 1 or cfg["model"].get("public_signals_version") != 1:
            raise ValueError("Requires the already-migrated public-v1 model and config")
        if ck["steps"] != cfg["rs4_program"]["start_steps"] + ck["rs4_program_state"]["relative_steps"]:
            raise ValueError("Checkpoint/program counter mismatch")
        # Configured sensor appearance, not a hard-coded runtime joint index.
        map_path = Path(__file__).resolve().parents[1] / "stadiums/rs_one.hbs"
        data = json.loads(map_path.read_text(encoding="utf-8"))
        geom = data["haxballrl"]
        candidates = discover_joint_barriers(data, geom["field_half_w"], geom["field_half_h"])
        if not candidates:
            raise ValueError("Catalog map has no supported RS4 paired joint topology")
        settings = dict(cfg.get("public_signals", {}))
        settings.update(version=1, auto_joints=True, joint_profile="rs4_paired_lateral_v1")
        settings["red_colors"] = list(dict.fromkeys([*settings.get("red_colors", []), *RED_COLORS]))
        settings["blue_colors"] = list(dict.fromkeys([*settings.get("blue_colors", []), *BLUE_COLORS]))
        summary = dict(run=run, branch=branch, catalog_candidates=candidates, auto_joints=True,
                       preserved_steps=ck["steps"], phase=ck["rs4_program_state"]["phase_index"],
                       checkpoint_modified=False, budget_expansion=0)
        if dry_run:
            return summary
        cfg["run_name"] = f"{run}/{branch}"
        cfg["public_signals"] = settings
        backup = path.with_name("config_before_public_joints.yaml")
        if not backup.exists():
            shutil.copy2(path, backup)
        atomic_yaml(cfg, path)
        manifest["public_joint_adaptation"] = summary
        manifest["evaluation_directory"] = "evaluations_public_joints_v1"
        atomic_json(manifest, manifest_path)
        return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="rs4_v3_public")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        result = configure(args.run, dry_run=args.dry_run)
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2))
    if not args.dry_run:
        print(f"Continue: python -u -m tools.run_rs4_v3 --run {args.run} --resume")


if __name__ == "__main__":
    main()
