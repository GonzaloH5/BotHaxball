"""Copy a stopped RS4 v3 program and add public cues without resetting progress."""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import torch
import yaml

from train.model import build_model
from train.rs4_migration import migrate_adam, optimizer_parameter_names
from train.checkpoints import atomic_torch_save
from train.runtime import load_config
from tools.run_rs4_v3 import atomic_json, refresh_ledger

ROOT = Path(__file__).resolve().parents[1]


def migrate_public_checkpoint(checkpoint):
    config = dict(checkpoint["model_config"])
    if config.get("type") not in ("set", "recurrent_set") or config.get("rule_observation") != "masked":
        raise ValueError("Requires a masked universal set/recurrent RS4 model")
    if config.get("public_signals_version", 0):
        raise ValueError("Checkpoint already uses public signals")
    source, target = build_model(config), build_model({**config, "public_signals_version": 1})
    source.load_state_dict(checkpoint["model"])
    state = target.state_dict()
    for key, value in checkpoint["model"].items():
        if state[key].shape != value.shape:
            raise ValueError(f"Incompatible preserved tensor {key}")
        state[key] = value.clone()
    target.load_state_dict(state)
    opt, transferred, new = migrate_adam(checkpoint, source, target)
    result = copy.deepcopy(checkpoint)
    result.update(model=target.state_dict(), model_config=target.config(), opt=opt,
                  optimizer_param_names=optimizer_parameter_names(target), frozen_normalizers=True)
    result["public_signal_migration"] = dict(version=1, source_steps=checkpoint["steps"],
                                           transferred_adam=transferred, new_parameters=new,
                                           initial_residual="zero", observation_width="unchanged")
    return result


def _under_runs(name):
    if not name or name in (".", "..") or any(c in name for c in "/\\:"):
        raise ValueError("Run must be a simple name, not a path")
    path = (ROOT / "runs" / name).resolve()
    root = (ROOT / "runs").resolve()
    if root not in path.parents:
        raise ValueError("Run must be a named directory inside runs/")
    return path


def prepare(source_run="rs4_v3", run="rs4_v3_public", *, dry_run=False, barrier_discs=(), barrier_segments=()):
    source, destination = _under_runs(source_run), _under_runs(run)
    if not source.is_dir() or destination.exists():
        raise ValueError("Source missing or destination exists; original program is never overwritten")
    if any(i <= 0 for i in barrier_discs) or any(i < 0 for i in barrier_segments):
        raise ValueError("Barrier disc IDs must exclude ball 0; segment IDs must be nonnegative")
    lock = None
    try:
        if (source / ".runner.lock").exists():
            lock = (source / ".runner.lock").open("r+b")
            try:
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                raise ValueError("Stop the source runner and wait for checkpoint save before migration") from error
        ledger = json.loads((source / "ledger.json").read_text(encoding="utf-8"))
        manifest = json.loads((source / "specialization.json").read_text(encoding="utf-8"))
        if manifest.get("kind") != "rs4_v3_program" or manifest.get("version") != 3:
            raise ValueError("Source must be a prepared RS4 v3 program")
        branch = ledger.get("selected")
        if branch not in ("control", "memory") or ledger.get("complete"):
            raise ValueError("Requires an unfinished program with completed architecture selection")
        cfg = load_config(source / branch / "config.yaml")
        checkpoint_path = source / branch / "latest.pt"
        from tools.prepare_rs4_v3 import file_hash
        original_hash = file_hash(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint["steps"] != cfg["rs4_program"]["start_steps"] + checkpoint["rs4_program_state"]["relative_steps"]:
            raise ValueError("Inconsistent checkpoint/program step counter")
        migrated = migrate_public_checkpoint(checkpoint)
        migrated.setdefault("env", {})["public_signal_config"] = dict(version=1, barrier_discs=list(barrier_discs),
                                                                      barrier_segments=list(barrier_segments))
        summary = dict(source=str(source), destination=str(destination), branch=branch,
                       preserved_steps=checkpoint["steps"], phase=checkpoint["rs4_program_state"]["phase_index"],
                       public_signals_version=1, budget_expansion=0,
                       barrier_discs=list(barrier_discs), barrier_segments=list(barrier_segments))
        if dry_run:
            return summary
        size = sum(p.stat().st_size for p in source.rglob("*") if p.is_file())
        if shutil.disk_usage(destination.parent).free < size + max(128 * 1024**2, 2 * checkpoint_path.stat().st_size):
            raise OSError("Insufficient space to preserve a full independent program copy")
        shutil.copytree(source, destination, ignore=shutil.ignore_patterns(".runner.lock", "runner.pid"))
        if file_hash(checkpoint_path) != original_hash:
            raise ValueError("Source checkpoint changed during copying; copied run must not be used")
        for name in ("control", "memory"):
            config = load_config(destination / name / "config.yaml")
            config["run_name"] = f"{run}/{name}"
            reference = config.get("bc_reference")
            if reference:
                old = str(source).replace("\\", "/")
                reference = str(reference).replace("\\", "/")
                if reference.startswith(old + "/"):
                    reference = str(destination).replace("\\", "/") + reference[len(old):]
                reference = reference.replace(f"runs/{source_run}/", f"runs/{run}/")
                config["bc_reference"] = reference
            if name == branch:
                config["model"]["public_signals_version"] = 1
                config["public_signals"] = dict(version=1, barrier_discs=list(barrier_discs), barrier_segments=list(barrier_segments))
            (destination / name / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        shutil.copy2(destination / branch / "latest.pt", destination / "public_source.pt")
        atomic_torch_save(migrated, destination / branch / "latest.pt")
        manifest_path = destination / "specialization.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        # Preserve previous evaluation evidence, but use fresh caches for the new
        # observation/scorer source identity. No anchor, phase or LR resets.
        manifest["evaluation_directory"] = "evaluations_public_v1"
        manifest["public_signal_migration"] = {**summary, "source_sha256": original_hash}
        atomic_json(manifest, manifest_path)
        ledger["public_signal_migration"] = manifest["public_signal_migration"]
        atomic_json(refresh_ledger(destination, ledger), destination / "ledger.json")
        return summary
    finally:
        if lock:
            lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", default="rs4_v3")
    parser.add_argument("--run", default="rs4_v3_public")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--barrier-discs", nargs="*", type=int, default=[])
    parser.add_argument("--barrier-segments", nargs="*", type=int, default=[])
    args = parser.parse_args()
    try:
        result = prepare(args.source_run, args.run, dry_run=args.dry_run,
                         barrier_discs=args.barrier_discs, barrier_segments=args.barrier_segments)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2))
    if not args.dry_run:
        print(f"Continue: python -u -m tools.run_rs4_v3 --run {args.run} --resume")


if __name__ == "__main__":
    main()
