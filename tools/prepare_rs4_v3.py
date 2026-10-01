"""Preparar RS4 v3: dos pilotos, referencia congelada y presupuesto compartido.

La preparación jamás entrena ni reemplaza runs existentes. --dry-run sólo lee.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

import torch
import yaml

from train.checkpoints import atomic_torch_save
from train.rs4_program import ProgramState, VERSION
from train.runtime import load_config
from .prepare_rs4 import ROOT, TASK, file_hash

PROFILE = ROOT / "train/config_rs4_v3.yaml"


def _merge(base, update):
    result = copy.deepcopy(base)
    for key, value in update.items():
        result[key] = _merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else copy.deepcopy(value)
    return result


def _code_version():
    try:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=True).stdout)
        return dict(revision=revision, dirty=dirty)
    except (OSError, subprocess.CalledProcessError):
        return dict(revision=None, dirty=None)


def candidate_config(source_cfg, profile, run, branch, start_steps):
    """Conservar arquitectura base y referencias; cambiar entrenamiento deliberadamente."""
    cfg = _merge(source_cfg, {key: value for key, value in profile.items()
                             if key not in ("model", "bc_reference", "env", "tasks_file", "scripted_readiness")})
    cfg["model"] = copy.deepcopy(source_cfg["model"])
    cfg["model"]["type"] = "set" if branch == "control" else "recurrent_set"
    if branch == "memory":
        cfg["model"].update(memory_size=32, pooling="attentive_meanmax")
    cfg["run_name"] = f"{run}/{branch}"
    cfg["ppo"].pop("lr_schedule", None)  # nuevo controlador propio, anclado en rs4_program
    cfg["rs4_program"]["start_steps"] = int(start_steps)
    cfg["rs4_program"]["candidate"] = branch
    cfg["ppo"]["total_steps"] = int(start_steps) + cfg["rs4_program"]["pilot_steps"]
    cfg.setdefault("runtime", {})["preserve_bc_reference"] = True
    cfg["runtime"]["freeze_normalizers"] = True
    cfg["rs4_tactics"]["start_steps"] = int(start_steps)
    cfg["rs4_tactics"]["decay_steps"] = cfg["rs4_program"]["branch_budget_steps"]
    ProgramState.from_config(cfg)
    return cfg


def prepare(source="runs/rs4_adapt/latest.pt", run="rs4_v3", *, dry_run=False,
            bc_reference=None, bc_validation=None):
    if not run or run in (".", "..") or any(char in run for char in '/\\:'):
        raise ValueError("run debe ser un nombre simple, sin rutas")
    source = Path(source).resolve()
    destination = ROOT / "runs" / run
    if destination.exists():
        raise FileExistsError(f"{destination} ya existe; usar run_rs4_v3 --resume, no preparar otra vez")
    config_source = source.with_name("config.yaml")
    if not source.is_file() or not config_source.is_file():
        raise FileNotFoundError("Se necesitan latest.pt y config.yaml de una fuente RS4 detenida")
    digest = file_hash(source)
    source_config_bytes = config_source.read_bytes()
    cfg = load_config(config_source)
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    required = ("model", "model_config", "opt", "league", "steps", "iteration", "task_state")
    if any(key not in checkpoint for key in required):
        raise ValueError("La fuente debe ser un checkpoint PPO completo, incluido Adam y liga")
    if checkpoint["model_config"].get("type") != "set" or checkpoint.get("env", {}).get("obs_layout") != "universal":
        raise ValueError("La migración v3 requiere la base SetActorCritic universal sin memoria")
    if TASK not in checkpoint["task_state"] or len(cfg["stages"]) != 1 or set(cfg["stages"][0]["tasks"]) != {TASK}:
        raise ValueError("La fuente debe estar especializada exclusivamente en rs4_4v4")
    if checkpoint.get("rs4_program_state") is not None or cfg.get("rs4_program"):
        raise ValueError("La fuente ya pertenece a un programa v3; reanudar sin reiniciar sus anclas")
    if isinstance(checkpoint["steps"], bool) or not isinstance(checkpoint["steps"], int) or checkpoint["steps"] < 0:
        raise ValueError("Pasos heredados inválidos")
    if bool(bc_reference) != bool(bc_validation):
        raise ValueError("Una referencia BC nueva requiere --bc-reference y --bc-validation juntos")
    teacher = None
    validation_identity = None
    chosen_teacher = bc_reference if bc_reference is not None else cfg.get("bc_reference")
    if chosen_teacher:
        teacher = Path(str(chosen_teacher).replace("\\", "/"))
        teacher = teacher if teacher.is_absolute() else ROOT / teacher
        if not teacher.is_file():
            raise FileNotFoundError(f"Falta la referencia BC heredada: {teacher}")
        teacher_identity = file_hash(teacher)
        teacher_ck = torch.load(teacher, map_location="cpu", weights_only=False)
        teacher_model = teacher_ck.get("model_config", {})
        if teacher_model.get("type") != "set":
            raise ValueError("La referencia BC débil debe ser feedforward SetActorCritic; no activar memoria como teacher KL")
        for field in ("self_dim", "ent_dim", "n_actions"):
            if teacher_model.get(field) != checkpoint["model_config"].get(field):
                raise ValueError(f"La referencia BC no conserva el contrato de observación/acciones: {field}")
        if bc_validation:
            validation_path = Path(bc_validation).resolve()
            validation = json.loads(validation_path.read_text(encoding="utf-8"))
            if (validation.get("gate", {}).get("approved_as_weak_reference") is not True
                    or validation.get("candidate_sha256") != teacher_identity):
                raise ValueError("El teacher nuevo no tiene una validación aprobada para ese checkpoint exacto")
            current_teacher = cfg.get("bc_reference")
            if current_teacher:
                current_teacher = Path(str(current_teacher).replace("\\", "/"))
                current_teacher = current_teacher if current_teacher.is_absolute() else ROOT / current_teacher
                if not current_teacher.is_file() or validation.get("source_sha256") != file_hash(current_teacher):
                    raise ValueError("La validación BC debe comparar contra el teacher actual, no contra otra referencia")
            validation_identity = dict(path=str(validation_path), sha256=file_hash(validation_path))
        cfg = copy.deepcopy(cfg)
        cfg["bc_reference"] = f"runs/{run}/teacher.pt"
    from train.rs4_migration import migrate_checkpoint
    profile = load_config(PROFILE)
    configurations = {branch: candidate_config(cfg, profile, run, branch, checkpoint["steps"])
                      for branch in ("control", "memory")}
    migrated = {}
    for branch, target_cfg in configurations.items():
        migrated[branch] = migrate_checkpoint(checkpoint, target_cfg)
        migrated[branch]["rs4_program_state"] = ProgramState.from_config(target_cfg).state_dict()
        migrated[branch]["stage"], migrated[branch]["stage_steps"] = 0, 0
        # Referencia inicial protegida y una historia acotada, antes de que el
        # loader legacy pudiera expulsar una referencia durante la restauración.
        history = []
        for saved in checkpoint["league"]:
            if not isinstance(saved, dict):
                raise ValueError("La liga fuente tiene un formato anterior sin metadatos; actualizar el checkpoint con el entrenador actual antes de preparar v3")
            history.append(copy.deepcopy(saved))
        limit = target_cfg["league"]["max_size"] - 1
        protected = [i for i, member in enumerate(history) if member.get("protected", False)]
        if len(protected) > limit:
            raise ValueError("La fuente contiene más referencias protegidas que el límite de liga v3")
        recent = [i for i in range(len(history)) if i not in protected][-max(0, limit - len(protected)):]
        if limit == len(protected):
            recent = []
        keep = set(protected + recent)
        history = [member for i, member in enumerate(history) if i in keep]
        history.append(dict(name="v3_initial_reference", model=copy.deepcopy(checkpoint["model"]),
                            model_config=copy.deepcopy(checkpoint["model_config"]),
                            elo=float(checkpoint.get("learner_elo", 1000.)), wins=1., games=2.,
                            match_points=1., matches=2., protected=True, snapshot_steps=0))
        migrated[branch]["league"] = history
    manifest = dict(kind="rs4_v3_program", version=VERSION, source=str(source), source_sha256=digest,
                    source_config_sha256=hashlib.sha256(source_config_bytes).hexdigest(),
                    source_steps=checkpoint["steps"], source_iteration=checkpoint["iteration"],
                    reference_label="RS4 anterior congelado", code_version=_code_version(),
                    total_budget_steps=profile["rs4_program"]["total_budget_steps"],
                    selected_branch_budget_steps=profile["rs4_program"]["branch_budget_steps"],
                    pilot_steps=profile["rs4_program"]["pilot_steps"], candidates=["control", "memory"])
    historical = checkpoint["league"]
    selected_history = sorted(set((0, len(historical) - 1))) if historical else []
    manifest["historical_opponents"] = [dict(path=f"reference_history/anchor_{j}.pt", name=historical[i]["name"])
                                        for j, i in enumerate(selected_history)]
    manifest["bc_teacher"] = (dict(source=str(teacher), sha256=teacher_identity, path="teacher.pt",
                                   validation=validation_identity) if teacher else None)
    for target in migrated.values():
        target["program_manifest"] = copy.deepcopy(manifest)
        target["evaluation_version"] = VERSION
        target["source_reference"] = dict(name="v3_initial_reference", sha256=digest)
    estimate = source.stat().st_size * 4 + 20 * 1024**2 + (teacher.stat().st_size if teacher else 0)
    existing_parent = destination.parent if destination.parent.exists() else ROOT
    if shutil.disk_usage(existing_parent).free < estimate:
        raise OSError(f"Espacio insuficiente para preparación segura: reservar al menos {estimate:,} bytes")
    if (digest != file_hash(source) or source_config_bytes != config_source.read_bytes()
            or teacher is not None and teacher_identity != file_hash(teacher)):
        raise ValueError("La fuente cambió durante la lectura; detener el entrenamiento y repetir")
    if dry_run:
        return dict(destination=str(destination), manifest=manifest, candidates=configurations,
                    estimated_bytes=estimate, source_unchanged=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="_prepare_rs4_v3_", dir=destination.parent) as directory:
        temporary = Path(directory)
        shutil.copy2(source, temporary / "parent.pt")
        shutil.copy2(config_source, temporary / "parent_config.yaml")
        if teacher:
            shutil.copy2(teacher, temporary / "teacher.pt")
            if file_hash(temporary / "teacher.pt") != teacher_identity:
                raise ValueError("La referencia BC cambió durante la copia; no se preparó el programa")
        if bc_validation:
            shutil.copy2(validation_path, temporary / "teacher_validation.json")
            if file_hash(temporary / "teacher_validation.json") != validation_identity["sha256"]:
                raise ValueError("La validación BC cambió durante la copia")
        if file_hash(temporary / "parent.pt") != digest or (temporary / "parent_config.yaml").read_bytes() != source_config_bytes:
            raise ValueError("La fuente cambió durante la copia; no se preparó el programa")
        if selected_history:
            history_dir = temporary / "reference_history"
            history_dir.mkdir()
            for item, index in zip(manifest["historical_opponents"], selected_history):
                member = historical[index]
                opponent = dict(model=member["model"], model_config=member.get("model_config", checkpoint["model_config"]))
                atomic_torch_save(opponent, temporary / item["path"])
                item["sha256"] = file_hash(temporary / item["path"])
        for branch in ("control", "memory"):
            candidate = temporary / branch
            candidate.mkdir()
            migrated[branch]["program_manifest"] = copy.deepcopy(manifest)
            atomic_torch_save(migrated[branch], candidate / "latest.pt")
            (candidate / "config.yaml").write_text(yaml.safe_dump(configurations[branch], sort_keys=False), encoding="utf-8")
        ledger = dict(version=VERSION, source_steps=checkpoint["steps"], total_budget_steps=manifest["total_budget_steps"],
                      candidates={branch: dict(useful_steps=0, complete=False, evaluation=None, benchmark=None)
                                  for branch in ("control", "memory")},
                      selected=None, selection=None, consumed_steps=0, complete=False, assessments=[])
        (temporary / "specialization.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        (temporary / "ledger.json").write_text(json.dumps(ledger, indent=2), encoding="utf-8")
        if (digest != file_hash(source) or source_config_bytes != config_source.read_bytes()
                or teacher is not None and teacher_identity != file_hash(teacher)):
            raise ValueError("La fuente cambió durante preparación; no activar las copias")
        # mkdir exclusivo evita sobrescribir un programa creado en paralelo.
        destination.mkdir()
        for entry in temporary.iterdir():
            entry.rename(destination / entry.name)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="runs/rs4_adapt/latest.pt")
    parser.add_argument("--run", default="rs4_v3")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--bc-reference", help="Teacher feedforward nuevo, sólo con validación aprobada")
    parser.add_argument("--bc-validation", help="validation.json ligado al SHA256 del teacher nuevo")
    args = parser.parse_args()
    try:
        result = prepare(args.source, args.run, dry_run=args.dry_run,
                         bc_reference=args.bc_reference, bc_validation=args.bc_validation)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    if args.dry_run:
        print(json.dumps(result, indent=2))
    else:
        print(f"Preparado {result}; fuente intacta, pilotos 200M + 200M, presupuesto total 6B.")
        print("No se inició entrenamiento. Ejecutar en el Pod:")
        print(f"python -u -m tools.run_rs4_v3 --run {args.run} --resume")


if __name__ == "__main__":
    main()
