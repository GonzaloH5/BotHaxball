"""Orquestar pilotos y programa RS4 v3, secuencialmente y con ledger reanudable.

Sólo este comando inicia entrenamiento. Preparación y --dry-run jamás lo hacen.
Todas las evaluaciones/benchmarks ocurren cuando PPO está detenido.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import signal
import sys
import tempfile

import torch

from train.checkpoints import atomic_torch_save
from train.rs4_program import ProgramState, VERSION
from train.runtime import load_config
from .prepare_rs4 import ROOT, file_hash


def atomic_json(value, path):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def normalized_evaluation(report, reference=None):
    """Guardar el informe original y añadir sólo el resumen del contrato de fases."""
    result = copy.deepcopy(report)
    reference = reference or report.get("reference", {})
    result.setdefault("skills", copy.deepcopy(report.get("functional", {}).get("skills", {})))
    result.setdefault("baseline", copy.deepcopy(reference.get("functional", {}).get("skills", {})))
    if "matches" not in result:
        result["matches"] = dict(points=report.get("full_games", {}).get("mean_points"),
                                 baseline_points=reference.get("full_games", {}).get("mean_points"))
    return result


def select_candidate(control, memory, control_benchmark, memory_benchmark):
    """Umbrales acordados; datos incompletos/no finitos nunca aprueban memoria."""
    def numeric(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    cf, mf = control.get("functional", {}), memory.get("functional", {})
    cg, mg = control.get("full_games", {}), memory.get("full_games", {})
    c_speed, m_speed = control_benchmark.get("steps_per_second"), memory_benchmark.get("steps_per_second")
    valid_speed = numeric(c_speed) and numeric(m_speed) and c_speed > 0 and m_speed >= .7 * c_speed
    functional = (numeric(cf.get("mean_success")) and numeric(mf.get("mean_success"))
                  and mf["mean_success"] >= cf["mean_success"] + .05)
    matches = (numeric(cg.get("mean_points")) and numeric(mg.get("mean_points"))
               and mg["mean_points"] >= cg["mean_points"] - .03)
    seeds = []
    for seed in (51, 73, 91):
        a = cf.get("by_seed", {}).get(str(seed), cf.get("by_seed", {}).get(seed, {})).get("mean_success")
        b = mf.get("by_seed", {}).get(str(seed), mf.get("by_seed", {}).get(seed, {})).get("mean_success")
        if numeric(a) and numeric(b) and b > a:
            seeds.append(seed)
    devices_match = control_benchmark.get("device") == memory_benchmark.get("device")
    # El benchmark compara perfiles idénticos: fase A y misma mezcla de compañeros.
    mixing_match = control_benchmark.get("rs4_profile") == memory_benchmark.get("rs4_profile")
    safe_memory = memory_benchmark.get("memory_gate_passed") is True
    same_contract = (control_benchmark.get("comparison_contract") is not None
                     and control_benchmark.get("comparison_contract") == memory_benchmark.get("comparison_contract"))
    gates = dict(throughput=bool(valid_speed), functional=bool(functional), matches=bool(matches),
                 seed_consistency=len(seeds) >= 2, same_device=devices_match,
                 same_profile=mixing_match, same_contract=same_contract, memory=bool(safe_memory))
    return dict(selected="memory" if all(gates.values()) else "control", gates=gates,
                positive_seeds=seeds, throughput_ratio=m_speed / c_speed if numeric(c_speed) and numeric(m_speed) and c_speed > 0 else None)


def sporting_acceptance(report):
    """Éxito deportivo requiere evidencia repetida; no sólo consumir el presupuesto."""
    reference = report.get("reference", {})
    candidate_f, baseline_f = report.get("functional", {}), reference.get("functional", {})
    candidate_g, baseline_g = report.get("full_games", {}), reference.get("full_games", {})
    def improved(candidate, baseline, field):
        a, b = candidate.get(field), baseline.get(field)
        return all(isinstance(v, (int, float)) and math.isfinite(v) for v in (a, b)) and a > b
    seeds_functional, seeds_matches = [], []
    for seed in (51, 73, 91):
        key = str(seed)
        if improved(candidate_f.get("by_seed", {}).get(key, {}), baseline_f.get("by_seed", {}).get(key, {}), "mean_success"):
            seeds_functional.append(seed)
        if improved(candidate_g.get("by_seed", {}).get(key, {}), baseline_g.get("by_seed", {}).get(key, {}), "mean_points"):
            seeds_matches.append(seed)
    gates = dict(functional_mean=improved(candidate_f, baseline_f, "mean_success"),
                 match_mean=improved(candidate_g, baseline_g, "mean_points"),
                 functional_seeds=len(seeds_functional) >= 2, match_seeds=len(seeds_matches) >= 2)
    return dict(passed=all(gates.values()), gates=gates,
                positive_functional_seeds=seeds_functional, positive_match_seeds=seeds_matches)


def _checkpoint(directory, branch):
    path = directory / branch / "latest.pt"
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = load_config(directory / branch / "config.yaml")
    state = ProgramState.from_config(cfg, ck.get("rs4_program_state"))
    expected_steps = cfg["rs4_program"]["start_steps"] + state.relative_steps
    if ck["steps"] != expected_steps:
        raise ValueError(f"{branch}: checkpoint y ledger de muestras útiles no coinciden")
    return path, ck, cfg, state


def refresh_ledger(directory, ledger):
    for branch in ("control", "memory"):
        _, _, _, state = _checkpoint(directory, branch)
        ledger["candidates"][branch]["useful_steps"] = state.relative_steps
    work_path = directory / "benchmarks" / "work_ledger.json"
    if work_path.exists():
        work = json.loads(work_path.read_text(encoding="utf-8"))
        ledger["diagnostic_ppo_steps"] = sum(attempt["charged_steps"] for attempt in work["attempts"])
        ledger["diagnostic_uses_upper_bound"] = any(attempt["status"] != "reconciled" for attempt in work["attempts"])
    ledger["retained_steps"] = sum(item["useful_steps"] for item in ledger["candidates"].values())
    ledger["consumed_steps"] = ledger["retained_steps"] + ledger.get("diagnostic_ppo_steps", 0)
    if ledger["consumed_steps"] > ledger["total_budget_steps"]:
        raise ValueError("El programa excedió 6B muestras útiles; no se amplía ni se oculta el exceso")
    return ledger


def _invoke(command, *, env=None):
    print("\n" + subprocess.list2cmdline(command), flush=True)
    options = dict(cwd=ROOT)
    if env is not None:
        options["env"] = env
    if os.name == "nt":
        options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    child = subprocess.Popen(command, **options)
    try:
        code = child.wait()
    except KeyboardInterrupt:
        # Señalar únicamente nuestro hijo; no matar grupos ni otros entrenadores.
        if child.poll() is None:
            child.send_signal(signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGINT)
        while child.poll() is None:
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                print("Esperando que el proceso hijo termine de guardar el checkpoint; no se fuerza SIGKILL.", flush=True)
            except KeyboardInterrupt:
                print("La parada ya está solicitada; esperando el guardado del proceso hijo.", flush=True)
        raise
    if code:
        raise subprocess.CalledProcessError(code, command)


def _render_phase(directory, phase, ledger):
    """Un HTML acotado por fase; sólo invocado tras evaluación, con PPO detenido."""
    previous = ledger.setdefault("replays", {}).get(phase)
    if previous:
        saved = directory / previous["path"]
        if saved.is_file() and file_hash(saved) == previous["html_sha256"]:
            return  # no reemplazar una grabación válida por cambios de bookkeeping
    checkpoint = directory / ("champion.pt" if phase == "F" else f"phase_{phase}_champion.pt")
    if not checkpoint.is_file():
        return
    output_dir = directory / "replays"
    output_dir.mkdir(exist_ok=True)
    style = "ABCDEF".index(phase) % 3
    output = output_dir / f"phase_{phase}_vs_r3_style_{style}.html"
    environment = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", NUMBA_NUM_THREADS="4")
    try:
        _invoke([sys.executable, "-m", "eval.render", str(checkpoint), f"scripted:r3:{style}",
                 "--task", "rs4_4v4", "--minutes", "1", "--seed", "51", "--out", str(output)], env=environment)
    except subprocess.CalledProcessError as error:
        ledger.setdefault("replay_errors", {})[phase] = dict(returncode=error.returncode)
        print(f"Replay de fase {phase} no generado; el campeón permanece guardado.", flush=True)
        return
    ledger["replays"][phase] = dict(path=output.relative_to(directory).as_posix(), html_sha256=file_hash(output),
                                   checkpoint_sha256=file_hash(checkpoint), opponent=f"scripted:r3:{style}",
                                   seed=51, minutes=1)


def _evaluate(directory, branch, *, full, games, full_games, minutes, seeds, label):
    from eval.rs4_v3 import evaluation_source_fingerprint
    candidate, _, _, _ = _checkpoint(directory, branch)
    reports = directory / "evaluations"
    reports.mkdir(exist_ok=True)
    output = reports / f"{branch}_{label}.json"
    identity = dict(checkpoint_sha256=file_hash(candidate), reference_sha256=file_hash(directory / "parent.pt"),
                    games=full_games if full else games, minutes=minutes if full else min(minutes, .5),
                    seeds=list(seeds), evaluation_sources=evaluation_source_fingerprint())
    manifest = json.loads((directory / "specialization.json").read_text(encoding="utf-8"))
    historical = []
    for item in manifest.get("historical_opponents", []):
        rival = directory / item["path"]
        if file_hash(rival) != item["sha256"]:
            raise ValueError("Una referencia histórica cambió; no comparar con otra matriz")
        historical.append(str(rival))
    identity["historical_opponents"] = historical
    teacher = manifest.get("bc_teacher")
    if teacher:
        teacher_path = directory / teacher["path"]
        if file_hash(teacher_path) != teacher["sha256"]:
            raise ValueError("La referencia BC congelada cambió; no continuar con otros compañeros")
        identity["teacher_sha256"] = teacher["sha256"]
    reference_cache = reports / ("reference_full.json" if full else "reference_control.json")
    cache = output.with_suffix(".identity.json")
    if not output.exists() or not cache.exists() or json.loads(cache.read_text(encoding="utf-8")) != identity:
        command = [sys.executable, "-m", "tools.evaluate_rs4_v3", "--checkpoint", str(candidate),
                   "--reference", str(directory / "parent.pt"), "--out", str(output),
                   "--games", str(identity["games"]), "--minutes", str(identity["minutes"]),
                   "--seeds", *map(str, seeds), "--reference-report", str(reference_cache)]
        if historical:
            command.extend(["--opponents", *historical])
        if teacher:
            command.extend(["--teammate-reference", str(teacher_path)])
        _invoke(command)
        atomic_json(identity, cache)
    report = normalized_evaluation(json.loads(output.read_text(encoding="utf-8")))
    return output, report


def _persist_evaluation(directory, branch, output, report, ledger, *, full=False, closing_phase=None):
    candidate, ck, cfg, state = _checkpoint(directory, branch)
    old_phase = state.phase_index
    if not state.evaluations or state.last_evaluation_steps != state.relative_steps:
        state.record_evaluation(report)
    ck["rs4_program_state"] = state.state_dict()
    atomic_torch_save(ck, candidate)
    ledger["candidates"][branch]["evaluation"] = str(output.relative_to(directory))
    ledger["assessments"].append(dict(branch=branch, steps=state.relative_steps,
                                       phase=state.phase["id"], report=str(output.relative_to(directory))))
    if state.complete and full and ledger.get("selected") == branch:
        ledger["final_report"] = str(output.relative_to(directory))
    # Selección de campeón independiente, no por training reward/último checkpoint.
    skills = report.get("functional", {}).get("mean_success")
    points = report.get("full_games", {}).get("mean_points")
    if full and all(isinstance(v, (int, float)) and math.isfinite(v) for v in (skills, points)):
        score = (float(points), float(skills))
        previous = ledger.get("champion", {})
        old = tuple(previous.get("score", (-math.inf, -math.inf)))
        if ledger.get("selected") == branch and score > old:
            champion = directory / "champion.pt"
            atomic_torch_save(ck, champion)
            ledger["champion"] = dict(branch=branch, steps=state.relative_steps, score=list(score),
                                     report=str(output.relative_to(directory)), sha256=file_hash(champion))
        if ledger.get("selected") == branch and closing_phase is not None:
            phase = "ABCDEF"[closing_phase]
            previous = ledger.setdefault("phase_champions", {}).get(phase, {})
            if score > tuple(previous.get("score", (-math.inf, -math.inf))):
                path = directory / f"phase_{phase}_champion.pt"
                atomic_torch_save(ck, path)
                ledger["phase_champions"][phase] = dict(branch=branch, steps=state.relative_steps,
                    score=list(score), report=str(output.relative_to(directory)), sha256=file_hash(path))
    atomic_json(refresh_ledger(directory, ledger), directory / "ledger.json")
    if full and closing_phase is not None and ledger.get("selected") == branch:
        _render_phase(directory, "ABCDEF"[closing_phase], ledger)
        atomic_json(refresh_ledger(directory, ledger), directory / "ledger.json")
    return state.phase_index != old_phase


def _benchmarks(directory, warmup, iters):
    reports = directory / "benchmarks"
    reports.mkdir(exist_ok=True)
    work_path = reports / "work_ledger.json"
    work = json.loads(work_path.read_text(encoding="utf-8")) if work_path.exists() else dict(version=VERSION, attempts=[])
    code_root = Path(__file__).resolve().parents[1]
    code_files = ("tools/benchmark_multitask.py", "train/rs4_trainer.py", "train/rs4_program.py",
                  "train/model.py", "train/recurrent_model.py", "train/rs4_inference.py", "env/rs4_v3.py")
    source_identity = {path: file_hash(code_root / path) for path in code_files}
    results = {"control": [], "memory": []}
    # Alternar orden atenúa efectos de calentamiento/temperatura y carga del host.
    for repeat, branches in enumerate((("control", "memory"), ("memory", "control"), ("control", "memory"))):
        for branch in branches:
            candidate, _, cfg, state = _checkpoint(directory, branch)
            output = reports / f"{branch}_{repeat}.json"
            identity = dict(checkpoint_sha256=file_hash(candidate), warmup=warmup, iters=iters,
                            benchmark_sources=source_identity)
            cache = output.with_suffix(".identity.json")
            if not output.exists() or not cache.exists() or json.loads(cache.read_text(encoding="utf-8")) != identity:
                maximum = (warmup + iters) * cfg["ppo"]["rollout_len"] * cfg["env"]["agents"]
                available = cfg["rs4_program"]["phases"][-1]["steps"] - cfg["rs4_program"]["final_unassisted_steps"]
                if sum(attempt["charged_steps"] for attempt in work["attempts"]) + maximum > available:
                    raise ValueError("Los diagnósticos agotarían los 200M finales reservados; no se inicia otra prueba")
                attempt = dict(branch=branch, repeat=repeat, identity=identity, charged_steps=maximum,
                               reserved_steps=maximum, status="reserved")
                work["attempts"].append(attempt)
                # Reservar ANTES del subprocess: un crash/SIGINT no hace gratis
                # una actualización sobre la copia descartada.
                atomic_json(work, work_path)
                try:
                    _invoke([sys.executable, "-m", "tools.benchmark_multitask", "--config", str(directory / branch / "config.yaml"),
                             "--checkpoint", str(candidate), "--warmup", str(warmup), "--iters", str(iters),
                             "--json-output", str(output), "--max-memory-gib", "7.0", "--companion-fraction", "0.1"])
                except subprocess.CalledProcessError as error:
                    if branch != "memory":
                        raise  # sin baseline válida no se puede aceptar ningún perfil
                    failed = json.loads(output.read_text(encoding="utf-8")) if output.exists() else {}
                    # Si el proceso murió antes de publicar conteos, cobrar una
                    # cota superior conservadora: nunca fingir coste cero.
                    failed.update(steps_per_second=0., memory_gate_passed=False,
                                  diagnostic_ppo_samples=failed.get("diagnostic_ppo_samples", maximum),
                                  diagnostic_upper_bound="diagnostic_ppo_samples" not in failed,
                                  benchmark_failed=True, failure_returncode=error.returncode)
                    atomic_json(failed, output)
                measured = json.loads(output.read_text(encoding="utf-8"))
                actual = measured.get("diagnostic_ppo_samples")
                if actual is not None and not measured.get("diagnostic_upper_bound", False):
                    if isinstance(actual, bool) or not isinstance(actual, int) or not 0 <= actual <= maximum:
                        raise ValueError("El benchmark publicó un conteo de muestras incompatible con la reserva")
                    attempt["charged_steps"], attempt["status"] = actual, "reconciled"
                atomic_json(work, work_path)
                atomic_json(identity, cache)
            row = json.loads(output.read_text(encoding="utf-8"))
            row["rs4_profile"] = {key: state.settings()[key] for key in
                                 ("phase_id", "exercise_fraction", "frozen_teammates_fraction", "opponent_mix")}
            results[branch].append(row)
    summaries = {}
    for branch, rows in results.items():
        summary = copy.deepcopy(rows[-1])
        summary["steps_per_second"] = statistics.median(row["steps_per_second"] for row in rows)
        summary["repetitions"] = rows
        summaries[branch] = summary
        atomic_json(summary, reports / f"{branch}_summary.json")
    summaries["control"]["diagnostic_work_total"] = sum(attempt["charged_steps"] for attempt in work["attempts"])
    summaries["control"]["diagnostic_work_ledger"] = str(work_path.relative_to(directory))
    return summaries


def _train_chunk(directory, branch, target_relative_steps):
    _, _, cfg, state = _checkpoint(directory, branch)
    if target_relative_steps > state.effective_branch_limit or target_relative_steps <= state.relative_steps:
        raise ValueError("El chunk debe avanzar sin ampliar el presupuesto de la rama")
    end = cfg["rs4_program"]["start_steps"] + target_relative_steps
    _invoke([sys.executable, "-u", "-m", "train.multitask", "--config", str(directory / branch / "config.yaml"),
             "--run", cfg["run_name"], "--resume", "--override", f"ppo.total_steps={end}"])
    _, _, _, after = _checkpoint(directory, branch)
    if after.relative_steps < target_relative_steps:
        print("El entrenador guardó antes de completar el segmento; se detiene el runner, no inicia otro chunk.", flush=True)
        raise KeyboardInterrupt


def execution_outline(directory):
    manifest = json.loads((directory / "specialization.json").read_text(encoding="utf-8"))
    return dict(version=VERSION, source_steps=manifest["source_steps"],
                pilot_steps=manifest["pilot_steps"], total_budget_steps=manifest["total_budget_steps"],
                selected_branch_budget_steps=manifest["selected_branch_budget_steps"],
                order=["control: piloto 200M con evaluaciones cada100M", "memory: piloto 200M con evaluaciones cada100M",
                       "benchmark alternado 3 repeticiones y evaluación completa de ambas",
                       "selección por velocidad/habilidades/partidos/semillas", "rama elegida hasta5.8B; descartada detenida",
                       "evaluación final, campeón y fallos pendientes"],
                requires_stopped_source=True, starts_training=False)


def run(run_name="rs4_v3", *, resume=False, dry_run=False, games=16, full_games=128,
        minutes=2., seeds=(51, 73, 91), warmup=3, iters=8):
    if not run_name or run_name in (".", "..") or any(c in run_name for c in '/\\:'):
        raise ValueError("run debe ser un nombre simple")
    if games < 2 or games % 2 or full_games < games or full_games % 2 or minutes <= 0:
        raise ValueError("games/full-games pares >=2, full-games>=games, minutos positivos")
    if tuple(seeds) != (51, 73, 91):
        raise ValueError("La selección exige semillas 51,73,91")
    directory = ROOT / "runs" / run_name
    manifest = json.loads((directory / "specialization.json").read_text(encoding="utf-8"))
    if manifest.get("kind") != "rs4_v3_program" or manifest.get("version") != VERSION:
        raise ValueError("No es un programa RS4 v3 preparado")
    if file_hash(directory / "parent.pt") != manifest["source_sha256"]:
        raise ValueError("La referencia inicial cambió; no comparar ni continuar")
    ledger = refresh_ledger(directory, json.loads((directory / "ledger.json").read_text(encoding="utf-8")))
    if dry_run:
        return execution_outline(directory)
    if not resume:
        raise ValueError("El programa sólo continúa sus checkpoints preparados: usar --resume")
    # Advisory lock liberado por el SO también tras Ctrl+C/crash; el archivo no
    # se borra, por lo que no hay una carrera sobre un lock recién recreado.
    lock = (directory / ".runner.lock").open("a+b")
    try:
        if os.name == "nt":
            import msvcrt
            if lock.tell() == 0:
                lock.write(b"0")
                lock.flush()
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if ledger["complete"]:
            for phase in sorted(ledger.get("phase_champions", {})):
                _render_phase(directory, phase, ledger)
            atomic_json(ledger, directory / "ledger.json")
            return ledger
        pilot = manifest["pilot_steps"]
        if ledger["selected"] is None:
            for branch in ("control", "memory"):
                _, _, cfg, state = _checkpoint(directory, branch)
                while state.relative_steps < pilot:
                    target = min(pilot, state.last_evaluation_steps + cfg["rs4_program"]["evaluation_every_steps"])
                    if target > state.relative_steps:
                        _train_chunk(directory, branch, target)
                        atomic_json(refresh_ledger(directory, ledger), directory / "ledger.json")
                    output, report = _evaluate(directory, branch, full=False, games=games, full_games=full_games,
                                               minutes=minutes, seeds=seeds, label=f"pilot_{target}")
                    _persist_evaluation(directory, branch, output, report, ledger)
                    _, _, cfg, state = _checkpoint(directory, branch)
                if state.evaluation_due:
                    output, report = _evaluate(directory, branch, full=False, games=games, full_games=full_games,
                                               minutes=minutes, seeds=seeds, label=f"pilot_{state.relative_steps}")
                    _persist_evaluation(directory, branch, output, report, ledger)
                ledger["candidates"][branch]["complete"] = True
            reports = {}
            for branch in ("control", "memory"):
                output, reports[branch] = _evaluate(directory, branch, full=True, games=games, full_games=full_games,
                                                   minutes=minutes, seeds=seeds, label="pilot_full")
                ledger["candidates"][branch]["evaluation"] = str(output.relative_to(directory))
            benchmarks = _benchmarks(directory, warmup, iters)
            ledger["diagnostic_ppo_steps"] = benchmarks["control"].get("diagnostic_work_total", sum(
                row.get("diagnostic_ppo_samples", row.get("samples", 0) + row.get("warmup_samples", 0))
                for benchmark in benchmarks.values() for row in benchmark.get("repetitions", [benchmark])))
            ledger["diagnostic_budget_definition"] = ("PPO sobre copias temporales descartadas para medir velocidad; "
                                                       "se cobra dentro de 6B y se resta de consolidación, sin mover anclas.")
            selection = select_candidate(reports["control"], reports["memory"], benchmarks["control"], benchmarks["memory"])
            ledger["selection"], ledger["selected"] = selection, selection["selected"]
            # El punto de partida también compite por campeón. Un run nuevo que
            # empeore todo no debe desplazar al modelo inicial por ser más reciente.
            baseline = reports["control"].get("reference", {})
            score = (baseline.get("full_games", {}).get("mean_points"),
                     baseline.get("functional", {}).get("mean_success"))
            if all(isinstance(v, (int, float)) and math.isfinite(v) for v in score):
                parent = torch.load(directory / "parent.pt", map_location="cpu", weights_only=False)
                atomic_torch_save(parent, directory / "champion.pt")
                ledger["champion"] = dict(branch="reference", steps=0, score=list(score),
                                           report="evaluations/reference_full.json", sha256=file_hash(directory / "champion.pt"))
            for branch in ("control", "memory"):
                ledger["candidates"][branch]["benchmark"] = f"benchmarks/{branch}_summary.json"
            atomic_json(refresh_ledger(directory, ledger), directory / "ledger.json")
            print(f"Selección: {ledger['selected']} | gates {selection['gates']}", flush=True)
        winner = ledger["selected"]
        candidate, checkpoint, cfg, state = _checkpoint(directory, winner)
        state.charge_diagnostics(ledger.get("diagnostic_ppo_steps", 0))
        checkpoint["rs4_program_state"] = state.state_dict()
        atomic_torch_save(checkpoint, candidate)
        for phase in sorted(ledger.get("phase_champions", {})):
            _render_phase(directory, phase, ledger)
        atomic_json(refresh_ledger(directory, ledger), directory / "ledger.json")
        while not state.complete:
            if state.transitions:
                transition = state.transitions[-1]
                phase = transition["from_phase"]
                if (transition["steps"] == state.relative_steps
                        and phase not in ledger.get("phase_champions", {})):
                    output, report = _evaluate(directory, winner, full=True, games=games, full_games=full_games,
                                               minutes=minutes, seeds=seeds, label=f"phase_{phase}_recovered_{state.relative_steps}")
                    _persist_evaluation(directory, winner, output, report, ledger, full=True,
                                          closing_phase="ABCDEF".index(phase))
                    _, _, cfg, state = _checkpoint(directory, winner)
            old_phase = state.phase_index
            interval = cfg["rs4_program"]["evaluation_every_steps"]
            nominal_end = state.phase_start_steps + state.phase["steps"]
            target = min(state.effective_branch_limit, state.last_evaluation_steps + interval)
            if old_phase < 5:
                target = min(target, nominal_end)
            if target > state.relative_steps:
                _train_chunk(directory, winner, target)
                atomic_json(refresh_ledger(directory, ledger), directory / "ledger.json")
            _, _, cfg, state = _checkpoint(directory, winner)
            pending_close = None
            if state.transitions and state.transitions[-1]["steps"] > state.last_evaluation_steps:
                pending_close = "ABCDEF".index(state.transitions[-1]["from_phase"])
            full = state.complete or state.phase_index != old_phase or pending_close is not None
            output, report = _evaluate(directory, winner, full=full, games=games, full_games=full_games,
                                       minutes=minutes, seeds=seeds, label=f"steps_{state.relative_steps}")
            advanced = _persist_evaluation(directory, winner, output, report, ledger, full=full,
                                            closing_phase=(pending_close if pending_close is not None else old_phase) if full else None)
            if advanced and not full:
                output, report = _evaluate(directory, winner, full=True, games=games, full_games=full_games,
                                           minutes=minutes, seeds=seeds, label=f"phase_{old_phase}_full_{state.relative_steps}")
                _persist_evaluation(directory, winner, output, report, ledger, full=True, closing_phase=old_phase)
            _, _, cfg, state = _checkpoint(directory, winner)
        ledger["complete"] = True
        ledger["pending_skills"] = list(state.skill_debts)
        if not ledger.get("final_report"):
            output, report = _evaluate(directory, winner, full=True, games=games, full_games=full_games,
                                       minutes=minutes, seeds=seeds, label="final")
            _persist_evaluation(directory, winner, output, report, ledger, full=True, closing_phase=5)
        final_report = json.loads((directory / ledger["final_report"]).read_text(encoding="utf-8"))
        ledger["sporting_acceptance"] = sporting_acceptance(final_report)
        ledger["sporting_success"] = not state.skill_debts and ledger["sporting_acceptance"]["passed"]
        atomic_json(refresh_ledger(directory, ledger), directory / "ledger.json")
        print(f"Programa finalizado; consumido {ledger['consumed_steps']:,}/6B; habilidades pendientes {state.skill_debts}", flush=True)
        return ledger
    finally:
        # Reflejar reservas de un benchmark interrumpido también en el ledger
        # visible; no empezar otro subprocess ni modificar el modelo detenido.
        try:
            atomic_json(refresh_ledger(directory, ledger), directory / "ledger.json")
        except (OSError, ValueError):
            print("No se pudo refrescar ledger.json; los checkpoints y las reservas diagnósticas siguen conservados.", flush=True)
        lock.close()


def main():
    def stop_requested(signum, frame):
        raise KeyboardInterrupt
    # nohup puede heredar SIGINT ignorado: instalar explícitamente ambas señales.
    signal.signal(signal.SIGINT, stop_requested)
    signal.signal(signal.SIGTERM, stop_requested)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="rs4_v3")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--games", type=int, default=16, help="Partidos pares por rival/semilla en controles")
    parser.add_argument("--full-games", type=int, default=128)
    parser.add_argument("--minutes", type=float, default=2.)
    parser.add_argument("--seeds", type=int, nargs="+", default=[51, 73, 91])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=8)
    args = parser.parse_args()
    try:
        result = run(args.run, resume=args.resume, dry_run=args.dry_run, games=args.games,
                     full_games=args.full_games, minutes=args.minutes, seeds=args.seeds,
                     warmup=args.warmup, iters=args.iters)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.error(str(error))
    except KeyboardInterrupt:
        print("RS4 v3 detenido. No se inició otro segmento; reanudar con --resume.", flush=True)
        return
    if args.dry_run:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
