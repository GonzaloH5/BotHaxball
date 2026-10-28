"""Bounded A/B: cooperative curriculum versus the same trainer without exercises.

Runs a 100-update pilot, checks stability/BC strength, then continues to the
predeclared budget and evaluates on test exactly once. Does not export or deploy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parent.parent


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def read_rows(path):
    return [json.loads(s) for s in Path(path).read_text(encoding="utf-8").splitlines() if s.strip()]


def common_args(a):
    return ["--bc", a.bc, "--init", a.init, "--device", a.device,
            "--envs", str(a.envs), "--rollout", str(a.rollout), "--threads", str(a.threads),
            "--lambda-dist", "0.05", "--lambda-decay", "1", "--lambda-min", "0.05",
            "--gamma", "0.999", "--gae", "0.98", "--ent", "0.002",
            "--critic-warmup", "20", "--human-starts", "0.4", "--bank-recordings", str(a.recordings),
            "--pool-frac", "0.5", "--pool", a.init, "--delays", "8:1,9:1,10:1,11:1",
            "--delay-feature", "mixed", "--eval-every", str(a.eval_every),
            "--eval-matches", "16", "--eval-selfplay", "16", "--dev-recordings", "0",
            "--rsone-every", "0", "--no-keep-evals", "--snapshot-every", "100", "--pool-max", "6",
            "--shaping-kind", "none", "--style-coef", "0", "--auto-pass-arm-update", "0",
            "--forfeit-reasons", "1,2,3,4", "--stop-on-drift", "0",
            "--pass-ref-key", "sanguchito_train", "--pass-macro", "--pass-ready", "--macro-trace",
            "--seed", str(a.seed)]


def branch_command(a, branch, updates):
    command = [sys.executable, "-m", "learn.x4_ppo", *common_args(a),
               "--out", str(Path(a.root) / branch), "--updates", str(updates), "--resume"]
    if branch == "curriculum":
        command += ["--coop-frac", "0.5", "--coop-min-episodes", "256",
                    "--coop-threshold", "0.15", "--coop-bank-limit", "6000"]
    else:
        command += ["--coop-frac", "0"]
    return command


def pilot_gate(root, target):
    reasons = []
    summary = {}
    for branch in ("control", "curriculum"):
        path = Path(root) / branch
        logs = read_rows(path / "log.jsonl")
        evals = read_rows(path / "eval.jsonl")
        final = logs[-1]
        initial_ev, final_ev = evals[0], evals[-1]
        initial = initial_ev["vs_bc"]["score"]
        final_score = final_ev["vs_bc"]["score"]
        nonfinite = sum(x.get("nonfinite_skipped", 0) for x in logs)
        task_samples = sum(x.get("coop_actor_samples", 0) for x in logs)
        successes = sum(x.get("coop", {}).get("successes", 0) for x in logs)
        episodes = sum(x.get("coop", {}).get("episodes", 0) for x in logs)
        summary[branch] = dict(update=final["update"], initial_bc=initial, final_bc=final_score,
                               skipped_nonfinite=nonfinite, task_actor_samples=task_samples,
                               exercise_successes=successes, exercise_episodes=episodes,
                               stages=final.get("curriculum_stage", {}))
        if final["update"] != target or final_ev.get("update", -1) != target:
            reasons.append(branch + ": incomplete pilot/evaluation")
        if not all(ev["vs_bc"].get("valid_for_strength", False) for ev in (initial_ev, final_ev)):
            reasons.append(branch + ": incomplete BC matches")
        if nonfinite or any(x.get("restored_nonfinite") for x in logs):
            reasons.append(branch + ": nonfinite update")
        if final_score < 0.65 or final_score < initial - 0.15:
            reasons.append(branch + ": material loss of BC strength")
        if branch == "curriculum" and (task_samples <= 0 or episodes < 32 or successes < 1):
            reasons.append("curriculum: exercises lack measured executions/successes")
    return dict(continue_training=not reasons, reasons=reasons, branches=summary,
                limitation="Pilot gate checks feasibility, not improvement or human competitiveness.")


def final_verdict(report, candidate, control, original):
    matches = report["matches"]
    direct = matches[candidate][control]
    against_initial = matches[candidate][original]
    score = direct["point_fraction"]["valor"]
    interval = direct["point_fraction"]["ic90"]
    stable = all(x["valid_for_strength"] for rows in matches.values() for x in rows.values())
    original_score = against_initial["point_fraction"]["valor"]
    robust = bool(stable and score is not None and score >= 0.55
                  and interval[0] is not None and interval[0] > 0.5
                  and original_score is not None and original_score >= 0.5)
    return dict(verdict="improved_simulated_strength" if robust else "no_confirmed_improvement",
                strength_checks=dict(complete_matches=stable, candidate_vs_control=score,
                                     candidate_vs_control_ci90=interval,
                                     candidate_vs_original=original_score),
                human_competitiveness="unverified",
                cooperative_metrics=direct.get("natural_metrics", {}),
                limitation="A strength verdict does not certify cooperation; examine retention, pressure losses, and probes.")


def evaluation_command(a, split, out):
    candidate, control = [str(Path(a.root) / b / "last.pt") for b in ("curriculum", "control")]
    pilot = split == "dev"
    return [sys.executable, "-m", "learn.x4_coop_eval",
            "--policy", candidate, "--reference", control, "--reference", a.init,
            "--bc", a.bc, "--matches", str(min(a.matches, 4) if pilot else a.matches),
            "--probe-states", str(min(a.probe_states, 32) if pilot else a.probe_states),
            "--recordings", str(min(a.eval_recordings, 15) if pilot else a.eval_recordings),
            "--minutes", "1" if pilot else "3", "--device", a.device,
            "--threads", str(a.threads), "--split", split,
            "--seed", str(a.seed + (5000 if pilot else 10000)), "--out", str(out)]


class Run:
    def __init__(self, a):
        self.a = a
        self.root = Path(a.root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.deadline = time.monotonic() + a.wall_hours * 3600
        self.state_path = self.root / "run_state.json"
        self.state = dict(status="starting", started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                          pid=os.getpid(), phases={})
        self.save()

    def save(self):
        atomic_json(self.state_path, self.state)

    def execute(self, name, commands):
        handles, processes = [], []
        try:
            for tag, command in commands.items():
                output = open(self.root / (tag + ".out"), "a", encoding="utf-8")
                handles.append(output)
                env = dict(os.environ, OMP_NUM_THREADS="1", NUMBA_NUM_THREADS=str(self.a.threads))
                processes.append((tag, subprocess.Popen(command, cwd=ROOT, env=env,
                                                        stdout=output, stderr=subprocess.STDOUT)))
            self.state["status"] = name
            self.state["phases"][name] = dict(commands=commands,
                                               pids={tag: p.pid for tag, p in processes}, status="running")
            self.save()
            while any(p.poll() is None for _, p in processes):
                if time.monotonic() >= self.deadline:
                    raise TimeoutError("predeclared wall-clock budget exhausted")
                if any(p.poll() not in (None, 0) for _, p in processes):
                    raise RuntimeError("a child process failed; inspect its .out file")
                time.sleep(2)
            codes = {tag: p.returncode for tag, p in processes}
            self.state["phases"][name].update(status="finished", exit_codes=codes)
            self.save()
            if any(codes.values()):
                raise RuntimeError(f"{name} failed: {codes}")
        finally:
            for _, p in processes:
                if p.poll() is None:
                    p.terminate()
                    try:
                        p.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        p.kill()
                        p.wait()
            for handle in handles:
                handle.close()

    def main(self):
        a = self.a
        source_arguments = dict(vars(a))
        references = self.root / "references"
        references.mkdir()
        fingerprints = {}
        for key in ("bc", "init"):
            source = ROOT / getattr(a, key)
            destination = references / (key + ".pt")
            shutil.copy2(source, destination)
            fingerprints[key] = dict(source=str(source), frozen_copy=str(destination),
                                     sha256=hashlib.sha256(destination.read_bytes()).hexdigest())
            setattr(a, key, str(destination))
        files = ("learn/x4_ppo.py", "learn/x4_policy.py", "learn/x4_pass_macro.py",
                 "learn/x4_coop_curriculum.py", "learn/x4_macro_trace.py", "learn/x4_coop_eval.py",
                 "learn/x4_coop_run.py", "env/rs4z/core.py", "env/rs4z/kernel.py")
        manifest = dict(arguments=vars(a), code_sha256={f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest()
                                                       for f in files},
                        source_arguments=source_arguments, frozen_checkpoints=fingerprints,
                        split_manifest_sha256=hashlib.sha256((ROOT / "reports/x4/splits.json").read_bytes()).hexdigest(),
                        pilot_commands={b: branch_command(a, b, a.pilot_updates)
                                        for b in ("control", "curriculum")},
                        final_budget_updates=a.updates,
                        train_decisions_per_branch=a.envs * a.rollout * a.updates,
                        hypothesis="A curriculum for carrier and off-ball players transfers useful cooperation to 4v4.",
                        adaptive_validation_split="dev", final_evaluation_split="test",
                        automatic_export=False, automatic_deployment=False)
        atomic_json(self.root / "manifest.json", manifest)
        self.execute("pilot", {b: branch_command(a, b, a.pilot_updates)
                               for b in ("control", "curriculum")})
        gate = pilot_gate(self.root, a.pilot_updates)
        atomic_json(self.root / "pilot_gate.json", gate)
        self.state["pilot_gate"] = gate
        self.save()
        if not gate["continue_training"]:
            self.state["status"] = "stopped_at_pilot"
            self.save()
            return 2
        candidate, control = [str(self.root / b / "last.pt") for b in ("curriculum", "control")]
        # This diagnostic uses dev only. It is deliberately too small to certify
        # improvement; preserve its result even if the final budget later fails.
        self.execute("pilot_evaluation_dev", {"pilot_evaluation_dev": evaluation_command(
            a, "dev", self.root / "pilot_evaluation_dev.json")})
        if a.updates > a.pilot_updates:
            self.execute("training", {b: branch_command(a, b, a.updates)
                                     for b in ("control", "curriculum")})
        out = self.root / "evaluation_test.json"
        self.execute("evaluation", {"evaluation": evaluation_command(a, "test", out)})
        report = json.loads(out.read_text(encoding="utf-8"))
        verdict = final_verdict(report, candidate, control, a.init)
        atomic_json(self.root / "verdict.json", verdict)
        self.state.update(status="finished", verdict=verdict,
                          finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        self.save()
        return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="runs/x4_coop_20261007")
    ap.add_argument("--bc", default="runs/x4_bc/final_sangu_rsone/best.pt")
    ap.add_argument("--init", default="runs/x4_estilo/rl_control/best.pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--envs", type=int, default=512)
    ap.add_argument("--rollout", type=int, default=128)
    ap.add_argument("--pilot-updates", type=int, default=100)
    ap.add_argument("--updates", type=int, default=1500)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--recordings", type=int, default=80)
    ap.add_argument("--eval-recordings", type=int, default=40)
    ap.add_argument("--matches", type=int, default=32)
    ap.add_argument("--probe-states", type=int, default=128)
    ap.add_argument("--threads", type=int, default=5)
    ap.add_argument("--wall-hours", type=float, default=4)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    if a.pilot_updates <= 20 or a.updates < a.pilot_updates or a.eval_every < 1:
        ap.error("invalid pilot/final budget")
    if a.pilot_updates % a.eval_every or a.updates % a.eval_every:
        ap.error("pilot and final budgets must finish at an evaluation boundary")
    if min(a.envs, a.rollout, a.recordings, a.eval_recordings, a.matches, a.threads) < 1 or a.wall_hours <= 0:
        ap.error("sizes and wall-clock budget must be positive")
    if Path(a.root).exists() and any(Path(a.root).iterdir()):
        ap.error("run directory must be empty; use a new --root (no implicit checkpoints/logs)")
    run = Run(a)
    try:
        return run.main()
    except Exception as e:
        run.state.update(status="failed", error=f"{type(e).__name__}: {e}")
        run.save()
        raise


if __name__ == "__main__":
    raise SystemExit(main())
