"""Alternated, warmed RS4 v3 architecture benchmark on disposable run copies.

No training run/checkpoint is modified. Run this on the same Pod, with no other
GPU workload. This measures the engineering gate, not football improvement.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys

from .benchmark_multitask import acceptance


def compare_reports(control, memory, minimum_speed_ratio=0.7, max_memory_gib=None):
    if not control or len(control) != len(memory):
        raise ValueError("Each architecture requires the same positive number of repetitions")
    contracts = [report.get("comparison_contract") for report in control + memory]
    if any(contract != contracts[0] for contract in contracts[1:]):
        raise ValueError("Comparison requires identical environment, PPO and frozen-teammate mixture")
    summary = {}
    for name, reports in (("control", control), ("memory", memory)):
        summary[name] = dict(
            steps_per_second=statistics.median(report["steps_per_second"] for report in reports),
            cuda_peak_allocated_gib=max(report.get("cuda_peak_allocated_gib", 0.0) for report in reports),
            comparison_contract=contracts[0], repetitions=reports)
    summary["acceptance"] = acceptance(summary["memory"], summary["control"],
                                       minimum_speed_ratio, max_memory_gib)
    if max_memory_gib is not None:
        control_gate = acceptance(summary["control"], max_memory_gib=max_memory_gib)
        if not control_gate["accepted"]:
            summary["acceptance"]["accepted"] = False
            summary["acceptance"]["reasons"].extend("control: " + reason for reason in control_gate["reasons"])
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-config", required=True, type=Path)
    parser.add_argument("--control-checkpoint", required=True, type=Path)
    parser.add_argument("--memory-config", required=True, type=Path)
    parser.add_argument("--memory-checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--companion-fraction", type=float, default=0.1)
    parser.add_argument("--minimum-speed-ratio", type=float, default=0.7)
    parser.add_argument("--max-memory-gib", type=float, default=7.0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.repetitions < 1 or args.warmup < 1 or args.iters < 1:
        parser.error("repetitions/warmup/iters must be positive")
    for filename in (args.control_config, args.control_checkpoint, args.memory_config, args.memory_checkpoint):
        if not filename.is_file():
            parser.error(f"Missing comparison input: {filename}")
    if args.output.exists():
        parser.error("Output already exists; keep previous benchmark evidence")
    args.output.mkdir(parents=True)
    reports, order = {"control": [], "memory": []}, []
    for repetition in range(args.repetitions):
        # AB BA AB avoids consistently giving one architecture the colder host.
        for name in (("control", "memory") if repetition % 2 == 0 else ("memory", "control")):
            output = args.output / f"{repetition:02d}-{name}.json"
            command = [sys.executable, "-u", "-m", "tools.benchmark_multitask",
                       "--config", str(getattr(args, f"{name}_config")),
                       "--checkpoint", str(getattr(args, f"{name}_checkpoint")),
                       "--device", args.device, "--warmup", str(args.warmup), "--iters", str(args.iters),
                       "--companion-fraction", str(args.companion_fraction),
                       "--max-memory-gib", str(args.max_memory_gib), "--json-output", str(output)]
            print("Benchmark " + name + f" repetition {repetition + 1}/{args.repetitions}", flush=True)
            subprocess.run(command, check=True)
            reports[name].append(json.loads(output.read_text(encoding="utf-8")))
            order.append(name)
    summary = compare_reports(reports["control"], reports["memory"],
                              args.minimum_speed_ratio, args.max_memory_gib)
    summary["execution_order"] = order
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["acceptance"], indent=2))
    if not summary["acceptance"]["accepted"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
