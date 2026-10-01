"""Evaluate stopped RS4 v3 checkpoints independently, without training helpers."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from eval.rs4_v3 import evaluate
from tools.audit_rs4_simulator import digest
import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--reference")
    ap.add_argument("--opponents", nargs="*", default=[])
    ap.add_argument("--teammate-reference")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[51, 73, 91])
    ap.add_argument("--games", type=int, default=16)
    ap.add_argument("--functional-games", type=int, default=16)
    ap.add_argument("--minutes", type=float, default=2)
    ap.add_argument("--reference-report", help="Reuse an immutable evaluated baseline")
    ap.add_argument("--torch-threads", type=int, default=2)
    ap.add_argument("--physics-threads", type=int, default=4)
    args = ap.parse_args()
    from train.runtime import cpu_budget
    if args.torch_threads < 1 or args.physics_threads < 1:
        ap.error("Thread counts must be positive")
    torch_threads = min(args.torch_threads, cpu_budget())
    torch.set_num_threads(torch_threads)
    from numba import get_num_threads, set_num_threads
    physics_threads = min(args.physics_threads, cpu_budget(), get_num_threads())
    set_num_threads(physics_threads)
    print(f"Independent evaluation runtime: cpu | Torch {torch_threads} | physics {physics_threads}", flush=True)
    for path in [args.checkpoint, args.reference, args.teammate_reference, *args.opponents]:
        if path and not Path(path).is_file():
            ap.error(f"Checkpoint missing: {path}")
    if args.games < 1 or args.functional_games < 1 or args.minutes <= 0:
        ap.error("Games and minutes must be positive")
    common = dict(seeds=args.seeds, games=args.games, functional_games=args.functional_games,
                  minutes=args.minutes, teammate_reference=args.teammate_reference)
    refs = [args.reference] if args.reference else []
    report = evaluate(args.checkpoint, references=[*refs, *args.opponents], **common)
    cache_exists = args.reference_report and Path(args.reference_report).is_file()
    reference_identity = digest(args.reference) if args.reference else None
    if cache_exists:
        reference = json.loads(Path(args.reference_report).read_text(encoding="utf-8"))
        if (reference.get("seeds") != args.seeds or reference.get("evaluation_contract") != report["evaluation_contract"]
                or reference.get("suite") != report["suite"] or reference.get("checkpoint_sha256") != reference_identity):
            ap.error("Reference report uses a different contract, suite, seeds or reference checkpoint")
        if reference.get("source_fingerprint") != report["source_fingerprint"]:
            ap.error("Reference report was measured with different simulator/scorer source code; use a new reference-report path")
    elif args.reference:
        # Compare against exactly the same opponent matrix (including the frozen
        # reference itself), colors, seeds and teammate mix.
        reference = evaluate(args.reference, references=[*refs, *args.opponents], **common)
        reference["checkpoint_sha256"] = reference_identity
        if args.reference_report:
            cache = Path(args.reference_report)
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(reference, indent=2), encoding="utf-8")
    else:
        reference = None
    report["checkpoint_sha256"] = digest(args.checkpoint)
    report["runtime"] = {"device": "cpu", "torch_threads": torch_threads, "physics_threads": physics_threads}
    if reference:
        report["reference"] = reference
        report["baseline"] = reference["functional"]["skills"]
        report["matches"] = {"points": report["full_games"]["mean_points"],
                             "baseline_points": reference["full_games"]["mean_points"]}
    else:
        report["matches"] = {"points": report["full_games"]["mean_points"]}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"RS4 v3: points {report['full_games']['mean_points']:.3f}; functional {report['functional']['mean_success']:.3f}")
    print(f"Results: {out.resolve()}")


if __name__ == "__main__":
    main()
