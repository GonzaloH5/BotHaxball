"""Pre-registro de compuertas RS4-Z: referencia de RS-Pro L5 (y L3, informativa) en cada batería.

Corre ANTES de entrenar y congela `reports/rs4z/gates.json` (reglas + referencias + huella de código).
Si cambia RS-Pro, el entorno o las baterías, hay que recalibrar y eso es un experimento nuevo.

  python -m tools.rs4z_calibrate_gates --episodes 512
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from eval.rs4z.batteries import BATTERIES, RSProCandidate, run_battery
from eval.rs4z.gates import RULES, STAGE_GATES

ROOT = Path(__file__).resolve().parent.parent
CODE = ["env/rs4z/kernel.py", "env/rs4z/core.py", "env/rs4z/drills.py", "env/rs4z/contract.py",
        "bots/rspro/brain.py", "bots/rspro/policy.py", "bots/rspro/geom.py", "eval/rs4z/batteries.py",
        "eval/rs4z/gates.py", "eval/rs4z/metrics.py", "eval/rs4z/runner.py"]


def code_hash():
    h = hashlib.sha256()
    for f in CODE:
        h.update((ROOT / f).read_bytes())
    return h.hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="reports/rs4z/gates.json")
    args = ap.parse_args()
    t0 = time.time()
    reference, info = {}, {}
    for name in BATTERIES:
        reference[name] = run_battery(RSProCandidate(5, seed=args.seed), name, episodes=args.episodes, seed=args.seed)
        info[name] = run_battery(RSProCandidate(3, seed=args.seed), name, episodes=args.episodes // 2, seed=args.seed)
        print(name, "L5", {t: round(v["success"], 3) for t, v in reference[name].items()},
              "L3", {t: round(v["success"], 3) for t, v in info[name].items()}, flush=True)
    report = dict(version="RS4-Z-gates-1", code_hash=code_hash(), episodes=args.episodes, seed=args.seed,
                  rules=RULES, stage_gates=STAGE_GATES, reference=reference, reference_l3=info,
                  seconds=time.time() - t0)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
