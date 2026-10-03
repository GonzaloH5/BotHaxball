"""Evaluación greedy periódica de un run RS4 mientras entrena.

Cada ``--every`` pasos útiles copia ``latest.pt`` y juega partidos greedy (como
``deploy/bot.js --temp 0``) contra cada rival. Escribe una línea legible y otra
JSON por cruce en ``pod_logs/<run>_eval.log`` / ``.jsonl``. No modifica el run.

  python -m tools.rs4_eval_watch --run rs4_v5c --every 50000000 \\
      --opponent scripted:r3 --opponent runs/rs4_v3_public/champion.pt --opponent self
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import torch

from tools.rs4_style_gap import simulated_style

ROOT = Path(__file__).resolve().parent.parent


def checkpoint_steps(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    state = ck.get("rs4_program_state") or {}
    return int(state.get("relative_steps", ck.get("steps", 0)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--every", type=int, default=50_000_000)
    ap.add_argument("--opponent", action="append", required=True)
    ap.add_argument("--games", type=int, default=16)
    ap.add_argument("--poll", type=int, default=60, help="segundos entre revisiones de latest.pt")
    ap.add_argument("--stop-after", type=int, default=0, help="terminar al evaluar estos pasos (0 = nunca)")
    a = ap.parse_args()
    run_dir = ROOT / "runs" / a.run
    logs = ROOT / "pod_logs"
    logs.mkdir(exist_ok=True)
    text, jsonl = logs / f"{a.run}_eval.log", logs / f"{a.run}_eval.jsonl"
    done = {json.loads(line)["steps"] for line in jsonl.read_text().splitlines()} if jsonl.exists() else set()
    while True:
        latest = run_dir / "latest.pt"
        if latest.exists():
            steps = checkpoint_steps(latest)
            mark = steps // a.every * a.every
            if mark > 0 and mark not in done:
                snapshot = run_dir / f"eval_{mark // 1_000_000}M.pt"
                shutil.copy2(latest, snapshot)
                rows = []
                for opponent in a.opponent:
                    rival = str(snapshot) if opponent == "self" else opponent  # "self": espejo
                    result = simulated_style(str(snapshot), rival, games=a.games)
                    rows.append(dict(opponent=opponent, points=result["red_points"],
                                     goals_per_game=result["goals_per_game"], scoreless=result["scoreless"],
                                     passes_per_min=result["red"]["passes_per_min"],
                                     pass_turnover_ratio=result["red"]["pass_turnover_ratio"]))
                with jsonl.open("a") as f:
                    f.write(json.dumps(dict(steps=mark, checkpoint_steps=steps, matches=rows)) + "\n")
                with text.open("a", encoding="utf-8") as f:
                    f.write(f"{time.strftime('%H:%M')} | {a.run} {mark / 1e6:.0f}M | " + " | ".join(
                        f"vs {Path(r['opponent']).stem or r['opponent']}: puntos {r['points']:.2f}, "
                        f"goles/p {r['goals_per_game']:.2f}, 0-0 {r['scoreless']:.0%}, pases/min {r['passes_per_min']:.1f}"
                        for r in rows) + "\n")
                done.add(mark)
                if a.stop_after and mark >= a.stop_after:
                    return
        time.sleep(a.poll)


if __name__ == "__main__":
    main()
