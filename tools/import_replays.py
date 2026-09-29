"""Prepara los replays de liga (replays_real/stadiums/<carpeta>/*.hbr2) para el entrenamiento.

python -m tools.import_replays [--minutes 3] [--folders rsx4,rsx6,...]

Para cada carpeta:
  1. clasifica los replays por mapa (bridge/replays_by_map.js) y descarta prácticas ("Training") y mapas
     que no son el principal de la carpeta;
  2. convierte a JSONL (bridge/replay_to_jsonl.js) el primer replay de cada mapa principal y cuenta cuánto
     interviene el script de la sala (eventos disc_props por minuto);
  3. deja el estadio exportado junto al JSONL (runs/real/leagues/) para importarlo al catálogo y validarlo
     con bridge/compare_sim.py --gate.
Imprime un resumen por mapa. No modifica stadiums/ ni train/tasks.yaml.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "replays_real" / "stadiums"
OUT = ROOT / "runs" / "real" / "leagues"
SKIP = ("training", "penales")


def node(*args) -> str:
    r = subprocess.run(["node", *map(str, args)], cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-2000:])
    return r.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=3.0)
    ap.add_argument("--folders", default=None)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    folders = a.folders.split(",") if a.folders else sorted(p.name for p in SRC.iterdir() if p.is_dir())
    summary = []
    for folder in folders:
        reps = json.loads(node("bridge/replays_by_map.js", SRC / folder))
        by_map = defaultdict(list)
        for r in reps:
            if "error" in r or any(s in r["stadium"].lower() for s in SKIP):
                continue
            by_map[r["stadium"].strip()].append(r)
        for stadium, rs in sorted(by_map.items(), key=lambda kv: -sum(r["minutes"] for r in kv[1])):
            minutes = sum(r["minutes"] for r in rs)
            pm = sum(r["minutes"] * r["players"] for r in rs)
            first = max(rs, key=lambda r: r["minutes"])
            node("bridge/replay_to_jsonl.js", first["file"], "--out", OUT, "--max-minutes", a.minutes)
            base = OUT / ("replay_" + Path(first["name"]).stem.replace(" ", "_"))
            jsonl = next(OUT.glob(Path(first["name"]).stem.replace(" ", "_").replace("(", "?").replace(")", "?")
                                  .join(["replay_", ".jsonl"])), None) or base.with_suffix(".jsonl")
            n_script = n_ticks = 0
            if jsonl.exists():
                for line in jsonl.open(encoding="utf-8"):
                    if '"disc_props"' in line:
                        n_script += 1
                    elif '"type":"tick"' in line:
                        n_ticks += 1
            per_min = n_script / max(n_ticks / 3600, 1e-9)
            summary.append({"folder": folder, "stadium": stadium, "replays": len(rs), "minutes": round(minutes),
                            "player_minutes": round(pm), "script_per_min": round(per_min, 1),
                            "sample": str(jsonl.relative_to(ROOT)) if jsonl.exists() else None,
                            "files": [r["name"] for r in rs]})
            print(f"{folder:9s} {stadium[:36]:36s} {len(rs):3d} replays {minutes:5.0f} min {pm:6.0f} jug·min | "
                  f"script {per_min:6.1f} eventos/min | {jsonl.name if jsonl.exists() else '(sin muestra)'}", flush=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nresumen en {OUT / 'summary.json'}")


if __name__ == "__main__":
    main()
