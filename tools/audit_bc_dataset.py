"""Cobertura de recs y shards existentes, sin entrenar ni modificar checkpoints.

python -m tools.audit_bc_dataset --out data/bc/coverage_report.json
Los tamaños de equipo son muestras, no partidas independientes. Las carpetas pueden
contener otros mapas; no confundir número de archivos con cobertura de una modalidad.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .build_bc_dataset import OUT, ROOT, SKIP, SRC, catalog_by_name, node


def summarize_folder(folder, recordings, val_frac=.12):
    report = {"recordings": len(recordings), "minutes": 0.0, "maps": {},
              "datasets": 0, "samples": 0, "train_replays": 0, "val_replays": 0,
              "train_samples": 0, "val_samples": 0, "team_sizes": {}, "pending": [], "excluded": []}
    cat = catalog_by_name()
    maps, sizes = Counter(), Counter()
    for r in recordings:
        if "error" in r:
            report["excluded"].append({"file": r["name"], "reason": "invalid replay", "error": r["error"]})
            continue
        report["minutes"] += r["minutes"]
        maps[r["stadium"]] += 1
        meta = cat.get(r["stadium"].strip())
        reason = None
        if any(s in r["stadium"].lower() for s in SKIP):
            reason = "training/penalties"
        elif any(str(n).upper().startswith("[BOT]") for n in r.get("names", [])):
            reason = "bot in recording"
        elif meta is None or meta["script"]:
            reason = "map outside supported BC catalog"
        if reason:
            report["excluded"].append({"file": r["name"], "map": r["stadium"], "reason": reason})
        elif not (folder / (Path(r["name"]).stem + ".npz")).exists():
            report["pending"].append(r["name"])
    for path in sorted(folder.glob("*.npz")):
        with np.load(path, allow_pickle=False) as z:
            n = len(z["act"])
            if not np.isin(z["act"], np.arange(18)).all() or len(z["T"]) != n:
                raise ValueError(f"Invalid labels or row count: {path}")
            count = 0
            for key in z.files:
                if key.startswith("obs_"):
                    obs = z[key]
                    if not np.isfinite(obs).all():
                        raise ValueError(f"Non-finite observations: {path}")
                    count += len(obs)
            if count != n or "act_lag6" not in z.files or len(z["act_lag6"]) != n:
                raise ValueError(f"Missing observations/lag labels: {path}")
            t, freq = np.unique(z["T"], return_counts=True)
            sizes.update({int(a): int(b) for a, b in zip(t, freq)})
            val = int(hashlib.md5(path.name.encode()).hexdigest(), 16) % 1000 < val_frac * 1000
            split = "val" if val else "train"
            report[f"{split}_replays"] += 1
            report[f"{split}_samples"] += n
            report["samples"] += n
            report["datasets"] += 1
    report["minutes"] = round(report["minutes"], 2)
    report["maps"], report["team_sizes"] = dict(maps), dict(sorted(sizes.items()))
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folders", help="comma-separated folder names")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    names = args.folders.split(",") if args.folders else sorted(p.name for p in SRC.iterdir() if p.is_dir())
    report = {"validation_fraction": .12, "root": str(ROOT), "folders": {}}
    for name in names:
        if name in (".", "..") or Path(name).name != name or not (SRC / name).is_dir():
            ap.error(f"Invalid recordings folder: {name}")
        recordings = json.loads(node("bridge/replays_by_map.js", SRC / name))
        r = summarize_folder(OUT / name, recordings)
        report["folders"][name] = r
        print(f"{name:10s} {r['recordings']:3d} recs | {r['minutes']:7.1f} min | "
              f"{r['datasets']:3d} datasets | {r['samples']:9,d} samples | "
              f"val {r['val_replays']} recs | pending {len(r['pending'])} | excluded {len(r['excluded'])}", flush=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Report: {args.out.resolve()}")


if __name__ == "__main__":
    main()
