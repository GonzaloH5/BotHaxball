"""Particiones entrenamiento / desarrollo / prueba de las grabaciones RS ONE 4v4 (PLAN_RS4.md 2.3).

* Deduplicación por contenido (SHA-256, ya registrada en data/rs4_jsonl/index.json).
* Sesión = fecha del nombre (AAAA-MM-DD) o, si no hay fecha, sala normalizada: todas las
  grabaciones de una sesión van a la misma partición (mismos jugadores, misma noche).
* Contaminación declarada: las grabaciones que entrenaron los imitadores BC v2/v5
  (`data/bc_rs4_v2`, partición por hash md5 del nombre con 12% de validación) no pueden ir
  a prueba. Prueba = sesiones sin ninguna grabación de entrenamiento BC.
* Desarrollo = sesiones elegidas por hash estable entre las restantes (~20% de grabaciones).

  python -m tools.rs4_dataset_splits --out reports/rs4_b1/splits.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def bc_role(stem, bc_dir, val_frac=0.12):
    """Rol de una grabación en el entrenamiento BC v2/v5 (train.bc: hash md5 del nombre del npz)."""
    if not (bc_dir / f"{stem}.npz").exists():
        return "no_usada"
    value = int(hashlib.md5(f"{stem}.npz".encode()).hexdigest(), 16) % 1000
    return "validacion_bc" if value < val_frac * 1000 else "entrenamiento_bc"


def session_key(name, room):
    match = re.search(r"(20\d\d)-(\d\d)-(\d\d)", name)
    if match:
        return "fecha:" + "-".join(match.groups())
    text = unicodedata.normalize("NFKD", room or name).encode("ascii", "ignore").decode().lower()
    text = re.sub(r"[^a-z0-9]+", " ", text).strip()
    text = re.sub(r"\b\d+\b$", "", text).strip()  # "MrHOST x IHL - Lima - 1/2" -> misma sesión
    return "sala:" + (text or name)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default=str(ROOT / "data" / "rs4_jsonl"))
    ap.add_argument("--bc-dir", default=str(ROOT / "data" / "bc_rs4_v2" / "rsx4"))
    ap.add_argument("--dev-fraction", type=float, default=0.2)
    ap.add_argument("--out", default=str(ROOT / "reports" / "rs4_b1" / "splits.json"))
    a = ap.parse_args()
    index = json.loads((Path(a.cache) / "index.json").read_text(encoding="utf-8"))
    bc_dir = Path(a.bc_dir)
    recordings = {}
    for name, row in index["recordings"].items():
        stadiums = [s.get("name") for s in row.get("stadiums") or []]
        if row.get("status") != "ok" or "Real Soccer ONE" not in stadiums:
            continue
        stem = Path(name).stem
        recordings[name] = dict(sha256=row["sha256"], jsonl=row["jsonl"].replace("\\", "/"), room=row.get("room"),
                                session=session_key(name, row.get("room")), bc=bc_role(stem, bc_dir),
                                only_rs_one=stadiums == ["Real Soccer ONE"])
    sessions = defaultdict(list)
    for name, row in recordings.items():
        sessions[row["session"]].append(name)
    split = {}
    clean = [s for s, names in sessions.items() if all(recordings[n]["bc"] != "entrenamiento_bc" for n in names)]
    for session in clean:
        split[session] = "prueba"
    rest = sorted((s for s in sessions if s not in split), key=lambda s: hashlib.sha256(s.encode()).hexdigest())
    target = a.dev_fraction * len(recordings)
    dev_count = 0
    for session in rest:
        if dev_count < target:
            split[session] = "desarrollo"
            dev_count += len(sessions[session])
        else:
            split[session] = "entrenamiento"
    for name, row in recordings.items():
        row["split"] = split[row["session"]]
        row["contaminacion"] = ("usada para entrenar imitadores BC" if row["bc"] == "entrenamiento_bc"
                                else "usada para validar/elegir BC" if row["bc"] == "validacion_bc" else None)
    summary = defaultdict(lambda: defaultdict(int))
    for row in recordings.values():
        summary[row["split"]]["grabaciones"] += 1
        summary[row["split"]][row["bc"]] += 1
    summary_sessions = defaultdict(int)
    for session, value in split.items():
        summary_sessions[value] += 1
    result = dict(version="RS4-b1-splits-1", rules=__doc__.strip().splitlines()[2:10],
                  summary={k: dict(v, sesiones=summary_sessions[k]) for k, v in summary.items()},
                  sessions={s: dict(split=split[s], recordings=sorted(n)) for s, n in sessions.items()},
                  recordings=recordings)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result["summary"], indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
