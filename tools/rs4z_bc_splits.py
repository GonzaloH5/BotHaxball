"""Particiones del imitador BC RS4-Z: las de RS4-b1 más las grabaciones nuevas, sin tocar las de RS4-b1.

Las grabaciones ya particionadas conservan su partición (las compuertas y el banco de estados usan
`reports/rs4_b1/splits.json`). Una grabación nueva hereda la partición de su sesión si la sesión ya
existe (mismos jugadores, misma noche: evita fuga entre entrenamiento y prueba); si la sesión es nueva
va a entrenamiento. Sólo RS ONE puro (`only_rs_one`), con la misma regla que `tools/rs4_dataset_splits.py`.

  python -m tools.rs4z_bc_splits --out reports/rs4z/bc_splits.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.rs4_dataset_splits import session_key

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default=str(ROOT / "data" / "rs4_jsonl"))
    ap.add_argument("--base", default=str(ROOT / "reports" / "rs4_b1" / "splits.json"))
    ap.add_argument("--out", default=str(ROOT / "reports" / "rs4z" / "bc_splits.json"))
    a = ap.parse_args()
    base = json.loads(Path(a.base).read_text(encoding="utf-8"))
    index = json.loads((Path(a.cache) / "index.json").read_text(encoding="utf-8"))
    recordings = dict(base["recordings"])
    session_split = {row["session"]: row["split"] for row in recordings.values()}
    added = {}
    for name, row in index["recordings"].items():
        if name in recordings or row.get("status") != "ok":
            continue
        stadiums = [s.get("name") for s in row.get("stadiums") or []]
        if "Real Soccer ONE" not in stadiums:
            continue
        session = session_key(name, row.get("room"))
        split = session_split.get(session, "entrenamiento")
        recordings[name] = dict(sha256=row["sha256"], jsonl=row["jsonl"].replace("\\", "/"), room=row.get("room"),
                                session=session, bc="no_usada", only_rs_one=stadiums == ["Real Soccer ONE"],
                                split=split, contaminacion=None, added="2026-10-06")
        added[name] = split
    summary = {}
    for row in recordings.values():
        if row["only_rs_one"]:
            summary[row["split"]] = summary.get(row["split"], 0) + 1
    out = dict(version="RS4-Z-bc-splits-1", base=str(a.base), rules=__doc__.strip().splitlines()[2:6],
               added=added, summary_only_rs_one=summary, recordings=recordings)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(dict(added=len(added), added_only_rs_one=sum(recordings[n]["only_rs_one"] for n in added),
                          summary_only_rs_one=summary), ensure_ascii=False))


if __name__ == "__main__":
    main()
