"""Importa un estadio real (exportado por bridge/record.js o replay_to_jsonl.js) al catálogo stadiums/.

python -m tools.import_stadium <origen.hbs> <nombre> [--field W H] [--source "texto"]

--field fija las medidas de la cancha (semiancho, semialto) cuando bg no las describe bien
(mapas con script: el fondo dibujado es más grande que la cancha). Se guardan en la clave "haxballrl".
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sim.stadium import STADIUM_DIR, load_stadium


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("name")
    ap.add_argument("--field", nargs=2, type=float, default=None)
    ap.add_argument("--source", default=None)
    a = ap.parse_args()
    d = json.loads(Path(a.src).read_text(encoding="utf-8"))
    meta = d.setdefault("haxballrl", {})
    if a.field:
        meta["field_half_w"], meta["field_half_h"] = a.field
    meta["source"] = a.source or Path(a.src).name
    out = STADIUM_DIR / f"{a.name}.hbs"
    out.write_text(json.dumps(d), encoding="utf-8")
    st = load_stadium(a.name)
    print(f"{out.name}: {st.name} | cancha {st.field_half_w:g}x{st.field_half_h:g} | arco x={st.goal_x:g} "
          f"±{st.goal_half_height:g} | {len(st.d_pos)} discos, {len(st.s_p0)} segmentos | spawn {st.spawn_distance:g}")


if __name__ == "__main__":
    main()
