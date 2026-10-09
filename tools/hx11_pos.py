"""Vocabulario de `--pos` → condición de la política (P2 de la revisión de la auditoría).

En el entrenamiento, la imitación se condiciona con el ancla medida de cada jugador (posición media relativa al
centroide de campo, con y sin pelota) y la línea de su rol dominante (`learn/hx11_data.py`). En la sala el usuario
pide un puesto por nombre; acá se traduce a esa condición con los humanos del banco:

* cada rol de D2 (línea-lado en su formación) se asigna a un puesto del plan (§4): LI, LD, DFC, CRI, CRD, MCD, MC,
  MCO, MI, MD, EI, ED, DC (las dos puntas de un 4-4-2 o 3-5-2 son DC);
* el ancla del puesto es la mediana de las anclas de los jugadores-mitad con ese puesto, y la línea, la más común.

Es la parte del etiquetador que queda en el camino del bot. La valida el usuario (E1 liviano); hasta entonces, cada
puesto informa cuántos jugadores-mitad lo sostienen y su dispersión.

  python -m tools.hx11_pos     # → reports/hx11/pos_vocab.{md,json}
"""
from __future__ import annotations

import collections
import json
from pathlib import Path

import numpy as np

from learn.hx11_data import LINES, anchors
from tools.hx11_d0 import apt_names, load_d0

ROOT = Path(__file__).resolve().parent.parent
D1 = ROOT / "data" / "hx11" / "d1"
D2 = ROOT / "data" / "hx11" / "d2"
REPORTS = ROOT / "reports" / "hx11"
BACK3 = {"3-4-3", "3-5-2"}
BACK5 = {"5-3-2", "5-4-1"}


def position(role, formation):
    """Puesto del plan (§4) de un rol de D2 en su formación, o None."""
    if role == "ARQ":
        return "ARQ"
    line, side = role.split("-")
    if line == "DEF":
        if formation in BACK5:
            return {"I": "CRI", "D": "CRD"}.get(side, "DFC")
        if formation in BACK3:
            return "DFC"
        return {"I": "LI", "D": "LD"}.get(side, "DFC")
    if line == "MCD":
        return "MCD"
    if line == "MED":
        if formation in BACK3:
            return {"I": "CRI", "D": "CRD"}.get(side, "MC")
        if formation == "4-2-3-1":
            return {"I": "MI", "D": "MD"}.get(side, "MCO")
        if formation in ("4-3-3",):
            return "MC"
        return {"I": "MI", "D": "MD"}.get(side, "MC")
    if line == "DEL":
        return {"I": "EI", "D": "ED"}.get(side, "DC")
    return None


def main():
    vals = collections.defaultdict(list)
    lines = collections.defaultdict(collections.Counter)
    for n in apt_names():
        if not (D2 / f"{n}.json").exists():
            continue
        d2 = json.loads((D2 / f"{n}.json").read_text(encoding="utf-8"))
        o, _ = load_d0(ROOT / "data" / "hx11" / "d0" / n)
        owner = np.load(D1 / f"{n}.npz")["owner"]
        gk = np.where(o["gk_pid"] >= 0, o["gk_pid"], o["gk_inferred"]).astype(np.int32)
        anc = anchors(o, owner, gk)
        for t in d2["equipos"]:
            f = t["formacion"]["sin_pelota"]["formacion"]
            for p, info in t["jugadores"].items():
                if info["muestras"] < 1000 or info["nombre_rol"].startswith("R"):
                    continue
                pos = position(info["nombre_rol"], f)
                if pos is None:
                    continue
                for h in (0, 1):
                    a = anc.get((int(p), h))
                    if a is not None and np.any(a):
                        vals[pos].append(a)
                        lines[pos][info["nombre_rol"].split("-")[0]] += 1
    vocab = {}
    for pos, v in sorted(vals.items()):
        v = np.array(v, np.float64)
        ln = lines[pos].most_common(1)[0][0]
        med = np.median(v, 0)
        dev = v - med
        disp = [float(np.median(np.hypot(dev[:, 2 * k], dev[:, 2 * k + 1]))) for k in (0, 1)]
        vocab[pos] = dict(linea=ln, linea_idx=LINES.index(ln), ancla=[round(float(x), 1) for x in med],
                          dispersion_px=[round(x, 1) for x in disp], jugadores_mitad=len(v))
    (REPORTS / "pos_vocab.json").write_text(json.dumps(vocab, indent=1, ensure_ascii=False), encoding="utf-8")
    md = ["# Vocabulario de `--pos`", "",
          "P2 de la revisión de la auditoría, con [`tools/hx11_pos.py`](../../tools/hx11_pos.py). El ancla es relativa "
          "al centroide de los jugadores de campo, con el ataque hacia +x e y > 0 a la derecha. La dispersión es la "
          "mediana de la distancia de cada jugador-mitad al ancla del puesto (con y sin pelota). **Falta que la valide el "
          "usuario (E1 liviano).**", "",
          "| Puesto | Línea | Ancla con pelota (x, y) | Ancla sin pelota (x, y) | Dispersión (px) | Jugadores-mitad |",
          "|---|---|---|---|---|---|"]
    for pos, r in vocab.items():
        a = r["ancla"]
        md.append(f"| {pos} | {r['linea']} | ({a[0]}, {a[1]}) | ({a[2]}, {a[3]}) | {r['dispersion_px'][0]} / "
                  f"{r['dispersion_px'][1]} | {r['jugadores_mitad']} |")
    (REPORTS / "pos_vocab.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({k: (v["linea"], v["ancla"], v["jugadores_mitad"]) for k, v in vocab.items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
