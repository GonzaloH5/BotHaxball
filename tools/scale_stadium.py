"""Escala la GEOMETRÍA de un estadio .hbs sin tocar la física.

python -m tools.scale_stadium stadiums/x6.hbs stadiums/x6_half.hbs --scale 0.5

Se escalan posiciones (vértices, discos, arcos, planos, spawn, cancha, círculo central).
NO se escalan radios, masas, bCoef, damping, aceleraciones ni kickStrength: jugador y pelota
se comportan idéntico que en el mapa real, sólo que las distancias son más cortas.
Las curvas (ángulos) son invariantes a la escala uniforme.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def scale_stadium(d: dict, s: float, suffix: str) -> dict:
    d = json.loads(json.dumps(d))
    d["name"] = f"{d.get('name', 'stadium')} {suffix}"
    for k in ("width", "height", "spawnDistance"):
        if k in d:
            d[k] = d[k] * s
    bg = d.get("bg", {})
    for k in ("width", "height", "kickOffRadius", "cornerRadius", "goalLine"):
        if k in bg:
            bg[k] = bg[k] * s
    for v in d.get("vertexes", []):
        v["x"] *= s
        v["y"] *= s
    for seg in d.get("segments", []):
        for k in ("x", "y"):  # campos auxiliares del editor
            if k in seg:
                seg[k] *= s
    for disc in d.get("discs", []):
        if "pos" in disc:
            disc["pos"] = [disc["pos"][0] * s, disc["pos"][1] * s]
    bw0, bh0 = bg.get("width", 0) / s if bg else 0, bg.get("height", 0) / s if bg else 0
    for p in d.get("planes", []):
        nx, ny = p["normal"]
        line0 = bw0 if abs(nx) > abs(ny) else bh0
        margin = -p["dist"] - line0  # cuánto más allá de la línea está el plano (en el mapa original)
        if line0 and 0 < margin and ("ball" in (p.get("cMask") or []) or p.get("trait") == "ballArea"):
            # planos que frenan la pelota afuera: se conserva el margen absoluto, porque el radio
            # de la pelota no se escala y la regla de "salió entera" (línea + radio) necesita espacio
            p["dist"] = -(line0 * s + margin)
        else:
            p["dist"] *= s
    for g in d.get("goals", []):
        g["p0"] = [g["p0"][0] * s, g["p0"][1] * s]
        g["p1"] = [g["p1"][0] * s, g["p1"][1] * s]
    for key in ("redSpawnPoints", "blueSpawnPoints"):
        if key in d:
            d[key] = [[x * s, y * s] for x, y in d[key]]
    for j in d.get("joints", []):
        if j.get("length") is not None and not isinstance(j["length"], list):
            j["length"] *= s
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--scale", type=float, default=0.5)
    a = ap.parse_args()
    d = json.loads(Path(a.src).read_text(encoding="utf-8"))
    out = scale_stadium(d, a.scale, f"({int(a.scale * 100)}%)")
    Path(a.dst).write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"{a.dst}: {out['width']}x{out['height']}, spawn {out.get('spawnDistance')}, "
          f"arcos {[g['p0'] for g in out.get('goals', [])]}")


if __name__ == "__main__":
    main()
