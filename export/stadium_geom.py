"""Geometría de un estadio para el bot de Node (obs "universal"), con el MISMO código que el entrenamiento.

python -m export.stadium_geom <archivo.hbs | nombre del catálogo> [--out geom.json]
(sin --out imprime el JSON por stdout; deploy/bot.js lo llama al entrar a una sala o cambiar el mapa)

Salida: medidas de la cancha y del arco, radios, y los obstáculos (segmentos, círculos, planos) contra los
que chocan la pelota, un jugador rojo y uno azul, ya filtrados por cGroup/cMask y con los arcos discretizados.
"""
from __future__ import annotations

import argparse
import json
import sys

from env.geometry import StadiumRays
from sim.stadium import STADIUM_DIR, load_stadium


def _catalog_overrides() -> dict[str, dict]:
    """Medidas de cancha fijadas a mano ("haxballrl" en stadiums/*.hbs), por nombre de mapa.
    Las salas mandan el mapa sin esa clave: así un mapa conocido recupera sus medidas reales."""
    out = {}
    for f in STADIUM_DIR.glob("*.hbs"):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        ov = d.get("haxballrl", {})
        if "field_half_w" in ov:
            out[d.get("name", "").strip()] = ov
    return out


def geometry(name_or_path: str) -> dict:
    st = load_stadium(name_or_path)
    ov = _catalog_overrides().get(st.name.strip())
    if ov:
        st.field_half_w, st.field_half_h = float(ov["field_half_w"]), float(ov["field_half_h"])
    rays = StadiumRays(st)

    def pack(o):
        a, b, cc, cr, pn, pd = o
        return {"seg_a": a.tolist(), "seg_b": b.tolist(), "circ_c": cc.tolist(), "circ_r": cr.tolist(),
                "pl_n": pn.tolist(), "pl_d": pd.tolist()}

    return {
        "name": st.name, "field_half_w": st.field_half_w, "field_half_h": st.field_half_h,
        "goal_x": st.goal_x, "goal_half_height": st.goal_half_height, "kickoff_radius": st.kickoff_radius,
        "ball_radius": float(st.ball["radius"]), "player_radius": float(st.player["radius"]),
        "ball_invmass": float(st.ball["invMass"]),  # base del powershot (la pelota "cargada" pesa distinto)
        "obstacles": {"ball": pack(rays.ball), "red": pack(rays.red), "blue": pack(rays.blue)},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stadium")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    js = json.dumps(geometry(a.stadium))
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(js)
    else:
        sys.stdout.write(js)


if __name__ == "__main__":
    main()
