"""Ancla en vivo (2026-10-11): dónde pararse según el equipo real, no según el promedio del puesto.

En la sala el bot recibe el ancla genérica del puesto pedido (`reports/hx11/pos_vocab.json`). En la prueba larga, con
esa ancla un defensor se aleja 1,7 veces más de su zona que con la del humano de ese lugar (1,07), y medios y
delanteros también empeoran (`reports/hx11/larga_h2_6_puesto.md`). La del humano no existe en la sala, pero sí sus
compañeros: con sus posiciones del último minuto se puede deducir la formación del equipo y qué rol queda libre.

* Para cada compañero de campo (sin el bot ni el arquero): posición media relativa al centroide de campo del equipo,
  con y sin la pelota (como `hx11_data.anchors`, en coordenadas del equipo).
* Para cada formación del vocabulario (plantillas de rol con su ancla, `pos_vocab.json` → `roles`): se asignan los
  compañeros a los roles con el algoritmo húngaro (costo: distancia al cuadrado con y sin la pelota). Los roles que
  sobran son los huecos; el puesto pedido elige entre ellos (misma línea; MCD y MED se aceptan entre sí con una
  penalización), y si ninguno es de esa línea, se penaliza la formación.
* Gana la formación de menor costo; el ancla del bot es la del rol hueco elegido, **solo si es de la línea pedida**.
  Si no (por ejemplo, MCD en un 4-4-2 o un 4-3-3, donde el hueco es un medio de costado), se usa el ancla genérica del
  puesto: con el hueco aproximado el MCD empeoraba en la prueba larga (zona 1,09 → 1,34), y el resto mejoraba (DEF
  1,76 → 1,21, MED 1,27 → 1,23, DEL 1,15 → 1,09; 2026-10-11).
"""
from __future__ import annotations

import numpy as np

from tools.hx11_d2 import hungarian

LINE_OF_POS = {"ARQ": "ARQ", "LI": "DEF", "LD": "DEF", "DFC": "DEF", "CRI": "DEF", "CRD": "DEF", "MCD": "MCD",
               "MC": "MED", "MI": "MED", "MD": "MED", "MCO": "MED", "DC": "DEL", "EI": "DEL", "ED": "DEL"}
LINE_PENALTY = 4e5          # el hueco no es de la línea pedida (px²)
NEAR_LINE_PENALTY = 5e4     # MCD ↔ MED


def role_templates(vocab):
    """{formación: (nombres de rol, anclas (R, 4))} de los roles de campo del vocabulario."""
    out = {}
    for key, info in vocab.get("roles", {}).items():
        f, role = key.split(":")
        if role == "ARQ":
            continue
        out.setdefault(f, ([], []))
        out[f][0].append(role)
        out[f][1].append(info["ancla"])
    return {f: (names, np.asarray(anc, np.float64)) for f, (names, anc) in out.items() if len(names) == 10}


def _line_cost(role, want):
    if want is None:
        return 0.0
    line = role.split("-")[0]
    if line == want:
        return 0.0
    if {line, want} == {"MCD", "MED"}:
        return NEAR_LINE_PENALTY
    return LINE_PENALTY


def live_anchor(rel_wb, rel_nb, templates, want_line, fallback):
    """Ancla deducida si el rol libre es de la línea pedida; si no, `fallback` (la genérica del puesto)."""
    if len(rel_wb) < 5:
        return np.asarray(fallback, np.float64), None, None
    if len(rel_wb) > 9:
        # sin arquero identificado (en la sala, si nadie puso !gk), el más retrasado es el arquero
        keep = np.argsort((rel_wb[:, 0] + rel_nb[:, 0]))[len(rel_wb) - 9:]
        rel_wb, rel_nb = rel_wb[keep], rel_nb[keep]
    res = infer_anchor(rel_wb, rel_nb, templates, want_line)
    if res is None:
        return np.asarray(fallback, np.float64), None, None
    anc, f, role = res
    if want_line is not None and role.split("-")[0] != want_line:
        return np.asarray(fallback, np.float64), f, None
    return anc, f, role


def infer_anchor(rel_wb, rel_nb, templates, want_line=None):
    """Ancla (4: con pelota x, y; sin pelota x, y) del rol libre para el bot, la formación y el rol.
    rel_wb, rel_nb: (n, 2) posiciones medias relativas de los n compañeros de campo (n ≤ 9)."""
    n = len(rel_wb)
    best = None
    for f, (names, A) in templates.items():
        R = len(names)
        if n >= R:
            continue
        cost = np.zeros((R, R))
        for i in range(n):
            cost[i] = ((A[:, :2] - rel_wb[i]) ** 2).sum(1) + ((A[:, 2:] - rel_nb[i]) ** 2).sum(1)
        lc = np.array([_line_cost(r, want_line) for r in names])
        cost[n:] = lc[None]                 # filas de los huecos: el del bot prefiere la línea pedida
        asg = hungarian(cost)
        holes = [int(asg[i]) for i in range(n, R)]
        hole = min(holes, key=lambda j: lc[j])
        total = float(sum(cost[i, asg[i]] for i in range(n)) + lc[hole])
        if best is None or total < best[0]:
            best = (total, A[hole].copy(), f, names[hole])
    return None if best is None else (best[1], best[2], best[3])


def teammates_means(pos, pid, gk, owner, ticks, team, skip):
    """Posiciones medias relativas al centroide de campo (con y sin pelota) de los compañeros del equipo `team` en los
    ticks dados (juego abierto), sin el lugar `skip`. pos (T, 22, 2) px; pid (T, 22); gk (T, 2); owner (T,)."""
    sl = np.arange(team * 11, team * 11 + 11)
    att = 1.0 if team == 0 else -1.0
    p = pid[ticks][:, sl]
    q = pos[ticks][:, sl] * att
    field = (p >= 0) & (p != gk[ticks, team][:, None])
    cen = (q * field[..., None]).sum(1) / np.maximum(field.sum(1), 1)[:, None]
    rel = q - cen[:, None]
    own = owner[ticks]
    wb = (own >= 0) & ((own // 11) == team)
    nb = (own >= 0) & ((own // 11) != team)
    wbs, nbs = [], []
    for i, s in enumerate(sl):
        if s == skip:
            continue
        m = field[:, i]
        if m.sum() < 0.5 * len(ticks):
            continue
        a = rel[m & wb, i]
        b = rel[m & nb, i]
        if len(a) < 5 or len(b) < 5:
            continue
        wbs.append(a.mean(0))
        nbs.append(b.mean(0))
    return np.array(wbs).reshape(-1, 2), np.array(nbs).reshape(-1, 2)
