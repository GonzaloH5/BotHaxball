"""E4 de la auditoría del 2026-10-08: ¿cuántos partidos hacen falta para afirmar algo?

1. **Goles:** desvío de la diferencia de gol por partido en el banco. Tamaño por brazo para no-inferioridad
   n ≈ 2 (z₁₋α + z₁₋β)² σ² / δ², con α = 0,05 a una cola y potencia 0,8.
2. **Métricas de proceso por jugador-partido** (lo que se puede medir en la sala con pocos partidos), por línea del
   puesto (D2):
   * acciones con pelota, pases y pérdidas cada 10 min, pase completo, patadas por minuto;
   * **desvío entre partidos de un mismo jugador** (clave anónima del nombre, jugadores con ≥ 6 partidos): es el ruido
     que tendría el bot jugando N partidos;
   * rango humano p10–p90 entre jugadores-partido;
   * partidos que necesita el bot para que el IC 90% de su media mida a lo sumo 1/4 del rango humano.

Escribe `reports/hx11/e4_potencia.{md,json}` (sin nombres).

  python -m tools.hx11_potencia
"""
from __future__ import annotations

import collections
import json
import math
from pathlib import Path

import numpy as np

from tools.hx11_d0 import apt_names, load_d0

ROOT = Path(__file__).resolve().parent.parent
D0 = ROOT / "data" / "hx11" / "d0"
D1 = ROOT / "data" / "hx11" / "d1"
D2 = ROOT / "data" / "hx11" / "d2"
REPORTS = ROOT / "reports" / "hx11"
Z = 1.645 + 0.842
MIN_MIN = 5.0


def main():
    from tools.hx11_d1 import names_by_recording
    from tools.x4_index import player_key
    names_of = names_by_recording()
    diffs = []
    per = collections.defaultdict(lambda: collections.defaultdict(list))      # línea → métrica → valores
    by_player = collections.defaultdict(lambda: collections.defaultdict(list))  # (línea, métrica, clave) → valores
    for n in apt_names():
        if not (D2 / f"{n}.json").exists():
            continue
        o, meta = load_d0(D0 / n)
        sc = o["score"][-1].astype(int)
        diffs.append(int(sc[0] - sc[1]))
        ev = json.loads((D1 / f"{n}.json").read_text(encoding="utf-8"))["events"]
        d2 = json.loads((D2 / f"{n}.json").read_text(encoding="utf-8"))
        spid = o["slot_pid"]
        cnt = collections.defaultdict(collections.Counter)
        for e in ev:
            if e.get("lugar") is None or e["lugar"] < 0:
                continue
            key = (e["equipo"], int(spid[e["t"], e["lugar"]]))
            k = e["tipo"]
            if k in ("pase", "centro", "remate", "despeje", "conduccion", "perdida"):
                cnt[key]["con_pelota"] += 1
            if k == "pase":
                cnt[key]["pases"] += 1
                cnt[key]["pases_ok"] += e.get("resultado") == "completo"
            if k == "perdida":
                cnt[key]["perdidas"] += 1
            if e.get("patada"):
                cnt[key]["patadas"] += 1
        nm = names_of.get(n, {})
        for t in d2["equipos"]:
            for p, info in t["jugadores"].items():
                mins = info["muestras"] / 600
                if mins < MIN_MIN:
                    continue
                c = cnt[(t["equipo"], int(p))]
                line = info["nombre_rol"].split("-")[0]
                vals = dict(con_pelota_10min=c["con_pelota"] / mins * 10, pases_10min=c["pases"] / mins * 10,
                            perdidas_10min=c["perdidas"] / mins * 10, patadas_min=c["patadas"] / mins)
                if c["pases"] >= 5:
                    vals["pase_completo"] = c["pases_ok"] / c["pases"]
                name = nm.get(str(p))
                for k, v in vals.items():
                    per[line][k].append(v)
                    if name:
                        by_player[(line, k)][player_key(name)].append(v)
    sigma = float(np.std(diffs, ddof=1))
    goals = {f"{d:.1f}": math.ceil(2 * Z ** 2 * sigma ** 2 / d ** 2) for d in (0.1, 0.2, 0.3, 0.5)}
    lines = {}
    for line, mets in per.items():
        if line.startswith("R"):
            continue
        row = {}
        for k, v in mets.items():
            v = np.array(v)
            within = [np.std(x, ddof=1) for x in by_player[(line, k)].values() if len(x) >= 6]
            sd_w = float(np.median(within)) if within else float("nan")
            rng = float(np.percentile(v, 90) - np.percentile(v, 10))
            # IC 90% de la media del bot en N partidos: 2 × 1,645 × sd / √N ≤ rango / 4
            need = math.ceil((2 * 1.645 * sd_w / (rng / 4)) ** 2) if within and rng > 0 else None
            row[k] = dict(n=len(v), p10=round(float(np.percentile(v, 10)), 3), p50=round(float(np.median(v)), 3),
                          p90=round(float(np.percentile(v, 90)), 3), sd_mismo_jugador=round(sd_w, 3),
                          jugadores_repetidos=len(within), partidos_bot=need)
        lines[line] = row
    s = dict(partidos=len(diffs), sigma_diferencia_gol=round(sigma, 3), partidos_por_brazo_goles=goals, lineas=lines)
    (REPORTS / "e4_potencia.json").write_text(json.dumps(s, indent=1, ensure_ascii=False), encoding="utf-8")
    md = ["# E4: potencia estadística", "",
          f"Auditoría del 2026-10-08 (§9.2, E4), con [`tools/hx11_potencia.py`](../../tools/hx11_potencia.py) sobre "
          f"{len(diffs)} partidos aptos.", "",
          f"## Goles", "", f"- **Desvío de la diferencia de gol por partido:** {sigma:.2f}.",
          "- **Partidos por brazo** para no-inferioridad con margen δ (α = 0,05 a una cola, potencia 0,8):", "",
          "| δ (goles por partido) | Partidos por brazo |", "|---|---|"]
    md += [f"| {d} | {v} |" for d, v in goals.items()]
    md += ["", "Por goles, \"no es carga\" solo se puede medir en el simulador, y con cientos o miles de partidos por "
               "celda. En la sala se mide con métricas de proceso.", "",
           "## Métricas de proceso por línea del puesto", "",
           "**Partidos del bot:** los que hacen falta para que el IC 90% de su media mida a lo sumo 1/4 del rango "
           "humano p10–p90. El ruido se toma de la variación de un mismo jugador entre partidos (mediana del desvío, "
           "jugadores con ≥ 6 partidos).", "",
           "| Línea | Métrica | Jugadores-partido | p10 / p50 / p90 | Desvío del mismo jugador | Partidos del bot |",
           "|---|---|---|---|---|---|"]
    for line in sorted(lines):
        for k, r in lines[line].items():
            md.append(f"| {line} | {k} | {r['n']} | {r['p10']} / {r['p50']} / {r['p90']} | {r['sd_mismo_jugador']} | "
                      f"{r['partidos_bot']} |")
    (REPORTS / "e4_potencia.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(dict(sigma=s["sigma_diferencia_gol"], goles=goals), ensure_ascii=False))


if __name__ == "__main__":
    main()
