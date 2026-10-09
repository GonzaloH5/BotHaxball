"""Validación interna de D2 sin etiquetas humanas (auditoría del 2026-10-08, §2.4 y E1).

Lo que el usuario fija con `--pos` es un puesto por tramo, no un rol por tick. Antes de pedir etiquetas, se mide con
los propios datos si el puesto se puede identificar por tramo, y qué parte del movimiento por tick es desplazamiento
alrededor de un puesto estable y qué parte son permutas entre compañeros:

1. **Ancla por tramo:** posición media del jugador en coordenadas del equipo (relativa al centroide de los jugadores de
   campo, con el ataque hacia +x), por ventana de 3 min.
2. **Identificabilidad:** en la ventana w+1, el ancla de cada jugador ¿está más cerca de su propia ancla de la ventana w
   que de la de cualquier compañero? Si casi siempre, el ancla del tramo es una identidad estable del puesto aunque el
   rol por tick cambie.
3. **Deriva** del ancla entre ventanas seguidas contra la **separación** con el compañero más cercano (px).
4. **Rol dominante de D2:** acuerdo entre la primera y la segunda mitad del partido, y contra una asignación
   independiente (las anclas del partido de los jugadores de campo, asignadas por Hungarian a las medias de rol de D2).
5. **Desplazamiento por tick** respecto del ancla propia del partido (p50 y p90), y qué parte del tiempo fuera del rol
   dominante es una **permuta directa** (el compañero que ocupa mi rol dominante tiene como dominante el rol que ocupo
   yo) y qué parte es otra cosa.

No usa nombres. Escribe `reports/hx11/d2_validacion.{md,json}`.

  python -m tools.hx11_d2_valida [--jobs 3]
"""
from __future__ import annotations

import argparse
import collections
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from tools.hx11_d0 import apt_names, load_d0
from tools.hx11_d2 import ATT, GK_ROLE, N_ROLES, WINDOW, hungarian

ROOT = Path(__file__).resolve().parent.parent
D0 = ROOT / "data" / "hx11" / "d0"
D2 = ROOT / "data" / "hx11" / "d2"
REPORTS = ROOT / "reports" / "hx11"
MIN_WIN = 100        # muestras de un jugador en una ventana para tener ancla (100 × 6 ticks = 10 s)
MIN_PLAYER = 1000    # muestras de un jugador en el partido (100 s de juego abierto)


def match(name):
    o, _ = load_d0(D0 / name)
    z = np.load(D2 / f"{name}.npz")
    d2 = json.loads((D2 / f"{name}.json").read_text(encoding="utf-8"))
    ticks, role = z["ticks"], z["role"].astype(np.int64)
    pos = o["pos"][ticks].astype(np.float64) / 10
    pid = o["slot_pid"][ticks]
    out = dict(ident=[0, 0], drift=[], sep=[], halves=[0, 0], indep=[0, 0], disp=[], off=[0, 0, 0])
    for t in d2["equipos"]:
        team = t["equipo"]
        sl = np.arange(team * 11, team * 11 + 11)
        r = role[:, sl]
        field = (r >= 0) & (r < N_ROLES)
        if field.sum(1).max() < 8:
            continue
        q = pos[:, sl] * ATT[team]
        n = np.maximum(field.sum(1), 1)
        cen = (q * field[..., None]).sum(1) / n[:, None]
        rel = q - cen[:, None]
        p_ = np.where(field, pid[:, sl], -1)
        ids, cnt = np.unique(p_[p_ >= 0], return_counts=True)
        ids = ids[cnt >= MIN_PLAYER]
        if len(ids) < 8:
            continue
        win = ticks // WINDOW
        half = ticks >= np.median(ticks)
        anchors, dom_half, anchor_match = {}, {}, {}
        for p in ids:
            m = p_ == p
            f, c = np.nonzero(m)
            x = rel[f, c]
            anchor_match[p] = x.mean(0)
            out["disp"] += np.hypot(*(x - anchor_match[p]).T)[::10].round(1).tolist()
            rr = r[f, c]
            dom_half[p] = [np.bincount(rr[half[f] == h], minlength=N_ROLES).argmax() if (half[f] == h).sum() >= 300
                           else -1 for h in (False, True)]
            for w in np.unique(win[f]):
                sel = win[f] == w
                if sel.sum() >= MIN_WIN:
                    anchors[(p, int(w))] = x[sel].mean(0)
        # identificabilidad y deriva entre ventanas seguidas
        for w in np.unique(win):
            prev = {p: anchors[(p, int(w))] for p in ids if (p, int(w)) in anchors}
            nxt = {p: anchors[(p, int(w) + 1)] for p in ids if (p, int(w) + 1) in anchors}
            both = [p for p in prev if p in nxt]
            if len(both) < 8:
                continue
            P = np.array([prev[p] for p in both])
            for k, p in enumerate(both):
                d = np.hypot(*(P - nxt[p]).T)
                out["ident"][0] += int(np.argmin(d) == k)
                out["ident"][1] += 1
                out["drift"].append(round(float(d[k]), 1))
                out["sep"].append(round(float(np.sort(np.hypot(*(P - P[k]).T))[1]), 1))
        # rol dominante: mitades e independiente
        dom = {int(k): v["rol"] for k, v in t["jugadores"].items()}
        for p in ids:
            a, b = dom_half[p]
            if a >= 0 and b >= 0:
                out["halves"][0] += int(a == b)
                out["halves"][1] += 1
        mu = np.array(t["medias"])
        top = sorted(ids, key=lambda p: -(p_ == p).sum())[:N_ROLES]
        if len(top) == N_ROLES and np.isfinite(mu).all():
            A = np.array([anchor_match[p] for p in top])
            cost = ((A[:, None] - mu[None]) ** 2).sum(-1)
            asg = hungarian(cost)
            for k, p in enumerate(top):
                if int(p) in dom and dom[int(p)] < N_ROLES:
                    out["indep"][0] += int(asg[k] == dom[int(p)])
                    out["indep"][1] += 1
        # tiempo fuera del rol dominante: ¿permuta directa con el compañero que ocupa mi rol?
        holder = np.full((len(ticks), N_ROLES), -1, np.int64)
        f, c = np.nonzero(field)
        holder[f, r[f, c]] = p_[f, c]
        for p in ids:
            if int(p) not in dom or dom[int(p)] >= N_ROLES:
                continue
            dp = dom[int(p)]
            f, c = np.nonzero(p_ == p)
            rr = r[f, c]
            off = rr != dp
            out["off"][0] += int((~off).sum())
            if not off.any():
                continue
            other = holder[f[off], dp]
            swap = np.array([o_ >= 0 and dom.get(int(o_), -1) == int(x) for o_, x in zip(other, rr[off])])
            out["off"][1] += int(swap.sum())
            out["off"][2] += int((~swap).sum())
    out["name"] = name
    return out


def batch(jobs):
    names = [n for n in apt_names() if (D2 / f"{n}.npz").exists()]
    res = []
    with ProcessPoolExecutor(jobs) as ex:
        for k, r in enumerate(ex.map(match, names, chunksize=4), 1):
            res.append(r)
            if k % 100 == 0:
                print(f"{k}/{len(names)}", flush=True)
    tot = lambda key: np.sum([r[key] for r in res], axis=0)
    ident, halves, indep, off = tot("ident"), tot("halves"), tot("indep"), tot("off")
    drift = np.concatenate([r["drift"] for r in res])
    sep = np.concatenate([r["sep"] for r in res])
    disp = np.concatenate([r["disp"] for r in res])
    pct = lambda a: [round(float(np.percentile(a, q)), 1) for q in (10, 50, 90)]
    s = dict(partidos=len(res),
             identificabilidad=dict(aciertos=int(ident[0]), n=int(ident[1]), tasa=round(ident[0] / ident[1], 4)),
             deriva_px_p10_p50_p90=pct(drift), separacion_px_p10_p50_p90=pct(sep),
             deriva_sobre_separacion_p50=round(float(np.median(drift / np.maximum(sep, 1))), 3),
             rol_dominante_mitades=dict(n=int(halves[1]), acuerdo=round(halves[0] / halves[1], 4)),
             rol_dominante_vs_anclas=dict(n=int(indep[1]), acuerdo=round(indep[0] / indep[1], 4)),
             desplazamiento_tick_px_p10_p50_p90=pct(disp),
             tiempo_por_tick=dict(en_su_rol=round(off[0] / off.sum(), 4), permuta_directa=round(off[1] / off.sum(), 4),
                                  otro=round(off[2] / off.sum(), 4)))
    (REPORTS / "d2_validacion.json").write_text(json.dumps(s, indent=1, ensure_ascii=False), encoding="utf-8")
    (REPORTS / "d2_validacion.md").write_text(report(s), encoding="utf-8")
    print(json.dumps(s, ensure_ascii=False))
    return s


def report(s):
    i, h, a, t = s["identificabilidad"], s["rol_dominante_mitades"], s["rol_dominante_vs_anclas"], s["tiempo_por_tick"]
    return "\n".join([
        "# D2: validación interna sin etiquetas humanas", "",
        f"Responde a la auditoría del 2026-10-08 (§2.4, E1), con [`tools/hx11_d2_valida.py`](../../tools/hx11_d2_valida.py) "
        f"sobre {s['partidos']} partidos aptos. **No reemplaza la verdad humana:** mide si el puesto por tramo es una "
        "identidad estable en los datos y cuánto del rol por tick es ruido o movilidad.", "",
        "## El puesto por tramo", "",
        f"- **Identificabilidad del ancla de 3 min:** en la ventana siguiente, el ancla de cada jugador está más cerca de "
        f"la suya que de la de cualquier compañero en el **{100 * i['tasa']:.1f}%** de {i['n']} casos.",
        f"- **Deriva** del ancla entre ventanas, p10 / p50 / p90: {' / '.join(map(str, s['deriva_px_p10_p50_p90']))} px. "
        f"**Separación** con el compañero más cercano: {' / '.join(map(str, s['separacion_px_p10_p50_p90']))} px. "
        f"Deriva / separación, p50: {s['deriva_sobre_separacion_p50']}.",
        f"- **Desplazamiento por tick** respecto del ancla propia del partido: "
        f"{' / '.join(map(str, s['desplazamiento_tick_px_p10_p50_p90']))} px (p10 / p50 / p90).", "",
        "## El rol de plantilla de D2", "",
        f"- **Rol dominante, primera contra segunda mitad:** {100 * h['acuerdo']:.1f}% de {h['n']} jugadores-partido.",
        f"- **Rol dominante contra la asignación independiente por anclas:** {100 * a['acuerdo']:.1f}% de {a['n']}.",
        f"- **Tiempo por tick:** en su rol dominante {100 * t['en_su_rol']:.1f}%, en permuta directa con el compañero "
        f"que ocupa su rol {100 * t['permuta_directa']:.1f}%, otro {100 * t['otro']:.1f}%.", ""]) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=3)
    a = ap.parse_args()
    batch(a.jobs)


if __name__ == "__main__":
    main()
