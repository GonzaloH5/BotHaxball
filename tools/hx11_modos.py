"""Modos y versiones del script por partido (auditoría del 2026-10-08, §2.3 y E2).

La sala prende y apaga faltas, offside y powershot por partido, y sólo lo anuncia en una parte de los partidos. La
imitación no puede mezclar sin saberlo partidos donde sostener el contacto es gratis con partidos donde es falta. Por
partido apto se junta:

* **Lo anunciado:** "Modos: PowerShot [✅] • Offside [✅] • Faltas [✅]" y los avisos "El Offside ha sido (des)activado".
* **Faltas:** oportunidades = tramos de contacto de un rival con el poseedor (centros a <= 30 + 2,5 px, cortes de hasta
  6 ticks, pelota a <= 25 px del poseedor, juego abierto) que llegan a 110 ticks; cobros = decisiones FOUL de la sala.
  Calibrado con 66 cobros reales: con esta definición, el contacto antes del cobro mide p10 112, p50 115 y p90 126
  ticks. Con el umbral en 115 (la regla de S2) quedaba afuera el 40% de los cobros y salían más cobros que
  oportunidades.
* **Offside:** cobros = decisiones OFFSIDE; oportunidades sin cobro = pases completos de juego abierto (D1) cuyo receptor
  estaba en posición de offside al patear (campo rival, delante de la pelota y del anteúltimo rival).
* **Powershot:** anuncios "Powershot utilizado" y patadas tras >= 90 ticks pegado a la pelota (la carga empieza ahí).
* **Versión:** marcadores del script en los anuncios (`!gk`, "Modos:", la barra "Potencia del …", el powershot por
  niveles) y el mes.

Decisión por partido (umbrales fijados antes de mirar el resultado): un modo está **apagado** si tiene >= 3
oportunidades y 0 cobros (o 0 anuncios, para el powershot), **prendido** si tiene algún cobro o anuncio, e
**indeterminado** si no hay oportunidades. Lo anunciado manda sobre lo inferido; si se contradicen, se informa.

Escribe `data/hx11/modos.json` (por partido, sin nombres) y `reports/hx11/e2_modos.{md,json}`.

  python -m tools.hx11_modos [--jobs 3]
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from env.hx11 import contract as HC
from tools.hx11_d0 import OPEN, apt_names, load_d0

ROOT = Path(__file__).resolve().parent.parent
D0 = ROOT / "data" / "hx11" / "d0"
D1 = ROOT / "data" / "hx11" / "d1"
OUT = ROOT / "data" / "hx11" / "modos.json"
REPORTS = ROOT / "reports" / "hx11"
MIN_OPP = 3
PS_CONTACT = 90
FOUL_OPP = 110
P_OFF = 0.85
ATT = (1.0, -1.0)
MODES_ANN = re.compile(r"(PowerShot|Offside|Faltas)\s*\[(✅|❌)\]")
TOGGLE_ANN = re.compile(r"El (Offside|PowerShot|Powershot|Faltas?) ha sido (activad|desactivad)")


def meta_by_file():
    out = {}
    for f in sorted(glob.glob(str(ROOT / "data" / "hx11" / "meta_*.jsonl"))):
        for line in open(f, encoding="utf-8", errors="replace"):
            if line.strip():
                r = json.loads(line)
                out.setdefault(r["file"].removesuffix(".hbr2"), r.get("announcements") or [])
    return out


def contact_runs(pos, owner, ball, on, live):
    """Tramos de contacto rival-poseedor según la regla de falta de S2: (largo en ticks, último tick) de cada tramo."""
    T = len(owner)
    run = {}
    last = {}
    best = []
    for t in np.nonzero(live & (owner >= 0))[0]:
        o = int(owner[t])
        po = pos[t, o]
        if np.hypot(*(ball[t] - po)) - HC.PLAYER_R - HC.BALL_R > HC.FOUL_BALL_DIST:
            continue
        rv = np.arange(11, 22) if o < 11 else np.arange(11)
        rv = rv[on[t, rv]]
        d = np.hypot(*(pos[t, rv] - po).T)
        for r in rv[d <= 2 * HC.PLAYER_R + HC.FOUL_CONTACT_TOL]:
            key = (o, int(r))
            if key in last and t - last[key] <= HC.FOUL_GAP + 1:
                run[key] += t - last[key]
            else:
                if key in run:
                    best.append((run[key], last[key]))
                run[key] = 1
            last[key] = t
    best += [(run[k], last[k]) for k in run]
    return np.array(best, np.int64).reshape(-1, 2)


def match(args):
    name, ann = args
    o, meta = load_d0(D0 / name)
    owner = np.load(D1 / f"{name}.npz")["owner"]
    ev = json.loads((D1 / f"{name}.json").read_text(encoding="utf-8"))["events"]
    pos = o["pos"].astype(np.float64) / 10
    ball = o["ball_pos"].astype(np.float64) / 10
    on = o["slot_pid"] >= 0
    live = o["phase"] == OPEN
    dec = collections.Counter(d[1] for d in meta["decisions"])
    # faltas
    runs = contact_runs(pos, owner, ball, on, live)
    foul_ticks = np.array([d[0] for d in meta["decisions"] if d[1] == "FOUL"], np.int64)
    long_ = runs[runs[:, 0] >= FOUL_OPP]
    foul_opp = len(long_)
    # oportunidad cobrada: hay un FOUL entre el fin del contacto y 20 ticks después
    foul_opp_called = int(sum(((foul_ticks >= e) & (foul_ticks <= e + 20)).any() for e in long_[:, 1]))
    runs = runs[:, 0]
    # offside: pases completos de juego abierto al receptor en posición de offside al patear
    off_unc = 0
    for e in ev:
        if e["tipo"] != "pase" or e.get("saque") or e.get("resultado") != "completo" or e.get("a", -1) < 0:
            continue
        t, a, team = e["t"], e["a"], e["equipo"]
        att = ATT[team]
        xa = pos[t, a, 0] * att
        rv = np.arange(11, 22) if team == 0 else np.arange(11)
        xs = np.sort(pos[t, rv[on[t, rv]], 0] * att)[::-1]
        second = xs[1] if len(xs) >= 2 else -np.inf
        if xa > 0 and xa > ball[t, 0] * att and xa > second:
            off_unc += 1
    # powershot: patadas tras >= 90 ticks de contacto del pateador (cerca de la pelota)
    gap = np.hypot(*(pos - ball[:, None]).transpose(2, 0, 1)) - HC.PLAYER_R - HC.BALL_R
    gap[~on] = np.inf
    close = gap <= 1.0
    ps_opp = 0
    for t, s in zip(*np.nonzero(o["kick"])):
        if t >= PS_CONTACT and live[t] and close[t - PS_CONTACT:t, s].all():
            ps_opp += 1
    ps_ann = sum(1 for n in meta["numbers"] if n[1] == "powershot")
    # anuncios de la sala
    announced = {}
    toggles = []
    markers = set()
    for a in ann:
        m = a.get("m", "")
        for k, v in MODES_ANN.findall(m):
            announced[k.lower()] = v == "✅"
        for k, v in TOGGLE_ANN.findall(m):
            toggles.append((a.get("f0"), k.lower(), v == "activad"))
        if "Modos:" in m:
            markers.add("anuncia_modos")
        if "es el Arquero" in m:
            markers.add("gk")
        if "Potencia del" in m:
            markers.add("barra")
        if "Powershot utilizado" in m:
            markers.add("powershot_pct")
        if "Nivel" in m and "Power" in m:
            markers.add("powershot_niveles")
        if "Gambeta" in m or "gambeta" in m:
            markers.add("gambeta")

    def decide(calls, opp, ann_key):
        inf = "prendido" if calls > 0 else ("apagado" if opp >= MIN_OPP else "indeterminado")
        a = announced.get(ann_key)
        final = ("prendido" if a else "apagado") if a is not None else inf
        return dict(cobros=calls, oportunidades=opp, inferido=inf, anunciado=a, modo=final,
                    contradice=a is not None and inf != "indeterminado" and (inf == "prendido") != a)

    return dict(name=name,
                faltas=dict(decide(dec.get("FOUL", 0), foul_opp, "faltas"), oportunidades_cobradas=foul_opp_called),
                offside=decide(dec.get("OFFSIDE", 0), off_unc + dec.get("OFFSIDE", 0), "offside"),
                powershot=decide(ps_ann, ps_opp, "powershot"),
                offside_on_frac=round(float(o["offside_on"][live].mean()) if live.any() else 1.0, 3),
                cambios=toggles, marcadores=sorted(markers),
                contacto_p90=int(np.percentile(runs, 90)) if len(runs) else 0,
                contacto_100_115=int(((runs >= 100) & (runs < HC.FOUL_TICKS)).sum()),
                minutos=round(float(live.sum()) / 3600, 2))


def batch(jobs):
    names = apt_names()
    anns = meta_by_file()
    splits = json.loads((REPORTS / "splits.json").read_text(encoding="utf-8"))["recordings"]
    res = []
    with ProcessPoolExecutor(jobs) as ex:
        for k, r in enumerate(ex.map(match, [(n, anns.get(n, [])) for n in names], chunksize=4), 1):
            sp = splits.get(r["name"] + ".hbr2", {})
            r["mes"] = (sp.get("times") or ["?"])[0][:7]
            r["split"] = sp.get("split")
            res.append(r)
            if k % 100 == 0:
                print(f"{k}/{len(names)}", flush=True)
    # faltas sin cobro: P(apagado) con la tasa de cobro medida en los partidos con cobros (previa 0,5)
    on = [r["faltas"] for r in res if r["faltas"]["cobros"] > 0]
    miss = 1 - sum(f["oportunidades_cobradas"] for f in on) / max(1, sum(f["oportunidades"] for f in on))
    for r in res:
        f = r["faltas"]
        p_off = 0.0 if f["cobros"] > 0 else 1 / (1 + max(miss, 0.01) ** f["oportunidades"])
        f["p_apagado"] = round(p_off, 3)
        f["modo_l0"] = ("prendido" if f["anunciado"] or (f["anunciado"] is None and f["cobros"] > 0) else
                        "apagado" if f["anunciado"] is False or p_off >= P_OFF else "desconocido")
    OUT.write_text(json.dumps({r["name"]: r for r in res}, ensure_ascii=False), encoding="utf-8")
    s = summarize(res)
    s["faltas_sin_cobro_si_prendido"] = round(miss, 3)
    (REPORTS / "e2_modos.json").write_text(json.dumps(s, indent=1, ensure_ascii=False), encoding="utf-8")
    (REPORTS / "e2_modos.md").write_text(report(s), encoding="utf-8")
    print(json.dumps({k: s[k] for k in ("modos", "contradicciones")}, ensure_ascii=False))


def summarize(res):
    modes = {k: collections.Counter(r[k]["modo"] for r in res) for k in ("faltas", "offside", "powershot")}
    modes["faltas_l0"] = collections.Counter(r["faltas"]["modo_l0"] for r in res)
    contra = {k: sum(r[k]["contradice"] for r in res) for k in ("faltas", "offside", "powershot")}
    contra["faltas_l0"] = 0
    by_month = collections.defaultdict(lambda: collections.Counter())
    for r in res:
        by_month[r["mes"]]["partidos"] += 1
        for k in ("faltas", "offside", "powershot"):
            by_month[r["mes"]][f"{k}_{r[k]['modo']}"] += 1
        by_month[r["mes"]]["gk"] += "gk" in r["marcadores"]
        by_month[r["mes"]]["anuncia_modos"] += "anuncia_modos" in r["marcadores"]
    on = [r for r in res if r["faltas"]["modo"] == "prendido"]
    rate = (sum(r["faltas"]["cobros"] for r in on) / max(1, sum(r["faltas"]["oportunidades"] for r in on)))
    versions = collections.Counter(" + ".join(r["marcadores"]) or "ninguno" for r in res)
    return dict(partidos=len(res), modos={k: dict(v) for k, v in modes.items()}, contradicciones=contra,
                por_mes={k: dict(v) for k, v in sorted(by_month.items())},
                faltas_cobradas_sobre_oportunidades_si_prendido=round(rate, 3),
                contacto_entre_100_y_115_ticks=sum(r["contacto_100_115"] for r in res),
                partidos_con_cambio_de_modo=sum(1 for r in res if r["cambios"]),
                versiones=dict(versions.most_common()),
                por_parte={sp: {k: dict(collections.Counter(r[k]["modo"] for r in res if r["split"] == sp))
                                for k in ("faltas", "offside", "powershot")}
                           for sp in sorted({r["split"] for r in res if r["split"]})})


def report(s):
    lines = ["# E2: modos y versiones del script por partido", "",
             f"Auditoría del 2026-10-08 (§2.3, E2), con [`tools/hx11_modos.py`](../../tools/hx11_modos.py) sobre "
             f"{s['partidos']} partidos aptos. Los criterios están en el docstring y se fijaron antes de mirar el "
             f"resultado. Datos por partido en `data/hx11/modos.json`.", "",
             "## Modo de cada partido", "", "| Modo | Prendido | Apagado | Indeterminado | Anunciado contra inferido |",
             "|---|---|---|---|---|"]
    for k, v in s["modos"].items():
        lines.append(f"| {k} | {v.get('prendido', 0)} | {v.get('apagado', 0)} | "
                     f"{v.get('indeterminado', 0) + v.get('desconocido', 0)} | "
                     f"{s['contradicciones'][k]} contradicciones |")
    lines += ["", f"- **Faltas cobradas sobre oportunidades** (contacto >= {FOUL_OPP} ticks) en los partidos con faltas "
                  f"prendidas: {s['faltas_cobradas_sobre_oportunidades_si_prendido']}. **Oportunidades sin cobro** "
                  f"(ninguna falta hasta 20 ticks después del contacto) en esos partidos: "
                  f"{100 * s['faltas_sin_cobro_si_prendido']:.1f}%.",
              f"- **`faltas_l0`** (lo que usa la imitación): prendido si hubo cobros o anuncio; apagado si se anunció o si "
              f"P(apagado) >= {P_OFF}, con P(apagado) = 1 / (1 + q^k) para k oportunidades sin ningún cobro, q la tasa "
              f"sin cobro de arriba y previa 0,5; desconocido si no.",
              f"- **Contactos que llegan a 100–114 ticks** (justo debajo de la regla, firma de un exploit si el RL los "
              f"concentra): {s['contacto_entre_100_y_115_ticks']} en todo el banco.",
              f"- **Partidos con un cambio de modo anunciado a mitad de partido:** {s['partidos_con_cambio_de_modo']}.",
              "", "## Por mes", "", "| Mes | Partidos | Faltas prendidas / apagadas | Offside prendido / apagado | "
                                   "Powershot prendido / apagado | Con `!gk` | Anuncia modos |",
              "|---|---|---|---|---|---|---|"]
    for m, v in s["por_mes"].items():
        lines.append(f"| {m} | {v['partidos']} | {v.get('faltas_prendido', 0)} / {v.get('faltas_apagado', 0)} | "
                     f"{v.get('offside_prendido', 0)} / {v.get('offside_apagado', 0)} | "
                     f"{v.get('powershot_prendido', 0)} / {v.get('powershot_apagado', 0)} | {v['gk']} | "
                     f"{v['anuncia_modos']} |")
    lines += ["", "## Marcadores de versión del script (anuncios presentes)", "", "| Marcadores | Partidos |",
              "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in s["versiones"].items()]
    lines += ["", "## Por parte", "", "| Parte | Faltas | Offside | Powershot |", "|---|---|---|---|"]
    for sp, v in s["por_parte"].items():
        lines.append(f"| {sp} | {v['faltas']} | {v['offside']} | {v['powershot']} |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=3)
    a = ap.parse_args()
    batch(a.jobs)


if __name__ == "__main__":
    main()
