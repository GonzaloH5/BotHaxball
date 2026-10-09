"""D2 de HaxEleven (docs/PLAN_HAXELEVEN.md): puestos y formación de cada equipo, sobre D0 (posiciones) y D1 (posesión).

* **Coordenadas del equipo:** (x, y) × (+1 rojo, −1 azul): los dos atacan hacia +x y la izquierda del equipo es −y.
  Los roles se miden respecto del centroide de los jugadores de campo (Bialkowski et al. 2014, ICDM), así la forma no
  depende de dónde está el bloque.
* **Muestreo:** juego abierto, un tick de cada 6 (10 por segundo, como en Bialkowski).
* **Arquero:** el declarado con `!gk` (D0); si no hay, el inferido por posición en D0 (bloques de 1 min) y, si
  tampoco, el más cercano al arco propio en esa muestra. No entra en los roles.
* **Roles de campo (EM + Hungarian):** 10 plantillas gaussianas; en cada muestra, los jugadores de campo presentes se
  asignan uno a uno a las plantillas con el algoritmo húngaro (costo = −log verosimilitud), y las plantillas se
  reestiman con lo asignado (8 iteraciones). Arranca de la posición media de los 10 jugadores más presentes. Con
  menos de 10 jugadores quedan roles vacíos.
* **Formación** por ventana de 3 minutos de partido y por fase (con pelota / sin pelota, según el último toque de D1):
  medias de los roles, normalizadas (z por eje), contra plantillas canónicas (4-4-2, 4-3-3, 3-4-3, 3-5-2, 4-2-3-1,
  5-3-2, 4-5-1, 5-4-1, 4-1-4-1) con el húngaro; gana la de menor costo, "otra" si el costo pasa de `OTHER_COST`.
* **Nombre del rol** según la formación del partido sin pelota: línea (DEF, MCD, MED, DEL, de atrás hacia adelante) y
  lado (I, CI, C, CD, D, de izquierda a derecha del equipo). El arquero es ARQ.
* **Ancla:** posición media de cada rol en coordenadas del equipo (absolutas, no respecto del centroide), con y sin
  pelota: es el "puesto como condición" de §4 del plan.

Por partido, en data/hx11/d2/ (no versionado): `<nombre>.npz` (`ticks` muestreados y `role` (n, 22): rol 0–9, 10 =
arquero, −1 ausente) y `<nombre>.json` (plantillas, anclas, formaciones por ventana, rol dominante de cada jugador).

  python -m tools.hx11_d2 lote [--jobs 3] [--limit N]    # → data/hx11/d2 y reports/hx11/d2_puestos.{md,json}
  python -m tools.hx11_d2 muestra --matches 10            # data/hx11/d2/muestra_revision.md (con nombres: no versionar)
"""
from __future__ import annotations

import argparse
import collections
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from numba import njit

from env.hx11 import contract as HC
from tools.hx11_d0 import OPEN, apt_names, load_d0

ROOT = Path(__file__).resolve().parent.parent
D0 = ROOT / "data" / "hx11" / "d0"
D1 = ROOT / "data" / "hx11" / "d1"
OUT = ROOT / "data" / "hx11" / "d2"
REPORTS = ROOT / "reports" / "hx11"

STRIDE = 6
N_ROLES = 10
GK_ROLE = 10
EM_ITERS = 8
REG = 25.0                     # px², regularización de las covarianzas
WINDOW = 3 * 60 * 60           # ventana de formación: 3 minutos de partido
MIN_SAMPLES = 100              # muestras por ventana y fase para clasificar
OTHER_COST = 0.35
ATT = (1.0, -1.0)
FORMATIONS = {"4-4-2": (4, 4, 2), "4-3-3": (4, 3, 3), "3-4-3": (3, 4, 3), "3-5-2": (3, 5, 2),
              "4-2-3-1": (4, 2, 3, 1), "5-3-2": (5, 3, 2), "4-5-1": (4, 5, 1), "5-4-1": (5, 4, 1),
              "4-1-4-1": (4, 1, 4, 1)}
SIDES = {1: ("C",), 2: ("CI", "CD"), 3: ("I", "C", "D"), 4: ("I", "CI", "CD", "D"), 5: ("I", "CI", "C", "CD", "D")}
LINES = {3: ("DEF", "MED", "DEL"), 4: ("DEF", "MCD", "MED", "DEL")}


@njit(cache=True)
def hungarian(a):
    """Asignación de costo mínimo en una matriz cuadrada: ans[fila] = columna (algoritmo húngaro, O(n³))."""
    n = a.shape[0]
    inf = 1e18
    u = np.zeros(n + 1)
    v = np.zeros(n + 1)
    p = np.zeros(n + 1, np.int64)
    way = np.zeros(n + 1, np.int64)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(n + 1, inf)
        used = np.zeros(n + 1, np.bool_)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = inf
            j1 = 0
            for j in range(1, n + 1):
                if not used[j]:
                    cur = a[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    ans = np.empty(n, np.int64)
    for j in range(1, n + 1):
        ans[p[j] - 1] = j - 1
    return ans


@njit(cache=True)
def assign_frames(rel, present, mu, icov, logdet):
    """Rol de cada jugador de campo por muestra: rel (F, 10, 2), present (F, 10) → (F, 10) con −1 si no está."""
    F, n = present.shape
    R = mu.shape[0]
    out = np.full((F, n), -1, np.int64)
    cost = np.zeros((R, R))
    for f in range(F):
        cost[:, :] = 0.0
        for i in range(n):
            if not present[f, i]:
                continue
            for r in range(R):
                dx = rel[f, i, 0] - mu[r, 0]
                dy = rel[f, i, 1] - mu[r, 1]
                m = dx * (icov[r, 0, 0] * dx + icov[r, 0, 1] * dy) + dy * (icov[r, 1, 0] * dx + icov[r, 1, 1] * dy)
                cost[i, r] = 0.5 * m + 0.5 * logdet[r]
        ans = hungarian(cost)
        for i in range(n):
            if present[f, i]:
                out[f, i] = ans[i]
    return out


def templates(n_lines_counts):
    """Plantilla canónica de una formación (10 puntos, z por eje), de atrás hacia adelante y de izquierda a derecha."""
    pts = []
    k = len(n_lines_counts)
    for li, m in enumerate(n_lines_counts):
        x = -1.0 + 2.0 * li / (k - 1)
        for j in range(m):
            pts.append((x, j - (m - 1) / 2))
    p = np.array(pts, np.float64)
    return (p - p.mean(0)) / p.std(0)


CANON = {k: templates(v) for k, v in FORMATIONS.items()}


def classify(means):
    """(formación, costo, asignación rol → índice en la plantilla) para 10 medias de rol (coordenadas del equipo)."""
    ok = np.isfinite(means).all(1)
    if ok.sum() < N_ROLES:
        return "incompleta", None, None
    z = (means - means.mean(0)) / (means.std(0) + 1e-9)
    best = ("otra", np.inf, None)
    for name, c in CANON.items():
        d = ((z[:, None, :] - c[None, :, :]) ** 2).sum(-1)
        a = hungarian(d)
        cost = float(d[np.arange(N_ROLES), a].mean())
        if cost < best[1]:
            best = (name, cost, a)
    if best[1] > OTHER_COST:
        return "otra", best[1], best[2]
    return best


def role_names(formation, assign):
    """Nombre de cada uno de los 10 roles según la formación (línea-lado) y su asignación a la plantilla."""
    counts = FORMATIONS[formation]
    labels = []
    lines = LINES[len(counts)]
    for li, m in enumerate(counts):
        labels += [f"{lines[li]}-{s}" for s in SIDES[m]]
    return [labels[assign[r]] for r in range(N_ROLES)]


def team_frames(o, owner, team, ticks):
    """Jugadores de campo del equipo en las muestras: rel (F, 10, 2), abs (F, 10, 2), present, slots (F, 10), gk."""
    att = ATT[team]
    sl = np.arange(team * 11, team * 11 + 11)
    pos = o["pos"][ticks][:, sl].astype(np.float64) / 10 * att
    pid = o["slot_pid"][ticks][:, sl]
    on = pid >= 0
    gkp = o["gk_pid"][ticks, team]
    if "gk_inferred" in o:   # sin !gk: el inferido por posición en bloques de 1 min (D0)
        gkp = np.where(gkp >= 0, gkp, o["gk_inferred"][ticks, team])
    is_gk = (pid == gkp[:, None]) & (gkp[:, None] >= 0)
    # sin arquero declarado: el más cercano al arco propio
    no = ~is_gk.any(1) & on.any(1)
    if no.any():
        xs = np.where(on[no], pos[no, :, 0], np.inf)
        is_gk[np.nonzero(no)[0], xs.argmin(1)] = True
    F = len(ticks)
    rel = np.zeros((F, N_ROLES, 2))
    ab = np.zeros((F, N_ROLES, 2))
    present = np.zeros((F, N_ROLES), np.bool_)
    slots = np.full((F, N_ROLES), -1, np.int64)
    field = on & ~is_gk
    for f in range(F):
        c = np.nonzero(field[f])[0][:N_ROLES]
        if not len(c):
            continue
        q = pos[f, c]
        rel[f, :len(c)] = q - q.mean(0)
        ab[f, :len(c)] = q
        present[f, :len(c)] = True
        slots[f, :len(c)] = sl[c]
    gk_slot = np.where(is_gk.any(1), sl[is_gk.argmax(1)], -1)
    poss = np.where(owner[ticks] >= 0, owner[ticks] // 11, -1)
    phase = np.where(poss == team, 1, np.where(poss == 1 - team, 0, -1))   # 1 con pelota, 0 sin, −1 nadie
    return rel, ab, present, slots, pid, gk_slot, phase


def em_roles(rel, present, slots, pid_of_slot):
    """Plantillas de rol (medias y covarianzas) y rol de cada jugador por muestra."""
    F = len(rel)
    # arranque: posición media de los 10 jugadores (id) más presentes
    ids = pid_of_slot
    cnt = collections.Counter(ids[present].tolist())
    top = [p for p, _ in cnt.most_common(N_ROLES)]
    mu = np.zeros((N_ROLES, 2))
    for r, p in enumerate(top):
        mu[r] = rel[(ids == p) & present].mean(0)
    for r in range(len(top), N_ROLES):
        mu[r] = np.random.default_rng(r).normal(0, 100, 2)
    cov = np.tile(np.eye(2) * 150.0 ** 2, (N_ROLES, 1, 1))
    roles = None
    for _ in range(EM_ITERS):
        icov = np.linalg.inv(cov)
        logdet = np.log(np.linalg.det(cov))
        roles = assign_frames(rel, present, mu, icov, logdet)
        for r in range(N_ROLES):
            x = rel[roles == r]
            if len(x) > 10:
                mu[r] = x.mean(0)
                cov[r] = np.cov(x.T) + np.eye(2) * REG
    return mu, cov, roles


def analyze(name):
    o, meta = load_d0(D0 / name)
    owner = np.load(D1 / f"{name}.npz")["owner"]
    T = len(o["frame"])
    ticks = np.nonzero(o["phase"] == OPEN)[0]
    ticks = ticks[(ticks % STRIDE) == 0]
    role_all = np.full((len(ticks), 22), -1, np.int8)
    teams = []
    for team in (0, 1):
        rel, ab, present, slots, pid, gk_slot, phase = team_frames(o, owner, team, ticks)
        pid_of = np.where(slots >= 0, o["slot_pid"][ticks][np.arange(len(ticks))[:, None], np.maximum(slots, 0)], -1)
        mu, cov, roles = em_roles(rel, present, slots, pid_of)
        F = len(ticks)
        for f in range(F):
            for i in range(N_ROLES):
                if slots[f, i] >= 0:
                    role_all[f, slots[f, i]] = roles[f, i]
            if gk_slot[f] >= 0:
                role_all[f, gk_slot[f]] = GK_ROLE
        # anclas y formación del partido, con y sin pelota
        anchors, form = {}, {}
        for ph, key in ((1, "con_pelota"), (0, "sin_pelota")):
            m = phase[:, None] == ph
            a = np.full((N_ROLES, 2), np.nan)
            rm = np.full((N_ROLES, 2), np.nan)
            for r in range(N_ROLES):
                sel = (roles == r) & m
                if sel.sum() >= MIN_SAMPLES:
                    a[r] = ab[sel].mean(0)
                    rm[r] = rel[sel].mean(0)
            anchors[key] = a
            f_, cost, asg = classify(rm)
            form[key] = dict(formacion=f_, costo=None if cost is None else round(cost, 3),
                             asignacion=None if asg is None else asg.tolist())
        base = form["sin_pelota"] if form["sin_pelota"]["formacion"] in FORMATIONS else form["con_pelota"]
        if base["formacion"] in FORMATIONS:
            names = role_names(base["formacion"], base["asignacion"])
        else:
            f_, cost, asg = classify(mu)
            names = role_names(f_, asg) if f_ in FORMATIONS else [f"R{r}" for r in range(N_ROLES)]
        # formación por ventana
        windows = []
        for w0 in range(0, T, WINDOW):
            wm = (ticks >= w0) & (ticks < w0 + WINDOW)
            row = dict(desde=int(w0), hasta=int(min(T, w0 + WINDOW)))
            for ph, key in ((1, "con_pelota"), (0, "sin_pelota")):
                m = (wm & (phase == ph))[:, None]
                rm = np.full((N_ROLES, 2), np.nan)
                for r in range(N_ROLES):
                    sel = (roles == r) & m
                    if sel.sum() >= MIN_SAMPLES // 2:
                        rm[r] = rel[sel].mean(0)
                f_, cost, _ = classify(rm)
                row[key] = f_
            windows.append(row)
        # rol dominante de cada jugador (por id) y su estabilidad
        # los 10 más presentes reciben roles distintos (húngaro sobre las muestras en cada rol); el resto, su rol más
        # frecuente
        players = {}
        ids = np.unique(pid_of[pid_of >= 0])
        # un equipo sin jugadores de campo en las muestras deja `ids` vacío: counts tiene que seguir siendo (0, 10)
        counts = np.array([np.bincount(roles[(pid_of == p) & present], minlength=N_ROLES) for p in ids],
                          np.int64).reshape(-1, N_ROLES)
        top = np.argsort(-counts.sum(1))[:N_ROLES]
        dom = counts.argmax(1)
        if len(top) == N_ROLES:
            dom[top] = hungarian((-counts[top]).astype(np.float64))
        for k, p in enumerate(ids):
            c = counts[k]
            r = int(dom[k])
            players[int(p)] = dict(rol=r, nombre_rol=names[r], muestras=int(c.sum()),
                                   estabilidad=round(float(c[r] / c.sum()), 3))
        gk_ids = collections.Counter(o["slot_pid"][ticks[f], gk_slot[f]] for f in range(F) if gk_slot[f] >= 0)
        for p, n in gk_ids.items():
            players.setdefault(int(p), dict(rol=GK_ROLE, nombre_rol="ARQ", muestras=0, estabilidad=None))
            if n >= players[int(p)]["muestras"]:
                players[int(p)] = dict(rol=GK_ROLE, nombre_rol="ARQ", muestras=int(n), estabilidad=None)
        teams.append(dict(equipo=team, formacion=form, nombres=names, medias=mu.round(1).tolist(),
                          anclas={k: np.round(v, 1).tolist() for k, v in anchors.items()}, ventanas=windows,
                          jugadores=players))
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{name}.npz", ticks=ticks.astype(np.int32), role=role_all)
    d = dict(name=name, equipos=teams)
    (OUT / f"{name}.json").write_text(json.dumps(d), encoding="utf-8")
    return d


def summarize(ds):
    form = {k: collections.Counter() for k in ("con_pelota", "sin_pelota")}
    pair = collections.Counter()
    win = {k: collections.Counter() for k in ("con_pelota", "sin_pelota")}
    changes = []
    stab = []
    costs = []
    for d in ds:
        for t in d["equipos"]:
            for k in form:
                form[k][t["formacion"][k]["formacion"]] += 1
                if t["formacion"][k]["costo"] is not None:
                    costs.append(t["formacion"][k]["costo"])
            pair[(t["formacion"]["con_pelota"]["formacion"], t["formacion"]["sin_pelota"]["formacion"])] += 1
            seq = []
            for w in t["ventanas"]:
                for k in win:
                    win[k][w[k]] += 1
                seq.append(w["sin_pelota"])
            seq = [s for s in seq if s in FORMATIONS]
            changes.append(sum(a != b for a, b in zip(seq, seq[1:])))
            stab += [p["estabilidad"] for p in t["jugadores"].values()
                     if p["estabilidad"] is not None and p["muestras"] >= 1000]
    return dict(partidos=len(ds), equipos=2 * len(ds),
                formacion_partido={k: dict(v.most_common()) for k, v in form.items()},
                con_vs_sin={f"{a} | {b}": n for (a, b), n in pair.most_common(15)},
                formacion_ventana={k: dict(v.most_common()) for k, v in win.items()},
                cambios_por_partido=[float(np.mean(changes)), int(np.percentile(changes, 90))] if changes else None,
                estabilidad_rol=[round(float(np.percentile(stab, q)), 3) for q in (10, 50, 90)] if stab else None,
                costo_plantilla=[round(float(np.percentile(costs, q)), 3) for q in (10, 50, 90)] if costs else None)


def report(s):
    def table(c):
        n = sum(c.values())
        return ["| Formación | Equipos | % |", "|---|---|---|"] + [f"| {k} | {v} | {100 * v / n:.1f}% |"
                                                                for k, v in c.items()]
    lines = ["# D2: puestos y formación en HaxEleven", "",
             f"Etapa D2 de [`docs/PLAN_HAXELEVEN.md`](../../docs/PLAN_HAXELEVEN.md), con el método de "
             f"[`tools/hx11_d2.py`](../../tools/hx11_d2.py). Son {s['partidos']} partidos ({s['equipos']} equipos) del "
             f"estadio actual. Los datos están en [`d2_puestos.json`](d2_puestos.json).", "",
             "**Falta la validación:** el arquero sale de `!gk` y es exacto, pero los roles de campo y la formación no "
             "tienen verdad todavía. Hace falta la muestra del usuario (`python -m tools.hx11_d2 muestra`) o las "
             "alineaciones de Discord.", "",
             "## Formación del partido sin pelota", ""] + table(s["formacion_partido"]["sin_pelota"]) + [
             "", "## Formación del partido con pelota", ""] + table(s["formacion_partido"]["con_pelota"]) + [
             "", "## Con pelota | sin pelota (los 15 pares más comunes)", "", "| Par | Equipos |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in s["con_vs_sin"].items()]
    lines += ["", "## Ventanas de 3 minutos sin pelota", ""] + table(s["formacion_ventana"]["sin_pelota"])
    lines += ["", "## Estabilidad", "",
              f"- Cambios de formación sin pelota entre ventanas seguidas, por equipo y partido: media "
              f"{s['cambios_por_partido'][0]:.1f}, p90 {s['cambios_por_partido'][1]}.",
              f"- Parte de las muestras en que un jugador está en su rol dominante (jugadores con >= 100 s de juego): "
              f"p10 {s['estabilidad_rol'][0]}, p50 {s['estabilidad_rol'][1]}, p90 {s['estabilidad_rol'][2]}.",
              f"- Costo contra la plantilla canónica (z por eje; \"otra\" si pasa de {OTHER_COST}): p10 "
              f"{s['costo_plantilla'][0]}, p50 {s['costo_plantilla'][1]}, p90 {s['costo_plantilla'][2]}."]
    return "\n".join(lines) + "\n"


def batch(jobs, limit=None):
    names = [n for n in apt_names() if (D1 / f"{n}.npz").exists()][:limit]
    ds = []
    with ProcessPoolExecutor(jobs) as ex:
        for k, d in enumerate(ex.map(analyze, names, chunksize=4), 1):
            ds.append(d)
            if k % 50 == 0:
                print(f"{k}/{len(names)}", flush=True)
    s = summarize(ds)
    (REPORTS / "d2_puestos.json").write_text(json.dumps(s, indent=1, ensure_ascii=False), encoding="utf-8")
    (REPORTS / "d2_puestos.md").write_text(report(s), encoding="utf-8")
    print(json.dumps(s["formacion_partido"], ensure_ascii=False))
    return s


def sample(n_matches, seed=0):
    """Por partido y equipo: formación y puesto de cada jugador con su nombre, para que el usuario lo corrija."""
    from tools.hx11_d1 import names_by_recording
    rng = np.random.default_rng(seed)
    files = sorted(OUT.glob("*.json"))
    apt = set(apt_names())
    files = [f for f in files if not f.name.startswith("muestra") and f.stem in apt]
    names = names_by_recording()
    lines = ["# Muestra de puestos y formaciones D2 para revisar (con nombres: no versionar)", "",
             "Por equipo: la formación que se detectó y el puesto de cada jugador. Corregir lo que esté mal "
             "(formación real y puesto real de cada uno); alcanza con un partido por línea.", ""]
    for f in rng.permutation(files)[:n_matches]:
        d = json.loads(f.read_text(encoding="utf-8"))
        nm = names.get(d["name"], {})
        lines += [f"## {d['name']}", ""]
        for t in d["equipos"]:
            fo = t["formacion"]
            lines.append(f"**{'Rojo' if t['equipo'] == 0 else 'Azul'}:** sin pelota {fo['sin_pelota']['formacion']}, "
                         f"con pelota {fo['con_pelota']['formacion']}. Por ventana de 3 min (sin pelota): "
                         + ", ".join(w["sin_pelota"] for w in t["ventanas"]))
            lines.append("")
            lines.append("| Jugador | Puesto detectado | Tiempo jugado | Estabilidad | ¿Bien? / puesto real |")
            lines.append("|---|---|---|---|---|")
            order = sorted(t["jugadores"].items(), key=lambda kv: (-1 if kv[1]["rol"] == GK_ROLE else 0,
                                                                   -kv[1]["muestras"]))
            for p, r in order:
                if r["muestras"] < 300:
                    continue
                lines.append(f"| {nm.get(str(p), '?')} | {r['nombre_rol']} | {r['muestras'] / 600:.1f} min | "
                             f"{r['estabilidad'] if r['estabilidad'] is not None else '—'} | |")
            lines.append("")
    path = OUT / "muestra_revision.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    u = sub.add_parser("uno")
    u.add_argument("name")
    b = sub.add_parser("lote")
    b.add_argument("--jobs", type=int, default=3)
    b.add_argument("--limit", type=int, default=None)
    m = sub.add_parser("muestra")
    m.add_argument("--matches", type=int, default=10)
    m.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.cmd == "uno":
        d = analyze(a.name)
        for t in d["equipos"]:
            print(t["equipo"], json.dumps(t["formacion"]), t["nombres"])
            print("  ventanas", [(w["con_pelota"], w["sin_pelota"]) for w in t["ventanas"]])
            print("  jugadores", sorted(((v["nombre_rol"], v["muestras"], v["estabilidad"]) for v in
                                         t["jugadores"].values()), key=lambda x: -x[1])[:14])
    elif a.cmd == "lote":
        batch(a.jobs, a.limit)
    else:
        sample(a.matches, a.seed)


if __name__ == "__main__":
    main()
