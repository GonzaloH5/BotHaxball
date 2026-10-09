"""Referencias y marcas en HaxEleven (D2 y D3 de docs/PLAN_HAXELEVEN.md; pedido del usuario, 2026-10-08).

El usuario: en HaxBall todos tienen la misma velocidad y no hay regate ni físico distinto. La ventaja sale de ganar la
posición, y una línea de 4 que se queda en su lugar mientras el 9 humano se ubica para que un pase deje al defensor
detrás suyo es una jugada de gol fácil. La forma de D2 no lo ve: mide al equipo contra sí mismo. Esto lo mide contra el
rival, en cada muestra de juego abierto con la pelota del rival (un tick de cada 6):

* **Quién marca a quién** (Franks, Miller, Bornn y Goldsberry 2015, *Ann. Appl. Stat.*, 1405.0231, hecho para básquet).
  Para cada jugador de campo del equipo sin la pelota, un HMM cuyo estado oculto es el atacante de campo que marca. La
  posición esperada del defensor es γo·atacante + γb·pelota + γh·arco propio (γ suman 1), con ruido gaussiano. Se
  queda con la misma marca con P = 0,97 por muestra. γ y σ se ajustan por partido con 3 vueltas de Viterbi y mínimos
  cuadrados. En la prueba dio γo ≈ 0,9–0,96: el modelo termina eligiendo al atacante más cercano, así que γ no
  distingue marca personal de zonal. Sirve como referencia de cada defensor, no como estilo.
* **Seguimiento (personal o zonal):** ventanas de 3 s, una por segundo, con la referencia fija desde el comienzo
  (si el HMM la cambia en el medio, no cuenta). Los desplazamientos del defensor (dd), de esa referencia (da) y de la
  pelota (db) se miden respecto del centroide del resto de los jugadores de campo de su equipo, para descontar el
  corrimiento del bloque. Mínimos cuadrados de dd = βa·da + βb·db: βa ≈ 1 es marca personal (la acompaña aunque se
  vaya), βa ≈ 0 es zonal (la deja ir y se queda en su lugar del bloque). Cuando la referencia se movió al menos
  150 px respecto del bloque, al final de la ventana el defensor de campo más cercano a ella es el mismo (sigue), es
  otro (entrega) o no hay ninguno a 150 px (libre). Es geométrico: la referencia del HMM se pega demasiado para esto.
* **Del lado del arco:** el defensor está más cerca de su arco que su referencia (en x del equipo).
* **Margen sobre la espalda:** carrera al espacio a su espalda en el carril del atacante, el punto
  (min(x del defensor, x del atacante) − 200, y del atacante) en coordenadas del defensor. Tiempo de cada uno =
  reacción + distancia / vmax, desde donde estaría tras la reacción siguiendo su velocidad (el modelo de control de
  `tools.hx11_d3`, con la reacción ajustada ahí). Margen = tiempo del atacante − tiempo del defensor, en ticks:
  negativo es espalda ganada. Como todos tienen la misma velocidad, sólo dependen de la ubicación, la velocidad que
  traen y la reacción.
* **Habilitado:** el atacante no está en offside (campo propio, detrás de la pelota o a la altura del anteúltimo
  defensor, arquero incluido). Si la línea la marca otro defensor más atrasado que el que marca, es la línea la que
  lo habilita.
* **Peligro (OBSO liviano, Spearman 2018):** P(el equipo atacante controla ese espacio, con la pelota saliendo de donde
  está) × xG desde ahí. El xG es una logística de distancia y ángulo ajustada con los remates de D1 de train.
* **Marca perdida:** el margen de un defensor con su referencia habilitada cae debajo de −10 ticks (lo cruza en esa
  muestra) con peligro >= 0,02. Resultado en 5 s: el atacante recibe a la espalda del defensor, el equipo atacante
  remata o hace gol.

Por partido: `data/hx11/d2/<nombre>_marcas.npz` (referencia, margen y peligro por muestra y jugador). Resumen por línea
del puesto (D2) en `reports/hx11/d2_marcas.{md,json}`, con la comparación contra muestras al azar de la misma
posesión rival (si la marca perdida no anticipa más remates que cualquier momento, no sirve).

  python -m tools.hx11_marcas lote [--jobs 3]
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from numba import njit

from env.hx11 import contract as HC
from tools.hx11_d0 import OPEN, apt_names, load_d0
from tools.hx11_d3 import VMAX, control

ROOT = Path(__file__).resolve().parent.parent
D0 = ROOT / "data" / "hx11" / "d0"
D1 = ROOT / "data" / "hx11" / "d1"
D2 = ROOT / "data" / "hx11" / "d2"
REPORTS = ROOT / "reports" / "hx11"

STRIDE = 6
STAY = 0.97
EM_ITERS = 3
BEHIND = 200.0
LOST_MARGIN = -10.0
DANGER_MIN = 0.02
OUTCOME = 300
BLOCK_GAP = 30
FOLLOW = 180
FOLLOW_STEP = 60
MOVED = 150.0
FREE = 150.0
ATT = (1.0, -1.0)
TEAM = np.r_[np.zeros(11, np.int64), np.ones(11, np.int64)]


@njit(cache=True)
def viterbi(logE, stay):
    """Camino más probable de estados (F, K) con permanencia `stay` y cambio uniforme al resto."""
    F, K = logE.shape
    ls = math.log(stay)
    lm = math.log((1 - stay) / max(1, K - 1))
    delta = logE[0].copy()
    back = np.zeros((F, K), np.int64)
    for f in range(1, F):
        b1, b2 = -1, -1
        for k in range(K):
            if b1 < 0 or delta[k] > delta[b1]:
                b2 = b1
                b1 = k
            elif b2 < 0 or delta[k] > delta[b2]:
                b2 = k
        new = np.empty(K)
        for k in range(K):
            j = b1 if b1 != k else b2
            sw = delta[j] + lm if j >= 0 else -1e300
            st = delta[k] + ls
            if st >= sw:
                new[k] = st + logE[f, k]
                back[f, k] = k
            else:
                new[k] = sw + logE[f, k]
                back[f, k] = j
        delta = new
    path = np.empty(F, np.int64)
    path[F - 1] = np.argmax(delta)
    for f in range(F - 1, 0, -1):
        path[f - 1] = back[f, path[f]]
    return path


def follow(fr, q, b, dsl, asl, refs, d_ok, a_ok, pids):
    """Seguimiento de cada defensor a la referencia que tenía al empezar cada ventana (ver el docstring del módulo).

    fr: ticks de las muestras; q (F, 22, 2) y b (F, 2) en coordenadas del equipo que defiende; refs (F, 11) índice en
    `asl` o −1; d_ok y a_ok (F, 11) jugadores de campo presentes; pids (F, 22) para no mezclar a dos jugadores que
    pasaron por el mismo lugar. Devuelve, por defensor, las ecuaciones normales de dd = βa·da + βb·db y el destino de
    las referencias que se movieron."""
    F = len(fr)
    blk = np.cumsum(np.r_[0, np.diff(fr) > BLOCK_GAP])
    g = np.searchsorted(fr, fr + FOLLOW)
    gc = np.minimum(g, F - 1)
    st = np.nonzero((g < F) & (fr[gc] == fr + FOLLOW) & (blk[gc] == blk) & (fr % FOLLOW_STEP == 0))[0]
    g = gc[st]
    D = q[:, dsl]
    tot = (D * d_ok[..., None]).sum(1)
    n = d_ok.sum(1)
    res = []
    for i in range(11):
        k = refs[st, i]
        kk = np.maximum(k, 0)
        ok = ((k >= 0) & d_ok[st, i] & d_ok[g, i] & (n[st] > 1) & (n[g] > 1) & a_ok[st, kk] & a_ok[g, kk]
              & (pids[st, dsl[i]] == pids[g, dsl[i]]) & (pids[st, asl[kk]] == pids[g, asl[kk]]))
        s, e, kk = st[ok], g[ok], kk[ok]
        c0 = (tot[s] - D[s, i]) / (n[s] - 1)[:, None]
        c1 = (tot[e] - D[e, i]) / (n[e] - 1)[:, None]
        dd = (D[e, i] - c1) - (D[s, i] - c0)
        da = (q[e, asl[kk]] - c1) - (q[s, asl[kk]] - c0)
        db = (b[e] - c1) - (b[s] - c0)
        X = np.stack([da, db], -1).reshape(-1, 2)
        moved = np.hypot(da[:, 0], da[:, 1]) >= MOVED
        gap = np.linalg.norm(D[e] - q[e, asl[kk]][:, None], axis=-1)
        gap = np.where(d_ok[e], gap, np.inf)
        near = gap.min(1) <= FREE
        keep = near & (gap.argmin(1) == i)
        taken = near & ~keep
        res.append(dict(xtx=X.T @ X, xty=X.T @ dd.reshape(-1), n=int(ok.sum()), movida=int(moved.sum()),
                        sigue=int((moved & keep).sum()), entrega=int((moved & taken).sum())))
    return res


def beta(xtx, xty):
    """(βa, βb) de las ecuaciones normales del seguimiento."""
    return np.linalg.lstsq(np.asarray(xtx) + 1e-6 * np.eye(2), np.asarray(xty), rcond=None)[0]


def fit_xg(names):
    """Logística P(gol | distancia, ángulo) con los remates de D1 (Newton)."""
    X, y = [], []
    for n in names:
        for e in json.loads((D1 / f"{n}.json").read_text(encoding="utf-8"))["events"]:
            if e["tipo"] == "remate":
                att = ATT[e["equipo"]]
                X.append(_xg_feats(e["x"] * att, e["y"] * att))
                y.append(e["resultado"] == "gol")
    X, y = np.array(X), np.array(y, np.float64)
    w = np.zeros(X.shape[1])
    for _ in range(25):
        p = 1 / (1 + np.exp(-X @ w))
        H = X.T @ (X * (p * (1 - p))[:, None]) + 1e-3 * np.eye(len(w))
        w += np.linalg.solve(H, X.T @ (y - p))
    return w, len(y), float(y.mean())


def _xg_feats(x, y):
    """x, y en coordenadas del que remata (ataca hacia +x)."""
    dx = max(HC.LINE_W - x, 1.0)
    d = math.hypot(dx, y)
    return [1.0, d / 1000, abs(math.atan2(y, dx)), math.log(d / 100 + 1e-3)]


def xg(w, x, y):
    return 1 / (1 + math.exp(-float(np.dot(w, _xg_feats(x, y)))))


def match(args):
    name, w_xg, reaction, sigma = args
    o, _ = load_d0(D0 / name)
    owner = np.load(D1 / f"{name}.npz")["owner"]
    ev = json.loads((D1 / f"{name}.json").read_text(encoding="utf-8"))["events"]
    d2 = json.loads((D2 / f"{name}.json").read_text(encoding="utf-8"))
    pos = o["pos"].astype(np.float64) / 10
    vel = o["vel"].astype(np.float64) / 1000
    ball = o["ball_pos"].astype(np.float64) / 10
    pid = o["slot_pid"]
    on = pid >= 0
    gk = np.where(o["gk_pid"] >= 0, o["gk_pid"], o["gk_inferred"])
    T = len(owner)
    ticks = np.nonzero((o["phase"] == OPEN) & (owner >= 0))[0]
    ticks = ticks[ticks % STRIDE == 0]
    shots = [(e["t"], e["equipo"]) for e in ev if e["tipo"] == "remate"]
    goals = [(e["t"], e["equipo"]) for e in ev if e["tipo"] == "gol"]
    recs = [(e["t"], e["lugar"], e["x"], e["y"]) for e in ev if e["tipo"] == "recepcion"]
    role_of = {}
    for t in d2["equipos"]:
        for p, info in t["jugadores"].items():
            role_of[(t["equipo"], int(p))] = info["nombre_rol"]
    out = dict(gamma=[], sigma=[], lost=[], frames=collections.Counter(), by_role=collections.defaultdict(
        lambda: collections.Counter()), margins=collections.defaultdict(list), stability=collections.defaultdict(list),
        random=collections.Counter(), follow={}, follow_player=collections.defaultdict(list), follow_team=[])
    rng = np.random.default_rng(zlib.crc32(name.encode()))
    teamv = TEAM
    arrays = {}
    for D in (0, 1):
        A = 1 - D
        att = ATT[D]
        fr = ticks[owner[ticks] // 11 == A]
        if len(fr) < 50:
            continue
        dsl = np.arange(D * 11, D * 11 + 11)
        asl = np.arange(A * 11, A * 11 + 11)
        q = pos[fr] * att                         # coordenadas del equipo D: su arco en (−1800, 0)
        b = ball[fr] * att
        H = np.array([-HC.LINE_W, 0.0])
        is_gk = pid[fr] == gk[fr][:, [0] * 11 + [1] * 11]
        a_ok = on[fr][:, asl] & ~is_gk[:, asl]
        d_ok = on[fr][:, dsl] & ~is_gk[:, dsl]
        F = len(fr)
        brk = np.r_[0, np.nonzero(np.diff(fr) > BLOCK_GAP)[0] + 1, F]
        gamma = np.array([0.6, 0.1, 0.3])
        sig = 100.0
        refs = np.full((F, 11), -1, np.int64)
        for _ in range(EM_ITERS):
            mu = (gamma[0] * q[:, asl] + gamma[1] * b[:, None] + gamma[2] * H)          # (F, 11, 2)
            Xs, Ys = [], []
            for i in range(11):
                if not d_ok[:, i].any():
                    continue
                d2_ = ((q[:, dsl[i]][:, None] - mu) ** 2).sum(-1)
                logE = np.where(a_ok, -d2_ / (2 * sig ** 2), -1e9)
                for s0, s1 in zip(brk[:-1], brk[1:]):
                    m = d_ok[s0:s1, i]
                    if m.sum() < 2:
                        continue
                    idx = np.nonzero(m)[0] + s0
                    path = viterbi(logE[idx], STAY)
                    refs[idx, i] = path
                sel = refs[:, i] >= 0
                k = refs[sel, i]
                o_k = q[sel][np.arange(sel.sum()), asl[k]]
                Xs.append(np.stack([o_k - H, b[sel] - H], -1).reshape(-1, 2))
                Ys.append((q[sel, dsl[i]] - H).reshape(-1))
            if not Xs:
                break
            X, Y = np.concatenate(Xs), np.concatenate(Ys)
            c = np.linalg.lstsq(X, Y, rcond=None)[0]
            gamma = np.array([c[0], c[1], 1 - c[0] - c[1]])
            sig = float(np.sqrt(((Y - X @ c) ** 2).mean()))
        out["gamma"].append(gamma.round(3).tolist())
        out["sigma"].append(round(sig, 1))
        # margen, habilitación y peligro
        margin = np.full((F, 11), np.nan)
        danger = np.full((F, 11), np.nan)
        onside = np.zeros((F, 11), bool)
        line_by = np.full(F, -1, np.int64)
        pv = vel[fr] * att
        for f in range(F):
            xs_d = np.where(on[fr[f], dsl], q[f, dsl, 0], np.inf)
            order = np.argsort(xs_d)
            line = xs_d[order[1]] if np.isfinite(xs_d[order[1]]) else xs_d[order[0]]
            line_by[f] = dsl[order[1]]
            for i in range(11):
                k = refs[f, i]
                if k < 0:
                    continue
                pa, pd = q[f, asl[k]], q[f, dsl[i]]
                onside[f, i] = pa[0] >= 0 or pa[0] >= b[f, 0] or pa[0] >= line
                S = np.array([max(min(pd[0], pa[0]) - BEHIND, -HC.LINE_W), pa[1]])
                ta = reaction + np.hypot(*(S - (pa + pv[f, asl[k]] * reaction))) / VMAX
                td = reaction + np.hypot(*(S - (pd + pv[f, dsl[i]] * reaction))) / VMAX
                margin[f, i] = ta - td
                if onside[f, i] and margin[f, i] < 0:
                    pc = control(q[f], pv[f], teamv, on[fr[f]], b[f], S, A, reaction, sigma)
                    # xG desde S visto por el atacante: su ataque es hacia −x en coordenadas de D
                    danger[f, i] = pc * xg(w_xg, -S[0], -S[1])
        arrays[f"ref_{D}"] = refs.astype(np.int8)
        arrays[f"margin_{D}"] = margin.astype(np.float16)
        arrays[f"danger_{D}"] = danger.astype(np.float16)
        arrays[f"ticks_{D}"] = fr.astype(np.int32)
        fol = follow(fr, q, b, dsl, asl, refs, d_ok, a_ok, pid[fr])
        team_xtx, team_xty = np.zeros((2, 2)), np.zeros(2)
        # agregados por defensor (rol de D2)
        for i in range(11):
            sel = refs[:, i] >= 0
            if sel.sum() < 100:
                continue
            p = int(pid[fr[sel][0], dsl[i]])
            role = role_of.get((D, p), "?").split("-")[0]
            fo = fol[i]
            fr_ = out["follow"].setdefault(role, dict(xtx=np.zeros((2, 2)), xty=np.zeros(2), n=0, movida=0, sigue=0,
                                                      entrega=0))
            for key in fr_:
                fr_[key] = fr_[key] + fo[key]
            team_xtx += fo["xtx"]
            team_xty += fo["xty"]
            if fo["n"] >= 30:
                out["follow_player"][role].append(round(float(beta(fo["xtx"], fo["xty"])[0]), 3))
            pr = out["by_role"][role]
            pr["muestras"] += int(sel.sum())
            pr["del_lado_del_arco"] += int((q[sel, dsl[i], 0] < q[sel][np.arange(sel.sum()), asl[refs[sel, i]], 0]).sum())
            pr["espalda_ganada"] += int((onside[sel, i] & (margin[sel, i] < 0)).sum())
            out["margins"][role] += np.nan_to_num(margin[sel, i][::5], nan=0).round(1).tolist()
            # duración media de una misma marca (s)
            r = refs[sel, i]
            ch = int((np.diff(r) != 0).sum())
            out["stability"][role].append(sel.sum() / (ch + 1) * STRIDE / 60)
        if team_xtx[0, 0] > 0:
            out["follow_team"].append(round(float(beta(team_xtx, team_xty)[0]), 3))
        # marcas perdidas: cruce del margen con peligro
        for i in range(11):
            m = margin[:, i]
            for f in range(1, F):
                if not (onside[f, i] and m[f] < LOST_MARGIN <= m[f - 1] and fr[f] - fr[f - 1] <= BLOCK_GAP):
                    continue
                if not (danger[f, i] >= DANGER_MIN):
                    continue
                t0 = int(fr[f])
                k = refs[f, i]
                a_slot = int(asl[k])
                xd = q[f, dsl[i], 0]
                res = "nada"
                if any(t0 < g <= t0 + OUTCOME and tm == A for g, tm in goals):
                    res = "gol"
                elif any(t0 < s <= t0 + OUTCOME and tm == A for s, tm in shots):
                    res = "remate"
                elif any(t0 < t <= t0 + OUTCOME and s == a_slot and x * att < xd for t, s, x, y in recs):
                    res = "recibe_a_la_espalda"
                p = int(pid[t0, dsl[i]])
                out["lost"].append(dict(t=t0, defensor=int(dsl[i]), atacante=a_slot,
                                        rol=role_of.get((D, p), "?"), margen=round(float(m[f]), 1),
                                        peligro=round(float(danger[f, i]), 3),
                                        linea=bool(line_by[f] != dsl[i]), resultado=res))
        # referencia al azar: muestras de la misma posesión rival
        for f in rng.choice(F, size=min(F, 40), replace=False):
            t0 = int(fr[f])
            res = "gol" if any(t0 < g <= t0 + OUTCOME and tm == A for g, tm in goals) else \
                "remate" if any(t0 < s <= t0 + OUTCOME and tm == A for s, tm in shots) else "nada"
            out["random"][res] += 1
        out["frames"][D] += F
    (D2 / f"{name}_marcas.npz").parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(D2 / f"{name}_marcas.npz", **arrays)
    out["by_role"] = {k: dict(v) for k, v in out["by_role"].items()}
    out["margins"] = dict(out["margins"])
    out["stability"] = dict(out["stability"])
    out["frames"] = dict(out["frames"])
    out["random"] = dict(out["random"])
    out["follow"] = {k: {kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv) for kk, vv in v.items()}
                     for k, v in out["follow"].items()}
    out["follow_player"] = dict(out["follow_player"])
    out["name"] = name
    return out


def batch(jobs, limit=None):
    man = json.loads((D0 / "manifest.json").read_text(encoding="utf-8"))["recordings"]
    names = [n for n in apt_names() if (D2 / f"{n}.json").exists()][:limit]
    train = [n for n in names if man[n + ".hbr2"]["split"] == "train"]
    w, n_shots, rate = fit_xg(train)
    ctl = json.loads((REPORTS / "d3_valor.json").read_text(encoding="utf-8"))["control"] \
        if (REPORTS / "d3_valor.json").exists() else dict(reaccion=15.0, sigma=10.0)
    reaction, sigma = float(ctl["reaccion"]), float(ctl["sigma"])
    res = []
    with ProcessPoolExecutor(jobs) as ex:
        for k, r in enumerate(ex.map(match, [(n, w, reaction, sigma) for n in names], chunksize=2), 1):
            res.append(r)
            if k % 50 == 0:
                print(f"{k}/{len(names)}", flush=True)
    s = summarize(res, dict(w=w.round(4).tolist(), remates=n_shots, tasa_gol=round(rate, 4)), reaction, sigma)
    (REPORTS / "d2_marcas.json").write_text(json.dumps(s, indent=1, ensure_ascii=False), encoding="utf-8")
    (REPORTS / "d2_marcas.md").write_text(report(s), encoding="utf-8")
    print(json.dumps({k: v for k, v in s.items() if k in ("gamma", "marca_perdida", "azar")}, ensure_ascii=False))
    return s


def summarize(res, xg_info, reaction, sigma):
    gam = np.array([g for r in res for g in r["gamma"]])
    roles = collections.defaultdict(collections.Counter)
    margins = collections.defaultdict(list)
    stab = collections.defaultdict(list)
    lost = collections.defaultdict(collections.Counter)
    lost_line = collections.Counter()
    rnd = collections.Counter()
    frames = 0
    for r in res:
        for k, v in r["by_role"].items():
            roles[k].update(v)
        for k, v in r["margins"].items():
            margins[k] += v
        for k, v in r["stability"].items():
            stab[k] += v
        for e in r["lost"]:
            lost[e["rol"].split("-")[0]][e["resultado"]] += 1
            lost["todos"][e["resultado"]] += 1
            lost_line["la_linea_habilita" if e["linea"] else "el_mismo_habilita"] += 1
        rnd.update(r["random"])
        frames += sum(r["frames"].values())
    by_role = {}
    for k, v in roles.items():
        n = v["muestras"]
        mins = n * STRIDE / 3600
        lr = lost.get(k, collections.Counter())
        by_role[k] = dict(minutos_defendiendo=round(mins, 1), del_lado_del_arco=round(v["del_lado_del_arco"] / n, 3),
                          espalda_ganada=round(v["espalda_ganada"] / n, 4),
                          marcas_perdidas_10min=round(sum(lr.values()) / mins * 10, 3) if mins else None,
                          margen_p10_p50=[round(float(np.percentile(margins[k], q)), 1) for q in (10, 50)],
                          misma_marca_s=round(float(np.median(stab[k])), 1))
    fol = collections.defaultdict(lambda: dict(xtx=np.zeros((2, 2)), xty=np.zeros(2), n=0, movida=0, sigue=0,
                                               entrega=0))
    fol_p = collections.defaultdict(list)
    for r in res:
        for k, v in r["follow"].items():
            for kk, vv in v.items():
                fol[k][kk] = fol[k][kk] + np.asarray(vv)
        for k, v in r["follow_player"].items():
            fol_p[k] += v
    seguimiento = {}
    for k, v in fol.items():
        if v["n"] < 30:
            continue
        ba, bb = beta(v["xtx"], v["xty"])
        mv = int(v["movida"])
        bp = fol_p.get(k, [])
        seguimiento[k] = dict(ventanas=int(v["n"]), beta_atacante=round(float(ba), 3), beta_pelota=round(float(bb), 3),
                              beta_jugador_p10_p50_p90=[round(float(np.percentile(bp, q)), 3) for q in (10, 50, 90)]
                              if len(bp) >= 5 else None, jugadores=len(bp), referencia_movida=mv,
                              sigue=round(int(v["sigue"]) / mv, 3) if mv else None,
                              entrega=round(int(v["entrega"]) / mv, 3) if mv else None,
                              libre=round((mv - int(v["sigue"]) - int(v["entrega"])) / mv, 3) if mv else None)
    fol_t = [b for r in res for b in r["follow_team"]]
    tot = lost["todos"]
    n_l = sum(tot.values())
    n_r = sum(rnd.values())
    return dict(partidos=len(res), muestras=frames, reaccion=reaction, sigma=sigma, xg=xg_info,
                seguimiento=seguimiento, seguimiento_equipo=dict(
                    n=len(fol_t), p10_p50_p90=[round(float(np.percentile(fol_t, q)), 3) for q in (10, 50, 90)]
                    if fol_t else None),
                gamma=dict(p10=np.percentile(gam, 10, axis=0).round(3).tolist(),
                           p50=np.percentile(gam, 50, axis=0).round(3).tolist(),
                           p90=np.percentile(gam, 90, axis=0).round(3).tolist()),
                por_linea=by_role,
                marca_perdida=dict(n=n_l, **{k: round(v / n_l, 4) for k, v in tot.items()}, **dict(lost_line)),
                marca_perdida_por_linea={k: dict(v) for k, v in lost.items() if k != "todos"},
                azar=dict(n=n_r, **{k: round(v / n_r, 4) for k, v in rnd.items()}))


def report(s):
    g = s["gamma"]
    mp, az = s["marca_perdida"], s["azar"]
    shot_l = mp.get("remate", 0) + mp.get("gol", 0)
    shot_r = az.get("remate", 0) + az.get("gol", 0)
    lines = ["# Referencias y marcas en HaxEleven", "",
             f"Pedido del usuario (2026-10-08), en D2 y D3 de [`docs/PLAN_HAXELEVEN.md`](../../docs/PLAN_HAXELEVEN.md), "
             f"con el método de [`tools/hx11_marcas.py`](../../tools/hx11_marcas.py). Son {s['partidos']} partidos y "
             f"{s['muestras']} muestras con la pelota del rival. Los datos están en "
             f"[`d2_marcas.json`](d2_marcas.json).", "",
             "## Cómo marcan los humanos", "",
             f"- **Posición del que marca** = γo·atacante + γb·pelota + γh·arco propio. Por equipo y partido, p50: "
             f"γo {g['p50'][0]}, γb {g['p50'][1]}, γh {g['p50'][2]} (p10–p90 de γo: {g['p10'][0]}–{g['p90'][0]}). "
             f"En básquet, Franks et al. encontraron cerca de 0,6 / 0,1 / 0,3.",
             f"- **Modelo de carrera:** reacción {s['reaccion']:.0f} ticks y vmax {VMAX:.2f} px/tick, del ajuste de D3.",
             "", "| Línea del que marca | Minutos defendiendo | Del lado del arco | Espalda ganada (habilitado) | "
                 "Marcas perdidas cada 10 min | Margen p10 / p50 (ticks) | Misma marca (s, p50) |",
             "|---|---|---|---|---|---|---|"]
    for k, v in sorted(s["por_linea"].items()):
        lines.append(f"| {k} | {v['minutos_defendiendo']} | {100 * v['del_lado_del_arco']:.1f}% | "
                     f"{100 * v['espalda_ganada']:.2f}% | {v['marcas_perdidas_10min']} | {v['margen_p10_p50'][0]} / "
                     f"{v['margen_p10_p50'][1]} | {v['misma_marca_s']} |")
    se = s["seguimiento_equipo"]
    lines += ["", "## ¿Marca personal o zonal?", "",
              f"El HMM elige casi siempre al atacante más cercano (γo alto), así que no alcanza para esto. Se fija la "
              f"referencia al comienzo de ventanas de {FOLLOW // 60} s y se mide, respecto del resto del bloque, cuánto "
              f"la acompaña el defensor: dd = βa·da + βb·db. βa ≈ 1 es personal y βa ≈ 0 es zonal. Las tres columnas "
              f"de la derecha cuentan, cuando la referencia se movió al menos {MOVED:.0f} px respecto del bloque, quién "
              f"es al final el defensor más cercano a ella: el mismo, otro, o ninguno a {FREE:.0f} px (libre).", ""]
    if se["p10_p50_p90"]:
        lines += [f"- **Por equipo y partido** ({se['n']}): βa p10 / p50 / p90 = "
                  f"{' / '.join(str(x) for x in se['p10_p50_p90'])}.", ""]
    lines += ["| Línea | Ventanas | βa (todos) | βa por jugador p10 / p50 / p90 | βb (pelota) | Referencia movida | "
              "Sigue con ella | La toma otro | Libre |", "|---|---|---|---|---|---|---|---|---|"]
    for k, v in sorted(s["seguimiento"].items()):
        bp = " / ".join(str(x) for x in v["beta_jugador_p10_p50_p90"]) if v["beta_jugador_p10_p50_p90"] else "—"
        pct = [f"{100 * v[c]:.1f}%" if v[c] is not None else "—" for c in ("sigue", "entrega", "libre")]
        lines.append(f"| {k} | {v['ventanas']} | {v['beta_atacante']} | {bp} | {v['beta_pelota']} | "
                     f"{v['referencia_movida']} | {' | '.join(pct)} |")
    lines += ["", "## ¿La marca perdida anticipa peligro?", "",
              f"- **Marcas perdidas:** {mp['n']}. En los 5 s siguientes: gol {100 * mp.get('gol', 0):.1f}%, remate "
              f"(sin gol) {100 * mp.get('remate', 0):.1f}%, el atacante recibe a la espalda "
              f"{100 * mp.get('recibe_a_la_espalda', 0):.1f}%.",
              f"- **Muestras al azar** de la posesión rival ({az['n']}): gol {100 * az.get('gol', 0):.1f}%, remate "
              f"{100 * az.get('remate', 0):.1f}%.",
              f"- **Remate o gol:** {100 * shot_l:.1f}% contra {100 * shot_r:.1f}% al azar "
              f"(× {shot_l / shot_r:.1f})." if shot_r else "",
              f"- **Quién habilita:** la línea (otro defensor más atrasado) {mp.get('la_linea_habilita', 0)}, el mismo "
              f"que marca {mp.get('el_mismo_habilita', 0)}.",
              f"- **xG para el peligro:** logística de distancia y ángulo con {s['xg']['remates']} remates de train "
              f"(tasa de gol {s['xg']['tasa_gol']})."]
    return "\n".join(x for x in lines if x is not None) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("lote")
    b.add_argument("--jobs", type=int, default=3)
    b.add_argument("--limit", type=int, default=None)
    u = sub.add_parser("uno")
    u.add_argument("name")
    a = ap.parse_args()
    if a.cmd == "lote":
        batch(a.jobs, a.limit)
    else:
        r = match((a.name, np.array([-1.0, -3.0, -1.0, 0.0]), 15.0, 10.0))
        print(json.dumps({k: v for k, v in r.items() if k not in ("margins", "lost", "stability", "follow")},
                         ensure_ascii=False))
        print({k: (beta(v["xtx"], v["xty"]).round(3).tolist(), v["n"], v["movida"], v["sigue"], v["entrega"])
               for k, v in r["follow"].items()})
        print(len(r["lost"]), collections.Counter(e["resultado"] for e in r["lost"]), r["lost"][:3])


if __name__ == "__main__":
    main()
