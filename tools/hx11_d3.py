"""D3 de HaxEleven (docs/PLAN_HAXELEVEN.md): valor de las acciones y de los estados, sobre D0, D1 y D2.

Tres piezas, entrenadas con la partición de S0 (train para ajustar, dev para elegir, test y test_grupos para medir):

1. **Control del espacio** (pitch control de Spearman 2018, MIT Sloan, en la versión simplificada de tiempo de
   intercepción). Cada jugador llega a un punto en `reaccion + distancia / vmax` desde donde estaría tras la reacción
   siguiendo su velocidad. La pelota tarda lo que le lleva con el frenado de 0,99 por tick y una patada típica. El control
   se integra en el tiempo con una logística de ancho `sigma` y tasa `lam`. vmax = 2,86 px/tick sale de la física del
   mapa (aceleración 0,119, frenado 0,96). La reacción y sigma se ajustan por verosimilitud de los pases de D1: P(el
   equipo del que patea controla el punto donde termina el pase) contra si se completó.
2. **VAEP** (Decroos et al. 2019, KDD, 1802.07127). Para cada acción con pelota de D1 se estima P(el equipo que la hace
   marca en 10 s) y P(recibe un gol en 10 s), a partir de la acción y las dos anteriores (tipo, resultado, lugares
   en coordenadas del que actúa, distancia y ángulo al arco, patada, powershot, saque, tiempo entre acciones). Valor =
   ΔP(marcar) − ΔP(recibir). Los totales por jugador se agregan por puesto con los roles de D2.
3. **Valor de estado** (versión liviana del EPV de Fernández, Bornn y Cervone 2021, 2011.09426). Con lo que también
   hay en el simulador (pelota y velocidad en coordenadas del equipo con la pelota, distancia del poseedor y del rival
   más cercano, cuántos rivales y compañeros hay detrás de la pelota), P(gol a favor en 10 s) y P(en contra en 10 s)
   por tick muestreado. Φ = P(a favor) − P(en contra) es el potencial para el shaping de L2 (Ng 1999).

Los modelos son MLP de torch (2 capas de 64), con pérdida logística y parada temprana en dev.

  python -m tools.hx11_d3 todo [--jobs 3]    # → data/hx11/d3 y reports/hx11/d3_valor.{md,json}
"""
from __future__ import annotations

import argparse
import collections
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from numba import njit

from env.hx11 import contract as HC
from tools.hx11_d0 import OPEN, load_d0

ROOT = Path(__file__).resolve().parent.parent
D0 = ROOT / "data" / "hx11" / "d0"
D1 = ROOT / "data" / "hx11" / "d1"
D2 = ROOT / "data" / "hx11" / "d2"
OUT = ROOT / "data" / "hx11" / "d3"
REPORTS = ROOT / "reports" / "hx11"

ATT = (1.0, -1.0)
VMAX = 0.119 * 0.96 / (1 - 0.96)       # px/tick, velocidad terminal del jugador
BALL_DAMP = 0.99
BALL_V0 = 10.0                          # px/tick: patada típica (p50 8,7, p75 9,7)
LAM = 4.3 / 60                          # tasa de control, 1/tick (Spearman: 4,3/s)
HORIZON = 600                           # 10 s
ACTIONS = ("pase", "centro", "remate", "despeje", "conduccion", "perdida", "intercepcion", "robo", "recuperacion",
           "bloqueo", "atajada", "falta")
SUCCESS = {"pase": ("completo",), "centro": ("completo",), "despeje": ("completo",), "remate": ("gol",)}
STATE_STRIDE = 15


# ----------------------------------------------------------------------------- 1. control del espacio
@njit(cache=True)
def ball_time(d):
    r = (1 - BALL_DAMP) * d / BALL_V0
    if r < 0.9:
        return math.log(1 - r) / math.log(BALL_DAMP)
    return math.log(0.1) / math.log(BALL_DAMP) + (d - 0.9 * BALL_V0 / (1 - BALL_DAMP)) / (0.1 * BALL_V0)


@njit(cache=True)
def control(pos, vel, team, on, ball, target, my_team, reaction, sigma):
    """P(el equipo my_team controla `target` primero): integración de Spearman con tiempos de intercepción."""
    n = pos.shape[0]
    tb = ball_time(math.hypot(target[0] - ball[0], target[1] - ball[1]))
    tti = np.empty(n)
    for i in range(n):
        if not on[i]:
            tti[i] = 1e9
            continue
        rx = pos[i, 0] + vel[i, 0] * reaction
        ry = pos[i, 1] + vel[i, 1] * reaction
        tti[i] = reaction + math.hypot(target[0] - rx, target[1] - ry) / VMAX
    k = math.pi / math.sqrt(3.0) / sigma
    p = np.zeros(n)
    t = tb
    dt = 2.0
    tot = 0.0
    for _ in range(400):
        rem = 1.0 - tot
        if rem < 0.01:
            break
        add = 0.0
        for i in range(n):
            if tti[i] > t + 6 * sigma + 600:
                continue
            q = 1.0 / (1.0 + math.exp(-k * (t - tti[i])))
            dp = rem * q * LAM * dt
            p[i] += dp
            add += dp
        tot += add
        t += dt
    mine = 0.0
    for i in range(n):
        if team[i] == my_team:
            mine += p[i]
    return mine / tot if tot > 0 else 0.5


def pass_rows(name):
    """Pases con patada de juego abierto: (estado al patear, destino, equipo, completo, split)."""
    o, _ = load_d0(D0 / name)
    ev = json.loads((D1 / f"{name}.json").read_text(encoding="utf-8"))["events"]
    pos = o["pos"].astype(np.float64) / 10
    vel = o["vel"].astype(np.float64) / 1000
    ball = o["ball_pos"].astype(np.float64) / 10
    on = o["slot_pid"] >= 0
    rows = []
    for e in ev:
        if e["tipo"] != "pase" or not e.get("patada") or e.get("saque") or e["resultado"] not in ("completo", "rival"):
            continue
        t = e["t"]
        rows.append((pos[t], vel[t], on[t], ball[t], np.array([e["x"] + e["dx"], e["y"] + e["dy"]], np.float64),
                     e["equipo"], e["resultado"] == "completo"))
    return rows


def fit_control(train_rows, dev_rows):
    team = np.r_[np.zeros(11), np.ones(11)].astype(np.int64)

    def ll(rows, reaction, sigma):
        s, ps = 0.0, []
        for pos, vel, on, ball, tgt, tm, ok in rows:
            p = min(max(control(pos, vel, team, on, ball, tgt, tm, reaction, sigma), 1e-4), 1 - 1e-4)
            s += math.log(p if ok else 1 - p)
            ps.append(p)
        return s / max(1, len(rows)), np.array(ps)

    grid = [(r, s) for r in (5.0, 10.0, 15.0, 20.0, 30.0, 45.0) for s in (4.0, 7.0, 10.0, 15.0, 25.0)]
    res = {f"{r:.0f}/{s:.0f}": ll(train_rows, r, s)[0] for r, s in grid}
    best = max(grid, key=lambda g: res[f"{g[0]:.0f}/{g[1]:.0f}"])
    dev_ll, p = ll(dev_rows, *best)
    y = np.array([r[-1] for r in dev_rows], np.float64)
    base = y.mean()
    calib = []
    for lo in np.arange(0, 1.0, 0.1):
        m = (p >= lo) & (p < lo + 0.1)
        if m.sum():
            calib.append(dict(desde=round(float(lo), 1), n=int(m.sum()), p_media=round(float(p[m].mean()), 3),
                              completados=round(float(y[m].mean()), 3)))
    return dict(reaccion=best[0], sigma=best[1], ll_train=res, ll_dev=round(dev_ll, 4),
                ll_dev_base=round(float(base * math.log(base) + (1 - base) * math.log(1 - base)), 4),
                auc_dev=round(auc(y, p), 3), calibracion_dev=calib, n_train=len(train_rows), n_dev=len(dev_rows))


# ----------------------------------------------------------------------------- 2. VAEP
def action_table(name):
    """Acciones con pelota de un partido, en coordenadas del que actúa, con etiquetas de gol a favor y en contra."""
    d = json.loads((D1 / f"{name}.json").read_text(encoding="utf-8"))
    ev = d["events"]
    goals = [(e["t"], e["equipo"]) for e in ev if e["tipo"] == "gol"]
    acts = [e for e in ev if e["tipo"] in ACTIONS]
    rows = []
    for e in acts:
        att = ATT[e["equipo"]]
        x0, y0 = e["x"] * att, e["y"] * att
        x1, y1 = (e["x"] + e.get("dx", 0)) * att, (e["y"] + e.get("dy", 0)) * att
        k = e["tipo"]
        ok = e.get("resultado") in SUCCESS.get(k, ()) if k in SUCCESS else k not in ("perdida", "falta")
        rows.append(dict(t=e["t"], team=e["equipo"], slot=e["lugar"], tipo=k, ok=ok, x0=x0, y0=y0, x1=x1, y1=y1,
                         patada=bool(e.get("patada")), ps=(e.get("powershot") or 0) / 100,
                         saque=bool(e.get("saque"))))
    for r in rows:
        r["gf"] = any(r["t"] < g <= r["t"] + HORIZON and tm == r["team"] for g, tm in goals)
        r["gc"] = any(r["t"] < g <= r["t"] + HORIZON and tm != r["team"] for g, tm in goals)
    return rows, goals


def _geo(x, y):
    dx = HC.LINE_W - x
    return [x / HC.LINE_W, y / HC.LINE_H, math.hypot(dx, y) / 3600, abs(math.atan2(y, max(dx, 1.0)))]


def action_features(rows, goals):
    X = []
    for i, r in enumerate(rows):
        f = []
        for lag in range(3):
            j = i - lag
            if j < 0:
                f += [0.0] * (len(ACTIONS) + 14)
                continue
            p = rows[j]
            same = 1.0 if p["team"] == r["team"] else -1.0
            s = 1.0 if p["team"] == r["team"] else -1.0   # las anteriores se ven desde el equipo del que actúa
            oh = [1.0 if p["tipo"] == a else 0.0 for a in ACTIONS]
            f += oh + [float(p["ok"])] + _geo(p["x0"] * s, p["y0"] * s) + _geo(p["x1"] * s, p["y1"] * s) + [
                float(p["patada"]), p["ps"], float(p["saque"]), same,
                min(1.0, (r["t"] - p["t"]) / 600)]
        X.append(f)
    return np.array(X, np.float32)


def vaep_values(rows, ps, pc, goals):
    v = np.zeros(len(rows))
    for i, r in enumerate(rows):
        if i == 0 or any(rows[i - 1]["t"] < g <= r["t"] for g, _ in goals):
            prev_s, prev_c = 0.0, 0.0
        elif rows[i - 1]["team"] == r["team"]:
            prev_s, prev_c = ps[i - 1], pc[i - 1]
        else:
            prev_s, prev_c = pc[i - 1], ps[i - 1]
        v[i] = (ps[i] - prev_s) - (pc[i] - prev_c)
    return v


# ----------------------------------------------------------------------------- 3. valor de estado
def state_rows(name):
    """Ticks muestreados con dueño: rasgos que también tiene el simulador y etiquetas de gol en 10 s."""
    o, _ = load_d0(D0 / name)
    owner = np.load(D1 / f"{name}.npz")["owner"]
    ev = json.loads((D1 / f"{name}.json").read_text(encoding="utf-8"))["events"]
    goals = np.array([(e["t"], e["equipo"]) for e in ev if e["tipo"] == "gol"], np.int64).reshape(-1, 2)
    T = len(owner)
    ticks = np.nonzero((o["phase"] == OPEN) & (owner >= 0))[0]
    ticks = ticks[ticks % STATE_STRIDE == 0]
    pos = o["pos"][ticks].astype(np.float32) / 10
    on = o["slot_pid"][ticks] >= 0
    ball = o["ball_pos"][ticks].astype(np.float32) / 10
    bv = o["ball_vel"][ticks].astype(np.float32) / 1000
    own = owner[ticks].astype(np.int64)
    tm = own // 11
    att = np.where(tm == 0, 1.0, -1.0).astype(np.float32)
    b = ball * att[:, None]
    v = bv * att[:, None]
    p = pos * att[:, None, None]
    d = np.hypot(*(p - b[:, None]).transpose(2, 0, 1))
    d[~on] = np.inf
    team = np.r_[np.zeros(11), np.ones(11)].astype(np.int64)
    mine = team[None, :] == tm[:, None]
    d_own = d[np.arange(len(ticks)), own]
    d_riv = np.where(mine, np.inf, d).min(1)
    behind_riv = (~mine & on & (p[..., 0] > b[:, None, 0])).sum(1)    # rivales entre la pelota y su arco
    ahead_mine = (mine & on & (p[..., 0] > b[:, None, 0])).sum(1)     # compañeros por delante de la pelota
    X = np.c_[b[:, 0] / HC.LINE_W, b[:, 1] / HC.LINE_H, v / 10,
              np.hypot(HC.LINE_W - b[:, 0], b[:, 1]) / 3600, np.abs(np.arctan2(b[:, 1], np.maximum(HC.LINE_W - b[:, 0], 1))),
              np.minimum(d_own, 500) / 500, np.minimum(d_riv, 500) / 500, behind_riv / 11, ahead_mine / 11].astype(np.float32)
    gf = np.zeros(len(ticks), bool)
    gc = np.zeros(len(ticks), bool)
    for g, gt in goals:
        w = (ticks < g) & (ticks >= g - HORIZON)
        gf |= w & (tm == gt)
        gc |= w & (tm != gt)
    return X, gf, gc


# ----------------------------------------------------------------------------- modelos
def auc(y, p):
    y = np.asarray(y, bool)
    if y.all() or not y.any():
        return float("nan")
    r = np.argsort(np.argsort(p)) + 1
    n1 = y.sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1)))


def train_mlp(Xtr, ytr, Xdev, ydev, epochs=40, seed=0):
    import torch
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    net = torch.nn.Sequential(torch.nn.Linear(Xtr.shape[1], 64), torch.nn.ReLU(), torch.nn.Linear(64, 64),
                              torch.nn.ReLU(), torch.nn.Linear(64, 1))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    xt = torch.tensor((Xtr - mu) / sd)
    yt = torch.tensor(ytr, dtype=torch.float32)
    xd = torch.tensor((Xdev - mu) / sd)
    yd = torch.tensor(ydev, dtype=torch.float32)
    lossf = torch.nn.BCEWithLogitsLoss()
    best, state, bad = 1e9, None, 0
    for _ in range(epochs):
        perm = torch.randperm(len(xt))
        net.train()
        for k in range(0, len(xt), 2048):
            i = perm[k:k + 2048]
            opt.zero_grad()
            loss = lossf(net(xt[i]).squeeze(1), yt[i])
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            dl = float(lossf(net(xd).squeeze(1), yd))
        if dl < best - 1e-5:
            best, state, bad = dl, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 5:
                break
    net.load_state_dict(state)

    def predict(X):
        with torch.no_grad():
            return torch.sigmoid(net(torch.tensor((X - mu) / sd)).squeeze(1)).numpy()
    return predict, dict(net=net, mu=mu, sd=sd)


def reliability(y, p, bins=10):
    """Curva de confiabilidad por cuantiles de p (las P son chicas: bins de ancho fijo quedarían vacíos) y ECE."""
    y = np.asarray(y, np.float64)
    p = np.asarray(p, np.float64)
    q = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    idx = np.clip(np.searchsorted(q, p, side="right") - 1, 0, len(q) - 2)
    rows, ece = [], 0.0
    for b in range(len(q) - 1):
        m = idx == b
        if m.sum():
            rows.append(dict(p_media=round(float(p[m].mean()), 4), observada=round(float(y[m].mean()), 4),
                             n=int(m.sum())))
            ece += m.sum() / len(p) * abs(p[m].mean() - y[m].mean())
    return dict(ece=round(float(ece), 5), bins=rows)


def fit_logit(X, y, l2=1.0, iters=25):
    """Logística por Newton (modelo B: otra familia que el MLP)."""
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = np.c_[np.ones(len(X)), (X - mu) / sd].astype(np.float64)
    w = np.zeros(Z.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Z @ w))
        H = Z.T @ (Z * (p * (1 - p))[:, None]) + l2 * np.eye(len(w))
        step = np.linalg.solve(H, Z.T @ (y - p) - l2 * w)
        w += step
        if np.abs(step).max() < 1e-6:
            break
    return lambda X_: 1 / (1 + np.exp(-np.c_[np.ones(len(X_)), (X_ - mu) / sd] @ w))


def action_features_b(rows):
    """Rasgos del modelo B de VAEP: sólo la acción actual (tipo, resultado, origen y destino), sin las anteriores."""
    return np.array([[1.0 if r["tipo"] == a else 0.0 for a in ACTIONS] + [float(r["ok"])] + _geo(r["x0"], r["y0"])
                     + _geo(r["x1"], r["y1"]) + [float(r["patada"]), r["ps"], float(r["saque"])] for r in rows],
                    np.float32)


class GridValue:
    """Modelo B del valor de estado: P(gol en 10 s) por zona de la pelota (12 × 8, coordenadas del equipo con la
    pelota), con previa beta centrada en la tasa base. Otra familia y otros rasgos que el MLP."""
    NX, NY, K = 12, 8, 50.0

    def __init__(self, X, y):
        self.base = float(y.mean())
        i = self._cell(X)
        n = np.bincount(i, minlength=self.NX * self.NY)
        s = np.bincount(i, weights=y.astype(np.float64), minlength=self.NX * self.NY)
        self.p = (s + self.K * self.base) / (n + self.K)

    def _cell(self, X):
        cx = np.clip(((X[:, 0] + 1) / 2 * self.NX).astype(np.int64), 0, self.NX - 1)
        cy = np.clip(((X[:, 1] + 1) / 2 * self.NY).astype(np.int64), 0, self.NY - 1)
        return cx * self.NY + cy

    def __call__(self, X):
        return self.p[self._cell(X)]


def spearman(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if len(a) < 3:
        return float("nan")
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def split_half(per_player, min_matches=6, splits=200, seed=0):
    """Fiabilidad entre partidos de un valor por jugador: se parten sus partidos en dos mitades al azar y se
    correlacionan (Pearson) las medias de las mitades entre jugadores. Mediana sobre particiones, IC 90% por bootstrap
    de jugadores, y Spearman-Brown para el promedio de todos sus partidos."""
    rng = np.random.default_rng(seed)
    vals = [np.array(v) for v in per_player.values() if len(v) >= min_matches]
    if len(vals) < 10:
        return dict(jugadores=len(vals), r=None)
    rs = np.empty((splits, 2, len(vals)))
    for s in range(splits):
        for j, v in enumerate(vals):
            perm = rng.permutation(len(v))
            h = len(v) // 2
            rs[s, 0, j], rs[s, 1, j] = v[perm[:h]].mean(), v[perm[h:2 * h]].mean()
    r_split = np.array([np.corrcoef(rs[s, 0], rs[s, 1])[0, 1] for s in range(splits)])
    boot = []
    for _ in range(200):
        idx = rng.integers(0, len(vals), len(vals))
        boot.append(np.median([np.corrcoef(rs[s, 0, idx], rs[s, 1, idx])[0, 1] for s in range(0, splits, 10)]))
    r = float(np.median(r_split))
    return dict(jugadores=len(vals), r=round(r, 3), ic90=[round(float(np.percentile(boot, q)), 3) for q in (5, 95)],
                spearman_brown=round(2 * r / (1 + r), 3) if r > -1 else None)


def metrics(y, p):
    y = np.asarray(y, np.float64)
    p = np.clip(p, 1e-6, 1 - 1e-6)
    base = y.mean()
    ll = float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
    llb = float(-(base * math.log(base) + (1 - base) * math.log(1 - base))) if 0 < base < 1 else float("nan")
    return dict(n=int(len(y)), tasa=round(float(base), 4), logloss=round(ll, 4), logloss_base=round(llb, 4),
                brier=round(float(((p - y) ** 2).mean()), 5), auc=round(auc(y, p), 3))


def splits():
    man = json.loads((D0 / "manifest.json").read_text(encoding="utf-8"))["recordings"]
    out = collections.defaultdict(list)
    for n, r in man.items():
        b = n.removesuffix(".hbr2")
        if r.get("apto", True) and (D1 / f"{b}.json").exists():
            out["test" if r["split"].startswith("test") else r["split"]].append(b)
    return {k: sorted(v) for k, v in out.items()}


def _pc_job(name):
    return name, pass_rows(name)


def run_all(jobs):
    import torch
    sp = splits()
    OUT.mkdir(parents=True, exist_ok=True)
    rep = dict(partidos={k: len(v) for k, v in sp.items()})
    # 1. control del espacio: pases de una parte de los partidos (alcanza para 2 parámetros)
    rng = np.random.default_rng(0)
    pick_tr = list(rng.permutation(sp["train"])[:40])
    pick_dev = list(sp["dev"][:20])
    with ProcessPoolExecutor(jobs) as ex:
        got = dict(ex.map(_pc_job, pick_tr + pick_dev))
    tr = [r for n in pick_tr for r in got[n]]
    dv = [r for n in pick_dev for r in got[n]]
    tr = [tr[i] for i in rng.permutation(len(tr))[:6000]]
    rep["control"] = fit_control(tr, dv)
    print("control", json.dumps({k: v for k, v in rep["control"].items() if k != "ll_train"}), flush=True)
    # 2. VAEP
    tabs = {}
    for k, names in sp.items():
        for n in names:
            tabs[n] = action_table(n)
    data = {}
    for k, names in sp.items():
        X, gf, gc = [], [], []
        for n in names:
            rows, goals = tabs[n]
            if not rows:
                continue
            X.append(action_features(rows, goals))
            gf.append([r["gf"] for r in rows])
            gc.append([r["gc"] for r in rows])
        data[k] = (np.concatenate(X), np.concatenate(gf).astype(np.float32), np.concatenate(gc).astype(np.float32))
    ps_f, ps_m = train_mlp(data["train"][0], data["train"][1], data["dev"][0], data["dev"][1])
    pc_f, pc_m = train_mlp(data["train"][0], data["train"][2], data["dev"][0], data["dev"][2])
    # modelo B (validación, auditoría E3): logística sin las acciones anteriores
    xb = {k: np.concatenate([action_features_b(tabs[n][0]) for n in names if tabs[n][0]]) for k, names in sp.items()}
    psb = fit_logit(xb["train"], data["train"][1])
    pcb = fit_logit(xb["train"], data["train"][2])
    pa_t, pc_t = ps_f(data["test"][0]), pc_f(data["test"][0])
    rep["vaep"] = dict(marcar=metrics(data["test"][1], pa_t), recibir=metrics(data["test"][2], pc_t))
    rep["vaep_b"] = dict(marcar=metrics(data["test"][1], psb(xb["test"])),
                         recibir=metrics(data["test"][2], pcb(xb["test"])))
    rep["vaep_confiabilidad"] = dict(marcar=reliability(data["test"][1], pa_t),
                                     recibir=reliability(data["test"][2], pc_t),
                                     marcar_b=reliability(data["test"][1], psb(xb["test"])))
    print("vaep", json.dumps(rep["vaep"]), "b", json.dumps(rep["vaep_b"]), flush=True)
    torch.save(dict(marcar=ps_m, recibir=pc_m), OUT / "vaep.pt")
    # valores por acción, por jugador y por puesto (D2); fiabilidad entre partidos por jugador (clave anónima)
    from tools.hx11_d1 import names_by_recording
    from tools.x4_index import player_key
    names_of = names_by_recording()
    test_names = set(sp.get("test", []))
    by_type = collections.defaultdict(list)
    by_role = collections.defaultdict(list)
    per_a, per_b, per_pass = (collections.defaultdict(list) for _ in range(3))
    agree_act, agree_player = [], []
    for n, (rows, goals) in tabs.items():
        if not rows:
            continue
        X = action_features(rows, goals)
        v = vaep_values(rows, ps_f(X), pc_f(X), goals)
        Xb = action_features_b(rows)
        vb = vaep_values(rows, psb(Xb), pcb(Xb), goals)
        if n in test_names:
            agree_act.append(np.c_[v, vb])
        for r, x in zip(rows, v):
            by_type[r["tipo"]].append(x)
        d2p = D2 / f"{n}.json"
        if not d2p.exists():
            continue
        d2 = json.loads(d2p.read_text(encoding="utf-8"))
        o, _ = load_d0(D0 / n)
        spid = o["slot_pid"]
        tot, totb = collections.Counter(), collections.Counter()
        passes = collections.defaultdict(lambda: [0, 0])
        for r, x, xb_ in zip(rows, v, vb):
            key = (r["team"], int(spid[r["t"], r["slot"]]))
            tot[key] += x
            totb[key] += xb_
            if r["tipo"] == "pase":
                passes[key][0] += int(r["ok"])
                passes[key][1] += 1
        nm = names_of.get(n, {})
        for t in d2["equipos"]:
            for p, info in t["jugadores"].items():
                mins = info["muestras"] / 600
                if mins < 5:
                    continue
                key = (t["equipo"], int(p))
                va, vbb = tot[key] / mins * 10, totb[key] / mins * 10
                by_role[info["nombre_rol"].split("-")[0]].append(va)
                agree_player.append((va, vbb))
                name = nm.get(str(p))
                if name and info["nombre_rol"] != "ARQ":
                    k = player_key(name)
                    per_a[k].append(va)
                    per_b[k].append(vbb)
                    if passes[key][1] >= 5:
                        per_pass[k].append(passes[key][0] / passes[key][1])
        np.save(OUT / f"{n}_vaep.npy", v.astype(np.float32))
    aa = np.concatenate(agree_act)
    ap_ = np.array(agree_player)
    rep["vaep_acuerdo_a_b"] = dict(por_accion_test=round(spearman(aa[:, 0], aa[:, 1]), 3),
                                   por_jugador_partido=round(spearman(ap_[:, 0], ap_[:, 1]), 3))
    rep["fiabilidad_entre_partidos"] = dict(vaep_a=split_half(per_a), vaep_b=split_half(per_b),
                                            pase_completo=split_half(per_pass))
    print("fiabilidad", json.dumps(rep["fiabilidad_entre_partidos"]), json.dumps(rep["vaep_acuerdo_a_b"]), flush=True)
    rep["vaep_por_tipo"] = {k: dict(n=len(x), media=round(float(np.mean(x)), 4), suma=round(float(np.sum(x)), 1))
                            for k, x in sorted(by_type.items(), key=lambda kv: -np.sum(kv[1]))}
    rep["vaep_por_linea_10min"] = {k: dict(n=len(x), p10=round(float(np.percentile(x, 10)), 3),
                                           p50=round(float(np.percentile(x, 50)), 3),
                                           p90=round(float(np.percentile(x, 90)), 3))
                                   for k, x in sorted(by_role.items())}
    # 3. valor de estado
    sdata = {}
    for k, names in sp.items():
        parts = [state_rows(n) for n in names]
        sdata[k] = tuple(np.concatenate([p[i] for p in parts]) for i in range(3))
    sf_f, sf_m = train_mlp(sdata["train"][0], sdata["train"][1].astype(np.float32), sdata["dev"][0],
                           sdata["dev"][1].astype(np.float32))
    sc_f, sc_m = train_mlp(sdata["train"][0], sdata["train"][2].astype(np.float32), sdata["dev"][0],
                           sdata["dev"][2].astype(np.float32))
    Xt = sdata["test"][0]
    pf, pcn = sf_f(Xt), sc_f(Xt)
    rep["estado"] = dict(a_favor=metrics(sdata["test"][1], pf), en_contra=metrics(sdata["test"][2], pcn))
    # modelo B del estado (juez independiente del shaping): grilla de la pelota
    gf_b = GridValue(sdata["train"][0], sdata["train"][1])
    gc_b = GridValue(sdata["train"][0], sdata["train"][2])
    pfb, pcb_ = gf_b(Xt), gc_b(Xt)
    rep["estado_b"] = dict(a_favor=metrics(sdata["test"][1], pfb), en_contra=metrics(sdata["test"][2], pcb_))
    rep["estado_confiabilidad"] = dict(a_favor=reliability(sdata["test"][1], pf),
                                       en_contra=reliability(sdata["test"][2], pcn),
                                       a_favor_b=reliability(sdata["test"][1], pfb))
    rep["estado_acuerdo_a_b"] = dict(phi_spearman=round(spearman(pf - pcn, pfb - pcb_), 3),
                                     phi_pearson=round(float(np.corrcoef(pf - pcn, pfb - pcb_)[0, 1]), 3))
    # Φ por zona de la pelota (coordenadas del equipo con la pelota), en test
    bx = Xt[:, 0] * HC.LINE_W
    zones = {}
    for lo in range(-1800, 1800, 600):
        m = (bx >= lo) & (bx < lo + 600)
        if m.sum():
            zones[f"{lo}..{lo + 600}"] = dict(n=int(m.sum()), phi=round(float((pf[m] - pcn[m]).mean()), 4),
                                              gol_favor=round(float(sdata["test"][1][m].mean()), 4),
                                              gol_contra=round(float(sdata["test"][2][m].mean()), 4))
    rep["estado_por_zona"] = zones
    torch.save(dict(a_favor=sf_m, en_contra=sc_m), OUT / "estado.pt")
    print("estado", json.dumps(rep["estado"]), flush=True)
    (REPORTS / "d3_valor.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
    (REPORTS / "d3_valor.md").write_text(report(rep), encoding="utf-8")
    return rep


def report(r):
    c = r["control"]
    lines = ["# D3: valor de acciones y estados en HaxEleven", "",
             f"Etapa D3 de [`docs/PLAN_HAXELEVEN.md`](../../docs/PLAN_HAXELEVEN.md), con el método de "
             f"[`tools/hx11_d3.py`](../../tools/hx11_d3.py). Partidos por parte: {r['partidos']}. Los datos están en "
             f"[`d3_valor.json`](d3_valor.json).", "",
             "## 1. Control del espacio", "",
             f"- Ajuste por verosimilitud en {c['n_train']} pases de train: reacción **{c['reaccion']:.0f} ticks**, "
             f"sigma **{c['sigma']:.0f} ticks** (vmax {VMAX:.2f} px/tick de la física, tasa {LAM * 60:.1f}/s).",
             f"- En dev ({c['n_dev']} pases): log-verosimilitud media {c['ll_dev']} contra {c['ll_dev_base']} de la "
             f"tasa base, AUC {c['auc_dev']}.",
             "- La P está medida en el punto donde termina el pase. Para un pase cortado, ese punto es donde lo tocó el "
             "rival, así que la separación es optimista. Sirve para ver la calibración, no como precisión de un pase "
             "planeado.", "",
             "| P(control) | Pases | P media | Completados |", "|---|---|---|---|"]
    lines += [f"| {b['desde']:.1f}–{b['desde'] + 0.1:.1f} | {b['n']} | {b['p_media']} | {b['completados']} |"
              for b in c["calibracion_dev"]]
    v = r["vaep"]
    lines += ["", "## 2. VAEP (test)", "", "| Modelo | Acciones | Tasa | Log-loss | Base | Brier | AUC |",
              "|---|---|---|---|---|---|---|"]
    for k, m in v.items():
        lines.append(f"| {k} en 10 s | {m['n']} | {m['tasa']} | {m['logloss']} | {m['logloss_base']} | {m['brier']} "
                     f"| {m['auc']} |")
    lines += ["", "**Valor por tipo de acción (todas las partes):**", "", "| Acción | n | Media | Suma |",
              "|---|---|---|---|"]
    lines += [f"| {k} | {x['n']} | {x['media']} | {x['suma']} |" for k, x in r["vaep_por_tipo"].items()]
    lines += ["", "**VAEP por jugador cada 10 minutos, por línea del puesto (D2; jugadores con >= 5 min):**", "",
              "| Línea | Jugadores-partido | p10 | p50 | p90 |", "|---|---|---|---|---|"]
    lines += [f"| {k} | {x['n']} | {x['p10']} | {x['p50']} | {x['p90']} |" for k, x in r["vaep_por_linea_10min"].items()]
    e = r["estado"]
    lines += ["", "## 3. Valor de estado (test)", "", "| Modelo | Ticks | Tasa | Log-loss | Base | Brier | AUC |",
              "|---|---|---|---|---|---|---|"]
    for k, m in e.items():
        lines.append(f"| gol {k.replace('_', ' ')} en 10 s | {m['n']} | {m['tasa']} | {m['logloss']} | "
                     f"{m['logloss_base']} | {m['brier']} | {m['auc']} |")
    lines += ["", "**Φ = P(a favor) − P(en contra) según dónde está la pelota (x hacia el ataque del equipo con la "
              "pelota):**", "", "| x | Ticks | Φ medio | Gol a favor | Gol en contra |", "|---|---|---|---|---|"]
    lines += [f"| {k} | {z['n']} | {z['phi']} | {z['gol_favor']} | {z['gol_contra']} |"
              for k, z in r["estado_por_zona"].items()]
    lines += validation_report(r)
    return "\n".join(lines) + "\n"


def validation_report(r):
    """Sección de validación (auditoría del 2026-10-08, E3): calibración, modelo B independiente y fiabilidad."""
    def table(title, d):
        out = ["", f"**{title}** (test):", "", "| Modelo | Log-loss | Base | Brier | AUC |", "|---|---|---|---|---|"]
        out += [f"| {k} | {m['logloss']} | {m['logloss_base']} | {m['brier']} | {m['auc']} |" for k, m in d.items()]
        return out

    def calib(title, c):
        out = ["", f"**Confiabilidad, {title}** (deciles de P; ECE {c['ece']}):", "", "| P media | Observada | n |",
               "|---|---|---|"]
        out += [f"| {b['p_media']} | {b['observada']} | {b['n']} |" for b in c["bins"]]
        return out

    f = r["fiabilidad_entre_partidos"]

    def fr(k, label):
        x = f[k]
        if x.get("r") is None:
            return f"- {label}: pocos jugadores ({x['jugadores']})."
        return (f"- **{label}:** r = {x['r']} entre mitades de sus partidos (IC 90% {x['ic90'][0]} a {x['ic90'][1]}), "
                f"Spearman-Brown {x['spearman_brown']}, {x['jugadores']} jugadores con >= 6 partidos.")

    lines = ["", "## 4. Validación del valor (auditoría E3)", "",
             "Dos modelos de familias y rasgos distintos. **A** (MLP, con contexto) es el candidato a shaping; **B** "
             "(logística sin acciones previas para VAEP; grilla de la pelota para el estado) es el juez independiente. "
             "Ninguno se usa a la vez para entrenar y para juzgar."]
    lines += table("VAEP, modelo B", r["vaep_b"])
    lines += calib("VAEP A, marcar en 10 s", r["vaep_confiabilidad"]["marcar"])
    lines += calib("VAEP A, recibir en 10 s", r["vaep_confiabilidad"]["recibir"])
    lines += calib("VAEP B, marcar en 10 s", r["vaep_confiabilidad"]["marcar_b"])
    lines += table("Valor de estado, modelo B", r["estado_b"])
    lines += calib("estado A, gol a favor en 10 s", r["estado_confiabilidad"]["a_favor"])
    lines += calib("estado A, gol en contra en 10 s", r["estado_confiabilidad"]["en_contra"])
    lines += calib("estado B, gol a favor en 10 s", r["estado_confiabilidad"]["a_favor_b"])
    a, e = r["vaep_acuerdo_a_b"], r["estado_acuerdo_a_b"]
    lines += ["", "**Acuerdo entre A y B:**", "",
              f"- VAEP por acción (test): Spearman {a['por_accion_test']}; por jugador-partido cada 10 min: "
              f"{a['por_jugador_partido']}.",
              f"- Φ por tick (test): Spearman {e['phi_spearman']}, Pearson {e['phi_pearson']}.",
              "", "**Fiabilidad entre partidos del valor por jugador** (clave anónima del nombre; sin arqueros):", "",
              fr("vaep_a", "VAEP A cada 10 min"), fr("vaep_b", "VAEP B cada 10 min"),
              fr("pase_completo", "Pase completo (referencia simple)")]
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("todo")
    t.add_argument("--jobs", type=int, default=3)
    a = ap.parse_args()
    run_all(a.jobs)


if __name__ == "__main__":
    main()
