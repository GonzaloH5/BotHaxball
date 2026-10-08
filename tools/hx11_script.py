"""Arqueología del script de HaxEleven (docs/PLAN_HAXELEVEN.md, S2): las reglas de la sala medidas en las grabaciones.

El script de la sala está ofuscado y no se ejecuta. Sus reglas se infieren de lo que hace en las grabaciones (lo que la
física no explica: `setDiscProperties`) y de sus anuncios. Lee los partidos de `tools.hx11_npz`.

* `saques`: cada colocación de la pelota por el script, su tipo (lateral, córner, saque de arco, tiro libre, penal) por
  el punto, el modo de la pelota mientras espera (color y masa), quién patea y la potencia anunciada.
* `faltas` / `ajuste_faltas`: los cobros de falta y el ajuste del temporizador de contacto (tolerancia, cortes,
  distancia de la pelota al poseedor, umbral) contra todos los cobros.
* `offside`: la regla de offside medida (patada de un compañero, desvíos, exenciones) contra los cobros.
* `powershot`: demora, curva y niveles de la carga, % anunciado y comba de la patada.
* `barra`: la barra de fuerza de la pelota parada por tipo de saque y la comba de esas patadas.

  python -m tools.hx11_script <subcomando> data/hx11/npz/*.npz --out reports/hx11/s2_<subcomando>.json
"""
from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

import numpy as np

from tools.hx11_npz import load

BASE_INV = 1.65             # invMass de la pelota en juego (mapa)
BALL_R, PLAYER_R = 8.0, 15.0
WHITE, BLUE_DARK, ORANGE = 0xFFFFFF, 0x00050A, 0xEBC81C
POWER_ANN = re.compile(r"Potencia del (corner|saque|tiro libre|penal\w*): (\d+)%")
PS_ANN = re.compile(r"Powershot utilizado\. (\d+)%")


class Match:
    def __init__(self, base):
        self.name = Path(str(base).removesuffix(".npz")).name
        self.z, self.events, self.header = load(base)
        z = self.z
        self.frame = z["frame"]
        self.T = len(self.frame)
        self.fidx = {int(f): i for i, f in enumerate(self.frame)}
        self.pcol = {int(p): c for c, p in enumerate(z["pid"])}
        self.kicks = [(self.at(e["frame"]), self.pcol.get(e["playerId"])) for e in self.events if e["name"] == "kick"]
        self.kicks = [(i, c) for i, c in self.kicks if i is not None and c is not None]
        self.kick_at = collections.defaultdict(list)
        for i, c in self.kicks:
            self.kick_at[i].append(c)
        self.ann = [(self.at(e["frame"]), e["msg"]) for e in self.events if e["name"] == "announcement"]
        self.ann = [(i, msg) for i, msg in self.ann if i is not None]
        self.placements = []  # (i, x, y) colocaciones de la pelota por el script
        for e in self.events:
            if e["name"] == "disc_props" and not e["player"] and e["id"] == 0 and e["d1"] and e["d1"][0] is not None:
                i = self.at(e["frame"])
                if i is not None:
                    self.placements.append((i, float(e["d1"][0]), float(e["d1"][1])))

    def at(self, frame):
        """Índice del tick de ese frame, o del último anterior (pausas)."""
        f = int(frame)
        for _ in range(600):
            if f in self.fidx:
                return self.fidx[f]
            f -= 1
        return None

    def edge_dist(self, i):
        """Distancia borde a borde de cada jugador a la pelota en el tick i (NaN si no está en la cancha)."""
        z = self.z
        d = np.linalg.norm(z["pl_pos"][i] - z["ball_pos"][i], axis=1) - PLAYER_R - BALL_R
        d[z["pl_team"][i] == 0] = np.nan
        return d

    def team(self, i, c):
        return int(self.z["pl_team"][i, c])


def piece_kind(x, y):
    ax, ay = abs(x), abs(y)
    if abs(ay - 1230) < 0.5:
        return "lateral"
    if abs(ax - 1765) < 0.5 and abs(ay - 1165) < 0.5:
        return "corner"
    if abs(ax - 1572) < 0.5:
        return "saque_arco"
    if abs(ax - 1275) < 0.5 and ay < 0.5:
        return "penal"
    return "tiro_libre"


def ball_mode(color):
    if np.isnan(color):
        return "?"
    c = int(color)
    if c == BLUE_DARK:
        return "azul"
    if c == ORANGE:
        return "naranja"
    if c == WHITE:
        return "blanca"
    return "otro"


def pieces(m: Match):
    """Una fila por colocación de la pelota: tipo, modo y masa mientras espera, ejecutor y potencia anunciada."""
    z = m.z
    out = []
    pl = m.placements
    for n, (i, x, y) in enumerate(pl):
        nxt = pl[n + 1][0] if n + 1 < len(pl) else m.T
        # primera patada después de la colocación (la del mismo tick es la jugada que la provocó)
        kick = next(((ki, kc) for ki, kc in m.kicks if i < ki <= nxt), None)
        same = m.kick_at.get(i, [])
        end = kick[0] if kick else min(nxt, i + 1200)
        seg = slice(i + 1, max(i + 2, end))
        inv = z["ball_invMass"][seg]
        colors = collections.Counter(ball_mode(c) for c in z["ball_color"][seg])
        ann = [(ai - i, msg) for ai, msg in m.ann if i <= ai <= end + 2]
        power = next((int(g.group(2)) for _, msg in ann for g in [POWER_ANN.search(msg)] if g), None)
        out.append(dict(
            match=m.name, frame=int(m.frame[i]), kind=piece_kind(x, y), x=round(x, 1), y=round(y, 1),
            prev_ball=[round(float(v), 1) for v in z["ball_pos"][max(0, i - 1)]],
            kick_same_tick_team=[m.team(i, c) for c in same],
            kicker_team=m.team(kick[0], kick[1]) if kick else None,
            wait=int(kick[0] - i) if kick else None, replaced_after=int(nxt - i) if nxt < m.T else None,
            mode=colors.most_common(1)[0][0] if colors else None,
            inv_min=float(np.nanmin(inv)) if len(inv) else None, inv_max=float(np.nanmax(inv)) if len(inv) else None,
            inv_at_kick=float(z["ball_invMass"][kick[0] - 1]) if kick else None,
            power=power,
            announcements=[(d, msg[:80]) for d, msg in ann if not re.search(r":\s", msg[:60])][:6]))
    return out


def last_toucher(m: Match, i_end, eps=1.0, lookback=3600):
    """(tick, jugador) del último toque antes de i_end: pelota a <= eps px del borde del jugador, o patada."""
    kicks = {i: cs for i, cs in m.kick_at.items() if i_end - lookback <= i < i_end}
    for j in range(i_end - 1, max(0, i_end - lookback), -1):
        if j in kicks:
            return j, kicks[j][0]
        d = m.edge_dist(j)
        if np.nanmin(d) <= eps:
            return j, int(np.nanargmin(d))
    return None, None


def contact_before(m: Match, i_end, c, eps=1.0, lookback=1200):
    """Para cada rival del jugador c: ticks seguidos de contacto con c (centros a <= 2r + eps) que terminan en i_end - 1,
    y ticks de contacto acumulados en la ventana."""
    z = m.z
    j0 = max(0, i_end - lookback)
    pos = z["pl_pos"][j0:i_end]
    team = z["pl_team"][j0:i_end]
    tc = team[-1, c]
    d = np.linalg.norm(pos - pos[:, c:c + 1, :], axis=2)
    touching = (d <= 2 * PLAYER_R + eps) & (team != 0) & (team != tc)
    out = {}
    for r in np.where(touching.any(axis=0))[0]:
        col = touching[:, r]
        run = 0
        for v in col[::-1]:
            if not v:
                break
            run += 1
        out[int(r)] = dict(run=run, total=int(col.sum()))
    return out


def fouls(m: Match, eps=1.0):
    """Faltas: doble colocación de la pelota (la primera congela la pelota donde está, la segunda ~15 ticks después la
    deja para el tiro libre o el penal) sin patada en el tick de la primera (si hay patada, es offside)."""
    rows = []
    pl = m.placements
    for n, (i, x, y) in enumerate(pl[:-1]):
        i2, x2, y2 = pl[n + 1]
        if not (0 < i2 - i <= 30 and abs(x2 - x) + abs(y2 - y) < 1e-3 or piece_kind(x2, y2) == "penal" and 0 < i2 - i <= 30):
            continue
        if m.kick_at.get(i) or piece_kind(x, y) in ("lateral", "corner", "saque_arco"):
            continue
        tj, toucher = last_toucher(m, i, eps)
        kick = next(((ki, kc) for ki, kc in m.kicks if i2 < ki <= i2 + 1500), None)
        cont = contact_before(m, i, toucher, eps) if toucher is not None else {}
        best = max(cont.items(), key=lambda kv: kv[1]["run"]) if cont else (None, dict(run=0, total=0))
        rows.append(dict(match=m.name, frame=int(m.frame[i]), kind=piece_kind(x2, y2), spot=[round(x2, 1), round(y2, 1)],
                         possessor_team=m.team(i, toucher) if toucher is not None else None,
                         since_touch=int(i - tj) if tj is not None else None,
                         taker_team=m.team(kick[0], kick[1]) if kick else None,
                         rival_contact_run=best[1]["run"], rival_contact_total=best[1]["total"],
                         rivals_in_contact=len([1 for v in cont.values() if v["run"] > 0])))
    return rows


def possession(m: Match, eps=0.5):
    """Por tick: el último jugador que tocó la pelota (a <= eps px del borde, o patada), -1 si nadie aún, y el tick
    en que empezó su posesión (su primer toque después del toque de otro)."""
    z = m.z
    d = np.linalg.norm(z["pl_pos"] - z["ball_pos"][:, None, :], axis=2) - PLAYER_R - BALL_R
    d[z["pl_team"] == 0] = np.inf
    d = np.nan_to_num(d, nan=np.inf)
    who = np.full(m.T, -1, np.int32)
    start = np.full(m.T, -1, np.int32)
    cur, cur_start = -1, -1
    near = d.argmin(axis=1)
    touch = d[np.arange(m.T), near] <= eps
    for i in range(m.T):
        c = m.kick_at[i][0] if i in m.kick_at else (int(near[i]) if touch[i] else -1)
        if c >= 0 and c != cur:
            cur, cur_start = c, i
        who[i], start[i] = cur, cur_start
    return who, start


def touch_segments(m: Match, eps=0.5):
    """Posesiones seguidas: [(primer toque, último toque, jugador)], con toque = pelota a <= eps px o patada."""
    z = m.z
    d = np.linalg.norm(z["pl_pos"] - z["ball_pos"][:, None, :], axis=2) - PLAYER_R - BALL_R
    d[z["pl_team"] == 0] = np.inf
    d = np.nan_to_num(d, nan=np.inf)
    near = d.argmin(axis=1)
    touch = d[np.arange(m.T), near] <= eps
    segs = []
    for i in range(m.T):
        c = m.kick_at[i][0] if i in m.kick_at else (int(near[i]) if touch[i] else -1)
        if c < 0:
            continue
        if segs and segs[-1][2] == c:
            segs[-1][1] = i
        else:
            segs.append([i, i, c])
    return segs


def offside_eval(paths, need_ball=(False, True), max_wait=900, match_tol=3):
    """Regla de offside medida, versión con desvíos: el pase es una PATADA de un compañero; el receptor es el primer
    compañero (distinto del que pateó) que toca la pelota después, aunque antes la haya tocado un rival sin patearla (un
    desvío no anula el offside). Una patada de cualquier otro jugador antes de esa recepción la corta (rival: juega la
    pelota; compañero: es otro pase). Posiciones en el tick de la patada; línea = anteúltimo rival (arquero incluido); el
    receptor debe estar en campo rival y, si need_ball, por delante de la pelota. Exentas las ejecuciones de lateral,
    córner y saque de arco. Devuelve también la lista de cobros predichos por partido."""
    tot = {nb: collections.Counter() for nb in need_ball}
    misses = {nb: [] for nb in need_ball}
    extra = {nb: [] for nb in need_ball}
    for p in paths:
        m = Match(p)
        z = m.z
        who, _ = possession(m)
        ok, on = open_play(m), offside_on(m)
        real = np.array([i for i, poss, taker in calls(m, who) if poss is not None and taker is not None
                         and taker != poss])
        segs = touch_segments(m)
        seg_starts = np.array([s[0] for s in segs])
        exempt = set()
        for pi, px, py in m.placements:
            if piece_kind(px, py) in ("lateral", "corner", "saque_arco"):
                first = next((ki for ki, _ in m.kicks if ki > pi), None)
                if first is not None:
                    exempt.add(first)
        kick_ticks = np.array([k for k, _ in m.kicks])
        team, pos = z["pl_team"], z["pl_pos"]
        for nb in need_ball:
            pred = []
            for n, (k, kc) in enumerate(m.kicks):
                if not (ok[k] and on[k]) or k in exempt:
                    continue
                a = m.team(k, kc)
                nxt_kick = kick_ticks[n + 1] if n + 1 < len(kick_ticks) else m.T
                # primer toque de un compañero (no el que pateó) después de la patada y hasta la próxima patada (inclusive:
                # la recepción puede ser una patada)
                rec = None
                for s in np.where((seg_starts > k) & (seg_starts <= min(nxt_kick, k + max_wait)))[0]:
                    t, _, q = segs[s]
                    if q == kc:
                        break
                    if m.team(t, q) == a:
                        rec = (t, q)
                        break
                if rec is None:
                    continue
                t, q = rec
                if not ok[t - 1]:
                    continue
                sgn = 1.0 if a == 1 else -1.0
                dx = np.sort(sgn * pos[k, team[k] == 3 - a, 0])[::-1]
                if len(dx) < 2:
                    continue
                qx = sgn * pos[k, q, 0]
                if qx > dx[1] and qx > 0 and (qx > sgn * z["ball_pos"][k, 0] or not nb):
                    pred.append(t)
            pred = np.array(sorted(set(pred)))
            hp = sum(1 for t in pred if len(real) and np.abs(real - t).min() <= match_tol)
            hr = sum(1 for i in real if len(pred) and np.abs(pred - i).min() <= match_tol)
            tot[nb].update(pred=len(pred), pred_ok=hp, real=len(real), real_ok=hr)
            misses[nb] += [(m.name, int(m.frame[i])) for i in real if not len(pred) or np.abs(pred - i).min() > match_tol]
            extra[nb] += [(m.name, int(m.frame[t])) for t in pred if not len(real) or np.abs(real - t).min() > match_tol]
        del m
    return {f"delante_pelota={nb}": dict(
        predichos=c["pred"], aciertos_predichos=c["pred_ok"], cobros=c["real"], cobros_explicados=c["real_ok"],
        precision=round(c["pred_ok"] / max(1, c["pred"]), 3), recall=round(c["real_ok"] / max(1, c["real"]), 3),
        no_explicados=misses[nb], predichos_sin_cobro=extra[nb]) for nb, c in tot.items()}


def powershot(paths, eps_list=(0.0, 0.5, 1.0, 2.0)):
    """Carga del powershot en juego abierto: cada vez que la masa de la pelota sube desde 1,65 (inicio de carga), cuántos
    ticks llevaba el poseedor en contacto (para cada tolerancia eps), la curva de la masa desde el inicio, qué la termina,
    el % anunciado contra la masa al patear y la gravedad (comba) que deja la patada."""
    delays = {e: [] for e in eps_list}
    curve = collections.defaultdict(list)       # ticks desde el inicio -> masas
    ends = collections.Counter()
    ann_vs_inv, grav = [], []
    for p in paths:
        m = Match(p)
        z = m.z
        inv, gx, gy = z["ball_invMass"], z["ball_gx"], z["ball_gy"]
        ok = open_play(m)
        who, _ = possession(m)
        d = np.linalg.norm(z["pl_pos"] - z["ball_pos"][:, None, :], axis=2) - PLAYER_R - BALL_R
        kicks = dict(m.kicks)
        ps_ann = {}
        for i, msg in m.ann:
            g = PS_ANN.search(msg)
            if g:
                ps_ann[i] = int(g.group(1))
        for s in range(1, m.T):
            if not (ok[s] and abs(inv[s - 1] - BASE_INV) < 1e-4 and inv[s] > BASE_INV + 1e-4):
                continue
            c = who[s]
            if c < 0:
                continue
            for e in eps_list:
                n = 0
                for j in range(s - 1, max(0, s - 600), -1):
                    if d[j, c] <= e:
                        n += 1
                    else:
                        break
                delays[e].append(n)
            j = s
            while j < m.T and inv[j] > BASE_INV + 1e-4 and ok[j] and j - s < 900:
                curve[j - s].append(float(inv[j]))
                j += 1
            # qué la terminó: patada del poseedor en los 3 ticks previos, salida del juego abierto, o nada visible
            if j < m.T and any(j - 3 <= k <= j for k in kicks):
                ends["patada"] += 1
            elif j < m.T and not ok[j]:
                ends["saque/pausa"] += 1
            else:
                ends["sin patada (perdió el contacto o la tocó otro)"] += 1
        for k, kc in m.kicks:
            if not ok[k]:
                continue
            a = next((v for i, v in ps_ann.items() if k <= i <= k + 2), None)
            if a is not None:
                ann_vs_inv.append((a, float(inv[k - 1]), float(inv[k])))
                # gravedad que queda en la pelota después de la patada
                g = [(j - k, float(gx[j]), float(gy[j])) for j in range(k, min(m.T, k + 200)) if gx[j] or gy[j]]
                if g:
                    v = z["ball_vel"][k + 1]
                    pv = z["pl_vel"][k, kc]
                    grav.append(dict(pct=a, start=g[0][0], ticks=len(g), g0=[round(g[0][1], 4), round(g[0][2], 4)],
                                     ball_v=[round(float(v[0]), 3), round(float(v[1]), 3)],
                                     player_v=[round(float(pv[0]), 3), round(float(pv[1]), 3)],
                                     g_last=[round(g[-1][1], 4), round(g[-1][2], 4)]))
        del m
    res = dict(delay={str(e): dict(n=len(v), p10=float(np.percentile(v, 10)), p50=float(np.median(v)),
                                   p90=float(np.percentile(v, 90))) for e, v in delays.items() if v},
               curve={k: dict(n=len(v), p50=round(float(np.median(v)), 4), p10=round(float(np.percentile(v, 10)), 4),
                              p90=round(float(np.percentile(v, 90)), 4))
                      for k, v in sorted(curve.items()) if k % 10 == 0 and len(v) >= 20},
               ends=dict(ends), announced=ann_vs_inv, gravity=grav)
    return res


def power_bar(paths):
    """Barra de fuerza de la pelota parada: la masa de la pelota mientras espera (desde la colocación definitiva hasta la
    patada), su período, rango y fase al empezar, por tipo; el % anunciado contra la masa al patear; y la gravedad
    (comba) que deja la patada."""
    traces = collections.defaultdict(list)
    ann = collections.defaultdict(list)
    grav = collections.defaultdict(list)
    for p in paths:
        m = Match(p)
        z = m.z
        inv = z["ball_invMass"]
        for r in pieces(m):
            if r["wait"] is None or r["power"] is None and r["kind"] not in ("lateral", "penal"):
                continue
            i = m.fidx[r["frame"]]
            k = i + r["wait"]
            tr = inv[i + 1:k + 1].astype(float)
            traces[r["kind"]].append(tr)
            if r["power"] is not None:
                ann[r["kind"]].append((r["power"], float(inv[k - 1]), float(inv[k])))
            kc = next((c for ki, c in m.kicks if ki == k), None)
            g = [(j - k, float(z["ball_gx"][j]), float(z["ball_gy"][j])) for j in range(k, min(m.T, k + 200))
                 if z["ball_gx"][j] or z["ball_gy"][j]]
            if g and kc is not None:
                pv = z["pl_vel"][k, kc]
                grav[r["kind"]].append(dict(g0=[round(g[0][1], 4), round(g[0][2], 4)], start=g[0][0], ticks=len(g),
                                            player_v=[round(float(pv[0]), 3), round(float(pv[1]), 3)],
                                            ball_v=[round(float(v), 3) for v in z["ball_vel"][min(m.T - 1, k + 1)]]))
        del m
    out = {}
    for kind, trs in traces.items():
        allv = np.concatenate([t for t in trs if len(t)]) if trs else np.array([])
        first = [t[:200] for t in trs if len(t) >= 200]
        # período: primer máximo local después del primer mínimo, en la traza media de las primeras 200
        mean = np.nanmean(np.vstack(first), axis=0) if first else None
        out[kind] = dict(n=len(trs), inv_min=float(np.nanmin(allv)) if len(allv) else None,
                         inv_max=float(np.nanmax(allv)) if len(allv) else None,
                         start_values=collections.Counter(round(float(t[0]), 3) for t in trs if len(t)).most_common(5),
                         mean_first_200=[round(float(x), 3) for x in mean[::5]] if mean is not None else None,
                         announced=ann.get(kind, []), gravity=grav.get(kind, []))
    return out


def open_play(m: Match):
    """Ticks de juego abierto: en juego, pelota de juego (no la de un saque) y sin el script colocando la pelota."""
    z = m.z
    mode = np.array([ball_mode(c) for c in z["ball_color"]])
    ok = (z["state"] == 1) & (mode != "azul") & (mode != "naranja")
    for i, _, _ in m.placements:
        ok[i:i + 30] = False
    return ok


def foul_timer(m: Match, who, start, ok, tau=2.0, gap=6, ball_d=50.0):
    """Para cada tick, la duración del contacto más largo de un rival con el poseedor dentro de su posesión (tolerancia
    tau px; separaciones de hasta `gap` ticks no lo cortan). Cuenta sólo mientras el poseedor tiene la pelota a <= ball_d
    px de su borde. Devuelve (T,) en ticks."""
    z = m.z
    pos, team = z["pl_pos"], z["pl_team"]
    bdist = np.linalg.norm(pos - z["ball_pos"][:, None, :], axis=2) - PLAYER_R - BALL_R
    run = np.zeros(m.T, np.int32)
    since = {}   # rival -> tick de inicio del contacto en curso
    last = {}    # rival -> último tick en contacto
    prev_who = -1
    for i in range(m.T):
        c = who[i]
        if c < 0 or not ok[i] or not bdist[i, c] <= ball_d:
            since.clear(), last.clear()
            prev_who = c
            continue
        if c != prev_who:
            since.clear(), last.clear()
        prev_who = c
        dc = np.linalg.norm(pos[i] - pos[i, c], axis=1)
        tc = team[i, c]
        rivals = np.where((dc <= 2 * PLAYER_R + tau) & (team[i] != 0) & (team[i] != tc))[0]
        for r in rivals:
            r = int(r)
            if r not in since or i - last[r] > gap + 1:
                since[r] = max(i, int(start[i]))
            last[r] = i
        best = 0
        for r, s0 in since.items():
            if i - last[r] <= gap:
                best = max(best, i - s0)
        run[i] = best
    return run


def foul_labels(m: Match):
    """Ticks de las faltas cobradas (primera colocación de la doble colocación, sin patada en ese tick y con tiro libre
    o penal para el equipo que tenía la pelota) y de los cobros que no son falta (offside, etc.)."""
    fl, other = [], []
    for r in fouls(m, eps=1.0):
        i = m.fidx[r["frame"]]
        (fl if r["taker_team"] == r["possessor_team"] else other).append(i)
    return fl, other


def fit_fouls(paths, taus=(2.0, 3.0, 4.0), gaps=(6, 15), ball_ds=(25.0, 50.0, 100.0, 200.0),
              thresholds=range(100, 131), tol=15):
    """Barre la tolerancia de contacto (tau), las separaciones permitidas (gap), la distancia máxima de la pelota al
    poseedor (ball_d) y el umbral N. Para cada combinación: el temporizador al cobrar cada falta, y con cada N los
    disparos del modelo que coinciden con un cobro (±tol ticks) y los que no. Procesa un partido por vez."""
    cfgs = [(t, g, b) for t in taus for g in gaps for b in ball_ds]
    at_call = {c: [] for c in cfgs}
    hits = {(c, n): 0 for c in cfgs for n in thresholds}
    false = {(c, n): 0 for c in cfgs for n in thresholds}
    n_fouls = n_calls = 0
    for p in paths:
        m = Match(p)
        who, start = possession(m)
        ok = open_play(m)
        fl, other = foul_labels(m)
        calls = np.array(sorted(fl + other))
        n_fouls += len(fl)
        n_calls += len(calls)
        for c in cfgs:
            run = foul_timer(m, who, start, ok, *c)
            at_call[c] += [int(run[max(0, i - 1)]) for i in fl]
            for n in thresholds:
                fire = np.where((run[1:] >= n) & (run[:-1] < n))[0] + 1
                near = np.array([len(calls) > 0 and np.abs(calls - f).min() <= tol for f in fire], bool)
                hits[c, n] += int(near.sum())
                false[c, n] += int((~near).sum())
        del m
    res = dict(fouls=n_fouls, calls=n_calls, configs={})
    for c in cfgs:
        at = np.array(at_call[c])
        best = max(thresholds, key=lambda n: hits[c, n] - false[c, n])
        res["configs"][f"tau={c[0]},gap={c[1]},ball_d={c[2]}"] = dict(
            at_call_p10=float(np.percentile(at, 10)), at_call_p50=float(np.median(at)),
            at_call_p90=float(np.percentile(at, 90)), best_N=best, hits=hits[c, best], false=false[c, best],
            recall=hits[c, best] / max(1, n_fouls), by_N={n: (hits[c, n], false[c, n]) for n in thresholds})
    return res


OFFSIDE_TOGGLE = re.compile(r"Offside (ha sido |\[)(des)?activad|Offside \[(✅|❌)\]")


def offside_on(m: Match):
    """Por tick: el modo offside de la sala está prendido (anuncios "El Offside ha sido (des)activado" y "Modos: ...
    Offside [✅/❌]"; por defecto prendido)."""
    on = np.ones(m.T, bool)
    for i, msg in sorted(m.ann):
        if "Offside" not in msg:
            continue
        if "desactivad" in msg or "Offside [❌]" in msg:
            on[i:] = False
        elif "activad" in msg or "Offside [✅]" in msg:
            on[i:] = True
    return on


def calls(m: Match, who):
    """Cobros de tiro libre (doble colocación de la pelota): (tick, equipo que tenía la pelota, equipo que saca)."""
    out = []
    pl = m.placements
    for n, (i, x, y) in enumerate(pl[:-1]):
        i2 = pl[n + 1][0]
        if not 0 < i2 - i <= 30 or piece_kind(x, y) in ("lateral", "corner", "saque_arco"):
            continue
        kick = next(((ki, kc) for ki, kc in m.kicks if i2 < ki <= i2 + 1500), None)
        poss = m.team(i, who[i]) if who[i] >= 0 else None
        out.append((i, poss, m.team(kick[0], kick[1]) if kick else None))
    return out


def _dump(res, out):
    if out:
        Path(out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("saques", "faltas", "ajuste_faltas", "offside", "powershot", "barra"):
        sp = sub.add_parser(name)
        sp.add_argument("matches", nargs="+")
        sp.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.cmd == "saques":
        rows = [r for p in a.matches for r in pieces(Match(p))]
        _dump(rows, a.out)
        by = collections.defaultdict(list)
        for r in rows:
            by[r["kind"]].append(r)
        for k, rs in sorted(by.items(), key=lambda x: -len(x[1])):
            modes = collections.Counter(r["mode"] for r in rs)
            print(f"{k:11s} {len(rs):5d} | modo {dict(modes)} | potencia anunciada {sum(r['power'] is not None for r in rs)}")
    elif a.cmd == "faltas":
        rows = [r for p in a.matches for r in fouls(Match(p))]
        _dump(rows, a.out)
        for r in rows:
            print(r)
    elif a.cmd == "ajuste_faltas":
        res = fit_fouls(a.matches)
        _dump(res, a.out)
        print(f"{res['fouls']} faltas, {res['calls']} cobros (faltas y otros)")
        for k, v in sorted(res["configs"].items(), key=lambda kv: kv[1]["false"] - kv[1]["hits"]):
            print(k, {x: v[x] for x in v if x != "by_N"})
    elif a.cmd == "offside":
        res = offside_eval(a.matches)
        _dump(res, a.out)
        for k, v in res.items():
            print(k, {x: v[x] for x in v if x not in ("no_explicados", "predichos_sin_cobro")})
    elif a.cmd == "powershot":
        res = powershot(a.matches)
        _dump(res, a.out)
        print("demora", res["delay"], "fin", res["ends"])
    elif a.cmd == "barra":
        res = power_bar(a.matches)
        _dump(res, a.out)
        for k, v in res.items():
            print(k, v["n"], v["inv_min"], v["inv_max"], v["start_values"])


if __name__ == "__main__":
    main()
