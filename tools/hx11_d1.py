"""D1 de HaxEleven (docs/PLAN_HAXELEVEN.md): eventos de fútbol de cada partido, leídos del caché D0 (`tools.hx11_d0`).

Definiciones [definición: no son mediciones; se validan con la muestra que revisa el usuario]. Coordenadas del mapa,
"hacia el ataque" = x × (+1 rojo, −1 azul); la cancha mide 3600 px ≈ 105 m (34,3 px/m).

* **Toque:** patada, o pelota a <= 0,6 px del borde del jugador en juego abierto (0,5 de S2 + la cuantización de D0).
  Si hay varios a la vez, gana el que patea; si nadie patea, el más cercano. La patada que ejecuta una pelota parada o el
  saque inicial también es un toque.
* **Tramo:** toques seguidos del mismo jugador sin parada en el medio. Cada tramo termina en una **acción**, que sale de
  su último toque: patada (si la última patada está a <= 3 ticks del último toque) o empuje con el cuerpo.
* **Resultado** de la acción, lo primero que pasa después: toque de un compañero (`completo`), de un rival (`rival`),
  `gol`, `gol_en_contra`, `salida` (con el saque que da y para quién), `falta` u `offside` cobrados.
* **Acción con patada**, en este orden:
  - `remate`: gol, o la pelota cruza la línea del arco rival a |y| <= 270 (palo + 100), o iba hacia ahí cuando la tocó
    un rival (proyección recta desde el tick anterior, si llega con el frenado de la pelota); desde <= 1500 px de la
    línea. Al arco si |y| <= 178 (palo + radio de la pelota).
  - `centro`: desde el último tercio (>= 600 hacia el ataque) y por fuera del área (|y| >= 450), y la pelota entra al
    área rival (ampliada 100 px) antes del siguiente toque o parada.
  - `despeje`: desde el tercio propio (<= −600), de primera (tramo de <= 10 ticks) o con un rival a <= 60 px de la
    pelota, fuerte (>= 9 px/tick, ~p60 de las patadas) y no hacia el arco propio.
  - `pase`: el resto. Largo si recorre >= 1100 px (32 m; StatsBomb usa 35 yardas): el recorrido real si llega a un
    compañero; si no, el alcance de la patada (velocidad × 0,99 / 0,01, sin comba).
  - Además: `powershot` (% del anuncio de la sala en ese tick), `comba` (la pelota sale con gravedad), `saque` (si la
    patada ejecuta una pelota parada), `barra` (% del anuncio) y `inv` (masa inversa de la pelota antes).
* **Acción sin patada:** `pase` con el cuerpo si llega a un compañero a >= 60 px, `remate` si es gol, `perdida` si la
  toca un rival o sale para el rival.
* **Del que toca después:**
  - `recepcion`: pase, centro o despeje completo; `de_primera` si patea en <= 6 ticks;
  - `atajada`: remate tocado por el arquero rival (declarado con `!gk`);
  - `bloqueo`: remate tocado por otro rival, o patada cortada a <= 8 ticks y <= 60 px;
  - `intercepcion`: patada de pase, centro o despeje cortada más lejos;
  - `robo`: pérdida con el que la tenía a <= 10 px del borde;
  - `recuperacion`: pérdida con la pelota suelta.
* `conduccion`: tramo en el que la pelota se mueve >= 150 px (4,4 m) entre el primer toque y la acción, con el avance
  hacia el ataque.
* `gol`: el autor es el último toque, del equipo que hace el gol (o `propio` si es del otro). La `asistencia` es el pase
  completo que inició el tramo del autor.
* `falta` (decisión FOUL de la sala): la `recibe` el poseedor; la comete el rival que más ticks estuvo en contacto con
  él (centros a <= 33 px) en los 130 ticks anteriores.
* `offside` (decisión OFFSIDE): el atacante que toca la pelota al cobrarse (o el más cercano al punto del cobro) y el
  último compañero que le pateó (`pasador`).
* `saque`: cada pelota parada (fase 2 de D0). Tipo, equipo, ejecutor, ticks hasta sacar, % de la barra (anuncio), si
  el primer toque de otro es propio, si el equipo conserva la pelota 3 s y si hace gol en 15 s.
* Lo propio de HaxBall (usuario, 2026-10-08: la ventaja sale de ganar la posición, usar al rival o apoyarse en un
  compañero para pasarlo):
  - `pared`: pase completo A → B y B se la devuelve a A, sin parada y en <= 3 s; cuántos rivales que A tenía por
    delante (entre él y el arco rival) quedan detrás suyo al recibir, y cuánto avanzó A;
  - `usar`: patada que rebota en un rival a <= 100 px y en <= 30 ticks sin que la controle (no patea, <= 6 ticks de
    toque), y a dónde va después: rebote propio, a un compañero, al rival, saque o gol (como en `tools/x4_mechanics`:
    no se sabe la intención, son todas las patadas contra un rival cercano);
  - `cuerpeo`: el dueño de la pelota, a <= 8 px de ella, con un rival pegado (centros a <= 33 px) que está más lejos de
    la pelota y detrás suyo; tramos de >= 30 ticks, y lo que pasa en 1,5 s: falta a favor, apoyo de un compañero,
    patea, la toca el rival o parada.

Por partido, en data/hx11/d1/ (no versionado): `<nombre>.json` (eventos, sin nombres) y `<nombre>.npz` (`owner`: último
toque por tick, −1 con la pelota muerta; `seg`: índice del tramo).

  python -m tools.hx11_d1 lote [--jobs 3] [--limit N]   # D0 → D1 y reports/hx11/d1_eventos.{md,json}
  python -m tools.hx11_d1 muestra --per-kind 4          # data/hx11/d1/muestra_revision.md (con nombres: no versionar)
"""
from __future__ import annotations

import argparse
import collections
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from env.hx11 import contract as HC
from tools.hx11_d0 import FREEZE, GOAL, KICKOFF, OPEN, OUT_PENDING, PIECE, apt_names, load_d0

ROOT = Path(__file__).resolve().parent.parent
D0 = ROOT / "data" / "hx11" / "d0"
OUT = ROOT / "data" / "hx11" / "d1"
REPORTS = ROOT / "reports" / "hx11"

EPS = 0.6
KICK_END = 3
DAMP = 0.99
SHOT_WIN = HC.GOAL_HALF_H + 100.0
ON_TARGET = HC.GOAL_HALF_H + HC.BALL_R
SHOT_RANGE = 1500.0
THIRD = 600.0
CROSS_WIDE = HC.BOX_HALF_H + 100.0
BOX_PAD = 100.0
FIRST_TIME = 10
PRESSURE = 60.0
CLEAR_SPEED = 9.0
LONG_PASS = 1100.0
PUSH_PASS = 60.0
BLOCK_TICKS, BLOCK_DIST = 8, 60.0
STEAL_GAP = 10.0
ONE_TOUCH = 6
CARRY_MIN = 150.0
FOUL_LOOKBACK, FOUL_CONTACT = 130, 2 * HC.PLAYER_R + 3.0
KEEP_TICKS, GOAL_AFTER = 180, 900
PARED_TICKS = 180
USE_TICKS, USE_DIST = 30, 100.0
SHIELD_GAP, SHIELD_JOIN, SHIELD_MIN, SHIELD_AFTER = 8.0, 6, 30, 90

TEAM = np.r_[np.zeros(11, np.int64), np.ones(11, np.int64)]
ATT = (1.0, -1.0)
PIECE_ANN = {HC.CORNER: "corner", HC.GOAL_KICK: "saque", HC.FREE_KICK: "tiro libre", HC.PENALTY: "penal"}


def detect(o, meta):
    """Eventos de un partido de D0: (lista de eventos ordenada por tick, owner, seg, control)."""
    T = len(o["frame"])
    ball = o["ball_pos"].astype(np.float32) / 10
    bv = o["ball_vel"].astype(np.float32) / 1000
    pos = o["pos"].astype(np.float32) / 10
    pid = o["slot_pid"]
    # arquero declarado con !gk; si no hay, el inferido por posición (D0)
    gk = np.where(o["gk_pid"] >= 0, o["gk_pid"], o.get("gk_inferred", o["gk_pid"]))
    on = pid >= 0
    ph, pk, pt = o["phase"], o["piece_kind"], o["piece_team"]
    kick = o["kick"].astype(bool) & on
    rel = pos - ball[:, None]
    gap = np.hypot(rel[..., 0], rel[..., 1]) - (HC.PLAYER_R + HC.BALL_R)
    gap[~on] = np.inf
    near = gap.argmin(1)
    toucher = np.where((gap[np.arange(T), near] <= EPS) & (ph == OPEN), near, -1)
    for t in np.nonzero(kick.any(1) & ~np.isin(ph, (OUT_PENDING, FREEZE, GOAL)))[0]:
        ks = np.nonzero(kick[t])[0]
        toucher[t] = ks[np.argmin(gap[t, ks])]
    live = ph == OPEN
    stops = np.nonzero(~live[1:] & live[:-1])[0] + 1
    score = o["score"].astype(np.int64)
    dec = {d[0]: d for d in meta["decisions"]}
    ann = collections.defaultdict(list)
    for i, kind, pct in meta["numbers"]:
        ann[i].append((kind, pct))
    # cada anuncio de powershot va a una sola patada, la más cercana a ±1 tick (antes se pegaba a dos patadas seguidas:
    # 101,6% de los anuncios en el lote del 2026-10-08)
    kick_ticks = np.nonzero(kick.any(1))[0]
    ps_at = {}
    for i, kind, pct in meta["numbers"]:
        if kind != "powershot" or not len(kick_ticks):
            continue
        j = np.searchsorted(kick_ticks, i)
        cand = [int(kick_ticks[c]) for c in (j - 1, j) if 0 <= c < len(kick_ticks)]
        best = min(cand, key=lambda t: (abs(t - i), t))
        if abs(best - i) <= 1 and best not in ps_at:
            ps_at[best] = pct

    def announced(i, kinds, tol):
        for d in range(-tol, tol + 1):
            for kind, pct in ann.get(i + d, ()):
                if kind.startswith(kinds):
                    return pct
        return None

    def stop_info(j):
        """(resultado, equipo que hace el gol o que saca, tipo de saque) de la parada que empieza en j."""
        p = ph[j]
        if p == GOAL:
            w = score[min(T - 1, j + 10)] - score[max(0, j - 2)]
            return "gol", 0 if w[0] > 0 else 1 if w[1] > 0 else -1, 0
        if p in (OUT_PENDING, PIECE):
            k = j
            while k < T and ph[k] == OUT_PENDING:
                k += 1
            if k < T and ph[k] == PIECE:
                return "salida", int(pt[k]), int(pk[k])
            return "salida", -1, 0
        if p == FREEZE:
            d = next((dec[j + e] for e in (0, -1, 1, -2, 2) if j + e in dec), None)
            if d is not None and d[1] in ("FOUL", "OFFSIDE"):
                return ("falta" if d[1] == "FOUL" else "offside"), int(d[2]) if d[2] is not None else -1, int(d[3])
        return "parada", -1, 0

    def is_gk(t, s):
        return pid[t, s] >= 0 and pid[t, s] == gk[t, TEAM[s]]

    # tramos
    segs = []
    tt = np.nonzero(toucher >= 0)[0]
    nstop = np.searchsorted(stops, tt, "right")
    for t, ns in zip(tt, nstop):
        s = int(toucher[t])
        if segs and segs[-1]["s"] == s and segs[-1]["ns"] == ns:
            segs[-1]["b"] = int(t)
        else:
            segs.append(dict(a=int(t), b=int(t), s=s, ns=int(ns), kicks=[]))
        if kick[t, s]:
            segs[-1]["kicks"].append(int(t))
    owner = np.full(T, -1, np.int8)
    seg_id = np.full(T, -1, np.int32)
    ev = []
    action = [None] * len(segs)
    info = [None] * len(segs)
    for k, g in enumerate(segs):
        s = g["s"]
        team = int(TEAM[s])
        att = ATT[team]
        nxt = segs[k + 1] if k + 1 < len(segs) else None
        stop = int(stops[g["ns"]]) if g["ns"] < len(stops) else T
        by_touch = nxt is not None and nxt["a"] < stop
        end = nxt["a"] if by_touch else stop
        owner[g["a"]:end] = s
        seg_id[g["a"]:end] = k
        tk = g["kicks"][-1] if g["kicks"] else -1
        kicked = tk >= 0 and g["b"] - tk <= KICK_END
        a = tk if kicked else g["b"]
        o0 = ball[a]
        dest = ball[min(end, T - 1)]
        dist = float(np.hypot(*(dest - o0)))
        extra = {}
        if by_touch:
            res = "completo" if TEAM[nxt["s"]] == team else "rival"
        elif stop < T:
            res, who, kind = stop_info(stop)
            if res == "gol":
                res = "gol" if who == team else "gol_en_contra" if who >= 0 else "parada"
            elif res == "salida":
                extra = dict(saque_para=who, saque_tipo=HC.KIND_NAMES.get(kind, "?"))
            elif res in ("falta", "offside"):
                extra = dict(cobro_para=who)
        else:
            res = "fin"
        # recorrido de la carrera previa (conducción)
        c0 = ball[g["a"]]
        cd = float(np.hypot(*(o0 - c0)))
        if cd >= CARRY_MIN:
            ev.append(dict(t=g["a"], tipo="conduccion", lugar=s, equipo=team, x=round(float(c0[0])),
                           y=round(float(c0[1])), fin=a, dist=round(cd), avance=round(float((o0[0] - c0[0]) * att))))
        kind = None
        if kicked:
            v0 = bv[min(a + 1, T - 1)]
            sp = float(np.hypot(*v0))
            path = ball[a:end + 1]
            shot = on_target = False
            xs = path[:, 0] * att
            cr = np.nonzero(xs >= HC.LINE_W)[0]
            if len(cr):
                yv = abs(float(path[cr[0], 1]))
                shot, on_target = yv <= SHOT_WIN, yv <= ON_TARGET
            elif res == "rival" and end - 1 > a:
                p, v = ball[end - 1], bv[end - 1]
                if v[0] * att > 0.1:
                    tl = (att * HC.LINE_W - p[0]) / v[0]
                    yv = abs(float(p[1] + v[1] * tl))
                    if np.hypot(tl * v[0], tl * v[1]) <= np.hypot(*v) * DAMP / (1 - DAMP):
                        shot, on_target = yv <= SHOT_WIN, yv <= ON_TARGET
            shot = (shot and o0[0] * att >= HC.LINE_W - SHOT_RANGE) or res == "gol"
            in_box = (xs >= HC.BOX_X - BOX_PAD) & (np.abs(path[:, 1]) <= HC.BOX_HALF_H + BOX_PAD)
            rv = on[a] & (TEAM != team)
            press = float(np.hypot(*(pos[a, rv] - o0).T).min()) if rv.any() else np.inf
            if shot:
                kind = "remate"
                extra.update(al_arco=bool(on_target or res == "gol"),
                             dist_arco=round(float(np.hypot(att * HC.LINE_W - o0[0], o0[1]))))
            elif o0[0] * att >= THIRD and abs(o0[1]) >= CROSS_WIDE and in_box.any():
                kind = "centro"
            elif (o0[0] * att <= -THIRD and (g["b"] - g["a"] <= FIRST_TIME or press <= PRESSURE)
                  and sp >= CLEAR_SPEED and v0[0] * att > -0.3 * sp):
                kind = "despeje"
            else:
                kind = "pase"
                extra["largo"] = bool((dist if res == "completo" else sp * DAMP / (1 - DAMP)) >= LONG_PASS)
            prev = ph[a - 1] if a > 0 else OPEN
            if prev == PIECE:
                extra["saque"] = HC.KIND_NAMES.get(int(pk[a - 1]), "?")
                if int(pk[a - 1]) in PIECE_ANN:
                    extra["barra"] = announced(a, PIECE_ANN[int(pk[a - 1])], 2)
            elif prev == KICKOFF:
                extra["saque"] = "inicial"
            else:
                t_ps = next((t for t in (a, a - 1, a + 1) if t in ps_at), None)
                ps = ps_at.pop(t_ps) if t_ps is not None else None
                if ps is not None:
                    extra["powershot"] = ps
            extra.update(patada=True, vel=round(sp, 2), inv=round(float(o["ball_inv"][max(0, a - 1)]), 3),
                         comba=bool(np.abs(o["ball_g"][min(a + 1, T - 1)].astype(np.float32)).max() > 1e-5),
                         presion=round(press) if np.isfinite(press) else None)
        elif res == "gol":
            kind = "remate"
            extra.update(patada=False, al_arco=True, dist_arco=round(float(np.hypot(att * HC.LINE_W - o0[0], o0[1]))))
        elif res == "completo" and dist >= PUSH_PASS:
            kind = "pase"
            extra.update(patada=False, largo=dist >= LONG_PASS)
        elif res == "rival" or (res == "salida" and extra.get("saque_para") != team):
            kind = "perdida"
        action[k] = kind
        info[k] = dict(kind=kind, res=res, a=a, by_touch=by_touch, end=end, stop=stop, kicked=kicked, team=team)
        if kind is not None:
            ev.append(dict(t=a, tipo=kind, lugar=s, equipo=team, x=round(float(o0[0])), y=round(float(o0[1])),
                           fin=int(end), resultado=res, a=nxt["s"] if by_touch else -1,
                           dx=round(float(dest[0] - o0[0])), dy=round(float(dest[1] - o0[1])), **extra))
        if by_touch:
            B, bt = nxt["s"], nxt["a"]
            base = dict(t=bt, lugar=B, equipo=int(TEAM[B]), x=round(float(ball[bt, 0])), y=round(float(ball[bt, 1])),
                        de=s)
            if TEAM[B] == team and kind in ("pase", "centro", "despeje"):
                first = bool(nxt["kicks"] and nxt["kicks"][-1] - bt <= ONE_TOUCH and nxt["b"] - bt <= ONE_TOUCH)
                ev.append(dict(tipo="recepcion", accion=kind, de_primera=first, **base))
            elif TEAM[B] != team:
                if kind == "remate":
                    ev.append(dict(tipo="atajada" if is_gk(bt, B) else "bloqueo", accion=kind, **base))
                elif kicked and kind is not None:
                    blk = bt - a <= BLOCK_TICKS and dist <= BLOCK_DIST
                    ev.append(dict(tipo="bloqueo" if blk else "intercepcion", accion=kind, **base))
                elif kind == "perdida":
                    ev.append(dict(tipo="robo" if gap[bt, s] <= STEAL_GAP else "recuperacion", accion=kind, **base))
        if res in ("gol", "gol_en_contra"):
            scorer_team = team if res == "gol" else 1 - team
            assist = -1
            if res == "gol" and k > 0:
                p = segs[k - 1]
                if TEAM[p["s"]] == team and p["s"] != s and action[k - 1] in ("pase", "centro") and p["ns"] == g["ns"]:
                    assist = p["s"]
            ev.append(dict(t=int(end), tipo="gol", lugar=s, equipo=scorer_team, propio=res == "gol_en_contra",
                           asistencia=assist, x=round(float(o0[0])), y=round(float(o0[1]))))
    # faltas y offsides (decisiones de la sala)
    for j, typ, t, kd, x, y in meta["decisions"]:
        if typ not in ("FOUL", "OFFSIDE") or t is None:
            continue
        if typ == "FOUL":
            f = int(owner[j - 1]) if j > 0 else -1
            if f < 0 or TEAM[f] != t:
                cand = np.nonzero(on[j] & (TEAM == t))[0]
                f = int(cand[np.argmin(np.hypot(*(pos[j, cand] - (x, y)).T))]) if len(cand) else -1
            if f < 0:
                continue
            w0 = max(0, j - FOUL_LOOKBACK)
            riv = np.nonzero(TEAM != t)[0]
            dd = np.hypot(*(pos[w0:j, riv] - pos[w0:j, f][:, None]).transpose(2, 0, 1))
            cnt = ((dd <= FOUL_CONTACT) & on[w0:j, riv] & on[w0:j, f][:, None]).sum(0)
            c = int(riv[np.argmax(cnt)])
            ev.append(dict(t=j, tipo="falta", lugar=c, equipo=1 - t, x=round(x), y=round(y), sobre=f,
                           penal=kd == HC.PENALTY, contacto=int(cnt.max())))
            ev.append(dict(t=j, tipo="falta_recibida", lugar=f, equipo=t, x=round(x), y=round(y), por=c))
        else:
            att_team = 1 - t
            # el infractor toca la pelota en el mismo tick del cobro (ya en congelamiento): patada o contacto
            off = -1
            for i in range(j, max(0, j - 4), -1):
                c = np.nonzero((kick[i] | (gap[i] <= EPS)) & (TEAM == att_team))[0]
                if len(c):
                    off = int(c[np.argmin(gap[i, c])])
                    break
            by_touch = off >= 0
            if off < 0:
                cand = np.nonzero(on[j] & (TEAM == att_team))[0]
                if not len(cand):
                    continue
                spot = np.array((x - HC.OFFSIDE_SPOT_DX * ATT[att_team], y), np.float32)
                off = int(cand[np.argmin(np.hypot(*(pos[j, cand] - spot).T))])
            passer = next((p["s"] for p in reversed(segs) if p["a"] < j and p["kicks"] and p["s"] != off
                           and TEAM[p["s"]] == att_team and j - p["kicks"][-1] <= GOAL_AFTER), -1)
            ev.append(dict(t=j, tipo="offside", lugar=off, equipo=att_team, x=round(x), y=round(y), pasador=passer,
                           por_toque=by_touch))
    # pelotas paradas
    pc = np.r_[False, ph == PIECE, False]
    starts = np.nonzero(pc[1:-1] & ~pc[:-2])[0]
    ends = np.nonzero(pc[1:-1] & ~pc[2:])[0] + 1
    goals_at = np.nonzero(np.diff(score, axis=0).max(1) > 0)[0] + 1
    for j0, j1 in zip(starts, ends):
        kd, team = int(pk[j0]), int(pt[j0])
        if team < 0:
            continue
        taker = -1
        if j1 < T:
            ks = np.nonzero(kick[j1] & (TEAM == team))[0]
            if len(ks):
                taker = int(ks[0])
        row = dict(t=int(j0), tipo="saque", lugar=taker, equipo=team, saque=HC.KIND_NAMES.get(kd, "?"),
                   x=round(float(o["spot"][j0, 0]) / 10), y=round(float(o["spot"][j0, 1]) / 10), espera=int(j1 - j0),
                   ejecutado=taker >= 0)
        if taker >= 0:
            w = owner[j1:min(T, j1 + KEEP_TICKS)]
            other = w[(w >= 0) & (w != taker)]
            row["primero_propio"] = bool(len(other) and TEAM[other[0]] == team) if len(other) else None
            kept = not (other.size and (TEAM[other] != team).any())
            s2 = stops[(stops > j1) & (stops < j1 + KEEP_TICKS)]
            if kept and len(s2):
                res, who, _ = stop_info(int(s2[0]))
                kept = not (res == "gol" and who != team) and not (res in ("salida", "falta", "offside") and who != team)
            row["conserva_3s"] = bool(kept)
            gl = goals_at[(goals_at > j1) & (goals_at <= j1 + GOAL_AFTER)]
            row["gol_15s"] = bool(len(gl) and (score[gl[0]] - score[gl[0] - 1])[team] > 0)
            if kd in PIECE_ANN:
                row["barra"] = announced(int(j1), PIECE_ANN[kd], 2)
        ev.append(row)
    ev += haxball_events(segs, info, owner, pos, ball, gap, on, kick, live, meta["decisions"], stop_info)
    ev.sort(key=lambda e: (e["t"], e["tipo"]))
    # controles contra lo que anota la sala
    n_goals = int(np.clip(np.diff(score, axis=0), 0, None).sum())
    ctl = dict(
        minutos_juego=round(float(live.sum()) / 3600, 2),
        goles_marcador=n_goals, goles_detectados=sum(e["tipo"] == "gol" for e in ev),
        powershot_anuncios=sum(1 for n in meta["numbers"] if n[1] == "powershot"),
        powershot_en_patada=sum(1 for e in ev if "powershot" in e),
        barra_anuncios=sum(1 for n in meta["numbers"] if n[1] != "powershot"),
        barra_en_saque=sum(1 for e in ev if e["tipo"] == "saque" and e.get("barra") is not None),
        faltas=sum(e["tipo"] == "falta" for e in ev),
        faltas_contacto_60=sum(e["tipo"] == "falta" and e["contacto"] >= 60 for e in ev),
        offsides=sum(e["tipo"] == "offside" for e in ev),
        offsides_por_toque=sum(e["tipo"] == "offside" and e["por_toque"] for e in ev),
        offsides_con_pasador=sum(e["tipo"] == "offside" and e["pasador"] >= 0 for e in ev),
    )
    return ev, owner, seg_id, ctl


def haxball_events(segs, info, owner, pos, ball, gap, on, kick, live, decisions, stop_info):
    """Pared, usar al rival y cuerpeo (lo que el usuario describe como la ventaja en HaxBall: ganar la posición,
    usar al rival y apoyarse en un compañero para pasarlo)."""
    T = len(owner)
    ev = []
    # pared: A -> B -> A con pases completos, sin parada, en <= 3 s, y A queda por delante de algún rival que tenía
    # entre él y el arco rival al pasarla
    for k in range(len(segs) - 2):
        i0, i1 = info[k], info[k + 1]
        A, B, C = segs[k]["s"], segs[k + 1]["s"], segs[k + 2]["s"]
        if not (i0["kind"] in ("pase", "centro") and i0["res"] == "completo" and i1["kind"] in ("pase", "centro")
                and i1["res"] == "completo" and C == A and B != A and segs[k]["ns"] == segs[k + 2]["ns"]):
            continue
        t0, t2 = i0["a"], segs[k + 2]["a"]
        if t2 - t0 > PARED_TICKS:
            continue
        att = ATT[i0["team"]]
        riv = np.nonzero(on[t0] & on[t2] & (TEAM != i0["team"]))[0]
        before = (pos[t0, riv, 0] - pos[t0, A, 0]) * att > 0
        after = (pos[t2, riv, 0] - pos[t2, A, 0]) * att < 0
        ev.append(dict(t=int(t0), tipo="pared", lugar=A, equipo=i0["team"], con=B, x=round(float(ball[t0, 0])),
                       y=round(float(ball[t0, 1])), fin=int(t2), rivales_superados=int((before & after).sum()),
                       avance=round(float((pos[t2, A, 0] - pos[t0, A, 0]) * att))))
    # usar: patada que rebota en un rival cercano sin que la controle, y a dónde va después
    for k in range(len(segs) - 1):
        i0 = info[k]
        if not (i0["kicked"] and i0["by_touch"] and i0["res"] == "rival"):
            continue
        g1 = segs[k + 1]
        a, bt = i0["a"], g1["a"]
        if bt - a > USE_TICKS or np.hypot(*(ball[bt] - ball[a])) > USE_DIST or g1["kicks"] or g1["b"] - g1["a"] > 6:
            continue
        team = i0["team"]
        i1 = info[k + 1]
        if i1["by_touch"]:
            nx = segs[k + 2]["s"]
            res = ("rebote_propio" if nx == segs[k]["s"] else "rebote_companero") if TEAM[nx] == team else "rival"
        elif i1["stop"] < T:
            r, who, _ = stop_info(i1["stop"])
            if r == "gol":
                res = "gol" if who == team else "gol_en_contra"
            else:
                res = r + ("_a_favor" if who == team else "_en_contra")
        else:
            res = "fin"
        ev.append(dict(t=int(a), tipo="usar", lugar=segs[k]["s"], equipo=team, rival=g1["s"],
                       x=round(float(ball[a, 0])), y=round(float(ball[a, 1])), resultado=res))
    # cuerpeo: el dueño, a <= 8 px de la pelota, con un rival pegado (centros a <= 33 px) más lejos de la pelota y
    # detrás suyo; tramos de >= 30 ticks (cortes de <= 6 se unen)
    idx = np.nonzero(live & (owner >= 0))[0]
    if len(idx):
        A = owner[idx].astype(np.int64)
        pa = pos[idx, A]
        pb = ball[idx]
        cc = pa[:, None] - pos[idx]
        dist = np.hypot(cc[..., 0], cc[..., 1])
        db = np.hypot(*(pos[idx] - pb[:, None]).transpose(2, 0, 1))
        dA = np.hypot(*(pa - pb).T)
        sb = pb - pa
        cos = (cc[..., 0] * sb[:, None, 0] + cc[..., 1] * sb[:, None, 1]) / (dist * dA[:, None] + 1e-9)
        rival = on[idx] & (TEAM[None, :] != TEAM[A][:, None])
        sh = rival & (dist <= 2 * HC.PLAYER_R + 3.0) & (db > dA[:, None]) & (cos > 0.3)
        ok = (gap[idx, A] <= SHIELD_GAP) & sh.any(1)
        runs = []
        for j in np.nonzero(ok)[0]:
            t = int(idx[j])
            r = int(np.argmax(sh[j]))
            if runs and runs[-1][2] == A[j] and t - runs[-1][1] <= SHIELD_JOIN:
                runs[-1][1] = t
                runs[-1][3][r] += 1
            else:
                runs.append([t, t, int(A[j]), collections.Counter({r: 1})])
        fouls = [(d[0], d[2]) for d in decisions if d[1] == "FOUL" and d[2] is not None]
        for t0, t1, s, rc in runs:
            if t1 - t0 + 1 < SHIELD_MIN:
                continue
            team = int(TEAM[s])
            res = "nada"
            if any(t0 <= j <= t1 + SHIELD_AFTER and tm == team for j, tm in fouls):
                res = "falta_a_favor"
            else:
                for t in range(t1 + 1, min(T, t1 + 1 + SHIELD_AFTER)):
                    o_ = owner[t]
                    if not live[t]:
                        res = "parada"
                    elif kick[t, s]:
                        res = "patea"
                    elif o_ >= 0 and TEAM[o_] != team:
                        res = "rival"
                    elif o_ >= 0 and o_ != s:
                        res = "apoyo"
                    else:
                        continue
                    break
            ev.append(dict(t=int(t0), tipo="cuerpeo", lugar=s, equipo=team, rival=rc.most_common(1)[0][0],
                           x=round(float(ball[t0, 0])), y=round(float(ball[t0, 1])), ticks=int(t1 - t0 + 1),
                           resultado=res))
    return ev


def one(name):
    o, meta = load_d0(D0 / name)
    ev, owner, seg, ctl = detect(o, meta)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{name}.npz", owner=owner, seg=seg)
    (OUT / f"{name}.json").write_text(json.dumps(dict(name=name, control=ctl, events=ev)), encoding="utf-8")
    return name, ctl, collections.Counter(e["tipo"] for e in ev)


def summarize(names):
    """Totales de todos los partidos de D1: frecuencias por minuto de juego abierto, resultados y controles."""
    tot = collections.Counter()
    ctl = collections.Counter()
    res = collections.defaultdict(collections.Counter)
    attrs = collections.defaultdict(collections.Counter)
    ps = collections.defaultdict(collections.Counter)
    waits = collections.defaultdict(list)
    for n in names:
        d = json.loads((OUT / f"{n}.json").read_text(encoding="utf-8"))
        for k, v in d["control"].items():
            ctl[k] += v
        for e in d["events"]:
            k = e["tipo"]
            tot[k] += 1
            if "resultado" in e:
                res[k][e["resultado"]] += 1
            if k == "pase":
                attrs[k]["largo" if e.get("largo") else "corto"] += 1
                attrs[k]["patada" if e.get("patada") else "cuerpo"] += 1
                if e.get("largo"):
                    res["pase_largo"][e["resultado"]] += 1
            if k == "remate":
                attrs[k]["al_arco" if e.get("al_arco") else "afuera"] += 1
            if k == "recepcion":
                attrs[k]["de_primera" if e["de_primera"] else "controla"] += 1
            if "powershot" in e:
                b = min(100, (e["powershot"] // 25) * 25)
                ps[f"{b}-{b + 24}%" if b < 100 else "100%"][e["tipo"] + ":" + e["resultado"]] += 1
            if k == "saque":
                attrs[f"saque_{e['saque']}"]["ejecutado" if e["ejecutado"] else "no_ejecutado"] += 1
                if e["ejecutado"]:
                    attrs[f"saque_{e['saque']}"]["conserva_3s" if e["conserva_3s"] else "pierde_3s"] += 1
                    attrs[f"saque_{e['saque']}"]["gol_15s"] += int(e["gol_15s"])
                    waits[e["saque"]].append(e["espera"])
            if k == "gol":
                attrs[k]["en_contra" if e["propio"] else "a_favor"] += 1
                attrs[k]["con_asistencia"] += int(e["asistencia"] >= 0)
            if k == "falta":
                attrs[k]["penal"] += int(e["penal"])
    mins = ctl["minutos_juego"]
    return dict(partidos=len(names), minutos_juego=round(mins, 1), control=dict(ctl),
                por_minuto={k: round(v / mins, 3) for k, v in tot.most_common()}, totales=dict(tot),
                resultados={k: dict(v) for k, v in res.items()}, atributos={k: dict(v) for k, v in attrs.items()},
                powershot={k: dict(v) for k, v in sorted(ps.items())},
                espera_saque={k: [int(np.percentile(v, q)) for q in (10, 50, 90)] for k, v in waits.items()})


def pct(a, b):
    return f"{100 * a / b:.1f}%" if b else "—"


def report(s):
    c = s["control"]
    r, at = s["resultados"], s["atributos"]
    lines = [
        "# D1: eventos de HaxEleven",
        "",
        f"Etapa D1 de [`docs/PLAN_HAXELEVEN.md`](../../docs/PLAN_HAXELEVEN.md), con las definiciones de "
        f"[`tools/hx11_d1.py`](../../tools/hx11_d1.py). Son {s['partidos']} partidos del estadio actual "
        f"(\"HaxEleven RSx11 [1800, 1200]\") y {s['minutos_juego']:.0f} minutos de juego abierto. Los datos están en "
        f"[`d1_eventos.json`](d1_eventos.json).",
        "",
        "## Controles contra lo que anota la sala",
        "",
        "| Control | Sala | Detectado | Acuerdo |",
        "|---|---|---|---|",
        f"| Goles (subidas del marcador) | {c['goles_marcador']} | {c['goles_detectados']} | "
        f"{pct(min(c['goles_detectados'], c['goles_marcador']), c['goles_marcador'])} |",
        f"| Powershot anunciado, en una patada de juego abierto | {c['powershot_anuncios']} | "
        f"{c['powershot_en_patada']} | {pct(c['powershot_en_patada'], c['powershot_anuncios'])} |",
        f"| % de la barra anunciado, en una pelota parada ejecutada | {c['barra_anuncios']} | {c['barra_en_saque']} | "
        f"{pct(c['barra_en_saque'], c['barra_anuncios'])} |",
        f"| Faltas con autor en contacto >= 1 s | {c['faltas']} | {c['faltas_contacto_60']} | "
        f"{pct(c['faltas_contacto_60'], c['faltas'])} |",
        f"| Offsides con el infractor tocando la pelota | {c['offsides']} | {c['offsides_por_toque']} | "
        f"{pct(c['offsides_por_toque'], c['offsides'])} |",
        f"| Offsides con pasador | {c['offsides']} | {c['offsides_con_pasador']} | "
        f"{pct(c['offsides_con_pasador'], c['offsides'])} |",
        "",
        "## Frecuencia por minuto de juego abierto (los 22 jugadores)",
        "",
        "| Evento | Por minuto | Total |",
        "|---|---|---|",
    ]
    lines += [f"| {k} | {v} | {s['totales'][k]} |" for k, v in s["por_minuto"].items()]
    lines += ["", "## Resultado de cada acción", "", "| Acción | Completo | Rival | Salida | Gol | Otro | n |",
              "|---|---|---|---|---|---|---|"]
    for k in ("pase", "pase_largo", "centro", "remate", "despeje", "perdida"):
        v = r.get(k, {})
        n = sum(v.values())
        other = n - sum(v.get(x, 0) for x in ("completo", "rival", "salida", "gol"))
        lines.append(f"| {k} | {pct(v.get('completo', 0), n)} | {pct(v.get('rival', 0), n)} | "
                     f"{pct(v.get('salida', 0), n)} | {pct(v.get('gol', 0), n)} | {pct(other, n)} | {n} |")
    lines += ["", "## Atributos", ""]
    for k, v in at.items():
        lines.append(f"- **{k}:** " + ", ".join(f"{a} {b}" for a, b in v.items()))
    lines += ["", "## Powershot: resultado según la carga", "", "| Carga | Acciones | Gol | Completo | Rival | Salida |",
              "|---|---|---|---|---|---|"]
    for b, v in s["powershot"].items():
        n = sum(v.values())
        agg = collections.Counter()
        for key, x in v.items():
            agg[key.split(":")[1]] += x
        lines.append(f"| {b} | {n} | {pct(agg['gol'], n)} | {pct(agg['completo'], n)} | {pct(agg['rival'], n)} | "
                     f"{pct(agg['salida'], n)} |")
    lines += ["", "## Espera hasta sacar (ticks: p10, p50, p90)", ""]
    lines += [f"- {k}: {v}" for k, v in s["espera_saque"].items()]
    return "\n".join(lines) + "\n"


def batch(jobs, limit=None):
    man = json.loads((D0 / "manifest.json").read_text(encoding="utf-8"))
    names = apt_names(man)[:limit]
    done = []
    with ProcessPoolExecutor(jobs) as ex:
        for k, (n, ctl, cnt) in enumerate(ex.map(one, names, chunksize=4), 1):
            done.append(n)
            if k % 50 == 0:
                print(f"{k}/{len(names)}", flush=True)
    s = summarize(done)
    (REPORTS / "d1_eventos.json").write_text(json.dumps(s, indent=1, ensure_ascii=False), encoding="utf-8")
    (REPORTS / "d1_eventos.md").write_text(report(s), encoding="utf-8")
    print(json.dumps(s["control"]))
    return s


def mmss(sec):
    sec = max(0, int(round(sec)))
    return f"{sec // 60}:{sec % 60:02d}"


def names_by_recording():
    out = {}
    for f in sorted((ROOT / "data" / "hx11").glob("meta_*.jsonl")):
        for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.strip():
                r = json.loads(line)
                out.setdefault(r["file"].removesuffix(".hbr2"), r.get("names") or {})
    return out


def sample(per_kind, seed=0):
    """Muestra para revisar a ojo en las repeticiones (con nombres: queda en data/, no se versiona)."""
    rng = np.random.default_rng(seed)
    files = sorted(OUT.glob("*.json"))
    apt = set(apt_names())
    files = [f for f in files if f.name != "muestra_revision.json" and f.stem in apt]
    kinds = ["pase", "pase_largo", "pase_cuerpo", "centro", "remate", "atajada", "despeje", "intercepcion", "bloqueo",
             "robo", "conduccion", "falta", "offside", "saque", "gol", "pared", "usar", "cuerpeo"]
    pick = {k: [] for k in kinds}
    for f in rng.permutation(files)[:max(per_kind * 3, 12)]:
        d = json.loads(f.read_text(encoding="utf-8"))
        for e in d["events"]:
            k = e["tipo"]
            if k == "pase" and e.get("largo"):
                k = "pase_largo"
            elif k == "pase" and not e.get("patada"):
                k = "pase_cuerpo"
            if k in pick:
                pick[k].append((d["name"], e))
    names = names_by_recording()
    color = ("rojo", "azul")
    rows = []
    cache = {}
    for k in kinds:
        items = pick[k]
        for i in (rng.permutation(len(items))[:per_kind] if len(items) > per_kind else range(len(items))):
            rec, e = items[i]
            if rec not in cache:
                o, _ = load_d0(D0 / rec)
                cache[rec] = (o["frame"], o["time"], o["score"], o["slot_pid"])
            frame, clock, score, spid = cache[rec]
            t = e["t"]
            nm = names.get(rec, {})

            def who(slot, tt=t):
                if slot is None or slot < 0:
                    return "—"
                p = int(spid[min(tt, len(spid) - 1), slot])
                return f"{nm.get(str(p), '?')} ({color[slot // 11]})"

            det = {kk: v for kk, v in e.items() if kk not in ("t", "tipo", "lugar", "equipo", "x", "y", "fin")}
            for kk in ("a", "de", "sobre", "por", "pasador", "asistencia", "con", "rival"):
                if kk in det:
                    det[kk] = who(det[kk])
            rows.append((k, rec, mmss(frame[t] / 60), mmss(float(clock[t])), f"{score[t, 0]}-{score[t, 1]}",
                         who(e["lugar"]), f"({e['x']}, {e['y']})",
                         ", ".join(f"{kk}={v}" for kk, v in det.items())))
    lines = ["# Muestra de eventos D1 para revisar (con nombres: no versionar)", "",
             "Abrir la repetición en el visor de HaxBall e ir al tiempo de la repetición. Marcar en la columna "
             "\"¿Bien?\" sí, no o dudoso; si es no, qué fue en realidad.", "",
             "| # | Evento | Repetición | Tiempo de la repetición | Reloj | Marcador | Jugador | Pelota | Detalle | ¿Bien? |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for n, (k, rec, tr, tc, sc, w, xy, det) in enumerate(rows, 1):
        lines.append(f"| {n} | {k} | {rec} | {tr} | {tc} | {sc} | {w} | {xy} | {det} | |")
    path = OUT / "muestra_revision.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(rows)} eventos en {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    u = sub.add_parser("uno")
    u.add_argument("name")
    b = sub.add_parser("lote")
    b.add_argument("--jobs", type=int, default=3)
    b.add_argument("--limit", type=int, default=None)
    m = sub.add_parser("muestra")
    m.add_argument("--per-kind", type=int, default=4)
    m.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.cmd == "uno":
        n, ctl, cnt = one(a.name)
        print(n, json.dumps(ctl), dict(cnt.most_common()))
    elif a.cmd == "lote":
        batch(a.jobs, a.limit)
    else:
        sample(a.per_kind, a.seed)


if __name__ == "__main__":
    main()
