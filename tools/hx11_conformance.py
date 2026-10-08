"""Conformidad del simulador con las grabaciones HaxEleven (docs/PLAN_HAXELEVEN.md, S1 y luego S4).

`fisica`: física de HaxBall en juego abierto. Carga el estado grabado de un tick en un BatchSim de un partido
(posiciones, velocidades y las PROPIEDADES que dejó el script de la sala: invMass de la pelota con la carga del
powershot, invMass/bCoef/cMask/cGroup de jugadores, barreras, gravedad de la pelota), avanza k ticks con las entradas
grabadas aplicando en cada tick las propiedades grabadas, y mide el error contra el juego. Se excluyen las transiciones
en que el script mueve un disco simulado (colocaciones de saque), los goles, los cambios de plantel y los tramos con
gravedad en un jugador (el motor sólo modela la de la pelota). `--align` elige qué propiedades valen en la transición
t -> t+1: las del tick t ("t") o las del t+1 ("t1").

`bench`: decisiones de partido por segundo del motor con 11v11 en el estadio (y 4v4 de referencia).

  node bridge/hx11_ticks.js <grabación.hbr2> --out data/hx11/ticks --max-minutes 3
  python -m tools.hx11_conformance fisica data/hx11/ticks/*.jsonl --out reports/hx11/s1_fisica.json
  python -m tools.hx11_conformance bench
"""
from __future__ import annotations

import argparse
import collections
import json
import time
from pathlib import Path

import numpy as np

from sim.physics import MOVE_DIRS, BatchSim
from sim.stadium import load_stadium

STADIUM = "haxeleven_x11"
HORIZONS = (1, 10, 30, 60)
PROPS = ("radius", "invMass", "bCoef", "damping", "cMask", "cGroup", "gx", "gy")

# entrada de HaxBall: 1 arriba, 2 abajo, 4 izquierda, 8 derecha, 16 patear -> acción 0..17 del sim
_DIR2MOVE = {(int(x), int(y)): i for i, (x, y) in enumerate(MOVE_DIRS)}


def input_to_action(inp: int) -> int:
    dx = ((inp >> 3) & 1) - ((inp >> 2) & 1)
    dy = ((inp >> 1) & 1) - (inp & 1)
    return _DIR2MOVE[(dx, dy)] + (9 if inp & 16 else 0)


def players_dict(p):
    """Jugador del tick en el formato compacto [id, equipo, input, patada activa, disco] o el viejo (dict)."""
    if isinstance(p, dict):
        return p
    return dict(id=p[0], team=p[1], input=p[2], kicking=bool(p[3]), disc=p[4])


class Recording:
    """Ticks de `bridge/hx11_ticks.js` con el estado completo reconstruido (los discos quietos y las propiedades sin
    cambio no se repiten en el archivo)."""

    def __init__(self, path):
        self.path = Path(path)
        self.header, ticks, self.events = None, [], []
        for line in open(path, encoding="utf-8"):
            try:
                o = json.loads(line)
            except json.JSONDecodeError:  # última línea cortada
                continue
            if o["type"] == "tick":
                o["players"] = [players_dict(p) for p in o["players"]]
                ticks.append(o)
            elif o["type"] == "header":
                self.header = o
            else:
                self.events.append(o)
        self.ticks = ticks
        T = len(ticks)
        nd = max(t["nd"] for t in ticks)
        self.pos = np.full((T, nd, 2), np.nan)
        self.vel = np.full((T, nd, 2), np.nan)
        self.prop = {k: np.full((T, nd), np.nan) for k in PROPS}
        cur_pv = np.full((nd, 4), np.nan)
        cur_pr = {k: np.full(nd, np.nan) for k in PROPS}
        for j, t in enumerate(ticks):
            for src in (t["m"], t["d"]):
                for i, pv in src.items():
                    cur_pv[int(i)] = [np.nan if v is None else v for v in pv]
            for i, ch in t["p"].items():
                for k, v in ch.items():
                    if k in cur_pr:
                        cur_pr[k][int(i)] = np.nan if v is None else v
            self.pos[j] = cur_pv[:, :2]
            self.vel[j] = cur_pv[:, 2:]
            for k in PROPS:
                self.prop[k][j] = cur_pr[k]
        self.frame = np.array([t["frame"] for t in ticks])
        # discos que el script coloca o lanza (setDiscProperties con x, y, vx o vy), por frame
        self.teleports = collections.defaultdict(set)
        for e in self.events:
            if e["name"] == "disc_props" and e.get("d1") and any(v is not None for v in e["d1"][:4]):
                self.teleports[e["frame"]].add(("p" if e["player"] else "d", e["id"]))


def roster(t):
    return tuple(sorted((p["id"], p["team"]) for p in t["players"]))


class Replayer:
    """BatchSim de un partido alineado con un tramo de plantel fijo de la grabación."""

    def __init__(self, st, rec: Recording, j0: int, align: str):
        self.rec, self.align = rec, align
        t0 = rec.ticks[j0]
        # mismo orden de discos que el juego: HaxBall resuelve los choques de a pares en ese orden y, con varios
        # jugadores en contacto a la vez (cuerpear), otro orden da otro resultado (hasta 0,07 px por tick)
        self.players = sorted(t0["players"], key=lambda p: p["disc"])
        n_red = sum(p["team"] == 1 for p in self.players)
        self.sim = BatchSim(1, n_red, len(self.players) - n_red, st)
        self.sim.player_team[:] = [0 if p["team"] == 1 else 1 for p in self.players]
        self.stadium_idx = [0] + [1 + int(i) for i in st.d_src]
        self.pids = [p["id"] for p in self.players]

    def disc_idx(self, j):
        by_id = {p["id"]: p["disc"] for p in self.rec.ticks[j]["players"]}
        return np.array(self.stadium_idx + [by_id[i] for i in self.pids])

    def set_props(self, j):
        s, r = self.sim, self.rec
        idx = self.disc_idx(j)
        s.radius[:] = r.prop["radius"][j, idx]
        s.inv_env[0] = r.prop["invMass"][j, idx]
        s.bcoef[:] = r.prop["bCoef"][j, idx]
        s.damping[:] = r.prop["damping"][j, idx]
        s.group[:] = r.prop["cGroup"][j, idx].astype(np.int64)
        s.mask[0] = r.prop["cMask"][j, idx].astype(np.int64)
        s.ball_grav[0] = [r.prop["gx"][j, 0], r.prop["gy"][j, 0]]

    def load(self, j):
        s, r = self.sim, self.rec
        idx = self.disc_idx(j)
        s.pos[0] = r.pos[j, idx]
        s.vel[0] = r.vel[j, idx]
        by_id = {p["id"]: p for p in r.ticks[j]["players"]}
        # patada "gastada": tecla apretada pero isKicking=false
        s.kick_cancel[0] = [bool(by_id[i]["input"] & 16) and not by_id[i]["kicking"] for i in self.pids]
        s.kickoff[0] = False  # las máscaras del saque inicial vienen de la grabación
        self.set_props(j)

    def step(self, j):
        """Transición j -> j+1 con la entrada grabada en j y las propiedades de j (o de j+1)."""
        self.set_props(j if self.align == "t" else j + 1)
        by_id = {p["id"]: p for p in self.rec.ticks[j]["players"]}
        self.sim.step(np.array([[input_to_action(by_id[i]["input"]) for i in self.pids]]))

    def error(self, j):
        idx = self.disc_idx(j)
        real = self.rec.pos[j, idx]
        p = self.sim.pos[0]
        fp = self.sim.first_player
        return float(np.linalg.norm(p[0] - real[0])), float(np.linalg.norm(p[fp:] - real[fp:], axis=1).max())


def window_ok(rec, rp, i, k, states, reasons):
    """Tramo i..i+k comparable: frames seguidos, mismo plantel y marcador, estado pedido, sin colocaciones del script
    sobre discos simulados y sin gravedad en jugadores."""
    t = rec.ticks
    if t[i]["state"] not in states:
        reasons["estado"] += 1
        return False
    sim_discs = set(rp.stadium_idx)
    for j in range(i, i + k):
        a, b = t[j], t[j + 1]
        if b["frame"] - a["frame"] != 1:
            reasons["frames"] += 1
            return False
        if b["score"] != t[i]["score"] or roster(b) != roster(t[i]):
            reasons["gol/plantel"] += 1
            return False
        for kind, did in rec.teleports.get(b["frame"], set()) | rec.teleports.get(a["frame"], set()):
            if kind == "p" or did in sim_discs:
                reasons["colocación del script"] += 1
                return False
    idx = rp.disc_idx(i)[rp.sim.first_player:]
    g = np.abs(rec.prop["gx"][i:i + k + 1][:, idx]) + np.abs(rec.prop["gy"][i:i + k + 1][:, idx])
    if np.nanmax(g) > 0:
        reasons["gravedad en jugador"] += 1
        return False
    if np.isnan(rec.pos[i:i + k + 1][:, rp.disc_idx(i)]).any():
        reasons["disco sin posición"] += 1
        return False
    return True


def physics(recs, align, states=(1,)):
    st = load_stadium(STADIUM)
    out = {}
    worst = []
    for k in HORIZONS:
        eb, ep, reasons, n_props = [], [], collections.Counter(), 0
        for rec in recs:
            rps = {}
            for i in range(0, len(rec.ticks) - k - 1, max(1, k // 3)):
                key = roster(rec.ticks[i])
                if key not in rps:
                    rps[key] = Replayer(st, rec, i, align)
                rp = rps[key]
                if not window_ok(rec, rp, i, k, states, reasons):
                    continue
                rp.load(i)
                for j in range(i, i + k):
                    rp.step(j)
                e_b, e_p = rp.error(i + k)
                eb.append(e_b)
                ep.append(e_p)
                if k == 1:
                    changed = any(not np.array_equal(rec.prop[p][i], rec.prop[p][i + 1], equal_nan=True)
                                  for p in PROPS)
                    n_props += changed
                    worst.append((max(e_b, e_p), rec.path.name, int(rec.frame[i + 1]), e_b, e_p, changed))
        eb, ep = np.array(eb), np.array(ep)
        out[k] = dict(n=len(eb), reasons=dict(reasons),
                      ball=dict(p50=float(np.median(eb)), p90=float(np.percentile(eb, 90)), max=float(eb.max()))
                      if len(eb) else None,
                      players=dict(p50=float(np.median(ep)), p90=float(np.percentile(ep, 90)), max=float(ep.max()))
                      if len(ep) else None)
        if k == 1:
            out[k]["transitions_with_prop_change"] = n_props
    worst.sort(key=lambda x: -x[0])
    out["worst_1tick"] = [dict(err=round(w[0], 4), file=w[1], frame=w[2], ball=round(w[3], 4), players=round(w[4], 4),
                               prop_change=w[5]) for w in worst[:15]]
    return out


def bench(n_envs=256, steps=600):
    res = {}
    for name, (nr, nb, stadium) in {"11v11 haxeleven_x11": (11, 11, STADIUM),
                                     "4v4 sanguchito_rs_x4": (4, 4, "sanguchito_rs_x4")}.items():
        sim = BatchSim(n_envs, nr, nb, stadium, seed=0)
        rng = np.random.default_rng(0)
        acts = rng.integers(0, 18, size=(steps, n_envs, nr + nb))
        sim.step(acts[0])  # compilación
        t0 = time.perf_counter()
        for a in acts:
            sim.step(a)
        dt = time.perf_counter() - t0
        res[name] = dict(n_envs=n_envs, steps=steps, ticks_per_s=round(n_envs * steps / dt),
                         player_ticks_per_s=round(n_envs * steps * (nr + nb) / dt))
    return res


# ============================================================================ S4: árbitro en sombra
SHADOW_TOL = dict(PLACE=4, FOUL=12, OFFSIDE=4)   # ticks de tolerancia para emparejar una decisión con la de la sala


def room_decisions(m):
    """Decisiones de la sala leídas de la grabación: [(tick, tipo, equipo 0/1, kind, x, y)] con tipo PLACE (colocación
    para un saque), FOUL u OFFSIDE (primera colocación de una doble colocación: congela la pelota)."""
    from env.hx11 import contract as HC
    from tools.hx11_script import piece_kind, possession

    kinds = dict(lateral=HC.LATERAL, corner=HC.CORNER, saque_arco=HC.GOAL_KICK, tiro_libre=HC.FREE_KICK,
                 penal=HC.PENALTY)
    who, _ = possession(m)

    def taker_after(t0):
        return next((m.team(ki, kc) - 1 for ki, kc in m.kicks if ki > t0), None)

    pl, out, n = m.placements, [], 0
    while n < len(pl):
        i, x, y = pl[n]
        kind = piece_kind(x, y)
        nxt = pl[n + 1] if n + 1 < len(pl) else None
        double = (nxt is not None and 0 < nxt[0] - i <= 30 and kind in ("tiro_libre", "penal")
                  and (abs(nxt[1] - x) + abs(nxt[2] - y) < 1e-3 or piece_kind(nxt[1], nxt[2]) == "penal"))
        if double:
            i2, x2, y2 = nxt
            taker = taker_after(i2)
            poss = m.team(i, who[i]) - 1 if who[i] >= 0 else None
            k2 = kinds[piece_kind(x2, y2)]
            out.append((i, "FOUL" if taker == poss else "OFFSIDE", taker, k2, x, y))
            out.append((i2, "PLACE", taker, k2, x2, y2))
            n += 2
        else:
            out.append((i, "PLACE", taker_after(i), kinds[kind], x, y))
            n += 1
    return out


def shadow(path):
    """Árbitro del simulador (env/hx11) sobre los estados grabados de un partido, tick a tick.

    En cada tick se cargan posiciones y velocidades de la pelota y de los jugadores, las patadas y los goles de la
    grabación, y se corre sólo el árbitro. Cada decisión suya (colocación, falta, offside) se empareja con una de la sala
    del mismo tipo y equipo a <= SHADOW_TOL ticks; si no la hay, es un falso positivo y se deshace. Si la sala decide
    algo que el árbitro no decidió en ese margen, se fuerza la decisión de la sala (así cada regla se mide sin
    arrastrar errores). Se comparan además, por tick, la masa de la pelota, la barrera, el disco del punto y los
    jugadores sin masa."""
    from env.hx11 import contract as HC
    from env.hx11 import kernel as K
    from env.hx11.core import HX11Env
    from tools.hx11_script import Match, offside_on

    m = Match(path)
    z = m.z
    env = HX11Env(1)
    fp, P, nr = env.fp, env.P, env.n_red
    pos, vel, mask, group, radius, inv, grav = (env.pos[0], env.vel[0], env.mask[0], env.group[0], env.radius[0],
                                               env.inv[0], env.grav[0])
    ri, rf, fs, fl = env.ri[0], env.rf[0], env.fs[0], env.fl[0]
    hold_cdf = HC.fixed_hold_cdf(env.hold_ticks)   # duración típica del valor fijo (sin sorteo) para comparar
    on = offside_on(m)
    room = room_decisions(m)
    matched = [False] * len(room)
    by_tick = collections.defaultdict(list)
    for n, r in enumerate(room):
        by_tick[r[0]].append(n)
    goals = {m.at(e["frame"]): (1 if e["team"] == 1 else -1) for e in m.events if e["name"] == "goal"}
    di = list(z["disc_ids"])
    d1, d2 = di.index(1), di.index(2)
    team_rec, ppos, pvel = z["pl_team"], z["pl_pos"], z["pl_vel"]
    bpos, bvel, binv = z["ball_pos"], z["ball_vel"], z["ball_invMass"]
    slot_of = {}                       # columna de la grabación -> lugar en el entorno
    free = {1: list(range(nr)), 2: list(range(nr, P))}
    kicked = np.zeros(P, np.bool_)
    kvel = np.zeros((P, 2))
    res = collections.defaultdict(list)
    cnt = collections.Counter()
    pending = []                       # decisiones de la sala sin emparejar todavía
    suppress = {}                      # tipo -> tick hasta el que un falso positivo repetido no se vuelve a contar
    last_kick_kind, last_kick_charged = -1, False   # origen de la comba vigente

    def decision_of_event(e, i):
        if e in (K.EV_PLACE, K.EV_TIMEOUT):
            return ("PLACE", int(ri[K.RI_TEAM]), int(ri[K.RI_KIND]), float(rf[K.RF_SPOT_X]), float(rf[K.RF_SPOT_Y]))
        if e == K.EV_FOUL:
            return ("FOUL", int(ri[K.RI_TEAM]), int(ri[K.RI_KIND]), float(rf[K.RF_SPOT_X]), float(rf[K.RF_SPOT_Y]))
        if e == K.EV_OFFSIDE:
            return ("OFFSIDE", int(ri[K.RI_TEAM]), int(ri[K.RI_KIND]), float(rf[K.RF_SPOT_X]), float(rf[K.RF_SPOT_Y]))
        return None

    for i in range(m.T):
        # ---- plantel: cada jugador conserva su lugar mientras está en la cancha
        cols_now = {int(c): int(team_rec[i, c]) for c in np.where(team_rec[i] > 0)[0]}
        for c in list(slot_of):
            if cols_now.get(c) is None or (slot_of[c] < nr) != (cols_now[c] == 1):
                s = slot_of.pop(c)
                free[1 if s < nr else 2].append(s)
        for c, t in cols_now.items():
            if c not in slot_of and free[t]:
                slot_of[c] = free[t].pop(0)
        # ---- estado grabado
        pos[0] = bpos[i]
        vel[0] = bvel[i]
        for s in range(P):
            pos[fp + s] = (3000.0 if s < nr else -3000.0, 0.0)   # lugares vacíos: lejos, detrás de su arco rival
            vel[fp + s] = 0.0
        for c, s in slot_of.items():
            pos[fp + s] = ppos[i, c]
            vel[fp + s] = pvel[i, c]
        kicked[:] = False
        for c in m.kick_at.get(i, []):
            if c in slot_of:
                kicked[slot_of[c]] = True
                kvel[slot_of[c]] = pvel[max(0, i - 1), c]
        ri[K.RI_OFFSIDE_ON] = int(on[i])
        if z["state"][i] == 0:
            if ri[K.RI_MODE] != K.KICKOFF:
                ri[K.RI_MODE] = K.KICKOFF
                ri[K.RI_TICKS] = 0
                ri[K.RI_LAST] = -1
                ri[K.RI_OFF_MASK] = 0
                ri[K.RI_PS_T] = -1
                ri[K.RI_PS_CONTACT] = 0
            ri[K.RI_KO] = 1
        else:
            ri[K.RI_KO] = 0
        snap = (ri.copy(), rf.copy(), fs.copy(), fl.copy(), inv.copy(), radius.copy(), mask.copy(), group.copy(),
                grav.copy())
        K.referee(pos, vel, mask, group, radius, inv, grav, ri, rf, fs, fl, kicked, kvel, goals.get(i, 0), fp, P, nr,
                  env.home, env.spawn, env.bar, env.hold, hold_cdf, env.ramp)
        dec = decision_of_event(int(ri[K.RI_EVENT]), i)
        if dec is not None:
            typ, team, kind, x, y = dec
            tol = SHADOW_TOL[typ]
            same = [n for j in range(i - tol, i + tol + 1) for n in by_tick.get(j, [])
                    if not matched[n] and room[n][1] == typ]
            exact = [n for n in same if room[n][2] in (team, None) and (typ != "PLACE" or room[n][3] == kind)]
            if exact:
                n = min(exact, key=lambda n: abs(room[n][0] - i))
                matched[n] = True
                j = room[n][0]
                res[f"{typ}:{HC.KIND_NAMES[kind]}:desfase"].append(i - j)
                res[f"{typ}:{HC.KIND_NAMES[kind]}:error_punto"].append(float(np.hypot(x - room[n][4], y - room[n][5])))
                if typ == "PLACE":
                    ri[K.RI_TICKS] = i - j      # el reloj del saque, alineado con la sala (fase de la barra)
                cnt[f"{typ}:aciertos"] += 1
            elif same:
                # misma decisión con otro equipo o tipo de saque: un desacuerdo; se impone la de la sala
                n = min(same, key=lambda n: abs(room[n][0] - i))
                matched[n] = True
                j, _, rteam, rkind, rx, ry = room[n]
                cnt[f"{typ}:desacuerdos"] += 1
                res[f"{typ}:desacuerdos_ejemplos"].append((m.name, int(m.frame[j]), HC.KIND_NAMES[kind], team,
                                                           HC.KIND_NAMES[rkind], rteam))
                if rteam is not None:
                    if typ == "PLACE":
                        K._place(pos, vel, inv, mask, group, radius, grav, ri, rf, rkind, rteam, rx, ry, fp, P, nr,
                                 env.home, env.hold, hold_cdf)
                    else:
                        K._freeze(pos, vel, grav, radius, ri, rf, rkind, rteam, rx, ry)
                    ri[K.RI_TICKS] = i - j
            else:
                # falso positivo: se deshace, se vuelve a juego abierto y no se cuenta de nuevo por 2 s
                ri[:], rf[:], fs[:], fl[:], inv[:], radius[:], mask[:], group[:], grav[:] = snap
                if ri[K.RI_MODE] in (K.OUT_PENDING, K.FREEZE):
                    ri[K.RI_MODE] = K.OPEN
                    ri[K.RI_TICKS] = 0
                if i >= suppress.get(typ, -1):
                    cnt[f"{typ}:falsos"] += 1
                    res[f"{typ}:falsos_ejemplos"].append((m.name, int(m.frame[i]), HC.KIND_NAMES[kind], team))
                suppress[typ] = i + 120
        # ---- decisiones de la sala que el árbitro no tomó a tiempo: se fuerzan
        for n in by_tick.get(i, []):
            pending.append(n)
        for n in list(pending):
            j, typ, team, kind, x, y = room[n]
            if matched[n]:
                pending.remove(n)
                continue
            if i >= j + SHADOW_TOL[typ] and team is not None:
                pending.remove(n)
                matched[n] = True
                cnt[f"{typ}:no_vistos"] += 1
                res[f"{typ}:no_vistos_ejemplos"].append((m.name, int(m.frame[j]), HC.KIND_NAMES[kind], team))
                if typ == "PLACE":
                    K._place(pos, vel, inv, mask, group, radius, grav, ri, rf, kind, team, x, y, fp, P, nr, env.home,
                             env.hold, hold_cdf)
                else:
                    K._freeze(pos, vel, grav, radius, ri, rf, kind, team, x, y)
                ri[K.RI_TICKS] = i - j
        # ---- comba: gravedad de la pelota que deja el árbitro contra la de la sala (vale para el tick siguiente)
        rg = (float(z["ball_gx"][i]) if not np.isnan(z["ball_gx"][i]) else 0.0,
              float(z["ball_gy"][i]) if not np.isnan(z["ball_gy"][i]) else 0.0)
        if rg != (0.0, 0.0) or grav[0] != 0.0 or grav[1] != 0.0:
            src = "carga" if last_kick_charged else HC.KIND_NAMES.get(last_kick_kind, "juego")
            res[f"comba:{src}"].append(float(np.hypot(grav[0] - rg[0], grav[1] - rg[1])))
            res[f"comba:{src}:sala"].append(float(np.hypot(*rg)))
        if int(ri[K.RI_EVENT]) == K.EV_PIECE_KICK:
            last_kick_kind, last_kick_charged = int(ri[K.RI_AFTER_KIND]), False
        elif kicked.any():
            last_kick_kind, last_kick_charged = -1, bool(snap[4][0] > HC.BASE_BALL_INV + 1e-3)
        # ---- masa de la pelota y discos, por tick
        mode = int(ri[K.RI_MODE])
        if z["state"][i] == 1 and not np.isnan(binv[i]):
            if mode == K.PIECE:
                res[f"masa:{HC.KIND_NAMES[int(ri[K.RI_KIND])]}"].append(abs(float(inv[0]) - float(binv[i])))
            elif mode == K.OPEN:
                key = "masa:carga" if binv[i] > HC.BASE_BALL_INV + 1e-3 or inv[0] > HC.BASE_BALL_INV + 1e-3 else "masa:juego"
                res[key].append(abs(float(inv[0]) - float(binv[i])))
        if mode == K.PIECE:
            kname = HC.KIND_NAMES[int(ri[K.RI_KIND])]
            rec_r = float(z["disc_radius"][i, d1])
            cnt[f"barrera:{kname}:ticks"] += 1
            cnt[f"barrera:{kname}:igual"] += int(abs(rec_r - float(radius[1])) < 0.5
                                                 and (rec_r < 0.5 or int(z["disc_cMask"][i, d1]) == int(mask[1])))
            # presencia del disco del punto (en reposo está a ~9999 px): el punto del árbitro puede diferir unos px
            rec_spot = bool(np.hypot(*(z["disc_pos"][i, d2] - (rf[K.RF_SPOT_X], rf[K.RF_SPOT_Y]))) < 50.0)
            cnt[f"punto:{kname}:ticks"] += 1
            cnt[f"punto:{kname}:igual"] += int(rec_spot == (radius[2] > 0))
            ghosts_rec = [bool(z["pl_invMass"][i, c] > 1e5) for c in slot_of]
            ghosts_sim = [bool(inv[fp + s] > 1e5) for s in slot_of.values()]
            cnt[f"sin_masa:{kname}:jugadores"] += len(ghosts_rec)
            cnt[f"sin_masa:{kname}:igual"] += sum(a == b for a, b in zip(ghosts_rec, ghosts_sim))
    cnt["sala:decisiones"] = len(room)
    for n, r in enumerate(room):
        if not matched[n]:
            cnt[f"{r[1]}:sin_cerrar"] += 1
    del m
    return res, cnt


def arbitro(paths):
    res, cnt = collections.defaultdict(list), collections.Counter()
    for p in paths:
        r, c = shadow(p)
        for k, v in r.items():
            res[k] += v
        cnt.update(c)
    out = dict(conteos=dict(cnt), resumen={})
    for typ in ("PLACE", "FOUL", "OFFSIDE"):
        a, f, nv, d = cnt[f"{typ}:aciertos"], cnt[f"{typ}:falsos"], cnt[f"{typ}:no_vistos"], cnt[f"{typ}:desacuerdos"]
        out["resumen"][typ] = dict(aciertos=a, desacuerdos=d, falsos=f, no_vistos=nv,
                                   recall=round(a / max(1, a + nv + d), 3), precision=round(a / max(1, a + f + d), 3))
    for k, v in sorted(res.items()):
        if k.endswith("_ejemplos"):
            out["resumen"][k] = v[:40]
            continue
        v = np.array(v, dtype=float)
        out["resumen"][k] = dict(n=len(v), p50=round(float(np.median(v)), 4), p90=round(float(np.percentile(v, 90)), 4),
                                 max=round(float(v.max()), 4), cero=round(float(np.mean(np.abs(v) < 1e-3)), 3))
    for k in list(cnt):
        if k.endswith(":ticks") or k.endswith(":jugadores"):
            base = k.rsplit(":", 1)[0]
            out["resumen"][f"{base}:acuerdo"] = round(cnt[f"{base}:igual"] / max(1, cnt[k]), 4)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fisica")
    f.add_argument("recordings", nargs="+")
    f.add_argument("--align", choices=("t", "t1", "ambos"), default="ambos")
    f.add_argument("--out", default=None)
    b = sub.add_parser("bench")
    b.add_argument("--envs", type=int, default=256)
    b.add_argument("--steps", type=int, default=600)
    s = sub.add_parser("arbitro")
    s.add_argument("matches", nargs="+", help="partidos de tools.hx11_npz (data/hx11/npz/*.npz)")
    s.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.cmd == "arbitro":
        res = arbitro(a.matches)
        if a.out:
            Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
        for k, v in res["resumen"].items():
            if not k.endswith("_ejemplos"):
                print(k, v)
        return
    if a.cmd == "fisica":
        recs = [Recording(p) for p in a.recordings]
        res = {al: dict(open_play=physics(recs, al), kickoff=physics(recs, al, states=(0,)))
               for al in (("t", "t1") if a.align == "ambos" else (a.align,))}
        res = dict(stadium=STADIUM, recordings=[Path(p).name for p in a.recordings], results=res)
        txt = json.dumps(res, indent=1, ensure_ascii=False)
        if a.out:
            Path(a.out).write_text(txt, encoding="utf-8")
        for al, r in res["results"].items():
            for part in ("open_play", "kickoff"):
                for k in HORIZONS:
                    x = r[part][k]
                    if x["ball"]:
                        print(f"[{al}] {part:9s} {k:3d} ticks n={x['n']:6d} | pelota p50 {x['ball']['p50']:.4f} "
                              f"p90 {x['ball']['p90']:.4f} máx {x['ball']['max']:.3f} | jugadores p50 "
                              f"{x['players']['p50']:.4f} p90 {x['players']['p90']:.4f} máx {x['players']['max']:.3f}")
                print(f"[{al}] {part} 1 tick: {r[part][1].get('transitions_with_prop_change')} con cambio de "
                      f"propiedades; excluidos {r[part][1]['reasons']}")
    else:
        print(json.dumps(bench(a.envs, a.steps), indent=1))


if __name__ == "__main__":
    main()
