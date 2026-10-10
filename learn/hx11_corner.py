"""Córners: cómo los sacan los bots (lazo reactivo con los 22, configuración de sala) contra los humanos del banco.

Pedido del usuario (2026-10-09): evaluar el córner como el lateral. Episodios desde córners reales (estado grabado en la
colocación + la colocación del árbitro del simulador), con las mismas métricas para bots y humanos:
* `espera`: ticks desde la colocación hasta la patada del equipo que saca;
* `italiano`: después de la patada, la pelota cruza la línea de fondo sin que nadie la toque ("CORNER ITALIANO? 🍕" de
  la sala: sale hacia afuera, pega en la red de afuera, no llega al área);
* `al_area`: la pelota entra al área grande del arco atacado antes del primer toque;
* `primer_toque`: atacante (equipo que saca, sin contar al que pateó), defensor o nadie en 180 ticks;
* `remate`: patada del equipo que saca dentro del área grande en los 300 ticks después del córner;
* `gol`: gol del equipo que saca en los 600 ticks después del córner.

  python -m learn.hx11_corner --model M [--split dev] [--corners 300] [--out reports/hx11/corner_bots]

Ajuste del córner (autoimitación filtrada, como `learn/hx11_ajuste_lateral.py` pero solo del que patea): con
`--colectar <npz>`, de los córners buenos (pateado desde atrás de la pelota, llega al área, no es italiano) se guardan
las decisiones del que pateó, desde la colocación hasta la patada (cómo rodea la pelota y desde dónde patea). El
ajuste es `learn.hx11_bc --init <modelo> --extra-data <npz> --extra-frac f`.
"""
from __future__ import annotations

import argparse
import collections
import json
import time
from pathlib import Path

import numpy as np
import torch

from env.hx11 import contract as C
from env.hx11 import kernel as K
from env.hx11.core import HX11Env
from learn.hx11_closed_loop import FRAME_SKIP, bits_of_action, load_state
from learn.hx11_data import D_MAX, D_MIN, MAX_BACK, ROT_MOVE, build, l0_split, load_match
from learn.hx11_policy import SetPolicy, StickyNoise, piece_rows, sample
from learn.hx11_selfplay import store_all, window_sp
from tools.hx11_d0 import PIECE

ROOT = Path(__file__).resolve().parent.parent
BOX_X, BOX_Y = C.BOX_X, C.BOX_Y


def behind(me, ball):
    """El jugador está detrás de la pelota del córner: del lado de la línea de fondo (95% de los humanos al patear)."""
    side = 1.0 if ball[0] > 0 else -1.0
    return (me[0] - ball[0]) * side > 0.0


KIND = C.CORNER          # tipo de pelota parada que se evalúa (--tipo): córner, tiro libre o saque de arco
TIPOS = {"corner": C.CORNER, "tiro_libre": C.FREE_KICK, "saque_arco": C.GOAL_KICK}


def corner_metrics(pos, ball, inp, kick, phase, score, team, t0, binv=None, bvel=None):
    """Métricas de un córner a partir de la colocación t0 (arreglos por tick: pos (T, 22, 2) px, ball (T, 2) px,
    inp (T, 22) teclas, kick (T, 22) patada efectiva o None, phase (T,), score (T, 2))."""
    T = len(ball)
    sl = np.arange(11) if team == 0 else np.arange(11, 22)
    # arco atacado: en el córner, el de la esquina; en las demás, el del rival (el rojo ataca hacia +x)
    side = (1.0 if ball[min(t0 + 10, T - 1), 0] > 0 else -1.0) if KIND == C.CORNER else (1.0 if team == 0 else -1.0)
    if kick is None:
        g = np.hypot(*(pos - ball[:, None]).transpose(2, 0, 1)) - C.PLAYER_R - C.BALL_R
        kb = (np.maximum(inp.astype(np.int64), 0) >> 4) & 1
        kick = (kb == 1) & (np.r_[[np.zeros(22, np.int64)], kb[:-1]] == 0) & (np.r_[[np.full(22, 9e9)], g[:-1]] <= 4.0)
    kk = np.flatnonzero(kick[t0:, sl].any(1))
    if not len(kk):
        return dict(sacado=False)
    tk = t0 + kk[0]
    kicker = sl[np.flatnonzero(kick[tk, sl])[0]]
    sy = 1.0 if ball[tk, 1] > 0 else -1.0
    rel = (pos[tk, kicker] - ball[tk]) * [side, sy]                      # marco normalizado: esquina en (+x, +y)
    v = (ball[min(tk + 2, T - 1)] - ball[tk]) * [side, sy]
    out = dict(sacado=True, espera=int(tk - t0), italiano=False, al_area=False, primer_toque="nadie", remate=False,
               gol=False, pateador=[round(float(rel[0]), 1), round(float(rel[1]), 1)], pateador_lugar=int(kicker),
               masa_inv=round(float(binv[tk]), 3) if binv is not None else None,
               rapidez=round(float(np.hypot(*bvel[min(tk + 2, T - 1)])), 2) if bvel is not None else None,
               direccion=round(float(np.degrees(np.arctan2(v[1], v[0]))), 1), afuera=False,
               avance=round(float((ball[min(tk + 120, T - 1), 0] - ball[tk, 0]) * side), 1))
    gap = np.hypot(*(pos - ball[:, None]).transpose(2, 0, 1)) - C.PLAYER_R - C.BALL_R
    for j in range(2, min(180, T - tk)):
        q = tk + j
        bx, by = ball[q]
        if abs(bx) > C.LINE_W + C.BALL_R and abs(by) > C.GOAL_HALF_H:
            out["italiano"] = KIND == C.CORNER
            out["afuera"] = True
            break
        if abs(by) > C.LATERAL_OUT_Y:
            out["afuera"] = True
            break
        if bx * side > BOX_X and abs(by) < BOX_Y:
            out["al_area"] = True
        touch = np.flatnonzero((gap[q] <= 0.6) | kick[q])
        touch = touch[touch != kicker] if j < 4 else touch
        if len(touch):
            w = touch[0]
            out["primer_toque"] = "atacante" if w in sl else "defensor"
            break
    for q in range(tk + 1, min(T, tk + 300)):
        if kick[q, sl].any() and ball[q, 0] * side > BOX_X and abs(ball[q, 1]) < BOX_Y:
            out["remate"] = True
            break
    s0 = score[tk].astype(np.int64)
    s1 = score[min(T - 1, tk + 600)].astype(np.int64)
    out["gol"] = bool(s1[team] > s0[team])
    return out


def summarize(rows):
    rr = [r for r in rows if r.get("sacado")]
    n = max(len(rr), 1)
    c = collections.Counter(r["primer_toque"] for r in rr)
    pk = np.array([r["pateador"] for r in rr]) if rr else np.zeros((0, 2))
    return dict(corners=len(rows), sacados=len(rr), espera_p50=float(np.median([r["espera"] for r in rr])) if rr else None,
                pateador_detras=round(float(np.mean(pk[:, 0] > 0)), 3) if len(pk) else None,
                masa_inv_p10_p50_p90=[round(float(np.percentile([r["masa_inv"] for r in rr], q)), 3) for q in (10, 50, 90)]
                if rr and rr[0].get("masa_inv") is not None else None,
                rapidez_p10_p50_p90=[round(float(np.percentile([r["rapidez"] for r in rr], q)), 2) for q in (10, 50, 90)]
                if rr and rr[0].get("rapidez") is not None else None,
                direccion_p50=float(np.median([r["direccion"] for r in rr])) if rr else None,
                italiano=round(sum(r["italiano"] for r in rr) / n, 3), al_area=round(sum(r["al_area"] for r in rr) / n, 3),
                afuera=round(sum(r.get("afuera", False) for r in rr) / n, 3),
                avance_p50=float(np.median([r["avance"] for r in rr])) if rr and "avance" in rr[0] else None,
                primer_toque_atacante=round(c["atacante"] / n, 3), primer_toque_defensor=round(c["defensor"] / n, 3),
                remate=round(sum(r["remate"] for r in rr) / n, 3), gol=round(sum(r["gol"] for r in rr) / n, 3))


def human(matches):
    rows = []
    for m in matches:
        st = np.flatnonzero((m.phase == PIECE) & (np.r_[-1, m.phase[:-1]] != PIECE) & (m.pkind == KIND))
        for t in st:
            if m.pteam[t] < 0 or t + 900 >= m.T:
                continue
            rows.append(corner_metrics(m.pos.astype(np.float64) / 10, m.ball.astype(np.float64) / 10, m.inp,
                                       m.kick.astype(bool), m.phase, m.score, int(m.pteam[t]), t,
                                       m.binv.astype(np.float64), m.bvel.astype(np.float64) / 1000))
    return rows


def good(r):
    """Córner bueno para la autoimitación: pateado desde atrás de la pelota, llega al área y no es italiano."""
    return bool(r.get("sacado") and r["pateador"][0] > 0 and r["al_area"] and not r["italiano"])


def bots(policy, matches, starts, H=900, n_env=16, seed=0, temp=0.7, renew_kick=0.33, every=FRAME_SKIP,
         collect=False, near_px=500.0, rule=False):
    """`rule`: en su propio córner el bot solo patea si está detrás de la pelota (regla de patada, 2026-10-09)."""
    rng = np.random.default_rng(seed)
    env = HX11Env(n_env, seed=seed)
    noise = StickyNoise(1.0, seed, renew_kick) if renew_kick < 1.0 else None
    rows = []
    keep_o, keep_e, keep_y = [], [], []
    for b0 in range(0, len(starts), n_env):
        chunk = starts[b0:b0 + n_env]
        W, D, teams = [], [], []
        for i, (mi, tp) in enumerate(chunk):
            m = matches[mi]
            w = window_sp(m, tp, H)
            w.score[:] = 0
            load_state(env, i, w, MAX_BACK)
            env.ri[i, K.RI_OFFSIDE_ON], env.ri[i, K.RI_FOULS_ON], env.ri[i, K.RI_PS_ON] = 1, 1, 1
            team = int(m.pteam[tp])
            sx, sy = (m.spot[tp].astype(np.float64) / 10).tolist()
            K._place(env.pos[i], env.vel[i], env.inv[i], env.mask[i], env.group[i], env.radius[i], env.grav[i],
                     env.ri[i], env.rf[i], KIND, team, sx, sy, env.fp, env.P, env.n_red, env.home, env.hold,
                     env.hold_cdf)
            store_all(env, i, w, MAX_BACK)
            W.append(w)
            D.append(rng.integers(D_MIN, D_MAX + 1, 22))
            teams.append(team)
        n = len(chunk)
        acts = np.zeros((n_env, 22), np.int64)
        buf = [collections.defaultdict(list) for _ in range(n)]     # lugar -> [(propia, entidades, acción)]
        for k in range(H):
            kk = MAX_BACK + k
            for i in range(n):
                W[i].inp[kk] = W[i].inp[kk - 1] if k > 0 else W[i].inp[kk]
            if k % every == 0:
                rows_o, rows_e, who = [], [], []
                for i in range(n):
                    w = W[i]
                    slots = np.nonzero(w.pid[kk] >= 0)[0]
                    o, e, _, _ = build(w, np.full(len(slots), kk), slots, D[i][slots], policy.extra)
                    rows_o.append(o)
                    rows_e.append(e)
                    who += [(i, int(s)) for s in slots]
                ot, et = np.concatenate(rows_o), np.concatenate(rows_e)
                with torch.no_grad():
                    lg = policy(torch.from_numpy(ot), torch.from_numpy(et))
                act = sample(lg, temp, kick_u=noise.get_kick([(b0, i, s) for i, s in who],
                                                             piece_rows([env.ri[i, K.RI_MODE] for i, s in who]))
                             if noise else None)
                for j, ((i, s), a) in enumerate(zip(who, act)):
                    a = int(a)
                    if collect and env.ri[i, K.RI_MODE] == K.PIECE and (s >= 11) == (teams[i] == 1) and \
                            np.hypot(*(env.pos[i, 0] - env.pos[i, env.fp + s])) <= near_px:
                        buf[i][s].append((ot[j], et[j], a))     # acción en coordenadas del equipo
                    if rule and a >= 9 and env.ri[i, K.RI_MODE] == K.PIECE and env.ri[i, K.RI_KIND] == C.CORNER and \
                            (s >= 11) == (env.ri[i, K.RI_TEAM] == 1) and \
                            not behind(env.pos[i, env.fp + s], env.pos[i, 0]):
                        a -= 9                                  # regla: sin patada desde el lado de la cancha
                    if s >= 11:
                        a = int(ROT_MOVE[a % 9] + 9 * (a // 9))
                    acts[i, s] = a
                    W[i].inp[kk, s] = bits_of_action(np.array([a]))[0]
            env.step(acts, frames=1)
            for i in range(n):
                store_all(env, i, W[i], kk + 1)
                W[i].score[kk + 1] = env.ri[i, [K.RI_SCORE0, K.RI_SCORE1]]
        for i in range(n):
            w = W[i]
            r = corner_metrics(w.pos.astype(np.float64) / 10, w.ball.astype(np.float64) / 10, w.inp, None,
                               w.phase, w.score, teams[i], MAX_BACK, w.binv.astype(np.float64),
                               w.bvel.astype(np.float64) / 1000)
            rows.append(r)
            if collect and good(r) and buf[i].get(r["pateador_lugar"]):
                b = buf[i][r["pateador_lugar"]]
                keep_o.append(np.stack([x[0] for x in b]))
                keep_e.append(np.stack([x[1] for x in b]))
                keep_y.append(np.array([x[2] for x in b], np.int64))
    if collect:
        data = (dict(own=np.concatenate(keep_o).astype(np.float32), ents=np.concatenate(keep_e).astype(np.float32),
                     y=np.concatenate(keep_y)) if keep_y else None)
        return rows, data
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", default="dev", choices=["train", "dev", "test"])
    ap.add_argument("--partidos", type=int, default=40)
    ap.add_argument("--corners", type=int, default=300)
    ap.add_argument("--n-env", type=int, default=16)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--renovar-patada", type=float, default=0.33)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default=None)
    ap.add_argument("--colectar", default=None, help="npz con las decisiones del que patea en los córners buenos")
    ap.add_argument("--sin-humanos", action="store_true")
    ap.add_argument("--regla", action="store_true", help="en su propio córner, solo patea desde atrás de la pelota")
    ap.add_argument("--tipo", default="corner", choices=list(TIPOS), help="pelota parada que se evalúa")
    a = ap.parse_args()
    global KIND
    KIND = TIPOS[a.tipo]
    torch.set_num_threads(a.threads)
    rng = np.random.default_rng(a.seed)
    names = sorted(rng.permutation(l0_split()[a.split])[:a.partidos].tolist())
    modos = json.loads((ROOT / "data" / "hx11" / "modos.json").read_text(encoding="utf-8"))
    matches = [load_match(nm, modos) for nm in names]
    hum = [] if a.sin_humanos else human(matches)
    starts = []
    for mi, m in enumerate(matches):
        st = np.flatnonzero((m.phase == PIECE) & (np.r_[-1, m.phase[:-1]] != PIECE) & (m.pkind == KIND))
        starts += [(mi, int(t)) for t in st if MAX_BACK + 1 <= t < m.T - 901 and m.pteam[t] >= 0]
    rng.shuffle(starts)
    starts = starts[:a.corners]
    ck = torch.load(a.model, map_location="cpu", weights_only=False)
    policy = SetPolicy(**ck["config"])
    policy.load_state_dict(ck["state"])
    policy.eval()
    t0 = time.time()
    out = bots(policy, matches, starts, n_env=a.n_env, seed=a.seed, temp=a.temp, renew_kick=a.renovar_patada,
               collect=bool(a.colectar), rule=a.regla)
    bot, data = out if a.colectar else (out, None)
    res = dict(modelo=a.model, split=a.split, partidos=len(names), bots=summarize(bot),
               humanos=summarize(hum) if hum else None, segundos=round(time.time() - t0))
    if a.colectar:
        res["buenos"] = int(sum(good(r) for r in bot))
        res["muestras"] = int(len(data["y"])) if data else 0
        if data:
            np.savez_compressed(a.colectar, **data)
    if a.out:
        Path(a.out + ".json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))




def human_takers(names, kind=C.CORNER, stride=3, seed=0, modos=None):
    """Decisiones de los humanos que sacaron la pelota parada `kind`, desde la colocación hasta su patada (propia,
    entidades, acción del equipo; retardo sorteado como en el entrenamiento). Para darles más peso en un ajuste corto:
    en el entrenamiento normal quedan diluidas entre las de juego abierto."""
    rng = np.random.default_rng(seed)
    keep_o, keep_e, keep_y = [], [], []
    for nm in names:
        m = load_match(nm, modos)
        st = np.flatnonzero((m.phase == PIECE) & (np.r_[-1, m.phase[:-1]] != PIECE) & (m.pkind == kind))
        for t0 in st:
            team = int(m.pteam[t0])
            if team < 0 or t0 < MAX_BACK + 1:
                continue
            sl = np.arange(11) if team == 0 else np.arange(11, 22)
            kk = np.flatnonzero(m.kick[t0:t0 + 1500][:, sl].any(1))
            if not len(kk):
                continue
            tk = t0 + kk[0]
            s = int(sl[np.flatnonzero(m.kick[tk, sl])[0]])
            ts = np.arange(t0, tk + 1, stride)
            ts = ts[(m.pid[ts, s] == m.pid[tk, s]) & (m.inp[ts, s] >= 0) & (m.pid[ts - MAX_BACK, s] == m.pid[tk, s])]
            if not len(ts):
                continue
            d = rng.integers(D_MIN, D_MAX + 1, len(ts))
            o, e, y, _ = build(m, ts, np.full(len(ts), s), d, False)
            keep_o.append(o); keep_e.append(e); keep_y.append(y)
    return dict(own=np.concatenate(keep_o).astype(np.float32), ents=np.concatenate(keep_e).astype(np.float32),
                y=np.concatenate(keep_y).astype(np.int64))


def main_humanos():
    ap = argparse.ArgumentParser(description="decisiones de los humanos que sacaron córners (para un ajuste corto)")
    ap.add_argument("--split-file", default="runs/hx11_l0/split.json")
    ap.add_argument("--tipo", default="corner", choices=["corner", "lateral"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sp = json.loads((ROOT / a.split_file).read_text(encoding="utf-8"))
    modos = json.loads((ROOT / "data" / "hx11" / "modos.json").read_text(encoding="utf-8"))
    data = human_takers(sp["train"], C.CORNER if a.tipo == "corner" else C.LATERAL, modos=modos)
    np.savez_compressed(a.out, **data)
    print(json.dumps(dict(muestras=int(len(data["y"])), patadas=int((data["y"] >= 9).sum()), out=a.out)))


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "humanos":
        sys.argv.pop(1)
        main_humanos()
    else:
        main()
