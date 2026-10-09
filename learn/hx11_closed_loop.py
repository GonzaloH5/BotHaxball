"""Banco de lazo cerrado no reactivo para L0 (puerta P3 de la revisión de la auditoría).

Desde un estado real de un partido de desarrollo o prueba (juego abierto), el simulador `env/hx11` arranca con las
posiciones, velocidades, masa y comba de la pelota grabadas. Los otros 21 jugadores juegan sus **teclas grabadas**
(no reaccionan al bot, como los agentes "non-reactive" de nuPlan; la física sí los empuja). El jugador evaluado juega
con la política, que decide cada tick con un retardo d sobre su propia historia simulada (`learn/hx11_data.build`).

Se compara con el humano de ese lugar en la misma ventana:
* desvío de posición respecto del humano a +1, +3 y +5 s;
* distancia media a la pelota (bot en el simulador, humano en la grabación);
* patadas y toques de pelota por minuto (la pasividad de X4: 0,55 contra 3,2 patadas por minuto);
* distancia media al ancla del puesto (centroide de campo + ancla del jugador, con o sin la pelota);
* cambios de tecla por segundo.

**Control del banco:** con `--policy humano` el lugar evaluado también juega sus teclas grabadas. Lo que diverja ahí
es error del banco (estado del árbitro, orden de discos), no del bot; se informa al lado.

Las ventanas se cortan en la primera parada (salida, falta, gol) del simulador o de la grabación. Los modos de falta y
offside del árbitro van apagados en estas ventanas cortas: la puerta mide movimiento y pelota, no reglas.

  python -m learn.hx11_closed_loop --model runs/hx11_l0/mini/model.pt --starts 400 --out runs/hx11_l0/mini/lazo.json
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import torch

from env.hx11 import kernel as K
from env.hx11.core import HX11Env
from learn.hx11_data import (D_MAX, D_MIN, HOLD_REP, MAX_BACK, MOVE_DX, MOVE_DY, ROT_MOVE, Match, build, l0_split,
                             load_match, move_of_input)
from learn.hx11_policy import SetPolicy, sample
from tools.hx11_d0 import OPEN

ROOT = Path(__file__).resolve().parent.parent
H = 300                       # 5 s
FRAME_SKIP = 3                # una decisión cada 3 ticks, como X4 (`env/rs4z`, `deploy/bot.js`): la historia propia de
                              # la observación está a 3, 6, … ticks; decidir cada tick la deja sin coherencia (45
                              # cambios de tecla por segundo contra 2,7 de los humanos, medido el 2026-10-08)
FAR = np.array([0.0, 5000.0])
TOUCH = 0.6


def bits_of_action(a):
    mv, kick = a % 9, a // 9
    dx, dy = MOVE_DX[mv], MOVE_DY[mv]
    return ((dx > 0) * 8 + (dx < 0) * 4 + (dy > 0) * 2 + (dy < 0) * 1 + kick * 16).astype(np.int8)


def action_of_bits(b):
    b = np.maximum(b.astype(np.int64), 0)
    return move_of_input(b) + 9 * ((b >> 4) & 1)


def starts(m: Match, owner, n, rng, slots=None, near=False):
    """(t0, lugar) con la ventana [t0, t0 + H] de juego abierto en la grabación y el jugador presente.

    `near`: el lugar es el jugador de campo más cercano a la pelota en t0 (las situaciones donde un humano interviene;
    al azar, el jugador casi nunca toca la pelota en 5 s y las patadas se cuentan de a una)."""
    T = m.T
    op = m.phase == OPEN
    # largo de juego abierto hacia adelante desde cada tick
    run = np.zeros(T + 1, np.int64)
    for t in range(T - 1, -1, -1):
        run[t] = run[t + 1] + 1 if op[t] else 0
    cand = np.nonzero((run[:T] >= H + 1) & (np.arange(T) >= MAX_BACK))[0]
    out = []
    for t0 in rng.choice(cand, size=min(n * 3, len(cand)), replace=False):
        ss = np.nonzero((m.pid[t0] >= 0) & (m.pid[t0] == m.pid[t0 + H]) & (m.pid[t0] == m.pid[t0 - MAX_BACK]))[0]
        if slots is not None:
            ss = np.intersect1d(ss, slots)
        if near and len(ss):
            gkp = np.r_[np.full(11, m.gk[t0, 0]), np.full(11, m.gk[t0, 1])]
            ss = ss[m.pid[t0, ss] != gkp[ss]]
            if len(ss):
                d = np.hypot(*(m.pos[t0, ss].astype(np.float64) - m.ball[t0].astype(np.float64)).T)
                ss = ss[[int(np.argmin(d))]]
        if len(ss):
            out.append((int(t0), int(rng.choice(ss))))
        if len(out) >= n:
            break
    return out


def window(m: Match, t0):
    """Copia mutable de los arreglos de la ventana [t0 − MAX_BACK, t0 + H] (índices desplazados)."""
    a, b = t0 - MAX_BACK, t0 + H + 1
    fields = {f.name: getattr(m, f.name) for f in dataclasses.fields(m)}
    w = {k: (v[a:b].copy() if isinstance(v, np.ndarray) and v.shape[:1] == (m.T,) else
             dict(v) if isinstance(v, dict) else v) for k, v in fields.items()}
    w["T"] = b - a
    return Match(**w), a


def load_state(env, i, w: Match, k):
    """Estado grabado del tick k de la ventana en el entorno i (juego abierto, árbitro sin faltas ni offside)."""
    fp = env.fp
    on = w.pid[k] >= 0
    env.pos[i, 0] = w.ball[k].astype(np.float64) / 10
    env.vel[i, 0] = w.bvel[k].astype(np.float64) / 1000
    env.pos[i, fp:] = np.where(on[:, None], w.pos[k].astype(np.float64) / 10, FAR)
    env.vel[i, fp:] = np.where(on[:, None], w.vel[k].astype(np.float64) / 1000, 0.0)
    env.radius[i] = env.base_radius
    env.radius[i, 1:3] = 0.0
    env.radius[i, fp:] = np.where(on, env.base_radius[fp:], 0.0)
    env.inv[i] = env.base_inv
    env.inv[i, 0] = float(w.binv[k])
    env.mask[i] = env.base_mask
    env.mask[i, fp:] = np.where(on, env.base_mask[fp:], 0)
    env.group[i] = env.base_group
    env.grav[i] = w.bg[k].astype(np.float64)
    # patada ya usada: la tecla sigue apretada desde una patada efectiva (hay que soltarla para volver a patear)
    held = (np.maximum(w.inp[:k + 1].astype(np.int64), 0) >> 4) & 1
    cancel = np.zeros(22, bool)
    for p in range(22):
        j = k
        while j >= 0 and held[j, p]:
            if w.kick[j, p]:
                cancel[p] = True
                break
            j -= 1
    env.kick_cancel[i] = cancel
    env.ri[i] = 0
    env.rf[i] = 0.0
    env.fs[i] = -1
    env.fl[i] = -1
    r = env.ri
    r[i, K.RI_MODE] = K.OPEN
    r[i, K.RI_LAST] = -1
    r[i, K.RI_PS_T] = -1
    r[i, K.RI_AFTER_T] = -1
    r[i, K.RI_LAT_ARMED] = 1
    r[i, K.RI_PREV_IN_Y] = 1
    r[i, K.RI_OFFSIDE_ON], r[i, K.RI_FOULS_ON], r[i, K.RI_PS_ON] = 0, 0, 1


def store(env, i, w: Match, k, acts):
    fp = env.fp
    on = w.pid[k] >= 0
    w.ball[k] = np.round(env.pos[i, 0] * 10).astype(np.int16)
    w.bvel[k] = np.round(env.vel[i, 0] * 1000).clip(-32767, 32767).astype(np.int16)
    w.pos[k] = np.where(on[:, None], np.round(env.pos[i, fp:] * 10), 0).clip(-32767, 32767).astype(np.int16)
    w.vel[k] = np.where(on[:, None], np.round(env.vel[i, fp:] * 1000), 0).clip(-32767, 32767).astype(np.int16)
    w.binv[k] = env.inv[i, 0]
    w.bg[k] = env.grav[i]
    w.phase[k] = env.ri[i, K.RI_MODE]


def rollout(policy, matches, owners, starts_list, n_env=64, seed=0, human=False, swap=None, every=FRAME_SKIP,
            temp_move=1.0, hyst=0.0, hold_scale=1.0, delay=None):
    """Corre las ventanas en lotes de n_env entornos; devuelve una fila de métricas por ventana.

    `swap` (vocabulario de `tools.hx11_pos`): al bot se le pide otro puesto, de otra línea (E8). Se pisa su ancla y su
    línea en la ventana y se mide la distancia al ancla pedida y a la propia."""
    rng = np.random.default_rng(seed)
    env = HX11Env(n_env, seed=seed)
    rows = []
    for b0 in range(0, len(starts_list), n_env):
        chunk = starts_list[b0:b0 + n_env]
        W, logs = [], []
        req = []
        for i, (mi, t0, s) in enumerate(chunk):
            w, a = window(matches[mi], t0)
            pid = int(w.pid[MAX_BACK, s])
            own_line = w.line.get(pid, 3)
            r = None
            if swap is not None and own_line != 0:
                cands = [k for k, v in swap.items() if v["linea_idx"] not in (0, own_line)]
                r = cands[int(rng.integers(len(cands)))]
                for h in (0, 1):
                    w.anchor[(pid, h)] = np.array(swap[r]["ancla"], np.float32)
                w.line[pid] = swap[r]["linea_idx"]
            req.append(r)
            dd = int(rng.integers(D_MIN, D_MAX + 1))
            W.append((w, a, s, dd if delay is None else int(delay), mi, t0))
            logs.append(window(matches[mi], t0)[0])
            load_state(env, i, w, MAX_BACK)
        n = len(chunk)
        alive = np.ones(n, bool)
        end = np.full(n, H)
        bot_act = np.zeros((n, H), np.int64)
        held = np.zeros(n, np.int64)
        kicks_eff = np.zeros(n, np.int64)
        cancel_prev = np.array([bool(env.kick_cancel[i, W[i][2]]) for i in range(n)])
        next_dec = np.zeros(n, np.int64)
        use_hold = getattr(policy, "hold", False)
        for k in range(H):
            kk = MAX_BACK + k
            acts = np.zeros((n_env, 22), np.int64)
            for i, (w, a, s, d, mi, t0) in enumerate(W):
                acts[i] = action_of_bits(logs[i].inp[kk])
            deciding = [i for i in range(len(W)) if (k >= next_dec[i] if use_hold else k % every == 0)]
            if not human and deciding:
                own, ents = [], []
                for i in deciding:
                    w, a, s, d, mi, t0 = W[i]
                    o, e, _, _ = build(w, np.array([kk]), np.array([s]), np.array([d]), policy.extra)
                    own.append(o)
                    ents.append(e)
                with torch.no_grad():
                    ot, et = torch.from_numpy(np.concatenate(own)), torch.from_numpy(np.concatenate(ents))
                    if use_hold:
                        lg, hl = policy.forward_hold(ot, et)
                        hc = torch.distributions.Categorical(logits=hl).sample().numpy()
                    else:
                        lg = policy(ot, et)
                    prev_team = [int(held[i]) if W[i][2] < 11 else int(ROT_MOVE[held[i] % 9] + 9 * (held[i] // 9))
                                 for i in deciding]
                    a_team = sample(lg, temp_move, prev=prev_team if k > 0 else None, hyst=hyst)
                for j, i in enumerate(deciding):
                    s = W[i][2]
                    act = int(a_team[j])
                    if s >= 11:
                        act = int(ROT_MOVE[act % 9] + 9 * (act // 9))
                    held[i] = act
                    next_dec[i] = k + (max(every, int(round(HOLD_REP[hc[j]] * hold_scale))) if use_hold else every)
            if not human:
                for i, (w, a, s, d, mi, t0) in enumerate(W):
                    acts[i, s] = held[i]
                    w.inp[kk, s] = bits_of_action(np.array([held[i]]))[0]
            for i, (w, a, s, d, mi, t0) in enumerate(W):
                bot_act[i, k] = acts[i, s]
            env.step(acts, frames=1)
            for i, (w, a, s, d, mi, t0) in enumerate(W):
                c = bool(env.kick_cancel[i, s])
                if alive[i] and c and not cancel_prev[i]:
                    kicks_eff[i] += 1           # la física ejecutó la patada (la tecla queda "usada")
                cancel_prev[i] = c
                if not alive[i]:
                    continue
                store(env, i, w, kk + 1, acts[i])
                if env.ri[i, K.RI_MODE] != K.OPEN:
                    alive[i] = False
                    end[i] = k + 1
            if not alive.any():
                break
        for i, (w, a, s, d, mi, t0) in enumerate(W):
            rows.append(metrics(w, logs[i], s, int(end[i]), bot_act[i, :end[i]], matches[mi], t0, owners[mi]))
            mins = int(end[i]) / 3600
            ks_ = np.arange(MAX_BACK + 1, MAX_BACK + int(end[i]) + 1)
            rows[-1]["patadas_efectivas_min_bot"] = kicks_eff[i] / mins
            rows[-1]["patadas_efectivas_min_humano"] = float(logs[i].kick[ks_, s].sum()) / mins
            if req[i] is not None:
                pid = int(w.pid[MAX_BACK, s])
                rows[-1]["dist_ancla_pedida_bot"] = zone_dist(w, s, int(end[i]), w.anchor[(pid, 0)], owners[mi], t0)
                rows[-1]["dist_ancla_pedida_humano"] = zone_dist(logs[i], s, int(end[i]), w.anchor[(pid, 0)],
                                                                 owners[mi], t0)
                rows[-1]["puesto_pedido"] = req[i]
            pid0 = int(matches[mi].pid[t0, s])
            from learn.hx11_data import LINES
            rows[-1].update(partido=matches[mi].name, t0=t0, lugar=s, retardo=d,
                            linea=LINES[matches[mi].line.get(pid0, 3)])
    return rows


def metrics(w: Match, log: Match, s, end, acts, m: Match, t0, owner):
    k0 = MAX_BACK
    ks = np.arange(k0 + 1, k0 + end + 1)
    p_bot = w.pos[ks, s].astype(np.float64) / 10
    p_hum = log.pos[ks, s].astype(np.float64) / 10
    b_bot = w.ball[ks].astype(np.float64) / 10
    b_hum = log.ball[ks].astype(np.float64) / 10
    dev = np.hypot(*(p_bot - p_hum).T)
    out = dict(ticks=int(end))
    for sec in (1, 3, 5):
        out[f"desvio_{sec}s"] = float(dev[sec * 60 - 1]) if end >= sec * 60 else None
    out["vel_bot"] = float(np.hypot(*np.diff(p_bot, axis=0).T).mean()) if end > 1 else None
    out["vel_humano"] = float(np.hypot(*np.diff(p_hum, axis=0).T).mean()) if end > 1 else None
    out["dist_pelota_bot"] = float(np.hypot(*(p_bot - b_bot).T).mean())
    out["dist_pelota_humano"] = float(np.hypot(*(p_hum - b_hum).T).mean())
    from env.hx11 import contract as HC
    gap_bot = np.hypot(*(p_bot - b_bot).T) - HC.PLAYER_R - HC.BALL_R
    gap_hum = np.hypot(*(p_hum - b_hum).T) - HC.PLAYER_R - HC.BALL_R
    # patada: la tecla se aprieta (flanco) con la pelota a <= 4 px del borde; la misma regla para los dos.
    # acts[j] se aplica en el paso que produce ks[j]: el flanco se mira contra la distancia del estado anterior.
    gb_prev = np.r_[np.hypot(*(w.pos[k0, s] / 10 - w.ball[k0] / 10)) - HC.PLAYER_R - HC.BALL_R, gap_bot[:-1]]
    gh_prev = np.r_[np.hypot(*(log.pos[k0, s] / 10 - log.ball[k0] / 10)) - HC.PLAYER_R - HC.BALL_R, gap_hum[:-1]]
    kb = (acts >= 9).astype(np.int64)
    kb_prev = np.r_[(max(int(log.inp[k0, s]), 0) >> 4) & 1, kb[:-1]]
    kick_bot = int(((kb == 1) & (kb_prev == 0) & (gb_prev <= 4.0)).sum())
    hk = (np.maximum(log.inp[ks, s].astype(np.int64), 0) >> 4) & 1
    hk_prev = np.r_[(max(int(log.inp[k0, s]), 0) >> 4) & 1, hk[:-1]]
    kick_hum = int(((hk == 1) & (hk_prev == 0) & (gh_prev <= 4.0)).sum())
    mins = end / 3600
    out["patadas_min_bot"] = kick_bot / mins
    out["patadas_min_humano"] = kick_hum / mins
    out["toques_min_bot"] = float((gap_bot <= TOUCH).sum()) / 60 / mins
    out["toques_min_humano"] = float((gap_hum <= TOUCH).sum()) / 60 / mins
    out["cambios_s_bot"] = float((np.diff(acts) != 0).sum()) / (end / 60)
    ha = action_of_bits(log.inp[ks, s])
    out["cambios_s_humano"] = float((np.diff(ha) != 0).sum()) / (end / 60)
    # distancia al ancla propia del jugador (la grabada, aunque se le haya pedido otro puesto)
    pid = int(m.pid[t0, s])
    anc = m.anchor.get((pid, int(t0 >= m.T // 2)), np.zeros(4, np.float32))

    out["dist_ancla_bot"] = zone_dist(w, s, end, anc, owner, t0)
    out["dist_ancla_humano"] = zone_dist(log, s, end, anc, owner, t0)
    return out


def zone_dist(arr: Match, s, end, anc, owner, t0):
    """Distancia media del jugador s a su zona: centroide de campo de su equipo + ancla (con o sin la pelota)."""
    ks = np.arange(MAX_BACK + 1, MAX_BACK + end + 1)
    team = 0 if s < 11 else 1
    att = 1.0 if team == 0 else -1.0
    sl = np.arange(team * 11, team * 11 + 11)
    own_ball = owner[t0 + 1:t0 + end + 1]
    with_ball = (own_ball >= 0) & ((own_ball // 11) == team)
    q = arr.pos[ks][:, sl].astype(np.float64) / 10 * att
    f = (arr.pid[ks][:, sl] >= 0) & (arr.pid[ks][:, sl] != arr.gk[ks, team][:, None])
    cen = (q * f[..., None]).sum(1) / np.maximum(f.sum(1), 1)[:, None]
    target = cen + np.where(with_ball[:, None], anc[:2], anc[2:])
    me = arr.pos[ks, s].astype(np.float64) / 10 * att
    return float(np.hypot(*(me - target).T).mean())


def summarize(rows):
    keys = sorted({k for r in rows for k, v in r.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
                  - {"t0", "lugar", "retardo"})
    out = {}
    for k in keys:
        v = np.array([r[k] for r in rows if r.get(k) is not None], np.float64)
        if len(v):
            out[k] = dict(media=round(float(v.mean()), 3), p10=round(float(np.percentile(v, 10)), 3),
                          p50=round(float(np.percentile(v, 50)), 3), p90=round(float(np.percentile(v, 90)), 3))
            if "_min_" in k or "_s_" in k:
                # tasas de conteo: total sobre el tiempo total. La media de las tasas por ventana la dominan las
                # ventanas cortadas por una salida (una patada en medio segundo = 120 por minuto; 2026-10-08)
                T = np.array([r["ticks"] for r in rows if r.get(k) is not None], np.float64)
                out[k]["media"] = round(float((v * T).sum() / T.sum()), 3)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=None)
    ap.add_argument("--policy", default="modelo", choices=("modelo", "humano"))
    ap.add_argument("--split", default="dev")
    ap.add_argument("--matches", type=int, default=10)
    ap.add_argument("--starts", type=int, default=200)
    ap.add_argument("--n-env", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--intercambio", action="store_true", help="E8: pedirle al bot un puesto de otra línea")
    ap.add_argument("--every", type=int, default=FRAME_SKIP, help="ticks entre decisiones del bot")
    ap.add_argument("--temp", type=float, default=1.0, help="temperatura de la dirección (la patada va sin temperatura)")
    ap.add_argument("--cerca", action="store_true", help="evaluar al jugador de campo más cercano a la pelota")
    ap.add_argument("--hyst", type=float, default=0.0, help="histéresis de la dirección (0 = sin)")
    ap.add_argument("--hold-scale", type=float, default=1.0, help="escala de la duración sorteada (cabeza de duración)")
    ap.add_argument("--delay", type=int, default=None, help="retardo fijo del bot (por defecto, sorteado en [8, 36])")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(4)
    modos = json.loads((ROOT / "data" / "hx11" / "modos.json").read_text(encoding="utf-8"))
    rng = np.random.default_rng(a.seed)
    names = sorted(rng.permutation(l0_split()[a.split])[:a.matches].tolist())
    matches = [load_match(n, modos) for n in names]
    owners = [np.load(ROOT / "data" / "hx11" / "d1" / f"{n}.npz")["owner"] for n in names]
    per = max(1, a.starts // len(matches))
    sl = [(mi, t0, s) for mi, m in enumerate(matches) for t0, s in starts(m, owners[mi], per, rng, near=a.cerca)]
    policy = None
    if a.policy == "modelo":
        ck = torch.load(a.model, map_location="cpu", weights_only=False)
        policy = SetPolicy(**ck["config"])
        policy.load_state_dict(ck["state"])
        policy.eval()
    t0 = time.time()
    swap = (json.loads((ROOT / "reports" / "hx11" / "pos_vocab.json").read_text(encoding="utf-8"))
            if a.intercambio else None)
    rows = rollout(policy, matches, owners, sl, n_env=a.n_env, seed=a.seed, human=a.policy == "humano", swap=swap,
                   every=a.every, temp_move=a.temp, hyst=a.hyst, hold_scale=a.hold_scale, delay=a.delay)
    res = dict(politica=a.policy, modelo=a.model, cada=a.every, temp=a.temp, hyst=a.hyst, cerca=a.cerca, partidos=names, ventanas=len(rows), resumen=summarize(rows),
               segundos=round(time.time() - t0))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(dict(res, filas=rows), indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "partidos"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
