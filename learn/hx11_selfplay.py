"""Lazo cerrado reactivo: L0 jugando los 22 lugares en el simulador (primer prototipo jugable y primera mirada a L1).

Arranca desde estados reales de partidos de desarrollo (saques iniciales y juego abierto), con la historia previa de la
grabación, y desde ahí los 22 juegan con la política: cada uno decide cada tick con su retardo (sorteado por jugador en
[8, 36]) sobre la historia simulada. El árbitro va como en la sala del usuario (offside, faltas y powershot prendidos).

Se mide lo mismo en la simulación y en ventanas humanas del mismo largo de los mismos partidos (la referencia de la
puerta de realismo, P5; esto es solo un primer vistazo, no la puerta):
* patadas por jugador y por minuto, toques de pelota por jugador y por minuto;
* goles cada 10 min; parte del tiempo con la pelota en juego;
* largo y ancho del equipo (jugadores de campo), distancia media de cada jugador a su zona (centroide + ancla);
* cambios de posesión por minuto (último toque de otro equipo).

Guarda las posiciones (sin nombres) para verlo con `learn/hx11_replay_html.py`.

  python -m learn.hx11_selfplay --model runs/hx11_l0/mini/model.pt --games 8 --seconds 60 --out runs/hx11_l0/mini/self
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from env.hx11 import contract as HC
from env.hx11 import kernel as K
from env.hx11.core import HX11Env
from learn.hx11_closed_loop import FRAME_SKIP, bits_of_action, load_state
from learn.hx11_data import D_MAX, D_MIN, HOLD_REP, MAX_BACK, ROT_MOVE, Match, build, l0_split, load_match
from learn.hx11_policy import Ensemble, SetPolicy, StickyNoise, near_rows, piece_rows, policy_logits, sample
from tools.hx11_d0 import KICKOFF, OPEN

ROOT = Path(__file__).resolve().parent.parent


def window_sp(m: Match, t0, H):
    """Ventana [t0 − MAX_BACK, t0 + H] con la historia grabada; lo que pasa del final del partido se rellena."""
    a = t0 - MAX_BACK
    n = MAX_BACK + H + 1
    out = {}
    for f in m.__dataclass_fields__:
        v = getattr(m, f)
        if isinstance(v, np.ndarray) and v.shape[:1] == (m.T,):
            w = np.repeat(v[min(m.T - 1, a + n - 1):min(m.T, a + n)], n, axis=0)
            k = min(n, m.T - a)
            w[:k] = v[a:a + k]
            out[f] = w.copy()
        else:
            out[f] = dict(v) if isinstance(v, dict) else v
    out["T"] = n
    return Match(**out)


def store_all(env, i, w: Match, k):
    fp = env.fp
    on = w.pid[k] >= 0
    w.ball[k] = np.round(env.pos[i, 0] * 10).clip(-32767, 32767).astype(np.int16)
    w.bvel[k] = np.round(env.vel[i, 0] * 1000).clip(-32767, 32767).astype(np.int16)
    w.pos[k] = np.where(on[:, None], np.round(env.pos[i, fp:] * 10), 0).clip(-32767, 32767).astype(np.int16)
    w.vel[k] = np.where(on[:, None], np.round(env.vel[i, fp:] * 1000), 0).clip(-32767, 32767).astype(np.int16)
    w.binv[k] = env.inv[i, 0]
    w.bg[k] = env.grav[i]
    mode = int(env.ri[i, K.RI_MODE])
    w.phase[k] = mode
    w.pkind[k] = env.ri[i, K.RI_KIND] if mode in (K.PIECE, K.OUT_PENDING) else 0
    w.pteam[k] = env.ri[i, K.RI_TEAM] if mode in (K.PIECE, K.OUT_PENDING, K.KICKOFF) else -1
    w.pticks[k] = min(int(env.ri[i, K.RI_TICKS]), 32767)
    w.spot[k] = np.round(env.rf[i, [K.RF_SPOT_X, K.RF_SPOT_Y]] * 10).astype(np.int16)
    w.offside[k] = True


def game_metrics(w: Match, k0, k1):
    """Métricas de una ventana [k0, k1) de un partido (simulado o grabado)."""
    ks = np.arange(k0, k1)
    mins = len(ks) / 3600
    pos = w.pos[ks].astype(np.float64) / 10
    ball = w.ball[ks].astype(np.float64) / 10
    on = w.pid[ks] >= 0
    gap = np.hypot(*(pos - ball[:, None]).transpose(2, 0, 1)) - HC.PLAYER_R - HC.BALL_R
    gap[~on] = np.inf
    kk = (np.maximum(w.inp[ks].astype(np.int64), 0) >> 4) & 1
    kprev = (np.maximum(w.inp[ks - 1].astype(np.int64), 0) >> 4) & 1
    gprev = np.vstack([gap[:1], gap[:-1]])
    kicks = ((kk == 1) & (kprev == 0) & (gprev <= 4.0)).sum()
    touch = gap <= 0.6
    n_on = max(on.sum(1).mean(), 1)
    # posesión por último toque
    last = -1
    changes = 0
    poss = np.full(len(ks), -1)
    for t in range(len(ks)):
        tt = np.nonzero(touch[t])[0]
        if len(tt):
            team = int(tt[0] >= 11)
            if last >= 0 and team != last:
                changes += 1
            last = team
        poss[t] = last
    goals = int(np.clip(np.diff(w.score[ks].astype(np.int64), axis=0), 0, None).sum())
    shape = []
    for team in (0, 1):
        sl = slice(team * 11, team * 11 + 11)
        q = pos[:, sl]
        f = on[:, sl] & (w.pid[ks][:, sl] != w.gk[ks, team][:, None])
        xs = np.where(f, q[..., 0], np.nan)
        ys = np.where(f, q[..., 1], np.nan)
        shape.append((np.nanmax(xs, 1) - np.nanmin(xs, 1), np.nanmax(ys, 1) - np.nanmin(ys, 1)))
    pos_m = positioning(w, ks, pos, ball, on, poss)
    open_frac = float((w.phase[ks] == OPEN).mean())
    sp = np.hypot(*np.diff(pos, axis=0).transpose(2, 0, 1))[on[1:] & on[:-1]]
    sp = sp[sp < 10]
    nearest = np.where(on, np.hypot(*(pos - ball[:, None]).transpose(2, 0, 1)), np.inf).min(1)
    return dict(patadas_jugador_min=float(kicks / n_on / mins), toques_jugador_min=float(touch.sum() / 60 / n_on / mins),
                cambios_posesion_min=changes / mins, goles_10min=goals / mins * 10, largo_equipo=float(np.nanmean([s[0] for s in shape])),
                ancho_equipo=float(np.nanmean([s[1] for s in shape])), juego_abierto=open_frac,
                vel_media=float(sp.mean()), mas_cercano_pelota_p50=float(np.median(nearest)), **pos_m)


def positioning(w: Match, ks, pos, ball, on, poss):
    """Posicionamiento en juego abierto, por equipo (coordenadas del equipo, ataca hacia +x):
    * dist_zona: distancia de cada jugador de campo a su zona (centroide de campo + ancla, con o sin la pelota según el
      último toque), la misma zona que ve la red;
    * cerca_pelota_150: jugadores de campo del equipo a <= 150 px de la pelota (amontonamiento);
    * vecino_mas_cercano: distancia al compañero de campo más cercano (separación);
    * bloque_dx: |x del centroide de campo − x de la pelota| (qué tan lejos queda el bloque de la jugada);
    * persigue: jugadores de campo del equipo a <= 300 px de la pelota que van hacia ella (> 1 px/tick)."""
    op = w.phase[ks] == OPEN
    if op.sum() < 60:
        return dict(dist_zona=np.nan, cerca_pelota_150=np.nan, vecino_mas_cercano=np.nan, bloque_dx=np.nan,
                    persigue=np.nan)
    vel = w.vel[ks][op].astype(np.float64) / 1000
    zd, near, mate, blk, chase = [], [], [], [], []
    for team in (0, 1):
        sl = np.arange(team * 11, team * 11 + 11)
        att = 1.0 if team == 0 else -1.0
        q = pos[op][:, sl] * att
        b = ball[op] * att
        pid = w.pid[ks][op][:, sl]
        f = on[op][:, sl] & (pid != w.gk[ks][op][:, team][:, None])
        nf = np.maximum(f.sum(1), 1)
        cen = (q * f[..., None]).sum(1) / nf[:, None]
        rel = q - cen[:, None]
        wb = poss[op] == team
        for i in range(11):
            for pp in np.unique(pid[f[:, i], i]):
                mm = f[:, i] & (pid[:, i] == pp)
                a = [w.anchor.get((int(pp), h)) for h in (0, 1)]
                a = [x for x in a if x is not None]
                if not a:
                    continue
                a = np.mean(a, 0)
                anc = np.where(wb[mm][:, None], a[None, 0:2], a[None, 2:4])
                zd.append(np.hypot(*(rel[mm, i] - anc).T))
        dball = np.hypot(*(q - b[:, None]).transpose(2, 0, 1))
        near.append(np.where(f, dball <= 150, False).sum(1))
        dd = np.hypot(*(q[:, :, None] - q[:, None, :]).transpose(3, 0, 1, 2))
        dd[:, np.arange(11), np.arange(11)] = np.inf
        dd = np.where(f[:, :, None] & f[:, None, :], dd, np.inf).min(2)
        mate.append(dd[f & np.isfinite(dd)])
        blk.append(np.abs(cen[:, 0] - b[:, 0]))
        to_b = (b[:, None] - q) / np.maximum(dball, 1e-6)[..., None]
        v_in = (vel[:, sl] * att * to_b).sum(-1)
        chase.append(np.where(f & (dball <= 300), v_in > 1.0, False).sum(1))
    cat = lambda xs: float(np.mean(np.concatenate(xs))) if xs and len(np.concatenate(xs)) else np.nan
    return dict(dist_zona=cat(zd), cerca_pelota_150=cat(near), vecino_mas_cercano=cat(mate), bloque_dx=cat(blk),
                persigue=cat(chase))


def play(policy, matches, n_games, H, seed=0, n_env=8, every=FRAME_SKIP, temp_move=1.0, delay=None, hyst=0.0,
         only_kickoff=False, ens_k=0, ens_m=0.0, renew=1.0, renew_kick=1.0):
    rng = np.random.default_rng(seed)
    env = HX11Env(n_env, seed=seed)
    ens = Ensemble(min(ens_k, policy.chunk), ens_m, every) if (ens_k > 1 and getattr(policy, "chunk", 1) > 1) else None
    noise = StickyNoise(renew, seed, renew_kick) if (renew < 1.0 or renew_kick < 1.0) else None
    res, logs_ref, traj = [], [], []
    starts = []
    for mi, m in enumerate(matches):
        ko = np.nonzero((m.phase == KICKOFF) & (np.arange(m.T) >= MAX_BACK))[0]
        op = np.nonzero((m.phase == OPEN) & (np.arange(m.T) >= MAX_BACK) & (np.arange(m.T) < m.T - H - 1))[0]
        pool = list(ko[np.r_[True, np.diff(ko) > 1]]) + ([] if only_kickoff else
                                                         list(rng.choice(op, size=min(4, len(op)), replace=False)))
        for t0 in pool:
            starts.append((mi, int(t0)))
    rng.shuffle(starts)
    starts = starts[:n_games]
    for b0 in range(0, len(starts), n_env):
        chunk = starts[b0:b0 + n_env]
        W, D = [], []
        for i, (mi, t0) in enumerate(chunk):
            w = window_sp(matches[mi], t0, H)
            w.score[MAX_BACK:] = 0
            w.score[:MAX_BACK] = 0
            load_state(env, i, w, MAX_BACK)
            env.ri[i, K.RI_OFFSIDE_ON], env.ri[i, K.RI_FOULS_ON], env.ri[i, K.RI_PS_ON] = 1, 1, 1
            if matches[mi].phase[t0] == KICKOFF:
                env.ri[i, K.RI_MODE] = K.KICKOFF
                env.ri[i, K.RI_KO] = 1
                env.ri[i, K.RI_KO_TEAM] = max(int(matches[mi].pteam[t0]), 0)
            W.append(w)
            D.append(rng.integers(D_MIN, D_MAX + 1, 22) if delay is None else np.full(22, delay))
        n = len(chunk)
        acts = np.zeros((n_env, 22), np.int64)
        prev = np.full((n_env, 22), -1, np.int64)       # última acción en coordenadas del equipo (histéresis)
        next_dec = np.zeros((n_env, 22), np.int64)
        use_hold = getattr(policy, "hold", False)
        for k in range(H):
            kk = MAX_BACK + k
            for i, w in enumerate(W):
                w.inp[kk] = w.inp[kk - 1] if k > 0 else w.inp[kk]
            rows_own, rows_ent, who = [], [], []
            for i, w in enumerate(W):
                slots = np.nonzero(w.pid[kk] >= 0)[0]
                slots = slots[(next_dec[i, slots] <= k) if use_hold else np.full(len(slots), k % every == 0)]
                if not len(slots):
                    continue
                o, e, _, _ = build(w, np.full(len(slots), kk), slots, D[i][slots], policy.extra)
                rows_own.append(o)
                rows_ent.append(e)
                who += [(i, int(s)) for s in slots]
            if who:
                with torch.no_grad():
                    ot, et = torch.from_numpy(np.concatenate(rows_own)), torch.from_numpy(np.concatenate(rows_ent))
                    if use_hold:
                        lg, hl = policy.forward_hold(ot, et)
                        hc = torch.distributions.Categorical(logits=hl).sample().numpy()
                    else:
                        lg = policy_logits(policy, ot, et, ens, [(b0, i, s) for i, s in who], [k] * len(who))
                    act = sample(lg, temp_move, prev=np.array([prev[i, s] for i, s in who]), hyst=hyst,
                                 gumbel=noise.get([(b0, i, s) for i, s in who],
                                                  near_rows([env.pos[i, env.fp + s] for i, s in who],
                                                            [env.pos[i, 0] for i, s in who]))
                                 if noise and renew < 1.0 else None,
                                    kick_u=noise.get_kick([(b0, i, s) for i, s in who],
                                                          piece_rows([env.ri[i, K.RI_MODE] for i, s in who]))
                                    if noise and renew_kick < 1.0 else None)
                for j, ((i, s), a) in enumerate(zip(who, act)):
                    a = int(a)
                    prev[i, s] = a
                    if s >= 11:
                        a = int(ROT_MOVE[a % 9] + 9 * (a // 9))
                    acts[i, s] = a
                    W[i].inp[kk, s] = bits_of_action(np.array([a]))[0]
                    next_dec[i, s] = k + (int(HOLD_REP[hc[j]]) if use_hold else every)
            env.step(acts, frames=1)
            for i, w in enumerate(W):
                store_all(env, i, w, kk + 1)
                w.score[kk + 1] = env.ri[i, [K.RI_SCORE0, K.RI_SCORE1]]
        for i, (mi, t0) in enumerate(chunk):
            r = game_metrics(W[i], MAX_BACK + 1, MAX_BACK + H)
            r.update(partido=matches[mi].name, t0=t0, inicio="saque" if matches[mi].phase[t0] == KICKOFF else "abierto")
            res.append(r)
            traj.append(dict(pos=W[i].pos[MAX_BACK:].tolist(), ball=W[i].ball[MAX_BACK:].tolist(),
                             pid=(W[i].pid[MAX_BACK] >= 0).tolist(), phase=W[i].phase[MAX_BACK:].tolist(),
                             score=W[i].score[MAX_BACK:].tolist(), pkind=W[i].pkind[MAX_BACK:].tolist(),
                             pteam=W[i].pteam[MAX_BACK:].tolist(), spot=W[i].spot[MAX_BACK:].tolist()))
            # referencia humana: la misma ventana en la grabación (si el partido alcanza)
            m = matches[mi]
            if t0 + H < m.T:
                ref = game_metrics(window_sp(m, t0, H), MAX_BACK + 1, MAX_BACK + H)
                logs_ref.append(ref)
    return res, logs_ref, traj


def summarize(rows):
    keys = [k for k in rows[0] if isinstance(rows[0][k], float)]
    return {k: dict(media=round(float(np.nanmean([r[k] for r in rows])), 3),
                    p10=round(float(np.nanpercentile([r[k] for r in rows], 10)), 3),
                    p90=round(float(np.nanpercentile([r[k] for r in rows], 90)), 3)) for k in keys}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--seconds", type=int, default=60)
    ap.add_argument("--matches", type=int, default=6)
    ap.add_argument("--n-env", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--every", type=int, default=FRAME_SKIP)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--delay", type=int, default=None, help="retardo fijo para los 22 (sin esto: sorteado en [8, 36])")
    ap.add_argument("--hyst", type=float, default=0.0, help="histéresis de la dirección, como en el cerebro de la sala")
    ap.add_argument("--solo-saques", action="store_true", help="arrancar solo desde saques iniciales (como la sala)")
    ap.add_argument("--ens", type=int, default=0, help="acciones en bloque: ensamble temporal de K decisiones (0 = no)")
    ap.add_argument("--ens-m", type=float, default=0.0, help="peso exp(−m·antigüedad) del ensamble (0 = parejo)")
    ap.add_argument("--renovar", type=float, default=1.0, help="ruido persistente de la dirección (1 = no)")
    ap.add_argument("--renovar-patada", type=float, default=1.0, help="ruido persistente de la tecla de patear")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    modos = json.loads((ROOT / "data" / "hx11" / "modos.json").read_text(encoding="utf-8"))
    rng = np.random.default_rng(a.seed)
    names = sorted(rng.permutation(l0_split()[a.split])[:a.matches].tolist())
    matches = [load_match(n, modos) for n in names]
    ck = torch.load(a.model, map_location="cpu", weights_only=False)
    policy = SetPolicy(**ck["config"])
    policy.load_state_dict(ck["state"])
    policy.eval()
    t0 = time.time()
    rows, ref, traj = play(policy, matches, a.games, a.seconds * 60, seed=a.seed, n_env=a.n_env, every=a.every,
                           temp_move=a.temp, delay=a.delay, hyst=a.hyst, only_kickoff=a.solo_saques, ens_k=a.ens,
                           ens_m=a.ens_m, renew=a.renovar, renew_kick=a.renovar_patada)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res = dict(modelo=a.model, juegos=len(rows), segundos_por_juego=a.seconds, simulado=summarize(rows),
               humano_mismas_ventanas=summarize(ref) if ref else None, tiempo=round(time.time() - t0))
    (out / "resumen.json").write_text(json.dumps(dict(res, filas=rows), indent=1), encoding="utf-8")
    (out / "trayectorias.json").write_text(json.dumps(traj), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
