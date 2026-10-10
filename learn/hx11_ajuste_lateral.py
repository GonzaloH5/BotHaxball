"""Ajuste corto de L0 para el lateral (L0.5 dirigido): autoimitación filtrada (self-imitation learning, Oh et al. 2018).

Con las reglas del lateral sacadas de las grabaciones (env/hx11, 2026-10-09), los 22 bots sacaban mal el 48% de los
laterales contra el 2% de los humanos: meten la pelota empujando y patean desde adentro, o patean paralelo a la línea.

1. `colectar`: episodios de lateral en el simulador desde laterales reales del banco de entrenamiento (estado grabado en
   la colocación + la colocación del árbitro del simulador), con los 22 jugando la política tal como en la sala
   (temperatura 0,7 en la dirección, patada persistente). Cada episodio termina en:
   * `bien`: patada desde afuera y la pelota entra a la cancha;
   * `mal`: el simulador lo cobra (pateado desde adentro, metida y dejada adentro, o no entró), o sale otra vez;
   * `tiempo`: se acabó el plazo del saque (cesión al rival);
   * `sin_resolver`: no terminó en H ticks.
   De los `bien` se guardan las decisiones del equipo que saca cerca de la pelota (propia, entidades, acción del equipo).
2. El ajuste lo hace `learn.hx11_bc --init <modelo> --extra-data <colecta.npz> --extra-frac f`: cada lote mezcla datos
   humanos (para no olvidar el resto del juego) con esas decisiones.
3. Se mide otra vez la tasa de `bien` y, antes de usar el modelo, la puerta P3 completa.

  python -m learn.hx11_ajuste_lateral colectar --model M --laterales 1500 --out runs/hx11_l0/ajuste_lateral/colecta_r1
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
from learn.hx11_data import D_MAX, D_MIN, MAX_BACK, ROT_MOVE, build, load_match
from learn.hx11_policy import SetPolicy, StickyNoise, piece_rows, sample
from learn.hx11_selfplay import kick_blocked, store_all, window_sp
from tools.hx11_d0 import PIECE

ROOT = Path(__file__).resolve().parent.parent


def lateral_starts(matches, H, rng, per_match=None):
    """(partido, tick de la colocación) de los laterales del banco con H ticks por delante."""
    out = []
    for mi, m in enumerate(matches):
        st = np.flatnonzero((m.phase == PIECE) & (np.r_[-1, m.phase[:-1]] != PIECE) & (m.pkind == C.LATERAL))
        st = st[(st >= MAX_BACK + 1) & (st < m.T - H - 1) & (m.pteam[st] >= 0)]
        if per_match is not None and len(st) > per_match:
            st = rng.choice(st, per_match, replace=False)
        out += [(mi, int(t)) for t in st]
    rng.shuffle(out)
    return out


def run(policy, matches, starts, H=720, n_env=16, seed=0, temp=0.7, renew_kick=0.33, near_px=200.0, every=FRAME_SKIP,
        rule=False):
    """Episodios de lateral. Devuelve (resultados, muestras de los `bien`: propia, entidades, acción del equipo)."""
    rng = np.random.default_rng(seed)
    env = HX11Env(n_env, seed=seed)
    noise = StickyNoise(1.0, seed, renew_kick) if renew_kick < 1.0 else None
    results, keep_o, keep_e, keep_y = [], [], [], []
    for b0 in range(0, len(starts), n_env):
        chunk = starts[b0:b0 + n_env]
        W, D, info = [], [], []
        for i, (mi, tp) in enumerate(chunk):
            m = matches[mi]
            w = window_sp(m, tp, H)
            w.score[:] = 0
            load_state(env, i, w, MAX_BACK)
            env.ri[i, K.RI_OFFSIDE_ON], env.ri[i, K.RI_FOULS_ON], env.ri[i, K.RI_PS_ON] = 1, 1, 1
            team = int(m.pteam[tp])
            sx, sy = (m.spot[tp].astype(np.float64) / 10).tolist()
            K._place(env.pos[i], env.vel[i], env.inv[i], env.mask[i], env.group[i], env.radius[i], env.grav[i],
                     env.ri[i], env.rf[i], C.LATERAL, team, sx, sy, env.fp, env.P, env.n_red, env.home, env.hold,
                     env.hold_cdf)
            store_all(env, i, w, MAX_BACK)
            W.append(w)
            D.append(rng.integers(D_MIN, D_MAX + 1, 22))
            info.append(dict(team=team, state="pendiente", end=None, buf_o=[], buf_e=[], buf_y=[]))
        n = len(chunk)
        acts = np.zeros((n_env, 22), np.int64)
        for k in range(H):
            live = [i for i in range(n) if info[i]["state"] in ("pendiente", "pateado")]
            if not live:
                break
            kk = MAX_BACK + k
            for i in range(n):
                W[i].inp[kk] = W[i].inp[kk - 1] if k > 0 else W[i].inp[kk]
            if k % every == 0:
                rows_o, rows_e, who = [], [], []
                for i in live:
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
                    if rule and a >= 9 and kick_blocked(W[i], kk - int(D[i][s]) - 1, s, int(D[i][s]), False, True):
                        a = a % 9
                    if info[i]["state"] == "pendiente" and (s >= 11) == (info[i]["team"] == 1):
                        ball, me = env.pos[i, 0], env.pos[i, env.fp + s]
                        if np.hypot(*(ball - me)) <= near_px:
                            info[i]["buf_o"].append(ot[j])
                            info[i]["buf_e"].append(et[j])
                            info[i]["buf_y"].append(a)              # acción en coordenadas del equipo (como build)
                    if s >= 11:
                        a = int(ROT_MOVE[a % 9] + 9 * (a // 9))
                    acts[i, s] = a
                    W[i].inp[kk, s] = bits_of_action(np.array([a]))[0]
            ev = env.step(acts, frames=1)
            for i in range(n):
                store_all(env, i, W[i], kk + 1)
                st = info[i]["state"]
                if st not in ("pendiente", "pateado"):
                    continue
                e = int(ev[i, 0])
                if st == "pendiente" and e == K.EV_PIECE_KICK:
                    info[i]["state"] = st = "pateado"
                    info[i]["kick_t"] = k
                if st == "pateado" and env.ri[i, K.RI_LAT_ARMED] == 1:
                    info[i]["state"], info[i]["end"] = "bien", k
                elif e == K.EV_OUT:
                    # mal desde "pendiente": pateado con la pelota adentro, o metida sin patear y dejada adentro;
                    # desde "pateado": no entró (paralelo o hacia afuera), o la tocó un rival afuera
                    info[i]["state"], info[i]["end"] = "mal", k
                    info[i]["mal_desde"] = st
                elif e == K.EV_TIMEOUT:
                    info[i]["state"], info[i]["end"] = "tiempo", k
        for i, (mi, tp) in enumerate(chunk):
            st = info[i]["state"]
            if st in ("pendiente", "pateado"):
                st = "sin_resolver"
            results.append(dict(partido=matches[mi].name, t=tp, resultado=st, ticks=info[i]["end"],
                                patada=info[i].get("kick_t"), mal_desde=info[i].get("mal_desde"),
                                muestras=len(info[i]["buf_y"])))
            if st == "bien" and info[i]["buf_y"]:
                keep_o.append(np.stack(info[i]["buf_o"]))
                keep_e.append(np.stack(info[i]["buf_e"]))
                keep_y.append(np.array(info[i]["buf_y"], np.int64))
    data = None
    if keep_y:
        data = dict(own=np.concatenate(keep_o).astype(np.float32), ents=np.concatenate(keep_e).astype(np.float32),
                    y=np.concatenate(keep_y))
    return results, data


def summarize(results):
    c = collections.Counter(r["resultado"] for r in results)
    n = max(len(results), 1)
    good = [r["ticks"] for r in results if r["resultado"] == "bien"]
    md = collections.Counter(r.get("mal_desde") for r in results if r["resultado"] == "mal")
    return dict(laterales=len(results), **{k: round(v / n, 3) for k, v in sorted(c.items())},
                mal_sin_patear=round(md["pendiente"] / n, 3), mal_pateado=round(md["pateado"] / n, 3),
                ticks_hasta_bien_p50=float(np.median(good)) if good else None)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("colectar")
    c.add_argument("--model", required=True)
    c.add_argument("--split", default="train", choices=["train", "dev", "test"])
    c.add_argument("--split-file", default="runs/hx11_l0/split.json")
    c.add_argument("--partidos", type=int, default=60)
    c.add_argument("--laterales", type=int, default=1500)
    c.add_argument("--n-env", type=int, default=16)
    c.add_argument("--temp", type=float, default=0.7)
    c.add_argument("--renovar-patada", type=float, default=0.33)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--regla", action="store_true", help="regla de patada del lateral (kick_rule_ok)")
    c.add_argument("--threads", type=int, default=4)
    c.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    sp = json.loads((ROOT / a.split_file).read_text(encoding="utf-8"))
    if a.split == "test":
        from learn.hx11_data import l0_split
        sp = dict(sp, test=l0_split()["test"])
    rng = np.random.default_rng(a.seed)
    names = sorted(rng.permutation(sp[a.split])[:a.partidos].tolist())
    modos = json.loads((ROOT / "data" / "hx11" / "modos.json").read_text(encoding="utf-8"))
    matches = [load_match(nm, modos) for nm in names]
    starts = lateral_starts(matches, 720, rng)[:a.laterales]
    ck = torch.load(a.model, map_location="cpu", weights_only=False)
    policy = SetPolicy(**ck["config"])
    policy.load_state_dict(ck["state"])
    policy.eval()
    t0 = time.time()
    results, data = run(policy, matches, starts, n_env=a.n_env, seed=a.seed, temp=a.temp, renew_kick=a.renovar_patada,
                           rule=a.regla)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    res = dict(modelo=a.model, split=a.split, partidos=len(names), resumen=summarize(results),
               muestras=int(len(data["y"])) if data else 0, segundos=round(time.time() - t0))
    if data:
        np.savez_compressed(str(out) + ".npz", **data)
    Path(str(out) + ".json").write_text(json.dumps(dict(res, filas=results), ensure_ascii=False, indent=1),
                                        encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
