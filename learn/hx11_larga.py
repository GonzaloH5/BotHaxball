"""Prueba larga (2026-10-10): el bot como fantasma durante 60 s entre humanos grabados, en todos los puestos.

Los dos partidos reales del bot mostraron fallas que la puerta P3 (ventanas de 5 s) no ve: deambula de lado a lado
(su posición lateral varía ±460 px contra ±140–340 de sus compañeros), juega adelantado, no cubre, casi no participa y
se pega al compañero que saca un córner o el saque inicial. El usuario pidió que juegue bien en cualquier puesto.

* **Fantasma:** en cada ventana de 60 s de un partido grabado, cada lugar ocupado se juega con la red como un fantasma:
  se mueve con la física del jugador (aceleración, patada sostenida, amortiguación), pero no choca ni toca la pelota.
  El resto del partido (pelota, los otros 21, fases) es el grabado, así que no se desvía en un minuto, que es lo que
  pasaba con el lazo cerrado. La red ve lo de siempre: la grabación con su propio fantasma en su lugar, con retardo.
  Tras un gol, en el saque inicial, el fantasma vuelve a la posición en la que el humano de su lugar reaparece.
* **Métricas** (fantasma contra el humano de ese lugar, en los mismos ticks):
  - zona: distancia a su zona (centroide de campo del equipo + ancla del jugador, con o sin la pelota);
  - dispersión x / y respecto del centroide del equipo (desvío estándar, juego abierto);
  - del lado de su arco cuando el rival tiene la pelota (fracción de ticks);
  - repliegue: cuánto retrocede en los 3 s siguientes a una pérdida de su equipo (px hacia su arco);
  - va a la pelota cuando es el más cercano de su equipo (velocidad de acercamiento, px/tick);
  - cerca de la pelota (< 150 px), distancia a la pelota (mediana), velocidad, cambios de tecla por segundo;
  - estorbo en pelotas paradas y saques iniciales de su equipo que saca un compañero: segundos a menos de 35 px del
    que saca o de 45 px de la pelota, por saque.
* **Criterio:** como P3, razón fantasma/humano por línea del puesto con IC 90% por bootstrap de ventanas; pasa si se
  superpone con [0,75; 1,33]. El estorbo se informa como diferencia (el humano casi nunca estorba).

  python -m learn.hx11_larga --model runs/hx11_l0/l0_ajuste_corner_h2/model_final.pt --out reports/hx11/larga_h2
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from learn import hx11_closed_loop as CL
from learn.hx11_closed_loop import bits_of_action
from learn.hx11_data import D_MAX, D_MIN, LINES, MAX_BACK, MOVE_DX, MOVE_DY, ROT_MOVE, build, l0_split, load_match
from learn.hx11_policy import KICKOFF_KIND, SetPolicy, StickyNoise, keep_away, near_rows, piece_rows, sample
from learn.hx11_selfplay import window_sp
from tools.hx11_d0 import KICKOFF, OPEN

ROOT = Path(__file__).resolve().parent.parent
H = 3600
ACC, KACC, DAMP, KDAMP = 0.119, 0.1, 0.96, 0.96       # jugador (stadium del simulador)
_U = np.stack([MOVE_DX, MOVE_DY], 1).astype(np.float64)
MOVE_UNIT = _U / np.maximum(np.hypot(*_U.T), 1)[:, None]
LO, HI = 0.75, 1.33
ROL_A_PUESTO = {"ARQ": "ARQ", "DEF-I": "LI", "DEF-D": "LD", "DEF-C": "DFC", "DEF-CI": "CRI", "DEF-CD": "CRD",
                "MCD-C": "MCD", "MCD-CI": "MCD", "MCD-CD": "MCD", "MED-I": "MI", "MED-D": "MD", "MED-C": "MC",
                "MED-CI": "MC", "MED-CD": "MC", "DEL-C": "DC", "DEL-CI": "DC", "DEL-CD": "DC", "DEL-I": "EI",
                "DEL-D": "ED"}
RATIO_KEYS = ("zona", "disp_x", "disp_y", "lado_arco", "repliegue", "va_a_la_pelota", "cerca_pelota", "dist_pelota",
              "vel", "cambios_tecla_s")


def ghost_window(policy, m, t0, slots, owner, rng, temp=0.7, renew=0.33, renew_kick=0.33, every=3, moves=None,
                 anchors=None, length=None, start=None):
    """Juega los lugares `slots` como fantasmas en [t0, t0 + H). Devuelve {lugar: ventana con el fantasma}.
    `anchors` ({lugar: ancla de 4}): el ancla que ve la red en vez de la propia del humano (como en la sala, donde el
    bot recibe el ancla genérica del puesto pedido)."""
    Hw = length or H
    base = window_sp(m, t0, Hw)
    W = {s: window_sp(m, t0, Hw) for s in slots}
    for s, anc in (anchors or {}).items():
        pid = int(m.pid[t0, s])
        W[s].anchor[(pid, 0)] = W[s].anchor[(pid, 1)] = np.asarray(anc, np.float32)
    pos = {s: (np.asarray(start[s], np.float64) if start and s in start else base.pos[MAX_BACK, s].astype(np.float64) / 10)
           for s in slots}
    for s in slots:
        W[s].pos[MAX_BACK, s] = np.round(pos[s] * 10)
    vel = {s: base.vel[MAX_BACK, s].astype(np.float64) / 1000 for s in slots}
    act = {s: 0 for s in slots}
    D = {s: int(rng.integers(D_MIN, D_MAX + 1)) for s in slots}
    noise = StickyNoise(renew, int(rng.integers(1 << 30)), renew_kick)
    for k in range(Hw):
        kk = MAX_BACK + k
        if k % every == 0:
            own, ents = [], []
            for s in slots:
                o, e, _, _ = build(W[s], np.array([kk]), np.array([s]), np.array([D[s]]), policy.extra)
                own.append(o)
                ents.append(e)
            ot = np.concatenate(own)
            keys = [(t0, s) for s in slots]
            fresh = near_rows([pos[s] for s in slots], [base.ball[kk].astype(np.float64) / 10 for _ in slots])
            with torch.no_grad():
                lg = policy(torch.from_numpy(ot), torch.from_numpy(np.concatenate(ents)))
            a_team = sample(lg, temp, gumbel=noise.get(keys, fresh) if renew < 1 else None,
                            kick_u=noise.get_kick(keys, piece_rows([base.phase[kk]] * len(slots)))
                            if renew_kick < 1 else None)
            if moves is not None:
                a_team = moves(ot, a_team)
            for j, s in enumerate(slots):
                a = int(a_team[j])
                act[s] = int(ROT_MOVE[a % 9] + 9 * (a // 9)) if s >= 11 else a
        ph_prev, ph = int(base.phase[kk]), int(base.phase[kk + 1])
        for s in slots:
            if ph == KICKOFF and ph_prev != KICKOFF:
                # saque inicial: el fantasma reaparece donde reaparece el humano de su lugar
                pos[s] = base.pos[kk + 1, s].astype(np.float64) / 10
                vel[s] = np.zeros(2)
            a = act[s]
            kick = a >= 9
            vel[s] = vel[s] + MOVE_UNIT[a % 9] * (KACC if kick else ACC)
            pos[s] = np.clip(pos[s] + vel[s], [-1950, -1350], [1950, 1350])
            vel[s] = vel[s] * (KDAMP if kick else DAMP)
            w = W[s]
            w.pos[kk + 1, s] = np.round(pos[s] * 10)
            w.vel[kk + 1, s] = np.clip(np.round(vel[s] * 1000), -32767, 32767)
            w.inp[kk + 1, s] = bits_of_action(np.array([a]))[0]
    return base, W


def slot_metrics(arr, base, s, m, t0, owner):
    """Métricas del lugar s en la ventana `arr` (fantasma o humano), con la pelota y los demás de `base`."""
    ks = np.arange(MAX_BACK + 1, MAX_BACK + H)
    team = 0 if s < 11 else 1
    att = 1.0 if team == 0 else -1.0
    ph = base.phase[ks]
    op = ph == OPEN
    me = arr.pos[ks, s].astype(np.float64) / 10
    ball = base.ball[ks].astype(np.float64) / 10
    sl = np.arange(team * 11, team * 11 + 11)
    q = arr.pos[ks][:, sl].astype(np.float64) / 10
    f = (base.pid[ks][:, sl] >= 0) & (base.pid[ks][:, sl] != base.gk[ks, team][:, None])
    cen = (q * f[..., None]).sum(1) / np.maximum(f.sum(1), 1)[:, None]
    rel = (me - cen) * att
    pid = int(m.pid[t0, s])
    anc = m.anchor.get((pid, int(t0 >= m.T // 2)), np.zeros(4, np.float32))
    own_ball = owner[t0 + 1:t0 + H]
    has = own_ball >= 0
    ours = has & ((own_ball // 11) == team)
    theirs = has & ((own_ball // 11) != team)
    target = cen + np.where(ours[:, None], anc[:2], anc[2:]) * att
    zona = np.hypot(*(me - target).T)
    db = np.hypot(*(me - ball).T)
    out = dict(zona=float(zona[op].mean()) if op.any() else np.nan,
               disp_x=float(rel[op, 0].std()) if op.sum() > 30 else np.nan,
               disp_y=float(rel[op, 1].std()) if op.sum() > 30 else np.nan,
               cerca_pelota=float((db[op] < 150).mean()) if op.any() else np.nan,
               dist_pelota=float(np.median(db[op])) if op.any() else np.nan)
    th = op & theirs
    out["lado_arco"] = float(((me[th, 0] - ball[th, 0]) * att < 0).mean()) if th.sum() > 30 else np.nan
    # repliegue: pérdidas de su equipo (de nosotros a ellos) en juego abierto
    lost = np.nonzero(ours[:-1] & theirs[1:] & op[1:])[0] + 1
    back = [-(me[min(i + 180, len(me) - 1), 0] - me[i, 0]) * att for i in lost if i + 60 < len(me)]
    out["repliegue"] = float(np.mean(back)) if back else np.nan
    # va a la pelota cuando es el más cercano de su equipo (con el fantasma en su lugar)
    others = [qq for qq in sl if qq != s]
    dm = np.where(base.pid[ks][:, others] >= 0,
                  np.hypot(*(base.pos[ks][:, others].astype(np.float64) / 10 - ball[:, None]).transpose(2, 0, 1)),
                  np.inf).min(1)
    nearest = op & (db < dm) & (db > 40)
    close = -np.diff(db)                                   # acercamiento por tick
    nz = nearest[:-1]
    out["va_a_la_pelota"] = float(close[nz].mean()) if nz.sum() > 30 else np.nan
    sp = np.hypot(*np.diff(me, axis=0).T)
    out["vel"] = float(sp[op[1:] & (sp < 10)].mean()) if op.any() else np.nan
    inp = arr.inp[ks, s].astype(np.int64)
    out["cambios_tecla_s"] = float(((inp[1:] != inp[:-1]) & op[1:]).sum()) / max(op.sum() / 60, 1e-6)
    # estorbo en pelotas paradas / saques iniciales de su equipo que saca un compañero
    est, n_p = 0.0, 0
    piece = np.isin(ph, (2, 5)) & (base.pteam[ks] == team)
    edges = np.flatnonzero(np.diff(np.r_[0, piece.astype(np.int8), 0]))
    for a, b in zip(edges[::2], edges[1::2]):
        bpos = ball[b - 1]
        d_all = np.where(base.pid[ks[b - 1], sl] >= 0,
                         np.hypot(*(base.pos[ks[b - 1], sl].astype(np.float64) / 10 - bpos).T), np.inf)
        taker = int(sl[int(np.argmin(d_all))])
        if taker == s:
            continue                                        # en la grabación lo sacó el humano de este lugar
        tk = base.pos[ks[a:b], taker].astype(np.float64) / 10
        near = (np.hypot(*(me[a:b] - tk).T) < 35) | (db[a:b] < 45)
        est += near.sum() / 60
        n_p += 1
    out["estorbo_s"] = est / n_p if n_p else np.nan
    out["saques"] = n_p
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", default="test", choices=["test", "dev"])
    ap.add_argument("--partidos", type=int, default=16)
    ap.add_argument("--ventanas", type=int, default=3, help="ventanas de 60 s por partido")
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--renovar", type=float, default=0.33)
    ap.add_argument("--renovar-patada", type=float, default=0.33)
    ap.add_argument("--ancla", default="propia", choices=["propia", "puesto"],
                    help="propia: la del humano de ese lugar; puesto: la genérica del puesto (como en la sala)")
    ap.add_argument("--seguridad", action="store_true",
                    help="red de seguridad de la sala: no acercarse en saques que saca otro (keep_away)")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    rng = np.random.default_rng(a.seed)
    modos = json.loads((ROOT / "data" / "hx11" / "modos.json").read_text(encoding="utf-8"))
    pool = l0_split()["test"] if a.split == "test" else json.loads(
        (ROOT / "runs/hx11_l0/split.json").read_text(encoding="utf-8"))["dev"]
    names = sorted(rng.permutation(pool)[:a.partidos].tolist())
    ck = torch.load(a.model, map_location="cpu", weights_only=False)
    policy = SetPolicy(**ck["config"])
    policy.load_state_dict(ck["state"])
    policy.eval()
    vocab = json.loads((ROOT / "reports" / "hx11" / "pos_vocab.json").read_text(encoding="utf-8"))
    rows = []
    t_start = time.time()
    for n in names:
        m = load_match(n, modos)
        owner = np.load(ROOT / "data" / "hx11" / "d1" / f"{n}.npz")["owner"]
        cand = np.arange(MAX_BACK + 1, m.T - H - 2, 600)
        for t0 in rng.choice(cand, size=min(a.ventanas, len(cand)), replace=False):
            t0 = int(t0)
            ok = (m.pid[t0 - MAX_BACK:t0 + H + 1] == m.pid[t0]).all(0) & (m.pid[t0] >= 0)
            slots = [int(s) for s in np.nonzero(ok)[0]]
            if len(slots) < 10:
                continue
            anchors = None
            if a.ancla == "puesto":
                d2 = json.loads((ROOT / "data" / "hx11" / "d2" / f"{n}.json").read_text(encoding="utf-8"))
                role = {int(p): info["nombre_rol"] for t in d2["equipos"] for p, info in t["jugadores"].items()}
                anchors = {}
                for s in slots:
                    pid = int(m.pid[t0, s])
                    gk = int(m.gk[t0, int(s >= 11)]) == pid
                    key = "ARQ" if gk else ROL_A_PUESTO.get(role.get(pid, ""), "MC")
                    anchors[s] = vocab[key]["ancla"]
            moves = (lambda ot, act: keep_away(ot, act, (2, 3, 4, KICKOFF_KIND))) if a.seguridad else None
            base, W = ghost_window(policy, m, t0, slots, owner, rng, a.temp, a.renovar, a.renovar_patada,
                                   moves=moves, anchors=anchors)
            for s in slots:
                pid = int(m.pid[t0, s])
                gk = int(m.gk[t0, int(s >= 11)]) == pid
                line = "ARQ" if gk else LINES[m.line.get(pid, 3)]
                r = dict(partido=n, t0=t0, lugar=s, linea=line)
                for who, arr in (("bot", W[s]), ("humano", base)):
                    for k, v in slot_metrics(arr, base, s, m, t0, owner).items():
                        r[f"{k}_{who}"] = v
                rows.append(r)
        print(f"{n}: {len(rows)} filas ({time.time() - t_start:.0f} s)", flush=True)
    res = {}
    for line in ("ARQ", "DEF", "MCD", "MED", "DEL"):
        rr = [r for r in rows if r["linea"] == line]
        if len(rr) < 10:
            continue
        res[line] = {"ventanas": len(rr)}
        for key in RATIO_KEYS:
            b = np.array([r[f"{key}_bot"] for r in rr], np.float64)
            h = np.array([r[f"{key}_humano"] for r in rr], np.float64)
            ok = np.isfinite(b) & np.isfinite(h)
            if ok.sum() < 10:
                continue
            b, h = b[ok], h[ok]
            boot = []
            br = np.random.default_rng(3)
            for _ in range(500):
                i = br.integers(0, len(b), len(b))
                if abs(h[i].mean()) > 1e-9:
                    boot.append(b[i].mean() / h[i].mean())
            lo, hi = np.percentile(boot, [5, 95])
            ratio = b.mean() / h.mean() if abs(h.mean()) > 1e-9 else np.nan
            res[line][key] = dict(bot=round(float(b.mean()), 3), humano=round(float(h.mean()), 3),
                                  razon=round(float(ratio), 3), ic90=[round(float(lo), 3), round(float(hi), 3)],
                                  pasa=bool(hi >= LO and lo <= HI))
        e_b = np.array([r["estorbo_s_bot"] for r in rr], np.float64)
        e_h = np.array([r["estorbo_s_humano"] for r in rr], np.float64)
        okk = np.isfinite(e_b) & np.isfinite(e_h)
        res[line]["estorbo_s_por_saque"] = dict(bot=round(float(e_b[okk].mean()), 3) if okk.any() else None,
                                                humano=round(float(e_h[okk].mean()), 3) if okk.any() else None,
                                                saques=int(sum(r["saques_bot"] for r in rr)))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    Path(str(out) + ".json").write_text(json.dumps(dict(modelo=a.model, split=a.split, partidos=names, temp=a.temp,
                                                        renovar=a.renovar, resultado=res, filas=rows),
                                                   indent=1, ensure_ascii=False), encoding="utf-8")
    md = [f"# Prueba larga (fantasma 60 s entre humanos): {Path(a.model).parent.name}", "",
          f"Modelo `{a.model}`, {len(names)} partidos de {a.split}, {len(rows)} ventanas por lugar. Razón fantasma/humano "
          f"con IC 90%; pasa si se superpone con [{LO}; {HI}]. Estorbo: segundos por saque a < 35 px del que saca o "
          f"< 45 px de la pelota, en pelotas paradas y saques iniciales de su equipo que saca otro.", "",
          "| Línea | Métrica | Bot | Humano | Razón | IC 90% | ¿Pasa? |", "|---|---|---|---|---|---|---|"]
    for line, d in res.items():
        for key, v in d.items():
            if key == "ventanas":
                continue
            if key == "estorbo_s_por_saque":
                md.append(f"| {line} | estorbo (s por saque, {v['saques']} saques) | {v['bot']} | {v['humano']} | | | |")
                continue
            md.append(f"| {line} | {key} | {v['bot']} | {v['humano']} | {v['razon']} | {v['ic90'][0]}–{v['ic90'][1]} | "
                      f"{'sí' if v['pasa'] else '**no**'} |")
    Path(str(out) + ".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({l: {k: (v["razon"], v["pasa"]) if isinstance(v, dict) and "razon" in v else v
                          for k, v in d.items()} for l, d in res.items()}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
