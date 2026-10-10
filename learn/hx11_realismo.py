"""Puerta de realismo P5 de L1 (docs/PLAN_HAXELEVEN.md): ¿la población de bots juega como los humanos?

* **Sintéticos** (`sinteticos`): partidos de 22 bots con `learn.hx11_selfplay.play` desde estados reales del banco de
  entrenamiento (saques iniciales y juego abierto), con la configuración de la población (temperatura, ruido
  persistente, reglas de patada). Se guardan los clips compactos (`clip_of`) en un .npz.
* **Referencia y techo** (`puerta`): ventanas del mismo largo de los partidos humanos de PRUEBA (referencia) y de
  ENTRENAMIENTO (techo). Para cada métrica, distancia de Wasserstein-1 entre las distribuciones por ventana:
  sintético contra prueba (d) y entrenamiento contra prueba (techo). Dentro del techo si d ≤ techo.
* **Criterio del plan:** ≥ 80% de las métricas dentro del techo, ninguna colectiva a más de 2 techos, y un clasificador
  real contra sintético (C2ST) sobre clips de 10 s con AUC ≤ 0,70. El C2ST va sobre rasgos resumidos de cada clip
  con una red chica y validación cruzada agrupada por partido; además del AUC, dice qué rasgos delatan al bot.

Las métricas se calculan igual para los tres conjuntos (arreglos de `clip_of`):
* colectivas: goles, remates, cambios de posesión, pases, pase completo;
* tácticas: largo, ancho y profundidad del bloque, separación entre compañeros, cortes por falta u offside;
* temporales: duración de la posesión, juego abierto, espera de los saques, laterales y laterales repetidos;
* mecánicas: patadas cargadas, contacto con la pelota, rachas largas de contacto, velocidad de jugadores y pelota,
  cambios de tecla, el más cercano a la pelota.

  python -m learn.hx11_realismo sinteticos --model runs/hx11_l0/l0_quienva_e1/model.pt --regla-saques --regla-lateral \\
      --juegos 200 --out runs/hx11_l0/realismo/e1.npz
  python -m learn.hx11_realismo puerta --sinteticos runs/hx11_l0/realismo/e1.npz --out reports/hx11/realismo_e1
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from learn.hx11_data import MAX_BACK, l0_split, load_match
from learn.hx11_selfplay import clip_of, window_sp
from tools.hx11_d0 import KICKOFF, OPEN

ROOT = Path(__file__).resolve().parent.parent
TOUCH = 15.0 + 8.0 + 0.5          # toque: centro de la pelota a <= 0,5 px del borde del jugador
KICK_DV = 3.0                     # patada: la pelota gana >= 3 px/tick en un tick con un jugador encima
CHARGED_INV = 1.70                # patada cargada: masa inversa de la pelota > 1,70 antes de la patada
GOAL_X, GOAL_HALF = 1810.0, 170.0
CLIP_C2ST = 600                   # clips de 10 s
FAMILIAS = {
    "colectiva": ("goles_10min", "remates_min", "cambios_posesion_min", "pases_min", "pase_completo"),
    "tactica": ("largo_equipo", "ancho_equipo", "profundidad", "vecino_mas_cercano", "cortes_10min"),
    "temporal": ("duracion_posesion_s", "juego_abierto", "espera_saque", "laterales_10min", "lateral_repetido"),
    "mecanica": ("cargadas_min", "contacto_s_jug_min", "racha_p90", "vel_jugador", "vel_pelota", "cambios_tecla_s",
                 "mas_cercano_abierto"),
}


# ----------------------------------------------------------------------------- métricas por ventana
def _events(c):
    """Toques, último que tocó (posesión), patadas y patadas cargadas, tick a tick."""
    pos = c["pos"].astype(np.float32) / 10
    ball = c["ball"].astype(np.float32) / 10
    on = c["on"]
    d = np.where(on, np.hypot(*(pos - ball[:, None]).transpose(2, 0, 1)), np.inf)
    near = d.argmin(1)
    touch = d.min(1) <= TOUCH
    speed = np.r_[0.0, np.hypot(*np.diff(ball, axis=0).T)]
    dv = np.r_[0.0, np.diff(speed)]
    kick = (dv >= KICK_DV) & (d.min(1) <= TOUCH + 6) & (speed < 40)
    binv = c["binv"].astype(np.float32)
    charged = kick & (np.r_[binv[:1], binv[:-1]] > CHARGED_INV)
    return pos, ball, d, near, touch, speed, kick, charged


def clip_metrics(c):
    T = len(c["phase"])
    ph = c["phase"].astype(np.int64)
    op = ph == OPEN
    om = max(op.sum() / 3600, 1e-6)
    mins = T / 3600
    pos, ball, d, near, touch, speed, kick, charged = _events(c)
    on, gk = c["on"], c["gk"]
    n_on = max(on[0].sum(), 1)
    # posesión: último jugador que tocó, en juego abierto
    last = -1
    owner = np.full(T, -1)
    passes = changes = 0
    spells, spell_t = [], 0
    for t in range(T):
        if not op[t]:
            if last >= 0:
                spells.append(t - spell_t)
            last = -1
            continue
        if touch[t]:
            s = int(near[t])
            if last >= 0 and s != last:
                if (s < 11) == (last < 11):
                    passes += 1
                else:
                    changes += 1
                    spells.append(t - spell_t)
                    spell_t = t
            elif last < 0:
                spell_t = t
            last = s
        owner[t] = last
    # remates: patada hacia la boca del arco rival desde menos de 900 px
    shots = 0
    for t in np.nonzero(kick & op)[0]:
        if t + 1 >= T:
            continue
        s = int(near[t])
        side = 1.0 if s < 11 else -1.0              # rojo ataca hacia +x
        v = ball[t + 1] - ball[t]
        if v[0] * side <= 0 or abs(GOAL_X * side - ball[t, 0]) > 900:
            continue
        k = (GOAL_X * side - ball[t, 0]) / v[0]
        y = ball[t, 1] + k * v[1]
        if abs(y) <= GOAL_HALF * 1.2:
            shots += 1
    goals = int(np.clip(np.diff(c["score"].astype(np.int64), axis=0), 0, None).sum())
    # bloque (jugadores de campo, juego abierto)
    lg, an, pr, vec = [], [], [], []
    ks = np.nonzero(op)[0][::6]
    for team in (0, 1):
        sl = slice(team * 11, team * 11 + 11)
        f = on[ks, sl] & ~gk[ks, sl]
        xs = np.where(f, pos[ks, sl, 0], np.nan)
        ys = np.where(f, pos[ks, sl, 1], np.nan)
        lg.append(np.nanmax(xs, 1) - np.nanmin(xs, 1))
        an.append(np.nanmax(ys, 1) - np.nanmin(ys, 1))
        pr.append(np.nanpercentile(xs, 75, axis=1) - np.nanpercentile(xs, 25, axis=1))
        q = pos[ks][:, sl]
        dd = np.hypot(*(q[:, :, None] - q[:, None]).transpose(3, 0, 1, 2))
        dd = np.where(f[:, :, None] & f[:, None] & ~np.eye(11, dtype=bool)[None], dd, np.inf)
        m = dd.min(2)
        vec.append(m[np.isfinite(m)])
    # cortes (falta u offside: fase congelada) y pelotas paradas
    cuts = int(((ph[1:] == 3) & (ph[:-1] != 3)).sum())
    starts = np.nonzero((ph[1:] == 2) & (ph[:-1] != 2))[0] + 1
    waits, lat_new, lat_rep, prev_spot = [], 0, 0, None
    pk = c["pkind"].astype(np.int64)
    spot = c["spot"].astype(np.float32) / 10
    for a in starts:
        b = a
        while b < T and ph[b] == 2:
            b += 1
        if b < T:
            waits.append(b - a)
        if pk[a] == 1:
            if prev_spot is not None and np.hypot(*(spot[a] - prev_spot)) < 1.0:
                lat_rep += 1
            else:
                lat_new += 1
            prev_spot = spot[a]
        else:
            prev_spot = None
    # contacto
    touch_p = (d <= TOUCH) & op[:, None]
    runs = []
    for s in range(22):
        x = np.r_[0, touch_p[:, s].astype(np.int8), 0]
        e = np.flatnonzero(np.diff(x))
        runs += list(e[1::2] - e[::2])
    # velocidades y teclas
    sp = np.hypot(*np.diff(pos, axis=0).transpose(2, 0, 1))
    okv = on[1:] & on[:-1] & op[1:, None] & (sp < 10)
    inp = c["inp"].astype(np.int64)
    chg = ((inp[1:] != inp[:-1]) & on[1:] & on[:-1] & (inp[1:] >= 0) & (inp[:-1] >= 0) & op[1:, None]).sum()
    bs = speed[op]
    return dict(
        goles_10min=goals / mins * 10, remates_min=shots / om, cambios_posesion_min=changes / om, pases_min=passes / om,
        pase_completo=passes / max(passes + changes, 1),
        largo_equipo=float(np.nanmean(np.concatenate(lg))) if len(ks) else np.nan,
        ancho_equipo=float(np.nanmean(np.concatenate(an))) if len(ks) else np.nan,
        profundidad=float(np.nanmean(np.concatenate(pr))) if len(ks) else np.nan,
        vecino_mas_cercano=float(np.mean(np.concatenate(vec))) if len(ks) else np.nan,
        cortes_10min=cuts / om * 10,
        duracion_posesion_s=float(np.mean(spells)) / 60 if spells else np.nan,
        juego_abierto=float(op.mean()), espera_saque=float(np.median(waits)) if waits else np.nan,
        laterales_10min=(lat_new + lat_rep) / mins * 10, lateral_repetido=lat_rep / max(lat_new + lat_rep, 1),
        cargadas_min=int((charged & op).sum()) / om,
        contacto_s_jug_min=touch_p.sum() / 60 / n_on / om,
        racha_p90=float(np.percentile(runs, 90)) if runs else 0.0,
        vel_jugador=float(sp[okv].mean()) if okv.any() else np.nan,
        vel_pelota=float(bs[bs < 40].mean()) if len(bs) else np.nan,
        cambios_tecla_s=float(chg) / max(okv.sum(), 1) * 60,
        mas_cercano_abierto=float(np.median(d.min(1)[op])) if op.any() else np.nan,
    )


def clip_features(c, a, b):
    """Rasgos resumidos de un clip [a, b) para el C2ST (los mismos en sintético y real)."""
    sub = {k: (v[a:b] if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == len(c["phase"]) else v)
           for k, v in c.items()}
    pos, ball, d, near, touch, speed, kick, charged = _events(sub)
    on = sub["on"]
    ph = sub["phase"].astype(np.int64)
    op = ph == OPEN
    sp = np.hypot(*np.diff(pos, axis=0).transpose(2, 0, 1))
    okv = on[1:] & on[:-1] & (sp < 10)
    spv = sp[okv] if okv.any() else np.zeros(1)
    dm = d.min(1)
    inp = sub["inp"].astype(np.int64)
    chg = ((inp[1:] != inp[:-1]) & okv & (inp[1:] >= 0) & (inp[:-1] >= 0)).sum() / max(okv.sum(), 1)
    xs = np.where(on, pos[..., 0], np.nan)
    ys = np.where(on, pos[..., 1], np.nan)
    dfin = d[np.isfinite(d)]
    return np.array([
        op.mean(), (ph == 2).mean(),
        spv.mean(), np.percentile(spv, 90), (spv < 0.3).mean(),
        speed.mean(), np.percentile(speed, 90),
        np.median(dm), np.percentile(dm, 10), (dm <= TOUCH).mean(),
        (dfin < 150).mean(), (dfin < 300).mean(),
        kick.sum(), (charged & op).sum(), chg,
        np.nanmean(np.nanstd(xs, 1)), np.nanmean(np.nanstd(ys, 1)),
        np.abs(ball[:, 1]).mean() / 1000, np.abs(ball[:, 0]).mean() / 1000,
        np.nanmean(np.abs(xs)) / 1000,
    ], np.float32)


# ----------------------------------------------------------------------------- conjuntos
def human_clips(names, H, rng, per_match=4, modos=None):
    """Ventanas grabadas de H ticks desde saques iniciales y momentos de juego abierto (como los sintéticos)."""
    out = []
    for nm in names:
        m = load_match(nm, modos)
        ar = np.arange(m.T)
        ko = np.nonzero((m.phase == KICKOFF) & (ar >= MAX_BACK) & (ar < m.T - H - 1))[0]
        ko = ko[np.r_[True, np.diff(ko) > 1]] if len(ko) else ko
        op = np.nonzero((m.phase == OPEN) & (ar >= MAX_BACK) & (ar < m.T - H - 1))[0]
        cand = list(ko) + list(rng.choice(op, size=min(per_match, len(op)), replace=False))
        for t0 in rng.permutation(cand)[:per_match]:
            out.append((nm, clip_of(window_sp(m, int(t0), H), MAX_BACK)))
    return out


def save_clips(path, clips, meta):
    keys = list(clips[0][1])
    np.savez_compressed(path, nombres=np.array([n for n, _ in clips]), meta=json.dumps(meta),
                        **{k: np.stack([c[k] for _, c in clips]) for k in keys})


def load_clips(path):
    z = np.load(path, allow_pickle=False)
    keys = [k for k in z.files if k not in ("nombres", "meta")]
    n = len(z["nombres"])
    return [(str(z["nombres"][i]), {k: z[k][i] for k in keys}) for i in range(n)], json.loads(str(z["meta"]))


# ----------------------------------------------------------------------------- C2ST
def auc(y, s):
    order = np.argsort(s)
    r = np.empty(len(s))
    r[order] = np.arange(1, len(s) + 1)
    pos = y == 1
    n1, n0 = pos.sum(), (~pos).sum()
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1))


def c2st(Xr, gr, Xs, gs, folds=5, seed=0):
    """AUC de una red chica que separa clips reales (0) de sintéticos (1), con validación cruzada por grupo (partido o
    juego: los clips de una misma ventana no se reparten entre entrenamiento y prueba). Devuelve el AUC y la caída de
    AUC al desordenar cada rasgo (importancia por permutación)."""
    X = np.concatenate([Xr, Xs]).astype(np.float32)
    y = np.r_[np.zeros(len(Xr)), np.ones(len(Xs))]
    g = np.r_[np.asarray(gr), np.asarray(gs) + (max(gr) + 1)]
    rng = np.random.default_rng(seed)
    ug = rng.permutation(np.unique(g))
    fold = {int(u): i % folds for i, u in enumerate(ug)}
    fid = np.array([fold[int(v)] for v in g])
    scores = np.zeros(len(y))
    imp = np.zeros(X.shape[1])
    torch.manual_seed(seed)
    for f in range(folds):
        tr, te = fid != f, fid == f
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        xt = torch.from_numpy((X[tr] - mu) / sd)
        yt = torch.from_numpy(y[tr].astype(np.float32))
        net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 64), torch.nn.ReLU(), torch.nn.Linear(64, 64),
                                  torch.nn.ReLU(), torch.nn.Linear(64, 1))
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
        for _ in range(300):
            perm = torch.randperm(len(xt))
            for j in range(0, len(xt), 256):
                b = perm[j:j + 256]
                loss = torch.nn.functional.binary_cross_entropy_with_logits(net(xt[b])[:, 0], yt[b])
                opt.zero_grad()
                loss.backward()
                opt.step()
        xe = (X[te] - mu) / sd
        with torch.no_grad():
            scores[te] = net(torch.from_numpy(xe))[:, 0].numpy()
            base = auc(y[te], scores[te])
            for k in range(X.shape[1]):
                xp = xe.copy()
                xp[:, k] = rng.permutation(xp[:, k])
                imp[k] += base - auc(y[te], net(torch.from_numpy(xp))[:, 0].numpy())
    return auc(y, scores), imp / folds


FEAT_NAMES = ["juego_abierto", "pelota_parada", "vel_jug_media", "vel_jug_p90", "jug_quietos", "vel_pelota_media",
              "vel_pelota_p90", "cercano_p50", "cercano_p10", "alguien_toca", "jug_a_150", "jug_a_300", "patadas",
              "cargadas", "cambios_tecla", "dispersion_x", "dispersion_y", "pelota_y", "pelota_x", "jug_x"]


# ----------------------------------------------------------------------------- comandos
def cmd_sinteticos(a):
    from learn.hx11_policy import SetPolicy
    from learn.hx11_selfplay import play
    torch.set_num_threads(a.threads)
    sp = json.loads((ROOT / a.split_file).read_text(encoding="utf-8"))
    rng = np.random.default_rng(a.seed)
    names = sorted(rng.permutation(sp["train"])[:a.partidos].tolist())
    modos = json.loads((ROOT / "data" / "hx11" / "modos.json").read_text(encoding="utf-8"))
    matches = [load_match(n, modos) for n in names]
    ck = torch.load(a.model, map_location="cpu", weights_only=False)
    policy = SetPolicy(**ck["config"])
    policy.load_state_dict(ck["state"])
    policy.eval()
    t0 = time.time()
    _, _, traj = play(policy, matches, a.juegos, a.segundos * 60, seed=a.seed, n_env=a.n_env, temp_move=a.temp,
                      delay=a.delay, renew=a.renovar, renew_kick=a.renovar_patada, rule=a.regla_saques,
                      rule_lateral=a.regla_lateral)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(modelo=a.model, juegos=len(traj), segundos=a.segundos, temp=a.temp, renovar=a.renovar,
                renovar_patada=a.renovar_patada, regla_saques=a.regla_saques, regla_lateral=a.regla_lateral,
                retardo=a.delay, segundos_reloj=round(time.time() - t0))
    save_clips(out, [(f"juego{i}", c) for i, c in enumerate(traj)], meta)
    print(json.dumps(meta), flush=True)


def w1(a, b):
    a, b = np.sort(a[np.isfinite(a)]), np.sort(b[np.isfinite(b)])
    if not len(a) or not len(b):
        return np.nan
    q = np.linspace(0, 1, 201)
    return float(np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q))))


def cmd_puerta(a):
    torch.set_num_threads(a.threads)
    syn, meta = load_clips(a.sinteticos)
    H = len(syn[0][1]["phase"])
    modos = json.loads((ROOT / "data" / "hx11" / "modos.json").read_text(encoding="utf-8"))
    rng = np.random.default_rng(a.seed)
    cache = ROOT / "runs" / "hx11_l0" / "realismo" / f"humanos_{H}.npz"
    if cache.exists():
        hum, hm = load_clips(cache)
    else:
        test = l0_split()["test"]
        sp = json.loads((ROOT / a.split_file).read_text(encoding="utf-8"))
        train = sorted(rng.permutation(sp["train"])[:len(test)].tolist())
        hum = ([("prueba:" + n, c) for n, c in human_clips(test, H, rng, a.por_partido, modos)]
               + [("entrenamiento:" + n, c) for n, c in human_clips(train, H, rng, a.por_partido, modos)])
        cache.parent.mkdir(parents=True, exist_ok=True)
        save_clips(cache, hum, dict(H=H))
    test_c = [c for n, c in hum if n.startswith("prueba:")]
    train_c = [c for n, c in hum if n.startswith("entrenamiento:")]
    syn_c = [c for _, c in syn]
    M = {k: [clip_metrics(c) for c in v] for k, v in (("sintetico", syn_c), ("prueba", test_c),
                                                      ("entrenamiento", train_c))}
    rows, ok_n, coll_bad = [], 0, []
    for fam, keys in FAMILIAS.items():
        for k in keys:
            vs, vt, vr = (np.array([r[k] for r in M[x]], np.float64) for x in ("sintetico", "prueba", "entrenamiento"))
            d, ceil = w1(vs, vt), w1(vr, vt)
            ratio = d / max(ceil, 1e-9)
            ok = bool(d <= ceil)
            ok_n += ok
            if fam == "colectiva" and ratio > 2:
                coll_bad.append(k)
            rows.append(dict(familia=fam, metrica=k, sintetico=float(np.nanmean(vs)), prueba=float(np.nanmean(vt)),
                             entrenamiento=float(np.nanmean(vr)), d=d, techo=ceil, veces_techo=ratio, dentro=ok))
    frac = ok_n / len(rows)
    # C2ST sobre clips de 10 s
    def feats(clips):
        X, g = [], []
        for j, c in enumerate(clips):
            for s0 in range(0, H - CLIP_C2ST + 1, CLIP_C2ST):
                X.append(clip_features(c, s0, s0 + CLIP_C2ST))
                g.append(j)
        return np.stack(X), np.array(g)
    Xr, gr = feats(test_c)
    Xs, gs = feats(syn_c)
    A, imp = c2st(Xr, gr, Xs, gs, seed=a.seed)
    top = sorted(zip(FEAT_NAMES, imp.tolist()), key=lambda x: -x[1])[:8]
    passed = frac >= 0.8 and not coll_bad and A <= 0.70
    res = dict(sinteticos=a.sinteticos, meta=meta, clips=dict(sintetico=len(syn_c), prueba=len(test_c),
                                                              entrenamiento=len(train_c)),
               dentro_del_techo=round(frac, 3), colectivas_a_mas_de_2_techos=coll_bad, c2st_auc=round(A, 3),
               c2st_rasgos=[(n, round(v, 4)) for n, v in top], pasa=passed, metricas=rows)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    Path(str(out) + ".json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    md = [f"# Puerta de realismo P5: {Path(a.sinteticos).stem}", "",
          f"- Sintéticos: {len(syn_c)} juegos de {H // 60} s ({meta.get('modelo')}); referencia: {len(test_c)} ventanas "
          f"humanas de prueba; techo: {len(train_c)} de entrenamiento contra prueba.",
          f"- **Dentro del techo: {frac:.0%}** (criterio ≥ 80%). Colectivas a más de 2 techos: "
          f"{', '.join(coll_bad) or 'ninguna'}.",
          f"- **C2ST (clips de 10 s): AUC {A:.3f}** (criterio ≤ 0,70). Rasgos que más delatan: "
          + ", ".join(f"{n} ({v:.3f})" for n, v in top[:5]) + ".",
          f"- **{'Pasa' if passed else 'No pasa'}.**", "",
          "| familia | métrica | sintético | prueba | entrenamiento | d | techo | veces | dentro |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['familia']} | {r['metrica']} | {r['sintetico']:.3f} | {r['prueba']:.3f} | "
                  f"{r['entrenamiento']:.3f} | {r['d']:.3f} | {r['techo']:.3f} | {r['veces_techo']:.1f} | "
                  f"{'sí' if r['dentro'] else '**no**'} |")
    Path(str(out) + ".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "metricas"}, ensure_ascii=False), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sinteticos")
    s.add_argument("--model", required=True)
    s.add_argument("--split-file", default="runs/hx11_l0/split.json")
    s.add_argument("--partidos", type=int, default=60)
    s.add_argument("--juegos", type=int, default=200)
    s.add_argument("--segundos", type=int, default=90)
    s.add_argument("--n-env", type=int, default=8)
    s.add_argument("--temp", type=float, default=0.7)
    s.add_argument("--renovar", type=float, default=0.33)
    s.add_argument("--renovar-patada", type=float, default=0.33)
    s.add_argument("--delay", type=int, default=None)
    s.add_argument("--regla-saques", action="store_true")
    s.add_argument("--regla-lateral", action="store_true")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--threads", type=int, default=4)
    s.add_argument("--out", required=True)
    p = sub.add_parser("puerta")
    p.add_argument("--sinteticos", required=True)
    p.add_argument("--split-file", default="runs/hx11_l0/split.json")
    p.add_argument("--por-partido", type=int, default=4, help="ventanas humanas por partido")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    cmd_sinteticos(a) if a.cmd == "sinteticos" else cmd_puerta(a)


if __name__ == "__main__":
    main()
