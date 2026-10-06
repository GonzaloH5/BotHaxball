"""Métricas de juego colectivo y de parecido humano (E0 del plan), iguales para grabaciones y simulación.

Entrada común (`Episode`): estados muestreados cada `stride` ticks (3 = una decisión) de tramos 4v4, con
pelota, jugadores (lugares 0..3 rojos, 4..7 azules), dirección/patada de la entrada y patadas. Las
grabaciones se leen del caché `tools.x4_ticks`; la simulación se graba con `EpisodeRecorder` desde
`RS4ZEnv`. Así el humano y la política se miden con el mismo código (WOSAC, arXiv 2305.12032: métricas
distribucionales contra los logs; Liu 2022, arXiv 2105.12196: pases, largo de pase).

Familias de métricas (muestras por partido o por tick, todas en el marco propio de cada equipo):
* posesión y pase: pases/min, largo de pase, pases por posesión, duración de la posesión, pérdidas/min;
* forma del equipo: profundidad, ancho, centroide respecto de la pelota, distancias a la pelota ordenadas
  (1.º..4.º más cercano), distancia al compañero más cercano, jugadores detrás de la pelota;
* actividad: % de jugadores parados, velocidad, cambios de tecla por segundo, patadas por minuto,
  % de ticks sin dirección;
* pelota: x de la pelota (territorio), velocidad, goles por minuto;
* saques: duración desde la colocación hasta la liberación por tipo.

Distancia entre dos conjuntos: Wasserstein-1 por métrica, normalizada por el desvío de la referencia.

  python -m tools.x4_metrics --ticks data/x4_ticks --splits reports/x4/splits.json --map sanguchito_rs_x4 \\
      --out reports/x4/human_metrics.json
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TICK_S = 1.0 / 60.0
PLAYER_R = 15.0
CONTACT_EPS = 2.0          # px de margen para contar contacto con la pelota en un muestreo
STILL_SPEED = 0.1          # px/tick: jugador parado
MIN_PASS = 40.0            # px que tiene que recorrer la pelota para contar un pase
MOVE_DX = np.array([0, 0, 1, 1, 1, 0, -1, -1, -1])
MOVE_DY = np.array([0, -1, -1, 0, 1, 1, 1, 0, -1])


@dataclass
class Episode:
    """Tramo 4v4 muestreado cada `stride` ticks. Coordenadas del mundo (rojo ataca +x)."""
    stride: int
    ball: np.ndarray            # (T, 4) x, y, vx, vy
    ball_r: float
    pos: np.ndarray             # (T, 8, 2)
    vel: np.ndarray             # (T, 8, 2)
    move: np.ndarray            # (T, 8) 0..8 dirección de la entrada (mundo)
    kick_key: np.ndarray        # (T, 8) bool tecla de patada
    kicked: np.ndarray          # (T, 8) bool patada efectiva en la ventana del muestreo
    open_play: np.ndarray       # (T,) bool juego abierto (sin saque, sin saque inicial, sin cobro pendiente)
    restart_kind: np.ndarray    # (T,) 0..3
    restart_age: np.ndarray     # (T,)
    kickoff: np.ndarray         # (T,) bool
    goals: int = 0
    segment: np.ndarray = None  # (T,) id de tramo continuo (cortes de la grabación o del plantel)
    meta: dict = field(default_factory=dict)
    restart_team: np.ndarray = None  # (T,) equipo que saca (-1 sin saque, 0 rojo, 1 azul)
    goal_ev: np.ndarray = None       # (T,) +1 gol rojo, -1 gol azul en la ventana del muestreo, 0 nada


# ------------------------------------------------------------------------------------- humanos
def _move_of_input(inp):
    inp = inp.astype(np.int64)
    dx = ((inp >> 3) & 1) - ((inp >> 2) & 1)
    dy = ((inp >> 1) & 1) - (inp & 1)
    table = {(int(x), int(y)): i for i, (x, y) in enumerate(zip(MOVE_DX, MOVE_DY))}
    out = np.zeros(inp.shape, np.int64)
    for (x, y), i in table.items():
        out[(dx == x) & (dy == y)] = i
    return out


def episodes_from_ticks(npz, map_id=None, stride=3):
    """Tramos 4v4 estables de una grabación del caché `x4_ticks` (uno por plantel continuo)."""
    d = dict(np.load(npz)) if not isinstance(npz, dict) else npz
    frame = d["frame"]
    finite = np.isfinite(d["ball"][:, :5]).all(1) & np.isfinite(d["pos"]).all((1, 2)) & np.isfinite(d["vel"]).all((1, 2))
    ok = (d["n_red"] == 4) & (d["n_blue"] == 4) & np.isin(d["state"], (0, 1)) & finite
    if map_id is not None:
        ok &= d["map_id"] == map_id
    pid = d["pid"]
    T = len(frame)
    # cortes: hueco de frames, cambio de plantel o fuera de `ok`
    brk = np.ones(T, bool)
    brk[1:] = (np.diff(frame) != 1) | np.any(pid[1:] != pid[:-1], axis=1) | ~ok[1:] | ~ok[:-1]
    seg = np.cumsum(brk)
    # patadas por (frame, id) → lugar
    kick_slot = np.zeros((T, 8), bool)
    if len(d["kicks"]):
        idx = {int(f): i for i, f in enumerate(frame)}
        for f, p in d["kicks"]:
            i = idx.get(int(f))
            if i is None:
                continue
            hit = np.flatnonzero(pid[i] == p)
            if len(hit):
                kick_slot[i, hit[0]] = True
    out = []
    for s in np.unique(seg[ok]):
        rows = np.flatnonzero((seg == s) & ok)
        if len(rows) < 60 * stride:
            continue
        # muestreo cada `stride` frames; la patada cuenta si ocurrió en la ventana [f, f+stride)
        start = rows[0] + (-frame[rows[0]]) % stride
        samp = np.arange(start, rows[-1] + 1, stride)
        kicked = np.zeros((len(samp), 8), bool)
        for j in range(stride):
            kicked |= kick_slot[np.minimum(samp + j, rows[-1])]
        st = d["state"][samp]
        rk = d["restart_kind"][samp]
        pend = d["pending"][samp]
        goal_ev = np.zeros(len(samp), np.int8)
        for g, team in (d["goals"] if len(d["goals"]) else ()):
            if frame[rows[0]] <= g <= frame[rows[-1]]:
                j = int(np.searchsorted(frame[samp], g, side="right")) - 1
                if 0 <= j < len(samp):
                    goal_ev[j] = 1 if team == 1 else -1
        ep = Episode(stride=stride, ball=d["ball"][samp, :4].astype(np.float64), ball_r=float(np.median(d["ball"][rows, 4])),
                     pos=d["pos"][samp].astype(np.float64), vel=d["vel"][samp].astype(np.float64),
                     move=_move_of_input(d["inp"][samp]), kick_key=(d["inp"][samp] & 16) != 0, kicked=kicked,
                     open_play=(st == 1) & (rk == 0) & (pend == 0), restart_kind=rk.astype(np.int64),
                     restart_age=d["restart_age"][samp].astype(np.int64), kickoff=st == 0,
                     goals=int(np.sum(np.isin(d["goals"][:, 0], frame[rows]))) if len(d["goals"]) else 0,
                     segment=np.zeros(len(samp), np.int64), meta=dict(frames=(int(frame[rows[0]]), int(frame[rows[-1]]))),
                     restart_team=np.where(rk > 0, d["restart_team"][samp], -1).astype(np.int64), goal_ev=goal_ev)
        out.append(ep)
    return out


# ------------------------------------------------------------------------------------- simulación
class EpisodeRecorder:
    """Graba un partido de `RS4ZEnv` (fila `row`) en el formato `Episode`. Llamar `pre(env)` antes y
    `record(env, ev, actions)` después de cada `env.step(actions)` (acciones en el marco propio).

    Alineación con las grabaciones: en `episodes_from_ticks` el muestreo t es el estado en el frame f y las patadas
    y goles de la ventana [f, f+3). Acá el estado se toma antes del paso (`pre`) y las patadas y goles son los de ese
    paso. Sin `pre`, se usa el estado después del paso (versión anterior: la pelota pateada ya salió del pie)."""

    def __init__(self, env, row=0):
        self.env, self.row = env, row
        self.rows = []
        self._pre = None

    def _state(self, env):
        from env.rs4z import kernel as K
        n = self.row
        ko = env.ri[n, K.RI_KO] != 0
        rk = int(env.ri[n, K.RI_KIND]) if env.ri[n, K.RI_TEAM] >= 0 else 0
        pend = int(env.ri[n, K.RI_PEND]) > 0
        return (env.pos[n, 0].copy(), env.vel[n, 0].copy(), env.player_pos[n].copy(), env.player_vel[n].copy(),
                (not ko) and rk == 0 and not pend, rk, int(env.ri[n, K.RI_TICKS]), ko,
                int(env.ri[n, K.RI_TEAM]) if rk > 0 else -1)

    def pre(self, env):
        self._pre = self._state(env)

    def record(self, env, ev, actions):
        from env.rs4z.core import MIRROR_ACTION
        n = self.row
        world = np.asarray(actions[n], dtype=np.int64).copy()
        world[4:] = MIRROR_ACTION[world[4:]]
        bp, bv, pp, pv, open_, rk, rticks, ko, rteam = self._pre if self._pre is not None else self._state(env)
        self._pre = None
        self.rows.append((bp, bv, pp, pv, world % 9, world >= 9, ev["kicked"][n].copy(), open_, rk, rticks, ko,
                          int(np.sign(ev["goal"][n])), rteam))

    def episode(self):
        r = self.rows
        cols = list(zip(*r))
        return Episode(stride=self.env.frame_skip, ball=np.hstack([np.array(cols[0]), np.array(cols[1])]),
                       ball_r=float(self.env.radius[self.row, 0]), pos=np.array(cols[2]), vel=np.array(cols[3]),
                       move=np.array(cols[4]), kick_key=np.array(cols[5]), kicked=np.array(cols[6]),
                       open_play=np.array(cols[7]), restart_kind=np.array(cols[8]), restart_age=np.array(cols[9]),
                       kickoff=np.array(cols[10]), goals=int(sum(abs(g) for g in cols[11])),
                       segment=np.zeros(len(r), np.int64), restart_team=np.array(cols[12], np.int64),
                       goal_ev=np.array(cols[11], np.int8))


def episodes_from_npz(path, stride=3):
    """Partidos de self-play guardados por el trainer en <out>/evals/ep_*.npz (`Trainer.keep_eval`)."""
    d = np.load(path)
    idx = sorted({int(k.split("/")[0]) for k in d.files})
    out = []
    for i in idx:
        g = lambda k, dt=None: (d[f"{i}/{k}"].astype(dt) if dt else d[f"{i}/{k}"]) if f"{i}/{k}" in d.files else None
        T = len(d[f"{i}/ball"])
        out.append(Episode(stride=stride, ball=g("ball", np.float64), ball_r=float(d[f"{i}/ball_r"]),
                           pos=g("pos", np.float64), vel=g("vel", np.float64), move=g("move", np.int64),
                           kick_key=g("kick_key"), kicked=g("kicked"), open_play=g("open_play"),
                           restart_kind=g("restart_kind", np.int64), restart_age=g("restart_age", np.int64),
                           kickoff=g("kickoff"), goals=int(np.abs(g("goal_ev", np.int64)).sum()) if g("goal_ev") is not None else 0,
                           segment=np.zeros(T, np.int64), restart_team=g("restart_team", np.int64),
                           goal_ev=g("goal_ev", np.int8)))
    return out


# ------------------------------------------------------------------------------------- métricas
def _touch_owner(ep):
    """Dueño de la pelota en cada muestreo: lugar con patada en la ventana o en contacto; -1 si ninguno,
    -2 si hay contacto de los dos equipos."""
    d = np.hypot(ep.pos[:, :, 0] - ep.ball[:, None, 0], ep.pos[:, :, 1] - ep.ball[:, None, 1])
    contact = d <= PLAYER_R + ep.ball_r + CONTACT_EPS
    touch = contact | ep.kicked
    owner = np.full(len(d), -1, np.int64)
    red = touch[:, :4].any(1)
    blue = touch[:, 4:].any(1)
    both = red & blue
    one = touch.any(1) & ~both
    # entre varios del mismo equipo, el más cercano (la patada manda)
    score = np.where(ep.kicked, -1.0, d)
    score = np.where(touch, score, np.inf)
    owner[one] = np.argmin(score[one], axis=1)
    owner[both] = -2
    return owner, d


def possession_sequence(ep):
    """Pases, pérdidas y posesiones de un tramo (una sola definición para grabaciones y simulación).

    Toques en el orden en que ocurren (dueño de la pelota en cada muestreo, `_touch_owner`). Entre dos toques
    seguidos de jugadores distintos: mismo equipo y la pelota recorrió ≥ MIN_PASS en juego abierto → pase;
    distinto equipo → pérdida (incluye despejes y pelotas divididas). Una posesión dura del primer toque de un
    equipo al primer toque del otro.
    """
    owner, dist = _touch_owner(ep)
    T = len(ep.ball)
    open_ = ep.open_play
    seq = []   # (índice, lugar)
    for i in range(T):
        o = owner[i]
        if o >= 0 and (not seq or seq[-1][1] != o):
            seq.append((i, int(o)))
        elif o == -2:
            seq.append((i, -2))
    passes, pass_len, losses, poss_len, poss_passes = 0, [], 0, [], []
    clean = [(i, o) for i, o in seq if o >= 0]
    cur_team, cur_start, cur_passes = None, None, 0
    for k, (i, o) in enumerate(clean):
        t = o // 4
        if cur_team is None:
            cur_team, cur_start, cur_passes = t, i, 0
            continue
        i0, a = clean[k - 1]
        if t != cur_team:
            losses += 1
            poss_len.append((i - cur_start) * ep.stride * TICK_S)
            poss_passes.append(cur_passes)
            cur_team, cur_start, cur_passes = t, i, 0
        elif a != o:
            travel = float(np.hypot(*(ep.ball[i, :2] - ep.ball[i0, :2])))
            if travel >= MIN_PASS and open_[i0:i + 1].all():
                passes += 1
                pass_len.append(travel)
                cur_passes += 1
    return dict(passes=passes, losses=losses, pass_len=pass_len, poss_len=poss_len, poss_passes=poss_passes,
                owner=owner, dist=dist)


def episode_counts(ep):
    """Totales de un tramo para tasas agregadas (suma de eventos / suma de minutos) e intervalos por bootstrap."""
    ps = possession_sequence(ep)
    minutes = len(ep.ball) * ep.stride * TICK_S / 60.0
    return dict(minutes=minutes, open_minutes=float(ep.open_play.sum()) * ep.stride * TICK_S / 60.0,
                passes=ps["passes"], losses=ps["losses"], goals=int(ep.goals), kicks=float(ep.kicked.sum()),
                possessions=len(ps["poss_len"]), pass_len=ps["pass_len"])


def episode_samples(ep):
    """Muestras de cada métrica para un tramo (listas de floats) y totales para las tasas."""
    T = len(ep.ball)
    minutes = T * ep.stride * TICK_S / 60.0
    open_ = ep.open_play
    s = {}
    ps = possession_sequence(ep)
    dist = ps["dist"]
    passes, losses, pass_len, poss_len, poss_passes = (ps[k] for k in ("passes", "losses", "pass_len", "poss_len",
                                                                         "poss_passes"))
    s["passes_per_min"] = [passes / minutes] if minutes > 0 else []
    s["losses_per_min"] = [losses / minutes] if minutes > 0 else []
    s["pass_length"] = pass_len
    s["possession_s"] = poss_len
    s["passes_per_possession"] = poss_passes
    s["goals_per_min"] = [ep.goals / minutes] if minutes > 0 else []
    # --- forma del equipo en juego abierto (marco propio)
    idx = np.flatnonzero(open_)
    shape = {k: [] for k in ("depth", "width", "centroid_minus_ball_x", "dist_ball_1", "dist_ball_2", "dist_ball_3",
                             "dist_ball_4", "nearest_mate", "behind_ball", "ball_x_own")}
    for team, sl, sign in ((0, slice(0, 4), 1.0), (1, slice(4, 8), -1.0)):
        px = ep.pos[idx, sl, 0] * sign
        py = ep.pos[idx, sl, 1]
        bx = ep.ball[idx, 0] * sign
        shape["depth"] += list(px.max(1) - px.min(1))
        shape["width"] += list(py.max(1) - py.min(1))
        shape["centroid_minus_ball_x"] += list(px.mean(1) - bx)
        dd = np.sort(dist[idx, sl], axis=1)
        for k in range(4):
            shape[f"dist_ball_{k + 1}"] += list(dd[:, k])
        pp = np.stack([px, py], -1)
        m = np.hypot(pp[:, :, None, 0] - pp[:, None, :, 0], pp[:, :, None, 1] - pp[:, None, :, 1])
        m[:, np.arange(4), np.arange(4)] = np.inf
        shape["nearest_mate"] += list(m.min(2).ravel())
        shape["behind_ball"] += list((px < bx[:, None]).sum(1).astype(float))
        shape["ball_x_own"] += list(bx)
    # submuestreo de las métricas por tick (cada 10 muestreos: ~0,5 s) para no inflar la referencia
    for k, v in shape.items():
        s[k] = v[::10]
    # --- actividad
    speed = np.hypot(ep.vel[idx, :, 0], ep.vel[idx, :, 1])
    s["player_speed"] = list(speed.ravel()[::40])
    s["still_frac"] = [float((speed < STILL_SPEED).mean())] if len(idx) else []
    s["no_direction_frac"] = [float((ep.move[idx] == 0).mean())] if len(idx) else []
    changes = (np.diff(ep.move, axis=0) != 0) | (np.diff(ep.kick_key.astype(np.int8), axis=0) != 0)
    s["key_changes_per_s"] = [float(changes.sum() / (8 * T * ep.stride * TICK_S))] if T > 1 else []
    s["kicks_per_min"] = [float(ep.kicked.sum() / 8 / minutes)] if minutes > 0 else []
    bspeed = np.hypot(ep.ball[idx, 2], ep.ball[idx, 3])
    s["ball_speed"] = list(bspeed[::10])
    # --- saques: edad al liberarse (el muestreo siguiente ya no tiene saque)
    for kind, name in ((1, "lateral"), (2, "corner"), (3, "goal_kick")):
        rel = np.flatnonzero((ep.restart_kind[:-1] == kind) & (ep.restart_kind[1:] != kind))
        s[f"restart_s_{name}"] = list(ep.restart_age[rel] * TICK_S)
    s["kickoff_wait_s"] = []
    ko = ep.kickoff.astype(np.int8)
    starts = np.flatnonzero(np.diff(np.concatenate([[0], ko])) == 1)
    ends = np.flatnonzero(np.diff(np.concatenate([ko, [0]])) == -1)
    for a, b in zip(starts, ends):
        if b + 1 < T:   # el saque inicial terminó dentro del tramo
            s["kickoff_wait_s"].append((b - a + 1) * ep.stride * TICK_S)
    return s, minutes


def merge(samples_list):
    out = {}
    for s in samples_list:
        for k, v in s.items():
            out.setdefault(k, []).extend(float(x) for x in v)
    return out


def wasserstein1(a, b):
    a, b = np.sort(np.asarray(a, float)), np.sort(np.asarray(b, float))
    if len(a) == 0 or len(b) == 0:
        return None
    q = np.linspace(0.0, 1.0, 201)[1:-1]
    return float(np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q))))


def summarize(samples):
    out = {}
    for k, v in samples.items():
        v = np.asarray(v, float)
        if len(v) == 0:
            out[k] = dict(n=0)
            continue
        out[k] = dict(n=int(len(v)), mean=float(v.mean()), std=float(v.std()),
                      p10=float(np.percentile(v, 10)), p50=float(np.percentile(v, 50)), p90=float(np.percentile(v, 90)))
    return out


def compare(candidate, reference):
    """W1 por métrica, normalizada por el desvío de la referencia."""
    out = {}
    for k, ref in reference.items():
        cand = candidate.get(k, [])
        w = wasserstein1(cand, ref)
        sd = float(np.std(ref)) if len(ref) else 0.0
        out[k] = dict(w1=w, w1_norm=None if w is None or sd == 0 else w / sd, n_cand=len(cand), n_ref=len(ref))
    return out


def recording_samples(path, map_id, stride=3):
    eps = episodes_from_ticks(path, map_id=map_id, stride=stride)
    ss, minutes = [], 0.0
    for ep in eps:
        s, m = episode_samples(ep)
        ss.append(s)
        minutes += m
    return merge(ss), minutes


def _job(args):
    path, map_id = args
    try:
        s, m = recording_samples(path, map_id)
        return path, s, m, None
    except Exception as error:
        return path, None, 0.0, f"{type(error).__name__}: {error}"


def main():
    from tools.x4_ticks import MAP_IDS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ticks", default=str(ROOT / "data" / "x4_ticks"))
    ap.add_argument("--splits", default=str(ROOT / "reports" / "x4" / "splits.json"))
    ap.add_argument("--map", default="sanguchito_rs_x4")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=str(ROOT / "reports" / "x4" / "human_metrics.json"))
    ap.add_argument("--samples-out", default="", help="muestras crudas (por defecto data/<out>.samples.npz)")
    a = ap.parse_args()
    splits = json.loads(Path(a.splits).read_text(encoding="utf-8"))["recordings"]
    map_id = MAP_IDS[a.map]
    jobs, split_of = [], {}
    for name, row in splits.items():
        p = Path(a.ticks) / f"{Path(name).stem}.npz"
        if p.exists() and a.map in (row.get("maps") or {}):
            jobs.append((str(p), map_id))
            split_of[str(p)] = row["split"]
    from multiprocessing import get_context
    by_split, minutes = {}, {}
    with get_context("fork").Pool(a.workers) as pool:
        for path, s, m, err in pool.imap_unordered(_job, jobs):
            if err:
                print("error", path, err, flush=True)
                continue
            sp = split_of[path]
            by_split.setdefault(sp, []).append(s)
            minutes[sp] = minutes.get(sp, 0.0) + m
    merged = {k: merge(v) for k, v in by_split.items()}
    ref = merged.get("train", {})
    report = dict(version="x4-metrics-1", map=a.map, minutes=minutes,
                  summary={k: summarize(v) for k, v in merged.items()},
                  ceiling={k: compare(v, ref) for k, v in merged.items() if k != "train"})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    np.savez_compressed(Path(a.samples_out or ROOT / "data" / (Path(a.out).stem + ".samples.npz")),
                        **{f"{sp}/{k}": np.asarray(v, np.float32) for sp, d in merged.items() for k, v in d.items()})
    print(json.dumps(dict(minutes=minutes, ceiling={sp: {k: (round(v["w1_norm"], 3) if v["w1_norm"] is not None else None)
                                                          for k, v in c.items()} for sp, c in report["ceiling"].items()}),
                     indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
