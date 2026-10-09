"""D0 de HaxEleven (docs/PLAN_HAXELEVEN.md): caché compacto por tick de cada partido, para la imitación y las etiquetas.

Por partido, `data/hx11/d0/<nombre>.npz` (no versionado) con:

* `frame`, `state`, `score`, `time`, `offside_on`.
* Pelota: `ball_pos`, `ball_vel` (int16: ×10 y ×1000), `ball_inv` (masa inversa: carga del powershot o barra, float16),
  `ball_g` (gravedad = comba, float16).
* 22 lugares fijos (0–10 rojos, 11–21 azules): cada jugador conserva su lugar mientras está en la cancha en el mismo
  equipo. `slot_pid` (id del jugador, −1 vacío), `pos`, `vel` (int16, igual que la pelota), `input` (teclas, −1 vacío),
  `kick` (patada en ese tick).
* `gk_pid` (T, 2): arquero declarado de cada equipo con `!gk` (anuncio "X es el Arquero del Red/Blue"), −1 si no hay.
* Fase de juego por tick, con los modos del árbitro del simulador (`env/hx11/kernel.py`): `phase` (0 juego abierto,
  1 salida pendiente, 2 pelota parada, 3 congelamiento de falta u offside, 4 gol, 5 saque inicial), `piece_kind`
  (`env/hx11/contract.py`), `piece_team` (0 rojo, 1 azul, −1), `phase_ticks` (ticks desde que empezó) y `spot`
  (punto del saque, int16 ×10). Sale de las decisiones de la sala (`tools.hx11_conformance.room_decisions`), las
  patadas y los goles: es la verdad de la sala, no la del árbitro del simulador.

Y `<nombre>.json`: estadio, decisiones de la sala, patadas (tick, lugar, masa de la pelota antes), anuncios con números
(powershot utilizado, potencia de la barra) y cobertura. Sin nombres de jugadores.

  node bridge/hx11_ticks.js <grabación> --max-minutes 0 --stdout | python -m tools.hx11_d0 uno - --out data/hx11/d0/<nombre>
  python -m tools.hx11_d0 lote --stadium "HaxEleven RSx11 [1800, 1200]" --jobs 4
  python -m tools.hx11_d0 resumen        # reports/hx11/d0_cache.{md,json} (solo agregados)
"""
from __future__ import annotations

import argparse
import collections
import io
import json
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from env.hx11 import contract as HC
from tools.hx11_npz import convert

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "hx11" / "d0"
N_SLOTS, N_TEAM = 22, 11
OPEN, OUT_PENDING, PIECE, FREEZE, GOAL, KICKOFF = 0, 1, 2, 3, 4, 5
GK_BLOCK = 3600           # bloques de 1 min para inferir el arquero por posición (con 10 s: 89% contra el declarado)
MIN_SIDE = 9              # jugadores en el equipo más chico (mediana en juego abierto) para considerar el partido 11v11
GK_ANN = re.compile(r"^\[✅\]\s+(.+?)\s+es el Arquero del .*?(Red|Blue)")
PS_ANN = re.compile(r"Powershot utilizado\. (\d+)%")
POWER_ANN = re.compile(r"Potencia del (corner|saque|tiro libre|penal\w*): (\d+)%")


def q16(x, scale):
    """Cuantización a int16 (NaN → 0; fuera de rango, recortado)."""
    v = np.nan_to_num(np.asarray(x, np.float64) * scale, nan=0.0)
    return np.clip(np.round(v), -32767, 32767).astype(np.int16)


def slots(team):
    """Lugar fijo de cada columna de la grabación por tick: (T, 22) con el índice de columna o −1."""
    T, P = team.shape
    out = np.full((T, N_SLOTS), -1, np.int16)
    slot_of = {}
    free = {1: list(range(N_TEAM)), 2: list(range(N_TEAM, N_SLOTS))}
    for i in range(T):
        now = {c: int(team[i, c]) for c in np.nonzero(team[i] > 0)[0]}
        for c in list(slot_of):
            s = slot_of[c]
            if now.get(c) != (1 if s < N_TEAM else 2):
                del slot_of[c]
                free[1 if s < N_TEAM else 2].append(s)
                free[1 if s < N_TEAM else 2].sort()
        for c, t in now.items():
            if c not in slot_of and free[t]:
                slot_of[c] = free[t].pop(0)
        for c, s in slot_of.items():
            out[i, s] = c
    return out


def infer_gk(pos, slot_pid, block=GK_BLOCK):
    """Arquero de cada equipo por posición: el jugador más cercano a su arco, el más frecuente en bloques de 1 min."""
    T = len(pos)
    out = np.full((T, 2), -1, np.int32)
    for t in range(2):
        sl = slice(t * N_TEAM, (t + 1) * N_TEAM)
        gx = -HC.LINE_W if t == 0 else HC.LINE_W
        p = pos[:, sl].astype(np.float32) / 10
        d = np.hypot(p[..., 0] - gx, p[..., 1])
        d[slot_pid[:, sl] < 0] = np.inf
        ok = np.isfinite(d).any(1)
        near = slot_pid[:, sl][np.arange(T), d.argmin(1)]
        for b in range(0, T, block):
            w = near[b:b + block][ok[b:b + block]]
            if len(w):
                v, c = np.unique(w, return_counts=True)
                out[b:b + block, t] = v[c.argmax()]
        # sólo vale mientras ese jugador está en el equipo
        out[:, t] = np.where((slot_pid[:, sl] == out[:, t:t + 1]).any(1), out[:, t], -1)
    return out


def build(arrays, name):
    """Arreglos y resumen del D0 de un partido a partir de la salida de `tools.hx11_npz.convert(..., return_arrays)`."""
    from tools.hx11_conformance import room_decisions
    from tools.hx11_script import Match, offside_on

    z, events, header = arrays
    m = Match(arrays=arrays, name=name)
    T = m.T
    team = z["pl_team"]
    sl = slots(team)
    valid = sl >= 0
    col = np.where(valid, sl, 0).astype(np.int64)
    ti = np.arange(T)[:, None]
    pid = np.asarray(z["pid"])
    out = dict(
        frame=z["frame"].astype(np.int32), state=z["state"].astype(np.int8), score=z["score"].astype(np.int16),
        time=z["time"].astype(np.float32),
        ball_pos=q16(z["ball_pos"], 10), ball_vel=q16(z["ball_vel"], 1000),
        ball_inv=np.nan_to_num(z["ball_invMass"], nan=HC.BASE_BALL_INV).astype(np.float16),
        ball_g=np.stack([np.nan_to_num(z["ball_gx"]), np.nan_to_num(z["ball_gy"])], axis=1).astype(np.float16),
        slot_pid=np.where(valid, pid[col], -1).astype(np.int32),
        pos=q16(np.where(valid[..., None], z["pl_pos"][ti, col], 0.0), 10),
        vel=q16(np.where(valid[..., None], z["pl_vel"][ti, col], 0.0), 1000),
        input=np.where(valid, z["pl_input"][ti, col], -1).astype(np.int8),
    )
    kick = np.zeros((T, N_SLOTS), np.int8)
    for i, c in m.kicks:
        s = np.nonzero(sl[i] == c)[0]
        if len(s):
            kick[i, s[0]] = 1
    out["kick"] = kick
    # arquero declarado
    names = {p["id"]: p["name"] for p in (header or {}).get("players", [])}
    for e in events:
        if e["name"] == "join":
            names[e["id"]] = e.get("player")
    gk = np.full((T, 2), -1, np.int32)
    cur = [-1, -1]
    # el !gk suele escribirse con el partido detenido (la sala no arranca sin arquero): va al primer tick posterior
    decl = sorted((int(np.searchsorted(z["frame"], e["frame"])), e["msg"]) for e in events
                  if e["name"] == "announcement" and GK_ANN.match(e["msg"]))
    decl = [(i, msg) for i, msg in decl if i < T]
    k = 0
    for i in range(T):
        while k < len(decl) and decl[k][0] <= i:
            g = GK_ANN.match(decl[k][1])
            t = 0 if g.group(2) == "Red" else 1
            cand = [p for p, nm in names.items() if nm is not None and nm.strip() == g.group(1).strip()]
            on = [p for p in cand if p in set(out["slot_pid"][i, t * N_TEAM:(t + 1) * N_TEAM].tolist())]
            cur[t] = on[0] if on else (cand[-1] if cand else -1)
            k += 1
        for t in range(2):
            if cur[t] >= 0 and cur[t] not in out["slot_pid"][i, t * N_TEAM:(t + 1) * N_TEAM]:
                cur[t] = -1
        gk[i] = cur
    out["gk_pid"] = gk
    # sin declaración (versiones del script sin !gk): el arquero inferido por posición
    inf = infer_gk(out["pos"], out["slot_pid"])
    out["gk_inferred"] = inf
    out["gk_src"] = np.where(gk >= 0, 1, np.where(inf >= 0, 2, 0)).astype(np.int8)
    # fase de juego (verdad de la sala)
    phase = np.full(T, OPEN, np.int8)
    kind = np.zeros(T, np.int8)
    pteam = np.full(T, -1, np.int8)
    pticks = np.zeros(T, np.int16)
    spot = np.zeros((T, 2), np.int16)
    st = z["state"]
    bp = z["ball_pos"]
    dec = room_decisions(m)
    kicks_by_team = collections.defaultdict(list)
    for i, c in m.kicks:
        kicks_by_team[m.team(i, c) - 1].append(i)
    stops = sorted([d[0] for d in dec] + [m.at(e["frame"]) or 0 for e in events if e["name"] == "goal"]
                   + list(np.nonzero(np.diff(st.astype(np.int16), prepend=st[0]) != 0)[0]))

    def next_stop(j):
        return next((s for s in stops if s > j), T)

    def put(a, b, ph, k_, t_, sx, sy):
        a, b = max(0, a), min(T, b)
        if b <= a:
            return
        phase[a:b] = ph
        kind[a:b] = k_
        pteam[a:b] = -1 if t_ is None else t_
        pticks[a:b] = np.clip(np.arange(b - a), 0, 32767)
        spot[a:b] = (round(sx * 10), round(sy * 10))

    for n, (j, typ, t, k_, x, y) in enumerate(dec):
        if typ in ("FOUL", "OFFSIDE"):
            j2 = next((d[0] for d in dec[n + 1:] if d[1] == "PLACE"), next_stop(j))
            put(j, j2, FREEZE, k_, t, x, y)
            continue
        end = next((ki for ki in kicks_by_team.get(t, []) if ki > j), T) if t is not None else next_stop(j)
        end = min(end, next_stop(j))
        put(j, end, PIECE, k_, t, x, y)
        if k_ in (HC.LATERAL, HC.CORNER, HC.GOAL_KICK) and (n == 0 or dec[n - 1][1] == "PLACE"):
            d = j - 1
            while d > j - 40 and d > 0 and (abs(bp[d, 1]) > HC.LATERAL_OUT_Y
                                             or abs(bp[d, 0]) > HC.LINE_W + HC.OUT_MARGIN and abs(bp[d, 1]) > HC.GOAL_HALF_H):
                d -= 1
            if d < j - 1:
                put(d + 1, j, OUT_PENDING, k_, t, x, y)
    for e in events:
        if e["name"] == "goal":
            g = m.at(e["frame"])
            if g is not None:
                end = next((s for s in range(g + 1, T) if st[s] == 0), min(T, g + HC.GOAL_RESET_TICKS))
                put(g, end, GOAL, 0, None, 0.0, 0.0)
    ko = st == 0
    phase[ko] = KICKOFF
    out.update(phase=phase, piece_kind=kind, piece_team=pteam, phase_ticks=pticks, spot=spot,
               offside_on=offside_on(m).astype(np.bool_))
    # resumen: patadas con la masa de la pelota antes, anuncios con números, decisiones
    inv = z["ball_invMass"]
    kicks = [(int(i), int(np.nonzero(sl[i] == c)[0][0]), round(float(inv[max(0, i - 1)]), 4))
             for i, c in m.kicks if (sl[i] == c).any()]
    nums = []
    for i, msg in m.ann:
        g = PS_ANN.search(msg)
        if g:
            nums.append((int(i), "powershot", int(g.group(1))))
        g = POWER_ANN.search(msg)
        if g:
            nums.append((int(i), g.group(1), int(g.group(2))))
    meta = dict(name=name, stadium=(header or {}).get("stadium"), T=int(T),
                stadiums=[e["stadium"] for e in events if e["name"] == "stadium_change"],
                full_11v11=float(np.mean(valid.sum(axis=1) == N_SLOTS)),
                gk_declared=[float(np.mean(gk[:, 0] >= 0)), float(np.mean(gk[:, 1] >= 0))],
                gk_known=[float(np.mean(out["gk_src"][:, t] > 0)) for t in range(2)],
                gk_inferred_agree=[int(((gk[:, t] >= 0) & (inf[:, t] == gk[:, t])).sum()) for t in range(2)],
                gk_declared_ticks=[int((gk[:, t] >= 0).sum()) for t in range(2)],
                phases={int(k): int(v) for k, v in zip(*np.unique(phase, return_counts=True))},
                decisions=[(int(d[0]), d[1], d[2], int(d[3]), round(d[4], 1), round(d[5], 1)) for d in dec],
                kicks=kicks, numbers=nums)
    return out, meta


def save(out, meta, base: Path):
    base.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(base) + ".npz", **out)
    Path(str(base) + ".json").write_text(json.dumps(meta), encoding="utf-8")


def load_d0(base):
    base = str(base).removesuffix(".npz")
    return dict(np.load(base + ".npz")), json.loads(Path(base + ".json").read_text(encoding="utf-8"))


def one(src, out_base, max_players=48):
    lines = (io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8", errors="replace") if src == "-"
             else open(src, encoding="utf-8", errors="replace"))
    # 48 columnas alcanzan casi siempre (máximo de 33 jugadores distintos en los 31 partidos de S2) y cuestan la mitad
    # de RAM que 96; el lote reintenta con 128 si un partido tiene más
    arrays = convert(lines, None, max_players=max_players, return_arrays=True)
    name = Path(out_base).name
    out, meta = build(arrays, name)
    save(out, meta, Path(out_base))
    return meta


def batch(stadium, jobs, limit=None):
    """Todos los partidos únicos (S0) cuyo estadio principal es `stadium`, `jobs` a la vez. Saltea los ya hechos."""
    splits = json.loads((ROOT / "reports" / "hx11" / "splits.json").read_text(encoding="utf-8"))["recordings"]
    names = sorted(n for n, r in splits.items() if r["stadium"] == stadium)[:limit]
    src = ROOT / "replays_real" / "stadiums" / "haxelevenx11"
    py = sys.executable
    t0 = time.time()
    done = []

    def run(n):
        base = OUT / n.removesuffix(".hbr2")
        if Path(str(base) + ".npz").exists():
            return n, "ya estaba"
        for mp in (48, 128):
            cmd = (f'node "{ROOT / "bridge" / "hx11_ticks.js"}" "{src / n}" --max-minutes 0 --stdout | '
                   f'"{py}" -m tools.hx11_d0 uno - --out "{base}" --max-players {mp}')
            r = subprocess.run(cmd, shell=True, cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
                               errors="replace")
            if r.returncode == 0 or "subir max_players" not in r.stderr:
                break
        return n, "ok" if r.returncode == 0 else ("error: " + r.stderr[-300:])

    with ThreadPoolExecutor(jobs) as ex:
        for k, (n, status) in enumerate(ex.map(run, names), 1):
            done.append((n, status))
            if k % 10 == 0 or not status.startswith(("ok", "ya")):
                print(f"{k}/{len(names)} {n}: {status} ({time.time() - t0:.0f} s)", flush=True)
    # manifiesto con la partición de S0
    rows = {}
    for n, status in done:
        base = OUT / n.removesuffix(".hbr2")
        if not Path(str(base) + ".json").exists():
            continue
        meta = json.loads(Path(str(base) + ".json").read_text(encoding="utf-8"))
        rows[n] = dict(split=splits[n]["split"], session=splits[n]["session"], groups=splits[n]["groups"], T=meta["T"],
                       full_11v11=round(meta["full_11v11"], 3), gk_declared=meta["gk_declared"], gk_known=meta["gk_known"],
                       gk_inferred_agree=meta["gk_inferred_agree"], gk_declared_ticks=meta["gk_declared_ticks"],
                       phases=meta["phases"], decisions=len(meta["decisions"]), kicks=len(meta["kicks"]))
    (OUT / "manifest.json").write_text(json.dumps(dict(stadium=stadium, recordings=rows), indent=0), encoding="utf-8")
    roster()
    print(f"{len(rows)} partidos en {OUT} ({time.time() - t0:.0f} s)")


def roster():
    """Agrega al manifiesto el plantel en juego abierto: mediana del equipo más chico (`min_lado_p50`) y `apto`.

    Algunas grabaciones del estadio actual no son partidos 11v11 (2026-10-08: 5 de 399, con 1, 2, 6, 6 y 7 jugadores
    en el equipo más chico; una de 1 contra 1 hizo fallar D2). D2, D3, marcas y la imitación usan solo los aptos."""
    path = OUT / "manifest.json"
    man = json.loads(path.read_text(encoding="utf-8"))
    for n, r in man["recordings"].items():
        z = np.load(OUT / f"{n.removesuffix('.hbr2')}.npz")
        on = z["slot_pid"][z["phase"] == OPEN] >= 0
        m = np.minimum(on[:, :N_TEAM].sum(1), on[:, N_TEAM:].sum(1))
        r["min_lado_p50"] = float(np.median(m)) if len(m) else 0.0
        r["apto"] = r["min_lado_p50"] >= MIN_SIDE
    path.write_text(json.dumps(man, indent=0), encoding="utf-8")
    return man


def apt_names(man=None):
    """Nombres (sin .hbr2) de los partidos aptos del manifiesto, ordenados."""
    man = man or json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    return sorted(n.removesuffix(".hbr2") for n, r in man["recordings"].items() if r.get("apto", True))
    return rows


def summary():
    """Agregados del manifiesto (sin nombres): horas, cobertura 11v11, arquero declarado, fases, decisiones."""
    man = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    rows = man["recordings"]
    by = collections.defaultdict(lambda: collections.Counter())
    ph = collections.Counter()
    T = 0
    for r in rows.values():
        b = by[r["split"]]
        b["partidos"] += 1
        b["ticks"] += r["T"]
        b["ticks_11v11"] += r["T"] * r["full_11v11"]
        b["decisiones"] += r["decisions"]
        b["patadas"] += r["kicks"]
        T += r["T"]
        for k, v in r["phases"].items():
            ph[int(k)] += v
    gk = np.array([r["gk_declared"] for r in rows.values()])
    full = np.array([r["full_11v11"] for r in rows.values()])
    names = {OPEN: "juego abierto", OUT_PENDING: "salida pendiente", PIECE: "pelota parada", FREEZE: "congelamiento",
             GOAL: "gol", KICKOFF: "saque inicial"}
    s = dict(estadio=man["stadium"], partidos=len(rows), horas=round(T / 216000, 1),
             por_parte={k: dict(partidos=v["partidos"], horas=round(v["ticks"] / 216000, 1),
                                horas_11v11=round(v["ticks_11v11"] / 216000, 1), decisiones=v["decisiones"],
                                patadas=v["patadas"]) for k, v in sorted(by.items())},
             fases={names[k]: round(v / T, 4) for k, v in sorted(ph.items())},
             once_contra_once=[round(float(np.percentile(full, q)), 3) for q in (10, 50, 90)],
             arquero_declarado=[round(float(np.percentile(gk.min(1), q)), 3) for q in (10, 50, 90)],
             partidos_sin_gk_declarado=int((gk.max(1) < 0.01).sum()),
             arquero_conocido=[round(float(np.percentile(np.array([r["gk_known"] for r in rows.values()]).min(1), q)),
                                     3) for q in (10, 50, 90)],
             inferido_vs_declarado=round(sum(sum(r["gk_inferred_agree"]) for r in rows.values())
                                         / max(1, sum(sum(r["gk_declared_ticks"]) for r in rows.values())), 4))
    (ROOT / "reports" / "hx11" / "d0_cache.json").write_text(json.dumps(s, indent=1, ensure_ascii=False),
                                                              encoding="utf-8")
    lines = ["# D0: caché por tick de HaxEleven", "",
             f"Etapa D0 de [`docs/PLAN_HAXELEVEN.md`](../../docs/PLAN_HAXELEVEN.md), con "
             f"[`tools/hx11_d0.py`](../../tools/hx11_d0.py). Estadio \"{s['estadio']}\": {s['partidos']} partidos, "
             f"{s['horas']} h. Un archivo por partido en `data/hx11/d0/` (no versionado, sin nombres en el resumen).",
             "",
             "- **Precisión de la cuantización:** posiciones a 0,05 px y velocidades a 0,0005 px/tick.",
             f"- **11 contra 11 completo,** parte de los ticks por partido: p10 {s['once_contra_once'][0]}, "
             f"p50 {s['once_contra_once'][1]}, p90 {s['once_contra_once'][2]}.",
             f"- **Arquero declarado con `!gk`** (el peor de los dos equipos), parte de los ticks: p10 "
             f"{s['arquero_declarado'][0]}, p50 {s['arquero_declarado'][1]}, p90 {s['arquero_declarado'][2]}. "
             f"{s['partidos_sin_gk_declarado']} partidos no tienen ningún `!gk` (versiones del script sin ese comando).",
             f"- **Arquero inferido** donde no hay declarado: el más cercano a su arco, el más frecuente en bloques de "
             f"1 min. Coincide con el declarado en el {100 * s['inferido_vs_declarado']:.1f}% de los ticks que tienen "
             f"declarado. Arquero conocido (declarado o inferido), el peor equipo: p10 {s['arquero_conocido'][0]}, "
             f"p50 {s['arquero_conocido'][1]}.",
             "", "| Parte | Partidos | Horas | Horas 11v11 | Decisiones de la sala | Patadas |", "|---|---|---|---|---|---|"]
    lines += [f"| {k} | {v['partidos']} | {v['horas']} | {v['horas_11v11']} | {v['decisiones']} | {v['patadas']} |"
              for k, v in s["por_parte"].items()]
    lines += ["", "**Fases (parte de los ticks):** " + ", ".join(f"{k} {100 * v:.1f}%" for k, v in s["fases"].items())]
    (ROOT / "reports" / "hx11" / "d0_cache.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(s, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    u = sub.add_parser("uno")
    u.add_argument("src")
    u.add_argument("--out", required=True)
    u.add_argument("--max-players", type=int, default=48)
    b = sub.add_parser("lote")
    b.add_argument("--stadium", default="HaxEleven RSx11 [1800, 1200]")
    b.add_argument("--jobs", type=int, default=4)
    b.add_argument("--limit", type=int, default=None)
    sub.add_parser("resumen")
    sub.add_parser("plantel")
    a = ap.parse_args()
    if a.cmd == "uno":
        meta = one(a.src, a.out, a.max_players)
        print(f"{meta['name']}: {meta['T']} ticks, 11v11 {meta['full_11v11']:.2f}, arquero {meta['gk_declared']}, "
              f"fases {meta['phases']}")
    elif a.cmd == "lote":
        batch(a.stadium, a.jobs, a.limit)
    elif a.cmd == "plantel":
        man = roster()["recordings"]
        bad = {n: r["min_lado_p50"] for n, r in man.items() if not r["apto"]}
        print(f"{len(man) - len(bad)} aptos de {len(man)}; no aptos (mediana del equipo más chico): "
              f"{sorted(bad.values())}")
    else:
        summary()


if __name__ == "__main__":
    main()
