"""Demostraciones humanas corregidas para el candidato C (PLAN_RS4.md 2.3 y 3).

Correcciones respecto de los datos de BC v2/v5 (data/bc_rs4_v2):
* particiones por sesión de reports/rs4_b1/splits.json (la prueba nunca se lee aquí);
* etiqueta causal única: la tecla del mismo tick de la decisión (desfase 0, medido en
  reports/rs4_b1/alignment_audit.json: explica el 91% de los cambios de tecla; +6, el 25%);
* fases separadas por decisión y jugador (reports/rs4_b1/rules_audit.json):
    0 juego activo · 1 espera previa al saque inicial · 2 último segundo antes del saque inicial ·
    3 saque en curso, equipo que saca · 4 saque en curso, equipo rival (restricciones del árbitro);
* secuencias continuas por jugador que no cruzan cambios de plantel, marcador, mapa ni huecos.

Observación: la universal del entrenamiento (tools/build_bc_dataset.Loader). El bloque de reglas
queda enmascarado por el modelo y la proyección pública nace en cero, así que no depende del árbitro.

Salida: data/rs4_b1_demos/<partición>/<grabación>.npz con obs (n, 8, 127) float16, act (n, 8),
seg (n,), phase (n, 8), ball_dist (n, 8), frame (n,), y por grabación los toques humanos para
medir pases/pérdidas con la misma herramienta que la compuerta de C (touch_metrics).

  python -m tools.rs4_b1_demos --workers 8
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FINAL_KICKOFF_TICKS = 60
PHASES = ("activo", "espera_saque_inicial", "ultimo_segundo_saque_inicial", "saque_propio", "saque_rival")
PASS_MIN_DISTANCE = 0.04 * 1150.0  # como team_pass_min_dist_frac del entorno


def touch_metrics(touch, team, ball):
    """Pases y pérdidas desde la secuencia de toques, igual para humanos y simulación.

    touch (T, 8) bool: contacto o patada en cada decisión; team (8,); ball (T, 2). Un toque de un
    único jugador define al dueño; pase = dueño nuevo del mismo equipo con la pelota desplazada
    >= PASS_MIN_DISTANCE; pérdida = dueño nuevo del otro equipo. Devuelve (pases, pérdidas) por equipo.
    """
    passes, turnovers = np.zeros(2, dtype=np.int64), np.zeros(2, dtype=np.int64)
    owner, origin = -1, None
    for t in np.flatnonzero(touch.sum(axis=1) == 1):
        player = int(np.flatnonzero(touch[t])[0])
        if owner >= 0 and player != owner:
            if team[player] == team[owner]:
                if np.hypot(*(ball[t] - origin)) >= PASS_MIN_DISTANCE:
                    passes[team[owner]] += 1
            else:
                turnovers[team[owner]] += 1
        if player != owner:
            origin = ball[t].copy()
        owner = player
    return passes, turnovers


def _intervals(audit):
    kickoffs = [(k["frame"], k["frame"] + k["duration"], k["team"] - 1) for k in audit["kickoffs"]]
    restarts = [(r["frame"], r["frame"] + r["duration"], r["team"] - 1) for r in audit["restarts"]
                if r.get("team") in (1, 2) and r.get("duration") is not None]
    return kickoffs, restarts


def _phases(frames, teams, kickoffs, restarts):
    """phase (n, 8) para frames (n,) y equipos (n, 8)."""
    phase = np.zeros(teams.shape, dtype=np.uint8)
    for start, end, _ in kickoffs:
        inside = (frames >= start) & (frames < end)
        phase[inside] = np.where((end - frames[inside]) <= FINAL_KICKOFF_TICKS, 2, 1)[:, None]
    for start, end, owner in restarts:
        inside = (frames >= start) & (frames < end)
        phase[inside] = np.where(teams[inside] == owner, 3, 4)
    return phase


def iter_items(path):
    from tools.rs4_jsonl_cache import open_ticks
    with open_ticks(path) as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def process(path, audit, out):
    from tools.build_bc_dataset import Loader, catalog_by_name, replay_stadium
    kickoffs, restarts = _intervals(audit)
    kicks = set()  # (jugador, frame) de los eventos "kick" del motor original
    catalog = catalog_by_name()
    loader, pending, rows = None, [], []
    seg, last_frame, roster, score = 0, None, None, None
    obs_parts, act_parts, meta = [], [], []

    def flush():
        nonlocal pending
        if pending and loader is not None:
            for start in range(0, len(pending), 512):
                chunk = pending[start:start + 512]
                obs, act = loader.obs([(tick, players, since) for tick, players, since, _ in chunk])
                obs_parts.append(obs.reshape(len(chunk), 8, -1).astype(np.float16))
                act_parts.append(act.reshape(len(chunk), 8))
                meta.extend(info for *_, info in chunk)
        pending = []

    def boundary():
        nonlocal seg, last_frame, roster, score
        flush()
        seg += 1
        last_frame, roster, score = None, None, None

    since = 0
    for item in iter_items(path):
        if item.get("type") == "event" and item.get("name") == "kick":
            kicks.add((item.get("playerId"), int(item["frame"])))
            continue
        if item.get("type") == "header" or item.get("name") == "stadium_change":
            boundary()
            entry = catalog.get((item.get("stadium") or "").strip())
            loader = None
            if entry and entry["stadium"] == "rs_one" and item.get("stadiumFile"):
                stadium, _ = replay_stadium(Path(path).parent / item["stadiumFile"], entry)
                loader = Loader(stadium, 4, False, True, batch=512)
            continue
        if item.get("type") != "tick":
            if item.get("name") in ("game_stop", "game_start", "player_join", "player_leave", "team_change"):
                boundary()
            continue
        if loader is None or item.get("state") not in (0, 1):
            boundary()
            continue
        players = sorted(item["players"], key=lambda p: (p["team"], p["id"]))
        valid = len(players) == 8 and sum(p["team"] == 1 for p in players) == 4
        valid &= all(0 <= p["disc"] < len(item["discs"]) for p in players)
        if valid:
            discs = [item["discs"][0]] + [item["discs"][p["disc"]] for p in players]
            valid = all(all(d.get(k) is not None and np.isfinite(d[k]) for k in ("x", "y", "vx", "vy")) for d in discs)
        if not valid:
            boundary()
            continue
        signature = tuple((p["id"], p["team"]) for p in players)
        current = tuple(item.get("score") or (0, 0))
        frame = int(item["frame"])
        if (roster is not None and roster != signature) or (score is not None and score != current):
            boundary()
        since = 0 if item["state"] == 0 else since + 1
        if frame % 3:
            roster, score = signature, current
            continue
        if last_frame is not None and frame != last_frame + 3:
            boundary()
        roster, score, last_frame = signature, current, frame
        ball = item["discs"][0]
        position = np.array([[item["discs"][p["disc"]]["x"], item["discs"][p["disc"]]["y"]] for p in players])
        radius = np.array([item["discs"][p["disc"]].get("r", 15.0) for p in players])
        gap = np.hypot(position[:, 0] - ball["x"], position[:, 1] - ball["y"])
        info = dict(seg=seg, frame=frame, ball=(ball["x"], ball["y"]), dist=gap - radius - ball.get("r", 8.325),
                    ids=[p["id"] for p in players], team=np.array([p["team"] - 1 for p in players]))
        pending.append((item, players, since, info))
    boundary()
    if not meta:
        return dict(decisions=0, note="sin tramos 4v4 válidos")
    obs, act = np.concatenate(obs_parts), np.concatenate(act_parts)
    frames = np.array([m["frame"] for m in meta], dtype=np.int64)
    teams = np.stack([m["team"] for m in meta])
    phase = _phases(frames, teams, kickoffs, restarts)
    segs = np.array([m["seg"] for m in meta], dtype=np.int32)
    ball = np.array([m["ball"] for m in meta])
    # Toque como en el simulador (árbitro rs_one_v1): contacto exacto o patada en la ventana de la decisión.
    touch = np.stack([(m["dist"] <= 0.01) | np.array([any((pid, f) in kicks for f in range(m["frame"] - 2, m["frame"] + 1))
                                                       for pid in m["ids"]]) for m in meta])
    passes, turnovers = np.zeros(2, dtype=np.int64), np.zeros(2, dtype=np.int64)
    active = phase.max(axis=1) == 0
    for s in np.unique(segs):
        rows = (segs == s) & active
        if rows.sum() > 1:
            p, t = touch_metrics(touch[rows], teams[rows][0], ball[rows])
            passes += p
            turnovers += t
    np.savez_compressed(out, obs=obs, act=act, seg=segs, phase=phase, frame=frames, touch=touch,
                        ball_dist=np.stack([m["dist"] for m in meta]).astype(np.float16), team=teams.astype(np.int8))
    return dict(decisions=int(len(obs)), segments=int(len(np.unique(segs))),
                phases={name: int((phase == i).sum()) for i, name in enumerate(PHASES)},
                active_minutes=float(active.sum() * 3 / 3600), passes=passes.tolist(), turnovers=turnovers.tolist())


def _job(job):
    name, path, audit, out = job
    try:
        return name, process(path, audit, out)
    except Exception as error:
        print(f"error {name}: {error!r}", flush=True)
        return name, None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default=str(ROOT / "data" / "rs4_jsonl"))
    ap.add_argument("--splits", default=str(ROOT / "reports" / "rs4_b1" / "splits.json"))
    ap.add_argument("--audit", default=str(ROOT / "reports" / "rs4_b1" / "rules_audit.json"))
    ap.add_argument("--out", default=str(ROOT / "data" / "rs4_b1_demos"))
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    splits = json.loads(Path(a.splits).read_text(encoding="utf-8"))
    audits = {r["sha256"]: r for r in json.loads(Path(a.audit).read_text(encoding="utf-8"))["recordings"]}
    jobs = []
    for name, row in splits["recordings"].items():
        if not row["only_rs_one"] or row["split"] not in ("entrenamiento", "desarrollo") or row["sha256"] not in audits:
            continue
        target = Path(a.out) / row["split"] / (Path(name).stem + ".npz")
        target.parent.mkdir(parents=True, exist_ok=True)
        jobs.append((name, str(Path(a.cache) / row["jsonl"].replace("\\", "/")), audits[row["sha256"]], str(target)))
    started = time.time()
    if a.workers > 1:
        from multiprocessing import get_context
        with get_context("spawn").Pool(a.workers) as pool:
            results = pool.map(_job, jobs)
    else:
        results = [_job(job) for job in jobs]
    manifest = dict(version="RS4-b1-demos-1", label_lag=0, decision_ticks=3, phases=PHASES,
                    final_kickoff_ticks=FINAL_KICKOFF_TICKS, seconds=round(time.time() - started, 1),
                    recordings={name: dict(split=splits["recordings"][name]["split"], **(r or {"error": True}))
                                for name, r in results})
    for split in ("entrenamiento", "desarrollo"):
        rows = [r for name, r in results if r and r["decisions"] and splits["recordings"][name]["split"] == split]
        passes = sum(sum(r["passes"]) for r in rows)
        turnovers = sum(sum(r["turnovers"]) for r in rows)
        manifest[split] = dict(recordings=len(rows), decisions=sum(r["decisions"] for r in rows),
                               active_minutes=sum(r["active_minutes"] for r in rows),
                               passes=passes, turnovers=turnovers, pass_turnover_ratio=passes / max(turnovers, 1))
    (Path(a.out) / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    for split in ("entrenamiento", "desarrollo"):
        print(split, json.dumps(manifest[split], ensure_ascii=False))


if __name__ == "__main__":
    main()
