"""Dataset de imitación humana RS4-Z: observación v2 exacta del actor y acción humana, desde las grabaciones.

Una fila por jugador y decisión (cada 3 ticks) en tramos 4v4 estables: juego abierto, saques
(lateral, córner, saque de arco) y espera del saque inicial. La observación sale de
`env/rs4z/obs_v2.observe` sobre un `RS4ZEnv` cargado con el estado grabado; así el imitador ve
exactamente lo mismo que la política entrenada y puede jugar en el simulador o en la liga.

Sin fuga de la etiqueta: el estado de patada (`kick_cancel`, "pateando") y la acción aplicada se
toman del tick anterior a la decisión. El historial de acciones propias y la latencia se guardan,
pero el entrenamiento BC los anula (el imitador C de RS4-b1 copiaba la tecla anterior).

Etiqueta (marco propio): dirección del input en el cuadro de decisión; patada si la tecla está
apretada en ese cuadro o si el jugador pateó dentro de la ventana de 3 ticks.

  python -m tools.rs4z_bc_dataset --out data/rs4z_bc --workers 10
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
KIND_CODE = {"lateral": 1, "corner": 2, "goal_kick": 3}
STRIDE = 3


def _action(inp, dir2move):
    dx = ((inp >> 3) & 1) - ((inp >> 2) & 1)
    dy = ((inp >> 1) & 1) - (inp & 1)
    return dir2move[(dx, dy)] + (9 if inp & 16 else 0)


def _restart_map(ticks, events):
    """frame → (tipo 1..3, equipo 0/1, edad en ticks) de los saques con equipo conocido, y frames dudosos."""
    from tools.rs4_rules_audit import rules
    from tools.rs4z_conformance import restart_frames
    restarts, _ = rules(ticks, events)
    known = {}
    for r in restarts:
        if r["team"] is None or r["kind"] not in KIND_CODE:
            continue
        for f in range(r["frame"], r["frame"] + r["duration"]):
            known[f] = (KIND_CODE[r["kind"]], int(r["team"]) - 1, f - r["frame"])
    compact = [dict(frame=t["frame"]) for t in ticks]
    doubtful = restart_frames(compact, events) - set(known)
    return known, doubtful


def _kick_strength(rec_dir):
    for hbs in sorted(Path(rec_dir).glob("*.hbs")):
        try:
            data = json.loads(hbs.read_text(encoding="utf-8"))
            ks = (data.get("playerPhysics") or {}).get("kickStrength")
            if ks is not None:
                return float(ks)
        except Exception:
            pass
    return None


def extract(path, rec_index, batch=512):
    """Filas (obs, acción, meta) de una grabación."""
    from bridge.compare_sim import _DIR2MOVE
    from env.rs4z import kernel as K
    from env.rs4z.core import MIRROR_ACTION, RS4ZEnv
    from env.rs4z.obs_v2 import OBS_DIM, observe
    from tools.rs4_rules_audit import load
    from tools.rs4z_conformance import mass_phases

    header, ticks, events = load(path)
    by = {t["frame"]: t for t in ticks}
    known, doubtful = _restart_map(ticks, events)
    phase = mass_phases([dict(frame=t["frame"]) for t in ticks], events)
    kicks = {}
    for f, evs in events.items():
        for e in evs:
            if e["name"] == "kick":
                kicks.setdefault(f, set()).add(e.get("playerId"))
    ks = _kick_strength(Path(path).parent)
    # saque inicial: inicio de cada tramo con state 0
    ko_start = {}
    start = None
    for t in ticks:
        if t["state"] == 0:
            if start is None:
                start = t["frame"]
            ko_start[t["frame"]] = start
        else:
            start = None

    env = RS4ZEnv(batch, contract="v2_lateral", frame_skip=STRIDE, deadline=0, kickoff_deadline=0, max_delay=0)
    if ks is not None:
        env.rf[:, K.RF_KSTR] = ks
    obs_buf = np.empty((batch, 8, OBS_DIM), dtype=np.float32)
    out_obs, out_act, out_meta = [], [], []
    pending = []

    def roster(t):
        return tuple(sorted((p["id"], p["team"]) for p in t["players"]))

    def flush():
        if not pending:
            return
        observe(env, obs_buf)
        for i, (labels, meta) in enumerate(pending):
            out_obs.append(obs_buf[i].astype(np.float16))
            out_act.append(labels)
            out_meta.append(meta)
        pending.clear()

    for t in ticks:
        f = t["frame"]
        if f % STRIDE or t["state"] not in (0, 1) or len(t["players"]) != 8:
            continue
        if sum(p["team"] == 1 for p in t["players"]) != 4 or f in doubtful:
            continue
        prev, nxt1, nxt2 = by.get(f - 1), by.get(f + 1), by.get(f + 2)
        if prev is None or nxt1 is None or nxt2 is None:
            continue
        r0 = roster(t)
        if roster(prev) != r0 or roster(nxt1) != r0 or roster(nxt2) != r0:
            continue
        m = phase.get(f)
        order = sorted(t["players"], key=lambda p: (p["team"], p["id"]))
        prev_by = {p["id"]: p for p in prev["players"]}
        discs = t["discs"]
        pp = np.array([[discs[p["disc"]]["x"], discs[p["disc"]]["y"]] for p in order])
        pv = np.array([[discs[p["disc"]]["vx"], discs[p["disc"]]["vy"]] for p in order])
        ball = discs[0]
        # estado de patada y acción aplicada, del tick anterior (sin mirar el input de la decisión)
        held = np.array([bool(prev_by[p["id"]]["input"] & 16) and not prev_by[p["id"]]["kicking"] for p in order])
        applied = np.array([_action(prev_by[p["id"]]["input"], _DIR2MOVE) for p in order], dtype=np.int64)
        n = len(pending)
        env.radius[n, 0] = ball["r"]
        env.place(n, ball_pos=(ball["x"], ball["y"]), ball_vel=(ball["vx"], ball["vy"]), player_pos=pp,
                  player_vel=pv, kick_held=held, mass_phase=0 if m == 0.5 else 1)
        env._s_act[n] = applied
        env.delay[n] = 0
        hist = np.zeros((8, env.H), dtype=np.int64)
        for h in range(min(3, env.H)):
            old = by.get(f - STRIDE * (h + 1))
            if old is not None and roster(old) == r0:
                old_by = {p["id"]: p for p in old["players"]}
                hist[:, h] = [_action(old_by[p["id"]]["input"], _DIR2MOVE) for p in order]
        env.act_hist[n] = hist
        kind, team, age = known.get(f, (0, -1, 0))
        if t["state"] == 0:
            env.ri[n, K.RI_KO] = 1
            env.ri[n, K.RI_KO_TEAM] = int(t.get("ko") or 1) - 1
            env.ri[n, K.RI_KO_TICKS] = f - ko_start.get(f, f)
            kind, team, age = 0, -1, 0
        if kind:
            env.ri[n, K.RI_TEAM] = team
            env.ri[n, K.RI_KIND] = kind
            env.ri[n, K.RI_TICKS] = age
        window = set()
        for g in (f, f + 1, f + 2):
            window |= kicks.get(g, set())
        labels = np.zeros(8, dtype=np.int8)
        for slot, p in enumerate(order):
            a = _action(p["input"], _DIR2MOVE)
            if a < 9 and p["id"] in window:
                a += 9
            if slot >= 4:
                a = int(MIRROR_ACTION[a])
            labels[slot] = a
        d = np.hypot(pp[:, 0] - ball["x"], pp[:, 1] - ball["y"])
        state_code = 4 if t["state"] == 0 else kind
        meta = np.stack([np.full(8, rec_index), np.full(8, f), np.arange(8), np.full(8, state_code),
                         np.round(d).astype(np.int64)], axis=1).astype(np.int32)
        pending.append((labels, meta))
        if len(pending) == batch:
            flush()
    flush()
    if not out_obs:
        return None
    obs = np.concatenate(out_obs).reshape(-1, OBS_DIM)
    act = np.concatenate(out_act)
    meta = np.concatenate(out_meta)
    return obs, act, meta


def _job(args):
    path, rec_index, out = args
    try:
        res = extract(path, rec_index)
    except Exception as error:  # una grabación rota no debe tumbar el resto
        import traceback
        traceback.print_exc()
        return rec_index, None, str(error)
    if res is None:
        return rec_index, None, "sin filas"
    obs, act, meta = res
    np.savez(out, obs=obs, act=act, meta=meta)
    return rec_index, len(act), None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default=str(ROOT / "data" / "rs4_jsonl"))
    ap.add_argument("--splits", default=str(ROOT / "reports" / "rs4_b1" / "splits.json"))
    ap.add_argument("--out", default=str(ROOT / "data" / "rs4z_bc"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    splits = json.loads(Path(a.splits).read_text(encoding="utf-8"))
    out = Path(a.out)
    (out / "parts").mkdir(parents=True, exist_ok=True)
    rows = [(name, row) for name, row in sorted(splits["recordings"].items()) if row["only_rs_one"]]
    if a.limit:
        rows = rows[:a.limit]
    jobs = [(str(Path(a.cache) / row["jsonl"].replace("\\", "/")), i, str(out / "parts" / f"{i:03d}.npz"))
            for i, (name, row) in enumerate(rows)]
    results = {}
    if a.workers > 1:
        from multiprocessing import get_context
        with get_context("spawn").Pool(a.workers) as pool:
            for idx, count, err in pool.imap_unordered(_job, jobs):
                results[idx] = (count, err)
                print(f"{idx:3d} {rows[idx][0][:50]:50s} filas={count} {err or ''}", flush=True)
    else:
        for job in jobs:
            idx, count, err = _job(job)
            results[idx] = (count, err)
            print(f"{idx:3d} {rows[idx][0][:50]:50s} filas={count} {err or ''}", flush=True)
    manifest = dict(version="rs4z-bc-1", stride=STRIDE, obs_version="rs4z-obs-v2",
                    recordings=[dict(index=i, name=name, split=row["split"], rows=results.get(i, (None,))[0],
                                     error=results.get(i, (None, None))[1]) for i, (name, row) in enumerate(rows)],
                    label="dirección del input en el cuadro de decisión; patada = tecla en el cuadro o patada en la ventana",
                    meta_columns=["recording", "frame", "slot", "state(0 abierto,1 lateral,2 córner,3 arco,4 inicial)",
                                  "dist_pelota"])
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    by_split = {}
    for r in manifest["recordings"]:
        by_split.setdefault(r["split"], 0)
        by_split[r["split"]] += r["rows"] or 0
    print(json.dumps(by_split))


if __name__ == "__main__":
    main()
