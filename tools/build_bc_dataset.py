"""Dataset de imitación (behavior cloning) a partir de los replays de liga.

python -m tools.build_bc_dataset [--stride 12] [--folders bigx3,futsalx3,...] [--limit N]

Para cada replay de replays_real/stadiums/<carpeta>/ cuyo mapa está en el catálogo (train/tasks.yaml):
  1. lo reproduce entero con node-haxball (bridge/replay_to_jsonl.js, motor original del juego);
  2. cada `stride` ticks (en saque y en juego, con equipos del mismo tamaño) carga el estado en un HaxballEnv
     con la obs "universal" —el MISMO código que usa el entrenamiento, así imitar y jugar ven lo mismo—;
  3. guarda la obs de cada jugador y la tecla que apretó de verdad (en su marco propio: espejada si es azul).
Se descartan prácticas ("Training", penales) y mapas que no son del catálogo. Peso por carpeta en WEIGHTS
(rsx6 es de sala pública: nivel más bajo, pesa la mitad).

Salida: data/bc/<carpeta>/<replay>.npz con obs (float16, n x D), act (uint8), weight, stadium, T, rules.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tools.replay_files import duplicate_recordings  # noqa: E402
from env.haxball_env import MIRROR_ACTION, HaxballEnv  # noqa: E402
from sim.physics import KICK_REACH, MOVE_DIRS  # noqa: E402
from sim.stadium import BLUEKO, PLAYER_MASK, REDKO, STADIUM_DIR, load_stadium  # noqa: E402

SRC = ROOT / "replays_real" / "stadiums"
OUT = ROOT / "data" / "bc"
SKIP = ("training", "penales")
WEIGHTS = {"rsx6": 0.5}
_DIR2MOVE = {(int(x), int(y)): i for i, (x, y) in enumerate(MOVE_DIRS)}


def input_to_action(inp: int) -> int:
    """input de HaxBall (1 arriba, 2 abajo, 4 izq, 8 der, 16 patear) -> acción 0..17 del sim (mundo)."""
    dx = ((inp >> 3) & 1) - ((inp >> 2) & 1)
    dy = ((inp >> 1) & 1) - (inp & 1)
    return _DIR2MOVE[(dx, dy)] + (9 if inp & 16 else 0)


def catalog_by_name() -> dict[str, dict]:
    """nombre del mapa (como viene en el replay) -> estadio del catálogo y sus reglas (de train/tasks.yaml).
    Sólo los mapas de tareas que están en el currículo (train/config_multi.yaml): p. ej. Classic no."""
    tasks = yaml.safe_load((ROOT / "train" / "tasks.yaml").read_text(encoding="utf-8"))
    stages = yaml.safe_load((ROOT / "train" / "config_multi.yaml").read_text(encoding="utf-8"))["stages"]
    active = {t for st in stages for t in st["tasks"]}
    rules = tasks["rules"]
    by_stadium = {}
    for name, spec in tasks["tasks"].items():
        if name in active:
            by_stadium.setdefault(spec["stadium"], rules[spec["rules"]])
    out = {}
    for key, r in by_stadium.items():
        d = json.loads((STADIUM_DIR / f"{key}.hbs").read_text(encoding="utf-8"))
        name = d.get("name", key).strip()
        if name not in out:
            out[name] = {"stadium": key, "powershot": bool(r.get("powershot")),
                         "out_of_bounds": bool(r.get("out_of_bounds")), "script": r.get("script")}
    # variantes de nombre del mismo mapa en otras salas (el estadio exacto sale de cada replay)
    for alias, key in ALIASES.items():
        target = next((v for v in out.values() if v["stadium"] == key), None)
        if target is not None:
            out.setdefault(alias, target)
    return out


ALIASES = {
    "AF Official 3v3 by Vitão ®": "af_futsalx3",
    "AF Official 3v3 by Vit�o �": "af_futsalx3",
    "Futsal x3 by Bazinga": "futsalx3",
    "Futsal x3  by Bazinga": "futsalx3",
    "Futsal X3 by Bazinga": "futsalx3",
    "Futsal x4 ; By Bazinga!": "futsal_x4",
    "Futsal X4 by Bazinga": "futsal_x4",
}


def node(*args) -> str:
    r = subprocess.run(["node", *map(str, args)], cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-2000:])
    return r.stdout


def read_ticks(path: Path):
    ticks = []
    for line in path.open(encoding="utf-8"):
        if not line.startswith('{"type":"tick"'):
            continue
        o = json.loads(line)
        ticks.append(o)
    return ticks


class Loader:
    """Carga lotes de ticks reales en un HaxballEnv universal (mismo T y estadio) y devuelve obs + acciones."""

    def __init__(self, stadium: str, T: int, powershot: bool, out_of_bounds: bool, batch: int = 512):
        self.env = HaxballEnv(batch, T, stadium, obs_layout="universal", max_entities=2 * T - 1,
                              powershot=powershot, out_of_bounds=out_of_bounds, seed=0, random_reset_prob=1.0)
        self.env.reset()
        st = self.env.sim.st
        src = st.d_src if st.d_src is not None else np.arange(st.d_pos.shape[0])
        self.stadium_idx = np.array([0] + [1 + int(i) for i in src])
        self.B, self.T = batch, T
        self.r_sum = st.player["radius"] + st.ball["radius"]

    def obs(self, rows):
        """rows: lista de (tick, jugadores ordenados rojo->azul, ticks desde el saque). Devuelve (obs, act)."""
        env, sim = self.env, self.env.sim
        n = len(rows)
        fp = sim.first_player
        for k, (t, players, since) in enumerate(rows):
            d = np.array([[q["x"], q["y"], q["vx"], q["vy"]] for q in t["discs"]], dtype=np.float64)
            idx = np.concatenate([self.stadium_idx, [p["disc"] for p in players]])
            sim.pos[k] = d[idx, :2]
            sim.vel[k] = d[idx, 2:]
            sim.kick_cancel[k] = [bool(p["input"] & 16) and not p["kicking"] for p in players]
            gap = np.linalg.norm(sim.pos[k, fp:] - sim.pos[k, 0], axis=1) - self.r_sum
            sim.touch[k] = gap < KICK_REACH
            sim.kickoff[k] = t["state"] == 0
            sim.kickoff_team[k] = 1 if t.get("ko") == 2 else 0
            sim.mask[k] = sim.base_mask
            if sim.kickoff[k]:
                sim.mask[k, fp:] = PLAYER_MASK | (REDKO if sim.kickoff_team[k] == 0 else BLUEKO)
            env.ticks[k] = min(since, env.max_ticks)
            if sim.ps_on:
                b = t.get("ball", {})
                sim.inv_env[k, 0] = b.get("im", sim.inv_env[k, 0])
                sim.ball_grav[k] = (b.get("gx", 0.0), b.get("gy", 0.0))
                sim.ps_comba[k] = b.get("gx", 0.0) != 0.0 or b.get("gy", 0.0) != 0.0
                sim.ps_charge[k] = 0
                sim.ps_held[k] = -1
        obs = env.observe()[:n]
        return obs.reshape(-1, obs.shape[-1]), self.actions(rows, "input")

    def actions(self, rows, key: str = "input") -> np.ndarray:
        """Teclas de cada jugador (en su marco propio) tomadas de p[key] (input o input_lagN)."""
        blue = self.env.sim.player_team == 1
        act = np.zeros((len(rows), self.env.P), dtype=np.uint8)
        for k, row in enumerate(rows):
            a = np.array([input_to_action(p.get(key, p["input"])) for p in row[1]])
            a[blue] = MIRROR_ACTION[a[blue]]
            act[k] = a
        return act.reshape(-1)


# Teclas guardadas además de la del instante: la de antes (para detectar cambios de decisión) y las de después
# (tiempo de reacción humano ~100-200 ms). Sin esto el modelo aprende a "seguir moviéndose como venía"
# (copycat): acierta ~50% de las teclas copiando la propia velocidad y flota por la cancha.
LAGS = (-3, 6, 12)


STADIUM_CACHE = OUT / "_stadiums"


def replay_stadium(hbs: Path, meta: dict) -> tuple[str, int]:
    """Estadio EXACTO del replay (un mismo nombre puede tener versiones con otros discos) con las medidas de
    cancha del mapa del catálogo ("haxballrl"). Se guarda en data/bc/_stadiums/<hash>.hbs.
    Devuelve (ruta, cantidad de discos del estadio contando la pelota) para verificar cada tick."""
    import hashlib
    d = json.loads(hbs.read_text(encoding="utf-8"))
    cat = json.loads((STADIUM_DIR / f"{meta['stadium']}.hbs").read_text(encoding="utf-8"))
    if "haxballrl" in cat:
        # medidas de cancha, discos del script a ignorar y puntos de saque reales (no las notas)
        d["haxballrl"] = {k: v for k, v in cat["haxballrl"].items()
                          if k.startswith("field_") or k in ("ignore_discs", "red_spawn", "blue_spawn")}
    text = json.dumps(d, sort_keys=True)
    STADIUM_CACHE.mkdir(parents=True, exist_ok=True)
    path = STADIUM_CACHE / (hashlib.md5(text.encode()).hexdigest()[:12] + ".hbs")
    if not path.exists():
        path.write_text(text, encoding="utf-8")
    ball_in_discs = d.get("ballPhysics") == "disc0" or not isinstance(d.get("ballPhysics"), dict)
    n_discs = len(d.get("discs", [])) + (0 if ball_in_discs else 1)
    return str(path), n_discs


def process_replay(file: Path, meta: dict, stride: int, tmp: Path, loaders: dict, rng, cat: dict,
                   stadium_filter=None, team_size=None) -> dict | None:
    node("bridge/replay_to_jsonl.js", file, "--out", tmp, "--max-minutes", 120)
    jsonl = max(tmp.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
    # el admin puede cambiar el mapa a mitad del replay: cada tick usa el estadio vigente (eventos stadium_change)
    changes = []  # (frame, ruta, discos, meta) ; meta None = mapa fuera del catálogo
    for line in jsonl.open(encoding="utf-8"):
        if line.startswith('{"type":"header"'):
            h = json.loads(line)
            changes.append((-1, *replay_stadium(tmp / h["stadiumFile"], meta), meta))
        elif '"stadium_change"' in line:
            e = json.loads(line)
            m = cat.get((e.get("stadium") or "").strip())
            if m is not None and e.get("stadiumFile") and not m["script"]:
                changes.append((e["frame"], *replay_stadium(tmp / e["stadiumFile"], m), m))
            else:
                changes.append((e["frame"], None, 0, None))
    ticks = read_ticks(jsonl)
    for f in tmp.iterdir():
        f.unlink()
    frames = [c[0] for c in changes]
    rows_by_key = defaultdict(list)
    since = 0
    offset = int(rng.integers(0, stride))
    import bisect
    for i, t in enumerate(ticks):
        since = 0 if t["state"] == 0 else since + 1
        if t["state"] not in (0, 1) or (i + offset) % stride:
            continue
        _, path, n_discs, m = changes[bisect.bisect_right(frames, t["frame"]) - 1]
        if m is None:
            continue
        if stadium_filter is not None and m["stadium"] != stadium_filter:
            continue
        players = sorted(t["players"], key=lambda p: (p["team"], p["id"]))
        n_red = sum(p["team"] == 1 for p in players)
        n_blue = len(players) - n_red
        if n_red == 0 or n_red != n_blue or any(p["disc"] < 0 for p in players):
            continue
        if team_size is not None and n_red != team_size:
            continue
        if len(t["discs"]) != n_discs + len(players):  # layout inesperado: se saltea el tick
            continue
        # posiciones vacías (NaN en el juego): scripts que esconden la pelota o discos (p. ej. Sanguchito X1 al
        # empezar) o jugadores al aparecer. Los discos del estadio escondidos a propósito (que el sim descarta)
        # no importan: se revisan sólo la pelota y los jugadores
        if t["discs"][0]["x"] is None or any(t["discs"][p["disc"]]["x"] is None or t["discs"][p["disc"]]["vx"] is None
                                             for p in players):
            continue
        # teclas del mismo jugador en otros instantes (si en ese tick no está, se usa la actual)
        players = [dict(p) for p in players]
        for lag in LAGS:
            j = min(max(i + lag, 0), len(ticks) - 1)
            by_id = {q["id"]: q["input"] for q in ticks[j]["players"]}
            for p in players:
                p[f"input_lag{lag}"] = by_id.get(p["id"], p["input"])
        rows_by_key[(path, n_red, m["powershot"], m["out_of_bounds"])].append((t, players, since))
    if not rows_by_key:
        return None
    obs_parts, act_parts, Ts = [], [], []
    lag_parts = {lag: [] for lag in LAGS}
    for key, rows in rows_by_key.items():
        T = key[1]
        if key not in loaders:
            loaders[key] = Loader(*key)
        ld = loaders[key]
        for s in range(0, len(rows), ld.B):
            batch = rows[s:s + ld.B]
            o, a = ld.obs(batch)
            if not np.isfinite(o).all():  # último control: nunca guardar datos inválidos (rompen el entrenamiento)
                keep = np.isfinite(o).all(axis=1)
                print(f"  aviso: {int((~keep).sum())} muestras con valores no finitos descartadas")
                o, a = o[keep], a[keep]
                lag_keep = keep
            else:
                lag_keep = None
            obs_parts.append(o.astype(np.float16))
            act_parts.append(a)
            for lag in LAGS:
                la = ld.actions(batch, f"input_lag{lag}")
                lag_parts[lag].append(la if lag_keep is None else la[lag_keep])
            Ts.append(np.full(len(a), T, dtype=np.uint8))
    return {"obs": obs_parts, "act": act_parts, "T": Ts, "n_ticks": len(ticks), "lags": lag_parts}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=12, help="un tick de cada `stride` (12 = 5 muestras/s por jugador)")
    ap.add_argument("--folders", default=None)
    ap.add_argument("--limit", type=int, default=0, help="máximo de replays por carpeta (0 = todos)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--stadium", default=None, help="filtra cada tramo, incluso cambios de mapa dentro del replay")
    ap.add_argument("--team-size", type=int, default=None)
    ap.add_argument("--out", default=None, help="dataset separado; RS4: data/bc_rs4_v2")
    a = ap.parse_args()
    if a.team_size is not None and a.team_size < 1:
        ap.error("--team-size debe ser >=1")
    out_root = Path(a.out).resolve() if a.out else OUT
    cat = catalog_by_name()
    folders = a.folders.split(",") if a.folders else sorted(p.name for p in SRC.iterdir() if p.is_dir())
    rng = np.random.default_rng(0)
    loaders: dict = {}
    tmp = Path(tempfile.mkdtemp(prefix="haxballrl_bc_"))
    total = 0
    try:
        for folder in folders:
            reps = json.loads(node("bridge/replays_by_map.js", SRC / folder))
            duplicates = duplicate_recordings(reps)
            done = 0
            for r in reps:
                if r["name"] in duplicates:
                    print(f"  salteo {folder}/{r['name']}: copia idéntica de {duplicates[r['name']]}", flush=True)
                    continue
                if "error" in r or any(s in r["stadium"].lower() for s in SKIP):
                    continue
                if any(str(n).upper().startswith("[BOT]") for n in r.get("names", [])):
                    # partidos contra nuestro propio bot: se imitarían sus jugadas (las malas). Sirven para
                    # diagnosticar (tools/analyze_match.py), no para imitar
                    print(f"  salteo {folder}/{r['name']}: juega un bot ({', '.join(r['names'])})")
                    continue
                meta = cat.get(r["stadium"].strip())
                if meta is None or meta["script"]:
                    # mapa fuera del catálogo, o con script propio que no se puede reconstruir (Pegeche)
                    print(f"  salteo {folder}/{r['name']}: mapa '{r['stadium']}' fuera del catálogo o con script propio")
                    continue
                if a.limit and done >= a.limit:
                    break
                out = out_root / folder / (Path(r["name"]).stem + ".npz")
                if out.exists() and not a.overwrite:
                    if a.stadium or a.team_size:
                        with np.load(out) as cached:
                            old_stadium = str(cached["selection_stadium"]) if "selection_stadium" in cached else ""
                            old_team = int(cached["selection_team_size"]) if "selection_team_size" in cached else 0
                        if old_stadium != (a.stadium or "") or old_team != (a.team_size or 0):
                            raise ValueError(f"{out}: filtros distintos o desconocidos; usar --out en un directorio nuevo")
                    done += 1
                    continue
                res = process_replay(Path(r["file"]), meta, a.stride, tmp, loaders, rng, cat,
                                     a.stadium, a.team_size)
                if res is None:
                    print(f"  {folder}/{r['name']}: sin muestras (equipos desparejos todo el partido?)")
                    continue
                out.parent.mkdir(parents=True, exist_ok=True)
                lag_arrays = {f"act_lag{lag}".replace("-", "m"): np.concatenate(v) for lag, v in res["lags"].items()}
                np.savez_compressed(out, act=np.concatenate(res["act"]), T=np.concatenate(res["T"]), **lag_arrays,
                                    weight=np.float32(WEIGHTS.get(folder, 1.0)), stadium=a.stadium or meta["stadium"],
                                    selection_stadium=a.stadium or "", selection_team_size=a.team_size or 0,
                                    powershot=meta["powershot"], out_of_bounds=meta["out_of_bounds"],
                                    **{f"obs_{i}": o for i, o in enumerate(res["obs"])})
                n = sum(len(x) for x in res["act"])
                total += n
                done += 1
                print(f"{folder:9s} {r['name'][:48]:48s} {meta['stadium']:14s} {n:7d} muestras "
                      f"({res['n_ticks'] / 3600:.0f} min)", flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"\ntotal nuevo: {total:,} muestras en {out_root}")


if __name__ == "__main__":
    main()
