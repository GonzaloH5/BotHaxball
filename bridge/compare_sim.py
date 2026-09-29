"""Compara grabaciones reales (bridge/record.js) contra sim/physics.py.

Carga el estado real de un tick en el simulador, aplica los mismos inputs y mide cuánto se
separa del juego real a 1, 10, 30, 60 y 180 ticks. Sólo lee el proyecto; no modifica nada.

Uso (desde la raíz del repo):
    python bridge/compare_sim.py runs/real/<grabacion>.jsonl [--stadium real|classic|ruta.hbs ...]

Por defecto compara con el estadio real grabado (el .hbs que exportó la sala).
Resultado de referencia (Classic, 2026-09-28): error 0 por tick; <0.005 a 30 ticks.
"""
from __future__ import annotations

import argparse
import json
import tempfile
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from sim.physics import MOVE_DIRS, BatchSim  # noqa: E402
from sim.stadium import BLUEKO, PLAYER_MASK, REDKO, load_stadium  # noqa: E402

HORIZONS = (1, 10, 30, 60, 180)


def load_rec(path):
    ticks, events, header = [], [], None
    for line in open(path, encoding="utf8"):
        try:
            o = json.loads(line)
        except json.JSONDecodeError:  # última línea cortada si la grabación se interrumpió
            continue
        if o["type"] == "tick":
            ticks.append(o)
        elif o["type"] == "header":
            header = o
        else:
            events.append(o)
    return header, ticks, events


def real_stadium(hbs_path):
    """El .hbs exportado por la sala trae la pelota como discs[0] + "ballPhysics": "disc0".
    Si sim/stadium.py todavía no lo entiende, lo adapto en un archivo temporal."""
    try:
        return load_stadium(str(hbs_path))
    except Exception:
        pass
    d = json.loads(Path(hbs_path).read_text(encoding="utf8"))
    if d.get("ballPhysics") == "disc0":
        d["ballPhysics"] = dict(d["discs"][0])
        d["discs"] = d["discs"][1:]
    tmp = Path(tempfile.gettempdir()) / "haxballrl_real_stadium.hbs"
    tmp.write_text(json.dumps(d), encoding="utf8")
    return load_stadium(str(tmp))


# input de HaxBall: 1 arriba, 2 abajo, 4 izq, 8 der, 16 patear  ->  acción 0..17 del sim
_DIR2MOVE = {(int(x), int(y)): i for i, (x, y) in enumerate(MOVE_DIRS)}


def input_to_action(inp: int) -> int:
    dx = ((inp >> 3) & 1) - ((inp >> 2) & 1)
    dy = ((inp >> 1) & 1) - (inp & 1)
    return _DIR2MOVE[(dx, dy)] + (9 if inp & 16 else 0)


class Replayer:
    """Carga ticks reales en un BatchSim de un solo partido y lo avanza con los inputs grabados."""

    def __init__(self, st, ticks):
        t0 = ticks[0]
        self.players = sorted(t0["players"], key=lambda p: (p["team"], p["id"]))  # rojos primero, como el sim
        n_red = sum(p["team"] == 1 for p in self.players)
        self.sim = BatchSim(1, n_red, len(self.players) - n_red, st)
        self.nd = 1 + st.d_pos.shape[0]  # pelota + discos del estadio que simula el sim
        # índices en la grabación: pelota (0) + discos del estadio conservados (el sim descarta los decorativos)
        src = st.d_src if st.d_src is not None else np.arange(st.d_pos.shape[0])
        self.stadium_idx = [0] + [1 + int(i) for i in src]
        if max(self.stadium_idx) >= len(t0["discs"]) - len(self.players):
            raise SystemExit(f"layout de discos distinto: grabación {len(t0['discs'])} discos, "
                             f"el estadio pide índice {max(self.stadium_idx)}")

    def _by_id(self, t):
        by_id = {p["id"]: p for p in t["players"]}
        return [by_id[p["id"]] for p in self.players]

    def _discs(self, t):
        idx = self.stadium_idx + [p["disc"] for p in self._by_id(t)]
        return [t["discs"][i] for i in idx]

    def load(self, t):
        s = self.sim
        ds = self._discs(t)
        s.pos[0] = [[d["x"], d["y"]] for d in ds]
        s.vel[0] = [[d["vx"], d["vy"]] for d in ds]
        # patada "gastada": tecla apretada pero isKicking=false
        s.kick_cancel[0] = [bool(p["input"] & 16) and not p["kicking"] for p in self._by_id(t)]
        s.kickoff[0] = t["state"] == 0
        s.mask[0] = s.base_mask
        if s.kickoff[0]:
            # "ko" = equipo que saca (1 rojo, 2 azul); grabaciones viejas no lo tienen -> rojo
            s.kickoff_team[0] = 1 if t.get("ko") == 2 else 0
            s.mask[0, s.first_player:] = PLAYER_MASK | (REDKO if s.kickoff_team[0] == 0 else BLUEKO)

    def real(self, t):
        ds = self._discs(t)
        return np.array([[d["x"], d["y"]] for d in ds]), np.array([[d["vx"], d["vy"]] for d in ds])

    def step(self, t):
        """Transición t -> t+1: usa el input grabado en t (verificado: la otra alineación falla en patadas)."""
        self.sim.step(np.array([[input_to_action(p["input"]) for p in self._by_id(t)]]))


# frames donde el script de la sala tocó discos (eventos "disc_props" de replay_to_jsonl.js):
# esas transiciones no son física pura y se excluyen
SCRIPT_FRAMES: set[int] = set()


def contiguous(ticks, i, k):
    return all(ticks[j + 1]["frame"] - ticks[j]["frame"] == 1 and ticks[j + 1]["score"] == ticks[i]["score"]
               and ticks[j + 1]["frame"] not in SCRIPT_FRAMES and ticks[j]["frame"] not in SCRIPT_FRAMES
               and ticks[j].get("_ok", True) and ticks[j + 1].get("_ok", True)
               for j in range(i, i + k))


def mark_valid(rp, ticks) -> None:
    """Marca los ticks donde algún disco que usa el sim tiene posición vacía (NaN en el juego: scripts que
    'esconden' discos o la pelota, p. ej. Sanguchito X1): esos ticks no se comparan."""
    for t in ticks:
        idx = rp.stadium_idx + [p["disc"] for p in rp._by_id(t)]
        t["_ok"] = all(t["discs"][i]["x"] is not None and t["discs"][i]["vx"] is not None for i in idx)


def rollout_errors(rp, ticks, k, states):
    eb, ep = [], []
    fp = rp.sim.first_player
    for i in range(0, len(ticks) - k - 1, max(1, k // 3)):
        if ticks[i]["state"] not in states or not contiguous(ticks, i, k):
            continue
        rp.load(ticks[i])
        for j in range(k):
            rp.step(ticks[i + j])
        p = rp.sim.pos[0]
        rpos, _ = rp.real(ticks[i + k])
        eb.append(np.linalg.norm(p[0] - rpos[0]))
        ep.append(np.linalg.norm(p[fp:] - rpos[fp:], axis=1).max())
    return np.array(eb), np.array(ep)


def roster(t):
    return tuple(sorted((p["id"], p["team"]) for p in t["players"]))


def segments(ticks):
    """Tramos consecutivos con el mismo plantel (en salas grandes entran y salen jugadores)."""
    out, cur = [], []
    for t in ticks:
        if cur and (roster(t) != roster(cur[-1]) or len(t["discs"]) != len(cur[-1]["discs"])):
            out.append(cur)
            cur = []
        cur.append(t)
    if cur:
        out.append(cur)
    return out


def run(path, stadium_arg):
    header, ticks, events = load_rec(path)
    SCRIPT_FRAMES.clear()
    SCRIPT_FRAMES.update(e["frame"] for e in events if e.get("name") == "disc_props")
    st = real_stadium(Path(path).parent / header["stadiumFile"]) if stadium_arg == "real" else load_stadium(stadium_arg)
    goals = sum(e["name"] == "goal" for e in events)
    segs = []
    for seg in segments(ticks):
        if len(seg) < 2 or not seg[0]["players"]:
            continue
        try:
            rp = Replayer(st, seg)
            mark_valid(rp, seg)
            segs.append((rp, seg))
        except SystemExit as e:
            print(f"  (tramo de {len(seg)} ticks salteado: {e})")
    sizes = sorted({(sum(p["team"] == 1 for p in s[0]["players"]), sum(p["team"] == 2 for p in s[0]["players"])) for _, s in segs})
    print(f"\n=== {Path(path).name} | estadio: {stadium_arg} ({header.get('stadium')}) | {len(ticks)} ticks, {goals} goles, "
          f"{len(segs)} tramos de plantel fijo, formatos {', '.join(f'{r}v{b}' for r, b in sizes)}")

    for label, states in (("en juego", (1,)), ("saque", (0,))):
        n_state = sum(t["state"] in states for _, s in segs for t in s)
        if n_state == 0:
            print(f"  [{label}] sin ticks en la grabación")
            continue
        print(f"  [{label}] {n_state} ticks")
        for k in HORIZONS:
            parts = [rollout_errors(rp, s, k, states) for rp, s in segs]
            eb = np.concatenate([p[0] for p in parts])
            ep = np.concatenate([p[1] for p in parts])
            if len(eb) == 0:
                continue
            print(f"    {k:4d} ticks ({len(eb):4d} tramos): pelota med {np.median(eb):8.4f} p90 {np.percentile(eb, 90):8.3f}"
                  f" máx {eb.max():8.3f} | jugadores med {np.median(ep):8.4f} p90 {np.percentile(ep, 90):8.3f} máx {ep.max():8.3f}")
            if k == 1 and label == "en juego":
                GATE.append((Path(path).name, header.get("stadium"), len(eb),
                             float(np.percentile(eb, 90)), float(np.percentile(ep, 90))))

    # peores transiciones de 1 tick (para diagnosticar si algo no coincide)
    worst = []
    for rp, s in segs:
        for i in range(len(s) - 1):
            if not contiguous(s, i, 1):
                continue
            rp.load(s[i])
            rp.step(s[i])
            rpos, _ = rp.real(s[i + 1])
            err = np.abs(rp.sim.pos[0] - rpos).max(axis=1)
            d = int(err.argmax())
            kind = "pelota" if d == 0 else ("jugador" if d >= rp.sim.first_player else "disco del estadio")
            worst.append((err[d], s[i + 1]["frame"], s[i]["state"], f"{d} ({kind})", rp.sim.pos[0, d].copy(), rpos[d]))
    worst.sort(key=lambda x: -x[0])
    print("  peores 1-tick:")
    for e, fr, stt, d, p, r in worst[:5]:
        print(f"    frame {fr} (estado {stt}) disco {d}: err {e:.4f}  sim {np.round(p, 2)}  real {np.round(r, 2)}")

    for rp, s in segs:
        ko = next((t for t in s if t["state"] == 0 and t.get("_ok", True)), None)
        if ko:
            rp.sim.reset_kickoff([0], kickoff_team=0)
            rpos, _ = rp.real(ko)
            fp = rp.sim.first_player
            print(f"  posiciones de saque: sim {np.round(rp.sim.pos[0, fp:], 1).tolist()}  real {np.round(rpos[fp:], 1).tolist()}")
            break


# (grabación, estadio, n transiciones, p90 pelota, p90 jugadores) del error de 1 tick en juego
GATE: list[tuple] = []
GATE_P90 = 0.05  # error de 1 tick tolerado en el 90% de las transiciones (unidades del mapa)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("recordings", nargs="+")
    ap.add_argument("--stadium", nargs="+", default=["real"])
    ap.add_argument("--gate", action="store_true",
                    help=f"resumen final y código de salida 1 si algún mapa supera p90 {GATE_P90} en 1 tick")
    a = ap.parse_args()
    for rec in a.recordings:
        for s in a.stadium:
            run(rec, s)
    if a.gate:
        print(f"\n=== control de física (1 tick en juego, p90 <= {GATE_P90}) ===")
        bad = 0
        for name, st, n, pb, pp in GATE:
            ok = n >= 50 and pb <= GATE_P90 and pp <= GATE_P90
            bad += not ok
            print(f"  {'OK ' if ok else 'MAL'} {str(st)[:36]:36s} {n:6d} transiciones | pelota p90 {pb:.4f} | jugadores p90 {pp:.4f} | {name}")
        sys.exit(1 if bad or not GATE else 0)


if __name__ == "__main__":
    main()
